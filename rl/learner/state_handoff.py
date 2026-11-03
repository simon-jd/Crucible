"""Owned Linux RAM handoff for the pinned Torch 2.8 single-caller learner.

The parent stays Torch-free. A snapshot remains charged until owned consumers
exit/reap; closing one fd is not proof that mappings elsewhere were released.
No full-model memory eligibility is implied by the tiny CPU tests.
"""

import fcntl
import hashlib
import itertools
import json
import math
import os
import pickle
import random
import selectors
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace

from rl.learner.handoff_zip import DirectoryTail, check_save_records, preflight

# Some pinned Python builds omit these names despite supporting fcntl itself.
# Linux x86-64 UAPI: linux/fcntl.h:41-50, asm-generic/fcntl.h:189.
# Module-owned aliases only: never monkeypatch stdlib or bypass kernel checks.
if sys.platform != "linux" or os.uname().machine != "x86_64":
    raise RuntimeError("RAM handoff seal constants require supported Linux x86-64")
F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
F_GET_SEALS = getattr(fcntl, "F_GET_SEALS", 1034)
F_SEAL_SEAL = getattr(fcntl, "F_SEAL_SEAL", 1)
F_SEAL_SHRINK = getattr(fcntl, "F_SEAL_SHRINK", 2)
F_SEAL_GROW = getattr(fcntl, "F_SEAL_GROW", 4)
F_SEAL_WRITE = getattr(fcntl, "F_SEAL_WRITE", 8)
SEALS = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL
FORMAT = "crucible-training-ram-v1"
MAX_RAM = 16_000_000_000
MAX_METADATA_ITEMS = 100_000
MAX_NAME = 1024


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid {label}")
    return value


def _binding(value):
    if type(value) is not dict or set(value) != {"model", "tokenizer", "external_state"}:
        raise ValueError("binding requires model/tokenizer/external_state")
    if value["external_state"] != "fixed-lr:no-scheduler:no-scaler:no-loader:no-numpy":
        raise ValueError("unsupported external state")
    for key in ("model", "tokenizer"):
        if type(value[key]) is not str or not 0 < len(value[key]) <= MAX_NAME:
            raise ValueError("immutable model/tokenizer binding required")
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(fd, size):
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        block = os.pread(fd, min(65536, size - offset), offset)
        if not block:
            raise ValueError("truncated RAM snapshot")
        digest.update(block)
        offset += len(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Receipt:
    device: int
    inode: int
    size: int
    sha256: str
    seals: int
    version: int
    binding: str
    cap: int
    metadata_cap: int = 32768
    storage_cap: int = 8192

    def validate(self, fd):
        _integer(self.size, "snapshot size", 1)
        _integer(self.cap, "snapshot cap", 1)
        _integer(self.version, "version")
        _integer(self.metadata_cap, "metadata cap", 1)
        _integer(self.storage_cap, "storage cap", 1)
        if self.size > self.cap or self.seals != SEALS or self.metadata_cap > 8_000_000:
            raise ValueError("invalid snapshot bounds/seals")
        if _binding(json.loads(self.binding)) != self.binding:
            raise ValueError("noncanonical snapshot binding")
        stat = os.fstat(fd)
        if (stat.st_dev, stat.st_ino, stat.st_size) != (self.device, self.inode, self.size):
            raise ValueError("snapshot descriptor identity/size mismatch")
        if fcntl.fcntl(fd, F_GET_SEALS) != SEALS:
            raise ValueError("snapshot is not fully sealed")
        if _digest(fd, self.size) != self.sha256:
            raise ValueError("snapshot digest mismatch")


@dataclass(frozen=True)
class CaptureBudget:
    """Caller-declared aggregate reservations; runtime RSS is separately monitored."""

    max_bytes: int
    other_resident_bytes: int
    retained_snapshot_bytes: int = 0
    mapped_bytes: int = 0
    ram_limit: int = MAX_RAM
    metadata_cap: int = 32768
    storage_cap: int = 8192

    def check(self, payload_bytes, largest_storage):
        for key, value in asdict(self).items():
            _integer(value, key)
        if (
            not 0 < self.max_bytes
            or not 0 < self.ram_limit <= MAX_RAM
            or not 0 < self.metadata_cap <= 8_000_000
            or self.storage_cap <= 0
        ):
            raise ValueError("invalid RAM/capture limit")
        if largest_storage > self.storage_cap:
            raise ValueError("single-storage staging cap exceeded")
        if payload_bytes > self.max_bytes:
            raise ValueError("checkpoint byte cap exceeded")
        # Conservative metadata buffering + CPU staging + Python writer buffer.
        common = self.other_resident_bytes + self.retained_snapshot_bytes + self.metadata_charge
        # Serial stages, never simultaneously three complete payloads: this is
        # valid only with the enforced own-process mapping-release proof below.
        restore_peak = self.restore_charge(self.retained_snapshot_bytes, self.mapped_bytes)
        capture_peak = common + payload_bytes + 2 * largest_storage
        charge = max(restore_peak, capture_peak)
        if charge > self.ram_limit:
            raise ValueError("aggregate RAM reservation exceeded")
        return charge

    def restore_charge(self, retained_bytes, mapped_bytes):
        """Same transient staging reservation for every consumer admission."""
        return (
            self.other_resident_bytes
            + retained_bytes
            + mapped_bytes
            + self.metadata_charge
            + 2 * self.storage_cap
        )

    @property
    def metadata_charge(self):
        # Declared conservative heap/memo allowance, not a measured allocation quota.
        # Includes two metadata buffers, dictionary/memo overhead and64KiB each
        # for the Pickler callback buffer and streaming SHA chunk.
        # Extra eight units cover ZIP tails/directory buffers and bounded record
        # objects (<=cap//512), names/sets;4KiB covers fixed parser temporaries.
        return 16 * self.metadata_cap + 135168


class _BoundedWriter:
    def __init__(self, cap, fd=None, directory_tail=None):
        self.cap, self.fd, self.position = cap, fd, 0
        self.directory_tail = directory_tail
        self.failure = None

    def write(self, data):
        if self.failure is not None:
            raise self.failure
        if self.position + len(data) > self.cap:
            self.failure = ValueError("checkpoint byte cap exceeded before write")
            raise self.failure
        try:
            if self.fd is not None:
                view = memoryview(data)
                while view:
                    written = os.write(self.fd, view)
                    if written <= 0:
                        raise OSError("short RAM write")
                    view = view[written:]
            if self.directory_tail is not None:
                self.directory_tail.write(data)
            self.position += len(data)
            return len(data)
        except BaseException as error:
            self.failure = error
            raise

    def flush(self):
        pass


class _MetadataSink:
    """Enforce data.pkl limit before growing Torch's internal BytesIO."""

    def __init__(self, destination, cap):
        self.destination, self.cap, self.position = destination, cap, 0
        self.failure = None

    def write(self, data):
        if self.position + len(data) > self.cap:
            self.failure = ValueError("pickle metadata cap exceeded before buffer write")
            raise self.failure
        result = self.destination.write(data)
        self.position += result
        return result


def _save(torch, payload, writer, metadata_cap=32768):
    sinks = []

    class BoundedPickler(pickle.Pickler):
        def __init__(self, destination, *args, **kwargs):
            sink = _MetadataSink(destination, metadata_cap)
            sinks.append(sink)
            super().__init__(sink, *args, **kwargs)

    # Save-only adapter; Torch's derived Pickler retains its persistent_id logic.
    # Loading always uses the untouched default weights-only restricted loader.
    save_pickle = SimpleNamespace(__name__="pickle", Pickler=BoundedPickler)
    try:
        torch.save(payload, writer, pickle_module=save_pickle)
    except BaseException as error:
        # ZIP finalization can raise again; retain the original bounded-write cause.
        if writer.failure is not None and error is not writer.failure:
            raise writer.failure from error
        for sink in sinks:
            if sink.failure is not None and error is not sink.failure:
                raise sink.failure from error
        raise
    if writer.failure is not None:
        raise writer.failure
    return writer.position


@dataclass(frozen=True)
class Snapshot:
    fd: int
    receipt: Receipt


class SnapshotOwner:
    """Synchronous parent-only orchestration of trusted, non-spawning workers.

    Existing disk checkpoints must be declared as exact paths. Only this owner
    may spawn fd consumers; callers must not duplicate/pass/map its descriptors.
    A successful child exit is the mapping-release proof. Ambiguous cleanup
    permanently poisons the owner and retains its reservations/descriptors.
    """

    def __init__(
        self,
        *,
        max_bytes,
        other_resident_bytes,
        ram_limit=MAX_RAM,
        metadata_cap=32768,
        storage_cap=8192,
        disk_checkpoints=(),
        timeout=30.0,
    ):
        self.budget = CaptureBudget(
            max_bytes,
            other_resident_bytes,
            ram_limit=ram_limit,
            metadata_cap=metadata_cap,
            storage_cap=storage_cap,
        )
        self.budget.check(max_bytes, storage_cap)
        if not 0 < timeout <= 120:
            raise ValueError("finite timeout <=120 required")
        identities = [os.stat(path) for path in disk_checkpoints]
        if len({(stat.st_dev, stat.st_ino) for stat in identities}) != len(identities):
            raise ValueError("duplicate disk checkpoint")
        if len(identities) > 2:
            raise ValueError("at most two checkpoints")
        self.disk_count, self.timeout = len(identities), timeout
        self.snapshots, self.events = {}, []
        self.busy, self.failed, self.closed = False, False, False
        self.pending_fd = None
        self.last_process = None
        self.last_output = None

    def _ready(self):
        if self.closed or self.failed or self.busy:
            raise RuntimeError("snapshot owner is closed/failed/busy")

    def _child(self, command, fd, extra, source=None):
        env = os.environ.copy()
        env.update(extra)
        env["CRUCIBLE_HANDOFF_FD"] = str(fd)
        inherited = (fd,)
        if source is not None:
            env["CRUCIBLE_HANDOFF_SOURCE_FD"] = str(source.fd)
            env["CRUCIBLE_HANDOFF_SOURCE_RECEIPT"] = json.dumps(asdict(source.receipt))
            inherited += (source.fd,)
        process = subprocess.Popen(
            command,
            env=env,
            pass_fds=inherited,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.last_process = process
        output, deadline = bytearray(), time.monotonic() + self.timeout
        cause = None
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                eof = False
                while not eof:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("owned handoff child deadline exceeded")
                    for key, _ in selector.select(min(0.05, deadline - time.monotonic())):
                        block = os.read(key.fileobj.fileno(), 4096)
                        if not block:
                            eof = True
                            break
                        if len(output) + len(block) > 65536:
                            raise ValueError("owned child output cap exceeded")
                        output.extend(block)
                code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
                if code != 0:
                    raise RuntimeError(
                        f"owned handoff child exited {code}: {output.decode(errors='replace')}"
                    )
        except BaseException as error:
            cause = error
            cleanup_errors = []
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2.5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2.5)
                else:
                    process.wait(timeout=0)
            except BaseException as cleanup:  # noqa: BLE001 - retain cleanup uncertainty and primary cause
                cleanup_errors.append(cleanup)
                self.failed = True
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "handoff child and cleanup failed", [error, *cleanup_errors]
                )
            raise
        finally:
            self.events.append(
                {
                    "pid": process.pid,
                    "returncode": process.returncode,
                    "reaped": process.returncode is not None,
                    "error": None if cause is None else f"{type(cause).__name__}: {cause}",
                }
            )
            try:
                process.stdout.close()
            except BaseException as cleanup:
                self.failed = True
                if cause is not None:
                    raise BaseExceptionGroup("child and pipe cleanup failed", [cause, cleanup])
                raise
        return output.decode()

    def capture(self, command, *, version, binding, source=None):
        self._ready()
        _integer(version, "version")
        binding_text = _binding(binding)
        if source is not None:
            if self.snapshots.get(source.fd) is not source:
                raise ValueError("source snapshot is not owned here")
            source.receipt.validate(source.fd)
            if version <= source.receipt.version or source.receipt.binding != binding_text:
                raise ValueError("advance requires matching binding and increasing version")
        weights_bytes, weights_slots = getattr(self, "_serial_weights", (0, 0))
        if self.disk_count + len(self.snapshots) + weights_slots >= 2:
            raise ValueError("at most two checkpoints including pending RAM")
        retained = sum(item.receipt.size for item in self.snapshots.values()) + weights_bytes
        budget = CaptureBudget(
            self.budget.max_bytes,
            self.budget.other_resident_bytes,
            retained,
            mapped_bytes=0 if source is None else source.receipt.size,
            ram_limit=self.budget.ram_limit,
            metadata_cap=self.budget.metadata_cap,
            storage_cap=self.budget.storage_cap,
        )
        budget.check(budget.max_bytes, budget.storage_cap)
        fd = os.memfd_create("crucible-training", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
        self.pending_fd, self.busy = fd, True
        try:
            identity = os.fstat(fd)
            self.last_output = self._child(
                command,
                fd,
                {
                    "CRUCIBLE_HANDOFF_BUDGET": json.dumps(asdict(budget)),
                    "CRUCIBLE_HANDOFF_VERSION": str(version),
                    "CRUCIBLE_HANDOFF_BINDING": binding_text,
                },
                source,
            )
            stat = os.fstat(fd)
            if not 0 < stat.st_size <= budget.max_bytes:
                raise ValueError("child snapshot size outside cap")
            receipt = Receipt(
                identity.st_dev,
                identity.st_ino,
                stat.st_size,
                _digest(fd, stat.st_size),
                fcntl.fcntl(fd, F_GET_SEALS),
                version,
                binding_text,
                budget.max_bytes,
                budget.metadata_cap,
                budget.storage_cap,
            )
            receipt.validate(fd)
            snapshot = Snapshot(fd, receipt)
            self.snapshots[fd] = snapshot
            return snapshot
        except BaseException as error:
            self.events.append(
                {"stage": "capture-rejected", "error": f"{type(error).__name__}: {error}"}
            )
            if not self.failed:
                try:
                    os.close(fd)
                except BaseException as cleanup:  # noqa: BLE001 - preserve primary and close failures
                    self.failed = True
                    raise BaseExceptionGroup(
                        "capture and descriptor cleanup failed", [error, cleanup]
                    )
            raise
        finally:
            self.busy = False
            if not self.failed:
                self.pending_fd = None

    def capture_stream(self, command, *, version, binding, budget):
        from rl.learner.handoff_protocol import capture_initial

        return capture_initial(self, command, version=version, binding=binding, budget=budget)

    def advance_consuming(self, source, command, *, version, binding, budget):
        from rl.learner.handoff_protocol import advance_consuming

        return advance_consuming(
            self, source, command, version=version, binding=binding, budget=budget
        )

    def consume(self, snapshot, command):
        self._ready()
        if self.snapshots.get(snapshot.fd) is not snapshot:
            raise ValueError("snapshot is not owned here")
        snapshot.receipt.validate(snapshot.fd)
        # Count the mapped length again, even if RSS shares physical pages.
        charge = self.budget.restore_charge(
            sum(item.receipt.size for item in self.snapshots.values())
            + getattr(self, "_serial_weights", (0, 0))[0],
            snapshot.receipt.size,
        )
        if charge > self.budget.ram_limit:
            raise ValueError("aggregate RAM reservation exceeded for restore")
        self.busy = True
        try:
            return self._child(
                command,
                snapshot.fd,
                {
                    "CRUCIBLE_HANDOFF_RECEIPT": json.dumps(asdict(snapshot.receipt)),
                },
            )
        finally:
            self.busy = False

    def release(self, snapshot):
        self._ready()
        if self.snapshots.get(snapshot.fd) is not snapshot:
            raise ValueError("snapshot is not owned here")
        try:
            os.close(snapshot.fd)
        except BaseException:
            self.failed = True
            raise
        del self.snapshots[snapshot.fd]

    def close(self):
        if self.closed:
            return
        self._ready()
        for snapshot in tuple(self.snapshots.values()):
            self.release(snapshot)
        self.closed = True

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, kind, error, trace):
        try:
            self.close()
        except BaseException as cleanup:
            if error is not None:
                raise BaseExceptionGroup("handoff and owner cleanup failed", [error, cleanup])
            raise


def _torch():
    import torch

    if torch.__version__.split("+")[0] != "2.8.0":
        raise RuntimeError("handoff requires pinned Torch 2.8.0")
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("handoff supports world_size=1 only")
    return torch


def _local(value):
    torch = _torch()
    from torch.distributed.tensor import DTensor, Replicate, Shard

    distribution = None
    if isinstance(value, DTensor):
        mesh = value.device_mesh
        if mesh.ndim != 1 or mesh.size() != 1 or mesh.mesh.tolist() != [0]:
            raise ValueError("DTensor requires exact world1 rank0 mesh")
        placements = []
        for placement in value.placements:
            if type(placement) is Shard:
                placements.append(["shard", placement.dim])
            elif type(placement) is Replicate:
                placements.append(["replicate"])
            else:
                raise ValueError("unsupported DTensor placement")
        distribution = {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "mesh": [0],
            "mesh_names": mesh.mesh_dim_names,
            "device_type": mesh.device_type,
            "placements": placements,
        }
        with torch.no_grad():
            local = value.to_local()
    else:
        local = value
    if type(local) not in (torch.Tensor, torch.nn.Parameter):
        raise TypeError("ordinary completed local Tensor required")
    if local.layout != torch.strided or not local.is_contiguous() or local.storage_offset() != 0:
        raise ValueError("only contiguous whole-storage views supported")
    local = local.detach()
    if local.untyped_storage().nbytes() != local.numel() * local.element_size():
        raise ValueError("partial storage view unsupported")
    if local.device.type not in ("cpu", "cuda") or local.is_complex():
        raise ValueError("unsupported tensor device/dtype")
    if not bool(torch.isfinite(local).all()):
        raise ValueError("nonfinite handoff tensor")
    metadata = {
        "shape": list(local.shape),
        "stride": list(local.stride()),
        "dtype": str(local.dtype),
        "device": str(local.device),
        "distribution": distribution,
    }
    return local, metadata


def _model(model):
    torch = _torch()
    for module in model.modules():
        if type(module).get_extra_state is not torch.nn.Module.get_extra_state:
            raise ValueError("unsupported model extra state")
    tensors, schema, aliases, seen = {}, {}, {}, {}
    parameters = list(model.named_parameters(remove_duplicate=False))
    buffers = list(model.named_buffers(remove_duplicate=False))
    if not parameters or len(parameters) + len(buffers) > MAX_METADATA_ITEMS:
        raise ValueError("invalid model tensor count")
    for kind, values in (("parameter", parameters), ("buffer", buffers)):
        for name, value in values:
            if not name or len(name) > MAX_NAME or name in schema:
                raise ValueError("invalid/duplicate tensor name")
            tensor, metadata = _local(value)
            key = (str(tensor.device), tensor.untyped_storage()._cdata)
            if key in seen:
                canonical = seen[key]
                if metadata != schema[canonical]["tensor"]:
                    raise ValueError("inconsistent tied storage view")
                aliases[name] = canonical
            else:
                seen[key] = name
                tensors[name] = tensor
            schema[name] = {
                "kind": kind,
                "requires_grad": value.requires_grad,
                "tensor": metadata,
                "persistent": kind == "parameter"
                or name.rsplit(".", 1)[-1]
                not in model.get_submodule(
                    name.rsplit(".", 1)[0] if "." in name else ""
                )._non_persistent_buffers_set,
            }
    return {"schema": schema, "aliases": aliases, "tensors": tensors}


def _options(optimizer):
    torch = _torch()
    if type(optimizer) is torch.optim.AdamW:
        tag = "AdamW"
        keys = {
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
            "decoupled_weight_decay",
        }
    elif type(optimizer) is torch.optim.SGD:
        tag = "SGD"
        keys = {
            "lr",
            "momentum",
            "dampening",
            "weight_decay",
            "nesterov",
            "maximize",
            "foreach",
            "differentiable",
            "fused",
        }
    else:
        raise TypeError("exact SGD/AdamW class required; subclasses unsupported")
    for name, value in vars(optimizer).items():
        if name.endswith("hooks") and value:
            raise ValueError("optimizer hooks/external state unsupported")

    def validate(options):
        if set(options) != keys:
            raise ValueError("optimizer option keys mismatch/external state")
        for key, value in options.items():
            sequence = value if type(value) is tuple else (value,)
            if any(type(item) not in (int, float, bool, type(None)) for item in sequence):
                raise ValueError("unsupported optimizer option type")
            if any(type(item) is float and not math.isfinite(item) for item in sequence):
                raise ValueError("nonfinite optimizer option")
            if key in ("capturable", "differentiable", "fused") and value not in (False, None):
                raise ValueError("unsupported optimizer execution option")
        if tag == "AdamW" and options["decoupled_weight_decay"] is not True:
            raise ValueError("AdamW requires decoupled weight decay")
        return dict(options)

    defaults = validate(optimizer.defaults)
    groups = [
        validate({key: val for key, val in group.items() if key != "params"})
        for group in optimizer.param_groups
    ]
    return tag, defaults, groups


def _optimizer(policy, optimizer):
    tag, defaults, options = _options(optimizer)
    canonical = {id(value): name for name, value in policy.named_parameters()}
    groups, states, seen = [], {}, set()
    for group, option in zip(optimizer.param_groups, options, strict=True):
        names = []
        for parameter in group["params"]:
            name = canonical.get(id(parameter))
            if name is None or name in seen:
                raise ValueError("optimizer name/order ownership mismatch")
            seen.add(name)
            names.append(name)
            state = optimizer.state.get(parameter, {})
            allowed = {"momentum_buffer"} if tag == "SGD" else {"step", "exp_avg", "exp_avg_sq"}
            if tag == "AdamW" and option["amsgrad"]:
                allowed.add("max_exp_avg_sq")
            if state and set(state) != allowed:
                raise ValueError("optimizer state keys mismatch")
            if tag == "SGD" and state and option["momentum"] == 0:
                raise ValueError("unexpected SGD momentum state")
            states[name] = {}
            for key, value in state.items():
                tensor, metadata = _local(value)
                states[name][key] = {"tensor": tensor, "meta": metadata}
        groups.append({"names": names, "options": option})
    if seen != set(canonical.values()) or any(id(key) not in canonical for key in optimizer.state):
        raise ValueError("optimizer does not exactly own named policy parameters")
    return {"class": tag, "defaults": defaults, "groups": groups, "state": states}


def _rng(policy):
    torch = _torch()
    cuda = None
    devices = {str(value.device) for value in policy.parameters() if value.device.type == "cuda"}
    if torch.cuda.is_initialized():
        generators = torch.cuda.default_generators
        if len(generators) != 1 or devices != {"cuda:0"}:
            raise ValueError("CUDA RNG requires initialized single device cuda:0")
        cuda = {"device": "cuda:0", "count": 1, "state": generators[0].get_state()}
    elif devices:
        raise ValueError("CUDA parameters without already-initialized context")
    return {"python": random.getstate(), "cpu": torch.get_rng_state(), "cuda": cuda}


def _payload(learner, binding):
    torch = _torch()
    version, policy, optimizer = learner.policy_version, learner.policy, learner.optimizer
    payload = {
        "format": FORMAT,
        "version": version,
        "binding": _binding(binding),
        "config": asdict(learner.config),
        "default_dtype": str(torch.get_default_dtype()),
        "policy": _model(policy),
        "reference": _model(learner.reference),
        "optimizer": _optimizer(policy, optimizer),
        "rng": _rng(policy),
    }
    return payload, policy, optimizer


def _walk_tensors(value):
    torch = _torch()
    if type(value) is torch.Tensor:
        yield value
    elif type(value) is dict:
        for child in value.values():
            yield from _walk_tensors(child)
    elif type(value) in (tuple, list):
        for child in value:
            yield from _walk_tensors(child)


def _signature(payload):
    """Metadata and live identities, no tensor content copy or tensor equality."""
    torch = _torch()
    if type(payload) is torch.Tensor:
        return (
            "tensor",
            str(payload.dtype),
            tuple(payload.shape),
            payload.untyped_storage()._cdata,
            payload._version,
        )
    if type(payload) is dict:
        return {key: _signature(value) for key, value in payload.items()}
    if type(payload) in (tuple, list):
        return tuple(_signature(value) for value in payload)
    return payload


def _assert_unmapped(receipt):
    """Kernel-view proof for this consumer only, not an owner-fd/GC inference."""
    with open("/proc/self/maps", encoding="ascii") as maps:
        for count, line in enumerate(maps, 1):
            if count > 100000 or len(line) > 16384:
                raise RuntimeError("cannot prove mapping release within bounded maps scan")
            fields = line.split(maxsplit=5)
            if len(fields) < 5:
                raise RuntimeError("unrecognized process maps record")
            major, minor = (int(value, 16) for value in fields[3].split(":"))
            if os.makedev(major, minor) == receipt.device and int(fields[4]) == receipt.inode:
                raise RuntimeError("source snapshot mapping still live in this consumer")


def capture_ready_learner(learner, fd, *, binding, budget, source_fd=None, source_receipt=None):
    """Write and seal an initially empty inherited memfd; child must then exit."""
    # Acquire publishable state before inspecting descriptors or building views.
    _ = learner.policy_version, learner.policy, learner.optimizer
    torch = _torch()
    budget.check(budget.max_bytes, budget.storage_cap)

    def source_released():
        if source_receipt is None:
            if source_fd is not None or budget.mapped_bytes:
                raise ValueError("advance requires source receipt and mapping-release proof")
        else:
            if source_fd is None or budget.mapped_bytes != source_receipt.size:
                raise ValueError("advance source mapping reservation mismatch")
            if budget.retained_snapshot_bytes < source_receipt.size:
                raise ValueError("advance source payload reservation missing")
            source_receipt.validate(source_fd)
            _assert_unmapped(source_receipt)

    source_released()
    stat = os.fstat(fd)
    if stat.st_size != 0 or fcntl.fcntl(fd, F_GET_SEALS) != 0:
        raise ValueError("capture requires empty unsealed memfd")
    # Bound Python schema construction before building per-name dictionaries.
    # This is conservative planning, not a Python allocator quota.
    item_charge = 0
    for model in (learner.policy, learner.reference):
        for values in (
            model.named_parameters(remove_duplicate=False),
            model.named_buffers(remove_duplicate=False),
        ):
            for name, value in values:
                item_charge += 512 + 2 * len(name.encode()) + 64 * value.ndim
                if item_charge > 4 * budget.metadata_cap:
                    raise ValueError("metadata item/name cap exceeded before schema construction")
    for state in learner.optimizer.state.values():
        item_charge += 1024 * len(state)
        if item_charge > 4 * budget.metadata_cap:
            raise ValueError("optimizer metadata item cap exceeded before schema construction")
    payload, policy, optimizer = _payload(learner, binding)
    # Bound schema/RNG metadata before torch.save builds its own pickle buffer.
    metadata = _metadata(payload)
    metadata_bytes = 0
    for chunk in json.JSONEncoder().iterencode(metadata):
        metadata_bytes += len(chunk.encode())
        if metadata_bytes > budget.metadata_cap:
            raise ValueError("metadata exceeds checkpoint cap before serialization")
    # Validate state semantics on capture too, before either serialization pass.
    _validate_payload(
        payload, policy, learner.reference, optimizer, learner.config, payload["version"], binding
    )
    # RNG get_state returns fresh tiny tensors: compare RNG values separately.
    state_signature = _signature({key: value for key, value in payload.items() if key != "rng"})

    def recheck():
        source_released()
        if learner.policy is not policy or learner.optimizer is not optimizer:
            raise RuntimeError("learner ownership changed during capture")
        current, _, _ = _payload(learner, binding)
        if (
            _signature({key: value for key, value in current.items() if key != "rng"})
            != state_signature
        ):
            raise RuntimeError("learner changed during capture")
        if current["rng"]["python"] != payload["rng"]["python"]:
            raise RuntimeError("Python RNG changed during capture")
        if not torch.equal(current["rng"]["cpu"], payload["rng"]["cpu"]):
            raise RuntimeError("Torch RNG changed during capture")
        if (current["rng"]["cuda"] is None) != (payload["rng"]["cuda"] is None):
            raise RuntimeError("CUDA RNG ownership changed during capture")
        if current["rng"]["cuda"] is not None and not torch.equal(
            current["rng"]["cuda"]["state"], payload["rng"]["cuda"]["state"]
        ):
            raise RuntimeError("CUDA RNG changed during capture")

    unique = {
        tensor.untyped_storage()._cdata: tensor.untyped_storage().nbytes()
        for tensor in _walk_tensors(payload)
    }
    largest = max(unique.values(), default=0)
    if sum(unique.values()) > budget.max_bytes:
        raise ValueError("tensor storage floor exceeds checkpoint cap before write")
    budget.check(budget.max_bytes, largest)
    check_save_records(len(unique), budget.metadata_cap)
    recheck()
    tail = DirectoryTail(budget.metadata_cap)
    count = _save(
        torch, payload, _BoundedWriter(budget.max_bytes, directory_tail=tail), budget.metadata_cap
    )
    preflight(tail.read, count, budget.metadata_cap, budget.storage_cap)
    del tail
    charge = budget.check(count, largest)
    recheck()
    os.lseek(fd, 0, os.SEEK_SET)
    written = _save(torch, payload, _BoundedWriter(count, fd), budget.metadata_cap)
    if written != count or os.fstat(fd).st_size != count:
        raise ValueError("serialization size changed after exact preflight")
    recheck()
    fcntl.fcntl(fd, F_ADD_SEALS, SEALS)
    return {"bytes": count, "largest_storage": largest, "ram_charge": charge}


def _validate_tensor(tensor, metadata):
    torch = _torch()
    if type(tensor) is not torch.Tensor:
        raise ValueError("payload must contain only plain Tensor values")
    if (
        (str(tensor.dtype), list(tensor.shape), list(tensor.stride()))
        != (metadata["dtype"], metadata["shape"], metadata["stride"])
        or tensor.storage_offset() < 0
        or not tensor.is_contiguous()
    ):
        raise ValueError("payload tensor shape/dtype/stride mismatch")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("nonfinite payload tensor")


def _metadata(value):
    torch = _torch()
    if type(value) is torch.Tensor:
        return {"tensor_shape": list(value.shape), "tensor_dtype": str(value.dtype)}
    if type(value) is dict:
        return {key: _metadata(child) for key, child in value.items()}
    if type(value) in (tuple, list):
        return [_metadata(child) for child in value]
    return value


def _same(left, right):
    """Schema equality must not silently equate bool/int or list/tuple options."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(_same(left[key], right[key]) for key in left)
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _validate_model(saved, current):
    if type(saved) is not dict or set(saved) != {"schema", "aliases", "tensors"}:
        raise ValueError("model schema keys mismatch")
    if not _same(saved["schema"], current["schema"]) or not _same(
        saved["aliases"], current["aliases"]
    ):
        raise ValueError("model name/shape/dtype/alias schema mismatch")
    if set(saved["tensors"]) != set(current["tensors"]):
        raise ValueError("model canonical tensor keys mismatch")
    storages = set()
    for name, tensor in saved["tensors"].items():
        _validate_tensor(tensor, saved["schema"][name]["tensor"])
        identity = (tensor.untyped_storage()._cdata, tensor.storage_offset())
        if identity in storages:
            raise ValueError("unexpected canonical tensor alias")
        storages.add(identity)


def _validate_rng(saved, policy):
    torch = _torch()
    if type(saved) is not dict or set(saved) != {"python", "cpu", "cuda"}:
        raise ValueError("RNG schema mismatch")
    random.Random().setstate(saved["python"])
    torch.Generator(device="cpu").set_state(saved["cpu"])
    current = _rng(policy)
    if (current["cuda"] is None) != (saved["cuda"] is None):
        raise ValueError("CUDA RNG initialized-device mismatch")
    if saved["cuda"] is not None:
        state = saved["cuda"]
        if type(state) is not dict or set(state) != {"device", "count", "state"}:
            raise ValueError("CUDA RNG schema mismatch")
        if state["device"] != "cuda:0" or state["count"] != 1:
            raise ValueError("CUDA RNG device/count mismatch")
        if not torch.cuda.is_initialized():
            raise ValueError("CUDA RNG restore must not initialize CUDA")
        # Already initialized only; this path is deliberately untested on CPU.
        torch.Generator(device="cuda:0").set_state(state["state"])


def _validate_payload(saved, policy, reference, optimizer, config, version, binding):
    torch = _torch()
    required = {
        "format",
        "version",
        "binding",
        "config",
        "default_dtype",
        "policy",
        "reference",
        "optimizer",
        "rng",
    }
    if type(saved) is not dict or set(saved) != required or saved["format"] != FORMAT:
        raise ValueError("handoff schema/format mismatch")
    if type(saved["version"]) is not int or saved["version"] != version:
        raise ValueError("handoff version mismatch")
    if saved["binding"] != _binding(binding) or not _same(saved["config"], asdict(config)):
        raise ValueError("handoff binding/config mismatch")
    if saved["default_dtype"] != str(torch.get_default_dtype()):
        raise ValueError("default dtype mismatch")
    current_policy, current_reference = _model(policy), _model(reference)
    _validate_model(saved["policy"], current_policy)
    _validate_model(saved["reference"], current_reference)
    current = _optimizer(policy, optimizer)
    state = saved["optimizer"]
    if type(state) is not dict or set(state) != {"class", "defaults", "groups", "state"}:
        raise ValueError("optimizer schema mismatch")
    if any(not _same(state[key], current[key]) for key in ("class", "defaults", "groups")):
        raise ValueError("optimizer class/name/order/options mismatch")
    if set(state["state"]) != set(current["state"]):
        raise ValueError("optimizer named states mismatch")
    parameters = dict(policy.named_parameters())
    for group in state["groups"]:
        for name in group["names"]:
            entries = state["state"][name]
            allowed = (
                {"momentum_buffer"}
                if state["class"] == "SGD"
                else {"step", "exp_avg", "exp_avg_sq"}
            )
            if state["class"] == "AdamW" and group["options"]["amsgrad"]:
                allowed.add("max_exp_avg_sq")
            if type(entries) is not dict or (entries and set(entries) != allowed):
                raise ValueError("optimizer state keys mismatch")
            if state["class"] == "SGD" and entries and group["options"]["momentum"] == 0:
                raise ValueError("unexpected SGD momentum")
            param_meta = _local(parameters[name])[1]
            for key, entry in entries.items():
                if type(entry) is not dict or set(entry) != {"tensor", "meta"}:
                    raise ValueError("optimizer tensor schema mismatch")
                expected = param_meta
                if key == "step":
                    expected = {
                        "shape": [],
                        "stride": [],
                        "dtype": str(torch.get_default_dtype()),
                        "device": "cpu",
                        "distribution": None,
                    }
                if not _same(entry["meta"], expected):
                    raise ValueError("optimizer tensor name/shape/dtype/device mismatch")
                _validate_tensor(entry["tensor"], expected)
                if key == "step":
                    step = float(entry["tensor"])
                    if step < 0 or not step.is_integer():
                        raise ValueError("invalid AdamW scalar step")
    _validate_rng(saved["rng"], policy)
    # Canonical stored tensors may not hide extra aliases across models, moments
    # or RNG. Intended tied model aliases have one stored tensor plus alias names.
    ranges = []
    for tensor in _walk_tensors(saved):
        size = tensor.numel() * tensor.element_size()
        if size:
            ranges.append((str(tensor.device), tensor.data_ptr(), tensor.data_ptr() + size))
    ranges.sort()
    for first, second in itertools.pairwise(ranges):
        if first[0] == second[0] and first[2] > second[1]:
            raise ValueError("unexpected cross-state tensor storage alias")


def _restore_into(policy, reference, optimizer, config, fd, receipt, *, version, binding):
    """Drop the entire load/copy frame, then prove this consumer has no mapping."""
    rng = _restore_copies(
        policy, reference, optimizer, config, fd, receipt, version=version, binding=binding
    )
    _assert_unmapped(receipt)
    return rng


def _restore_copies(policy, reference, optimizer, config, fd, receipt, *, version, binding):
    """Called only by the Learner restoring-phase owner, never by cached aliases."""
    torch = _torch()
    receipt.validate(fd)
    if receipt.version != version or receipt.binding != _binding(binding):
        raise ValueError("receipt version/binding mismatch")
    # Bound end records, directory bytes/objects and each variable field before
    # Torch's eager ZIP reader or restricted pickle loader sees the artifact.
    preflight(
        lambda offset, count: os.pread(fd, count, offset),
        receipt.size,
        receipt.metadata_cap,
        receipt.storage_cap,
    )
    from mmap import MAP_PRIVATE

    with torch.serialization.set_default_mmap_options(MAP_PRIVATE):
        payload = torch.load(
            f"/proc/self/fd/{fd}", weights_only=True, mmap=True, map_location="cpu"
        )
    try:
        _validate_payload(payload, policy, reference, optimizer, config, version, binding)
        return _copy_payload(payload, policy, reference, optimizer)
    finally:
        # No view is returned; process exit/reap supplies the final ownership proof.
        del payload


def _copy_payload(payload, policy, reference, optimizer):
    """Copy an already completely validated candidate into owned destinations."""
    torch = _torch()
    with torch.no_grad():
        for model, name in ((policy, "policy"), (reference, "reference")):
            destinations = _model(model)["tensors"]
            for key, destination in destinations.items():
                destination.copy_(payload[name]["tensors"][key])
        parameters = dict(policy.named_parameters())
        for name, entries in payload["optimizer"]["state"].items():
            parameter = parameters[name]
            restored = {}
            for key, entry in entries.items():
                source = entry["tensor"]
                if key == "step":
                    restored[key] = source.clone()
                else:
                    # Allocate owned destination storage, never retain mmap COW state.
                    destination = torch.empty_like(parameter)
                    from torch.distributed.tensor import DTensor

                    local = (
                        destination.to_local() if isinstance(destination, DTensor) else destination
                    )
                    local.copy_(source)
                    restored[key] = destination
            if restored:
                optimizer.state[parameter] = restored
    # Return only tiny owned RNG copies; no mmap Tensor escapes this method.
    rng = {
        "python": payload["rng"]["python"],
        "cpu": payload["rng"]["cpu"].clone(),
        "cuda": None,
    }
    if payload["rng"]["cuda"] is not None:
        rng["cuda"] = {
            **payload["rng"]["cuda"],
            "state": payload["rng"]["cuda"]["state"].clone(),
        }
    return rng


def _restore_rng(rng):
    """Learner invokes after all model/optimizer validation and cleanup succeeds."""
    torch = _torch()
    random.setstate(rng["python"])
    torch.set_rng_state(rng["cpu"])
    if rng["cuda"] is not None:
        if not torch.cuda.is_initialized() or len(torch.cuda.default_generators) != 1:
            raise RuntimeError("CUDA context changed during restore")
        torch.cuda.default_generators[0].set_state(rng["cuda"]["state"])
