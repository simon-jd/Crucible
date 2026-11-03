"""Exclusive owned training/rollout phases; full-model/GPU admission UNVERIFIED."""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock

from rl.learner import handoff_protocol as protocol
from rl.learner import handoff_stream as stream
from rl.learner import state_handoff as state
from rl.rollout.engine import RolloutEngine
from rl.sync import ram_weights as weights


@dataclass
class WeightLease:
    capacity: int
    fd: int | None = None
    identity: tuple | None = None
    checkpoint: weights.RamPolicyCheckpoint | None = None
    phase: str = "reserved"
    closed: bool = False


class PhaseBridge:
    def __init__(
        self, *, budget, contract, model_path, base_revision, assets, runtime_root, timeout=30
    ):
        budget.check()
        self._budget = budget
        # Copy through bounded canonical primitives: caller cannot change a live contract.
        self._contract = stream._json(
            weights.encode(contract, budget.metadata_cap), budget.metadata_cap
        )
        self._contract_digest = weights.digest(self._contract)
        self._model_path = str(Path(model_path).absolute())
        self._assets = tuple(tuple(a) for a in assets)
        weights.verify_assets(self._model_path, self._assets)
        self._base_revision = base_revision
        if not isinstance(base_revision, str) or not base_revision:
            raise ValueError("base revision required")
        root = Path(__file__).resolve().parents[2]
        self._runtime_root = Path(runtime_root).absolute()
        if root not in self._runtime_root.parents or self._runtime_root.is_symlink():
            raise ValueError("owned project runtime root required")
        self._owner = state.SnapshotOwner(
            max_bytes=budget.source_cap,
            other_resident_bytes=budget.host_baseline,
            ram_limit=budget.ram_limit,
            metadata_cap=budget.metadata_cap,
            storage_cap=budget.storage_cap,
            timeout=timeout,
        )
        self._lock = RLock()
        self._source = self._weight = self._engine = None
        self._source_producer = None
        self._source_retirement = None
        self._failed = self._closed = self._unknown_process = False
        self._highwater = -1
        self._noop_dropped_stale = 0
        self._events = []

    @property
    def audit(self):
        pending = getattr(self._owner, "_serial_pending", None)
        retirement = getattr(self._owner, "_serial_retirement", None)
        training_slots = len(self._owner.snapshots)
        training_slots += 0 if pending is None else pending.reserved_slots
        if retirement is not None and retirement.source.fd not in self._owner.snapshots:
            training_slots += retirement.reserved_slots
        return {
            "failed": self._failed,
            "closed": self._closed,
            "source": None if self._source is None else asdict(self._source.receipt),
            "weight_phase": None if self._weight is None else self._weight.phase,
            "slots": training_slots + int(self._weight is not None and not self._weight.closed),
            "weight_reserved": 0
            if self._weight is None or self._weight.closed
            else self._weight.capacity,
            "unknown_process": self._unknown_process,
            "noop_dropped_stale": self._noop_dropped_stale,
            "events": tuple(self._events),
        }

    def _ready(self, training=False):
        if self._closed or self._failed or self._owner.failed or self._unknown_process:
            raise RuntimeError("bridge closed/failed/uncertain")
        if training and (self._weight is not None or self._engine is not None):
            raise RuntimeError("retire owned weights/engine before training")
        self._budget.check(0 if self._source is None else self._source.receipt.size)

    def _training_budget(self, budget):
        if budget.weights_slots or budget.weights_bytes or budget.retained_bytes:
            raise ValueError("bridge requires empty training weights reservation")
        if (
            budget.max_bytes > self._budget.source_cap
            or budget.host_baseline < self._budget.host_baseline
        ):
            raise ValueError("training cap/baseline exceeds bridge plan")

    def capture_initial(self, command, *, version, binding, budget):
        with self._lock:
            self._ready(training=True)
            self._training_budget(budget)
            if self._source is not None or self._owner.snapshots:
                raise RuntimeError("initial source already exists")
            try:
                self._source = self._owner.capture_stream(
                    command, version=version, binding=binding, budget=budget
                )
                self._source_producer = self._owner.last_process
                self._validate_accepted_source()
                self._highwater = version
                return self._source.receipt
            except BaseException:
                self._failed = True
                raise

    def _validate_accepted_source(self):
        if self._source is None or self._owner.snapshots.get(self._source.fd) is not self._source:
            raise ValueError("owned accepted source required")
        process = self._source_producer
        if process is None or process.returncode != 0:
            raise ValueError("original producer exit0/reap required")
        saved, _, _ = stream.read_metadata(
            self._source.fd, self._source.receipt, self._budget.source_limits
        )
        weights.validate_source(saved, self._source.receipt, self._contract)
        self._budget.check(self._source.receipt.size)

    def advance_learner(self, command, *, version, binding, budget, sample_versions):
        """Header preflight before destructive consumption; command owns actual batch.

        Each item is one complete group's sample versions. Caller binds these
        headers to its fixed worker batch. Unexpected worker no-op fails receipt
        acceptance after consumption; it must never manufacture a version.
        Audit noop_dropped_stale counts valid all-stale submissions only. Mixed
        batches belong to actual learner metrics; repeated submissions count again.
        """
        with self._lock:
            self._ready(training=True)
            self._training_budget(budget)
            if self._source is None or type(version) is not int or version <= self._highwater:
                raise ValueError("strict new source version required")
            if type(sample_versions) not in (tuple, list) or len(sample_versions) > 1024:
                raise ValueError("bounded group headers required")
            eligible = False
            submitted = 0
            for group in sample_versions:
                if (
                    type(group) not in (tuple, list)
                    or len(group) != self._contract["config"]["group_size"]
                ):
                    raise ValueError("complete group header required")
                if (
                    any(type(v) is not int or not 0 <= v <= self._highwater for v in group)
                    or len(set(group)) != 1
                ):
                    raise ValueError("invalid group policy version")
                eligible |= self._highwater - group[0] <= self._contract["config"]["max_staleness"]
                submitted += len(group)
            if not eligible:
                # All headers validated before the only accounting mutation.
                self._noop_dropped_stale += submitted
                return self._source.receipt  # known no-op: no child, no retirement
            try:
                source = self._owner.advance_consuming(
                    self._source, command, version=version, binding=binding, budget=budget
                )
                self._source = source
                self._source_producer = self._owner.last_process
                self._validate_accepted_source()
                self._highwater = version
                return source.receipt
            except BaseException:
                self._failed = True
                raise

    def export_inference(self):
        with self._lock:
            self._ready(training=True)
            self._validate_accepted_source()
            if len(self._owner.snapshots) != 1 or self._owner.disk_count:
                raise ValueError("two checkpoint admission")
            self._failed = True  # reserve BEFORE acquisition, including unknown syscall result
            lease = WeightLease(self._budget.max_bytes)
            self._weight = lease
            fd = None
            try:
                lease.phase = "acquiring"
                fd = os.memfd_create("crucible-inference", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
                lease.fd = fd
                lease.phase = "identity-unknown"
                info = os.fstat(fd)
                lease.identity = (info.st_dev, info.st_ino)
                lease.phase = "exporting"
                self._owner.last_process = None
                self._unknown_process = True
                python = str(Path(__file__).resolve().parents[2] / ".venv-ml/bin/python")
                self._owner._child(
                    [python, "-B", "-m", "rl.sync.ram_export"],
                    fd,
                    {
                        "CUDA_VISIBLE_DEVICES": "",
                        "CRUCIBLE_RAM_BUDGET": json.dumps(asdict(self._budget)),
                        "CRUCIBLE_RAM_CONTRACT": self._contract_digest,
                        "CRUCIBLE_RAM_PROVENANCE": weights.provenance(
                            self._model_path, self._base_revision, self._assets
                        ),
                    },
                    source=self._source,
                )
                self._unknown_process = False
                receipt = weights.make_receipt(
                    fd,
                    lease.identity,
                    self._source.receipt.version,
                    self._source.receipt.binding,
                    self._budget,
                )
                checkpoint = weights.RamPolicyCheckpoint(
                    self._model_path,
                    receipt.sha256,
                    receipt.version,
                    receipt,
                    self._contract_digest,
                    self._source.receipt.sha256,
                    self._base_revision,
                    self._assets,
                )
                records, _ = weights.read_weights(fd, checkpoint)
                actual = {r["name"]: {k: r[k] for k in ("shape", "dtype")} for r in records}
                if actual != self._contract["inference"]:
                    raise ValueError("export inference schema")
                lease.checkpoint, lease.phase = checkpoint, "accepted"
                self._source.receipt.validate(self._source.fd)
                self._events.append(
                    {
                        "exported": checkpoint.policy_version,
                        "pid": self._owner.last_process.pid,
                        "exitcode": self._owner.last_process.returncode,
                    }
                )
                self._failed = False
                return checkpoint
            except BaseException:
                if fd is not None:
                    lease.fd = fd
                process = self._owner.last_process
                if process is not None and process.returncode is not None:
                    self._unknown_process = False
                raise  # quarantine output; no unknown close or implicit retry

    def start_rollout(self, sampling=None, *, _backend_factory=None):
        with self._lock:
            self._ready()
            if _backend_factory is None and not (
                self._budget.gpu_total > 0
                and self._budget.gpu_baseline > 0
                and self._budget.rollout_host > 0
            ):
                raise ValueError("actual rollout requires explicit host/GPU occupancy plan")
            lease = self._weight
            if lease is None or lease.phase != "accepted" or self._engine is not None:
                raise RuntimeError("accepted exclusive weights required")
            weights.verify_assets(self._model_path, self._assets)
            lease.checkpoint.receipt.validate(lease.fd)
            from rl.rollout._worker import VLLMWorker

            factory = VLLMWorker if _backend_factory is None else _backend_factory

            def backend(checkpoint, config, **kwargs):
                return factory(
                    checkpoint, config, ram_fd=lease.fd, runtime_root=self._runtime_root, **kwargs
                )

            lease.phase = "starting"
            try:
                self._engine = RolloutEngine(lease.checkpoint, sampling, _backend_factory=backend)
                lease.phase = "active"
                return self._engine.policy_version
            except BaseException as error:
                self._engine = getattr(error, "recovery_engine", None)
                if self._engine is None:
                    self._unknown_process = True
                self._failed = True
                raise

    def generate(self, prompts, group_size, policy_version):
        with self._lock:
            self._ready()
            if self._engine is None:
                raise RuntimeError("no active rollout")
            try:
                return self._engine.generate(prompts, group_size, policy_version)
            except BaseException:
                self._failed = True
                raise

    def retire_rollout(self):
        with self._lock:
            if self._unknown_process:
                raise RuntimeError("unknown process retains weight reservation")
            previous_failed = self._failed
            self._failed = True
            try:
                if self._engine is not None:
                    self._engine.close()
                    self._engine = None
                lease = self._weight
                if lease is not None and not lease.closed:
                    if lease.fd is None or lease.identity is None:
                        raise RuntimeError("unknown weight acquisition/identity")
                    if lease.checkpoint is not None:
                        protocol.prove_absent(os.getpid(), lease.checkpoint.receipt, lease.fd)
                    lease.phase = "releasing"
                    protocol._close_identity(lease.fd, lease.identity)
                    lease.closed, lease.phase = True, "closed"
                self._weight = None
                self._failed = previous_failed
            except BaseException:
                self._failed = True
                raise

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._failed = True
            self.retire_rollout()
            if self._owner.failed:
                raise RuntimeError("training ownership uncertain; retain owner")
            if self._source is not None:
                if self._source_retirement is None:
                    self._source_retirement = protocol.SourceRetirement(self._source)
                retirement = self._source_retirement
                if not retirement.close_confirmed:
                    retirement.phase = "closing"
                    protocol.prove_absent(os.getpid(), self._source.receipt, self._source.fd)
                    protocol._close_identity(
                        self._source.fd, (self._source.receipt.device, self._source.receipt.inode)
                    )
                    retirement.close_confirmed = True
                    retirement.phase = "closed"
                dict.pop(self._owner.snapshots, self._source.fd, None)
                self._source = None
            self._owner.close()
            self._closed = True
