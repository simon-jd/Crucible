"""Narrow versioned training-state container: bounded tagged JSON + raw tensors.

No pickle, mmap, hidden storage reconstruction or CUDA discovery. CUDA transfer
requires an already initialized single-device context; that branch is unverified.
"""

import json
import math
import os
import struct
from dataclasses import asdict, dataclass

from rl.learner import state_handoff as state

MAGIC = b"CRUCSTR1"
HEADER = struct.Struct("<8sQ")
MAX_DEPTH = 32
MAX_ITEMS = 100000
MAX_DIM = 8
DTYPES = {
    "torch.float32": 4,
    "torch.bfloat16": 2,
    "torch.float16": 2,
    "torch.float64": 8,
    "torch.int64": 8,
    "torch.int32": 4,
    "torch.int16": 2,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.bool": 1,
}


@dataclass(frozen=True)
class SerialBudget:
    max_bytes: int
    host_baseline: int
    cpu_candidates: int
    cpu_destinations: int
    gpu_candidates: int = 0
    gpu_destinations: int = 0
    gpu_baseline: int = 0
    gpu_transient: int = 0
    gpu_limit: int = 24_000_000_000
    ram_limit: int = state.MAX_RAM
    storage_cap: int = 8192
    metadata_cap: int = 32768
    retained_bytes: int = 0
    weights_bytes: int = 0
    weights_slots: int = 0

    @property
    def metadata_charge(self):
        # JSON trees, tags, decoded schema, maps/protocol scans and 64KiB I/O.
        return 64 * self.metadata_cap + 262144

    def check(self, source_bytes=0, cpu=0, gpu=0):
        for name, value in asdict(self).items():
            state._integer(value, name)
        if not (
            0 < self.host_baseline
            and 0 < self.max_bytes
            and 0 < self.metadata_cap <= 8_000_000
            and 0 < self.storage_cap
            and 0 < self.ram_limit <= state.MAX_RAM
            and 0 < self.gpu_limit <= 24_000_000_000
            and self.weights_slots <= 1
        ):
            raise ValueError("invalid serial budget")
        if bool(self.weights_slots) != bool(self.weights_bytes):
            raise ValueError("weights bytes and slot must be declared together")
        if (
            cpu > self.cpu_candidates
            or gpu > self.gpu_candidates
            or cpu > self.cpu_destinations
            or gpu > self.gpu_destinations
        ):
            raise ValueError("candidate aggregate budget exceeded before allocation")
        common = self.host_baseline + self.retained_bytes + self.weights_bytes
        common += self.metadata_charge + 2 * self.storage_cap
        restore = common + source_bytes + self.cpu_candidates + self.cpu_destinations
        capture = common + self.max_bytes + self.cpu_destinations
        vram = self.gpu_baseline + self.gpu_candidates + self.gpu_destinations + self.gpu_transient
        if max(restore, capture) > self.ram_limit or vram > self.gpu_limit:
            raise ValueError("serial RAM/VRAM budget exceeded")
        return {"restore": restore, "capture": capture, "vram": vram}


def _cuda(torch, devices):
    if devices == {"cpu"} or not devices:
        return
    if devices - {"cpu", "cuda:0"} or not torch.cuda.is_initialized():
        raise ValueError("stream requires already initialized cuda:0")
    if len(torch.cuda.default_generators) != 1:
        raise ValueError("stream requires one initialized CUDA generator")


def _write(fd, data):
    view = memoryview(data)
    while view:
        n = os.write(fd, view[:65536])
        if n <= 0:
            raise OSError("short stream write")
        view = view[n:]


def _raw(tensor):
    return memoryview(tensor.detach().reshape(-1).view(state._torch().uint8).numpy()).cast("B")


def _encode(payload, budget):
    torch = state._torch()
    records, tensors, count = [], [], 0

    def visit(value, depth=0):
        nonlocal count
        count += 1
        if depth > MAX_DEPTH or count > min(MAX_ITEMS, budget.metadata_cap // 8):
            raise ValueError("manifest structure allowance exceeded")
        if type(value) is torch.Tensor:
            value, meta = state._local(value)
            nbytes = value.numel() * value.element_size()
            if (
                meta["dtype"] not in DTYPES
                or nbytes > budget.storage_cap
                or len(meta["shape"]) > MAX_DIM
            ):
                raise ValueError("unsupported dtype/storage cap")
            if len(records) >= min(MAX_ITEMS, budget.metadata_cap // 128):
                raise ValueError("stream record count allowance before write")
            stride, product = [], 1
            for dimension in reversed(meta["shape"]):
                stride.insert(0, product)
                product *= max(dimension, 1)
            if stride != meta["stride"]:
                raise ValueError("stream requires canonical contiguous strides")
            index = len(records)
            records.append(
                {
                    "dtype": meta["dtype"],
                    "device": meta["device"],
                    "shape": meta["shape"],
                    "stride": meta["stride"],
                    "offset": (records[-1]["offset"] + records[-1]["bytes"] if records else 0),
                    "bytes": nbytes,
                }
            )
            tensors.append(value)
            return ["tensor", index]
        if value is None or type(value) in (bool, int, float, str):
            if type(value) is int and abs(value) > 2**63 - 1:
                raise ValueError("manifest integer outside signed63-bit magnitude")
            if type(value) is float and not math.isfinite(value):
                raise ValueError("nonfinite manifest number")
            if type(value) is str and len(value.encode()) > budget.metadata_cap:
                raise ValueError("manifest string too long")
            return ["atom", value]
        if type(value) in (tuple, list):
            return [
                "tuple" if type(value) is tuple else "list",
                [visit(v, depth + 1) for v in value],
            ]
        if type(value) is dict:
            return ["dict", [[visit(k, depth + 1), visit(v, depth + 1)] for k, v in value.items()]]
        raise ValueError("unsupported manifest primitive")

    manifest = {"format": "crucible-stream-v1", "state": visit(payload), "records": records}
    encoded = bytearray()
    for part in json.JSONEncoder(separators=(",", ":"), ensure_ascii=True).iterencode(manifest):
        if len(encoded) + len(part) > budget.metadata_cap:
            raise ValueError("manifest byte cap exceeded before write")
        encoded.extend(part.encode("ascii"))
    return bytes(encoded), tensors


def _json(data, cap):
    # Reject deep/large decoded structures BEFORE the stdlib decoder allocates them.
    depth, tokens, quoted, escaped = 0, 0, False, False
    for byte in data:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
            continue
        if byte == 34:
            quoted = True
            tokens += 1
        elif byte in (91, 123):
            depth += 1
            tokens += 1
        elif byte in (93, 125):
            depth -= 1
        elif byte in (44, 58):
            tokens += 1
        if depth < 0 or depth > 2 * MAX_DEPTH + 8 or tokens > min(MAX_ITEMS, cap // 4):
            raise ValueError("decoded manifest structure allowance exceeded")
    if quoted or depth:
        raise ValueError("malformed manifest nesting")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate manifest key")
            result[key] = value
        return result

    def integer(text):
        if len(text) > 20 or abs(int(text)) > 2**63 - 1:
            raise ValueError("manifest integer overflow")
        return int(text)

    def floating(text):
        result = float(text)
        if not math.isfinite(result):
            raise ValueError("nonfinite manifest number")
        return result

    return json.loads(
        data,
        object_pairs_hook=pairs,
        parse_int=integer,
        parse_float=floating,
        parse_constant=lambda _: floating("nan"),
    )


@dataclass(frozen=True)
class TensorRecord:
    index: int
    metadata: dict


def _decode(tree, records, cap):
    count, used = 0, set()

    def visit(node, depth=0):
        nonlocal count
        count += 1
        if depth > MAX_DEPTH or count > min(MAX_ITEMS, cap // 8):
            raise ValueError("manifest decoded allowance exceeded")
        if type(node) is not list or len(node) != 2 or type(node[0]) is not str:
            raise ValueError("invalid manifest tag")
        tag, value = node
        if tag == "atom":
            if value is not None and type(value) not in (bool, int, float, str):
                raise ValueError("invalid manifest atom")
            return value
        if tag == "tensor":
            if type(value) is not int or not 0 <= value < len(records) or value in used:
                raise ValueError("invalid/aliased tensor record")
            used.add(value)
            return TensorRecord(value, records[value])
        if tag in ("list", "tuple") and type(value) is list:
            values = [visit(v, depth + 1) for v in value]
            return values if tag == "list" else tuple(values)
        if tag == "dict" and type(value) is list:
            result = {}
            for pair in value:
                if type(pair) is not list or len(pair) != 2:
                    raise ValueError("invalid tagged dictionary")
                key, val = (visit(v, depth + 1) for v in pair)
                if type(key) not in (str, int, float, bool, tuple, type(None)) or key in result:
                    raise ValueError("invalid/duplicate tagged key")
                result[key] = val
            return result
        raise ValueError("unknown manifest tag")

    payload = visit(tree)
    if used != set(range(len(records))):
        raise ValueError("unreferenced storage record")
    return payload


def read_manifest(fd, receipt, budget):
    """No Torch/tensor construction; receipt SHA streams bytes in bounded chunks."""
    budget.check(receipt.size)
    tree, records, start, cpu, gpu = _read_structure(fd, receipt, budget)
    budget.check(receipt.size, cpu, gpu)
    return _decode(tree, records, budget.metadata_cap), records, start


def read_metadata(fd, receipt, limits):
    """Structural reader only; caller must independently admit its allocation plan.

    Does not authorize restore or claim candidate/destination memory is available.
    Receipt validation and all original extent/structure checks remain mandatory.
    """
    tree, records, start, _, _ = _read_structure(fd, receipt, limits)
    return _decode(tree, records, limits.metadata_cap), records, start


def _read_structure(fd, receipt, limits):
    budget = limits
    for name in ("max_bytes", "metadata_cap", "storage_cap"):
        state._integer(getattr(limits, name), name, 1)
    if limits.metadata_cap > 8_000_000:
        raise ValueError("stream metadata cap")
    if receipt.size > budget.max_bytes:
        raise ValueError("stream input payload cap")
    receipt.validate(fd)
    header = os.pread(fd, HEADER.size, 0)
    if len(header) != HEADER.size:
        raise ValueError("truncated stream header")
    magic, length = HEADER.unpack(header)
    if (
        magic != MAGIC
        or not 0 < length <= budget.metadata_cap
        or HEADER.size + length > receipt.size
    ):
        raise ValueError("stream header/manifest bounds")
    data = os.pread(fd, length, HEADER.size)
    if len(data) != length:
        raise ValueError("truncated manifest")
    manifest = _json(data, budget.metadata_cap)
    if type(manifest) is not dict or set(manifest) != {"format", "state", "records"}:
        raise ValueError("manifest keys mismatch")
    if manifest["format"] != "crucible-stream-v1" or type(manifest["records"]) is not list:
        raise ValueError("manifest format/records mismatch")
    records = manifest["records"]
    if not 0 < len(records) <= min(MAX_ITEMS, budget.metadata_cap // 128):
        raise ValueError("record count allowance")
    offset, cpu, gpu = 0, 0, 0
    for record in records:
        if type(record) is not dict or set(record) != {
            "dtype",
            "device",
            "shape",
            "stride",
            "offset",
            "bytes",
        }:
            raise ValueError("record schema")
        dtype, device = record["dtype"], record["device"]
        if type(dtype) is not str or dtype not in DTYPES or device not in ("cpu", "cuda:0"):
            raise ValueError("unsupported record dtype/device")
        shape, stride = record["shape"], record["stride"]
        if (
            type(shape) is not list
            or type(stride) is not list
            or len(shape) != len(stride)
            or len(shape) > MAX_DIM
        ):
            raise ValueError("record dimensions")
        elements, expected_stride = 1, []
        for size in reversed(shape):
            if type(size) is not int or not 0 <= size <= budget.storage_cap:
                raise ValueError("record dimension bounds")
            expected_stride.insert(0, elements)
            elements *= max(size, 1)
            if elements > max(1, budget.storage_cap):
                raise ValueError("record shape product exceeds cap")
        if any(type(x) is not int for x in stride) or stride != expected_stride:
            raise ValueError("record must be contiguous")
        nbytes = (0 if 0 in shape else elements) * DTYPES[dtype]
        if (
            type(record["bytes"]) is not int
            or record["bytes"] != nbytes
            or nbytes > budget.storage_cap
        ):
            raise ValueError("record storage byte extent")
        if type(record["offset"]) is not int or record["offset"] != offset:
            raise ValueError("record overlap/gap/offset")
        offset += nbytes
        if device == "cpu":
            cpu += nbytes
        else:
            gpu += nbytes
    if HEADER.size + length + offset != receipt.size or receipt.size > budget.max_bytes:
        raise ValueError("stream truncated/trailing bytes")
    return manifest["state"], records, HEADER.size + length, cpu, gpu


def _check_tensor(record, expected):
    if type(record) is not TensorRecord:
        raise ValueError("expected tensor record")
    meta = record.metadata
    if any(not state._same(meta[k], expected[k]) for k in ("shape", "stride", "dtype", "device")):
        raise ValueError("tensor metadata mismatch before allocation")


def _prevalidate(saved, policy, reference, optimizer, config, version, binding):
    """All tensor reconstruction metadata is checked against trusted destinations."""
    if type(saved) is not dict or set(saved) != {
        "format",
        "version",
        "binding",
        "config",
        "default_dtype",
        "policy",
        "reference",
        "optimizer",
        "rng",
    }:
        raise ValueError("state schema")
    if (
        saved["format"] != state.FORMAT
        or type(saved["version"]) is not int
        or saved["version"] != version
    ):
        raise ValueError("state version/format")
    if saved["binding"] != state._binding(binding) or not state._same(
        saved["config"], asdict(config)
    ):
        raise ValueError("state binding/config")
    torch = state._torch()
    if saved["default_dtype"] != str(torch.get_default_dtype()):
        raise ValueError("default dtype mismatch")
    for name, model in (("policy", policy), ("reference", reference)):
        current, value = state._model(model), saved[name]
        if type(value) is not dict or set(value) != {"schema", "aliases", "tensors"}:
            raise ValueError("model schema")
        if not state._same(value["schema"], current["schema"]) or not state._same(
            value["aliases"], current["aliases"]
        ):
            raise ValueError("model schema/alias mismatch")
        if type(value["tensors"]) is not dict or set(value["tensors"]) != set(current["tensors"]):
            raise ValueError("model names mismatch")
        for key, tensor in value["tensors"].items():
            _check_tensor(tensor, current["schema"][key]["tensor"])
    current, value = state._optimizer(policy, optimizer), saved["optimizer"]
    if type(value) is not dict or set(value) != set(current):
        raise ValueError("optimizer schema")
    if any(not state._same(value[k], current[k]) for k in ("class", "defaults", "groups")):
        raise ValueError("optimizer options/order")
    if type(value["state"]) is not dict or set(value["state"]) != set(current["state"]):
        raise ValueError("optimizer names")
    parameters = dict(policy.named_parameters())
    for group in value["groups"]:
        for name in group["names"]:
            entries = value["state"][name]
            allowed = (
                {"momentum_buffer"}
                if value["class"] == "SGD"
                else {"step", "exp_avg", "exp_avg_sq"}
            )
            if value["class"] == "AdamW" and group["options"]["amsgrad"]:
                allowed.add("max_exp_avg_sq")
            if type(entries) is not dict or (entries and set(entries) != allowed):
                raise ValueError("optimizer entry keys")
            for key, entry in entries.items():
                expected = state._local(parameters[name])[1]
                if key == "step":
                    expected = {
                        "shape": [],
                        "stride": [],
                        "dtype": str(torch.get_default_dtype()),
                        "device": "cpu",
                        "distribution": None,
                    }
                if (
                    type(entry) is not dict
                    or set(entry) != {"tensor", "meta"}
                    or not state._same(entry["meta"], expected)
                ):
                    raise ValueError("optimizer entry metadata")
                _check_tensor(entry["tensor"], expected)
    rng, current_rng = saved["rng"], state._rng(policy)
    if type(rng) is not dict or set(rng) != {"python", "cpu", "cuda"}:
        raise ValueError("RNG schema")
    # Python RNG validation allocates no tensor and does not change global RNG.
    state.random.Random().setstate(rng["python"])
    _check_tensor(rng["cpu"], state._local(current_rng["cpu"])[1])
    if (rng["cuda"] is None) != (current_rng["cuda"] is None):
        raise ValueError("RNG initialized-device mismatch")
    if rng["cuda"] is not None:
        value = rng["cuda"]
        if (
            type(value) is not dict
            or set(value) != {"device", "count", "state"}
            or value["device"] != "cuda:0"
            or type(value["count"]) is not int
            or value["count"] != 1
        ):
            raise ValueError("CUDA RNG metadata")
        _check_tensor(value["state"], state._local(current_rng["cuda"]["state"])[1])


def _allocate_record(torch, record):
    return torch.empty(record["shape"], dtype=getattr(torch, record["dtype"][6:]), device="cpu")


def _read_record(fd, tensor, offset):
    view = _raw(tensor)
    while view:
        count = os.preadv(fd, [view[:65536]], offset)
        if count <= 0:
            raise ValueError("truncated raw storage")
        view, offset = view[count:], offset + count


def _materialize(value, tensors):
    if type(value) is TensorRecord:
        return tensors[value.index]
    if type(value) is dict:
        return {_materialize(k, tensors): _materialize(v, tensors) for k, v in value.items()}
    if type(value) in (list, tuple):
        values = [_materialize(v, tensors) for v in value]
        return values if type(value) is list else tuple(values)
    return value


def restore(policy, reference, optimizer, config, fd, receipt, *, version, binding, budget):
    saved, records, start = read_manifest(fd, receipt, budget)
    if receipt.version != version or receipt.binding != state._binding(binding):
        raise ValueError("receipt version/binding")
    _prevalidate(saved, policy, reference, optimizer, config, version, binding)
    torch = state._torch()
    devices = {record["device"] for record in records}
    _cuda(torch, devices)
    tensors = []
    for record in records:
        staging = _allocate_record(torch, record)
        _read_record(fd, staging, start + record["offset"])
        tensors.append(staging if record["device"] == "cpu" else staging.to(record["device"]))
        del staging
    payload = _materialize(saved, tensors)
    state._validate_payload(payload, policy, reference, optimizer, config, version, binding)
    rng = state._copy_payload(payload, policy, reference, optimizer)
    if "cuda:0" in devices:
        torch.cuda.synchronize(0)
    return rng


def capture(learner, fd, *, binding, budget):
    budget.check()
    payload, policy, optimizer = state._payload(learner, binding)
    state._validate_payload(
        payload, policy, learner.reference, optimizer, learner.config, payload["version"], binding
    )
    encoded, tensors = _encode(payload, budget)
    cpu = sum(t.numel() * t.element_size() for t in tensors if t.device.type == "cpu")
    gpu = sum(t.numel() * t.element_size() for t in tensors if t.device.type == "cuda")
    budget.check(cpu=cpu, gpu=gpu)
    total = HEADER.size + len(encoded) + cpu + gpu
    if total > budget.max_bytes:
        raise ValueError("stream payload cap")
    _cuda(state._torch(), {str(t.device) for t in tensors})
    stat = os.fstat(fd)
    if stat.st_size or state.fcntl.fcntl(fd, state.F_GET_SEALS):
        raise ValueError("stream capture requires empty unsealed output")
    signature = state._signature({k: v for k, v in payload.items() if k != "rng"})
    _write(fd, HEADER.pack(MAGIC, len(encoded)))
    _write(fd, encoded)
    for tensor in tensors:
        host = tensor if tensor.device.type == "cpu" else tensor.cpu()
        _write(fd, _raw(host))
        del host
    current = state._payload(learner, binding)[0]
    if (
        state._signature({k: v for k, v in current.items() if k != "rng"}) != signature
        or learner.policy is not policy
        or learner.optimizer is not optimizer
    ):
        raise RuntimeError("state changed during stream capture")
    torch = state._torch()
    if current["rng"]["python"] != payload["rng"]["python"] or not torch.equal(
        current["rng"]["cpu"], payload["rng"]["cpu"]
    ):
        raise RuntimeError("RNG changed during stream capture")
    if (current["rng"]["cuda"] is None) != (payload["rng"]["cuda"] is None):
        raise RuntimeError("CUDA RNG ownership changed during capture")
    if current["rng"]["cuda"] is not None and not torch.equal(
        current["rng"]["cuda"]["state"], payload["rng"]["cuda"]["state"]
    ):
        raise RuntimeError("CUDA RNG changed during capture")
    if os.fstat(fd).st_size != total:
        raise ValueError("stream capture length mismatch")
    state.fcntl.fcntl(fd, state.F_ADD_SEALS, state.SEALS)
    return {"bytes": total, "charge": budget.check(), "format": "crucible-stream-v1"}
