"""Torch-free contracts for owned anonymous inference weights (GPU UNVERIFIED)."""

import hashlib
import json
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

from rl.learner import handoff_stream as stream
from rl.learner import state_handoff as state

HEADER = struct.Struct("<8sQ")
MAGIC = b"CRUCWGT1"


def encode(value, cap):
    result = bytearray()
    for chunk in json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), allow_nan=False
    ).iterencode(value):
        raw = chunk.encode("ascii")
        if len(result) + len(raw) > cap:
            raise ValueError("weight metadata cap")
        result.extend(raw)
    return bytes(result)


def digest(value):
    return hashlib.sha256(encode(value, 8_000_000)).hexdigest()


def provenance(model_path, base_revision, assets):
    return digest({"model_path": model_path, "base_revision": base_revision, "assets": assets})


@dataclass(frozen=True)
class BridgeBudget:
    source_cap: int
    max_bytes: int
    host_baseline: int
    storage_cap: int = 8192
    metadata_cap: int = 32768
    ram_limit: int = 16_000_000_000
    rollout_host: int = 0
    gpu_baseline: int = 0
    gpu_total: int = 0

    def check(self, source_bytes=0):
        for name, value in asdict(self).items():
            state._integer(value, name)
        if not (
            0 < self.source_cap
            and 0 < self.max_bytes
            and 0 < self.host_baseline
            and 0 < self.storage_cap
            and 0 < self.metadata_cap <= 8_000_000
            and 0 < self.ram_limit <= state.MAX_RAM
        ):
            raise ValueError("bridge bounds/nonzero baseline")
        if source_bytes > self.source_cap:
            raise ValueError("bridge source cap")
        # Reserve future weights even before acquisition. No eviction assumptions.
        common = self.host_baseline + self.source_cap + self.max_bytes
        common += 64 * self.metadata_cap + 262144
        export = common + 3 * self.storage_cap
        rollout = common + self.rollout_host + 2 * self.storage_cap
        if (
            max(export, rollout) > self.ram_limit
            or self.gpu_baseline + self.gpu_total > 24_000_000_000
        ):
            raise ValueError("bridge RAM/VRAM budget")
        return {
            "export": export,
            "rollout": rollout,
            "future_weights_slots": 1,
            "future_weights_bytes": self.max_bytes,
        }

    @property
    def source_limits(self):
        return Limits(self.source_cap, self.metadata_cap, self.storage_cap)


@dataclass(frozen=True)
class Limits:
    max_bytes: int
    metadata_cap: int
    storage_cap: int


@dataclass(frozen=True)
class RamPolicyCheckpoint:
    model_path: str
    revision: str
    policy_version: int
    receipt: state.Receipt
    contract: str
    source_sha256: str
    base_revision: str
    assets: tuple

    def __post_init__(self):
        if not Path(self.model_path).is_absolute():
            raise ValueError("absolute immutable assets path required")
        if self.revision != self.receipt.sha256 or self.policy_version != self.receipt.version:
            raise ValueError("RAM checkpoint version/digest")

    def ack(self):
        return {
            "weights": self.revision,
            "source": self.source_sha256,
            "version": self.policy_version,
            "contract": self.contract,
            "assets": digest(self.assets),
            "provenance": provenance(self.model_path, self.base_revision, self.assets),
            "verified": "all-coordinates",
        }


def verify_assets(path, assets):
    """Exact caller-pinned config/tokenizer files; immutable HF links are allowed.

    This is separate from disk checkpoint verification and never reads base weights.
    """
    if not assets or "config.json" not in dict(assets):
        raise ValueError("pinned config/assets required")
    names = set()
    for name, expected in assets:
        if name in names or Path(name).name != name or name.endswith((".bin", ".safetensors")):
            raise ValueError("invalid asset name")
        names.add(name)
        h = hashlib.sha256()
        with (Path(path) / name).open("rb") as source:
            total = 0
            while block := source.read(65536):
                total += len(block)
                if total > 32_000_000:
                    raise ValueError("asset extent")
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError("base asset digest")


def verify_qwen_assets(checkpoint):
    """Actual runtime's fixed local Qwen profile; CPU fixtures do not call this."""
    revision = "7ae557604adf67be50417f59c2c2f167def9a775"
    path = (
        Path("/home/postedism/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots")
        / revision
    )
    required = {
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
    }
    if checkpoint.model_path != str(path) or checkpoint.base_revision != revision:
        raise ValueError("actual RAM loader requires fixed local Qwen revision")
    if {name for name, _ in checkpoint.assets} != required:
        raise ValueError("complete Qwen config/tokenizer asset pins required")
    if {p.name for p in path.iterdir()} != required | {"README.md", "LICENSE", "model.safetensors"}:
        raise ValueError("unexpected local Qwen asset set")
    verify_assets(path, checkpoint.assets)
    with (path / "config.json").open("rb") as source:
        raw = source.read(65537)
    if len(raw) > 65536:
        raise ValueError("Qwen config cap")
    config = stream._json(raw, 65536)
    if (
        config.get("model_type") != "qwen2"
        or config.get("architectures") != ["Qwen2ForCausalLM"]
        or config.get("tie_word_embeddings") is not True
        or config.get("quantization_config") is not None
    ):
        raise ValueError("supported Qwen architecture/config required")


def record_spec(record):
    return {k: record.metadata[k] for k in ("shape", "stride", "dtype", "device")}


def describe_source(saved):
    """Describe only; caller must pin/trust this contract independently of export."""
    result = {k: saved[k] for k in ("binding", "config")}
    for name in ("policy", "reference"):
        result[name] = {k: saved[name][k] for k in ("schema", "aliases")}
    result["optimizer"] = {k: saved["optimizer"][k] for k in ("class", "defaults", "groups")}
    result["rng"] = {
        "cpu": record_spec(saved["rng"]["cpu"]),
        "cuda": None
        if saved["rng"]["cuda"] is None
        else record_spec(saved["rng"]["cuda"]["state"]),
    }
    result["inference"] = {
        n: {
            "shape": m["tensor"]["shape"],
            "dtype": "torch.bfloat16"
            if m["tensor"]["dtype"] == "torch.float32"
            else m["tensor"]["dtype"],
        }
        for n, m in saved["policy"]["schema"].items()
        if m["persistent"] and n not in saved["policy"]["aliases"]
    }
    return result


def validate_source(saved, receipt, contract):
    """Contract is independently pinned by the caller, never inferred on export.

    Accepted producer ownership is checked by PhaseBridge. CPU validation cannot
    assert CUDA RNG semantics: only exact trusted size/type/structure are checked.
    """
    if set(saved) != {
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
        raise ValueError("training keys")
    if (
        saved["format"] != state.FORMAT
        or saved["version"] != receipt.version
        or saved["binding"] != receipt.binding
    ):
        raise ValueError("training lineage/version")
    if (
        saved["binding"] != contract["binding"]
        or saved["config"] != contract["config"]
        or saved["default_dtype"] != "torch.float32"
    ):
        raise ValueError("training contract")
    for kind in ("policy", "reference"):
        value = saved[kind]
        if set(value) != {"schema", "aliases", "tensors"}:
            raise ValueError("model keys")
        expected = contract[kind]
        if (
            digest(value["schema"]) != digest(expected["schema"])
            or value["aliases"] != expected["aliases"]
        ):
            raise ValueError("model schema/aliases")
        schema, aliases = value["schema"], value["aliases"]
        if set(value["tensors"]) != set(schema) - set(aliases):
            raise ValueError("model canonical records")
        for name, target in aliases.items():
            if target not in value["tensors"] or schema[name] != schema[target]:
                raise ValueError("invalid tied alias")
        for name, record in value["tensors"].items():
            meta = schema[name]["tensor"]
            if type(record) is not stream.TensorRecord or record_spec(record) != {
                k: meta[k] for k in record_spec(record)
            }:
                raise ValueError("record schema mismatch")
            if (
                schema[name]["kind"] not in ("parameter", "buffer")
                or type(schema[name]["persistent"]) is not bool
            ):
                raise ValueError("model kind/persistence")
            dtype = meta["dtype"]
            if (
                (dtype.startswith("torch.float") or dtype == "torch.bfloat16")
                and schema[name]["kind"] == "parameter"
                and dtype != ("torch.float32" if kind == "policy" else "torch.bfloat16")
            ):
                raise ValueError("FP32 policy/BF16 reference required")
            distribution = meta["distribution"]
            if distribution is not None:
                if (
                    distribution["mesh"] != [0]
                    or distribution["shape"] != meta["shape"]
                    or distribution["stride"] != meta["stride"]
                    or len(distribution["placements"]) != 1
                ):
                    raise ValueError("world1 full-storage distribution")
                placement = distribution["placements"][0]
                if placement != ["replicate"] and not (
                    len(placement) == 2
                    and placement[0] == "shard"
                    and type(placement[1]) is int
                    and 0 <= placement[1] < len(meta["shape"])
                ):
                    raise ValueError("unsupported distribution")
    optimizer = saved["optimizer"]
    if set(optimizer) != {"class", "defaults", "groups", "state"} or optimizer["class"] != "AdamW":
        raise ValueError("AdamW contract required")
    if digest({k: optimizer[k] for k in ("class", "defaults", "groups")}) != digest(
        contract["optimizer"]
    ):
        raise ValueError("optimizer options/groups")
    names = [n for group in optimizer["groups"] for n in group["names"]]
    parameters = {
        n
        for n, m in saved["policy"]["schema"].items()
        if m["kind"] == "parameter" and n not in saved["policy"]["aliases"]
    }
    if (
        len(names) != len(set(names))
        or set(names) != parameters
        or set(optimizer["state"]) != parameters
    ):
        raise ValueError("optimizer ownership")
    steps = set()
    for group in optimizer["groups"]:
        if any(group["options"].get(k) for k in ("capturable", "differentiable", "fused")):
            raise ValueError("unsupported Adam mode")
        for name in group["names"]:
            entries = optimizer["state"][name]
            required = {"step", "exp_avg", "exp_avg_sq"}
            if group["options"]["amsgrad"]:
                required.add("max_exp_avg_sq")
            if set(entries) != required:
                raise ValueError("complete initialized Adam state required")
            for key, entry in entries.items():
                expected = saved["policy"]["schema"][name]["tensor"]
                if key == "step":
                    expected = {
                        "shape": [],
                        "stride": [],
                        "dtype": "torch.float32",
                        "device": "cpu",
                        "distribution": None,
                    }
                    steps.add(entry["tensor"].index)
                if (
                    set(entry) != {"tensor", "meta"}
                    or entry["meta"] != expected
                    or record_spec(entry["tensor"])
                    != {k: expected[k] for k in record_spec(entry["tensor"])}
                ):
                    raise ValueError("Adam record metadata")
    rng = saved["rng"]
    if set(rng) != {"python", "cpu", "cuda"}:
        raise ValueError("RNG keys")
    import random

    random.Random().setstate(rng["python"])
    for key in ("cpu", "cuda"):
        value, expected = rng[key], contract["rng"][key]
        if key == "cuda" and value is not None:
            if (
                set(value) != {"device", "count", "state"}
                or value["device"] != "cuda:0"
                or type(value["count"]) is not int
                or value["count"] != 1
            ):
                raise ValueError("CUDA RNG structure")
            value = value["state"]
        if (value is None) != (expected is None):
            raise ValueError("RNG presence")
        if value is not None and (
            record_spec(value) != expected
            or expected["dtype"] != "torch.uint8"
            or expected["device"] != "cpu"
            or len(expected["shape"]) != 1
            or not 0 < expected["shape"][0] <= 65536
        ):
            raise ValueError("RNG size/type")
    inference = contract["inference"]
    persistent = {
        n
        for n, m in saved["policy"]["schema"].items()
        if m["persistent"] and n not in saved["policy"]["aliases"]
    }
    if set(inference) != persistent:
        raise ValueError("complete persistent inference state")
    for name, spec in inference.items():
        meta = saved["policy"]["schema"][name]["tensor"]
        target = "torch.bfloat16" if meta["dtype"] == "torch.float32" else meta["dtype"]
        if spec != {"shape": meta["shape"], "dtype": target}:
            raise ValueError("inference schema")
    return steps


def read_weights(fd, checkpoint):
    receipt = checkpoint.receipt
    receipt.validate(fd)
    raw = os.pread(fd, HEADER.size, 0)
    if len(raw) != HEADER.size:
        raise ValueError("weight header")
    magic, length = HEADER.unpack(raw)
    if (
        magic != MAGIC
        or not 0 < length <= receipt.metadata_cap
        or HEADER.size + length > receipt.size
    ):
        raise ValueError("weight manifest bounds")
    manifest = stream._json(os.pread(fd, length, HEADER.size), receipt.metadata_cap)
    if (
        set(manifest) != {"format", "version", "source", "contract", "provenance", "records"}
        or manifest["format"] != "crucible-weights-v1"
    ):
        raise ValueError("weight manifest schema")
    if manifest["provenance"] != provenance(
        checkpoint.model_path, checkpoint.base_revision, checkpoint.assets
    ):
        raise ValueError("sealed base/assets provenance mismatch")
    if (manifest["version"], manifest["source"], manifest["contract"]) != (
        checkpoint.policy_version,
        checkpoint.source_sha256,
        checkpoint.contract,
    ):
        raise ValueError("weight manifest binding")
    offset, names = 0, set()
    if (
        type(manifest["records"]) is not list
        or not 0 < len(manifest["records"]) <= receipt.metadata_cap // 128
    ):
        raise ValueError("weight record count")
    for record in manifest["records"]:
        if set(record) != {"name", "shape", "dtype", "offset", "bytes"}:
            raise ValueError("weight record keys")
        name, dtype, shape = record["name"], record["dtype"], record["shape"]
        if (
            type(name) is not str
            or not 0 < len(name) <= 1024
            or name in names
            or dtype not in stream.DTYPES
            or dtype.startswith("torch.float")
        ):
            raise ValueError("weight name/dtype")
        if type(shape) is not list or len(shape) > 8:
            raise ValueError("weight dimensions")
        elements = 1
        for size in shape:
            state._integer(size, "dimension")
            elements *= size
            if elements > receipt.storage_cap:
                raise ValueError("weight shape cap")
        nbytes = elements * stream.DTYPES[dtype]
        if (
            type(record["offset"]) is not int
            or record["offset"] != offset
            or type(record["bytes"]) is not int
            or record["bytes"] != nbytes
            or nbytes > receipt.storage_cap
        ):
            raise ValueError("weight extent")
        names.add(name)
        offset += nbytes
    if HEADER.size + length + offset != receipt.size:
        raise ValueError("weight trailing/truncated payload")
    return manifest["records"], HEADER.size + length


def make_receipt(fd, identity, version, binding, budget):
    info = os.fstat(fd)
    if (info.st_dev, info.st_ino) != identity or not 0 < info.st_size <= budget.max_bytes:
        raise ValueError("weight output identity/size")
    receipt = state.Receipt(
        *identity,
        info.st_size,
        state._digest(fd, info.st_size),
        state.SEALS,
        version,
        binding,
        budget.max_bytes,
        budget.metadata_cap,
        budget.storage_cap,
    )
    receipt.validate(fd)
    return receipt
