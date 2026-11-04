"""In-memory store for WeightSync: the same staged broadcast protocol, no disk.

`VersionStore` writes every received tensor to safetensors files, which suits
the restart-the-engine handoff but not a per-step sync into a running vLLM. A
`MemoryStore` keeps the received tensors where they arrived (GPU for NCCL), so the
rollout process can pass them straight to `model.load_weights`. The sender side
uses `keep=False`: it only checksums what it sends.

Both sides compute a content digest over the complete schema, the version and
per-tensor checksums, and WeightSync refuses to commit unless the digests agree.
`digest="sha256"` hashes every byte on the host (exact, about a second per
494M-parameter BF16 snapshot); `digest="sum64"` computes two position-weighted
64-bit sums of the raw bits on the tensor's device. sum64 detects lost, reordered
or corrupted chunks with overwhelming probability, but it is not a cryptographic hash.
Integer overflow wraps modulo 2**64, so the sums do not depend on device or
reduction order. Single owner; not thread safe beyond WeightSync's lock.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from threading import RLock

from rl.sync.version import canonical, check_time, digest_file

_CHUNK = 1 << 24
_MOD = 65521


def tensor_checksum(tensor, mode="sum64"):
    """Content checksum of a dense tensor's raw bits (shape/dtype are in the schema)."""
    import torch

    flat = tensor.detach().contiguous().reshape(-1)
    size = flat.element_size()
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[size]
    flat = flat.view(view)
    if mode == "sha256":
        return hashlib.sha256(flat.cpu().numpy().tobytes()).hexdigest()
    if mode != "sum64":
        raise ValueError("checksum mode must be sum64 or sha256")
    plain, weighted = 0, 0
    for offset in range(0, flat.numel(), _CHUNK):
        part = flat[offset : offset + _CHUNK].to(torch.int64)
        weights = torch.arange(offset, offset + part.numel(), device=part.device, dtype=torch.int64)
        weights = weights.remainder_(_MOD).add_(1)
        plain += int(part.sum())
        weighted += int((part * weights).sum())
    return f"{plain % 2**64:016x}{weighted % 2**64:016x}"


class MemoryStore:
    def __init__(self, schema, *, assets=None, keep=True, digest="sum64"):
        if digest not in ("sum64", "sha256"):
            raise ValueError("digest must be sum64 or sha256")
        if not schema:
            raise ValueError("complete nonempty schema required")
        self.schema = dict(schema)
        self.keep, self.digest_mode = keep, digest
        self.asset_hashes = {}
        if assets is not None:
            self.asset_hashes = {
                p.name: digest_file(p)
                for p in Path(assets).iterdir()
                if p.suffix in {".json", ".txt"} and not p.name.endswith(".index.json")
            }
        self.version = None
        self.tensors = None
        self.digest = None
        self._highwater = -1
        self._lock = RLock()
        self._in_doubt = False

    def contract(self):
        return {
            "schema": {
                k: [list(shape), str(dtype)] for k, (shape, dtype) in sorted(self.schema.items())
            },
            "assets": dict(self.asset_hashes),
        }

    def begin(self, version, deadline):
        check_time(deadline)
        if self._in_doubt:
            raise RuntimeError("store transaction in doubt")
        if type(version) is not int or version < 0 or version <= self._highwater:
            raise ValueError("version must strictly increase")
        return MemoryStage(self, version, deadline)

    def take(self, version):
        """Hand the committed tensors (in schema-sorted order) to the caller and drop them here."""
        with self._lock:
            if self._in_doubt:
                raise RuntimeError("store transaction in doubt")
            if version != self.version or self.tensors is None:
                raise ValueError(f"version {version} is not the committed, untaken candidate")
            tensors, self.tensors = self.tensors, None
            return tensors


class MemoryStage:
    def __init__(self, store, version, deadline):
        self.store, self.version, self.deadline = store, version, deadline
        self.contract = store.contract()
        self.seen, self.checksums, self.kept = set(), {}, {}
        self.bytes = 0
        self.committed = False
        self.digest = None

    def add(self, name, tensor):
        import torch

        check_time(self.deadline)
        if name not in self.store.schema or name in self.seen:
            raise ValueError("missing, duplicate or unexpected parameter name")
        shape, dtype = self.store.schema[name]
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError("dense Tensor required")
        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
            raise ValueError("parameter shape/dtype must match complete schema; no implicit cast")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("parameter must be finite")
        self.checksums[name] = tensor_checksum(tensor, self.store.digest_mode)
        if self.store.keep:
            self.kept[name] = tensor.detach()
        self.bytes += tensor.numel() * tensor.element_size()
        self.seen.add(name)
        check_time(self.deadline)

    def prepare(self):
        check_time(self.deadline)
        if self.store.contract() != self.contract:
            raise ValueError("schema/config contract changed during staging")
        if self.seen != set(self.store.schema):
            raise ValueError("incomplete parameter set")
        manifest = {
            "version": self.version,
            "contract": self.contract,
            "checksums": self.checksums,
            "mode": self.store.digest_mode,
            "tensor_bytes": self.bytes,
        }
        self.digest = hashlib.sha256(canonical(manifest)).hexdigest()

    def commit(self):
        check_time(self.deadline)
        store = self.store
        store._highwater = self.version
        store.version, store.digest = self.version, self.digest
        store.tensors = (
            [(name, self.kept[name]) for name in sorted(self.kept)] if store.keep else None
        )
        self.kept = {}
        self.committed = True

    def abort(self):
        if not self.committed:
            self.kept = {}
