"""Owned, complete snapshots; publication is distinct from engine activation.

Single-owner protection against accidental changes, not hostile-host attestation.
At most two directories (including partial staging) may exist. No automatic prune.
"""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from threading import RLock

from rl.rollout.engine import PolicyCheckpoint


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("weight synchronization transaction deadline exceeded")


def remove_tree(path):
    if path.exists():
        path.chmod(0o700)
        for file in path.iterdir():
            file.chmod(0o600)
        shutil.rmtree(path)


def verify_checkpoint(checkpoint):
    """Verify every owned byte against the content address, including asset/index files."""
    path = Path(checkpoint.model_path)
    if path.is_symlink():
        raise ValueError("checkpoint content must be owned, not linked")
    raw = (path / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != checkpoint.revision:
        raise ValueError("checkpoint content manifest mismatch")
    manifest = json.loads(raw)
    if manifest["version"] != checkpoint.policy_version:
        raise ValueError("checkpoint content version mismatch")
    files = manifest["files"]
    if {p.name for p in path.iterdir()} != set(files) | {"manifest.json"}:
        raise ValueError("checkpoint content file set mismatch")
    for name, expected in files.items():
        file = path / name
        if Path(name).name != name or file.is_symlink() or not file.is_file():
            raise ValueError("checkpoint content is not an owned regular file")
        if digest_file(file) != expected:
            raise ValueError(f"checkpoint content mismatch: {name}")
    return manifest


class VersionStore:
    """Schema is the complete unique checkpoint state, including persistent buffers.

    Construct with an explicit expected dtype. Generic push cannot change it.
    Assets are copied from the immutable base; no checkpoint weight symlinks.
    Access is single-process/single-owner; a store must not be independently opened
    by concurrent writers. Distributed ranks must each own a different root.
    """

    def __init__(self, root, schema, assets):
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        self.schema = dict(schema)
        import math

        import torch

        for name, (shape, dtype) in self.schema.items():
            if not isinstance(name, str) or not name or len(name) > 1024:
                raise ValueError("bounded parameter names required")
            if any(type(n) is not int or n < 0 for n in shape):
                raise ValueError("bounded dense tensor shape required")
            if math.prod(shape) * torch.empty((), dtype=dtype).element_size() > 1024**3:
                raise ValueError("individual tensor exceeds 1GiB staging bound")
        if not self.schema:
            raise ValueError("complete nonempty schema required")
        self.assets = Path(assets)
        self.asset_hashes = {
            p.name: digest_file(p)
            for p in self.assets.iterdir()
            if p.suffix in {".json", ".txt"} and not p.name.endswith(".index.json")
        }
        if "config.json" not in self.asset_hashes:
            raise ValueError("config.json asset required")
        self.active_version = None
        self._engine_owner = None
        self._lock = RLock()
        self._in_doubt = False
        self._ledger = self.root.parent / (self.root.name + ".highwater")
        self._highwater = max(self._entries(), default=-1)
        if self._ledger.exists():
            self._highwater = max(self._highwater, int(self._ledger.read_text()))

    def _reserve_engine(self, owner, checkpoint):
        # Process-local lease, including startup/failed cleanup. Not a filesystem lock.
        with self._lock:
            if self._engine_owner is not None or self.active_version is not None:
                raise RuntimeError("owned store already has an active or reserved engine")
            if self.checkpoint(checkpoint.policy_version) != checkpoint:
                raise ValueError("checkpoint does not match owned store")
            self._engine_owner = owner
            self.active_version = checkpoint.policy_version

    def _activate_engine(self, owner, version):
        with self._lock:
            if self._engine_owner is not owner:
                raise RuntimeError("engine does not own store reservation")
            self.active_version = version

    def _release_engine(self, owner):
        with self._lock:
            if self._engine_owner is owner:
                self._engine_owner = None
                self.active_version = None

    def contract(self):
        return {
            "schema": {
                k: [list(shape), str(dtype)] for k, (shape, dtype) in sorted(self.schema.items())
            },
            "assets": dict(self.asset_hashes),
        }

    def _entries(self):
        entries = {}
        for path in self.root.iterdir():
            if path.name.startswith("v") and "-" in path.name:
                number, digest = path.name[1:].split("-", 1)
                entries[int(number)] = PolicyCheckpoint(str(path), digest, int(number))
        return entries

    def checkpoint(self, version):
        with self._lock:
            if self._in_doubt:
                raise RuntimeError("store transaction in doubt; recover in a new owner phase")
            if type(version) is not int or version < 0:
                raise ValueError("version must be a nonnegative integer")
            try:
                checkpoint = self._entries()[version]
            except KeyError:
                raise ValueError(f"no complete checkpoint version {version}") from None
            manifest = verify_checkpoint(checkpoint)
            if manifest["contract"] != self.contract():
                raise ValueError("checkpoint content does not match store schema/config contract")
            return checkpoint

    def begin(self, version, deadline):
        check_time(deadline)
        if self._in_doubt:
            raise RuntimeError("store transaction in doubt")
        entries = self._entries()
        if (
            type(version) is not int
            or version < 0
            or version <= max(self._highwater, max(entries, default=-1))
        ):
            raise ValueError("version must strictly increase")
        if len(list(self.root.iterdir())) >= 2:
            raise ValueError("at most two snapshots including staging; discard nonactive first")
        return Stage(self, version, deadline)

    def discard(self, version):
        with self._lock:
            if version == self.active_version:
                raise ValueError("cannot discard active checkpoint")
            checkpoint = self.checkpoint(version)
            remove_tree(Path(checkpoint.model_path))


class Stage:
    def __init__(self, store, version, deadline):
        self.store, self.version, self.deadline = store, version, deadline
        self.contract = store.contract()
        self.path = store.root / f".stage-{version}"
        self.path.mkdir()
        self.seen, self.files, self.weight_map = set(), {}, {}
        self.bytes = 0
        self.committed = False
        try:
            for name, digest in store.asset_hashes.items():
                check_time(deadline)
                destination = self.path / name
                shutil.copyfile(store.assets / name, destination)
                if digest_file(destination) != digest:
                    raise ValueError("base asset content changed")
                self.files[name] = digest
        except BaseException as failure:
            try:
                self.abort()
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve failed-stage ownership
                store._in_doubt = True
                raise BaseExceptionGroup("staging and cleanup failed", [failure, cleanup_error])
            raise

    def add(self, name, tensor):
        import torch
        from safetensors.torch import save_file

        check_time(self.deadline)
        if name not in self.store.schema or name in self.seen:
            raise ValueError("missing, duplicate or unexpected parameter name")
        shape, dtype = self.store.schema[name]
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError("dense Tensor required")
        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
            raise ValueError("parameter shape/dtype must match complete schema; no implicit cast")
        tensor = tensor.detach().to(device="cpu").contiguous()
        if not torch.isfinite(tensor).all().item():
            raise ValueError("parameter must be finite")
        filename = f"tensor-{len(self.seen):05d}.safetensors"
        save_file({name: tensor}, str(self.path / filename), metadata={"format": "pt"})
        self.files[filename] = digest_file(self.path / filename)
        self.weight_map[name] = filename
        self.bytes += tensor.numel() * tensor.element_size()
        self.seen.add(name)
        check_time(self.deadline)

    def prepare(self):
        check_time(self.deadline)
        if self.store.contract() != self.contract:
            raise ValueError("schema/config contract changed during staging")
        if self.seen != set(self.store.schema):
            raise ValueError("incomplete parameter set")
        index = self.path / "model.safetensors.index.json"
        index.write_bytes(
            canonical({"metadata": {"total_size": self.bytes}, "weight_map": self.weight_map})
        )
        self.files[index.name] = digest_file(index)
        manifest = {
            "version": self.version,
            "contract": self.contract,
            "files": self.files,
            "tensor_bytes": self.bytes,
        }
        raw = canonical(manifest)
        (self.path / "manifest.json").write_bytes(raw)
        self.digest = hashlib.sha256(raw).hexdigest()
        for file in self.path.iterdir():
            with open(file, "rb") as stream:
                os.fsync(stream.fileno())
            file.chmod(0o444)
        check_time(self.deadline)

    def commit(self):
        check_time(self.deadline)
        destination = self.store.root / f"v{self.version}-{self.digest}"
        # Never reuse an owned version number after explicit discard/reopen.
        self.store._highwater = self.version
        ledger_tmp = self.store._ledger.with_suffix(".tmp")
        ledger_tmp.write_text(str(self.version))
        with open(ledger_tmp, "rb") as stream:
            os.fsync(stream.fileno())
        ledger_tmp.replace(self.store._ledger)
        self.path.chmod(0o555)
        self.path.rename(destination)  # no copy; old version is never touched
        self.path = destination
        self.committed = True
        try:
            with _directory_fd(self.store.root) as fd:
                os.fsync(fd)
            check_time(self.deadline)
        except BaseException:
            # Publication cannot be undone or declared timely after late/failed I/O.
            self.store._in_doubt = True
            raise

    def abort(self):
        if not self.committed:
            remove_tree(self.path)


class _directory_fd:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
        return self.fd

    def __exit__(self, *_exc):
        os.close(self.fd)
