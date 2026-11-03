"""Strict two-slot retention for this adapter's exclusive native checkpoint directory."""

import fcntl
import json
import os
import re
import shutil
from pathlib import Path

from cluster.drain.controller import atomic_write


class CheckpointRetention:
    def __init__(self, root, run_id):
        self.root, self.run_id, self.lock = Path(root).absolute(), run_id, None

    def _owned_root(self):
        if self.root == Path("/") or any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise ValueError("exclusive nonsymlink checkpoint path required")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.stat().st_uid != os.geteuid() or self.root.stat().st_mode & 0o022:
            raise ValueError("checkpoint directory owner mismatch")
        marker = self.root / ".crucible-owner.json"
        if marker.exists():
            if marker.is_symlink() or json.loads(marker.read_text()) != {"run_id": self.run_id}:
                raise ValueError("checkpoint run ownership mismatch")
        else:
            if any(p.name != ".crucible-retention.lock" for p in self.root.iterdir()):
                raise ValueError("refuse to claim a nonempty checkpoint directory")
            atomic_write(marker, json.dumps({"run_id": self.run_id}))

    def _confirmed(self):
        checkpoints = []
        for path in self.root.iterdir():
            if path.name in (".crucible-owner.json", ".crucible-retention.lock"):
                if path.is_symlink():
                    raise ValueError("symlink in checkpoint ownership metadata")
                continue
            if (
                path.is_symlink()
                or not path.is_dir()
                or not re.fullmatch(r"step-[0-9]+", path.name)
            ):
                raise ValueError("unknown checkpoint path; operator reconciliation required")
            marker = path / ".crucible-confirmed.json"
            metadata = path / ".metadata"
            if (
                marker.is_symlink()
                or metadata.is_symlink()
                or not marker.is_file()
                or not metadata.is_file()
            ):
                raise ValueError("unknown or incomplete checkpoint; never delete automatically")
            if any(p.stat().st_uid != os.geteuid() for p in (path, marker, metadata)):
                raise ValueError("checkpoint artifact owner mismatch")
            record = json.loads(marker.read_text())
            if (
                record.get("run_id") != self.run_id
                or record.get("step") != int(path.name[5:])
                or type(record.get("sequence")) is not int
                or record["sequence"] <= 0
                or not isinstance(record.get("checkpoint_id"), str)
                or not record["checkpoint_id"]
            ):
                raise ValueError("checkpoint ownership/identity mismatch")
            checkpoints.append((record["sequence"], path))
        return sorted(checkpoints)

    def begin(self, step, protected_checkpoint_id=None):
        if self.root == Path("/") or any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise ValueError("exclusive nonsymlink checkpoint path required")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = self.root / ".crucible-retention.lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        self.lock = os.fdopen(fd, "w")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._owned_root()
            confirmed = self._confirmed()
            self.sequence = max((s for s, _ in confirmed), default=0) + 1
            target = self.root / f"step-{step}"
            protected = confirmed[-1][1] if confirmed else None
            if protected_checkpoint_id is not None:
                matches = [
                    path
                    for _, path in confirmed
                    if json.loads((path / ".crucible-confirmed.json").read_text())["checkpoint_id"]
                    == protected_checkpoint_id
                ]
                if len(matches) != 1:
                    raise ValueError("active recovery anchor is not uniquely confirmed on disk")
                protected = matches[0]
            if target == protected:
                raise ValueError(
                    "refuse to overwrite the latest confirmed or active recovery anchor"
                )
            # After explicit rollback, protect the checkpoint actually loaded,
            # not the higher-step generation from the abandoned lineage.
            for _, path in confirmed:
                if path != protected:
                    shutil.rmtree(path)
            if target.exists():
                raise ValueError("target checkpoint path was not safely retired")
        except Exception:
            self.end()
            raise

    def confirm(self, step, checkpoint_id):
        target = self.root / f"step-{step}"
        metadata = target / ".metadata"
        if target.is_symlink() or metadata.is_symlink() or not metadata.is_file():
            raise ValueError("synchronous native DCP metadata missing")
        atomic_write(
            target / ".crucible-confirmed.json",
            json.dumps(
                {
                    "run_id": self.run_id,
                    "step": step,
                    "checkpoint_id": checkpoint_id,
                    "sequence": self.sequence,
                }
            ),
        )
        if len(self._confirmed()) > 2:
            raise ValueError("two-checkpoint budget exceeded")
        return step

    def end(self):
        if self.lock is not None:
            self.lock.close()
            self.lock = None
