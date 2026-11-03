import json
import os
import signal
from pathlib import Path

import pytest

from cluster.faults.scope import CgroupScope, Kernel, gate


class FakeKernel(Kernel):
    def __init__(self):
        self.ids, self.signals, self.writes, self.closed = set(), [], [], []
        self.outside = set()

    def magic(self, fd):
        return 0x63677270

    def write(self, fd, name, value):
        self.writes.append((name, value))
        root = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if name == "cgroup.procs":
            self.ids.add(int(value))
            (root / "cgroup.events").write_text("populated 1\n")
        if name == "cgroup.kill":
            self.ids.clear()
            (root / "cgroup.events").write_text("populated 0\n")

    def members(self, fd):
        return set(self.ids)

    def member_of(self, pid, path):
        return pid in self.ids and pid not in self.outside

    def pidfd_open(self, pid):
        return pid + 100000

    def pidfd_signal(self, fd, sig):
        self.signals.append((fd, sig))

    def close(self, fd):
        self.closed.append(fd)


@pytest.fixture
def owned_scope(tmp_path):
    delegated = tmp_path / "delegated"
    path = delegated / "run"
    path.mkdir(parents=True)
    path.chmod(0o700)
    for name, value in [
        ("cgroup.events", "populated 0\n"),
        ("cgroup.type", "domain\n"),
        ("cgroup.procs", ""),
        ("cgroup.kill", ""),
    ]:
        (path / name).write_text(value)
    info = path.stat()
    lease = tmp_path / "lease.json"
    lease.write_text(
        json.dumps(
            {
                "path": str(path),
                "delegated_root": str(delegated),
                "device": info.st_dev,
                "inode": info.st_ino,
                "uid": os.geteuid(),
                "owner": "owner",
                "run_id": "run",
                "exclusive": True,
                "cgroup_path": "/delegated/run",
            }
        )
    )
    lease.chmod(0o600)
    return lease, path, FakeKernel()


def test_scope_requires_exclusive_empty_owned_nonroot_delegation(owned_scope):
    lease, path, kernel = owned_scope
    original = json.loads(lease.read_text())
    for field, value in [
        ("owner", "other"),
        ("run_id", "other"),
        ("exclusive", False),
        ("path", original["delegated_root"]),
        ("inode", 0),
    ]:
        lease.write_text(json.dumps(original | {field: value}))
        with pytest.raises(ValueError):
            CgroupScope(lease, owner="owner", run_id="run", kernel=kernel)
    lease.write_text(json.dumps(original))
    (path / "cgroup.events").write_text("populated 1\n")
    with pytest.raises(ValueError, match="initially be empty"):
        CgroupScope(lease, owner="owner", run_id="run", kernel=kernel)
    assert kernel.writes == []


def test_scope_pins_identity_and_never_kills_a_replacement(owned_scope):
    lease, path, kernel = owned_scope
    scope = CgroupScope(lease, owner="owner", run_id="run", kernel=kernel)
    path.rename(path.with_name("old"))
    path.mkdir()
    with pytest.raises(ValueError, match="replaced"):
        scope.kill()
    assert kernel.writes == []
    scope.close()


def test_term_uses_pidfds_and_kill_waits_for_subtree_empty(owned_scope):
    lease, _path, kernel = owned_scope
    scope = CgroupScope(lease, owner="owner", run_id="run", kernel=kernel)
    scope.attach(10)
    kernel.ids.add(11)  # Descendant remains even after launcher exited.
    kernel.ids.discard(10)
    kernel.outside.add(11)  # PID snapshot raced with exit/reuse: never TERM it.
    scope.terminate()
    assert kernel.signals == []
    kernel.outside.clear()
    scope.terminate()
    assert kernel.signals == [(100011, signal.SIGTERM)]
    now = [0.0]
    with pytest.raises(TimeoutError):
        scope.wait_empty(0.04, clock=lambda: now[0], sleep=lambda t: now.__setitem__(0, now[0] + t))
    scope.kill()
    assert scope.wait_empty(1, clock=lambda: now[0]) == now[0]
    assert kernel.writes[-1] == ("cgroup.kill", "1")
    scope.close()


def test_startup_gate_never_executes_without_parent_membership_permission(monkeypatch):
    read, write = os.pipe()
    os.close(write)
    execs = []
    monkeypatch.setattr(os, "execvpe", lambda *args: execs.append(args))
    with pytest.raises(SystemExit, match="not authorized"):
        gate(read, ["training"])
    assert execs == []


def test_scope_rejects_changed_lease_owner_even_without_inode_replacement(owned_scope):
    lease, _path, kernel = owned_scope
    scope = CgroupScope(lease, owner="owner", run_id="run", kernel=kernel)
    changed = json.loads(lease.read_text()) | {"run_id": "new-tenant"}
    lease.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="ownership"):
        scope.terminate()
    with pytest.raises(ValueError, match="ownership"):
        scope.kill()
    assert kernel.signals == []
    assert kernel.writes == []
    scope.close()


def test_startup_cannot_release_training_before_confirmed_scope_attach(monkeypatch):
    from cluster.faults.supervisor import _OwnedTrainingGroup

    permits = []

    class Refused:
        def populated(self):
            return False

        def attach(self, pid):
            raise ValueError("membership not confirmed")

    monkeypatch.setattr(os, "write", lambda fd, body: permits.append(body))
    with pytest.raises(ValueError, match="membership"):
        _OwnedTrainingGroup(
            ["training"],
            env={},
            scope=Refused(),
            popen=lambda *a, **k: type("Child", (), {"pid": 100})(),
        )
    assert permits == []


def test_duplicate_lease_files_cannot_both_adopt_one_empty_scope(owned_scope):
    lease, _path, kernel = owned_scope
    first = CgroupScope(lease, owner="owner", run_id="run", kernel=kernel)
    duplicate = lease.with_name("duplicate.json")
    duplicate.write_text(lease.read_text())
    duplicate.chmod(0o600)
    with pytest.raises(BlockingIOError):
        CgroupScope(duplicate, owner="owner", run_id="run", kernel=kernel)
    assert kernel.writes == []
    first.close()
