"""Exclusive pre-delegated cgroup-v2 scope; never create/delegate a host cgroup.

The startup gate executes no training code before its parent attaches it. All
runtime syscalls are injectable; CPU tests use ordinary temporary files and shims.
"""

import ctypes
import fcntl
import json
import os
import signal
import stat
import sys
import time
from pathlib import Path


class Kernel:
    def magic(self, fd):
        data = ctypes.create_string_buffer(512)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.fstatfs(fd, data) != 0:
            raise OSError(ctypes.get_errno(), "fstatfs failed")
        return ctypes.c_long.from_buffer(data).value

    def read(self, fd, name):
        handle = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        with os.fdopen(handle) as stream:
            return stream.read()

    def write(self, fd, name, value):
        handle = os.open(name, os.O_WRONLY | os.O_NOFOLLOW, dir_fd=fd)
        with os.fdopen(handle, "w") as stream:
            stream.write(value)

    def members(self, fd):
        result = {int(pid) for pid in self.read(fd, "cgroup.procs").split()}
        for name in os.listdir(fd):
            try:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    raise ValueError("symlink in owned scope")
                if stat.S_ISDIR(info.st_mode):
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    try:
                        result.update(self.members(child))
                    finally:
                        os.close(child)
            except FileNotFoundError:
                continue  # An empty sub-cgroup may disappear during enumeration.
        return result

    def member_of(self, pid, cgroup_path):
        try:
            lines = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
        except FileNotFoundError:
            return False
        return any(
            line == f"0::{cgroup_path}" or line.startswith(f"0::{cgroup_path}/") for line in lines
        )

    pidfd_open = staticmethod(os.pidfd_open)
    pidfd_signal = staticmethod(signal.pidfd_send_signal)
    close = staticmethod(os.close)


class CgroupScope:
    def __init__(self, lease_path, *, owner, run_id, kernel=None):
        self.kernel = kernel or Kernel()
        self.lease_path = Path(lease_path).absolute()
        self.lease_fd = os.open(self.lease_path, os.O_RDONLY | os.O_NOFOLLOW)
        self.fd = None
        try:
            info = os.fstat(self.lease_fd)
            if info.st_uid != os.geteuid() or info.st_mode & 0o022:
                raise ValueError("scope lease must be owned and not group/world writable")
            fcntl.flock(self.lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lease_identity = (info.st_dev, info.st_ino)
            self.lease = json.loads(os.read(self.lease_fd, 65536))
            lease = self.lease
            self.path = Path(lease["path"]).absolute()
            delegated = Path(lease["delegated_root"]).absolute()
            if (
                lease.get("owner") != owner
                or lease.get("run_id") != run_id
                or lease.get("exclusive") is not True
                or lease.get("uid") != os.geteuid()
                or self.path == delegated
                or delegated not in self.path.parents
                or self.path == Path("/")
                or lease.get("cgroup_path") in (None, "", "/")
                or ".." in Path(lease["cgroup_path"]).parts
                or any(p.is_symlink() for p in (self.path, *self.path.parents))
            ):
                raise ValueError("exclusive non-root delegated scope ownership mismatch")
            self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            # Lock the actual scope inode as well as its lease, so duplicate
            # lease files cannot race to adopt the same initially empty tree.
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.identity = (lease["device"], lease["inode"])
            self.check()
            if self.kernel.magic(self.fd) != 0x63677270:
                raise ValueError("scope is not a cgroup-v2 filesystem")
            if self.kernel.read(self.fd, "cgroup.type").strip() != "domain":
                raise ValueError("only domain cgroups support this scope")
            if self.populated():
                raise ValueError("scope must initially be empty; never adopt shared work")
            # Require the subtree-kill interface before any worker can start.
            handle = os.open("cgroup.kill", os.O_WRONLY | os.O_NOFOLLOW, dir_fd=self.fd)
            os.close(handle)
            self.attached = False
        except Exception:
            self.close()
            raise

    def check(self):
        if any(p.is_symlink() for p in (self.path, *self.path.parents)):
            raise ValueError("scope path replaced by symlink")
        lease_info = self.lease_path.stat(follow_symlinks=False)
        info = self.path.stat(follow_symlinks=False)
        pinned = os.fstat(self.fd)
        if (
            json.loads(os.pread(self.lease_fd, 65536, 0)) != self.lease
            or lease_info.st_uid != os.geteuid()
            or lease_info.st_mode & 0o022
            or (lease_info.st_dev, lease_info.st_ino) != self.lease_identity
            or (info.st_dev, info.st_ino) != self.identity
            or (pinned.st_dev, pinned.st_ino) != self.identity
            or info.st_uid != self.lease["uid"]
            or info.st_mode & 0o022
            or stat.S_ISLNK(info.st_mode)
        ):
            raise ValueError("scope or lease replaced; ownership no longer certain")

    def populated(self):
        self.check()
        values = dict(
            line.split() for line in self.kernel.read(self.fd, "cgroup.events").splitlines()
        )
        if values.get("populated") not in ("0", "1"):
            raise ValueError("missing cgroup subtree population evidence")
        return values["populated"] == "1"

    def attach(self, pid):
        if self.attached or self.populated() or type(pid) is not int or pid <= 1:
            raise ValueError("only one blocked owned launcher may seed an empty scope")
        self.kernel.write(self.fd, "cgroup.procs", str(pid))
        if not self.kernel.member_of(pid, self.lease["cgroup_path"]):
            raise ValueError("launcher cgroup membership not confirmed; gate stays closed")
        self.attached = True

    def terminate(self):
        self.check()
        # Snapshot TERM is best effort. Pidfds prevent reused PIDs being signaled;
        # subtree KILL below covers forks after this snapshot and all descendants.
        for pid in self.kernel.members(self.fd):
            self.check()
            try:
                handle = self.kernel.pidfd_open(pid)
            except ProcessLookupError:
                continue
            try:
                if self.kernel.member_of(pid, self.lease["cgroup_path"]):
                    self.kernel.pidfd_signal(handle, signal.SIGTERM)
            except ProcessLookupError:
                pass
            finally:
                self.kernel.close(handle)

    def kill(self):
        self.check()
        self.kernel.write(self.fd, "cgroup.kill", "1")

    def wait_empty(self, timeout, *, clock=time.monotonic, sleep=time.sleep):
        deadline = clock() + timeout
        while self.populated():
            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError("owned cgroup subtree remains populated")
            sleep(min(0.02, remaining))
        return clock()

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        if self.lease_fd is not None:
            os.close(self.lease_fd)
            self.lease_fd = None


def gate(fd, argv):
    """Invoked with Python -I -S: no project/user imports or training before attach."""
    permit = os.read(fd, 1)
    os.close(fd)
    if permit != b"1" or not argv:
        raise SystemExit("owned scope attachment was not authorized")
    os.execvpe(argv[0], argv, os.environ)


if __name__ == "__main__":
    if len(sys.argv) < 5 or sys.argv[1] != "--gate" or sys.argv[3] != "--":
        raise SystemExit("internal startup gate only")
    gate(int(sys.argv[2]), sys.argv[4:])
