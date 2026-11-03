"""Foreground GPU command supervisor. Internal helpers permit CPU-only test injection."""

import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

LOCK_PATH = "/var/tmp/rtx4090.lock"
MAX_SECONDS = 4 * 60 * 60
FORBIDDEN = {
    "nohup",
    "setsid",
    "disown",
    "--daemon",
    "--daemonize",
    "--detach",
    "--background",
    "-d",
}
INTERRUPTS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)
LINUX_WALL = 0x40000000  # __WALL: include clone children with non-SIGCHLD exit signals
WAIT_OPTIONS = os.WEXITED | os.WNOWAIT | os.WNOHANG | LINUX_WALL


def validate_command(command):
    if not command or command[0].startswith("-"):
        raise ValueError("usage: with_gpu.sh [--] foreground-command [args...]")
    if any(Path(arg).name in FORBIDDEN or arg.split("=", 1)[0] in FORBIDDEN for arg in command):
        raise ValueError("background/daemon/detach commands are forbidden")


class OwnershipUnconfirmed(RuntimeError):
    """Do not signal numeric identities after losing their kernel ownership anchor."""


def _enable_subreaper():
    # Linux process-local setting; no capabilities, cgroup, or host configuration writes.
    prctl = ctypes.CDLL(None, use_errno=True).prctl
    prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    prctl.restype = ctypes.c_int
    if prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _own_children():
    # This dedicated single-threaded supervisor has no children from other work.
    task = os.getpid()
    children = {int(pid) for pid in Path(f"/proc/self/task/{task}/children").read_text().split()}
    if any(pid <= 0 for pid in children):
        raise OwnershipUnconfirmed("invalid own-child list")
    return children


class _OwnedWorkload:
    """Single reaper: retire group capability before reaping its ownership anchor."""

    def __init__(self, command):
        self.process = subprocess.Popen(command, start_new_session=True)
        self.leader = self.process.pid
        self.group_enabled = True

    @staticmethod
    def peek(pid):
        try:
            info = os.waitid(os.P_PID, pid, WAIT_OPTIONS)
        except ChildProcessError as exc:
            raise OwnershipUnconfirmed(f"child {pid} is no longer waitable by supervisor") from exc
        if info is not None and info.si_pid != pid:
            raise OwnershipUnconfirmed("waitid returned a different child identity")
        return info

    def anchor(self):
        try:
            info = self.peek(self.leader)
            if os.getpgid(self.leader) != self.leader:
                raise OwnershipUnconfirmed("leader no longer anchors its private process group")
            return info
        except (OSError, OwnershipUnconfirmed):
            self.group_enabled = False  # permanently revoke this numeric PGID
            raise

    def signal(self, sig):
        if self.group_enabled:
            self.anchor()  # live or unreaped zombie: PID/PGID cannot be reused
            try:
                os.killpg(self.leader, sig)
            except ProcessLookupError:
                pass
        # Adopted workers may have changed session/group (e.g. an accidental double fork).
        # Every PID is our unreaped direct child, not a match from a global /proc scan.
        children = _own_children()
        if self.group_enabled:
            children.discard(self.leader)
        for pid in children:
            if self.peek(pid) is None:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass

    def drained(self):
        if self.group_enabled:
            leader_info = self.anchor()
            for pid in _own_children() - {self.leader}:
                if self.peek(pid) is None:
                    return False
                self.reap(pid)
            if leader_info is None or _own_children() != {self.leader}:
                return False
            # A proc snapshot is discovery, NOT a complete descendant proof.
            # Retire group signalling before reaping; retain flock for kernel proof.
            self.group_enabled = False
            self.reap(self.leader)
        while True:
            try:
                info = os.waitid(os.P_ALL, 0, WAIT_OPTIONS)
            except ChildProcessError:
                # Subreaper + dedicated single reaper: ECHILD, not a proc snapshot
                # or leader exit, proves there is no living/reparenting descendant.
                if self.process.returncode is None:
                    self.process.returncode = 125  # lost leader status; no later Popen reap
                return True
            if info is None:
                return False  # at least one live child remains, even if proc omitted it
            self.reap(info.si_pid)

    def reap(self, pid):
        if pid == self.leader and self.group_enabled:
            raise OwnershipUnconfirmed("cannot reap an enabled process-group anchor")
        waited, status = os.waitpid(pid, os.WNOHANG | LINUX_WALL)
        if waited != pid:
            raise OwnershipUnconfirmed("exited owned child could not be reaped")
        if pid == self.leader and self.process.returncode is None:
            self.process.returncode = os.waitstatus_to_exitcode(status)


def _cleanup(workload, deadline, grace):
    kill_at = min(deadline - grace / 2, time.monotonic() + grace / 2)
    term_sent, unconfirmed = False, False

    def report(error="deadline exceeded"):
        nonlocal unconfirmed
        if not unconfirmed:
            try:
                print(
                    f"GPU guard: cleanup_unconfirmed; retaining flock isolation: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            except (OSError, ValueError):
                pass  # a closed diagnostic stream must not release an unconfirmed lock
            unconfirmed = True

    while True:
        try:
            if workload.drained():
                return not unconfirmed
        except (OSError, OwnershipUnconfirmed, ValueError) as exc:
            report(exc)
        if not term_sent:
            try:
                workload.signal(signal.SIGTERM)
            except (OSError, OwnershipUnconfirmed, ValueError) as exc:
                report(exc)
            term_sent = True
        if time.monotonic() >= kill_at:
            try:
                workload.signal(signal.SIGKILL)
            except (OSError, OwnershipUnconfirmed, ValueError) as exc:
                report(exc)
        if time.monotonic() >= deadline:
            report()
        # Failure is quarantine, never permission to release the lock or guess a PGID.
        # Retry only our child observations; uncatchable supervisor/host death is outside
        # user-space guarantees. Tests bound this path with syscall shims, not live hangs.
        time.sleep(0.01)


def _supervise(seconds, command):
    """Dedicated Linux supervisor; no Popen.poll/wait may reap its ownership anchor."""
    deadline = time.monotonic() + seconds
    received = []

    def interrupted(sig, _frame):
        received.append(sig)

    previous = {sig: signal.signal(sig, interrupted) for sig in INTERRUPTS}
    previous[signal.SIGCHLD] = signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    workload, confirmed, result = None, True, 125
    grace = min(1.0, seconds / 4)
    try:
        _enable_subreaper()
        try:
            os.waitid(os.P_ALL, 0, WAIT_OPTIONS)
        except ChildProcessError:
            pass  # kernel ECHILD authorizes this dedicated supervisor to launch work
        else:
            raise OwnershipUnconfirmed("dedicated supervisor already has unrelated children")
        workload = _OwnedWorkload(command)
        while True:
            info = workload.anchor()
            if received:
                result = 128 + received[0]
                break
            if info is not None:
                result = info.si_status if info.si_code == os.CLD_EXITED else 128 + info.si_status
                break
            if time.monotonic() >= deadline - grace:
                result = 124
                break
            time.sleep(min(0.02, seconds / 20))
    except (OSError, OwnershipUnconfirmed, ValueError) as exc:
        print(f"GPU guard: supervision failed: {exc}", file=sys.stderr, flush=True)
        result = 2 if workload is None else 125
    finally:
        try:
            if workload is not None:
                confirmed = _cleanup(workload, deadline, grace)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return result if confirmed else 125


def _locked_command(lock_path, seconds, command):
    """Test seam only. Production CLI exposes neither lock nor duration overrides."""
    validate_command(command)
    if not 0 < seconds <= MAX_SECONDS:
        raise ValueError("execution window must be positive and at most four hours")
    root = str(Path(__file__).resolve().parents[1])
    code = (
        f"import sys;sys.path.insert(0,{root!r});"
        "from scripts.gpu_guard import _supervise;"
        "sys.exit(_supervise(float(sys.argv[1]),sys.argv[2:]))"
    )
    # --no-fork transfers lock ownership to the supervisor. Popen(close_fds=True)
    # prevents workload descendants inheriting it; cleanup completes before unlock.
    return [
        "flock",
        "--no-fork",
        "--",
        str(lock_path),
        sys.executable,
        "-c",
        code,
        str(seconds),
        *command,
    ]


def _run_locked(lock_path, seconds, command):
    """CPU-only tests use temporary locks/short limits through this internal helper."""
    process = subprocess.Popen(_locked_command(lock_path, seconds, command))
    previous = {}
    try:

        def relay(sig, _frame):
            try:
                process.send_signal(sig)
            except ProcessLookupError:
                pass

        previous = {sig: signal.signal(sig, relay) for sig in INTERRUPTS}
        result = process.wait()
        return result if result >= 0 else 128 - result
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    try:
        locked = _locked_command(LOCK_PATH, MAX_SECONDS, command)
        # Keep the public wrapper PID as flock/supervisor, so TERM reaches cleanup.
        os.execvp(locked[0], locked)
    except (ValueError, OSError) as exc:
        print(f"GPU guard: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
