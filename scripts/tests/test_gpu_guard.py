import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import gpu_guard
from scripts.gpu_guard import LOCK_PATH, MAX_SECONDS, _run_locked, validate_command


def test_production_constants_and_no_cli_override():
    assert LOCK_PATH == "/var/tmp/rtx4090.lock"
    assert MAX_SECONDS == 4 * 60 * 60
    script = Path(__file__).resolve().parents[1] / "with_gpu.sh"
    result = subprocess.run(
        ["bash", str(script), "--lock", "/tmp/bypass", "true"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


@pytest.mark.parametrize(
    "command",
    [[], ["nohup", "true"], ["setsid", "true"], ["service", "--detach"], ["server", "--daemon"]],
)
def test_rejects_background_commands(command):
    with pytest.raises(ValueError):
        validate_command(command)


def launcher(lock, seconds, command):
    code = (
        "from scripts.gpu_guard import _run_locked; import sys; "
        "sys.exit(_run_locked(sys.argv[1], float(sys.argv[2]), sys.argv[3:]))"
    )
    return subprocess.Popen([sys.executable, "-c", code, str(lock), str(seconds), *command])


def test_lock_serializes_and_releases_on_normal_exit(tmp_path):
    lock, log = tmp_path / "lock", tmp_path / "log"
    code = (
        "import sys,time; f=open(sys.argv[1],'a'); "
        "f.write('start\\n');f.flush();time.sleep(.2);f.write('end\\n');f.close()"
    )
    one = launcher(lock, 2, [sys.executable, "-c", code, str(log)])
    two = launcher(lock, 2, [sys.executable, "-c", code, str(log)])
    assert one.wait(timeout=5) == two.wait(timeout=5) == 0
    assert log.read_text().splitlines() == ["start", "end", "start", "end"]


def test_timeout_kills_stubborn_child_and_unlocks(tmp_path):
    lock = tmp_path / "lock"
    started = time.monotonic()
    code = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"
    assert _run_locked(str(lock), 0.4, [sys.executable, "-c", code]) == 124
    assert time.monotonic() - started < 2
    assert _run_locked(str(lock), 1, [sys.executable, "-c", "pass"]) == 0


def test_signal_releases_lock_and_returns_nonzero(tmp_path):
    lock, pidfile = tmp_path / "lock", tmp_path / "pid"
    code = "import os,sys,time;open(sys.argv[1],'w').write(str(os.getpid()));time.sleep(30)"
    proc = launcher(lock, 2, [sys.executable, "-c", code, str(pidfile)])
    deadline = time.monotonic() + 3
    while not pidfile.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert pidfile.exists()
    os.kill(int(pidfile.read_text()), signal.SIGTERM)
    assert proc.wait(timeout=3) != 0
    assert _run_locked(str(lock), 1, [sys.executable, "-c", "pass"]) == 0


def test_wrapper_sigterm_cleans_up_workload_and_unlocks(tmp_path):
    lock, pidfile = tmp_path / "lock", tmp_path / "pid"
    code = "import os,sys,time;open(sys.argv[1],'w').write(str(os.getpid()));time.sleep(30)"
    proc = launcher(lock, 3, [sys.executable, "-c", code, str(pidfile)])
    deadline = time.monotonic() + 3
    while not pidfile.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert pidfile.exists()
    proc.terminate()
    assert proc.wait(timeout=2) == 143
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)
    assert _run_locked(str(lock), 1, [sys.executable, "-c", "pass"]) == 0


def test_normal_foreground_exit_cleans_background_descendants(tmp_path):
    lock, pidfile = tmp_path / "lock", tmp_path / "pid"
    child = "import time;time.sleep(30)"
    code = (
        "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',sys.argv[2]]);"
        "open(sys.argv[1],'w').write(str(p.pid))"
    )
    assert _run_locked(str(lock), 1, [sys.executable, "-c", code, str(pidfile), child]) == 0
    status = Path(f"/proc/{pidfile.read_text()}/stat")
    assert not status.exists() or status.read_text().split()[2] == "Z"


def test_execution_timeout_starts_after_waiting_for_lock(tmp_path):
    lock, started_file = tmp_path / "lock", tmp_path / "started"
    first_code = "import sys,time;open(sys.argv[1],'w').close();time.sleep(.4)"
    first = launcher(lock, 2, [sys.executable, "-c", first_code, str(started_file)])
    deadline = time.monotonic() + 3
    while not started_file.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started_file.exists()
    second = launcher(lock, 0.25, [sys.executable, "-c", "import time;time.sleep(.08)"])
    assert first.wait(timeout=4) == second.wait(timeout=4) == 0


class StopQuarantine(Exception):
    """Bound a syscall-shim quarantine test without leaving a real lock/process."""


def ownership_shim(monkeypatch, *, worker=False, lost=False):
    state = SimpleNamespace(started=False, reaped=False, signals=[], clock=0.0)

    class Leader:
        pid = 71001
        returncode = None

        def poll(self):
            state.reaped = True
            self.returncode = 0
            return 0

        def wait(self):
            state.reaped = True
            self.returncode = 0
            return 0

    leader = Leader()

    def spawn(*args, **kwargs):
        state.started = True
        return leader

    def peek(kind, pid, flags):
        assert flags & os.WNOWAIT, "ownership observation must not reap the anchor"
        if kind == os.P_ALL:
            if not state.started:
                raise ChildProcessError("shim: dedicated supervisor has no pre-existing child")
            if worker or lost:
                return None
            if state.reaped:
                raise ChildProcessError("shim: kernel confirms no owned children")
            return SimpleNamespace(si_pid=leader.pid, si_code=os.CLD_EXITED, si_status=0)
        assert kind == os.P_PID
        if lost or state.reaped:
            raise ChildProcessError("shim: original child gone; numeric ID reused")
        if pid == leader.pid:
            return SimpleNamespace(si_pid=pid, si_code=os.CLD_EXITED, si_status=0)
        assert pid == 71002
        return None  # kernel-stuck worker remains live despite signals

    def reap(pid, flags):
        assert pid == leader.pid
        assert not worker, "cannot reap anchor while an owned live worker remains"
        state.reaped = True
        return pid, 0

    def group_signal(pid, sig):
        if sig == 0:
            raise ProcessLookupError("shim: no live member in old group")
        state.signals.append((pid, sig, state.reaped))

    monkeypatch.setattr(gpu_guard.subprocess, "Popen", spawn)
    monkeypatch.setattr(gpu_guard, "_enable_subreaper", lambda: None, raising=False)
    monkeypatch.setattr(
        gpu_guard,
        "_own_children",
        lambda: (
            ({leader.pid, 71002} if worker else {leader.pid})
            if state.started and not state.reaped
            else set()
        ),
        raising=False,
    )
    monkeypatch.setattr(gpu_guard.os, "waitid", peek)
    monkeypatch.setattr(gpu_guard.os, "waitpid", reap)
    monkeypatch.setattr(gpu_guard.os, "getpgid", lambda pid: leader.pid)
    monkeypatch.setattr(gpu_guard.os, "killpg", group_signal)
    monkeypatch.setattr(
        gpu_guard.os, "kill", lambda pid, sig: state.signals.append((pid, sig, state.reaped))
    )
    monkeypatch.setattr(gpu_guard.time, "monotonic", lambda: state.clock)
    return state


def test_ownership_shim_never_signals_after_reaping_leader(monkeypatch):
    state = ownership_shim(monkeypatch)
    assert gpu_guard._supervise(1, ["fixture"]) == 0
    assert state.reaped
    assert not any(reaped for _, _, reaped in state.signals)


def test_ownership_shim_lost_anchor_never_signals_reused_pgid(monkeypatch, capsys):
    state = ownership_shim(monkeypatch, lost=True)

    def stop(_seconds):
        assert "cleanup_unconfirmed" in capsys.readouterr().err
        assert not state.reaped
        assert not state.signals
        raise StopQuarantine

    monkeypatch.setattr(gpu_guard.time, "sleep", stop)
    with pytest.raises(StopQuarantine):
        gpu_guard._supervise(1, ["fixture"])


def test_ownership_shim_unconfirmed_worker_keeps_anchor_and_lock(monkeypatch, capsys):
    state = ownership_shim(monkeypatch, worker=True)

    def advance(seconds):
        state.clock += seconds
        if state.clock > 1.1:
            assert "cleanup_unconfirmed" in capsys.readouterr().err
            assert not state.reaped, "return/reap would release the supervisor's flock"
            assert any(sig == signal.SIGKILL for _, sig, _ in state.signals)
            raise StopQuarantine

    monkeypatch.setattr(gpu_guard.time, "sleep", advance)
    with pytest.raises(StopQuarantine):
        gpu_guard._supervise(1, ["fixture"])


def test_ownership_shim_empty_snapshot_never_overrides_kernel_live_child(monkeypatch, capsys):
    state = ownership_shim(monkeypatch, worker=True)
    # /proc can omit live children during concurrent exits. It is only discovery.
    monkeypatch.setattr(
        gpu_guard, "_own_children", lambda: {71001} if state.started and not state.reaped else set()
    )

    def reap_leader(pid, flags):
        assert pid == 71001
        state.reaped = True
        return pid, 0

    def advance(seconds):
        state.clock += seconds
        if state.clock > 1.1:
            assert "cleanup_unconfirmed" in capsys.readouterr().err
            assert not any(reaped for _, _, reaped in state.signals), "no stale group signal"
            raise StopQuarantine

    monkeypatch.setattr(gpu_guard.os, "waitpid", reap_leader)
    monkeypatch.setattr(gpu_guard.time, "sleep", advance)
    with pytest.raises(StopQuarantine):
        gpu_guard._supervise(1, ["fixture"])


def test_ownership_shim_lost_anchor_can_finish_after_kernel_echild(monkeypatch):
    state = ownership_shim(monkeypatch, lost=True)
    original = gpu_guard.os.waitid

    def peek(kind, pid, flags):
        if kind == os.P_ALL:
            raise ChildProcessError("shim: no owned descendants remain")
        return original(kind, pid, flags)

    monkeypatch.setattr(gpu_guard.os, "waitid", peek)
    assert gpu_guard._supervise(1, ["fixture"]) == 125
    assert not state.signals


def test_ownership_shim_retired_pgid_is_never_signalled_after_reuse(monkeypatch):
    state = ownership_shim(monkeypatch)
    original = gpu_guard.os.waitid
    unseen_worker = [True]

    def peek(kind, pid, flags):
        if kind == os.P_ALL and state.reaped and unseen_worker[0]:
            return None
        return original(kind, pid, flags)

    def later_worker_exits(_seconds):
        assert state.reaped
        assert not state.signals
        unseen_worker[0] = False

    monkeypatch.setattr(gpu_guard.os, "waitid", peek)
    monkeypatch.setattr(gpu_guard.time, "sleep", later_worker_exits)
    assert gpu_guard._supervise(1, ["fixture"]) == 0
    assert not state.signals


def test_ownership_shim_proc_error_still_escalates_only_anchored_group(monkeypatch, capsys):
    state = ownership_shim(monkeypatch, worker=True)

    def children():
        if not state.started:
            return set()
        raise PermissionError("shim: own-child discovery unavailable")

    def advance(seconds):
        state.clock += seconds
        if state.clock > 1.1:
            assert "cleanup_unconfirmed" in capsys.readouterr().err
            assert any(sig == signal.SIGKILL for _, sig, _ in state.signals)
            assert all(pid == 71001 and not reaped for pid, _, reaped in state.signals)
            raise StopQuarantine

    monkeypatch.setattr(gpu_guard, "_own_children", children)
    monkeypatch.setattr(gpu_guard.time, "sleep", advance)
    with pytest.raises(StopQuarantine):
        gpu_guard._supervise(1, ["fixture"])


def test_ownership_shim_existing_child_refuses_launch_despite_empty_snapshot(monkeypatch):
    state = ownership_shim(monkeypatch)
    original = gpu_guard.os.waitid

    def peek(kind, pid, flags):
        if kind == os.P_ALL and not state.started:
            return None  # unrelated live child; proc's empty discovery cannot authorize launch
        return original(kind, pid, flags)

    monkeypatch.setattr(gpu_guard.os, "waitid", peek)
    assert gpu_guard._supervise(1, ["fixture"]) == 2
    assert not state.started
    assert not state.signals


def test_ownership_shim_broken_error_stream_cannot_release_unconfirmed_lock(monkeypatch):
    state = ownership_shim(monkeypatch, worker=True)

    class ClosedErrorStream:
        def write(self, _text):
            raise OSError("shim: diagnostic pipe closed")

        def flush(self):
            pass

    def advance(seconds):
        state.clock += seconds
        if state.clock > 1.1:
            assert not state.reaped
            raise StopQuarantine

    monkeypatch.setattr(gpu_guard.sys, "stderr", ClosedErrorStream())
    monkeypatch.setattr(gpu_guard.time, "sleep", advance)
    with pytest.raises(StopQuarantine):
        gpu_guard._supervise(1, ["fixture"])


def test_ownership_shim_kernel_proof_includes_non_sigchld_clone_children(monkeypatch, capsys):
    state = ownership_shim(monkeypatch, worker=True)
    original = gpu_guard.os.waitid
    monkeypatch.setattr(
        gpu_guard, "_own_children", lambda: {71001} if state.started and not state.reaped else set()
    )

    def peek(kind, pid, flags):
        if kind == os.P_ALL and state.reaped and not flags & 0x40000000:  # Linux __WALL
            raise ChildProcessError("shim: default wait hides live non-SIGCHLD clone child")
        return original(kind, pid, flags)

    def reap(pid, flags):
        assert pid == 71001
        state.reaped = True
        return pid, 0

    def advance(seconds):
        state.clock += seconds
        if state.clock > 1.1:
            assert "cleanup_unconfirmed" in capsys.readouterr().err
            raise StopQuarantine

    monkeypatch.setattr(gpu_guard.os, "waitid", peek)
    monkeypatch.setattr(gpu_guard.os, "waitpid", reap)
    monkeypatch.setattr(gpu_guard.time, "sleep", advance)
    with pytest.raises(StopQuarantine):
        gpu_guard._supervise(1, ["fixture"])


def test_double_fork_new_session_worker_is_reaped_before_unlock(tmp_path):
    lock, pidfile = tmp_path / "lock", tmp_path / "worker"
    code = """import os,signal,sys,time
first = os.fork()
if first == 0:
    os.setsid()
    if os.fork(): os._exit(0)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    open(sys.argv[1], 'w').write(str(os.getpid()))
    time.sleep(.8)
    os._exit(0)
os.waitpid(first, 0)
while not os.path.exists(sys.argv[1]): time.sleep(.005)
"""
    try:
        assert _run_locked(str(lock), 1, [sys.executable, "-c", code, str(pidfile)]) == 0
        assert not Path(f"/proc/{pidfile.read_text()}").exists()
        assert _run_locked(str(lock), 0.5, [sys.executable, "-c", "pass"]) == 0
    finally:
        # If a regression misses the worker, its own finite CPU sleep still ends it.
        time.sleep(0.9)
