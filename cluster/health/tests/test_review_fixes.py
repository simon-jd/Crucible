"""Review regressions: CPU proxies only, never real SSH/MPI/GPU execution."""

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cluster.health import rank_exec, remote_worker
from cluster.health.ib_perftest import build_pairs, run_pair
from cluster.health.remote_control import run_remote
from cluster.health.report import execute, summarize
from cluster.health.runner import Result
from cluster.health.tests.test_execution import CpuShims, site


@pytest.mark.parametrize("group,expected", [("affinity", 1), ("dcgm", 5)])
def test_initial_first_failure_stops_all_later_tasks(tmp_path, group, expected):
    plan = site(tmp_path)

    class FailsFirst(CpuShims):
        def run(self, argv, timeout):
            if f"{group}_" in " ".join(argv):
                self.calls.append(argv)
                return Result(argv, 9, "", "SYNTHETIC first failure")
            return super().run(argv, timeout)

    runner = FailsFirst(plan["inventory"])
    results = execute(plan, runner)
    assert len(results) == expected
    # Cleanup calls are permitted for the failed command; no later workload is.
    launches = [a for a in runner.calls if " stop " not in a[-1]]
    assert len(launches) == expected
    assert len(runner.calls) == 2 * expected  # Exactly launch + verified cleanup per task.
    assert not results[-1]["passed"]
    assert summarize(plan, results)["missing"]


def test_ib_ready_consumes_same_absolute_deadline(tmp_path):
    plan = site(tmp_path)
    stop_at = time.time() + 0.15

    class Process:
        def wait_ready(self, timeout):
            time.sleep(0.2)
            return True

        def terminate(self):
            pass

        def finish(self, timeout):
            return Result(["CPU server"], 124, "", "expired", True)

    class Runner:
        def __init__(self):
            self.calls = []

        def start(self, argv):
            self.calls.append(argv)
            return Process()

        def run(self, argv, timeout):
            self.calls.append(argv)
            token = argv[-1].split()[-1]
            return Result(argv, 0, json.dumps({"token": token, "verified": True}), "")

    runner = Runner()
    result = run_pair(
        build_pairs(plan["inventory"])[0],
        plan["inventory"],
        runner,
        timeout=10,
        deadline_epoch=stop_at,
    )
    assert not result["passed"]
    assert len([a for a in runner.calls if " run " in a[-1]]) == 1
    assert "--deadline-epoch" in runner.calls[0][-1]
    assert len(result["cleanup"]) == 2


def live(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.exists() and path.read_text().rsplit(")", 1)[1].split()[0] != "Z"


@pytest.mark.parametrize(
    "proof", ["", '{"verified": true, "token": "wrong"}', '{"verified": false}']
)
def test_successful_command_with_unverified_cleanup_fails(tmp_path, proof):
    plan = site(tmp_path)

    class Runner:
        def run(self, argv, timeout):
            return Result(argv, 0, proof if " stop " in argv[-1] else "SYNTHETIC success", "")

    result = run_remote(
        plan["inventory"],
        plan["inventory"]["nodes"][0],
        ["SYNTHETIC-success"],
        Runner(),
        time.time() + 1,
    )
    assert not result["passed"] and not result["cleanup"][0]["verified"]
    assert result["cleanup"][0]["command"]["stdout"] == proof


def test_rank_forwards_same_deadline_and_controller_owned_token(tmp_path, monkeypatch):
    inv = site(tmp_path)["inventory"]
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inv))
    deadline = time.time() + 1
    original_read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *a, **kw: "0" if str(p).startswith("/sys/") else original_read(p, *a, **kw),
    )
    monkeypatch.setattr(rank_exec.socket, "gethostname", lambda: "n0")
    monkeypatch.setenv("OMPI_COMM_WORLD_LOCAL_RANK", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "SYNTHETIC-before")
    monkeypatch.setenv("NCCL_IB_HCA", "SYNTHETIC-before")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            rank_exec.__file__,
            "--inventory",
            str(inventory_path),
            "--deadline-epoch",
            str(deadline),
            "--token-prefix",
            "CPU-prefix",
            "--",
            "SYNTHETIC-no-GPU-command",
        ],
    )
    commands = []
    monkeypatch.setattr(os, "execvp", lambda executable, argv: commands.append(argv))
    rank_exec.main()
    assert float(commands[0][commands[0].index("--deadline-epoch") + 1]) == deadline
    assert commands[0][commands[0].index("--token") + 1] == rank_exec.rank_token(
        "CPU-prefix", "n0", 0
    )


def test_tombstone_blocks_delayed_launch_after_verified_cleanup(tmp_path):
    token = "e" * 32
    base = [sys.executable, remote_worker.__file__]
    stopped = subprocess.run(
        base + ["stop", "--token", token, "--state-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=2,
        check=True,
    )
    assert json.loads(stopped.stdout)["verified"] is True
    marker = tmp_path / "must-not-start"
    launched = subprocess.run(
        base
        + [
            "run",
            "--token",
            token,
            "--state-dir",
            str(tmp_path),
            "--deadline-epoch",
            str(time.time() + 1),
            "--",
            sys.executable,
            "-c",
            f"from pathlib import Path;Path({str(marker)!r}).touch()",
        ],
        capture_output=True,
        timeout=2,
        check=False,
    )
    assert launched.returncode != 0 and not marker.exists()


def test_stop_does_not_signal_reused_pid(tmp_path, monkeypatch):
    path = tmp_path / ("f" * 32 + ".json")
    path.write_text(json.dumps({"pid": os.getpid(), "starttime": "wrong-old-identity"}))
    monkeypatch.setattr(os, "killpg", lambda *a: pytest.fail("signalled another task"))
    ticks = iter([0, 3])
    monkeypatch.setattr(remote_worker.time, "monotonic", lambda: next(ticks))
    assert remote_worker.stop(path)["verified"] is False
    assert path.exists()


def test_disconnected_proxy_does_not_extend_worker_descendant_deadline(tmp_path):
    """The proxy exits without signalling the detached worker, like a lost connection."""
    token = "d" * 32
    deadline = time.time() + 0.7
    descendant_file = tmp_path / "descendant.json"
    proxy_info = tmp_path / "proxy.json"
    payload = (
        "import json,os,subprocess,time,pathlib; "
        "p=subprocess.Popen(['sleep','60']); "
        f"pathlib.Path({str(descendant_file)!r}).write_text(json.dumps([os.getpid(),p.pid])); "
        "time.sleep(60)"
    )
    worker = [
        sys.executable,
        remote_worker.__file__,
        "run",
        "--token",
        token,
        "--deadline-epoch",
        str(deadline),
        "--state-dir",
        str(tmp_path),
        "--",
        sys.executable,
        "-c",
        payload,
    ]
    proxy = (
        "import subprocess,json,pathlib,sys; "
        "p=subprocess.Popen(json.loads(sys.argv[1]),start_new_session=True,"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        f"pathlib.Path({str(proxy_info)!r}).write_text(str(p.pid))"
    )
    subprocess.run([sys.executable, "-c", proxy, json.dumps(worker)], check=True, timeout=2)
    worker_pid = int(proxy_info.read_text())
    wait_until = time.monotonic() + 3
    try:
        while not descendant_file.exists() and time.monotonic() < wait_until:
            time.sleep(0.01)
        assert descendant_file.exists(), "detached worker never launched its CPU payload"
        pids = json.loads(descendant_file.read_text())
        while any(live(p) for p in [worker_pid, *pids]) and time.monotonic() < wait_until:
            time.sleep(0.02)
        assert not any(live(p) for p in [worker_pid, *pids])
        cleanup = subprocess.run(
            [
                sys.executable,
                remote_worker.__file__,
                "stop",
                "--token",
                token,
                "--state-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        assert cleanup.returncode == 0
        assert json.loads(cleanup.stdout)["verified"] is True
        (tmp_path / "cleanup-evidence.json").write_text(cleanup.stdout)
    finally:
        for path in [tmp_path / (token + ".json"), descendant_file]:
            if path.exists():
                data = json.loads(path.read_text())
                pids = [data["pid"]] if isinstance(data, dict) else data
                for pid in pids:
                    if live(pid):
                        os.kill(pid, signal.SIGKILL)
        if live(worker_pid):
            os.kill(worker_pid, signal.SIGKILL)


@pytest.mark.parametrize("failure", ["disconnect", "timeout", "exception", "cancel", "cleanup"])
def test_dcgm_proxy_failure_explicitly_cleans_and_checkpoints(tmp_path, failure):
    plan = site(tmp_path)
    plan["affinity"] = []
    plan["dcgm"] = plan["dcgm"][:1]
    plan["inventory"]["execution"]["stop_at_utc"] = (
        datetime.now(UTC) + timedelta(seconds=0.8)
    ).isoformat()
    descendant_file = tmp_path / "dcgm-descendants.json"
    worker_file = tmp_path / "dcgm-worker.pid"
    payload = (
        "import os,subprocess,pathlib,json,time; "
        "p=subprocess.Popen(['sleep','60']); "
        f"pathlib.Path({str(descendant_file)!r}).write_text(json.dumps([os.getpid(),p.pid])); "
        "time.sleep(60)"
    )

    class ProxyRunner:
        def run(self, argv, timeout):
            words = shlex.split(argv[-1])
            action = words[2]
            actual = [sys.executable, remote_worker.__file__, *words[2:]]
            index = actual.index("--") if "--" in actual else len(actual)
            actual[index:index] = ["--state-dir", str(tmp_path)]
            if action == "stop":
                if failure == "cleanup":
                    return Result(["SYNTHETIC unavailable cleanup transport"], 255, "", "lost")
                raw = subprocess.run(actual, capture_output=True, text=True, timeout=3, check=False)
                return Result(actual, raw.returncode, raw.stdout, raw.stderr)
            actual = actual[: actual.index("--") + 1] + [sys.executable, "-c", payload]
            proxy = (
                "import subprocess,pathlib,time,json,sys; "
                "p=subprocess.Popen(json.loads(sys.argv[1]),start_new_session=True,"
                "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                f"pathlib.Path({str(worker_file)!r}).write_text(str(p.pid)); "
                f"deadline=time.time()+0.5; path=pathlib.Path({str(descendant_file)!r}); "
                "exec('while not path.exists() and time.time()<deadline: time.sleep(0.01)')"
            )
            proxy_argv = [sys.executable, "-c", proxy, json.dumps(actual)]
            subprocess.run(proxy_argv, timeout=2, check=True)
            if failure == "exception":
                raise OSError("SYNTHETIC lost proxy")
            if failure == "cancel":
                raise KeyboardInterrupt("SYNTHETIC cancelled proxy")
            return Result(
                proxy_argv,
                124 if failure == "timeout" else 255,
                "",
                "SYNTHETIC disconnected",
                failure == "timeout",
            )

    checkpoints = []
    try:
        results = execute(
            plan, ProxyRunner(), checkpoint=lambda r: checkpoints.append(json.loads(json.dumps(r)))
        )
        assert len(results) == 1 and not results[0]["passed"]
        assert len(results[0]["cleanup"]) == 1
        assert results[0]["cleanup"][0]["verified"] is (failure != "cleanup")
        assert checkpoints[-1] == results
        pids = [int(worker_file.read_text()), *json.loads(descendant_file.read_text())]
        wait_until = time.monotonic() + 2
        while any(live(p) for p in pids) and time.monotonic() < wait_until:
            time.sleep(0.01)
        assert not any(live(p) for p in pids)
        print(
            "CPU_PROXY_EVIDENCE",
            json.dumps(
                {
                    "failure": failure,
                    "passed": results[0]["passed"],
                    "all_owned_processes_exited": True,
                    "cleanup": results[0]["cleanup"],
                }
            ),
        )
    finally:
        if descendant_file.exists():
            for pid in json.loads(descendant_file.read_text()):
                if live(pid):
                    os.kill(pid, signal.SIGKILL)
        if worker_file.exists() and live(int(worker_file.read_text())):
            os.kill(int(worker_file.read_text()), signal.SIGKILL)
