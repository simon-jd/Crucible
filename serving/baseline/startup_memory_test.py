import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from serving.baseline.startup_memory import run_probe, snapshot


def fake_process(proc, pid, parent, start=10, rss=3):
    directory = proc / str(pid)
    directory.mkdir()
    fields = ["0"] * 22
    fields[0], fields[1], fields[2], fields[19], fields[21] = (
        "S",
        str(parent),
        str(parent),
        str(start),
        str(rss),
    )
    (directory / "stat").write_text(f"{pid} (fixture) " + " ".join(fields))
    (directory / "status").write_text(
        "VmRSS:\t12 kB\nRssAnon:\t8 kB\nRssFile:\t4 kB\nRssShmem:\t0 kB\n"
    )
    (directory / "cmdline").write_bytes(b"fixture\0")


def test_snapshot_preserves_identity_and_all_rss_categories(tmp_path):
    fake_process(tmp_path, 100, 1)
    fake_process(tmp_path, 101, 100, start=11, rss=4)
    fake_process(tmp_path, 102, 1, rss=999)
    seen = {}
    result = snapshot(100, seen, proc=tmp_path)
    assert result["aggregate_rss_bytes"] == 7 * os.sysconf("SC_PAGE_SIZE")
    assert {x["pid"] for x in result["processes"]} == {100, 101}
    assert seen == {100: 10, 101: 11}
    assert result["processes"][1]["RssFile"] == 4096


def test_snapshot_rejects_pid_reuse_and_unreadable_owned_status(tmp_path, monkeypatch):
    from serving.baseline import startup_memory as memory

    fake_process(tmp_path, 100, 1)
    with pytest.raises(ValueError, match="identity"):
        snapshot(100, {100: 9}, proc=tmp_path)
    original = memory.bounded_read

    def denied(path):
        if path.name == "status":
            raise PermissionError("fixture owned status denied")
        return original(path)

    monkeypatch.setattr(memory, "bounded_read", denied)
    with pytest.raises(PermissionError, match="owned"):
        snapshot(100, {}, proc=tmp_path)


def test_snapshot_records_confirmed_disappearance(tmp_path):
    fake_process(tmp_path, 100, 1)
    seen = {100: 10, 101: 11}
    sample = snapshot(100, seen, proc=tmp_path)
    assert sample["events"] == [{"pid": 101, "start_ticks": 11, "event": "absent"}]


def test_real_owned_child_tree_rss_is_summed_and_reaped():
    code = """import signal,subprocess,sys,time
child=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(60)'])
def stop(*args):
 child.terminate();child.wait(timeout=2);sys.exit(0)
signal.signal(signal.SIGTERM,stop)
print(child.pid,flush=True)
time.sleep(60)
"""
    child = subprocess.Popen([sys.executable, "-B", "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        grandchild = int(child.stdout.readline())
        sample = snapshot(os.getpid(), {})
        ids = {x["pid"] for x in sample["processes"]}
        assert {os.getpid(), child.pid, grandchild} <= ids
        assert sample["aggregate_rss_bytes"] == sum(x["rss_bytes"] for x in sample["processes"])
    finally:
        child.terminate()
        child.wait(timeout=3)
        child.stdout.close()
    assert not Path(f"/proc/{grandchild}").exists()


@pytest.mark.parametrize("case", ["memory", "timeout", "evidence"])
def test_probe_preserves_trigger_or_error_and_reaps_child(tmp_path, case):
    def sample(*args, **kwargs):
        return {
            "aggregate_rss_bytes": 10_000_000_000 if case == "memory" else 1,
            "processes": [],
            "events": [],
            "fixture": "x" * (500 if case == "evidence" else 1),
        }

    result = run_probe(
        [sys.executable, "-B", "-c", "import time;time.sleep(60)"],
        {},
        tmp_path,
        None,
        lambda: None,
        sampler=sample,
        seconds=0.12,
        evidence_limit=100 if case == "evidence" else 16000,
    )
    assert (
        result["reason"]
        == {"memory": "early-memory-stop", "timeout": "deadline", "evidence": "error"}[case]
    )
    assert result["cleanup"]["reaped"] and not result["cleanup"]["forced_kill"]
    assert not Path(f"/proc/{result['cleanup']['pid']}").exists()
    assert json.loads((tmp_path / "terminal.json").read_text())["baseline_accepted"] is False
    if case == "memory":
        row = json.loads((tmp_path / "samples.jsonl").read_text().splitlines()[-1])
        assert row["aggregate_rss_bytes"] == 10_000_000_000
    if case == "evidence":
        assert "bound" in result["error"]


def test_probe_timeout_reaps_real_loopback_listener(tmp_path):
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    code = (
        "import socket,time; s=socket.socket();"
        f"s.bind(('127.0.0.1',{port}));s.listen();print('ready',flush=True);time.sleep(60)"
    )
    result = run_probe(
        [sys.executable, "-B", "-c", code], {}, tmp_path, port, lambda: None, seconds=0.35
    )
    assert result["reason"] == "deadline" and result["cleanup"]["listener_closed"]
    assert "ready" in (tmp_path / "server.log").read_text()


def test_proc_read_and_owned_process_bounds(tmp_path):
    fake_process(tmp_path, 100, 1)
    (tmp_path / "100/status").write_bytes(b"x" * 8193)
    with pytest.raises(ValueError, match="read bound"):
        snapshot(100, {}, proc=tmp_path)
    (tmp_path / "100/status").write_text(
        "VmRSS: 0 kB\nRssAnon: 0 kB\nRssFile: 0 kB\nRssShmem: 0 kB\n"
    )
    for pid in range(101, 165):
        fake_process(tmp_path, pid, 100)
    with pytest.raises(ValueError, match="process bound"):
        snapshot(100, {}, proc=tmp_path)


def test_probe_bounds_actual_child_stdout_before_copy_and_reaps(tmp_path):
    code = "import os,time; os.write(1,b'x'*100000);time.sleep(60)"
    result = run_probe(
        [sys.executable, "-B", "-c", code],
        {},
        tmp_path,
        None,
        lambda: None,
        seconds=1,
        evidence_limit=16000,
    )
    assert result["reason"] == "error" and "log/evidence bound" in result["error"]
    assert result["cleanup"]["reaped"]
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) <= 16000


def test_global_proc_enumeration_bound(monkeypatch):
    from types import SimpleNamespace

    from serving.baseline import startup_memory as memory

    fake = SimpleNamespace(iterdir=lambda: (Path(str(pid)) for pid in range(1, 8194)))
    monkeypatch.setattr(memory, "read_stat", lambda path: {"pid": int(path.name)})
    with pytest.raises(ValueError, match="enumeration bound"):
        snapshot(1, {}, proc=fake)


@pytest.mark.parametrize("split", [False, True])
def test_ready_log_stops_without_any_http_request(tmp_path, monkeypatch, split):
    import http.client

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("diagnostic must send no health/generate HTTP request")

    monkeypatch.setattr(http.client, "HTTPConnection", forbidden)
    marker = b"The server is fired up and ready to roll!"
    code = (
        f"import os,time;os.write(1,{marker[:20] if split else marker!r});"
        + (f"time.sleep(.15);os.write(1,{marker[20:]!r});" if split else "")
        + "time.sleep(60)"
    )
    result = run_probe(
        [sys.executable, "-B", "-c", code], {}, tmp_path, 0, lambda: None, seconds=0.8
    )
    assert result["reason"] == "ready-no-generation" and calls == []
    assert result["cleanup"]["reaped"]
    assert marker in (tmp_path / "server.log").read_bytes()


def test_periodic_resource_failure_persists_complete_sample_before_cleanup(tmp_path, monkeypatch):
    import time

    from serving.baseline import startup_memory as memory

    checks = []
    sample = {
        "aggregate_rss_bytes": 16_100_000_000,
        "monotonic": 123.5,
        "processes": [
            {
                "pid": 765,
                "start_ticks": 456,
                "rss_bytes": 16_100_000_000,
                "RssAnon": 15_000_000_000,
                "RssFile": 1_000_000_000,
                "RssShmem": 100_000_000,
                "category": "fixture-scheduler",
            }
        ],
        "events": [{"event": "distinctive-trigger"}],
    }
    error = RuntimeError('resource high-water stop: {"owned_rss_bytes":16100000000}')

    def inspect():
        checks.append(True)
        if len(checks) == 1:
            return {"initial": "accepted"}
        raise error

    def sampler(*args, **kwargs):
        time.sleep(1.02)  # Make the existing once-per-second inspection due.
        return dict(sample)

    original = memory.stop_child
    cleaned = []

    def check_then_cleanup(child, port):
        try:
            saved = [
                json.loads(line) for line in (tmp_path / "samples.jsonl").read_text().splitlines()
            ]
            assert saved == [{**sample, "resource_error": repr(error)}]
            stop = json.loads((tmp_path / "stop.json").read_text())
            assert stop["samples"] == 1 and stop["error"] == repr(error)
            assert child.poll() is None  # Evidence precedes real exact-child cleanup.
        finally:
            cleaned.append(original(child, port))
        return cleaned[-1]

    monkeypatch.setattr(memory, "stop_child", check_then_cleanup)
    result = memory.run_probe(
        [sys.executable, "-B", "-c", "import time;time.sleep(60)"],
        {},
        tmp_path,
        None,
        inspect,
        sampler=sampler,
        seconds=3,
    )
    assert result["reason"] == "error" and checks == [True, True]
    assert cleaned[0]["reaped"] and not Path(f"/proc/{cleaned[0]['pid']}").exists()
