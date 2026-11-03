"""Real CPU-only proofs for baseline trial, smoke and terminal-report lifecycle."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from serving.baseline import experiment


def test_error_description_is_truncated_utf8():
    description = experiment.error_info(ValueError("界" * 10000))
    assert description["truncated"] and len(description["repr"].encode()) <= 1024


def test_primary_failure_is_recorded_before_cleanup(tmp_path, monkeypatch):
    child = subprocess.Popen([sys.executable, "-B", "-c", "import time;time.sleep(30)"])
    original = experiment.stop_child
    seen = []

    def cleanup(process, port):
        trigger = json.loads((tmp_path / "terminal.json").read_text())
        assert trigger["stage"] == "before_cleanup" and trigger["aborted"]
        assert "primary CPU fixture" in trigger["primary_failure"]["repr"]
        seen.append(process.pid)
        return original(process, port)

    monkeypatch.setattr(experiment, "stop_child", cleanup)
    failure = RuntimeError("primary CPU fixture")
    returned = experiment.finish_full(
        tmp_path, child, None, {"azure": [], "burst": []}, [{"data": "界" * 2000}], failure
    )
    assert returned is failure and seen == [child.pid]
    assert child.poll() is not None and not Path(f"/proc/{child.pid}").exists()
    terminal = json.loads((tmp_path / "terminal.json").read_text())
    assert terminal["aborted"] and terminal["primary_failure"]
    assert len(terminal["planned_trials"]) == 30
    assert all(x["status"] == "not_run" for x in terminal["planned_trials"])
    assert json.loads((tmp_path / "resource-samples.json").read_text()) == [{"data": "界" * 2000}]


def test_partial_http_evidence_never_claims_full_body_hash(tmp_path, monkeypatch):
    def partial(endpoint, body, data, evidence, inspect):
        data.extend(b"x" * (70 << 10))
        raise TimeoutError("primary partial-body timeout")

    monkeypatch.setattr(experiment, "_native_response", partial)
    with pytest.raises(TimeoutError, match="primary partial-body"):
        experiment._native_cases(
            "http://127.0.0.1:1/generate",
            [("zero", [0], 0)],
            tmp_path,
            "native-smoke",
            lambda: None,
        )
    row = json.loads((tmp_path / "native-smoke.json").read_text())[0]
    assert row["observed_count_is_lower_bound"] and row["raw_bytes"] == 70 << 10
    assert row["accepted_contract"] is False and row["error"].startswith("TimeoutError")


def test_failed_report_invalidates_aggregates_without_mutating_raw_members(tmp_path, monkeypatch):
    order = experiment.trial_order()
    summaries = [
        {"trial": x, "status": "completed", "metrics": {"accepted": True, "goodput": 1}}
        for x in order
    ]
    write = experiment.write_json

    def failing_write(path, value):
        if Path(path).name == "resource-samples.json":
            raise OSError("fixture report write failure")
        write(path, value)

    monkeypatch.setattr(experiment, "write_json", failing_write)
    failure = experiment.finish_full(tmp_path, None, None, {"azure": summaries}, [], None)
    assert isinstance(failure, OSError)
    pairs = json.loads((tmp_path / "azure-pairs.json").read_text())
    assert pairs["execution_aborted"]
    assert all(
        not c["accepted_all_three"] and c["goodput_difference"] is None
        for c in pairs["comparisons"].values()
    )
    assert all(x["metrics"]["accepted"] for x in summaries)
    terminal = json.loads((tmp_path / "terminal.json").read_text())
    assert terminal["aborted"] and terminal["baseline_accepted"] is False


def test_trial_hash_failure_cannot_mask_primary_after_real_child_reap(tmp_path, monkeypatch):
    real_hash = experiment.sha256

    def hash_file(path):
        if path.name == "trial.json":
            raise OSError("fixture metadata hash error")
        return real_hash(path)

    monkeypatch.setattr(experiment, "sha256", hash_file)
    summary = {"planned_rows": 1}
    with pytest.raises(json.JSONDecodeError):
        experiment.execute_trial(
            [sys.executable, "-B", "-c", "print('not JSON')"],
            dict(os.environ),
            tmp_path / "trial.json",
            summary,
            lambda: None,
        )
    assert summary["cleanup"]["reaped"]
    assert not Path(f"/proc/{summary['pid']}").exists()
    assert "metadata hash error" in summary["metadata_error"]["repr"]


@pytest.mark.parametrize("exit_kind", ["SIGXFSZ", "nonzero", "zero"])
def test_late_native_exit_invalidates_both_comparisons(tmp_path, exit_kind):
    code = "import os,resource,signal;resource.setrlimit(resource.RLIMIT_CORE,(0,0));"
    code += {
        "SIGXFSZ": "signal.signal(signal.SIGXFSZ,signal.SIG_DFL);os.kill(os.getpid(),signal.SIGXFSZ)",
        "nonzero": "raise SystemExit(7)",
        "zero": "raise SystemExit(0)",
    }[exit_kind]
    with (tmp_path / "server.log").open("wb") as log:
        child = subprocess.Popen([sys.executable, "-B", "-c", code], stdout=log, stderr=log)
        child.wait(timeout=5)
    summaries = [
        {"trial": row, "status": "completed", "metrics": {"accepted": True, "goodput": 1}}
        for row in experiment.trial_order()
    ]
    failure = experiment.finish_full(
        tmp_path, child, None, {"azure": summaries, "burst": summaries}, [], None
    )
    assert isinstance(failure, RuntimeError)
    terminal = json.loads((tmp_path / "terminal.json").read_text())
    assert terminal["aborted"] and not terminal["run_completed"]
    assert not terminal["cleanup"]["termination_requested"]
    assert terminal["cleanup"]["returncode_before_cleanup"] == child.returncode
    assert terminal["cleanup"]["returncode"] == child.returncode
    assert "unexpected native exit" in terminal["primary_failure"]["repr"]
    for trace in ("azure", "burst"):
        pairs = json.loads((tmp_path / f"{trace}-pairs.json").read_text())
        assert pairs["execution_aborted"]
        assert all(not c["accepted_all_three"] for c in pairs["comparisons"].values())
    assert not Path(f"/proc/{child.pid}").exists()


def test_requested_native_term_is_expected_and_retains_exit_identity(tmp_path):
    import signal
    import time

    with (tmp_path / "server.log").open("wb") as log:
        child = subprocess.Popen(
            [sys.executable, "-B", "-c", "import os,time;os.write(1,b'ready');time.sleep(30)"],
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 5
            while not (tmp_path / "server.log").stat().st_size and time.monotonic() < deadline:
                time.sleep(0.01)
            assert child.poll() is None
            summaries = [
                {"trial": row, "status": "completed", "metrics": {"accepted": True, "goodput": 1}}
                for row in experiment.trial_order()
            ]
            failure = experiment.finish_full(
                tmp_path, child, None, {"azure": summaries, "burst": summaries}, [], None
            )
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
    assert failure is None
    terminal = json.loads((tmp_path / "terminal.json").read_text())
    assert terminal["run_completed"] and not terminal["aborted"]
    assert terminal["cleanup"]["termination_requested"]
    assert terminal["cleanup"]["returncode_before_cleanup"] is None
    assert terminal["cleanup"]["returncode"] == -signal.SIGTERM
    for trace in ("azure", "burst"):
        pairs = json.loads((tmp_path / f"{trace}-pairs.json").read_text())
        assert not pairs["execution_aborted"]
        assert all(c["accepted_all_three"] for c in pairs["comparisons"].values())
    assert not Path(f"/proc/{child.pid}").exists()


@pytest.mark.parametrize("case", ["zero", "error"])
@pytest.mark.parametrize("truncated", [True, False])
def test_real_http_content_length_completion_required(tmp_path, case, truncated):
    import http.client
    import http.server
    import threading

    body = b"small server rejection"

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(400)
            self.send_header("Content-Length", str(len(body) + (9 if truncated else 0)))
            self.end_headers()
            self.wfile.write(body)
            self.close_connection = True

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/generate"
        if truncated:
            with pytest.raises(http.client.IncompleteRead):
                experiment._native_cases(endpoint, [(case, [0], 0)], tmp_path, "native-smoke")
        else:
            experiment._native_cases(endpoint, [(case, [0], 0)], tmp_path, "native-smoke")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert not thread.is_alive()
    row = json.loads((tmp_path / "native-smoke.json").read_text())[0]
    assert row["accepted_contract"] is not truncated
    assert row.get("http_body_complete", False) is not truncated
    assert row["observed_count_is_lower_bound"] is truncated
    assert row["raw_bytes"] == len(body)
    assert row["remaining_content_length"] == (9 if truncated else 0)
    assert (tmp_path / f"native-smoke-{case}.raw").read_bytes() == body
