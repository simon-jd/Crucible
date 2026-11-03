import json
import signal
import sys
import time
from pathlib import Path

import pytest

from cluster.health import remote_worker
from cluster.health.ib_perftest import build_pairs, run_pair
from cluster.health.report import validate_site
from cluster.health.runner import CommandRunner, Result
from cluster.health.tests.test_execution import site
from cluster.health.tests.test_suite import inventory


def test_remote_worker_signal_reaps_owned_process(tmp_path):
    token = "b" * 32
    started = CommandRunner().start(
        [
            sys.executable,
            remote_worker.__file__,
            "run",
            "--token",
            token,
            "--deadline-epoch",
            str(time.time() + 60),
            "--state-dir",
            str(tmp_path),
            "--",
            "sleep",
            "60",
        ]
    )
    state = tmp_path / (token + ".json")
    deadline = time.monotonic() + 3
    while not state.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        pid = json.loads(state.read_text())["pid"]
        started.process.send_signal(signal.SIGTERM)
        result = started.finish(3)
        assert result.returncode == 128 + signal.SIGTERM
        assert not Path(f"/proc/{pid}").exists()
        assert not state.exists()
    finally:
        started.terminate()


def test_server_readiness_failure_cleans_both_tokens():
    class Process:
        def wait_ready(self, timeout):
            return False

        def terminate(self):
            pass

        def finish(self, timeout):
            return Result(["SYNTHETIC-server"], 124, "", "timeout", True)

    class Runner:
        def __init__(self):
            self.calls = []

        def start(self, argv):
            return Process()

        def run(self, argv, timeout):
            self.calls.append(argv)
            return Result(argv, 0, "", "")

    runner = Runner()
    result = run_pair(build_pairs(inventory())[0], inventory(), runner, timeout=0.1)
    assert not result["passed"]
    assert len(runner.calls) == 2
    assert all(" stop " in argv[-1] for argv in runner.calls)
    assert len({argv[-1].split()[-1] for argv in runner.calls}) == 2


def test_remote_worker_does_not_remove_another_token_state(tmp_path):
    token = "c" * 32
    state = tmp_path / (token + ".json")
    state.write_text("SYNTHETIC existing owner state")
    result = CommandRunner().run(
        [
            sys.executable,
            remote_worker.__file__,
            "run",
            "--token",
            token,
            "--deadline-epoch",
            str(time.time() + 0.1),
            "--state-dir",
            str(tmp_path),
            "--",
            "sleep",
            "60",
        ],
        timeout=2,
    )
    assert result.returncode != 0
    assert state.read_text() == "SYNTHETIC existing owner state"


@pytest.mark.parametrize(
    "patch",
    [
        {"max_cost_usd": 1},
        {"stop_at_utc": "2020-01-01T00:00:00+00:00"},
        {"stop_at_utc": "2099-01-01T00:00:00+00:00"},
        {"owner": ""},
        {"approval_reference": ""},
        {"stop_at_utc": "2026-10-23T00:00:00"},
    ],
)
def test_live_preflight_rejects_cost_or_deadline(tmp_path, patch):
    plan = site(tmp_path)
    plan["inventory"]["execution"].update(patch)
    with pytest.raises(ValueError):
        validate_site(plan)
