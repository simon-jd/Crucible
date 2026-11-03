"""HTTP protocol fixtures only; no Warden service is contacted."""

import base64
import hashlib
from pathlib import Path

import pytest

from rl.verifiers.warden import RunnerImage, WardenSandbox

PUBLIC = '[{"expected":2,"input":1}]'


def image():
    return RunnerImage(
        "sha256:" + "a" * 64,
        hashlib.sha256(PUBLIC.encode()).hexdigest(),
        hashlib.sha256(Path("rl/verifiers/_candidate.py").read_bytes()).hexdigest(),
    )


class ServiceFixture:
    def __init__(self):
        self.calls = []
        self.payload = None
        self.state = "succeeded"
        self.result = {
            "Code": 0,
            "OOMKilled": False,
            "TimedOut": False,
            "CgroupEmpty": True,
            "LogsTruncated": False,
            "DroppedLogBytes": 0,
            "PidsMaxEvents": 0,
            "PeakPids": 1,
        }
        self.logs = {"logs": ["2"], "status": "complete", "truncated": False, "dropped_bytes": 0}
        self.cleanup_pending = False
        self.post_error = None
        self.spec_override = {}

    def __call__(self, method, path, payload, key, timeout):
        assert timeout > 0
        self.calls.append((method, path, payload, key))
        if method == "POST":
            assert path == "/v1/runs"
            self.payload = payload
            if self.post_error:
                raise self.post_error
            return 201, {"run_id": "fixture-run"}
        if path.endswith("/logs"):
            return 200, self.logs
        spec = dict(self.payload, rootfs_readonly=True, uid=65534, no_new_privileges=True)
        spec.update(self.spec_override)
        return 200, {
            "run_id": "fixture-run",
            "state": self.state,
            "spec": spec,
            "result": self.result,
            "cleanup_pending": self.cleanup_pending,
        }


def client(fixture, runner=None):
    return WardenSandbox(
        "https://warden.invalid",
        "fixture-token",
        runner or image(),
        transport=fixture,
        poll_interval_s=0,
    )


def run(fixture):
    return client(fixture).run("def solve(x): return x+1", 1, PUBLIC, 1)


def test_warden_actual_schema_and_no_invented_fields():
    service = ServiceFixture()
    result = run(service)
    assert result.completed and result.cleanup_confirmed and result.stdout == "2"
    assert set(service.payload) == {"image_digest", "cmd", "runtime", "network", "limits"}
    assert service.payload["runtime"] == "gvisor" and service.payload["network"] == "none"
    assert service.payload["limits"] == {"cpu": 1, "memory_mib": 512, "pids": 128, "timeout_s": 1}
    request = __import__("json").loads(base64.b64decode(service.payload["cmd"][-1]))
    assert request == {"input": 1, "solution": "def solve(x): return x+1"}
    assert service.calls[0][3]
    assert [call[1] for call in service.calls] == [
        "/v1/runs",
        "/v1/runs/fixture-run",
        "/v1/runs/fixture-run/logs",
    ]


def test_warden_missing_image_does_not_contact_service():
    fixture = ServiceFixture()
    adapter = WardenSandbox("https://warden.invalid", "token", None, transport=fixture)
    result = adapter.run("def solve(x): return x", 1, PUBLIC, 1)
    assert result.error == "warden_runner_image_prerequisites_missing" and not fixture.calls


def test_warden_wrong_public_image_fails_closed():
    fixture = ServiceFixture()
    assert client(fixture).run("def solve(x): return x", 1, "[]", 1).error
    assert not fixture.calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("TimedOut", True),
        ("OOMKilled", True),
        ("CgroupEmpty", False),
        ("Code", 1),
        ("LogsTruncated", True),
        ("DroppedLogBytes", 2),
        ("PidsMaxEvents", 1),
        ("PeakPids", "1"),
        ("Code", False),
    ],
)
def test_warden_bad_runtime_result(field, value):
    fixture = ServiceFixture()
    fixture.result[field] = value
    assert not run(fixture).completed


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "partial"),
        ("status", "lost"),
        ("status", "pending"),
        ("truncated", True),
        ("dropped_bytes", 1),
        ("encoding", "unknown"),
        ("logs", "2"),
    ],
)
def test_warden_incomplete_logs(field, value):
    fixture = ServiceFixture()
    fixture.logs[field] = value
    assert not run(fixture).completed


def test_warden_base64_chunks_joined_before_decoding():
    fixture = ServiceFixture()
    fixture.logs.update(logs=["M", "g", "=="], encoding="base64")
    assert run(fixture).stdout == "2"


def test_warden_empty_complete_logs_are_valid_transport_result():
    fixture = ServiceFixture()
    fixture.logs["logs"] = []
    result = run(fixture)
    assert result.completed and result.stdout == ""


def test_warden_cleanup_pending_is_not_success():
    fixture = ServiceFixture()
    fixture.cleanup_pending = True
    result = run(fixture)
    assert not result.cleanup_confirmed and not result.completed


@pytest.mark.parametrize(
    "change",
    [
        {"runtime": "runc"},
        {"network": "host"},
        {"uid": 0},
        {"rootfs_readonly": False},
        {"cmd": ["echo", "PASS"]},
    ],
)
def test_warden_authoritative_spec_must_match(change):
    fixture = ServiceFixture()
    fixture.spec_override = change
    assert not run(fixture).completed


def test_warden_lost_post_retains_key_and_exact_retry():
    fixture = ServiceFixture()
    fixture.post_error = TimeoutError()
    adapter = client(fixture)
    uncertain = adapter.run("def solve(x): return x+1", 1, PUBLIC, 1)
    assert uncertain.timed_out and not uncertain.cleanup_confirmed and uncertain.run_id is None
    assert uncertain.idempotency_key in adapter.pending
    first = fixture.calls[0]
    fixture.post_error = None
    result = adapter.retry(uncertain.idempotency_key, 1)
    assert result.completed and fixture.calls[1] == first


def test_warden_queue_timeout_preserves_remote_identity_without_cancel():
    fixture = ServiceFixture()
    fixture.state = "queued"
    adapter = client(fixture)
    result = adapter.run("def solve(x): return x", 1, PUBLIC, 0.005)
    assert result.timed_out and result.run_id == "fixture-run" and not result.cleanup_confirmed
    assert result.idempotency_key in adapter.pending
    assert all("/cancel" not in call[1] for call in fixture.calls)


def test_warden_rejects_redirect_and_insecure_origin():
    from rl.verifiers._warden_http import _NoRedirect

    with pytest.raises(ValueError):
        WardenSandbox("http://localhost:8080", "token", image())
    with pytest.raises(ValueError):
        _NoRedirect().redirect_request(None, None, 302, None, None, "https://elsewhere.invalid")


@pytest.mark.parametrize("outcome", ["failed", "timeout", "interrupted", "succeeded"])
def test_warden_confirmed_clean_terminal_evicts_submission(outcome):
    fixture = ServiceFixture()
    fixture.state = outcome if outcome in ("interrupted", "succeeded") else "failed"
    fixture.result.update(Code=0 if outcome == "succeeded" else 1, TimedOut=outcome == "timeout")
    adapter = client(fixture)
    result = adapter.run("def solve(x): return x", 1, PUBLIC, 1)
    assert result.cleanup_confirmed and result.completed == (outcome == "succeeded")
    assert result.idempotency_key not in adapter.pending


@pytest.mark.parametrize(
    "problem", ["pending_cleanup", "malformed_result", "bad_range", "wrong_spec", "missing_result"]
)
def test_warden_uncertain_terminal_retains_submission(problem):
    fixture = ServiceFixture()
    if problem == "pending_cleanup":
        fixture.cleanup_pending = True
    elif problem == "malformed_result":
        fixture.result["Code"] = False
    elif problem == "bad_range":
        fixture.result["PeakPids"] = -1
    elif problem == "wrong_spec":
        fixture.spec_override["runtime"] = "runc"
    else:
        fixture.result = None
    sandbox = client(fixture)
    result = sandbox.run("def solve(x): return x", 1, PUBLIC, 1)
    assert not result.completed and not result.cleanup_confirmed
    assert result.idempotency_key in sandbox.pending


@pytest.mark.parametrize(
    "problem", ["partial", "encoding", "oversized", "invalid_base64", "timeout", "transport"]
)
def test_warden_known_clean_log_failure_evicts_and_preserves_cleanup(problem):
    fixture = ServiceFixture()
    if problem == "partial":
        fixture.logs["status"] = "partial"
    elif problem == "encoding":
        fixture.logs["encoding"] = "unknown"
    elif problem == "oversized":
        fixture.logs["logs"] = ["a" * 65537]
    elif problem == "invalid_base64":
        fixture.logs.update(encoding="base64", logs=["!"])

    def transport(method, path, payload, key, timeout):
        if path.endswith("/logs"):
            if problem == "timeout":
                raise TimeoutError
            if problem == "transport":
                raise OSError
        return fixture(method, path, payload, key, timeout)

    sandbox = client(transport)
    result = sandbox.run("def solve(x): return x", 1, PUBLIC, 1)
    assert result.cleanup_confirmed and not result.completed
    assert result.idempotency_key not in sandbox.pending


def test_warden_many_confirmed_failures_do_not_accumulate_payloads():
    fixture = ServiceFixture()
    fixture.state = "failed"
    fixture.result["Code"] = 1
    sandbox = client(fixture)
    for _ in range(200):
        result = sandbox.run("def solve(x): return x", 1, PUBLIC, 1)
        assert result.cleanup_confirmed and result.error == "remote_execution_failed"
        assert not sandbox.pending
