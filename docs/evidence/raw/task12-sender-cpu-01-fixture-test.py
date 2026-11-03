"""Focused preparation only; run after root source gate, never imports native code."""

import errno
import importlib.util
import json
import os
import queue
import signal
import socket
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

SOURCE = Path(__file__).with_name("task12-sender-cpu-01-fixture.py")
spec = importlib.util.spec_from_file_location("sender_cpu_fixture", SOURCE)
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


def request():
    body = b'{"input_ids":[1],"stream":true,"sampling_params":{"ignore_eos":true,"max_new_tokens":2,"temperature":0}}'
    return {
        f.digest(body): {
            "index": 0,
            "bytes": len(body),
            "body": body,
            "input": 1,
            "output": 2,
            "scheduled_ms": 0,
        }
    }


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def stub(text):
    # Explicit tiny Python child only; never the frozen replay executable.
    return [sys.executable, "-B", "-c", text]


def assert_cleanup(terminal):
    child = terminal["cleanup"]["child"]
    assert child["reaped"] and child["pid_absent"]
    fixture = terminal["cleanup"]["fixture"]
    assert fixture["listener_closed"] and fixture["port_listener_absent"]
    assert fixture["threads_alive"] == 0


def test_exact_burst_body_identity_and_bounds():
    trace = (f.ROOT / f.TRACE).read_bytes()
    ids = json.loads((f.ROOT / f.TOKENS).read_bytes())["token_ids"]
    expected = f.prepare(trace, ids)
    assert len(expected) == 23
    rows = sorted(expected.values(), key=lambda x: x["index"])
    assert [r["bytes"] for r in rows[:4]] == [6977, 172, 2673, 2701]
    assert (
        f.digest(rows[2]["body"])
        == "e7dc38eea9dba6aafd58b25f0e35324a76f649d35bcc0ff3876904ef17ca91c0"
    )
    assert sum(r["output"] for r in rows) == 1556
    assert max(r["bytes"] for r in rows) <= f.BODY_CAP
    duplicate = b"Timestamp,Request tokens,Response tokens\n" + b"553788,0,1\n" * 23
    with pytest.raises(ValueError, match="ambiguous"):
        f.prepare(duplicate, ids)
    oversized = trace.replace(b"1459,294", b"32768,294", 1)
    with pytest.raises(ValueError, match="oversized"):
        f.prepare(oversized, ids)


@pytest.mark.parametrize(
    "headers",
    [
        b"Content-Length: 2\r\nContent-Length: 2",
        b"Transfer-Encoding: chunked",
        b"Content-Length: 16385",
        b"Content-Length: nope",
        b"bad\x00: x",
    ],
)
def test_http_framing_rejects_malformed_and_bounds(headers):
    # Complete invalid headers need no receive operation.
    with socket.socket() as unused, pytest.raises(ValueError):
        f.read_request(
            unused, b"POST /generate HTTP/1.1\r\n" + headers + b"\r\n\r\n", completed=False
        )


def test_bounded_writer_refuses_before_append(tmp_path):
    path = tmp_path / "bounded"
    writer = f.BoundedFile(path, 8)
    try:
        writer.write(b"1234")
        with pytest.raises(ValueError, match="cap"):
            writer.write(b"56789")
        assert path.read_bytes() == b"1234"
    finally:
        writer.close()


def test_stopping_does_not_swallow_evidence_eio():
    fixture = f.Fixture.__new__(f.Fixture)
    fixture.stopping = threading.Event()
    fixture.stopping.set()
    fixture.errors = queue.Queue(maxsize=32)

    class BadEvidence:
        def write(self, _data):
            raise OSError(errno.EIO, "evidence write failed")

    class Connection:
        def close(self):
            self.closed = True

    fixture.events = BadEvidence()
    conn = Connection()
    fixture.handle(conn, 1)
    assert conn.closed
    assert "evidence write failed" in fixture.errors.get_nowait()["error"]
    with pytest.raises(OSError, match="evidence write failed"):
        fixture.socket_io(BadEvidence().write, b"x")  # EIO is never a closure errno.


def test_fast_normal_stub_keepalive_count_and_cleanup(tmp_path):
    expected, port = request(), free_port()
    body = next(iter(expected.values()))["body"]
    second = body.replace(b'input_ids":[1]', b'input_ids":[2]')
    expected[f.digest(second)] = dict(next(iter(expected.values())), index=1, body=second)
    code = f"""
import http.client, json
c = http.client.HTTPConnection("127.0.0.1", {port}, timeout=2)
c.request("POST", "/generate", body={body!r}, headers={{"Content-Type":"application/json"}})
r = c.getresponse()
b = r.read()
assert r.status == 200 and b.count(b"data:") == 2 and b"[DONE]" in b
assert b'"completion_tokens":2' in b and b'"output_ids":[0,0]' in b
fd = c.sock.fileno()
c.request("POST", "/generate", body={second!r}, headers={{"Content-Type":"application/json"}})
r = c.getresponse()
assert r.status == 200 and b"[DONE]" in r.read() and c.sock.fileno() == fd
c.close()
print(json.dumps({{"Run":{{"Aborted":False,"SenderValid":True,"Records":[{{"Status":"ok","OutputTokens":2,"ScheduledMs":0,"SentMs":0}}] * 2}}}}))
"""
    terminal = f.screen(stub(code), expected, tmp_path / "normal", port=port, child_seconds=3)
    assert terminal["screen_completed"], terminal
    assert terminal["result"]["records"] == 2
    assert_cleanup(terminal)


def test_timeout_stub_is_terminated_reaped_and_signal_restored(tmp_path):
    before = signal.getsignal(signal.SIGXFSZ)
    terminal = f.screen(
        stub("import time; time.sleep(10)"),
        request(),
        tmp_path / "timeout",
        port=free_port(),
        child_seconds=0.15,
    )
    assert not terminal["screen_completed"] and "deadline" in terminal["failure"]
    assert terminal["cleanup"]["child"]["termination_requested"]
    assert signal.getsignal(signal.SIGXFSZ) == before
    assert_cleanup(terminal)


def test_real_child_hard_cap_signal_and_cleanup(tmp_path):
    code = """
import errno, os, resource, signal
assert resource.getrlimit(resource.RLIMIT_FSIZE) == (65536,65536)
assert resource.getrlimit(resource.RLIMIT_CORE) == (0,0)
assert signal.getsignal(signal.SIGXFSZ) == signal.SIG_IGN
os.write(1, b"x" * 65536)
try:
    os.write(1, b"!")
except OSError as e:
    assert e.errno == errno.EFBIG
else:
    raise AssertionError("hard limit not enforced")
"""
    out = tmp_path / "cap"
    terminal = f.screen(stub(code), request(), out, port=free_port(), child_seconds=3)
    assert not terminal["screen_completed"] and "cap" in terminal["failure"]
    assert (out / "stdout.json").stat().st_size == 65536
    assert_cleanup(terminal)


@pytest.mark.parametrize("fail_at", [1, 2])
def test_first_or_later_proc_failure_still_cleans_original_popen(tmp_path, fail_at):
    original = f.process_sample
    child_calls = 0

    def denied(pid, expected_start=None):
        nonlocal child_calls
        if pid != os.getpid():
            child_calls += 1
            if child_calls >= fail_at:
                raise PermissionError("owned proc sample denied")
        return original(pid, expected_start)

    with patch.object(f, "process_sample", side_effect=denied):
        terminal = f.screen(
            stub("import time; time.sleep(10)"),
            request(),
            tmp_path / f"identity-{fail_at}",
            port=free_port(),
            child_seconds=3,
        )
    assert not terminal["screen_completed"]
    assert "owned proc sample denied" in terminal["failure"]
    assert terminal["cleanup"]["child"]["termination_requested"]
    assert_cleanup(terminal)


def test_pid_reuse_rejected_without_relaxing_sampler():
    current = f.process_sample(os.getpid())
    with pytest.raises(ValueError, match="identity changed"):
        f.process_sample(os.getpid(), current["starttime"] + 1)


@pytest.mark.parametrize("unknown", [False, True])
def test_duplicate_or_unknown_body_is_retained_failure(tmp_path, unknown):
    expected = request()
    body = next(iter(expected.values()))["body"]
    events = f.BoundedFile(tmp_path / "events", 4096)
    fixture = f.Fixture(expected, events, 0)
    peer, server = socket.socketpair()
    peer.settimeout(1)
    server.settimeout(1)
    second = body.replace(b"[1]", b"[9]") if unknown else body

    def packet(data):
        return (
            b"POST /generate HTTP/1.1\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(data)}\r\n\r\n".encode()
            + data
        )

    try:
        peer.sendall(packet(body) + packet(second))
        fixture.handle(server, 1)
        assert len(fixture.seen) == 1
        assert "unknown or duplicate" in fixture.errors.get_nowait()["error"]
    finally:
        peer.close()
        server.close()
        cleanup = fixture.close()
        events.close()
        assert cleanup["listener_closed"] and cleanup["threads_alive"] == 0
