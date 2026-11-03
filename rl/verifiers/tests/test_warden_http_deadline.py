"""Real HTTPResponse/socketpair I/O in the owned worker; no service or network.

Only the worker's opener is replaced by a trusted socketpair fixture. Request
serialization, fixed worker launch, stdlib response parsing, body read, parent
absolute deadline, kill/reap and response decoding follow production code.
"""

import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from rl.verifiers.tests.test_warden_adapter import PUBLIC, image
from rl.verifiers.warden import WardenSandbox


@contextmanager
def socket_response(monkeypatch, tmp_path, phase, body=b'{"ok":true}'):
    reader, writer = socket.socketpair()
    writer.settimeout(0.2)
    marker = tmp_path / "phase"
    fixture = tmp_path / "trusted_http_fixture.py"
    fixture.write_text(
        "import sys, socket, time\n"
        "from pathlib import Path\n"
        "from http.client import HTTPResponse\n"
        "from urllib.error import URLError\n"
        f"sys.path.insert(0, {str(Path.cwd())!r})\n"
        "from rl.verifiers import _warden_http as worker\n"
        "reader = socket.socket(fileno=int(sys.argv[1]))\n"
        f"marker = Path({str(marker)!r})\n"
        f"phase = {phase!r}\n"
        "class SocketOpener:\n"
        " def open(self, request, timeout):\n"
        '  assert request.get_header("Authorization") == "Bearer fixture-token"\n'
        "  reader.settimeout(timeout)\n"
        '  marker.write_text("connect" if phase == "connect" else "header")\n'
        '  if phase == "connect": time.sleep(5)\n'
        '  if phase == "inactivity": raise URLError(TimeoutError())\n'
        "  response = HTTPResponse(reader)\n"
        "  response.begin()\n"
        '  marker.write_text("body")\n'
        "  return response\n"
        "worker.build_opener = lambda *args: SocketOpener()\n"
        "worker.main()\n"
    )
    workers, original_args = [], []
    original_popen = subprocess.Popen

    def launch(argv, **kwargs):
        original_args.append(argv)
        assert argv[1:3] == ["-I", "-S"]
        assert Path(argv[3]).name == "_warden_http.py"
        # Pass only a fixture socket FD; credentials/body still travel on stdin.
        process = original_popen(
            [argv[0], "-I", "-S", str(fixture), str(reader.fileno())],
            pass_fds=(reader.fileno(),),
            **kwargs,
        )
        reader.close()  # Worker now owns the only reader endpoint.
        workers.append(process)
        return process

    monkeypatch.setattr("rl.verifiers.warden.subprocess.Popen", launch)
    stop = threading.Event()
    headers = f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n\r\n".encode()

    def send_response():
        try:
            for data, slow in ((headers, phase == "header"), (body, phase == "body")):
                if slow:
                    for byte in data:
                        if stop.wait(0.01):
                            return
                        writer.sendall(bytes([byte]))
                else:
                    writer.sendall(data)
        except OSError:
            pass
        finally:
            writer.close()

    sender = threading.Thread(target=send_response)
    sender.start()
    try:
        yield workers, marker, original_args
    finally:
        stop.set()
        reader.close()
        sender.join(timeout=1)
        assert not sender.is_alive()
        assert workers and all(worker.returncode is not None for worker in workers)
        assert all(worker.stdin.closed and worker.stdout.closed for worker in workers)


def adapter():
    return WardenSandbox("https://warden.invalid", "fixture-token", image())


@pytest.mark.parametrize("phase", ["connect", "header", "body"])
def test_actual_transport_has_absolute_deadline_and_reaps(monkeypatch, tmp_path, phase):
    # Each drip interval is below the inactivity timeout; total transfer is >1s.
    body = b'{"ok":' + b" " * 150 + b"true}"
    with socket_response(monkeypatch, tmp_path, phase, body) as (workers, marker, argv):
        sandbox = adapter()
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            sandbox._request("GET", "/v1/runs/fixture-run", None, None, 0.4)
        elapsed = time.monotonic() - started
        assert elapsed < 0.65, elapsed  # Allows scheduler overhead, not a >1s trickle.
        assert marker.read_text() == phase  # Must actually reach the tested phase.
        assert sandbox._http_worker is None
        assert workers[0].returncode is not None
        assert "fixture-token" not in " ".join(argv[0])


def test_actual_transport_prompt_response_and_reaping(monkeypatch, tmp_path):
    with socket_response(monkeypatch, tmp_path, "prompt"):
        sandbox = adapter()
        assert sandbox._request("GET", "/v1/runs/fixture-run", None, None, 1) == (200, {"ok": True})
        assert sandbox._http_worker is None


@pytest.mark.parametrize("body", [b"not-json", b" " * (2 * 1024**2 + 1)])
def test_actual_transport_invalid_or_oversized_body_is_bounded(monkeypatch, tmp_path, body):
    with socket_response(monkeypatch, tmp_path, "prompt", body):
        sandbox = adapter()
        with pytest.raises(ValueError, match="http_exchange_failed"):
            sandbox._request("GET", "/v1/runs/fixture-run", None, None, 2)
        assert sandbox._http_worker is None


def test_actual_transport_timed_out_post_retains_identity(monkeypatch, tmp_path):
    with socket_response(monkeypatch, tmp_path, "connect"):
        sandbox = adapter()
        result = sandbox.run("def solve(x): return x", 1, PUBLIC, 0.4)
        assert result.timed_out and not result.cleanup_confirmed and result.run_id is None
        assert result.idempotency_key in sandbox.pending
        assert sandbox.pending[result.idempotency_key]["payload"]["runtime"] == "gvisor"
        assert sandbox._http_worker is None


@pytest.mark.parametrize("failure", ["reap", "signal"])
def test_local_worker_cleanup_uncertainty_retains_owner_and_blocks_new_io(monkeypatch, failure):
    """Synthetic unreapable process only; never leaves a real process behind."""
    import io

    from rl.verifiers.warden import HTTPWorkerCleanupError

    class UnreapedFixture:
        pid = 12345
        returncode = None
        stdin = io.BytesIO()
        stdout = io.BytesIO()

        def poll(self):
            return self.returncode

        def communicate(self, *args, **kwargs):
            raise subprocess.TimeoutExpired("fixture-only", kwargs["timeout"])

    launches = []
    worker = UnreapedFixture()

    def launch(*args, **kwargs):
        launches.append(args)
        return worker

    def signal_owned(pid, signal):
        assert pid == worker.pid
        if failure == "signal":
            raise PermissionError("fixture-only")

    monkeypatch.setattr("rl.verifiers.warden.subprocess.Popen", launch)
    monkeypatch.setattr("rl.verifiers.warden.os.killpg", signal_owned)
    sandbox = adapter()
    result = sandbox.run("def solve(x): return x", 1, PUBLIC, 1)
    assert result.error == "local_http_cleanup_unknown:pid=12345"
    assert not result.cleanup_confirmed and result.idempotency_key in sandbox.pending
    assert sandbox._http_worker is worker
    with pytest.raises(HTTPWorkerCleanupError):
        sandbox._request("GET", "/v1/runs/fixture-run", None, None, 1)
    assert len(launches) == 1 and sandbox._http_worker is worker
    # Fixture becomes reaped: recovery may close its descriptors, no live I/O.
    worker.returncode = -9
    sandbox._close_http_worker()
    assert worker.stdin.closed and worker.stdout.closed and sandbox._http_worker is None


def test_actual_transport_wrapped_socket_timeout_preserves_timeout_type(monkeypatch, tmp_path):
    with socket_response(monkeypatch, tmp_path, "inactivity"):
        sandbox = adapter()
        with pytest.raises(TimeoutError):
            sandbox._request("GET", "/v1/runs/fixture-run", None, None, 1)
        assert sandbox._http_worker is None
