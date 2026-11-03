import hashlib
import http.server
import signal
import threading
import time

import pytest

from serving.baseline.deadline_http import absolute_deadline


@pytest.fixture
def trickle_server():
    state = {"mode": "redirect", "closed": threading.Event()}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            try:
                if self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", "/data")
                    self.send_header("Content-Length", "1000000000")
                    self.end_headers()
                    self.close_connection = False
                    return
                if state["mode"] == "header":
                    self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Trickle: ")
                    for _ in range(100):
                        self.wfile.write(b"x")
                        self.wfile.flush()
                        time.sleep(0.01)
                    self.wfile.write(b"\r\nContent-Length: 4\r\n\r\ndata")
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", "4")
                    self.end_headers()
                    for byte in b"data":
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        if state["mode"] == "body":
                            time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                state["closed"].set()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False  # server_close joins owned request threads.
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    yield state, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)
    assert not worker.is_alive()


@pytest.mark.parametrize("kind", ["wheel", "model"])
@pytest.mark.parametrize("mode", ["redirect", "header", "body"])
def test_absolute_http_deadline_and_redirect_without_drain(tmp_path, trickle_server, kind, mode):
    from serving.baseline.model_snapshot import cache_paths, ensure_file
    from serving.baseline.model_snapshot_test import item
    from serving.baseline.prepare_runtime import download_wheel

    state, origin = trickle_server
    state["mode"] = mode
    url = origin + ("/redirect" if mode == "redirect" else "/data")
    deadline = time.monotonic() + (0.8 if mode == "redirect" else 0.15)
    started = time.monotonic()
    if kind == "wheel":
        target = tmp_path / "data.whl"
        call = lambda: download_wheel(
            {"url": url, "compressed_bytes": 4, "sha256": hashlib.sha256(b"data").hexdigest()},
            target,
            test_loopback=True,
            deadline=deadline,
        )
    else:
        expected = item(b"data")
        target = cache_paths(tmp_path, expected)[0]
        call = lambda: ensure_file(
            expected,
            tmp_path,
            download=True,
            test_url=url,
            deadline=deadline,
            observe=lambda remaining: None,
        )
    if mode == "redirect":
        call()
        assert target.read_bytes() == b"data"
    else:
        with pytest.raises(TimeoutError):
            call()
        assert not target.exists()
    assert time.monotonic() - started < 1.2


def test_alarm_interrupts_blocking_phase_and_restores_state():
    previous = signal.getsignal(signal.SIGALRM)
    handler = lambda *args: None
    signal.signal(signal.SIGALRM, handler)
    try:
        with pytest.raises(TimeoutError), absolute_deadline(time.monotonic() + 0.05):
            time.sleep(1)
        assert signal.getsignal(signal.SIGALRM) is handler
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
        signal.setitimer(signal.ITIMER_REAL, 5)
        with pytest.raises(RuntimeError, match="active"), absolute_deadline(time.monotonic() + 1):
            pass
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 4
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def test_late_wheel_publication_never_becomes_success(tmp_path, trickle_server, monkeypatch):
    from serving.baseline import prepare_runtime

    _, origin = trickle_server
    original = prepare_runtime.os.link

    def late_link(source, target):
        time.sleep(0.5)
        original(source, target)

    monkeypatch.setattr(prepare_runtime.os, "link", late_link)
    target = tmp_path / "late.whl"
    with pytest.raises(TimeoutError):
        prepare_runtime.download_wheel(
            {
                "url": origin + "/data",
                "compressed_bytes": 4,
                "sha256": hashlib.sha256(b"data").hexdigest(),
            },
            target,
            test_loopback=True,
            deadline=time.monotonic() + 0.15,
        )
    assert not target.exists()
    assert target.with_suffix(".whl.partial").read_bytes() == b"data"


def test_owned_dns_child_is_killed_and_reaped_at_absolute_deadline(monkeypatch):
    from serving.baseline import deadline_http as http

    original = http.subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(http.subprocess, "Popen", spawn)
    monkeypatch.setattr(http, "_RESOLVE", "import time;time.sleep(60)")
    deadline = time.monotonic() + 0.1
    with pytest.raises(TimeoutError), absolute_deadline(deadline):
        http.resolve("fixture.invalid", 443, deadline)
    assert len(children) == 1 and children[0].poll() is not None
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_pending_alarm_is_drained_before_restoring_prior_handler():
    import os

    previous = signal.getsignal(signal.SIGALRM)
    called = []
    signal.signal(signal.SIGALRM, lambda *args: called.append(True))
    try:
        with absolute_deadline(time.monotonic() + 1):
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
            os.kill(os.getpid(), signal.SIGALRM)
        assert called == [] and signal.SIGALRM not in signal.sigpending()
    finally:
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize("phase", ["final-read", "hash"])
def test_late_final_block_or_model_hash_cannot_publish(
    tmp_path, trickle_server, monkeypatch, phase
):
    from serving.baseline import deadline_http, model_snapshot, prepare_runtime
    from serving.baseline.model_snapshot_test import item

    _, origin = trickle_server
    if phase == "final-read":
        original = deadline_http.open_response

        def delayed_open(*args, **kwargs):
            response = original(*args, **kwargs)
            read = response.read1

            def late_read(size):
                data = read(size)
                time.sleep(0.5)
                return data

            response.read1 = late_read
            return response

        monkeypatch.setattr(deadline_http, "open_response", delayed_open)
        target = tmp_path / "late.whl"
        with pytest.raises(TimeoutError):
            prepare_runtime.download_wheel(
                {
                    "url": origin + "/data",
                    "compressed_bytes": 4,
                    "sha256": hashlib.sha256(b"data").hexdigest(),
                },
                target,
                test_loopback=True,
                deadline=time.monotonic() + 0.15,
            )
    else:
        original = model_snapshot.verify_file

        def late_hash(*args, **kwargs):
            result = original(*args, **kwargs)
            time.sleep(0.5)
            return result

        monkeypatch.setattr(model_snapshot, "verify_file", late_hash)
        expected = item(b"data")
        target = model_snapshot.cache_paths(tmp_path, expected)[0]
        with pytest.raises(TimeoutError):
            model_snapshot.ensure_file(
                expected,
                tmp_path,
                download=True,
                test_url=origin + "/data",
                deadline=time.monotonic() + 0.15,
                observe=lambda remaining: None,
            )
    assert not target.exists()


def test_preexisting_timer_expiring_during_inspection_is_not_stolen(monkeypatch):
    import os

    previous = signal.getsignal(signal.SIGALRM)
    called = []
    signal.signal(signal.SIGALRM, lambda *args: called.append(True))

    def expired_during_inspection(which):
        os.kill(os.getpid(), signal.SIGALRM)
        return (0.0, 0.0)

    monkeypatch.setattr(signal, "getitimer", expired_during_inspection)
    try:
        with pytest.raises(RuntimeError, match="pending"), absolute_deadline(time.monotonic() + 1):
            pass
        assert called == [True]
    finally:
        signal.signal(signal.SIGALRM, previous)
