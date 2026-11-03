"""Real nonblocking transfer regressions, protected by external test safety timers."""

import multiprocessing as mp
import signal
import socket
import threading
import time
from contextlib import contextmanager

import pytest

from rl.rollout._rpc import HEADER, MAX_FRAME_BYTES, FramedSocket
from rl.rollout._worker import VLLMWorker
from rl.rollout.sampler import Prompt


@contextmanager
def safety_timer():
    def expired(*_args):
        raise AssertionError("external safety timer fired: operation deadline failed")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize("partial", [b"\0", HEADER.pack(100) + b"x"])
def test_partial_header_and_payload_respect_full_frame_deadline(partial):
    parent, peer = socket.socketpair()
    worker = object.__new__(VLLMWorker)
    worker._connection = FramedSocket(parent)
    peer.sendall(partial)
    start = time.monotonic()
    try:
        with safety_timer(), pytest.raises(TimeoutError, match="absolute deadline"):
            worker._receive(timeout=0.03)
        assert 0.02 <= time.monotonic() - start < 0.35
    finally:
        parent.close()
        peer.close()


def test_slow_header_does_not_reset_deadline():
    parent, peer = socket.socketpair()
    channel = FramedSocket(parent)
    stop = threading.Event()

    def drip():
        for value in HEADER.pack(100):
            if stop.wait(0.015):
                return
            peer.send(bytes([value]))

    thread = threading.Thread(target=drip)
    thread.start()
    try:
        with safety_timer(), pytest.raises(TimeoutError):
            channel.receive(time.monotonic() + 0.05)
    finally:
        stop.set()
        thread.join(timeout=1)
        parent.close()
        peer.close()
    assert not thread.is_alive()


@pytest.mark.parametrize("length", [0, MAX_FRAME_BYTES + 1, 2**64 - 1])
def test_invalid_length_rejected_before_payload_read(length):
    parent, peer = socket.socketpair()
    channel = FramedSocket(parent)
    peer.sendall(HEADER.pack(length))
    try:
        with safety_timer(), pytest.raises(ValueError, match="length outside limit"):
            channel.receive(time.monotonic() + 0.03)
    finally:
        parent.close()
        peer.close()


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), object(), {"bad": {1}}, {1: "not a JSON string key"}, "x" * 65537],
)
def test_nonprimitive_nonfinite_and_oversized_strings_rejected(value):
    parent, peer = socket.socketpair()
    try:
        with pytest.raises(ValueError):
            FramedSocket(parent).send(value, time.monotonic() + 0.1)
    finally:
        parent.close()
        peer.close()


def test_normal_json_roundtrip_and_expired_encode_decode(monkeypatch):
    import rl.rollout._rpc as rpc

    parent, peer = socket.socketpair()
    a, b = FramedSocket(parent), FramedSocket(peer)
    value = ["outputs", [{"text": "数学", "ids": [1, 2], "logprobs": [-0.1, -2.0]}]]
    try:
        deadline = time.monotonic() + 0.5
        a.send(value, deadline)
        assert b.receive(deadline) == value
        with pytest.raises(TimeoutError):
            a.send(value, time.monotonic() - 0.01)
        # A complete frame arriving in time cannot hide codec deadline expiry.
        a.send(value, time.monotonic() + 0.5)
        loads = rpc.json.loads

        def slow_loads(data):
            time.sleep(0.04)
            return loads(data)

        monkeypatch.setattr(rpc.json, "loads", slow_loads)
        with safety_timer(), pytest.raises(TimeoutError):
            b.receive(time.monotonic() + 0.02)
    finally:
        parent.close()
        peer.close()


def _idle_cpu_child(ready, ignore_term):
    if ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    time.sleep(30)


@contextmanager
def owned_cpu_worker(*, ignore_term=False):
    context = mp.get_context("fork")
    parent, peer = socket.socketpair()
    worker = object.__new__(VLLMWorker)
    worker._connection = FramedSocket(parent)
    worker._closed = worker._failed = False
    worker.REQUEST_SECONDS = worker.CLOSE_SEND_SECONDS = 0.03
    worker.GRACE_SECONDS = worker.TERM_SECONDS = worker.KILL_SECONDS = 0.03
    ready = context.Event()
    process = worker._process = context.Process(target=_idle_cpu_child, args=(ready, ignore_term))
    process.start()
    assert ready.wait(1), "CPU child did not initialize"
    try:
        yield worker, peer
    finally:
        parent.close()
        peer.close()
        if process.is_alive():
            process.kill()
        process.join(timeout=1)
        process.close()


def _fill_write_buffer(sock):
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    while True:
        try:
            sock.send(b"x" * 65536)
        except BlockingIOError:
            return


def test_blocked_generate_write_reaches_owned_process_cleanup():
    with owned_cpu_worker() as (worker, _peer):
        _fill_write_buffer(worker._connection.socket)
        owner = worker._process
        start = time.monotonic()
        with safety_timer(), pytest.raises(RuntimeError, match="exitcode") as error:
            worker.generate([Prompt("a", "hello")], 1)
        assert isinstance(error.value.__context__, TimeoutError)
        assert time.monotonic() - start < 0.35
        assert worker._process is owner and not owner.is_alive()
        assert worker._closed and worker._failed
        with pytest.raises(RuntimeError, match="closed or failed"):
            worker.generate([Prompt("a", "hello")], 1)


def test_partial_response_timeout_reaps_owned_child():
    with owned_cpu_worker() as (worker, peer):
        peer.sendall(HEADER.pack(100) + b"x")
        with safety_timer(), pytest.raises(RuntimeError, match="exitcode") as error:
            worker.generate([Prompt("a", "hello")], 1)
        assert isinstance(error.value.__context__, TimeoutError)
        assert not worker._process.is_alive()


def test_blocked_close_write_still_escalates_to_owned_kill():
    with owned_cpu_worker(ignore_term=True) as (worker, _peer):
        _fill_write_buffer(worker._connection.socket)
        owner = worker._process
        start = time.monotonic()
        with safety_timer(), pytest.raises(RuntimeError, match="exitcode=-9"):
            worker.close()
        assert time.monotonic() - start < 0.35
        assert worker._process is owner and not owner.is_alive()
        assert worker._closed and worker._failed


def test_live_owner_retained_if_cleanup_cannot_confirm_exit():
    class LiveOwner:
        def is_alive(self):
            return True

        def join(self, timeout):
            pass

        def terminate(self):
            pass

        def kill(self):
            pass

    parent, peer = socket.socketpair()
    worker = object.__new__(VLLMWorker)
    worker._connection = FramedSocket(parent)
    worker._closed = worker._failed = False
    owner = worker._process = LiveOwner()
    try:
        with pytest.raises(RuntimeError, match="did not exit"):
            worker.close()
        assert worker._process is owner and worker._failed and not worker._closed
        with pytest.raises(RuntimeError, match="closed or failed"):
            worker.generate([Prompt("a", "hello")], 1)
    finally:
        parent.close()
        peer.close()


def test_transport_close_error_cannot_skip_owned_process_cleanup(monkeypatch):
    with owned_cpu_worker() as (worker, _peer):
        close = worker._connection.close

        def broken_close():
            close()
            raise OSError("injected transport close error")

        monkeypatch.setattr(worker._connection, "close", broken_close)
        with safety_timer(), pytest.raises(RuntimeError, match="exitcode") as error:
            worker.close()
        assert isinstance(error.value.__context__, OSError)
        assert not worker._process.is_alive()
        assert worker._closed and worker._failed
