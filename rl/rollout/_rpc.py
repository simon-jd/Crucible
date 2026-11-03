"""Bounded primitive JSON frames over owned nonblocking Unix sockets.

Every complete operation uses an absolute monotonic deadline, including header,
payload, encoding/decoding and backpressure. No blocking Connection.send/recv or
serializer callbacks run here. Frame and primitive limits bound codec work;
deadlines are checked during validation/encoding and before/after JSON decoding.
"""

import json
import math
import selectors
import struct
import time

MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_STRING_CHARS = 64 * 1024
MAX_DEPTH = 12
HEADER = struct.Struct("!Q")


def check_deadline(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("rollout RPC absolute deadline exceeded")
    return remaining


def _validate(value, deadline, depth=0):
    check_deadline(deadline)
    if depth > MAX_DEPTH:
        raise ValueError("rollout RPC nesting exceeds limit")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("rollout RPC requires finite numbers")
        return
    if type(value) is str:
        if len(value) > MAX_STRING_CHARS:
            raise ValueError("rollout RPC string exceeds limit")
        return
    if type(value) in (list, tuple, dict):
        if len(value) > MAX_FRAME_BYTES:
            raise ValueError("rollout RPC container exceeds limit")
        for item in value:
            if type(value) is dict:
                if type(item) is not str:
                    raise ValueError("rollout RPC object keys must be strings")
                _validate(item, deadline, depth + 1)
                _validate(value[item], deadline, depth + 1)
            else:
                _validate(item, deadline, depth + 1)
        return
    raise ValueError("rollout RPC accepts only primitive JSON values")


class FramedSocket:
    def __init__(self, sock):
        self.socket = sock
        sock.setblocking(False)

    def _wait(self, event, deadline):
        with selectors.DefaultSelector() as selector:
            selector.register(self.socket, event)
            while not selector.select(check_deadline(deadline)):
                check_deadline(deadline)
        check_deadline(deadline)

    def _write(self, data, deadline):
        view = memoryview(data)
        while view:
            check_deadline(deadline)
            try:
                count = self.socket.send(view[:65536])
            except BlockingIOError:
                self._wait(selectors.EVENT_WRITE, deadline)
                continue
            if not count:
                raise ConnectionError("rollout RPC peer closed during write")
            view = view[count:]
        check_deadline(deadline)

    def _read(self, length, deadline):
        payload = bytearray()
        while len(payload) < length:
            check_deadline(deadline)
            try:
                chunk = self.socket.recv(min(65536, length - len(payload)))
            except BlockingIOError:
                self._wait(selectors.EVENT_READ, deadline)
                continue
            if not chunk:
                raise ConnectionError("rollout RPC peer closed during frame")
            payload.extend(chunk)
        check_deadline(deadline)
        return payload

    def send(self, value, deadline):
        _validate(value, deadline)
        payload = bytearray()
        encoder = json.JSONEncoder(allow_nan=False, ensure_ascii=False, separators=(",", ":"))
        for text in encoder.iterencode(value):
            check_deadline(deadline)
            chunk = text.encode("utf-8")
            if len(payload) + len(chunk) > MAX_FRAME_BYTES:
                raise ValueError("rollout RPC frame exceeds limit")
            payload.extend(chunk)
        check_deadline(deadline)
        self._write(HEADER.pack(len(payload)), deadline)
        self._write(payload, deadline)

    def receive(self, deadline):
        (length,) = HEADER.unpack(self._read(HEADER.size, deadline))
        if not 0 < length <= MAX_FRAME_BYTES:
            raise ValueError("rollout RPC frame length outside limit")
        payload = self._read(length, deadline)
        value = json.loads(payload)
        check_deadline(deadline)
        _validate(value, deadline)
        check_deadline(deadline)
        return value

    def close(self):
        self.socket.close()
