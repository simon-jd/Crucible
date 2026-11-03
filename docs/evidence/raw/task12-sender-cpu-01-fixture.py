"""Proposed one-shot CPU sender screen. Importing this file never starts a service.

Only main() may run the frozen 59-second replay. Focused tests use explicit tiny
CPU stubs. No native/model imports; fixture timings are not serving metrics.
"""
# ruff: noqa: BLE001
# Broad catches are lifecycle boundaries: every exception invalidates and is
# retained in bounded evidence, while remaining exact-owned cleanup continues.

import csv
import errno
import hashlib
import io
import json
import math
import os
import queue
import select
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BINARY = ".cache/task12-baseline/bin/replay"
TRACE = "serving/baseline/traces/burst-calibration.csv"
TOKENS = "docs/evidence/raw/task12-model-token-allowlist.json"
PINNED = {
    BINARY: "68bb43b4aee08ca371d05a75b467a1f163c1fe8e959ccb534811740b63dffa93",
    TRACE: "28c6162d5a4e894a60e92c6626fc04534f02e0b1e8472dff88e9e107f3c89364",
    TOKENS: "8e8c8418ca48ed5bb0fe5b810ca1dcf841a56fbe0fd5333ed4babae63926eaa4",
}
CHILD_CAP = 65536
RSS_CAP = 512_000_000
BODY_CAP = 16384
HEADER_CAP = 8192
PRLIMIT = "/usr/bin/prlimit"
GO_ENV = ("GODEBUG", "GOGC", "GOMEMLIMIT", "GOMAXPROCS", "GOTRACEBACK")


def encoded(value):
    return (json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def error_text(error):
    return repr(error).encode("utf-8", "replace")[:2048].decode("utf-8", "replace")


class BoundedFile:
    def __init__(self, path, cap):
        self.file = path.open("xb")
        self.cap = cap
        self.size = 0
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            if self.size + len(data) > self.cap:
                raise ValueError(f"evidence cap: {Path(self.file.name).name}")
            self.file.write(data)
            self.file.flush()
            self.size += len(data)

    def close(self):
        self.file.close()


def save(path, value, cap=65536):
    data = encoded(value)
    if len(data) > cap:
        raise ValueError(f"metadata cap: {path.name}")
    with path.open("xb") as out:
        out.write(data)


def prepare(trace_data, token_ids):
    """Static exact Go prepare reconstruction; reject ambiguous identities."""
    state = 17
    expected = {}
    rows = list(csv.DictReader(io.StringIO(trace_data.decode("ascii"))))
    if len(rows) != 23:
        raise ValueError("expected exactly 23 frozen Burst rows")
    if len(token_ids) != 4096 or any(type(x) is not int or x < 0 for x in token_ids):
        raise ValueError("invalid allowlist")
    for index, row in enumerate(rows):
        inputs, outputs = int(row["Request tokens"]), int(row["Response tokens"])
        if not (0 <= inputs <= 32768 and 0 <= outputs <= 294):
            raise ValueError("unexpected frozen token bounds")
        ids = []
        for _ in range(inputs):
            state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            ids.append(token_ids[(state >> 32) % len(token_ids)])
        body = json.dumps(
            {
                "input_ids": ids,
                "stream": True,
                "sampling_params": {
                    "ignore_eos": True,
                    "max_new_tokens": outputs,
                    "temperature": 0,
                },
            },
            separators=(",", ":"),
        ).encode()
        key = digest(body)
        if len(body) > BODY_CAP or key in expected:
            raise ValueError("oversized or ambiguous request body")
        expected[key] = {
            "index": index,
            "bytes": len(body),
            "input": inputs,
            "output": outputs,
            "body": body,
            "scheduled_ms": (float(row["Timestamp"]) - 553788) * 1000,
        }
    return expected


class OwnedSocketClosed(Exception):
    """Only a socket operation interrupted by this fixture's cleanup."""


def read_request(conn, pending, *, completed, recv=None):
    """Bounded HTTP/1 framing. Idle EOF/timeout is ordinary only after response."""
    recv = recv or conn.recv
    while b"\r\n\r\n" not in pending:
        if len(pending) >= HEADER_CAP:
            raise ValueError("request headers too large")
        try:
            part = recv(min(4096, HEADER_CAP - len(pending)))
        except TimeoutError:
            if completed and not pending:
                return None, b""
            raise
        if not part:
            if pending:
                raise ValueError("partial HTTP headers")
            return None, b""
        pending += part
    head, pending = pending.split(b"\r\n\r\n", 1)
    lines = head.split(b"\r\n")
    if lines[0] != b"POST /generate HTTP/1.1":
        raise ValueError("unexpected request line")
    headers = {}
    for line in lines[1:]:
        name, sep, value = line.partition(b":")
        name = name.lower()
        tokens = b"!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyz"
        if (
            not sep
            or not name
            or name in headers
            or any(c not in tokens for c in name)
            or any(c < 32 and c != 9 or c == 127 for c in value)
        ):
            raise ValueError("malformed/duplicate header")
        headers[name] = value.strip()
    length = headers.get(b"content-length", b"")
    if b"transfer-encoding" in headers or not length.isdigit():
        raise ValueError("missing/unsupported HTTP framing")
    count = int(length)
    if not 0 < count <= BODY_CAP:
        raise ValueError("request body bound")
    if headers.get(b"content-type") != b"application/json":
        raise ValueError("unexpected content type")
    while len(pending) < count:
        part = recv(min(4096, count - len(pending)))
        if not part:
            raise ValueError("partial HTTP body")
        pending += part
    return pending[:count], pending[count:]


class Fixture:
    def __init__(self, expected, events, port):
        self.expected, self.events = expected, events
        self.seen = set()
        self.errors = queue.Queue(maxsize=32)
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.connections, self.threads = [], []
        self.listener = socket.socket()
        try:
            self.listener.bind(("127.0.0.1", port))
            self.listener.listen(32)
            self.listener.setblocking(False)
        except BaseException:
            self.listener.close()
            raise
        self.port = self.listener.getsockname()[1]

    def accept_one(self):
        conn, _address = self.listener.accept()
        if len(self.connections) >= 32:
            conn.close()
            raise ValueError("connection/worker cap")
        conn.settimeout(2)
        self.connections.append(conn)
        thread = threading.Thread(
            target=self.handle, args=(conn, len(self.connections)), daemon=True
        )
        self.threads.append(thread)
        thread.start()

    def handle(self, conn, connection_id):
        pending, completed = b"", False
        try:
            self.events.write(
                encoded(
                    {
                        "event": "accept",
                        "connection": connection_id,
                        "monotonic_ns": time.monotonic_ns(),
                    }
                )
            )
            while not self.stopping.is_set():
                body, pending = read_request(
                    conn,
                    pending,
                    completed=completed,
                    recv=lambda n: self.socket_io(conn.recv, n),
                )
                if body is None:
                    return
                key = digest(body)
                with self.lock:
                    if key not in self.expected or key in self.seen:
                        raise ValueError("unknown or duplicate request body")
                    if len(self.seen) >= 23:
                        raise ValueError("request cap")
                    request = self.expected[key]
                    if body != request["body"]:
                        raise ValueError("body identity mismatch")
                    self.seen.add(key)
                # Exact known JSON bytes were generated and validated before listen.
                self.events.write(
                    encoded(
                        {
                            "event": "body_complete",
                            "connection": connection_id,
                            "index": request["index"],
                            "sha256": key,
                            "bytes": len(body),
                            "monotonic_ns": time.monotonic_ns(),
                        }
                    )
                )
                count = request["output"]
                response = (
                    b"data: "
                    + encoded(
                        {
                            "output_ids": [0] * count,
                            "meta_info": {
                                "completion_tokens": count,
                                "prompt_tokens": request["input"],
                                "finish_reason": {"type": "length"},
                            },
                        }
                    ).rstrip(b"\n")
                    + b"\n\ndata: [DONE]\n\n"
                )
                header = (
                    f"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    f"Content-Length: {len(response)}\r\n\r\n"
                ).encode()
                self.socket_io(conn.sendall, header + response)
                completed = True
                self.events.write(
                    encoded(
                        {
                            "event": "response_complete",
                            "connection": connection_id,
                            "index": request["index"],
                            "bytes": len(response),
                            "monotonic_ns": time.monotonic_ns(),
                        }
                    )
                )
        except OwnedSocketClosed:
            pass  # Defined only by socket_io; never wraps evidence/file operations.
        except Exception as exc:
            self.errors.put_nowait({"connection": connection_id, "error": error_text(exc)})
        finally:
            conn.close()

    def socket_io(self, operation, *args):
        try:
            result = operation(*args)
        except OSError as exc:
            if self.stopping.is_set() and exc.errno in (
                errno.EBADF,
                errno.ECONNRESET,
                errno.ENOTCONN,
                errno.EPIPE,
            ):
                raise OwnedSocketClosed() from exc
            raise
        if result == b"" and self.stopping.is_set():
            raise OwnedSocketClosed()
        return result

    def close(self):
        self.stopping.set()
        self.listener.close()
        errors = []
        for conn in self.connections:
            if conn.fileno() >= 0:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError as exc:
                    if exc.errno not in (errno.ENOTCONN, errno.EBADF):
                        errors.append(error_text(exc))
                conn.close()
        deadline = time.monotonic() + 3
        for thread in self.threads:
            thread.join(max(0, deadline - time.monotonic()))
        alive = sum(t.is_alive() for t in self.threads)
        while not self.errors.empty():
            errors.append(self.errors.get_nowait())
        # Read-only LISTEN state proof, avoiding TIME_WAIT's ambiguous bind error.
        try:
            with Path("/proc/net/tcp").open("rb") as inp:
                tcp = inp.read((1 << 20) + 1)
            if len(tcp) > 1 << 20:
                raise ValueError("TCP table read bound")
            address = f"0100007F:{self.port:04X}".encode()
            port_closed = not any(
                row.split()[1] == address and row.split()[3] == b"0A"
                for row in tcp.splitlines()[1:]
            )
        except Exception as exc:
            port_closed = False
            errors.append(error_text(exc))
        return {
            "threads_alive": alive,
            "connections": len(self.connections),
            "listener_closed": self.listener.fileno() == -1,
            "port_listener_absent": port_closed,
            "errors": errors,
        }


def process_sample(pid, expected_start=None):
    with Path(f"/proc/{pid}/stat").open("rb") as inp:
        data = inp.read(8193)
    if len(data) > 8192:
        raise ValueError("proc stat read bound")
    fields = data[data.rfind(b")") + 2 :].split()
    start, rss = int(fields[19]), int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
    if expected_start is not None and start != expected_start:
        raise ValueError(f"owned PID identity changed: {pid}")
    return {"pid": pid, "starttime": start, "rss": rss}


def spawn(command, stdout, stderr):
    # No preexec_fn. Main thread temporarily sets inherited signal disposition;
    # the trusted prlimit utility execs the one child with both hard limits set.
    if threading.active_count() != 1:
        raise ValueError("spawn requires no started fixture threads")
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise ValueError("sole-waiter Popen anchor requires default SIGCHLD")
    previous = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    try:
        return subprocess.Popen(
            [PRLIMIT, "--fsize=65536:65536", "--core=0:0", "--", *command],
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            restore_signals=False,
            close_fds=True,
        )
    finally:
        signal.signal(signal.SIGXFSZ, previous)


def child_caps(out):
    sizes = {name: (out / name).stat().st_size for name in ("stdout.json", "stderr.log")}
    if any(n >= CHILD_CAP for n in sizes.values()):
        raise ValueError(f"child output at hard cap: {sizes}")
    return sizes


def stop_child(child, identity):
    result = {"pid": child.pid, "starttime": identity, "termination_requested": False}
    code = child.poll()
    result["pre_cleanup_returncode"] = code
    if code is None:
        # Sole parent/waiter holds the original unreaped Popen. Its unreaped PID
        # cannot be reused; a failed diagnostic /proc read must not leak it.
        try:
            if identity is None:
                raise ValueError("initial starttime sample failed")
            process_sample(child.pid, identity)
        except Exception as exc:
            result["identity_error"] = error_text(exc)
        result["termination_requested"] = True
        child.terminate()
        try:
            code = child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            result["forced_kill"] = True
            child.kill()
            code = child.wait(timeout=3)
    result.update(returncode=code, reaped=child.returncode is not None)
    path = Path(f"/proc/{child.pid}")
    result["pid_absent"] = not path.exists()
    if path.exists():
        result["post_identity"] = process_sample(child.pid)
    return result


def validate_result(out, expected, returncode):
    child_caps(out)
    raw = json.loads((out / "stdout.json").read_bytes())
    run = raw["Run"]
    records = run["Records"]
    if run["Aborted"] is not False or len(records) != len(expected):
        raise ValueError("incomplete/aborted fixture run")
    ordered = sorted(expected.values(), key=lambda x: x["index"])
    late = []
    for index, (row, request) in enumerate(zip(records, ordered, strict=True)):
        if row["Status"] != "ok" or row["OutputTokens"] != request["output"]:
            raise ValueError("failed fixture request")
        if row["ScheduledMs"] != request["scheduled_ms"]:
            raise ValueError("changed arrival schedule")
        if not math.isfinite(row["SentMs"]) or row["SentMs"] < 0:
            raise ValueError("missing/nonfinite complete-write observation")
        lag = row["SentMs"] - row["ScheduledMs"]
        if lag > 5:
            late.append({"index": index, "lag_ms": lag})
    valid = len(late) * 100 <= len(records)
    stderr = (out / "stderr.log").read_bytes()
    if run["SenderValid"] is not valid:
        raise ValueError("sender flag mismatch")
    if (returncode, stderr) != (
        (0, b"") if valid else (1, b"sender-invalid run; no accepted performance\n")
    ):
        raise ValueError("unknown child exit/stderr")
    return {
        "sender_valid": valid,
        "late": late,
        "records": len(records),
        "native_performance_accepted": False,
        "coalesced_fixture": True,
    }


def screen(command, expected, out, *, port=31013, child_seconds=70, provenance=None):
    """Lifecycle seam; focused tests may supply ONLY an explicit tiny CPU stub."""
    start = time.monotonic()
    out.mkdir()  # fresh exact target, never reuse or delete outputs
    events = BoundedFile(out / "fixture.jsonl", 256 * 1024)
    samples = BoundedFile(out / "rss.jsonl", 256 * 1024)
    child = fixture = None
    identity = None
    failure = None
    cleanup = {}
    result = None
    stop_reason = None
    received_signal = []
    previous_handlers = {}
    stdout = stderr = None
    try:
        if provenance is not None:
            save(out / "provenance.json", provenance, 32768)
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(
                sig, lambda n, _f: received_signal.append(n) if len(received_signal) < 4 else None
            )
        own = process_sample(os.getpid())
        fixture = Fixture(expected, events, port)
        stdout, stderr = (out / "stdout.json").open("xb"), (out / "stderr.log").open("xb")
        child = spawn(command, stdout, stderr)
        identity = process_sample(child.pid)["starttime"]
        save(
            out / "launch.json",
            {
                "command": command,
                "pid": child.pid,
                "starttime": identity,
                "prlimit_sha256": digest(Path(PRLIMIT).read_bytes()),
                "hard_fsize": CHILD_CAP,
                "hard_core": 0,
                "inherited_sigxfsz": "SIG_IGN",
                "restore_signals": False,
                "fixture_port": fixture.port,
            },
            32768,
        )
        child_start, next_sample = time.monotonic(), 0.0
        while True:
            now = time.monotonic()
            if now >= next_sample:
                observed = [
                    process_sample(os.getpid(), own["starttime"]),
                    process_sample(child.pid, identity),
                ]
                sample = {
                    "elapsed": now - start,
                    "owned": observed,
                    "rss": sum(x["rss"] for x in observed),
                }
                samples.write(encoded(sample))  # retain full trigger before checks
                next_sample = now + 0.1
                if sample["rss"] >= RSS_CAP:
                    raise ValueError("owned aggregate RSS early stop")
            child_caps(out)
            if not fixture.errors.empty():
                raise ValueError(f"fixture failure: {fixture.errors.get_nowait()}")
            if received_signal:
                raise ValueError(f"controller signal: {received_signal[:4]}")
            if now - start >= 75 or now - child_start >= child_seconds:
                raise TimeoutError("CPU screen deadline")
            code = child.poll()
            if code is not None:
                if fixture.seen != set(expected):
                    raise ValueError("missing fixture requests")
                result = validate_result(out, expected, code)
                stop_reason = "child exited; original result retained"
                break
            ready, _, _ = select.select([fixture.listener], [], [], 0.02)
            if ready:
                fixture.accept_one()
    except Exception as exc:
        failure = error_text(exc)
        stop_reason = "diagnostic invalid"
    finally:
        # Reserved bounded stop evidence precedes exact child cleanup.
        try:
            save(
                out / "stop.json",
                {"reason": stop_reason, "error": failure, "elapsed": time.monotonic() - start},
            )
        except Exception as exc:
            failure = failure or error_text(exc)
            cleanup["stop_evidence_error"] = error_text(exc)
        if child is not None:
            try:
                cleanup["child"] = stop_child(child, identity)
                if cleanup["child"].get("identity_error"):
                    raise ValueError(cleanup["child"]["identity_error"])
            except Exception as exc:
                cleanup["child_error"] = error_text(exc)
                failure = failure or error_text(exc)
        if fixture is not None:
            try:
                cleanup["fixture"] = fixture.close()
                f = cleanup["fixture"]
                if f["threads_alive"] or f["errors"] or not f["port_listener_absent"]:
                    raise ValueError("fixture cleanup unproved")
            except Exception as exc:
                cleanup["fixture_error"] = error_text(exc)
                failure = failure or error_text(exc)
        for stream in (stdout, stderr, events, samples):
            if stream is not None:
                try:
                    stream.close()
                except Exception as exc:
                    cleanup.setdefault("file_close_errors", []).append(error_text(exc))
                    failure = failure or error_text(exc)
        if child is not None:
            try:
                child_caps(out)
                if not cleanup.get("child", {}).get("reaped"):
                    raise ValueError("child reap unproved")
            except Exception as exc:
                cleanup["final_child_error"] = error_text(exc)
                failure = failure or error_text(exc)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        terminal = {
            "screen_completed": failure is None,
            "failure": failure,
            "result": result if failure is None else None,
            "cleanup": cleanup,
            "elapsed": time.monotonic() - start,
            "native_performance_accepted": False,
        }
        save(out / "terminal.json", terminal)
    return terminal


def main():
    # No runtime/package imports, no model contents, and no env normalization.
    if any(os.environ.get(name) for name in GO_ENV):
        raise ValueError("unexpected Go runtime override; stop for review")
    data = {}
    for name, expected_hash in PINNED.items():
        path = ROOT / name
        if path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("input type/size bound")
        content = path.read_bytes()
        if digest(content) != expected_hash:
            raise ValueError(f"frozen input changed: {name}")
        data[name] = content
    expected = prepare(data[TRACE], json.loads(data[TOKENS])["token_ids"])
    if len(expected) != 23:
        raise ValueError("fixed request count")
    os.chdir(ROOT)
    command = [
        BINARY,
        "--trace",
        TRACE,
        "--format",
        "burstgpt",
        "--url",
        "http://127.0.0.1:31013/generate",
        "--token-ids",
        TOKENS,
        "--mult",
        "1",
        "--timeout",
        "60s",
        "--inflight",
        "192",
        "--seed",
        "17",
    ]
    out = ROOT / "docs/evidence/raw/task12-sender-cpu-01"
    # Identity receipt is in the newly-created output only; never overwrite raw.
    provenance = {
        "source_sha256": digest(Path(__file__).read_bytes()),
        "inputs": PINNED,
        "go_environment": {name: os.environ.get(name) for name in GO_ENV},
        "bodies": [
            {k: v for k, v in row.items() if k != "body"} | {"sha256": key}
            for key, row in expected.items()
        ],
        "fixture": "immediate single delta SSE; timing differs from native",
    }
    terminal = screen(command, expected, out, provenance=provenance)
    return 0 if terminal["screen_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
