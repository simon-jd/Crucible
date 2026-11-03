"""Warden's documented /v1/runs protocol. No assumed upload/cancel endpoint.

Needs an authenticated, explicitly configured gVisor service and imported image
containing the fixed bootstrap and readonly /tests/public.json. Current Warden
production prerequisites are absent; merely constructing this client proves none.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from rl.verifiers._warden_http import MAX_ENVELOPE_BYTES, MAX_REQUEST_BYTES
from rl.verifiers.code import encode
from rl.verifiers.types import RunResult


@dataclass(frozen=True)
class RunnerImage:
    """Trusted deployment metadata, checked against this exact public bundle.

    Operator must import/verify this immutable image in Warden. These fields are
    configuration, not measured proof of a runtime or image's contents.
    """

    image_digest: str
    public_tests_sha256: str
    bootstrap_sha256: str
    protocol: str = "crucible-solve-json-v1"


class HTTPWorkerCleanupError(RuntimeError):
    """Owned local worker has not yet been confirmed reaped; new I/O is blocked."""

    def __init__(self, pid):
        self.pid = pid
        super().__init__(f"local_http_cleanup_unknown:pid={pid}")


class WardenSandbox:
    def __init__(
        self,
        base_url: str,
        bearer_token: str,
        image: RunnerImage | None,
        *,
        transport=None,
        poll_interval_s: float = 0.05,
    ):
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
            or parsed.username
        ):
            raise ValueError("explicit_https_service_origin_required")
        if not bearer_token or any(c in bearer_token for c in "\r\n"):
            raise ValueError("bearer_token_required")
        if not 0 <= poll_interval_s <= 1:
            raise ValueError("invalid_poll_interval")
        self.base_url = base_url.rstrip("/")
        self._token = bearer_token
        self.image = image
        self.transport = transport or self._request
        self.poll_interval_s = poll_interval_s
        # Keep exact payload/idempotency identity even when POST's response is lost.
        # Caller must persist this map before process termination if retry is needed.
        self.pending = {}
        self._http_lock = threading.Lock()
        self._http_worker = None

    def _request(self, method, path, payload, key, timeout):
        deadline = time.monotonic() + timeout
        # Reserve part of this same budget to kill/reap a stalled local request.
        io_deadline = deadline - min(0.2, timeout / 4)
        if not self._http_lock.acquire(blocking=False):
            raise RuntimeError("http_request_already_active")
        try:
            if self._http_worker is not None:
                if self._http_worker.poll() is None:
                    raise HTTPWorkerCleanupError(self._http_worker.pid)
                self._close_http_worker()
            headers = {"Authorization": "Bearer " + self._token, "Content-Type": "application/json"}
            if key:
                headers["Idempotency-Key"] = key
            request = json.dumps(
                {
                    "url": self.base_url + path,
                    "method": method,
                    "payload": payload,
                    "headers": headers,
                    "deadline": io_deadline,
                },
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
            if len(request) > MAX_REQUEST_BYTES:
                raise ValueError("request_too_large")
            if time.monotonic() >= io_deadline:
                raise TimeoutError
            self._http_worker = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    str(Path(__file__).with_name("_warden_http.py").resolve()),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            )
            try:
                raw, _ = self._http_worker.communicate(
                    request, timeout=max(0, io_deadline - time.monotonic())
                )
                if self._http_worker.returncode != 0 or len(raw) > MAX_ENVELOPE_BYTES:
                    raise ValueError("http_worker_failed")
                result = json.loads(raw)
                if time.monotonic() >= deadline or result.get("error") == "timeout":
                    raise TimeoutError
                if (
                    "error" in result
                    or type(result.get("status")) is not int
                    or "body" not in result
                ):
                    raise ValueError("http_exchange_failed")
                return result["status"], result["body"]
            except subprocess.TimeoutExpired as error:
                raise TimeoutError from error
            finally:
                worker = self._http_worker
                if worker.poll() is None:
                    try:
                        os.killpg(worker.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except OSError:
                        if worker.poll() is None:
                            raise HTTPWorkerCleanupError(worker.pid) from None
                    try:
                        worker.communicate(timeout=max(0, deadline - time.monotonic()))
                    except (OSError, subprocess.TimeoutExpired):
                        if worker.poll() is None:
                            # Keep ownership and block subsequent network launches.
                            raise HTTPWorkerCleanupError(worker.pid) from None
                self._close_http_worker()
        finally:
            self._http_lock.release()

    def _close_http_worker(self):
        """Called only after poll/communicate has confirmed and reaped exit."""
        worker = self._http_worker
        for stream in (worker.stdin, worker.stdout):
            if stream is not None:
                stream.close()
        self._http_worker = None

    def run(self, solution, case_input, public_tests, timeout_s):
        bootstrap = Path(__file__).with_name("_candidate.py").read_bytes()
        image = self.image
        if (
            image is None
            or image.protocol != "crucible-solve-json-v1"
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", image.image_digest)
            or image.public_tests_sha256 != hashlib.sha256(public_tests.encode()).hexdigest()
            or image.bootstrap_sha256 != hashlib.sha256(bootstrap).hexdigest()
        ):
            return RunResult(
                error="warden_runner_image_prerequisites_missing", cleanup_confirmed=True
            )
        if not 0 < timeout_s <= 300:
            return RunResult(error="invalid_timeout", cleanup_confirmed=True)
        request = base64.b64encode(
            encode({"solution": solution, "input": case_input}).encode()
        ).decode()
        payload = {
            "image_digest": image.image_digest,
            "cmd": ["/usr/bin/python3", "-I", "-S", "/verifier/_candidate.py", request],
            "runtime": "gvisor",
            "network": "none",
            "limits": {
                "cpu": 1,
                "memory_mib": 512,
                "pids": 128,
                "timeout_s": max(1, math.ceil(timeout_s)),
            },
        }
        key = uuid.uuid4().hex
        self.pending[key] = {"payload": payload, "run_id": None}
        return self.retry(key, timeout_s)

    def retry(self, key: str, timeout_s: float = 10) -> RunResult:
        """Resume the exact submission; a client timeout NEVER cancels the run."""
        if key not in self.pending or not 0 < timeout_s <= 300:
            return RunResult(error="invalid_retry", idempotency_key=key)
        pending = self.pending[key]
        deadline = time.monotonic() + timeout_s
        run_id = pending["run_id"]
        definitive = False

        def call(method, path, payload=None, idempotency=None):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            status, body = self.transport(method, path, payload, idempotency, remaining)
            if status != (201 if method == "POST" else 200) or not isinstance(body, dict):
                raise ValueError("unexpected_response")
            return body

        try:
            if run_id is None:
                created = call("POST", "/v1/runs", pending["payload"], key)
                run_id = created.get("run_id")
                if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id):
                    raise ValueError("invalid_run_id")
                pending["run_id"] = run_id
            while True:
                state = call("GET", "/v1/runs/" + run_id)
                if state.get("run_id") != run_id:
                    raise ValueError("run_id_mismatch")
                if state.get("state") in ("succeeded", "failed", "interrupted"):
                    break
                if state.get("state") not in ("queued", "starting", "running", "pending"):
                    raise ValueError("unknown_state")
                time.sleep(min(self.poll_interval_s, max(0, deadline - time.monotonic())))
            result = state.get("result")
            if not isinstance(result, dict):
                raise TypeError("missing_runtime_result")
            cleaned = state.get("cleanup_pending") is False and result.get("CgroupEmpty") is True
            common = {"run_id": run_id, "idempotency_key": key, "cleanup_confirmed": False}
            if not cleaned:
                return RunResult(error="remote_cleanup_pending", **common)
            # Authenticated server result, never candidate-emitted markers.
            required_bool = ("OOMKilled", "TimedOut", "CgroupEmpty", "LogsTruncated")
            required_int = ("Code", "DroppedLogBytes", "PidsMaxEvents", "PeakPids")
            if any(type(result.get(field)) is not bool for field in required_bool) or any(
                type(result.get(field)) is not int for field in required_int
            ):
                return RunResult(error="invalid_runtime_result", **common)
            if (
                result["DroppedLogBytes"] < 0
                or result["PidsMaxEvents"] < 0
                or not 0 <= result["PeakPids"] <= 128
            ):
                return RunResult(error="invalid_runtime_result", **common)
            expected_spec = pending["payload"]
            spec = state.get("spec")
            if not isinstance(spec, dict) or any(
                spec.get(k) != v for k, v in expected_spec.items()
            ):
                return RunResult(error="runtime_spec_mismatch", **common)
            if (
                spec.get("rootfs_readonly") is not True
                or spec.get("uid") != 65534
                or spec.get("no_new_privileges") is not True
            ):
                return RunResult(error="unsafe_runtime_spec", **common)
            # Only this fully accepted terminal/spec/result/cleanup combination
            # makes remote work definitive. Every subsequent outcome must evict.
            definitive = True
            common["cleanup_confirmed"] = True
            if result["TimedOut"]:
                return RunResult(timed_out=True, **common)
            if (
                state["state"] != "succeeded"
                or state.get("error_code") is not None
                or result["Code"] != 0
                or result["OOMKilled"]
                or result["PidsMaxEvents"]
            ):
                return RunResult(
                    error="remote_execution_failed", exit_code=result["Code"], **common
                )
            logs = call("GET", "/v1/runs/" + run_id + "/logs")
            chunks = logs.get("logs")
            if (
                logs.get("status") != "complete"
                or logs.get("truncated") is not False
                or type(logs.get("dropped_bytes")) is not int
                or logs["dropped_bytes"] != 0
                or result["LogsTruncated"]
                or result["DroppedLogBytes"] != 0
                or not isinstance(chunks, list)
                or any(not isinstance(chunk, str) for chunk in chunks)
            ):
                return RunResult(error="incomplete_logs", **common)
            joined = "".join(chunks)
            if len(joined) > 2 * 1024**2:
                return RunResult(error="logs_limit", **common)
            if logs.get("encoding") == "base64":
                joined = base64.b64decode(joined, validate=True).decode("utf-8", errors="strict")
            elif logs.get("encoding", "utf-8") != "utf-8":
                return RunResult(error="unknown_log_encoding", **common)
            if len(joined.encode()) > 65536:
                return RunResult(error="answer_limit", output_truncated=True, **common)
            return RunResult(stdout=joined, exit_code=0, completed=True, **common)
        except HTTPWorkerCleanupError as error:
            return RunResult(
                error=str(error), cleanup_confirmed=definitive, run_id=run_id, idempotency_key=key
            )
        except TimeoutError:
            return RunResult(
                timed_out=True,
                cleanup_confirmed=definitive,
                error="remote_logs_timeout" if definitive else "remote_status_unknown",
                run_id=run_id,
                idempotency_key=key,
            )
        except Exception:  # noqa: BLE001 - preserve uncertain submission on any transport failure
            # Before accepted terminal cleanup, errors retain the exact retry identity.
            # After it, cleanup remains known and finally evicts the payload.
            return RunResult(
                error="warden_protocol_or_transport_error",
                cleanup_confirmed=definitive,
                run_id=run_id,
                idempotency_key=key,
            )
        finally:
            if definitive:
                self.pending.pop(key, None)
