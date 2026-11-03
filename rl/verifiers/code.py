"""Host-side trusted grader for pure Python solve(JSON) -> JSON tasks."""

from __future__ import annotations

import json
import time

from rl.rewards.hacks import detect_hacks
from rl.verifiers.types import Case, RunResult, Sandbox, TestBundle, Verdict

__all__ = ["Case", "RunResult", "Sandbox", "TestBundle", "Verdict", "verify_code"]
MAX_JSON_BYTES = 65536


def json_value(value, depth=0):
    """Bound the JSON tree and disallow bool/int confusion, NaN and huge values."""
    if depth > 16:
        raise ValueError("json_depth")
    if value is None or type(value) in (bool, str):
        return
    if type(value) is int:
        if value.bit_length() > 256:
            raise ValueError("integer_limit")
        return
    if type(value) is float:
        import math

        if not math.isfinite(value):
            raise ValueError("nonfinite")
        return
    if type(value) in (list, dict):
        if len(value) > 1024:
            raise ValueError("collection_limit")
        if isinstance(value, dict) and any(type(key) is not str for key in value):
            raise ValueError("nonstring_key")
        for child in value.values() if isinstance(value, dict) else value:
            json_value(child, depth + 1)
        return
    raise ValueError("not_json")


def encode(value):
    json_value(value)
    result = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    if len(result.encode()) > MAX_JSON_BYTES:
        raise ValueError("json_size")
    return result


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


def verify_code(solution: str, tests: TestBundle, sandbox: Sandbox, timeout_s: int = 10) -> Verdict:
    """Deadline covers the entire bundle, including queueing/startup/cleanup.

    Each input is run in a fresh process. JSON output is compared to host-only
    expectations; it cannot assert success or replace the trusted grader.
    """
    if type(timeout_s) is not int or not 1 <= timeout_s <= 300:
        return Verdict(False, "invalid_timeout")
    if not isinstance(solution, str) or not solution or len(solution) > 65536:
        return Verdict(False, "invalid_solution")
    try:
        if len(solution.encode()) > 65536:
            return Verdict(False, "invalid_solution")
    except UnicodeError:
        return Verdict(False, "invalid_solution")
    flags = detect_hacks(solution, "")
    if flags:
        return Verdict(False, "hack:" + ",".join(flags))
    try:
        if not isinstance(tests, TestBundle) or not tests.public or not tests.hidden:
            raise ValueError("public_and_hidden_required")
        cases = (*tests.public, *tests.hidden)
        if len(cases) > 64:
            raise ValueError("case_limit")
        for case in cases:
            encode(case.input)
            encode(case.expected)
        public = encode([{"input": case.input, "expected": case.expected} for case in tests.public])
    except (ValueError, TypeError, AttributeError, RecursionError):
        return Verdict(False, "invalid_test_bundle")
    deadline = time.monotonic() + timeout_s
    for index, case in enumerate(cases):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return Verdict(False, "timeout")
        try:
            result = sandbox.run(solution, case.input, public, remaining)
        except Exception:  # noqa: BLE001 - fail closed across trusted adapter failures
            # Implementations must clean their resources before raising. Exceptions
            # prove neither completion nor cleanup; never turn them into success.
            return Verdict(False, "sandbox_exception:cleanup_unknown")
        if (
            not isinstance(result, RunResult)
            or any(
                type(getattr(result, name)) is not bool
                for name in ("completed", "cleanup_confirmed", "timed_out", "output_truncated")
            )
            or any(
                not isinstance(getattr(result, name), str) for name in ("stdout", "stderr", "error")
            )
        ):
            return Verdict(False, "invalid_sandbox_result:cleanup_unknown")
        identity = (
            f":run_id={result.run_id}:idempotency_key={result.idempotency_key}"
            if result.run_id or result.idempotency_key
            else ""
        )
        if result.timed_out or time.monotonic() > deadline:
            return Verdict(
                False,
                "timeout" + (":cleanup_unknown" if not result.cleanup_confirmed else "") + identity,
            )
        if not result.cleanup_confirmed:
            return Verdict(False, "cleanup_unknown" + identity)
        if result.error or not result.completed:
            return Verdict(False, "sandbox_failed:" + result.error + identity)
        if type(result.exit_code) is not int or result.exit_code != 0 or result.output_truncated:
            return Verdict(False, "execution_failed_or_truncated" + identity)
        flags = detect_hacks(solution, result.stderr)
        if flags:
            return Verdict(False, "hack:" + ",".join(flags))
        try:
            if len(result.stdout.encode()) > MAX_JSON_BYTES:
                raise ValueError("output_limit")
            answer = json.loads(result.stdout, object_pairs_hook=_no_duplicates)
            if encode(answer) != encode(case.expected):
                # Do not echo hidden expected answers, inputs, or logs to generation.
                return Verdict(False, f"wrong_answer:case={index}")
        except (ValueError, TypeError, RecursionError):
            return Verdict(False, f"invalid_answer:case={index}")
    return Verdict(True, "all_public_and_hidden_cases_passed")
