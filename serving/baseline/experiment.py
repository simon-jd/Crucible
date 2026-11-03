"""Fixed calibration/evaluation protocol and owned native-server lifecycle."""

import argparse
import csv
import hashlib
import http.client
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from serving.baseline.model_snapshot import canonical_bytes, load_plan, verify_file
from serving.baseline.prepare_runtime import CAP, CLEANUP_RESERVE, sha256
from serving.baseline.runtime_entry import ROOT, inventory, runtime_environment
from serving.replay.tokenizer_allowlist import build as build_allowlist


def error_info(error):
    data = repr(error).encode("utf-8", errors="replace")
    return {
        "repr": data[:1024].decode("utf-8", errors="ignore"),
        "observed_utf8_bytes": len(data),
        "truncated": len(data) > 1024,
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def sender_valid(run):
    records = run["Records"]
    late = sum(
        not _finite(x["SentMs"])
        or not _finite(x["ScheduledMs"])
        or x["SentMs"] - x["ScheduledMs"] > 5
        for x in records
    )
    return (
        bool(records) and not run["Aborted"] and run["SenderValid"] and late * 100 <= len(records)
    )


def _defined(row):
    return row["OutputTokens"] >= 2 and _finite(row["TTFTms"]) and _finite(row["TPOTms"])


def calibrate(runs, identities, lineage=None):
    if len(runs) != 3 or len(set(identities)) != 3 or not all(map(sender_valid, runs)):
        raise ValueError("calibration requires three distinct sender-valid runs")
    rows = [r for run in runs for r in run["Records"]]
    valid = [r for r in rows if r["Status"] == "ok" and _defined(r)]
    if not valid:
        raise ValueError("no defined calibration samples")
    rank = math.ceil(0.99 * len(valid)) - 1
    slo = {metric: sorted(r[metric] for r in valid)[rank] for metric in ("TTFTms", "TPOTms")}
    slo.update(
        Samples=len(valid),
        identities=identities,
        quantile="pooled nearest-rank p99",
        planned=len(rows),
        failed=sum(r["Status"] != "ok" for r in rows),
        undefined=sum(not _defined(r) for r in rows),
        coalesced=sum(r["Coalesced"] for r in rows),
        lineage=lineage or {},
    )
    slo["sha256"] = _digest(slo)
    return slo


def evaluate(run, slo, identity, lineage=None):
    if (
        not identity
        or identity in slo["identities"]
        or slo["sha256"] != _digest({k: v for k, v in slo.items() if k != "sha256"})
        or slo["lineage"] != (lineage or {})
    ):
        raise ValueError("invalid evaluation/SLO lineage")
    rows = run["Records"]
    accepted = sender_valid(run) and _finite(run["DurationMs"]) and run["DurationMs"] > 0
    good = sum(
        r["Status"] == "ok"
        and _defined(r)
        and r["TTFTms"] <= slo["TTFTms"]
        and r["TPOTms"] <= slo["TPOTms"]
        for r in rows
    )
    return {
        "accepted": accepted,
        "goodput": good / (run["DurationMs"] / 1000) if accepted else None,
        "good": good if accepted else None,
        "planned": len(rows),
        "failed": sum(r["Status"] != "ok" for r in rows),
        "undefined": sum(not _defined(r) for r in rows),
        "coalesced": sum(r["Coalesced"] for r in rows),
        "slo_sha256": slo["sha256"],
        "duration_seconds": run["DurationMs"] / 1000,
        "denominator": "first selected scheduled request through final completion",
    }


def trial_order():
    trials = []
    for multiplier in (2, 4):
        for round_index in range(3):
            pair = [1, multiplier]
            if (round_index + (multiplier == 4)) % 2:
                pair.reverse()
            for value in pair:
                trials.append({"pair": f"{multiplier}x-{round_index + 1}", "multiplier": value})
    return trials


def matched_pairs(summaries, order):
    """Retain the fixed design, including invalid and never-run pair members."""
    slots = {}
    for summary in summaries:
        key = (summary["trial"]["pair"], summary["trial"]["multiplier"])
        if key in slots:
            raise ValueError("duplicate matched trial")
        slots[key] = summary
    pairs = {}
    for trial in order:
        key = (trial["pair"], trial["multiplier"])
        pairs.setdefault(trial["pair"], []).append(
            slots.get(key, {"trial": trial, "status": "not_run", "metrics": {"accepted": False}})
        )
    rows = []
    for name, members in pairs.items():
        accepted = len(members) == 2 and all(x["metrics"]["accepted"] for x in members)
        by_rate = {m["trial"]["multiplier"]: m for m in members}
        rate = max(by_rate)
        rows.append(
            {
                "pair": name,
                "multiplier": rate,
                "members": members,
                "accepted": accepted,
                "goodput_difference": by_rate[rate]["metrics"]["goodput"]
                - by_rate[1]["metrics"]["goodput"]
                if accepted
                else None,
            }
        )
    comparisons = {}
    for rate in (2, 4):
        subset = [r for r in rows if r["multiplier"] == rate]
        values = [r["goodput_difference"] for r in subset if r["accepted"]]
        accepted = len(subset) == len(values) == 3
        comparisons[f"{rate}x"] = {
            "accepted_all_three": accepted,
            "accepted_pairs": len(values),
            "planned_pairs": 3,
            "goodput_difference": {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
            if accepted
            else None,
        }
    return {"pairs": rows, "comparisons": comparisons}


def listener_closed(port):
    if port is None:
        return True
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return False
    except OSError:
        return True


def stop_child(child, port, grace=15):
    """Signal only this unreaped Popen child; descendant cleanup belongs to fixed guard."""
    forced = False
    before = child.poll()
    termination_requested = before is None
    if termination_requested:
        child.terminate()
        try:
            child.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            forced = True
            child.kill()
            child.wait(timeout=grace)
    deadline = time.monotonic() + grace
    while not listener_closed(port) and time.monotonic() < deadline:
        time.sleep(0.05)
    return {
        "pid": child.pid,
        "returncode": child.returncode,
        "returncode_before_cleanup": before,
        # Records the controller's request, not proof that TERM caused the exit.
        "termination_requested": termination_requested,
        "forced_kill": forced,
        "reaped": child.returncode is not None,
        "listener_closed": listener_closed(port),
        "descendant_confirmation": "required from outer fixed GPU guard before lock release",
    }


def _native_response(endpoint, body, data, evidence, inspect, seconds=30, limit=4 << 20):
    """One owned loopback POST; watchdog closes its exact socket at deadline."""
    parsed = urllib.parse.urlparse(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username
        or parsed.password
    ):
        raise ValueError("native smoke requires literal loopback HTTP")
    deadline = time.monotonic() + seconds
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=seconds)
    expired = threading.Event()
    timer = None
    try:
        connection.connect()
        owned_socket = connection.sock

        def expire():
            expired.set()
            try:
                owned_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        timer = threading.Timer(max(0, deadline - time.monotonic()), expire)
        timer.start()
        connection.request("POST", parsed.path, body, {"Content-Type": "application/json"})
        with connection.getresponse() as response:
            evidence["status"] = response.status
            evidence["http_body_complete"] = False
            evidence["http_framing"] = (
                "chunked"
                if response.chunked
                else "content-length"
                if response.length is not None
                else "connection-close"
            )
            while True:
                inspect()
                if time.monotonic() >= deadline:
                    raise TimeoutError("native response absolute deadline exceeded")
                block = response.read1(min(65536, limit + 1 - len(data)))
                if not block:
                    evidence["remaining_content_length"] = response.length
                    # read1() can return EOF with a positive fixed-length remainder.
                    # Chunked read1 verifies the terminator; close-delimited EOF is
                    # completion only when no fixed-length/chunk framing remains.
                    if response.length is not None and response.length > 0:
                        raise http.client.IncompleteRead(b"", response.length)
                    if response.chunked and response.chunk_left is not None:
                        raise http.client.IncompleteRead(b"")
                    evidence["http_body_complete"] = True
                    break
                data.extend(block)
                if len(data) > limit:
                    raise ValueError("native smoke response exceeded bound")
        if expired.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("native response absolute deadline exceeded")
    except (OSError, http.client.HTTPException):
        if expired.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("native response absolute deadline exceeded") from None
        raise
    finally:
        if timer is not None:
            timer.cancel()
            timer.join()
        connection.close()


def _native_cases(endpoint, cases, output, prefix, inspect=lambda: None):
    results = []
    for name, ids, count in cases:
        body = json.dumps(
            {
                "input_ids": ids,
                "stream": True,
                "sampling_params": {"max_new_tokens": count, "temperature": 0, "ignore_eos": True},
            }
        ).encode()
        if name == "error":
            body = b'{"input_ids":'  # Parser rejection; no invalid CUDA embedding.
        data = bytearray()
        evidence = {
            "case": name,
            "input_tokens": len(ids),
            "requested_output": count,
            "request_sha256": hashlib.sha256(body).hexdigest(),
            "status": None,
            "actual_output": None,
            "accepted_contract": False,
        }
        failure = None
        try:
            _native_response(endpoint, body, data, evidence, inspect)
            status = evidence["status"]
            total, finished, done = 0, False, False
            valid = 400 <= status < 500 and name in {"error", "zero"}
            if status == 200 and name != "error":
                for line in data.splitlines():
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if done:
                        raise ValueError("native data after DONE")
                    if payload == b"[DONE]":
                        done = True
                        continue
                    if finished:
                        raise ValueError("native data after finish")
                    event = json.loads(payload)
                    delta = event.get("output_ids", [])
                    if not isinstance(delta, list) or any(
                        type(x) is not int or not 0 <= x < 151936 for x in delta
                    ):
                        raise ValueError("native invalid token IDs")
                    total += len(delta)
                    meta = event.get("meta_info", {})
                    if (
                        type(meta.get("prompt_tokens")) is not int
                        or type(meta.get("completion_tokens")) is not int
                        or meta["prompt_tokens"] != len(ids)
                        or meta["completion_tokens"] != total
                    ):
                        raise ValueError("native prompt/delta token accounting failed")
                    reason = meta.get("finish_reason")
                    if reason is not None:
                        if not isinstance(reason, dict) or reason.get("type") not in {
                            "length",
                            "stop",
                        }:
                            raise ValueError("native invalid finish reason")
                        finished = True
                valid = total == count and finished and done
            evidence.update(actual_output=total, accepted_contract=valid)
            if not valid:
                raise ValueError(f"native contract failed: {name}")
        except BaseException as error:  # noqa: BLE001 - preserve primary failure through bounded cleanup/reporting
            failure = error
        finally:
            evidence.update(
                raw_sha256=hashlib.sha256(data).hexdigest(),
                raw_bytes=len(data),
                observed_count_is_lower_bound=not evidence.get("http_body_complete", False),
            )
            if failure is not None:
                evidence["error"] = error_info(failure)["repr"]
                evidence["error_info"] = error_info(failure)
            try:
                (output / f"{prefix}-{name}.raw").write_bytes(data)
            except BaseException as write_error:  # noqa: BLE001 - preserve primary failure through bounded cleanup/reporting
                failure = failure or write_error
                evidence["accepted_contract"] = False
                evidence["write_error"] = error_info(write_error)
            results.append(evidence)
            try:
                write_json(output / f"{prefix}.json", results)
            except BaseException as report_error:  # noqa: BLE001 - preserve primary failure through bounded cleanup/reporting
                if failure is None:
                    failure = report_error
        if failure is not None:
            raise failure
    return results


def native_smoke(endpoint, token_ids, vocab_size, output, inspect=lambda: None):
    return _native_cases(
        endpoint,
        [
            ("zero", token_ids[:8], 0),
            ("one", token_ids[:8], 1),
            ("multi", token_ids[:8], 4),
            ("error", [vocab_size], 1),
        ],
        output,
        "native-smoke",
        inspect,
    )


def fixed_warmups(endpoint, token_ids, specs, output, inspect=lambda: None):
    if not token_ids:
        raise ValueError("warmup requires verified token allowlist")
    cases = [
        (
            str(i),
            [token_ids[j % len(token_ids)] for j in range(spec["input_tokens"])],
            spec["output_tokens"],
        )
        for i, spec in enumerate(specs)
    ]
    return _native_cases(endpoint, cases, output, "warmup", inspect)


def _owned_rss():
    rows = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            rows[int(directory.name)] = (
                int(fields[1]),
                int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
            )
        except (OSError, ValueError, IndexError):
            continue
    owned = {os.getpid()}
    while True:
        found = {pid for pid, (ppid, _) in rows.items() if ppid in owned}
        if found <= owned:
            break
        owned |= found
    return sum(rows[pid][1] for pid in owned if pid in rows)


def check_resources(work, baseline_bytes, output=None, canonical=True):
    rescans = []
    runtime_bytes = canonical_bytes(work, rescans)
    evidence_bytes = canonical_bytes(output, rescans) if output is not None else 0
    # Full canonical scans happen between trials. During timed runs, scan only
    # the explicitly owned mutable roots and reserve50MB for other project growth.
    current = (
        canonical_bytes(ROOT, rescans)
        if canonical
        else baseline_bytes + runtime_bytes + evidence_bytes + 50_000_000
    )
    rss = _owned_rss()
    sample = {
        "monotonic": time.monotonic(),
        "project_bytes": current,
        "project_bytes_kind": "canonical" if canonical else "conservative mutable-root estimate",
        "runtime_cache_bytes": runtime_bytes,
        "evidence_bytes": evidence_bytes,
        "owned_rss_bytes": rss,
        "canonical_rescans": rescans,
    }
    if (
        current >= CAP - CLEANUP_RESERVE
        or runtime_bytes >= 400_000_000
        or runtime_bytes + evidence_bytes >= 450_000_000
        or rss >= 16_000_000_000
    ):
        raise RuntimeError("resource high-water stop: " + json.dumps(sample))
    return sample


def _wait(process, seconds, inspect):
    deadline = time.monotonic() + seconds
    while process.poll() is None:
        inspect()
        if time.monotonic() >= deadline:
            raise TimeoutError("owned subprocess deadline exceeded")
        time.sleep(1)
    return process.returncode


def verify_model_inputs(plan, manifest, model, tokens):
    if (
        manifest["status"] != "verified"
        or manifest["repo"] != plan["repo"]
        or manifest["revision"] != plan["revision"]
        or model.name != plan["revision"]
    ):
        raise ValueError("model manifest identity mismatch")
    rows = {row["name"]: row for row in manifest["files"]}
    if len(rows) != len(manifest["files"]) or set(rows) != {x["name"] for x in plan["files"]}:
        raise ValueError("model manifest file set mismatch")
    for item in plan["files"]:
        pointer = model / item["name"]
        blob = model.parent.parent / "blobs" / item["blob"]
        row = rows[item["name"]]
        if (
            not pointer.is_symlink()
            or pointer.resolve() != blob
            or row["blob_path"] != str(blob)
            or row["snapshot_path"] != str(pointer)
        ):
            raise ValueError("model manifest snapshot pointer mismatch")
        actual = verify_file(blob, item)
        if any(
            row[key] != actual[key] for key in ("bytes", "sha256", "algorithm", "upstream_hash")
        ):
            raise ValueError("model manifest content mismatch")
    if tokens != build_allowlist(model, plan["repo"], plan["revision"]):
        raise ValueError("token allowlist differs from actual verified model metadata")


def execute_trial(command, env, target, summary, inspect):
    """Direct output files; incomplete or failed runs are marked aborted."""
    summary.update(status="running", metrics={"accepted": False, "goodput": None})
    record = target.with_suffix(".summary.json")
    write_json(record, summary)
    child, failure = None, None
    try:
        with target.open("wb") as out, target.with_suffix(".stderr").open("wb") as err:
            child = subprocess.Popen(
                command, env=env, stdout=out, stderr=err
            )
            summary["pid"] = child.pid
            summary["returncode"] = _wait(child, 130, inspect)
        payload = target.read_bytes()
        result = json.loads(payload)["Run"]
        complete = len(result["Records"]) == summary["planned_rows"]
        failed = sum(row["Status"] != "ok" for row in result["Records"])
        with target.with_suffix(".stderr").open("rb") as source:
            stderr = source.read(4097)
        request_failure_exit = (
            summary["returncode"] == 1
            and stderr == b"run contains failed requests\n"
            and complete
            and result["Aborted"] is False
            and result["SenderValid"] is True
            and sender_valid(result)
            and failed > 0
        )
        if not complete or (summary["returncode"] != 0 and not request_failure_exit):
            result["Aborted"] = True
        summary.update(
            status="completed",
            actual_records=len(result["Records"]),
            actual_output_tokens=sum(r["OutputTokens"] for r in result["Records"]),
            failed_requests=failed,
            complete_request_failure_exit=request_failure_exit,
            effective_aborted=result["Aborted"],
        )
        return result
    except BaseException as error:
        failure = error
        summary.update(
            status="failed",
            error=error_info(error)["repr"],
            error_info=error_info(error),
            effective_aborted=True,
            metrics={"accepted": False, "goodput": None},
        )
        # Trigger/cause precedes exact child cleanup; optional evidence failure
        # must not replace the primary failure or bypass cleanup.
        try:
            write_json(record, summary)
        except BaseException as report_error:  # noqa: BLE001 - preserve primary failure through bounded cleanup/reporting
            summary["trigger_write_error"] = error_info(report_error)
        raise
    finally:
        try:
            if child is not None:
                summary["cleanup"] = stop_child(child, None)
                summary["returncode"] = child.returncode
        except BaseException as cleanup_error:  # noqa: BLE001 - preserve primary failure through bounded cleanup/reporting
            summary["cleanup_error"] = error_info(cleanup_error)
            if failure is None:
                failure = cleanup_error
        try:
            for key, path in (("raw", target), ("stderr", target.with_suffix(".stderr"))):
                if path.exists():
                    summary[key + "_sha256"] = sha256(path)
                    summary[key + "_bytes"] = path.stat().st_size
        except BaseException as metadata_error:  # noqa: BLE001 - preserve primary failure through bounded cleanup/reporting
            summary["metadata_error"] = error_info(metadata_error)
            failure = failure or metadata_error
        try:
            write_json(record, summary)
        except BaseException:
            if failure is None:
                raise
        if failure is not None and summary["status"] != "failed":
            raise failure


TRIAL_BOUNDARY = {
    "endpoint": "POST /flush_cache",
    "deadline_seconds": 30,
    "success_status": 200,
    "retry_status": 400,
    "placement": "after fixed warmups and every attempted calibration/evaluation trial",
    "measurement": "outside selected-request interval",
    "effect": "require scheduler idle, then reset cache and memory pools",
}


def flush_barrier(endpoint, target, inspect, seconds=30):
    began = time.monotonic()
    deadline = began + seconds
    report = {"accepted": False, "attempts": [], "deadline_seconds": seconds}
    total = 0
    failure = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("server idle barrier deadline exceeded")
            evidence = {"status": None, "started_seconds": time.monotonic() - began}
            report["attempts"].append(evidence)
            data = bytearray()
            try:
                _native_response(
                    endpoint + "/flush_cache",
                    b"{}",
                    data,
                    evidence,
                    inspect,
                    seconds=remaining,
                    limit=65536,
                )
            finally:
                evidence.update(
                    bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                    elapsed_seconds=time.monotonic() - began,
                )
            total += len(data)
            if total > 1 << 20:
                raise ValueError("idle barrier cumulative response bound exceeded")
            if evidence["status"] == 200:
                report["accepted"] = True
                return report
            if evidence["status"] != 400:
                raise ValueError("unexpected idle barrier HTTP status")
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    except BaseException as error:
        failure = error
        report["error"] = error_info(error)["repr"]
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - began
        try:
            write_json(target, report)
        except BaseException:
            if failure is None:
                raise


def execute_isolated_trial(command, env, target, summary, inspect, endpoint):
    failure = None
    try:
        return execute_trial(command, env, target, summary, inspect)
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            summary["boundary"] = flush_barrier(
                endpoint, target.with_suffix(".boundary.json"), inspect
            )
        except BaseException as error:
            summary["boundary_error"] = error_info(error)["repr"]
            if failure is None:
                failure = error
                raise
        finally:
            try:
                write_json(target.with_suffix(".summary.json"), summary)
            except BaseException:
                if failure is None:
                    raise


def finish_full(output, server, port, evaluations, samples, failure, attempts=()):
    """Record the terminal cause before cleanup; optional reports follow it."""
    attempted = {x["identity"]: x for x in attempts}
    planned = []
    for trace in ("azure", "burst"):
        for phase, count in (("calibration", 3), ("evaluation", 12)):
            for i in range(count):
                identity = f"{trace}-{phase}-{i}"
                item = attempted.get(identity)
                planned.append(
                    {
                        "identity": identity,
                        "status": item.get("status", "running") if item else "not_run",
                    }
                )
    terminal = {
        "aborted": failure is not None,
        "baseline_accepted": False,
        "primary_failure": error_info(failure) if failure is not None else None,
        "planned_trials": planned,
        "stage": "before_cleanup",
        "report_errors": [],
    }
    try:
        write_json(output / "terminal.json", terminal)
    except BaseException as error:  # noqa: BLE001 - preserve primary failure through bounded cleanup/reporting
        terminal["report_errors"].append(error_info(error))
        failure = failure or error
    try:
        cleanup = stop_child(server, port) if server is not None else {"not_started": True}
    except BaseException as error:  # noqa: BLE001 - preserve primary failure through bounded cleanup/reporting
        cleanup = {"cleanup_error": error_info(error), "confirmed": False}
        failure = failure or error
    terminal["cleanup"] = cleanup
    if not cleanup.get("not_started"):
        cleanup["native_exit_expected"] = (
            cleanup.get("termination_requested") is True
            and cleanup.get("returncode_before_cleanup") is None
            and cleanup.get("returncode") in (0, -signal.SIGTERM)
        )
        if not cleanup["native_exit_expected"]:
            failure = failure or RuntimeError(
                "unexpected native exit: "
                f"returncode={cleanup.get('returncode')}, "
                f"termination_requested={cleanup.get('termination_requested')}, "
                f"before_cleanup={cleanup.get('returncode_before_cleanup')}"
            )
    if not cleanup.get("not_started") and (
        not cleanup.get("listener_closed") or cleanup.get("forced_kill")
    ):
        failure = failure or RuntimeError("server cleanup did not finish normally")
    reports = [
        (output / "cleanup.json", cleanup),
        (output / "resource-samples.json", samples),
    ]
    for name, summaries in evaluations.items():
        try:
            pairs = matched_pairs(summaries, trial_order())
            pairs["execution_aborted"] = failure is not None
            if failure is not None:
                for comparison in pairs["comparisons"].values():
                    comparison["accepted_all_three"] = False
                    comparison["goodput_difference"] = None
            reports.append((output / f"{name}-pairs.json", pairs))
        except BaseException as error:  # noqa: BLE001 - preserve primary failure through bounded cleanup/reporting
            terminal["report_errors"].append(error_info(error))
            failure = failure or error
    for path, data in reports:
        try:
            write_json(path, data)
        except BaseException as error:  # noqa: BLE001 - preserve primary failure through bounded cleanup/reporting
            terminal["report_errors"].append(error_info(error))
            failure = failure or error
    if failure is not None:
        for name, summaries in evaluations.items():
            try:
                pairs = matched_pairs(summaries, trial_order())
                pairs["execution_aborted"] = True
                for comparison in pairs["comparisons"].values():
                    comparison["accepted_all_three"] = False
                    comparison["goodput_difference"] = None
                write_json(output / f"{name}-pairs.json", pairs)
            except BaseException as error:  # noqa: BLE001 - preserve primary failure through bounded cleanup/reporting
                terminal["report_errors"].append(error_info(error))
    if terminal["report_errors"] and failure is None:
        failure = RuntimeError("terminal metadata publication failed")
    terminal.update(
        stage="after_cleanup",
        aborted=failure is not None,
        run_completed=failure is None,
        baseline_accepted=False,
        primary_failure=error_info(failure) if failure is not None else None,
    )
    try:
        write_json(output / "terminal.json", terminal)
    except BaseException:
        if failure is None:
            raise
    if failure is not None and terminal["primary_failure"] is not None:
        # Caller already has its original exception in flight when applicable.
        return failure
    return None


def planned_tokens(path, kind):
    keys = (
        ("ContextTokens", "GeneratedTokens")
        if kind == "azure"
        else ("Request tokens", "Response tokens")
    )
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    return {
        "requested_input_tokens": sum(int(row[keys[0]]) for row in rows),
        "requested_output_tokens": sum(int(row[keys[1]]) for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "lock",
        "overlay",
        "cache-audit",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("protocol", "model", "model-manifest", "token-ids", "binary"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--abi-evidence", type=Path)
    parser.add_argument("--abi-evidence-sha256")
    parser.add_argument("--abi-only", action="store_true")
    parser.add_argument("--port", type=int, default=31012)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise InterruptedError(f"interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    if not args.abi_only and not listener_closed(args.port):
        raise ValueError("requested loopback port is occupied")
    lock = json.loads(args.lock.read_text())
    if not args.abi_only:
        if any(
            getattr(args, name) is None
            for name in (
                "protocol",
                "model",
                "model_manifest",
                "token_ids",
                "binary",
                "abi_evidence",
                "abi_evidence_sha256",
            )
        ):
            raise ValueError("full experiment requires frozen model/trace/protocol inputs")
        protocol = json.loads(args.protocol.read_text())
        expected_model = (
            Path.home()
            / ".cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots"
            / lock["model"]["revision"]
        )
        args.model = args.model.resolve()
        if args.model != expected_model:
            raise ValueError("model path revision mismatch")
        tokens = json.loads(args.token_ids.read_text())
        plan_path = Path(__file__).with_name("model-files.json")
        plan = load_plan(plan_path)
        manifest = json.loads(args.model_manifest.read_text())
        if (
            tokens["Revision"] != lock["model"]["revision"]
            or tokens["Model"] != lock["model"]["repo"]
            or manifest["plan_sha256"] != sha256(plan_path)
            or manifest["runtime_lock_sha256"] != sha256(args.lock)
        ):
            raise ValueError("token allowlist lineage mismatch")
        verify_model_inputs(plan, manifest, args.model, tokens)
        if sha256(args.abi_evidence) != args.abi_evidence_sha256:
            raise ValueError("prior ABI evidence hash mismatch")
        prior_abi = json.loads(args.abi_evidence.read_text())
        if prior_abi["inventory"] != inventory(
            lock, args.overlay.resolve(), ROOT / ".venv-ml/lib/python3.12/site-packages"
        ):
            raise ValueError("prior ABI inventory differs from current runtime")
        if protocol["trial_order"] != trial_order():
            raise ValueError("trial protocol order changed")
        if protocol["trial_boundary"] != TRIAL_BOUNDARY:
            raise ValueError("trial boundary protocol changed")
    args.output.mkdir(parents=True, exist_ok=False)
    work = ROOT / ".cache/task12-baseline/native-cache"
    work.mkdir(parents=True, exist_ok=True)
    base_bytes = canonical_bytes(ROOT)
    began = time.monotonic()
    samples = []

    def inspect(canonical=False):
        samples.append(check_resources(work, base_bytes, args.output, canonical))
        if len(samples) % 20 == 0:
            if args.abi_only:
                (args.output / "resource-samples.json").write_text(json.dumps(samples))
            else:
                write_json(args.output / "resource-samples.json", samples)

    endpoint = f"http://127.0.0.1:{args.port}"
    env = runtime_environment(work)
    entry = [
        str(ROOT / ".venv-ml/bin/python"),
        "-B",
        "-S",
        "-m",
        "serving.baseline.runtime_entry",
        "--lock",
        str(args.lock),
        "--overlay",
        str(args.overlay),
        "--work",
        str(work),
        "--cache-audit",
        str(args.cache_audit),
    ]
    if args.abi_only:
        with (args.output / "abi.log").open("wb") as log:
            inspect(canonical=True)
            abi = subprocess.Popen(
                entry + ["--mode", "abi", "--inventory-out", str(args.output / "abi.json")],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                if _wait(abi, 180, inspect):
                    raise RuntimeError("native ABI smoke failed")
            finally:
                cleanup = stop_child(abi, None)
                (args.output / "abi-cleanup.json").write_text(json.dumps(cleanup, indent=2) + "\n")
                (args.output / "resource-samples.json").write_text(json.dumps(samples) + "\n")
        inspect(canonical=True)
        (args.output / "resource-samples.json").write_text(json.dumps(samples) + "\n")
        return
    write_json(
        args.output / "abi-reference.json",
        {
            "path": str(args.abi_evidence.resolve()),
            "sha256": args.abi_evidence_sha256,
            "reused_prior_success": True,
            "rerun": False,
        },
    )
    server_log = (args.output / "server.log").open("wb")
    command = entry + [
        "--mode",
        "server",
        "--inventory-out",
        str(args.output / "server-inventory.json"),
        "--",
        "--model-path",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        *protocol["server_args"],
    ]
    server, failure = None, None
    attempts = []
    evaluations = {trace["name"]: [] for trace in protocol["traces"]}
    try:
        server = subprocess.Popen(
            command,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        ready_deadline = time.monotonic() + 900
        while True:
            inspect()
            if server.poll() is not None or time.monotonic() >= ready_deadline:
                raise RuntimeError("server readiness failed")
            try:
                with urllib.request.urlopen(endpoint + "/health", timeout=1) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                time.sleep(1)
        native_smoke(
            endpoint + "/generate",
            tokens["token_ids"],
            lock["model"]["vocab_size"],
            args.output,
            inspect,
        )
        fixed_warmups(
            endpoint + "/generate", tokens["token_ids"], protocol["warmup"], args.output, inspect
        )
        flush_barrier(endpoint, args.output / "warmup-boundary.json", inspect)
        runtime_lineage = {
            "model_manifest_sha256": sha256(args.model_manifest),
            "runtime_sha256": sha256(args.abi_evidence),
            "server_inventory_sha256": sha256(args.output / "server-inventory.json"),
            "lock_sha256": sha256(args.lock),
            "protocol_sha256": sha256(args.protocol),
            "token_allowlist_sha256": sha256(args.token_ids),
            "server_args": command,
            "sources_sha256": _digest(
                {
                    str(path.relative_to(ROOT)): sha256(path)
                    for path in [
                        Path(__file__),
                        ROOT / "serving/baseline/runtime_entry.py",
                        args.cache_audit.resolve(),
                    ]
                }
            ),
            "replay_binary_sha256": sha256(args.binary),
        }
        for trace in protocol["traces"]:
            lineage = {
                **runtime_lineage,
                "calibration_trace_sha256": trace["calibration"]["sha256"],
                "evaluation_trace_sha256": trace["evaluation"]["sha256"],
            }
            calibration, identities = [], []
            slo = None
            for phase in ("calibration", "evaluation"):
                csv_path = ROOT / trace[phase]["file"]
                if sha256(csv_path) != trace[phase]["sha256"]:
                    raise ValueError("frozen trace hash changed")
                order = (
                    [{"pair": f"calibration-{i}", "multiplier": 1} for i in range(3)]
                    if phase == "calibration"
                    else trial_order()
                )
                for ordinal, trial in enumerate(order):
                    if time.monotonic() - began >= 3 * 3600 + 15 * 60:
                        raise TimeoutError("stop-new-trials deadline reached")
                    inspect(canonical=True)
                    identity = f"{trace['name']}-{phase}-{ordinal}"
                    target = args.output / (identity + ".json")
                    command = [
                        str(args.binary),
                        "--trace",
                        str(csv_path),
                        "--format",
                        trace["format"],
                        "--url",
                        endpoint + "/generate",
                        "--token-ids",
                        str(args.token_ids),
                        "--mult",
                        str(trial["multiplier"]),
                        "--timeout",
                        "60s",
                        "--inflight",
                        "192",
                        "--seed",
                        str(protocol["seed"]),
                    ]
                    summary = {
                        "identity": identity,
                        "trial": trial,
                        "protocol_sha256": sha256(args.protocol),
                        "lock_sha256": sha256(args.lock),
                        "planned_rows": trace[phase]["planned_rows"],
                        **planned_tokens(csv_path, trace["format"]),
                    }
                    attempts.append(summary)
                    if phase == "evaluation":
                        evaluations[trace["name"]].append(summary)
                    result = execute_isolated_trial(
                        command, env, target, summary, inspect, endpoint
                    )
                    if phase == "calibration":
                        calibration.append(result)
                        identities.append(summary["raw_sha256"])
                    else:
                        if slo is None:
                            raise ValueError("missing frozen calibration")
                        summary["metrics"] = evaluate(result, slo, summary["raw_sha256"], lineage)
                        summary["metrics"]["planned"] = summary["planned_rows"]
                        write_json(
                            args.output / f"{trace['name']}-pairs.json",
                            matched_pairs(evaluations[trace["name"]], trial_order()),
                        )
                    write_json(target.with_suffix(".summary.json"), summary)
                if phase == "calibration":
                    slo = calibrate(calibration, identities, lineage)
                    slo.update(
                        trace_sha256=trace["calibration"]["sha256"],
                        protocol_sha256=sha256(args.protocol),
                    )
                    slo["sha256"] = _digest({k: v for k, v in slo.items() if k != "sha256"})
                    write_json(args.output / f"{trace['name']}-slo.json", slo)
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            final_error = finish_full(
                args.output, server, args.port, evaluations, samples, failure, attempts
            )
        finally:
            server_log.close()
        if failure is None and final_error is not None:
            raise final_error


if __name__ == "__main__":
    main()
