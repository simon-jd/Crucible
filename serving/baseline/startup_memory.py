"""Bounded startup attribution only; no generation or performance acceptance."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from serving.baseline.experiment import (
    check_resources,
    listener_closed,
    stop_child,
    verify_model_inputs,
)
from serving.baseline.model_snapshot import canonical_bytes, load_plan
from serving.baseline.prepare_runtime import sha256
from serving.baseline.runtime_entry import ROOT, runtime_environment


def bounded_read(path):
    with path.open("rb") as source:
        data = source.read(8193)
    if len(data) > 8192:
        raise ValueError("proc metadata read bound exceeded")
    return data


def read_stat(directory):
    text = bounded_read(directory / "stat").decode()
    fields = text.rsplit(")", 1)[1].split()
    row = {
        "pid": int(directory.name),
        "ppid": int(fields[1]),
        "pgrp": int(fields[2]),
        "start_ticks": int(fields[19]),
        "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
    }
    if row["rss_bytes"] < 0:
        raise ValueError("negative proc RSS")
    return row


def snapshot(root_pid, seen, *, proc=Path("/proc"), server_pid=None):
    rows, events = {}, []
    count = 0
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        count += 1
        if count > 8192:
            raise ValueError("proc enumeration bound exceeded")
        try:
            row = read_stat(directory)
        except FileNotFoundError:
            events.append({"pid": int(directory.name), "event": "enumeration-disappeared"})
            continue
        rows[row["pid"]] = row
    for pid, start in list(seen.items()):
        if pid in rows and rows[pid]["start_ticks"] != start:
            raise ValueError(f"owned PID identity changed: {pid}")
        if pid not in rows:
            if (proc / str(pid)).exists():
                raise ValueError(f"owned process metadata missing: {pid}")
            events.append({"pid": pid, "start_ticks": start, "event": "absent"})
            del seen[pid]
    if root_pid not in rows:
        raise ValueError("observer root identity missing")
    owned = {root_pid} | set(seen)
    while True:
        found = {pid for pid, row in rows.items() if row["ppid"] in owned}
        if found <= owned:
            break
        owned |= found
    if len(owned) > 64:
        raise ValueError("owned process bound exceeded")
    result = []
    for pid in sorted(owned):
        directory = proc / str(pid)
        row = rows[pid]
        # Any ambiguous/denied metadata of a known-owned process fails closed.
        # The caller retains the error before cleanup instead of treating it as0.
        status = bounded_read(directory / "status").decode()
        values = {}
        for line in status.splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "RssAnon", "RssFile", "RssShmem"}:
                amount, units = value.split()
                if units != "kB" or int(amount) < 0:
                    raise ValueError("invalid owned RSS component")
                values[key] = int(amount) * 1024
        if len(values) != 4:
            raise ValueError("owned RSS components missing")
        title = bounded_read(directory / "cmdline").split(b"\0", 1)[0]
        category = "unknown-owned-child"
        if pid == root_pid:
            category = "observer"
        elif pid == server_pid:
            category = "server-entry"
        elif title.startswith(b"sglang::scheduler"):
            category = "scheduler-title"
        elif title == b"sglang::detokenizer":
            category = "detokenizer-title"
        after = read_stat(directory)
        if row["start_ticks"] != after["start_ticks"]:
            raise ValueError(f"owned PID identity changed during sample: {pid}")
        # Both are the same stat-RSS observable. Keep the larger non-atomic read;
        # status components are attribution only, never deductions from the sum.
        row.update(
            rss_bytes=max(row["rss_bytes"], after["rss_bytes"]),
            category=category,
            status_components=values,
            **values,
        )
        result.append(row)
        seen[pid] = row["start_ticks"]
    return {
        "monotonic": time.monotonic(),
        "aggregate_rss_bytes": sum(x["rss_bytes"] for x in result),
        "processes": result,
        "events": events,
    }


READY_LOG = b"The server is fired up and ready to roll!"


def run_probe(
    command, env, output, port, inspect, *, sampler=snapshot, seconds=120, evidence_limit=16 << 20
):
    """Own one child; persist trigger/error before exact child cleanup."""
    result = {
        "baseline_accepted": False,
        "reason": "running",
        "early_rss_stop_bytes": 10_000_000_000,
        "unchanged_service_cap_bytes": 16_000_000_000,
        "samples": 0,
        "sampling_is_not_an_atomic_quota": True,
    }
    child = None
    began = time.monotonic()
    seen = {}
    try:
        with (
            (output / "server.log").open("xb") as log,
            (output / "samples.jsonl").open("x") as samples,
        ):
            result["initial_resources"] = inspect()
            child = subprocess.Popen(
                command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0
            )
            os.set_blocking(child.stdout.fileno(), False)
            result["pid"] = child.pid
            last_inspect = began
            next_sample = began
            reserve = min(65536, evidence_limit // 4)
            log_tail = b""
            saw_ready = False
            while True:
                # The child writes into a bounded OS pipe. Copy only bounded
                # chunks, and stop before the owned evidence budget is exhausted.
                for _ in range(8):
                    chunk = child.stdout.read(65536)
                    if not chunk:
                        break
                    if canonical_bytes(output) + len(chunk) + reserve > evidence_limit:
                        raise ValueError("diagnostic log/evidence bound exceeded")
                    log.write(chunk)
                    log.flush()
                    joined = log_tail + chunk
                    saw_ready |= READY_LOG in joined
                    log_tail = joined[-(len(READY_LOG) - 1) :]
                sample = sampler(os.getpid(), seen, server_pid=child.pid)
                resource_error = None
                if time.monotonic() - last_inspect >= 1:
                    try:
                        sample["resources"] = inspect()
                    except BaseException as error:  # noqa: BLE001 — re-raised after bounded flush
                        resource_error = error
                        sample["resource_error"] = repr(error)
                    last_inspect = time.monotonic()
                data = json.dumps(sample, separators=(",", ":")) + "\n"
                if canonical_bytes(output) + len(data.encode()) + reserve > evidence_limit:
                    raise ValueError("diagnostic evidence bound exceeded")
                samples.write(data)
                samples.flush()
                result["samples"] += 1
                if resource_error is not None:
                    raise resource_error
                if sample["aggregate_rss_bytes"] >= 10_000_000_000:
                    result["reason"] = "early-memory-stop"
                    break
                if child.poll() is not None:
                    result["reason"] = "server-exit"
                    break
                if time.monotonic() - began >= seconds or result["samples"] >= 1200:
                    result["reason"] = "deadline"
                    break
                if saw_ready:
                    result["readiness_observable"] = READY_LOG.decode()
                    result["reason"] = "ready-no-generation"
                    break
                next_sample += 0.1
                time.sleep(max(0, min(next_sample, began + seconds) - time.monotonic()))
    except BaseException as error:  # noqa: BLE001 — retain interruption evidence before cleanup
        result.update(reason="error", error=repr(error))
    finally:
        # Optional evidence I/O must never prevent owned process cleanup.
        try:
            (output / "stop.json").write_text(json.dumps(result, indent=2) + "\n")
        finally:
            if child is not None:
                try:
                    result["cleanup"] = stop_child(child, port)
                    if not (result["cleanup"]["reaped"] and result["cleanup"]["listener_closed"]):
                        result["reason"] = "cleanup-failed"
                finally:
                    child.stdout.close()
        result["elapsed_seconds"] = time.monotonic() - began
        (output / "terminal.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "lock",
        "overlay",
        "cache-audit",
        "protocol",
        "model",
        "model-manifest",
        "token-ids",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--port", type=int, default=31012)
    args = parser.parse_args()
    if not listener_closed(args.port):
        raise ValueError("diagnostic loopback port is occupied")
    plan_path = Path(__file__).with_name("model-files.json")
    plan = load_plan(plan_path)
    manifest = json.loads(args.model_manifest.read_text())
    tokens = json.loads(args.token_ids.read_text())
    expected_model = (
        Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots" / plan["revision"]
    )
    if (
        args.model.resolve() != expected_model
        or manifest["plan_sha256"] != sha256(plan_path)
        or manifest["runtime_lock_sha256"] != sha256(args.lock)
    ):
        raise ValueError("diagnostic model/lock lineage mismatch")
    verify_model_inputs(plan, manifest, args.model.resolve(), tokens)
    protocol = json.loads(args.protocol.read_text())
    if sha256(args.protocol) != "301bb8c4137c1b565fb7e3aca282ec14dc61cafd4a751b3333dcc9e50e117f61":
        raise ValueError("diagnostic original protocol changed")
    args.output.mkdir(parents=True, exist_ok=False)
    work = ROOT / ".cache/task12-baseline/native-cache"
    baseline = canonical_bytes(ROOT)
    env = runtime_environment(work)
    command = [
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
        "--mode",
        "server",
        "--inventory-out",
        str(args.output / "server-inventory.json"),
        "--",
        "--model-path",
        str(args.model.resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        *protocol["server_args"],
        "--skip-server-warmup",
    ]

    def terminated(signum, frame):
        raise InterruptedError("diagnostic terminated")

    previous = signal.signal(signal.SIGTERM, terminated)
    try:
        result = run_probe(
            command,
            env,
            args.output,
            args.port,
            lambda: check_resources(work, baseline, args.output, canonical=False),
        )
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0 if result["reason"] == "ready-no-generation" else 1


if __name__ == "__main__":
    raise SystemExit(main())
