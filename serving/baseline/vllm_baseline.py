"""Single-RTX-4090 serving baseline: vLLM OpenAI server + Go open-loop replay.

One invocation serves one trace (calibration + evaluation windows) from one
server process, and must run under scripts/with_gpu.sh. Steps:

1. start vLLM (gpu-memory-utilization 0.80, seed 17, no prefix cache) and wait;
2. fixed warmup requests, outside every measured window;
3. three 1x calibration runs; SLO = pooled nearest-rank p99 TTFT and TPOT over
   successful >=2-token requests, from sender-valid runs only;
4. three evaluation rounds of every level, order rotated per round;
5. stop the server and write summary.json.

"1x" is ``--base-mult`` times the source arrival rate; level L replays at
``base_mult * L``. Raw replay JSON for every trial is kept next to the summary.
The run stops early when owned RSS exceeds 16 GB, the runtime cache exceeds its
cap, or the 3h15m new-trial deadline passes; the summary records why.
"""

import argparse
import csv
import datetime
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

LEVELS = (1, 2, 4)
SLO_QUANTILE = 0.99
RSS_CAP = 16 * 10**9
NEW_TRIAL_DEADLINE_S = 3 * 3600 + 15 * 60
REQUEST_TIMEOUT_S = 60
MAX_INFLIGHT = 192
# (input tokens, output tokens), sent one second apart before any measurement.
WARMUP = ((128, 16), (1024, 32), (2048, 32), (4000, 32))


def eval_order(rounds, levels):
    return [list(levels[i % len(levels) :]) + list(levels[: i % len(levels)]) for i in range(rounds)]


def nearest_rank(values, q):
    """Same rank as replay.Calibrate: sorted[ceil(q*n)-1]."""
    if not values:
        raise ValueError("no samples")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _eligible(r):
    return r["Status"] == "ok" and r["OutputTokens"] >= 2 and r["TTFTms"] >= 0 and r["TPOTms"] >= 0


def calibrate(runs):
    """Pool >=2-token successes from exactly the given sender-valid runs."""
    ttft, tpot, failed, undefined = [], [], 0, 0
    for run in runs:
        result = run["Run"]
        if result["Aborted"] or not result["SenderValid"]:
            raise ValueError("calibration needs sender-valid, complete runs")
        for r in result["Records"]:
            if r["Status"] != "ok":
                failed += 1
            elif _eligible(r):
                ttft.append(r["TTFTms"])
                tpot.append(r["TPOTms"])
            else:
                undefined += 1
    return {
        "ttft_ms": nearest_rank(ttft, SLO_QUANTILE),
        "tpot_ms": nearest_rank(tpot, SLO_QUANTILE),
        "quantile": SLO_QUANTILE,
        "samples": len(ttft),
        "failed": failed,
        "undefined": undefined,
        "runs": len(runs),
    }


def _dist(values):
    if not values:
        return None
    return {
        "p50": nearest_rank(values, 0.50),
        "p90": nearest_rank(values, 0.90),
        "p99": nearest_rank(values, 0.99),
        "max": max(values),
    }


def summarize(output, slo):
    result = output["Run"]
    records = result["Records"]
    ok = [r for r in records if r["Status"] == "ok"]
    failures = [r for r in records if r["Status"] != "ok"]
    timeouts = [r for r in failures if "deadline exceeded" in r["Error"]]
    late = [r for r in records if r["SentMs"] < 0 or r["SentMs"] - r["ScheduledMs"] > 5]
    duration_s = result["DurationMs"] / 1000
    good = None
    if slo is not None:
        good = sum(_eligible(r) and r["TTFTms"] <= slo["ttft_ms"] and r["TPOTms"] <= slo["tpot_ms"] for r in records)
    statuses = {}
    for r in failures:
        statuses[r["Status"]] = statuses.get(r["Status"], 0) + 1
    return {
        "requests": len(records),
        "ok": len(ok),
        "eligible_ok": sum(_eligible(r) for r in records),
        "good": good,
        "timeouts": len(timeouts),
        "other_failures": len(failures) - len(timeouts),
        "failure_statuses": statuses,
        "late_sends": len(late),
        "sender_valid": bool(result["SenderValid"]) and not result["Aborted"],
        "duration_s": duration_s,
        "goodput_rps": None if good is None else good / duration_s,
        "completed_rps": len(ok) / duration_s,
        "output_tokens": sum(r["OutputTokens"] for r in records),
        "output_tok_per_s": sum(r["OutputTokens"] for r in ok) / duration_s,
        "ttft_ms": _dist([r["TTFTms"] for r in ok if r["TTFTms"] >= 0]),
        "tpot_ms": _dist([r["TPOTms"] for r in ok if r["TPOTms"] >= 0]),
    }


def _range(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def aggregate(trials, levels):
    """Median/min/max goodput over sender-valid rounds, plus paired level-minus-1x."""
    by_round = {}
    for t in trials:
        if t["sender_valid"] and t["goodput_rps"] is not None:
            by_round.setdefault(t["round"], {})[t["level"]] = t["goodput_rps"]
    out = {}
    for level in levels:
        values = [r[level] for r in by_round.values() if level in r]
        planned = sum(t["level"] == level for t in trials)
        entry = {"invalid_rounds": planned - len(values)}
        entry["goodput_rps"] = {**_range(values), "valid_rounds": len(values)} if values else None
        if level != levels[0]:
            diffs = [r[level] - r[levels[0]] for r in by_round.values() if level in r and levels[0] in r]
            entry["paired_minus_1x"] = _range(diffs) if diffs else None
        out[str(level)] = entry
    return out


def queue_depth(metrics_text):
    total = 0.0
    for line in metrics_text.splitlines():
        m = re.match(r"vllm:num_requests_(running|waiting)\{[^}]*\}\s+([0-9.eE+-]+)$", line)
        if m:
            total += float(m.group(2))
    return int(total)


def server_command(snapshot, name, port, max_model_len, max_num_seqs):
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", snapshot, "--served-model-name", name,
        "--host", "127.0.0.1", "--port", str(port),
        "--gpu-memory-utilization", "0.8", "--seed", "17",
        "--max-model-len", str(max_model_len), "--max-num-seqs", str(max_num_seqs),
        "--no-enable-prefix-caching", "--generation-config", "vllm",
    ]  # fmt: skip


def _get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, resp.read(1 << 20).decode()


def _owned_rss(root_pid):
    """Sum RSS of root_pid and every descendant (by /proc ppid links)."""
    parents, rss = {}, {}
    page = os.sysconf("SC_PAGE_SIZE")
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        parents[int(entry.name)] = int(fields[1])
        rss[int(entry.name)] = int(fields[21]) * page
    owned, frontier = {root_pid}, [root_pid]
    while frontier:
        pid = frontier.pop()
        for child, parent in parents.items():
            if parent == pid and child not in owned:
                owned.add(child)
                frontier.append(child)
    return sum(rss.get(p, 0) for p in owned)


def _dir_bytes(path):
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except OSError:
                pass
    return total


class Monitor(threading.Thread):
    """Samples owned RSS and runtime-cache size; sets .stop_reason on a breach."""

    def __init__(self, cache_root, cache_cap):
        super().__init__(daemon=True)
        self.cache_root, self.cache_cap = cache_root, cache_cap
        self.peak_rss = self.peak_cache = 0
        self.stop_reason = None
        self.halt = threading.Event()

    def run(self):
        while not self.halt.wait(2.0):
            rss = _owned_rss(os.getpid())
            self.peak_rss = max(self.peak_rss, rss)
            if rss > RSS_CAP and not self.stop_reason:
                self.stop_reason = f"owned RSS {rss} > {RSS_CAP}"
            cache = _dir_bytes(self.cache_root)
            self.peak_cache = max(self.peak_cache, cache)
            if cache > self.cache_cap and not self.stop_reason:
                self.stop_reason = f"runtime cache {cache} > {self.cache_cap}"


def _span(path):
    """Source-time seconds from first to last arrival (bounds the replay timeout)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    if "Timestamp" in head:
        stamps = [float(r[head.index("Timestamp")]) for r in body]
    else:
        col = head.index("TIMESTAMP")
        stamps = [datetime.datetime.fromisoformat(r[col][:26]).timestamp() for r in body]
    return stamps[-1] - stamps[0]


def run_replay(args, trace_csv, fmt, mult, out_json, port):
    cmd = [
        args.replay_bin, "-trace", str(trace_csv), "-format", fmt,
        "-url", f"http://127.0.0.1:{port}/v1/completions", "-mult", repr(mult),
        "-token-ids", args.token_ids, "-timeout", f"{REQUEST_TIMEOUT_S}s",
        "-inflight", str(MAX_INFLIGHT), "-seed", "17", "-api", "openai", "-model", args.model_name,
    ]  # fmt: skip
    limit = _span(trace_csv) / mult + REQUEST_TIMEOUT_S + 60
    with open(out_json, "wb") as stdout:
        proc = subprocess.run(cmd, stdout=stdout, stderr=subprocess.PIPE, timeout=limit, check=False)
    if proc.stderr:
        Path(str(out_json) + ".stderr").write_bytes(proc.stderr[:65536])
    return proc.returncode, json.loads(Path(out_json).read_text())


def wait_idle(port, limit_s=180):
    deadline = time.monotonic() + limit_s
    quiet = 0
    while time.monotonic() < deadline:
        _, text = _get(f"http://127.0.0.1:{port}/metrics")
        quiet = quiet + 1 if queue_depth(text) == 0 else 0
        if quiet >= 2:
            return True
        time.sleep(1)
    return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace-dir", type=Path, required=True, help="has calibration.csv, evaluation.csv")
    p.add_argument("--format", choices=("azure", "burstgpt"), required=True)
    p.add_argument("--base-mult", type=float, required=True, help="source-rate multiplier that defines 1x")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--token-ids", required=True)
    p.add_argument("--replay-bin", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    # Keeps the runtime cache small next to the project's disk share.
    p.add_argument("--cache-cap", type=int, default=300 * 10**6)
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args(argv)

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary = {
        "trace_dir": str(args.trace_dir),
        "format": args.format,
        "base_mult": args.base_mult,
        "levels": LEVELS,
        "model": args.model_name,
        "snapshot": args.snapshot,
        "request_timeout_s": REQUEST_TIMEOUT_S,
        "max_inflight": MAX_INFLIGHT,
        "calibration": [],
        "evaluation": [],
        "stop_reason": None,
    }
    cmd = server_command(args.snapshot, args.model_name, args.port, args.max_model_len, args.max_num_seqs)
    summary["server_command"] = cmd[1:]
    monitor = Monitor(args.cache_root, args.cache_cap)
    monitor.start()
    log = open(out / "server.log", "wb")  # noqa: SIM115 - closed after the server stops
    server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        t0 = time.monotonic()
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode} before ready")
            try:
                if _get(f"http://127.0.0.1:{args.port}/health")[0] == 200:
                    break
            except OSError:
                pass
            if time.monotonic() - t0 > 900:
                raise RuntimeError("server not ready in 900 s")
            time.sleep(2)
        summary["startup_s"] = time.monotonic() - t0

        warm = args.cache_root / "warmup.csv"
        warm.write_text(
            "Timestamp,Request tokens,Response tokens\n" + "".join(f"{i},{a},{b}\n" for i, (a, b) in enumerate(WARMUP))
        )
        code, warm_out = run_replay(args, warm, "burstgpt", 1.0, out / "warmup.json", args.port)
        summary["warmup"] = {"exit": code, **summarize(warm_out, None)}
        wait_idle(args.port)

        def trial(role, csv_name, level, rnd, slo):
            if monitor.stop_reason or server.poll() is not None:
                summary["stop_reason"] = summary["stop_reason"] or monitor.stop_reason or "server exited"
                return None
            if time.monotonic() - started > NEW_TRIAL_DEADLINE_S:
                summary["stop_reason"] = "new-trial deadline"
                return None
            name = f"{role}-r{rnd}-x{level}.json"
            mult = args.base_mult * level
            code, data = run_replay(args, args.trace_dir / csv_name, args.format, mult, out / name, args.port)
            idle = wait_idle(args.port)
            entry = {"file": name, "round": rnd, "level": level, "mult": mult, "exit": code, "idle_after": idle}
            entry.update(summarize(data, slo))
            print(json.dumps({k: entry[k] for k in ("file", "sender_valid", "ok", "timeouts", "goodput_rps")}), flush=True)
            return entry, data

        cal_outputs = []
        for rnd in range(args.rounds):
            res = trial("calibration", "calibration.csv", 1, rnd, None)
            if res is None:
                break
            summary["calibration"].append(res[0])
            if res[0]["sender_valid"]:
                cal_outputs.append(res[1])
        if len(cal_outputs) < 3:
            summary["stop_reason"] = summary["stop_reason"] or f"only {len(cal_outputs)} sender-valid calibrations"
        else:
            slo = calibrate(cal_outputs)
            summary["slo"] = slo
            for rnd, order in enumerate(eval_order(args.rounds, LEVELS)):
                for level in order:
                    res = trial("evaluation", "evaluation.csv", level, rnd, slo)
                    if res is None:
                        break
                    summary["evaluation"].append(res[0])
            summary["aggregate"] = aggregate(summary["evaluation"], LEVELS)
    except Exception as exc:  # noqa: BLE001 - recorded as the terminal cause
        summary["stop_reason"] = summary["stop_reason"] or f"{type(exc).__name__}: {exc}"
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
        except ProcessLookupError:
            pass
        log.close()
        monitor.halt.set()
        monitor.join()
        summary["server_exit"] = server.returncode
        summary["peak_owned_rss_bytes"] = monitor.peak_rss
        summary["peak_runtime_cache_bytes"] = monitor.peak_cache
        summary["wall_s"] = time.monotonic() - started
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0 if summary["stop_reason"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
