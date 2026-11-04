"""Condense an E3 run directory into per-trial and per-condition rows (stdout JSON).

  python3 docs/evidence/snowflake/e3/e3_table.py docs/evidence/snowflake/e3/main > .../e3-table.json

Requests served by b come from its vLLM access log ('POST /v1/completions' lines, the count is in the first line
of server-b.excerpt.log) minus one readiness completion per start of b.
"""
import csv
import datetime
import json
import re
import statistics
import sys
from pathlib import Path

run = Path(sys.argv[1])
s = json.loads((run / "summary.json").read_text())
phases = s["meta"]["phases"]
burst0 = phases[0][0]
burst1 = burst0 + phases[1][0]

gpu = []
with open(run / "gpu-samples.csv") as f:
    for row in csv.reader(f):
        try:
            t = datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.UTC)
            gpu.append((t.timestamp(), int(row[1]), float(row[2]), float(row[3])))
        except (ValueError, IndexError):
            continue


def gpu_window(t0, t1):
    out = {}
    for idx in (0, 1):
        xs = [(m, u) for t, i, m, u in gpu if i == idx and t0 <= t < t1]
        out[idx] = None if not xs else {"util_mean": round(statistics.mean(u for _, u in xs), 1),
                                        "mem_max_mib": max(m for m, _ in xs), "n": len(xs)}
    return out


trials = []
for t in s["trials"]:
    d = run / f"trial-{t['index']}-{t['condition']}"
    events = json.loads((d / "events.json").read_text())
    tl = json.loads((d / "timeline.json").read_text())
    start = next(e["t"] for e in events if e["event"] == "replay_start")
    starts = sum(e["event"] in ("replica_spawn", "replica_prestarted") for e in events)
    served_b = None
    for log in (d / "server-b.log", d / "server-b.excerpt.log"):
        if log.exists():
            text = log.read_text(errors="replace")
            m = re.match(r"# excerpt of .*?: (\d+) 'POST", text)
            served_b = (int(m.group(1)) if m else text.count("POST /v1/completions")) - starts
            break
    records = json.loads((d / "replay.json").read_text())["Run"]["Records"]
    slo = s["slo"]
    burst = [r for r in records if burst0 * 1000 <= r["ScheduledMs"] < burst1 * 1000]
    el = [r for r in burst if r["Status"] == "ok" and r["OutputTokens"] >= 2 and r["TTFTms"] >= 0 and r["TPOTms"] >= 0]
    burst_detail = {"requests": len(burst), "ok": sum(r["Status"] == "ok" for r in burst),
                    "ttft_within_slo": sum(r["TTFTms"] <= slo["ttft_ms"] for r in el),
                    "tpot_within_slo": sum(r["TPOTms"] <= slo["tpot_ms"] for r in el)}  # fmt: skip
    p = t["phases"]
    trials.append({
        "index": t["index"], "condition": t["condition"], "sender_valid": t["sender_valid"],
        "accounting": t["accounting"],
        "goodput": [round(x["goodput_rps"], 3) for x in p],
        "good": [x["good"] for x in p], "requests": [x["requests"] for x in p], "failed": [x["failed"] for x in p],
        "timeouts": [x["timeouts"] for x in p],
        "p50_tpot": [round((x["tpot_ms"] or {}).get("p50", float("nan")), 2) for x in p],
        "p50_ttft": [round((x["ttft_ms"] or {}).get("p50", float("nan")), 1) for x in p],
        "p99_ttft": [round((x["ttft_ms"] or {}).get("p99", float("nan")), 1) for x in p],
        "reactions": t["reactions"], "b_starts": starts, "served_by_b": served_b, "burst_detail": burst_detail,
        "gpu_burst": gpu_window(start + burst0, start + burst1),
        "prestart": next((e for e in events if e["event"] == "replica_prestarted"), None),
    })  # fmt: skip


def spread(v):
    v = [x for x in v if x is not None]
    return None if not v else [round(statistics.median(v), 3), round(min(v), 3), round(max(v), 3)]


by = {}
for cond in ("static", "auto", "static2"):
    rs = [r for r in trials if r["condition"] == cond]
    if not rs:
        continue
    by[cond] = {
        "n": len(rs),
        **{f"goodput_phase{i}": spread([r["goodput"][i] for r in rs]) for i in range(3)},
        **{f"p50_tpot_phase{i}": spread([r["p50_tpot"][i] for r in rs]) for i in range(3)},
        **{f"p50_ttft_phase{i}": spread([r["p50_ttft"][i] for r in rs]) for i in range(3)},
        **{f"p99_ttft_phase{i}": spread([r["p99_ttft"][i] for r in rs]) for i in range(3)},
        **{k: spread([r["reactions"][k] for r in rs]) for k in rs[0]["reactions"]},
        "completed": [r["accounting"]["completed"] for r in rs], "failed": [r["accounting"]["failed"] for r in rs],
        "retried": [r["accounting"]["retried"] for r in rs], "planned": [r["accounting"]["planned"] for r in rs],
        "balanced": all(r["accounting"]["balanced"] for r in rs), "sender_valid": all(r["sender_valid"] for r in rs),
        "served_by_b": [r["served_by_b"] for r in rs], "b_starts": [r["b_starts"] for r in rs],
        "burst_ok": [r["burst_detail"]["ok"] for r in rs],
        "burst_ttft_within_slo": [r["burst_detail"]["ttft_within_slo"] for r in rs],
        "burst_tpot_within_slo": [r["burst_detail"]["tpot_within_slo"] for r in rs],
        "gpu1_burst_util": [(r["gpu_burst"][1] or {}).get("util_mean") for r in rs],
        "gpu0_burst_util": [(r["gpu_burst"][0] or {}).get("util_mean") for r in rs],
    }  # fmt: skip
print(json.dumps({"slo": s["slo"], "order": s["meta"]["order"], "by_condition": by, "trials": trials,
                  "peak_owned_rss_bytes": s.get("peak_owned_rss_bytes"), "stop_reason": s.get("stop_reason"),
                  "replica_a_start_to_ready_s": s["meta"]["replica_a_start_to_ready_s"],
                  "loadavg": [s["meta"].get("loadavg_start"), s.get("loadavg_end")]}, indent=1))
