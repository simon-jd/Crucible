"""Offline E4 analysis: one SLO from the colocated layout's calibration runs, goodput for every layout.

  python -m serving.pd.analyze --slo-from colo=<dir> --layout colo=<dir> --layout dp2=<dir> ... \\
      --output summary.json [--markdown table.md]

Each <dir> is one layout's output from scripts/snowflake/payloads/e4_pd.py (summary.json plus the
raw replay JSON of every trial). The SLO is the Task 12 rule (pooled nearest-rank p99 TTFT and p99
TPOT over successful >=2-token requests of three sender-valid 1x calibration runs), computed with
serving.baseline.vllm_baseline.calibrate; goodput, latency and the round aggregation reuse the same
module, so the definitions match docs/evidence/serving-baseline.md.
"""

import argparse
import json
import statistics
from pathlib import Path

from serving.baseline import vllm_baseline as vb


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _dist(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"n": len(values), "p50": vb.nearest_rank(values, 0.5), "p90": vb.nearest_rank(values, 0.9),
            "max": max(values), "mean": statistics.fmean(values)}


def kv_trial_stats(proxy_recs, kv_rows, t0, t1):
    """KV transfer and proxy-phase statistics for the requests that arrived in [t0, t1]."""
    recs = [r for r in proxy_recs if t0 <= r["t_arrive"] <= t1]
    pd = [r for r in recs if r.get("mode") == "pd"]
    out = {"proxy_requests": len(recs), "proxy_errors": sum("error" in r for r in recs)}
    if not pd:
        return out
    joined = [(r, kv_rows[r["id"]]) for r in pd if r.get("id") in kv_rows]
    send_bytes = [k.get("send_bytes") for _, k in joined]
    send_s = [k.get("send_s") for _, k in joined]
    total_bytes = sum(b for b in send_bytes if b)
    total_send_s = sum(s for s in send_s if s)
    out.update({
        "pd_requests": len(pd),
        "with_kv": len(joined),
        "kv_bytes_total": total_bytes,
        "kv_bytes": _dist(send_bytes),
        "send_span_s": _dist([k.get("send_span_s") for _, k in joined]),
        "send_s": _dist(send_s),
        "recv_s": _dist([k.get("recv_s") for _, k in joined]),
        "decode_load_wait_s": _dist([k.get("load_wait_s") for _, k in joined]),
        "send_GBps": total_bytes / total_send_s / 1e9 if total_send_s else None,
        "pooled_tensors": sum(k.get("pooled", 0) for _, k in joined),
        "load_misses": sum(k.get("load_misses", 0) for _, k in joined),
        "send_failed": sum(k.get("send_failed", 0) for _, k in joined),
        "admit_wait_s": _dist([r["t_admit"] - r["t_arrive"] for r in pd if "t_admit" in r]),
        "prefill_s": _dist([r["t_prefill_done"] - r["t_admit"] for r in pd if "t_prefill_done" in r]),
        "decode_first_byte_s": _dist([r["t_first_byte"] - r["t_prefill_done"] for r in pd
                                      if "t_first_byte" in r and "t_prefill_done" in r]),
    })  # fmt: skip
    return out


def load_layout(path):
    path = Path(path)
    return json.loads((path / "summary.json").read_text()), path


def slo_from(path):
    summary, path = load_layout(path)
    runs = [json.loads((path / t["file"]).read_text()) for t in summary["calibration"]]
    valid = [r for r in runs if r["Run"]["SenderValid"] and not r["Run"]["Aborted"]]
    if len(valid) < 3:
        raise ValueError(f"{path}: only {len(valid)} sender-valid calibration runs")
    return vb.calibrate(valid)


def evaluate(path, slo):
    summary, path = load_layout(path)
    levels = tuple(summary["levels"])
    trials = []
    for t in summary["evaluation"]:
        data = json.loads((path / t["file"]).read_text())
        s = vb.summarize(data, slo)
        trials.append({"file": t["file"], "round": t["round"], "level": t["level"], **s, "kv": t.get("kv")})
    per_level = {}
    for level in levels:
        ts = [t for t in trials if t["level"] == level]
        if not ts:
            continue
        med = lambda xs: statistics.median(xs) if xs else None
        per_level[str(level)] = {
            "rounds": len(ts),
            "sender_valid": sum(t["sender_valid"] for t in ts),
            "good": [t["good"] for t in ts],
            "requests": ts[0]["requests"],
            "ok": [t["ok"] for t in ts],
            "timeouts": [t["timeouts"] for t in ts],
            "other_failures": [t["other_failures"] for t in ts],
            "output_tok_per_s": med([t["output_tok_per_s"] for t in ts]),
            "completed_rps": med([t["completed_rps"] for t in ts]),
            "ttft_p50_ms": med([t["ttft_ms"]["p50"] for t in ts if t["ttft_ms"]]),
            "ttft_p99_ms": med([t["ttft_ms"]["p99"] for t in ts if t["ttft_ms"]]),
            "tpot_p50_ms": med([t["tpot_ms"]["p50"] for t in ts if t["tpot_ms"]]),
            "tpot_p99_ms": med([t["tpot_ms"]["p99"] for t in ts if t["tpot_ms"]]),
            "kv_bytes_p50": med([t["kv"]["kv_bytes"]["p50"] for t in ts if t.get("kv") and t["kv"].get("kv_bytes")]),
            "kv_send_span_p50_s": med([t["kv"]["send_span_s"]["p50"] for t in ts
                                       if t.get("kv") and t["kv"].get("send_span_s")]),
            "kv_send_GBps": med([t["kv"]["send_GBps"] for t in ts if t.get("kv") and t["kv"].get("send_GBps")]),
            "decode_load_wait_p50_s": med([t["kv"]["decode_load_wait_s"]["p50"] for t in ts
                                           if t.get("kv") and t["kv"].get("decode_load_wait_s")]),
        }  # fmt: skip
    return {"layout": summary["layout"], "stop_reason": summary["stop_reason"], "trials": trials,
            "aggregate": vb.aggregate(trials, levels), "per_level": per_level,
            "server_args": summary.get("server_args"), "startup_s": summary.get("startup_s")}


def _fmt(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def markdown(result):
    slo = result["slo"]
    lines = [
        (f"SLO (colocated calibration, {slo['samples']} samples): TTFT <= {slo['ttft_ms']:.1f} ms, "
         f"TPOT <= {slo['tpot_ms']:.2f} ms"),
        "",
        ("| Layout | Level | Goodput req/s median [min, max] | Level - 1x | Good / sent | OK | Timeouts | "
         "p50 / p99 TTFT ms | p50 / p99 TPOT ms | Output tok/s |"),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for lay in result["layouts"]:
        for level, pl in lay["per_level"].items():
            agg = lay["aggregate"].get(level, {})
            g = agg.get("goodput_rps")
            d = agg.get("paired_minus_1x")
            good = f"{min(pl['good'])}–{max(pl['good'])}" if min(pl["good"]) != max(pl["good"]) else f"{pl['good'][0]}"
            ok = f"{min(pl['ok'])}–{max(pl['ok'])}" if min(pl["ok"]) != max(pl["ok"]) else f"{pl['ok'][0]}"
            to = f"{min(pl['timeouts'])}–{max(pl['timeouts'])}" if min(pl["timeouts"]) != max(pl["timeouts"]) \
                else f"{pl['timeouts'][0]}"
            lines.append(
                f"| {lay['layout']} | {level}x | "
                + (f"{g['median']:.3f} [{g['min']:.3f}, {g['max']:.3f}]" if g else "—") + " | "
                + (f"{d['median']:+.3f} [{d['min']:+.3f}, {d['max']:+.3f}]" if d else "—") + " | "
                + f"{good}/{pl['requests']} | {ok} | {to} | "
                + f"{_fmt(pl['ttft_p50_ms'], 0)} / {_fmt(pl['ttft_p99_ms'], 0)} | "
                + f"{_fmt(pl['tpot_p50_ms'])} / {_fmt(pl['tpot_p99_ms'])} | {_fmt(pl['output_tok_per_s'], 0)} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slo-from", required=True, help="dir of the layout whose calibration defines the SLO")
    p.add_argument("--layout", action="append", default=[], help="NAME=DIR, in table order")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--markdown", type=Path)
    a = p.parse_args(argv)
    slo = slo_from(a.slo_from.split("=", 1)[-1])
    result = {"slo": slo, "slo_source": a.slo_from, "layouts": []}
    for item in a.layout:
        name, _, d = item.partition("=")
        lay = evaluate(d, slo)
        lay["name"] = name
        result["layouts"].append(lay)
    a.output.write_text(json.dumps(result, indent=1) + "\n")
    if a.markdown:
        a.markdown.write_text(markdown(result))
    print(markdown(result))


if __name__ == "__main__":
    main()
