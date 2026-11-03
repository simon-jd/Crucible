"""Per-phase goodput, base-phase SLO and scale reaction times for one trial."""

import math

from serving.localscale.trace import phase_of

SLO_QUANTILE = 0.99


def nearest_rank(values, q):
    """Same rank as replay.Calibrate and the T12 baseline: sorted[ceil(q*n)-1]."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _eligible(r):
    return r["Status"] == "ok" and r["OutputTokens"] >= 2 and r["TTFTms"] >= 0 and r["TPOTms"] >= 0


def _dist(values):
    if not values:
        return None
    return {"p50": nearest_rank(values, 0.5), "p99": nearest_rank(values, 0.99), "max": max(values)}


def slo_from_base_phase(record_sets, phases):
    """Pooled p99 TTFT/TPOT over phase-0 (1x) successes of the given trials."""
    ttft, tpot = [], []
    for records in record_sets:
        for r in records:
            if phase_of(r["ScheduledMs"], phases) == 0 and _eligible(r):
                ttft.append(r["TTFTms"])
                tpot.append(r["TPOTms"])
    return {
        "ttft_ms": nearest_rank(ttft, SLO_QUANTILE),
        "tpot_ms": nearest_rank(tpot, SLO_QUANTILE),
        "quantile": SLO_QUANTILE,
        "samples": len(ttft),
    }


def phase_summary(records, phases, slo):
    out = []
    for index, (seconds, multiple) in enumerate(phases):
        rs = [r for r in records if phase_of(r["ScheduledMs"], phases) == index]
        ok = [r for r in rs if r["Status"] == "ok"]
        good = sum(
            _eligible(r) and r["TTFTms"] <= slo["ttft_ms"] and r["TPOTms"] <= slo["tpot_ms"] for r in rs
        )
        out.append({
            "phase": index, "seconds": seconds, "multiple": multiple, "requests": len(rs),
            "ok": len(ok), "failed": len(rs) - len(ok),
            "timeouts": sum("deadline exceeded" in r.get("Error", "") for r in rs),
            "good": good, "goodput_rps": good / seconds,
            "ttft_ms": _dist([r["TTFTms"] for r in ok if r["TTFTms"] >= 0]),
            "tpot_ms": _dist([r["TPOTms"] for r in ok if r["TPOTms"] >= 0]),
        })  # fmt: skip
    return out


def reactions(events, burst_start_s, load_drop_s):
    """Seconds from the scheduled burst start (or load drop) to each scale milestone."""
    start = next((e["t"] for e in events if e["event"] == "replay_start"), None)

    def first(kind, **match):
        for e in events:
            if e["event"] == kind and all(e.get(k) == v for k, v in match.items()):
                return e["t"] - start
        return None

    def rel(value, origin):
        return None if value is None else round(value - origin, 3)

    up = first("scale_decision", current=1, want=2)
    spawn, ready = first("replica_spawn", replica="b"), first("replica_ready", replica="b")
    return {
        "up_decision_s": rel(up, burst_start_s),
        "replica_ready_s": rel(ready, burst_start_s),
        "first_success_on_new_replica_s": rel(first("replica_first_success", replica="b"), burst_start_s),
        "spawn_to_ready_s": None if spawn is None or ready is None else round(ready - spawn, 3),
        "down_decision_after_drop_s": rel(first("scale_decision", current=2, want=1), load_drop_s),
        "stopped_after_drop_s": rel(first("replica_stopped", replica="b"), load_drop_s),
    }
