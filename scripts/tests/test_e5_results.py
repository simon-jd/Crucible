"""CPU tests for scripts/snowflake/e5_results.py (E5 summaries and redacted export)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("e5_results", ROOT / "scripts" / "snowflake" / "e5_results.py")
res = importlib.util.module_from_spec(spec)
spec.loader.exec_module(res)

QWEN = {"active_parameters": 357_898_112, "layers": 24, "heads": 14, "head_dim": 64}


def _bw(op, nbytes, gbps, world, ms=1.0):
    return {"op": op, "bytes": nbytes, "GBps": gbps, "ms": ms, "iters": 3,
            "busGBps": round(gbps * res.e5.bus_factor(op, world), 3)}


def test_bandwidth_gate_applies_the_burn_in_rule_to_the_slowest_rank():
    big = 256 << 20
    run = {"world": 4, "variant": "tuned", "pass": 0, "ranks": [
        {"check_ok": True, "bw": [_bw("all_reduce", 1 << 20, 0.1, 4), _bw("all_reduce", big, 0.7, 4),
                                  _bw("broadcast", big, 2.0, 4)]},
        {"check_ok": True, "bw": [_bw("all_reduce", 1 << 20, 0.1, 4), _bw("all_reduce", big, 0.5, 4),
                                  _bw("broadcast", big, 2.0, 4)]}]}
    gate = res.bandwidth_gate([run], theoretical_busbw_gbs=1.25)
    assert gate["thresholds"]["min_busbw_fraction"] == 0.70 and gate["thresholds"]["min_size_bytes"] == 128 << 20
    # slowest rank: 0.5 GB/s algbw -> 0.75 busbw < 0.875; the 1 MiB row is below min_size and not judged.
    assert gate["verdicts"] == {"4n/tuned/all_reduce": ["slow_bandwidth"], "4n/tuned/broadcast": ["pass"]}
    run["ranks"][1]["check_ok"] = False
    assert res.bandwidth_gate([run], 1.25)["verdicts"]["4n/tuned/broadcast"] == ["unchecked"]


def _scale_run(kind, world, rnd, tps, accum, step, local, last, alone=None):
    plan = res.e5.batch_plan(kind, world)
    split = res.e5.step_split([{"t": step, "local": local, "last": last, "opt": 0.1}], accum, alone)
    rank = {"global_tokens_per_s": tps, "split": split, "structure": QWEN, "losses": [2.0, 1.9],
            "peak_mem_gib": 12.0}
    if alone is not None:
        rank["comm_alone"] = {"bytes": 1_976_131_072, "seconds": [alone]}
    return {"kind": kind, "world": world, "round": rnd, "plan": plan, "ranks": [rank] * world}


def test_scaling_table_uses_the_one_node_run_as_reference():
    runs = []
    for rnd, jitter in ((1, 1.0), (2, 1.02), (3, 0.98)):
        runs.append(_scale_run("weak", 1, rnd, 8000 * jitter, 8, 2.05, 0.25, 0.25))
        runs.append(_scale_run("weak", 2, rnd, 8000 * jitter, 8, 4.1, 0.25, 2.3, alone=2.4))
    data, md = res.scaling_table({"runs": runs, "peak_flops": 125e12})
    one, two = data["rows"]
    assert one["world"] == 1 and one["scaling_efficiency"]["median"] == pytest.approx(1.0)
    assert two["tokens_per_s_per_gpu"]["median"] == pytest.approx(4000)
    assert two["scaling_efficiency"]["median"] == pytest.approx(0.5)
    assert two["exposed_comm_share"]["median"] == pytest.approx(2.05 / 4.1)
    assert one["exposed_comm_s"]["median"] == 0.0 and one["comm_alone_s"] is None
    assert "| weak | 2 | 32 | 8 |" in md


def test_export_redacts_and_refuses_leaks(tmp_path):
    src, dest = tmp_path / "run", tmp_path / "out"
    src.mkdir()
    (src / "report.json").write_text(json.dumps({"x": "https://1wtauthtoken:secret@Org-Acct.registry-local."
                                                      "snowflakecomputing.com/simple"}))
    (src / "instance0.log").write_text("Looking in indexes: https://1wtauthtoken:****@org-acct.registry-local."
                                       "snowflakecomputing.com/x\nplain org-acct mention\n")
    sizes = res.export(src, dest, secrets=["ORG-ACCT"])
    assert set(sizes) == {"report.json", "instance0.log"}
    text = (dest / "instance0.log").read_text() + (dest / "report.json").read_text()
    assert "org-acct" not in text.lower() and "secret" not in text and "<account>" in text
    (src / "instance1.log").write_text("token eyJhbGciOiJIUzI1NiJ9 alone is fine; mixed-Case Org-Acct is not\n")
    res.export(src, dest, secrets=["org-acct"])
    assert "Org-Acct" not in (dest / "instance1.log").read_text()
