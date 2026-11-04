import json

import pytest

from serving.pd import analyze


def record(ttft=10.0, tpot=5.0, status="ok", out=8, error=""):
    return {"ScheduledMs": 0.0, "SentMs": 0.0, "TTFTms": ttft, "TPOTms": tpot, "CompletedMs": 100.0,
            "OutputTokens": out, "Coalesced": False, "Status": status, "Error": error}


def replay(records, duration_ms=1000.0, valid=True):
    return {"Run": {"Records": records, "SenderValid": valid, "Aborted": False, "DurationMs": duration_ms}}


def write_layout(d, name, calibration, evaluation):
    d.mkdir()
    summary = {"layout": name, "levels": [1, 2], "stop_reason": None, "calibration": [], "evaluation": []}
    for i, data in enumerate(calibration):
        f = f"calibration-r{i}-x1.json"
        (d / f).write_text(json.dumps(data))
        summary["calibration"].append({"file": f, "round": i, "level": 1})
    for rnd, level, data, kv in evaluation:
        f = f"evaluation-r{rnd}-x{level}.json"
        (d / f).write_text(json.dumps(data))
        summary["evaluation"].append({"file": f, "round": rnd, "level": level, "kv": kv})
    (d / "summary.json").write_text(json.dumps(summary))
    return d


def test_slo_comes_from_the_colocated_calibration_and_applies_to_every_layout(tmp_path):
    cal = [replay([record(ttft=t, tpot=t / 10) for t in (10, 20, 100)]) for _ in range(3)]
    colo = write_layout(tmp_path / "colo", "colo", cal, [
        (0, 1, replay([record(ttft=50, tpot=5), record(ttft=150, tpot=5)]), None),
        (0, 2, replay([record(ttft=50, tpot=5)] * 2, duration_ms=500), None),
        (1, 2, replay([record(ttft=50, tpot=5)] * 2, duration_ms=500), None),
        (1, 1, replay([record(ttft=50, tpot=5)] * 2), None)])
    kv = {"kv_bytes": {"p50": 1000}, "send_span_s": {"p50": 0.1}, "send_GBps": 2.0,
          "decode_load_wait_s": {"p50": 0.001}}
    pd = write_layout(tmp_path / "pd", "pd-shm", [], [
        (0, 1, replay([record(ttft=50, tpot=20), record(ttft=50, tpot=5)]), kv),
        (0, 2, replay([record(ttft=50, tpot=5), record(status="transport_error", error="context deadline exceeded",
                                                         ttft=-1, tpot=-1, out=0)]), kv)])
    out = tmp_path / "summary.json"
    analyze.main(["--slo-from", f"colo={colo}", "--layout", f"colo={colo}", "--layout", f"pd-shm={pd}",
                  "--output", str(out), "--markdown", str(tmp_path / "t.md")])
    res = json.loads(out.read_text())
    import gzip  # the committed evidence keeps raw replay files gzipped; results must not change

    for f in pd.glob("evaluation-*.json"):
        f.with_name(f.name + ".gz").write_bytes(gzip.compress(f.read_bytes()))
        f.unlink()
    again = analyze.evaluate(pd, res["slo"])
    assert again["per_level"] == res["layouts"][1]["per_level"]
    assert res["slo"]["ttft_ms"] == 100 and res["slo"]["tpot_ms"] == 10 and res["slo"]["samples"] == 9
    colo_r, pd_r = res["layouts"]
    assert colo_r["per_level"]["1"]["good"] == [1, 2]
    assert colo_r["aggregate"]["2"]["goodput_rps"]["median"] == 4.0
    assert colo_r["aggregate"]["2"]["paired_minus_1x"]["median"] == pytest.approx(2.5)
    assert pd_r["per_level"]["1"]["good"] == [1]  # TPOT 20 ms misses the colocated SLO
    assert pd_r["per_level"]["2"]["timeouts"] == [1]
    assert pd_r["per_level"]["2"]["kv_send_GBps"] == 2.0
    table = (tmp_path / "t.md").read_text()
    assert "TTFT <= 100.0 ms" in table and "| pd-shm | 2x |" in table


def test_slo_needs_three_sender_valid_calibrations(tmp_path):
    cal = [replay([record()]), replay([record()]), replay([record()], valid=False)]
    d = write_layout(tmp_path / "c", "colo", cal, [])
    with pytest.raises(ValueError, match="sender-valid"):
        analyze.slo_from(d)


def test_kv_trial_stats_joins_proxy_records_and_probe_rows_inside_the_trial_window():
    proxy = [
        {"mode": "pd", "id": "a", "t_arrive": 10.0, "t_admit": 10.5, "t_prefill_done": 11.0, "t_first_byte": 11.25},
        {"mode": "pd", "id": "b", "t_arrive": 12.0, "t_admit": 12.0, "t_prefill_done": 12.5, "t_first_byte": 12.75},
        {"mode": "pd", "id": "c", "t_arrive": 99.0},  # another trial
        {"mode": "pd", "id": "d", "t_arrive": 13.0, "error": "x"},  # no probe row
    ]
    rows = {"a": {"send_bytes": 100, "send_s": 0.5, "send_span_s": 0.6, "load_wait_s": 0.01, "pooled": 2},
            "b": {"send_bytes": 300, "send_s": 0.5, "send_span_s": 0.7, "load_wait_s": 0.03, "pooled": 0},
            "c": {"send_bytes": 1, "send_s": 1}}
    s = analyze.kv_trial_stats(proxy, rows, 0, 50)
    assert s["proxy_requests"] == 3 and s["pd_requests"] == 3 and s["with_kv"] == 2 and s["proxy_errors"] == 1
    assert s["kv_bytes_total"] == 400 and s["send_GBps"] == pytest.approx(400 / 1.0 / 1e9)
    assert s["admit_wait_s"]["max"] == 0.5 and s["prefill_s"]["p50"] == 0.5
    assert s["decode_first_byte_s"]["p50"] == 0.25 and s["pooled_tensors"] == 2
    assert analyze.kv_trial_stats([{"mode": "dp", "t_arrive": 1.0}], {}, 0, 5) == {"proxy_requests": 1,
                                                                                    "proxy_errors": 0}
