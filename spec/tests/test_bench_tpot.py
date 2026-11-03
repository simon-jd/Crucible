"""Synthetic timings; checks the T15 TPOT and acceptance summary."""

import pytest

from spec.bench_tpot import aggregate, compare, parse_sse_line, request_record, summarize


def _rec(i, ttft, decode_s, tokens, verify=None, correct=None):
    meta = {"completion_tokens": tokens}
    if verify is not None:
        meta.update(spec_verify_ct=verify, spec_num_correct_drafts=correct)
    return request_record(f"p{i}", t_send=10.0, t_first=10.0 + ttft, t_end=10.0 + ttft + decode_s, meta=meta)


def test_tpot_excludes_first_token():
    rec = _rec(0, ttft=0.1, decode_s=0.99, tokens=100)
    assert rec["ttft_ms"] == pytest.approx(100.0)
    assert rec["tpot_ms"] == pytest.approx(10.0)
    assert _rec(1, 0.1, 0.0, 1)["tpot_ms"] is None


def test_summary_median_and_acceptance_weighted_by_verifies():
    records = [
        _rec(0, 0.1, 0.99, 100, verify=50, correct=50),
        _rec(1, 0.1, 1.98, 100, verify=25, correct=75),
        _rec(2, 0.1, 0.495, 100, verify=25, correct=25),
    ]
    s = summarize(records, wall_s=3.0)
    assert s["completed"] == 3 and s["failed"] == 0
    assert s["tpot_ms_median"] == pytest.approx(10.0)
    # 1 + (50+75+25)/(50+25+25)
    assert s["acceptance_length"] == pytest.approx(2.5)
    assert s["output_tokens"] == 300
    assert s["output_tok_per_s"] == pytest.approx(100.0)


def test_summary_counts_failures_and_baseline_has_no_acceptance():
    records = [_rec(0, 0.1, 0.99, 100), {"prompt_id": "p1", "status": "failed", "error": "x"}]
    s = summarize(records, wall_s=1.0)
    assert s["completed"] == 1 and s["failed"] == 1
    assert s["acceptance_length"] is None


def test_compare_speedup_is_base_over_spec():
    base = {"tpot_ms_median": 12.0, "output_tok_per_s": 80.0}
    spec = {"tpot_ms_median": 6.0, "output_tok_per_s": 120.0}
    assert compare(base, spec) == {"tpot_speedup": pytest.approx(2.0), "throughput_ratio": pytest.approx(1.5)}


def test_aggregate_rounds_pairs_modes_per_batch():
    def run(label, tpot, tps, accept=None):
        summary = {"tpot_ms_median": tpot, "output_tok_per_s": tps, "acceptance_length": accept,
                   "completed": 4, "failed": 0, "ttft_ms_median": 20.0}
        return {"label": label, "batches": [{"batch_size": 1, "summary": summary,
                                              "load_before": {"loadavg": [3.0, 0, 0]}}]}

    runs = [run("base-r1", 10.0, 100.0), run("eagle3-r1", 5.0, 180.0, 2.0),
            run("base-r2", 12.0, 90.0), run("eagle3-r2", 6.0, 160.0, 2.2)]
    (row,) = aggregate(runs)
    assert row["batch_size"] == 1
    assert row["base_tpot_ms"] == [10.0, 12.0] and row["eagle3_tpot_ms"] == [5.0, 6.0]
    assert row["tpot_speedup"] == [pytest.approx(2.0), pytest.approx(2.0)]
    assert row["acceptance_length"] == [2.0, 2.2]
    assert row["loadavg_1m"] == [3.0, 3.0, 3.0, 3.0]


def test_aggregate_other_mode_and_missing_batch_sizes():
    def run(label, sizes, accept=None):
        summary = {"tpot_ms_median": 10.0, "output_tok_per_s": 100.0, "acceptance_length": accept,
                   "completed": 4, "failed": 0, "ttft_ms_median": 20.0}
        return {"label": label, "batches": [{"batch_size": n, "summary": summary,
                                              "load_before": {"loadavg": [1.0, 0, 0]}} for n in sizes]}

    runs = [run("base-r1", [1, 4, 64]), run("s5k1d6-r1", [1, 64], 2.5), run("eagle3-r1", [4], 1.5)]
    rows = aggregate(runs, mode="s5k1d6")
    assert [r["batch_size"] for r in rows] == [1, 64]
    assert rows[0]["s5k1d6_tpot_ms"] == [10.0] and rows[0]["acceptance_length"] == [2.5]
    assert [r["batch_size"] for r in aggregate(runs)] == [4]


def test_parse_sse_line():
    assert parse_sse_line(b'data: {"text": "a", "meta_info": {"completion_tokens": 1}}\n') == {
        "text": "a",
        "meta_info": {"completion_tokens": 1},
    }
    assert parse_sse_line(b"data: [DONE]\n") == "done"
    assert parse_sse_line(b"\n") is None
