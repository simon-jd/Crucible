import pytest

from serving.baseline import vllm_baseline as vb


def record(ttft=10.0, tpot=5.0, status="ok", sent=0.0, sched=0.0, out=8, done=100.0, error=""):
    return {
        "ScheduledMs": sched,
        "SentMs": sent,
        "TTFTms": ttft,
        "TPOTms": tpot,
        "CompletedMs": done,
        "OutputTokens": out,
        "Coalesced": False,
        "Status": status,
        "Error": error,
    }


def replay_output(records, duration_ms=1000.0, valid=True, aborted=False):
    return {"Run": {"Records": records, "SenderValid": valid, "Aborted": aborted, "DurationMs": duration_ms}}


def test_eval_order_rotates_levels_so_each_round_has_a_different_first():
    assert vb.eval_order(3, (1, 2, 4)) == [[1, 2, 4], [2, 4, 1], [4, 1, 2]]


def test_nearest_rank_p99_matches_go_calibrate():
    values = list(range(1, 101))
    assert vb.nearest_rank(values, 0.99) == 99
    assert vb.nearest_rank([7.0], 0.99) == 7.0
    with pytest.raises(ValueError):
        vb.nearest_rank([], 0.99)


def test_calibration_pools_three_sender_valid_runs_and_ignores_failures():
    runs = [
        replay_output([record(ttft=t, tpot=t / 10) for t in (10, 20, 30)]),
        replay_output([record(ttft=40, tpot=4), record(status="stream_error", ttft=-1, tpot=-1)]),
        replay_output([record(ttft=50, tpot=5), record(out=1, tpot=-1)]),
    ]
    slo = vb.calibrate(runs)
    assert slo["ttft_ms"] == 50 and slo["tpot_ms"] == 5
    assert slo["samples"] == 5 and slo["failed"] == 1 and slo["undefined"] == 1
    runs[1]["Run"]["SenderValid"] = False
    with pytest.raises(ValueError, match="sender-valid"):
        vb.calibrate(runs)


def test_trial_summary_counts_timeouts_late_sends_and_goodput():
    records = [
        record(ttft=10, tpot=5, done=900),
        record(ttft=999, tpot=5),  # misses TTFT SLO
        record(status="transport_error", error="context deadline exceeded", ttft=-1, tpot=-1, sent=1, out=0),
        record(status="stream_error", error="stream read: context deadline exceeded", tpot=-1),  # 8 partial
        record(status="http_error", error="HTTP 400", ttft=-1, tpot=-1, sent=7, sched=1, out=0),
    ]
    s = vb.summarize(replay_output(records, duration_ms=2000), {"ttft_ms": 100, "tpot_ms": 10})
    assert s["requests"] == 5 and s["ok"] == 2 and s["good"] == 1
    assert s["timeouts"] == 2 and s["other_failures"] == 1
    assert s["late_sends"] == 1 and s["sender_valid"] is True
    assert s["goodput_rps"] == pytest.approx(0.5)
    assert s["output_tokens"] == 8 * 3


def test_summary_without_slo_still_reports_latency_and_failures():
    s = vb.summarize(replay_output([record(ttft=10), record(ttft=30)]), None)
    assert s["good"] is None and s["goodput_rps"] is None
    assert s["ttft_ms"]["p50"] == 10 and s["ttft_ms"]["max"] == 30


def test_aggregate_reports_median_range_and_paired_difference_from_valid_rounds_only():
    trials = []
    for rnd, goodputs in enumerate(({1: 1.0, 2: 1.5, 4: 0.5}, {1: 1.2, 2: 1.4, 4: 0.7}, {1: 0.9, 2: 1.9, 4: 9.0})):
        for level, g in goodputs.items():
            valid = not (rnd == 2 and level == 4)  # a sender-invalid round never counts
            trials.append({"round": rnd, "level": level, "sender_valid": valid, "goodput_rps": g})
    agg = vb.aggregate(trials, (1, 2, 4))
    assert agg["1"]["goodput_rps"] == {"median": 1.0, "min": 0.9, "max": 1.2, "valid_rounds": 3}
    assert agg["4"]["goodput_rps"]["valid_rounds"] == 2 and agg["4"]["invalid_rounds"] == 1
    assert agg["2"]["paired_minus_1x"] == {"median": pytest.approx(0.5), "min": pytest.approx(0.2), "max": pytest.approx(1.0)}


def test_vllm_metrics_parse_running_and_waiting():
    text = (
        "# HELP vllm:num_requests_running x\n"
        'vllm:num_requests_running{engine="0",model_name="m"} 3.0\n'
        'vllm:num_requests_waiting{engine="0",model_name="m"} 2.0\n'
    )
    assert vb.queue_depth(text) == 5


def test_server_command_respects_shared_gpu_rules():
    cmd = vb.server_command("/snap", "Qwen/Qwen3-8B-AWQ", port=8123, max_model_len=8192, max_num_seqs=64)
    joined = " ".join(cmd)
    assert "--gpu-memory-utilization 0.8" in joined
    assert "--seed 17" in joined and "--max-model-len 8192" in joined
    assert "--served-model-name Qwen/Qwen3-8B-AWQ" in joined and "--host 127.0.0.1" in joined
    assert "--no-enable-prefix-caching" in joined
    assert not any(flag in cmd for flag in ("--daemon", "-d"))
