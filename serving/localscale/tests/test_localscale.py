import csv
import json
import threading
import urllib.error
import urllib.request

import pytest

from serving.localscale import adapter, analyze, metrics, trace

PROM = """# HELP vllm:num_requests_waiting x
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_waiting{engine="0",model_name="m"} 7.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1",model_name="m"} 10.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.5",model_name="m"} 18.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="1.0",model_name="m"} 20.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="m"} 20.0
vllm:request_success_total{engine="0",finished_reason="length",model_name="m"} 20.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="m"} 2.0
"""


def test_parse_vllm_metrics():
    s = metrics.parse(PROM)
    assert (s.waiting, s.running, s.success) == (7, 3, 22)
    assert s.ttft_buckets == {0.1: 10.0, 0.5: 18.0, 1.0: 20.0, float("inf"): 20.0}


def test_quantile_interpolates_within_bucket_like_prometheus():
    before = {0.1: 0.0, 0.5: 0.0, 1.0: 0.0, float("inf"): 0.0}
    after = {0.1: 10.0, 0.5: 18.0, 1.0: 20.0, float("inf"): 20.0}
    # rank 0.95*20 = 19 lies in (0.5, 1.0], 1 of its 2 samples: 0.5 + 0.5*0.5.
    assert metrics.quantile(0.95, before, after) == pytest.approx(0.75)
    assert metrics.quantile(0.5, before, after) == pytest.approx(0.1)
    assert metrics.quantile(0.95, after, after) is None  # no new samples is missing, not zero


def test_quantile_in_inf_bucket_returns_last_finite_bound():
    before = {1.0: 0.0, float("inf"): 0.0}
    after = {1.0: 1.0, float("inf"): 10.0}
    assert metrics.quantile(0.95, before, after) == 1.0


def test_step_burst_trace_rates_and_lengths(tmp_path):
    lengths = [(100, 10), (200, 20), (300, 30)]
    phases = [(60, 1.0), (120, 4.0), (60, 1.0)]
    rows = trace.step_burst(lengths, phases, base_rps=2.0, seed=17)
    times = [r[0] for r in rows]
    assert times == sorted(times) and times[0] == 0 and times[-1] < 240
    counts = [sum(lo <= t < hi for t in times) for lo, hi in ((0, 60), (60, 180), (180, 240))]
    # Poisson counts near 120 / 960 / 120.
    assert 90 < counts[0] < 150 and 860 < counts[1] < 1060 and 90 < counts[2] < 150
    assert [r[1:] for r in rows[:4]] == [(100, 10), (200, 20), (300, 30), (100, 10)]
    assert trace.step_burst(lengths, phases, 2.0, 17) == rows
    out = tmp_path / "t.csv"
    trace.write_azure_csv(rows, out)
    with open(out) as f:
        body = list(csv.reader(f))
    assert body[0] == ["TIMESTAMP", "ContextTokens", "GeneratedTokens"]
    assert len(body) == len(rows) + 1 and body[1][1:] == ["100", "10"]


def test_phase_of_uses_scheduled_time():
    phases = [(60, 1.0), (120, 4.0), (60, 1.0)]
    assert [trace.phase_of(ms, phases) for ms in (0, 59999, 60000, 179999, 180000, 239999)] == [
        0, 0, 1, 1, 2, 2,
    ]  # fmt: skip


def _records():
    rec = []
    for i, (ms, ttft, tpot, status) in enumerate(
        [(1000, 50, 5, "ok"), (2000, 60, 6, "ok"), (61000, 900, 9, "ok"), (62000, 80, 5, "timeout"),
         (181000, 55, 5, "ok")]
    ):  # fmt: skip
        rec.append({"ScheduledMs": ms, "SentMs": ms + 1, "TTFTms": ttft, "TPOTms": tpot, "Status": status,
                    "OutputTokens": 10, "Error": "context deadline exceeded" if status != "ok" else ""})  # fmt: skip
    return rec


def test_phase_summary_and_goodput():
    phases = [(60, 1.0), (120, 4.0), (60, 1.0)]
    slo = {"ttft_ms": 100, "tpot_ms": 8}
    out = analyze.phase_summary(_records(), phases, slo)
    assert [p["requests"] for p in out] == [2, 2, 1]
    assert [p["good"] for p in out] == [2, 0, 1]
    assert out[1]["failed"] == 1 and out[1]["goodput_rps"] == 0
    assert out[0]["goodput_rps"] == pytest.approx(2 / 60)


def test_slo_from_phase_a_only():
    phases = [(60, 1.0), (120, 4.0), (60, 1.0)]
    slo = analyze.slo_from_base_phase([_records(), _records()], phases)
    assert slo["ttft_ms"] == 60 and slo["tpot_ms"] == 6 and slo["samples"] == 4


def test_reaction_times_relative_to_burst_start():
    events = [
        {"t": 100.0, "event": "replay_start"},
        {"t": 163.5, "event": "scale_decision", "current": 1, "want": 2},
        {"t": 164.0, "event": "replica_spawn", "replica": "b"},
        {"t": 190.0, "event": "replica_ready", "replica": "b"},
        {"t": 196.0, "event": "replica_first_success", "replica": "b"},
        {"t": 520.0, "event": "scale_decision", "current": 2, "want": 1},
        {"t": 545.0, "event": "replica_stopped", "replica": "b"},
    ]
    r = analyze.reactions(events, burst_start_s=60, load_drop_s=180)
    assert r == {
        "up_decision_s": 3.5, "replica_ready_s": 30.0, "first_success_on_new_replica_s": 36.0,
        "spawn_to_ready_s": 26.0, "down_decision_after_drop_s": 240.0, "stopped_after_drop_s": 265.0,
    }  # fmt: skip
    assert analyze.reactions([{"t": 0.0, "event": "replay_start"}], 60, 180)["up_decision_s"] is None


def test_adapter_serves_router_and_autoscaler_formats():
    state = adapter.State(model="m", replicas={"a": "http://127.0.0.1:1"}, desired=1)
    state.record("a", metrics.parse(PROM), now=100.0)
    state.serving["a"] = True
    server = adapter.serve(state, port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        snap = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/snapshot/a"))
        assert snap["Queue"] == 7 and snap["Prefixes"] == [] and snap["ObservedAt"].endswith("Z")
        agg = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/aggregate"))
        assert agg["replicas"] == 1 and agg["queue_per_replica"] == 7.0
        assert agg["p95_ttft_seconds"] is None  # one scrape: no window delta yet
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/snapshot/b")
        assert err.value.code == 404
        state.serving["a"] = False  # draining replica disappears from routing
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/snapshot/a")
        assert err.value.code == 503
    finally:
        server.shutdown()
        server.server_close()


def test_aggregate_window_p95_across_replicas():
    state = adapter.State(model="m", replicas={"a": "x", "b": "y"}, desired=2, window_s=10)
    empty = metrics.Sample(ttft_buckets={0.1: 0.0, 0.5: 0.0, 1.0: 0.0, float("inf"): 0.0})
    state.record("a", empty, now=0.0)
    state.record("b", empty, now=0.0)
    state.record("a", metrics.parse(PROM), now=5.0)
    state.record("b", empty, now=5.0)
    agg = state.aggregate(now=5.0)
    assert agg["p95_ttft_seconds"] == pytest.approx(0.75)
    assert agg["queue_per_replica"] == pytest.approx((7 + 0) / 2)


def test_fault_summary_places_failures_relative_to_the_kill():
    from serving.localscale.run import fault_summary

    records = [{"ScheduledMs": ms, "Status": st} for ms, st in
               [(10000, "ok"), (31000, "http_error"), (44000, "transport_error"), (50000, "ok")]]  # fmt: skip
    events = [{"t": 100.0, "event": "replay_start"}, {"t": 130.0, "event": "decode_killed"}]
    s = fault_summary(records, events)
    assert (s["planned"], s["ok"], s["failed"]) == (4, 2, 2)
    assert s["failure_statuses"] == {"http_error": 1, "transport_error": 1}
    assert s["kill_s"] == 30.0
    assert s["first_failure_after_kill_s"] == 1.0 and s["last_failure_after_kill_s"] == 14.0
    none = fault_summary(records[:1], events[:1])
    assert none["kill_s"] is None and none["failed"] == 0


def test_fault_order_alternates_conditions_by_pair():
    from serving.localscale import run

    assert run.alternate(["kill", "none"], 3) == ["kill", "none", "none", "kill", "kill", "none"]
    assert run.alternate(["kill", "kill-retry"], 2) == ["kill", "kill-retry", "kill-retry", "kill"]


def test_router_flags_per_fault_condition():
    from serving.localscale import run

    assert run.router_flags("kill") == [] and run.router_flags("none") == []
    assert run.router_flags("kill-retry") == ["-retry-dial"]
    assert run.kills("kill") and run.kills("kill-retry") and not run.kills("none")
    # kill-client-retry: the router also resends dial errors, the client resends cut streams once.
    assert run.router_flags("kill-client-retry") == ["-retry-dial"] and run.kills("kill-client-retry")
    assert run.replay_flags("kill-client-retry") == ["-client-retries", "1"]
    assert run.replay_flags("kill-retry") == [] and run.replay_flags("none") == []


def test_fault_summary_counts_client_retries():
    from serving.localscale.run import fault_summary

    cut = {"Status": "stream_error", "Error": "stream read: unexpected EOF"}
    records = [
        {"ScheduledMs": 1000, "Status": "ok"},
        {"ScheduledMs": 29000, "Status": "ok", "Attempts": [cut, {"Status": "ok"}]},
        {"ScheduledMs": 29500, "Status": "stream_error", "Attempts": [cut, cut]},
    ]
    s = fault_summary(records, [{"t": 0.0, "event": "replay_start"}, {"t": 30.0, "event": "decode_killed"}])
    assert (s["planned"], s["ok"], s["failed"]) == (3, 2, 1)
    assert s["retried"] == 2 and s["retried_ok"] == 1
    assert s["retried_attempt_statuses"] == {"stream_error": 2}


def test_router_log_stats_parse_ejections_and_totals():
    from serving.localscale import run

    log = (
        "adapter b: invalid or stale snapshot\n"
        "2026-10-27T20:00:01.5Z eject b retried=true: dial tcp 127.0.0.1:31102: connect: connection refused\n"
        "2026-10-27T20:00:02Z eject b retried=false: unexpected EOF\n"
        "router stats ejections=2 dial_retries=1\n"
    )
    st = run.router_log_stats(log)
    assert st["ejections"] == 2 and st["dial_retries"] == 1 and st["eject_lines"] == 2
    assert st["first_eject_t"] == pytest.approx(1793131201.5)
