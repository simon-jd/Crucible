import json

import pytest

from serving.k8s import run


def test_autoscaler_decisions_parse_kubectl_timestamps():
    log = (
        '2026-10-28T03:00:01.250000000Z {"current":1,"want":2,"applied":true,"mode":"apply"}\n'
        "2026-10-28T03:00:02.000000000Z Get metrics: connection refused\n"
    )
    (d,) = run.autoscaler_decisions(log)
    assert d["want"] == 2 and d["applied"] and d["t"] == pytest.approx(1793156401.25)


def test_pod_events_name_pods_from_log_files(tmp_path):
    (tmp_path / "fake-1-0a1b-22.jsonl").write_text(json.dumps({"t": 5.0, "event": "listening"}) + "\n")
    (tmp_path / "adapter-9f-1.jsonl").write_text(
        json.dumps({"t": 6.0, "event": "serving", "replica": "fake-1"}) + "\n" + json.dumps({"t": 1.0, "event": "old"}) + "\n"
    )
    events = run.pod_events(tmp_path, since=2.0)
    assert [(e["pod"], e["event"]) for e in events] == [("fake-1", "listening"), ("adapter", "serving")]


def test_reactions_measure_from_burst_start_and_load_drop():
    start = 1000.0
    events = [
        {"t": 900.0, "pod": "fake-1", "event": "first_request"},  # previous trial
        {"t": 1070.0, "pod": "fake-1", "event": "process_start"},
        {"t": 1070.5, "pod": "fake-1", "event": "listening"},
        {"t": 1072.0, "pod": "adapter", "event": "serving", "replica": "fake-1"},
        {"t": 1075.0, "pod": "fake-1", "event": "first_request"},
        {"t": 1400.0, "pod": "fake-1", "event": "drain_start"},
        {"t": 1405.0, "pod": "fake-1", "event": "drained"},
    ]
    decisions = [{"t": 1068.0, "current": 1, "want": 2, "applied": True}, {"t": 1399.0, "current": 2, "want": 1, "applied": True}]
    r = run.reactions(events, decisions, start, burst_s=60, drop_s=180)
    assert r["up_decision_s"] == 8.0 and r["up_applied"]
    assert r["new_pod_process_start_s"] == 10.0 and r["adapter_serving_new_pod_s"] == 12.0
    assert r["new_pod_first_request_s"] == 15.0
    assert r["down_decision_after_drop_s"] == 219.0 and r["drained_after_drop_s"] == 225.0


def test_reactions_without_scaling_are_none():
    r = run.reactions([], [{"t": 1001.0, "current": 1, "want": 1, "applied": False}], 1000.0, 60, 180)
    assert r["up_decision_s"] is None and not r["up_applied"] and r["new_pod_first_request_s"] is None
