from cluster.drain.controller import Controller, NodeMetrics
from cluster.faults.events import FaultTimeline, JsonlSink, ObservedSchedulerExecutor


def test_task3_detect_drain_and_restore_use_observer_clock_and_keep_sources(tmp_path):
    now = [100.0]
    journal = tmp_path / "timeline.jsonl"
    timeline = FaultTimeline("run", clock_id="cpu", clock=lambda: now[0], emit=JsonlSink(journal))
    timeline.record("injection_confirmed", "incident", "node-a")
    now[0] = 105
    controller = Controller(clock=lambda: 200)
    actions = timeline.observe_controller(
        controller, {"node-a": NodeMetrics(199, 3, True)}, {"node-a": "incident"}
    )
    assert [a.kind for a in actions] == ["drain", "requeue"]
    calls = []

    def shim(argv, stdin=None):
        calls.append(argv)
        if argv[:2] == ["scontrol", "update"]:
            now[0] += 3
        elif argv[:2] == ["scontrol", "requeue"]:
            now[0] += 2
        if argv[:3] == ["scontrol", "show", "job"]:
            state = "PENDING" if any(c[:2] == ["scontrol", "requeue"] for c in calls) else "RUNNING"
            return (
                '{"jobs":[{"job_id":7,"comment":"crucible:test","job_state":["'
                + state
                + '"],"nodes":"node-a"}]}'
            )
        if argv[:3] == ["scontrol", "show", "hostnames"]:
            return "node-a\n"
        return ""

    executor = ObservedSchedulerExecutor(
        "slurm",
        ["node-a"],
        "test",
        dry_run=False,
        runner=shim,
        clock=lambda: now[0],
        clock_id="cpu",
    )
    start = now[0]
    results = executor.execute(actions, [{"id": 7}])
    assert now[0] == 110  # requeue receipt is later than actual isolation at 108
    assert timeline.scheduler_results(results, {"node-a": "incident"}, started_at=start)
    timeline.record("recovery_start", "incident", "node-a")
    now[0] = 114
    timeline.record("recovery_complete", "incident", "node-a")
    assert timeline.metrics("incident") == {
        "detection_seconds": 5,
        "drain_seconds": 3,
        "detection_to_isolation_seconds": 3,
        "recovery_seconds": 4,
    }
    assert timeline.events[1]["evidence"]["sample_observed_at"] == 199
    assert journal.read_text().count("\n") == 5


def test_planned_or_failed_isolation_has_no_drain_latency_and_preserves_batch_failure():
    timeline = FaultTimeline("run", clock_id="cpu", clock=lambda: 10)
    result = [
        {"node": "a", "kind": "drain", "status": "drained"},
        {"node": "b", "kind": "drain", "status": "failed", "error": "denied"},
    ]
    assert not timeline.scheduler_results(result, {"a": "one", "b": "two"}, started_at=9)
    assert timeline.metrics("one")["drain_seconds"] is None
    assert timeline.events[0]["evidence"]["batch_results"][1]["error"] == "denied"
    dry = [{"node": "a", "kind": "drain", "status": "planned"}]
    assert not timeline.scheduler_results(dry, {"a": "one"}, started_at=9)
