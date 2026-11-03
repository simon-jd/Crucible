import json

import pytest
import yaml

from cluster.drain.controller import Action, Controller, NodeMetrics, SchedulerExecutor
from cluster.drain.metrics import PrometheusSource, StepTimeHook


def sample(t=1000, **kwargs):
    return NodeMetrics(observed_at=t, xid_errors=0, link_up=True, **kwargs)


def trio(t=1000, slow=1.3, run="same-config"):
    return {
        n: sample(t, step_time=v, step_observed_at=t, run_id=run)
        for n, v in zip(("a", "b", "c"), (1, 1, slow))
    }


def test_xid_triggers_drain():
    c = Controller(clock=lambda: 1000)
    actions = c.evaluate({"a": NodeMetrics(1000, 79, True)})
    assert [a.kind for a in actions] == ["drain", "requeue"]
    assert "XID" in actions[0].reason


def test_link_down_triggers_drain():
    assert (
        Controller(clock=lambda: 1000).evaluate({"a": NodeMetrics(1000, 0, False)})[0].kind
        == "drain"
    )


def test_step_time_outlier_alerts():
    c = Controller(clock=lambda: 1100)
    for t in (1000, 1050):
        assert c.evaluate(trio(t)) == []
    actions = c.evaluate(trio(1100))
    assert len(actions) == 1 and actions[0].kind == "alert" and actions[0].node == "c"


def test_no_flapping():
    now = [1000]
    c = Controller(clock=lambda: now[0])
    bad = lambda: {"a": NodeMetrics(now[0], 79, True)}
    assert len(c.evaluate(bad())) == 2
    now[0] = 1601
    assert c.evaluate(bad()) == []  # Persistent incident never repeats.
    now[0] = 1602
    c.evaluate({"a": sample(now[0])})
    now[0] = 1603
    assert len(c.evaluate(bad())) == 2
    c.evaluate({"a": sample(1604)})  # Future sample is not recovery.
    now[0] = 1605
    assert c.evaluate(bad()) == []


def test_recovered_incident_respects_exact_cooldown():
    now = [1000]
    c = Controller(clock=lambda: now[0])
    c.evaluate({"a": NodeMetrics(1000, 1, True)})
    now[0] = 1001
    c.evaluate({"a": sample(1001)})
    now[0] = 1599
    assert c.evaluate({"a": NodeMetrics(1599, 1, True)}) == []
    now[0] = 1600
    assert len(c.evaluate({"a": NodeMetrics(1600, 1, True)})) == 2


def test_duplicate_and_missing_comparison_do_not_advance_or_reset():
    c = Controller(clock=lambda: 1100)
    c.evaluate(trio())
    for _ in range(5):
        assert c.evaluate(trio()) == []
    assert c.evaluate({"c": trio(1020)["c"]})[0].reason.startswith("step time indeterminate")
    assert c.evaluate(trio(1050)) == []
    assert c.evaluate(trio(1100))[0].node == "c"


def test_different_runs_and_unhealthy_peers_not_compared():
    data = trio()
    data["a"] = sample(step_time=1, step_observed_at=1000, run_id="other")
    assert all("indeterminate" in a.reason for a in Controller(clock=lambda: 1000).evaluate(data))


@pytest.mark.parametrize("t,xid,link", [(800, 79, True), (1001, 0, False), (1000, None, True)])
def test_unknown_health_alerts_without_drain(t, xid, link):
    actions = Controller(clock=lambda: 1000).evaluate({"a": NodeMetrics(t, xid, link)})
    assert [a.kind for a in actions] == ["alert"]


def test_restart_preserves_incident_and_step_count(tmp_path):
    path = tmp_path / "state.json"
    c = Controller(clock=lambda: 1100, state_path=path)
    c.evaluate({"bad": NodeMetrics(1000, 1, True), **trio()})
    c.evaluate(trio(1050))
    restored = Controller(clock=lambda: 1100, state_path=path)
    actions = restored.evaluate({"bad": NodeMetrics(1100, 1, True), **trio(1100)})
    assert [a.kind for a in actions] == ["alert"]
    assert actions[0].node == "c"


def test_state_write_failure_returns_no_actions_and_preserves_memory(tmp_path, monkeypatch):
    c = Controller(clock=lambda: 1000, state_path=tmp_path / "state.json")

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(c, "_save", fail)
    with pytest.raises(OSError):
        c.evaluate({"a": NodeMetrics(1000, 79, True)})
    assert c.state == {}


def test_invalid_state_fails_closed(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"schema": 99}')
    with pytest.raises(ValueError):
        Controller(state_path=path)


def test_hook_reports_only_each_50_steps_and_preserves_sample_time(tmp_path):
    path = tmp_path / "training.prom"
    hook = StepTimeHook("a", "same-config", path, clock=lambda: 1234)
    for step in range(1, 50):
        assert not hook.observe(step, 1.5)
    assert hook.observe(50, 1.5)
    assert not hook.observe(50, 9)
    body = path.read_text()
    assert 'crucible_step_time_seconds{node="a",run_id="same-config"} 1.5' in body
    assert 'crucible_step_sample_timestamp_seconds{node="a",run_id="same-config"} 1234' in body
    assert 'crucible_step_index{node="a",run_id="same-config"} 50' in body


def test_prometheus_maps_inventory_and_source_timestamps():
    def fetch(query):
        timestamp = query.startswith("timestamp(")
        name = query.removeprefix("timestamp(").removesuffix(")")
        labels = {"node": "a"}
        value = 0
        if name == "DCGM_FI_DEV_XID_ERRORS":
            labels["UUID"] = "gpu-1"
        elif name == "node_network_carrier":
            labels["device"] = "ib0"
            value = 1
        elif name == "crucible_step_time_seconds":
            labels["run_id"] = "same-config"
            value = 1.3
        elif name == "crucible_step_sample_timestamp_seconds":
            labels["run_id"] = "same-config"
            value = 950
        return {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": labels,
                        "value": [
                            1000,
                            str((980 if "crucible_step" in name else 900) if timestamp else value),
                        ],
                    }
                ],
            },
        }

    source = PrometheusSource({"a": {"gpus": ["gpu-1"], "links": ["ib0"]}}, query=fetch)
    m = source.collect()["a"]
    assert (m.observed_at, m.xid_errors, m.link_up) == (900, 0, True)
    assert (m.step_observed_at, m.step_time, m.run_id) == (950, 1.3, "same-config")
    missing = PrometheusSource({"a": {"gpus": ["gpu-2"], "links": ["ib0"]}}, query=fetch)
    assert missing.collect()["a"].xid_errors is None


def slurm_job(comment="crucible:owner", nodes="a", state="RUNNING"):
    return {"jobs": [{"job_id": 42, "comment": comment, "nodes": nodes, "job_state": [state]}]}


def test_slurm_dry_run_never_calls_runner():
    def fail(*args, **kwargs):
        raise AssertionError("external command")

    executor = SchedulerExecutor("slurm", {"a"}, "owner", runner=fail)
    results = executor.execute(
        [Action("a", "drain", "XID"), Action("a", "requeue", "XID")], [{"id": "42"}]
    )
    assert all(r["status"] == "planned" for r in results)


def test_slurm_owned_job_requeued_and_verified():
    commands = []

    def runner(argv, stdin=None):
        commands.append(argv)
        if argv[:4] == ["scontrol", "show", "job", "42"]:
            first_read = sum(c[:3] == ["scontrol", "show", "job"] for c in commands) == 1
            return json.dumps(
                slurm_job(
                    nodes="a" if first_read else "", state="RUNNING" if first_read else "PENDING"
                )
            )
        if argv[:3] == ["scontrol", "show", "hostnames"]:
            return "a\n"
        return ""

    executor = SchedulerExecutor("slurm", {"a"}, "owner", dry_run=False, runner=runner)
    result = executor.execute(
        [Action("a", "drain", "XID 79"), Action("a", "requeue", "XID 79")], [{"id": "42"}]
    )
    assert result[-1]["status"] == "requeued"
    assert ["scontrol", "update", "nodename=a", "state=drain", "reason=XID 79"] in commands
    assert ["scontrol", "requeue", "42"] in commands


def test_unowned_slurm_job_never_requeued():
    calls = []

    def runner(argv, stdin=None):
        calls.append(argv)
        return json.dumps(slurm_job(comment="other-project"))

    result = SchedulerExecutor("slurm", {"a"}, "owner", dry_run=False, runner=runner).execute(
        [Action("a", "drain", "XID"), Action("a", "requeue", "XID")], [{"id": "42"}]
    )
    assert result[-1]["status"] == "failed"
    assert not any("requeue" in c for c in calls)


def k8s_objects():
    job = {"metadata": {"uid": "j1", "labels": {"crucible/owner": "owner"}}, "spec": {}}
    pod = {
        "metadata": {
            "uid": "p1",
            "ownerReferences": [{"uid": "j1", "kind": "Job", "name": "train", "controller": True}],
        },
        "spec": {"nodeName": "a"},
    }
    return job, pod


def test_k8s_evicts_only_owned_pod_with_uid_and_verifies_replacement():
    job, pod = k8s_objects()
    calls = []

    def runner(argv, stdin=None):
        calls.append((argv, stdin))
        if "job" in argv:
            return json.dumps(job)
        if "pod" in argv:
            return json.dumps(pod)
        if "pods" in argv:
            replacement = json.loads(json.dumps(pod))
            replacement["metadata"]["uid"] = "p2"
            replacement["spec"]["nodeName"] = "b"
            replacement["status"] = {"phase": "Running"}
            return json.dumps({"items": [pod] if len(calls) < 6 else [replacement]})
        return ""

    result = SchedulerExecutor("k8s", {"a"}, "owner", dry_run=False, runner=runner).execute(
        [Action("a", "drain", "link down"), Action("a", "requeue", "link down")],
        [{"namespace": "lab", "job": "train", "pod": "train-1"}],
    )
    assert result[-1]["status"] == "requeued"
    eviction = next(stdin for argv, stdin in calls if "--raw" in argv)
    assert json.loads(eviction)["deleteOptions"]["preconditions"]["uid"] == "p1"
    assert any("cordon" in argv for argv, _ in calls)
    assert any("taint" in argv for argv, _ in calls)


@pytest.mark.parametrize("failure", ["unowned", "wrong-node", "replacement-missing", "command"])
def test_k8s_failure_never_claims_requeued(failure):
    job, pod = k8s_objects()
    if failure == "unowned":
        job["metadata"]["labels"] = {}
    if failure == "wrong-node":
        pod["spec"]["nodeName"] = "other"

    def runner(argv, stdin=None):
        if failure == "command" and "cordon" in argv:
            raise RuntimeError("denied")
        return json.dumps(job if "job" in argv else pod if "pod" in argv else {"items": [pod]})

    result = SchedulerExecutor("k8s", {"a"}, "owner", dry_run=False, runner=runner).execute(
        [Action("a", "drain", "fault"), Action("a", "requeue", "fault")],
        [{"namespace": "lab", "job": "train", "pod": "train-1"}],
    )
    assert not any(r["status"] == "requeued" for r in result)
    assert any(r["status"] == "failed" for r in result)


def test_requeue_without_successful_drain_refused():
    results = SchedulerExecutor(
        "slurm", {"a"}, "owner", dry_run=False, runner=lambda *a, **kw: ""
    ).execute([Action("a", "requeue", "fault")], [{"id": "42"}])
    assert results[0]["status"] == "failed"


def test_configs_load_and_thresholds_are_runtime_values():
    from pathlib import Path

    root = Path(__file__).parents[1]
    rules = yaml.safe_load((root / "rules.yaml").read_text())
    assert rules["step_time_ratio"] == 1.2
    assert rules["consecutive_checks"] == 3
    assert rules["drain_cooldown_seconds"] == 600
    assert rules["report_every_steps"] == 50
    prom = yaml.safe_load((root / "prometheus.yaml").read_text())
    assert prom["rule_files"] == ["prometheus-rules.yaml"]
    assert {j["job_name"] for j in prom["scrape_configs"]} == {"dcgm", "node", "training"}


def test_step_boundary_recovery_and_alert_latch():
    now = [1000]
    c = Controller(clock=lambda: now[0])
    for t in (1000, 1010, 1020):
        now[0] = t
        assert c.evaluate(trio(t, slow=1.2)) == []
    for t in (1030, 1040):
        now[0] = t
        assert c.evaluate(trio(t)) == []
    now[0] = 1050
    assert len(c.evaluate(trio(1050))) == 1
    now[0] = 1060
    assert c.evaluate(trio(1060)) == []
    now[0] = 1070
    assert c.evaluate(trio(1070, slow=1)) == []
    for t in (1080, 1090):
        now[0] = t
        assert c.evaluate(trio(t)) == []
    now[0] = 1100
    assert len(c.evaluate(trio(1100))) == 1


def test_unknown_health_does_not_rearm_fault_after_restart(tmp_path):
    now = [1000]
    path = tmp_path / "state.json"
    c = Controller(clock=lambda: now[0], state_path=path)
    c.evaluate({"a": NodeMetrics(1000, 79, True)})
    now[0] = 2000
    assert c.evaluate({"a": NodeMetrics(1001, 0, True)})[0].kind == "alert"
    restored = Controller(clock=lambda: now[0], state_path=path)
    assert restored.evaluate({"a": NodeMetrics(2000, 79, True)}) == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0])
def test_invalid_step_time_is_indeterminate(value):
    data = trio()
    data["c"] = sample(step_time=value, step_observed_at=1000, run_id="same-config")
    actions = Controller(clock=lambda: 1000).evaluate(data)
    assert len(actions) == 3
    assert all("indeterminate" in a.reason for a in actions)


def test_default_k8s_dry_run_never_executes():
    def fail(*args, **kwargs):
        raise AssertionError("external command")

    results = SchedulerExecutor("k8s", {"a"}, "owner", runner=fail).execute(
        [Action("a", "drain", "fault"), Action("a", "requeue", "fault")], []
    )
    assert [r["status"] for r in results] == ["planned", "planned"]


def test_executor_retains_partial_failure_and_blocks_requeue():
    def runner(argv, stdin=None):
        if "taint" in argv:
            raise RuntimeError("simulated admission failure")
        return "cordoned"

    results = SchedulerExecutor("k8s", {"a"}, "owner", dry_run=False, runner=runner).execute(
        [Action("a", "drain", "fault"), Action("a", "requeue", "fault")], []
    )
    assert all(r["status"] == "failed" for r in results)
    assert [e["ok"] for e in results[0]["evidence"]] == [True, False]
    assert results[1]["evidence"] == []


def test_slurm_command_success_without_pending_is_failure():
    def runner(argv, stdin=None):
        if "job" in argv:
            return json.dumps(slurm_job())
        return "a\n"

    results = SchedulerExecutor("slurm", {"a"}, "owner", dry_run=False, runner=runner).execute(
        [Action("a", "drain", "fault"), Action("a", "requeue", "fault")], [{"id": "42"}]
    )
    assert results[-1]["status"] == "failed"
    assert "not confirmed" in results[-1]["jobs"][0]["error"]


def test_unmanaged_node_has_no_commands():
    def fail(*args, **kwargs):
        raise AssertionError("external command")

    result = SchedulerExecutor("slurm", {"a"}, "owner", dry_run=False, runner=fail).execute(
        [Action("other", "drain", "fault")], []
    )
    assert result[0]["status"] == "failed" and result[0]["evidence"] == []


def test_prometheus_http_uses_one_instant_and_missing_inventory_is_unknown(monkeypatch):
    import io
    from urllib.parse import parse_qs, urlparse

    from cluster.drain import metrics

    queries = []

    def urlopen(url, timeout):
        assert timeout == 10
        queries.append(parse_qs(urlparse(url).query))
        return io.StringIO(
            json.dumps({"status": "success", "data": {"resultType": "vector", "result": []}})
        )

    monkeypatch.setattr(metrics, "urlopen", urlopen)
    source = PrometheusSource({"a": {"gpus": ["gpu1"], "links": ["ib0"]}})
    data = source.collect()
    assert len(queries) == 8 and len({q["time"][0] for q in queries}) == 1
    actions = Controller().evaluate(data)
    assert len(actions) == 1 and actions[0].kind == "alert"


def test_source_failed_response_is_not_treated_as_empty_health():
    source = PrometheusSource(
        {"a": {"gpus": ["gpu1"], "links": ["ib0"]}}, query=lambda q: {"status": "error"}
    )
    with pytest.raises(ValueError, match="successful instant vector"):
        source.collect()


def test_hook_failure_does_not_mark_step_published(tmp_path, monkeypatch):
    from cluster.drain import metrics

    hook = StepTimeHook("a", "same-config", tmp_path / "step.prom")

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(metrics, "atomic_write", fail)
    with pytest.raises(OSError):
        hook.observe(50, 1)
    assert hook.last_step == 0


def test_new_fault_not_hidden_by_unchanged_oldest_health_timestamp():
    c = Controller(clock=lambda: 1050)
    assert c.evaluate({"a": NodeMetrics(1000, 0, True)}) == []
    # NIC scrape remains at 1000 while GPU has a newly observed XID at 1050.
    # Inventory adapter retains min=1000 for conservative health freshness.
    actions = c.evaluate({"a": NodeMetrics(1000, 79, True)})
    assert [a.kind for a in actions] == ["drain", "requeue"]


def test_hook_prometheus_controller_offline_pipeline(tmp_path):
    now = [1000]
    nodes = ("a", "b", "c")
    hooks = {
        n: StepTimeHook(n, "same-config", tmp_path / f"{n}.prom", clock=lambda: now[0])
        for n in nodes
    }
    inventory = {n: {"gpus": [f"gpu-{n}"], "links": ["ib0"]} for n in nodes}

    def query(expression):
        timestamp = expression.startswith("timestamp(")
        name = expression.removeprefix("timestamp(").removesuffix(")")
        rows = []
        for node in nodes:
            labels = {"node": node}
            if name == "DCGM_FI_DEV_XID_ERRORS":
                labels["UUID"], value = f"gpu-{node}", 0
            elif name == "node_network_carrier":
                labels["device"], value = "ib0", 1
            else:
                labels["run_id"] = "same-config"
                line = next(
                    line
                    for line in hooks[node].path.read_text().splitlines()
                    if line.startswith(name + "{")
                )
                value = float(line.rsplit(" ", 1)[1])
            rows.append({"metric": labels, "value": [now[0], str(now[0] if timestamp else value)]})
        return {"status": "success", "data": {"resultType": "vector", "result": rows}}

    source = PrometheusSource(inventory, query=query)
    controller = Controller(clock=lambda: now[0], state_path=tmp_path / "state.json")
    for i, t in enumerate((1000, 1050, 1100), 1):
        now[0] = t
        for node in nodes:
            assert hooks[node].observe(i * 50, 1.3 if node == "c" else 1)
        actions = controller.evaluate(source.collect())
        if i < 3:
            assert actions == []
        else:
            assert [(a.node, a.kind) for a in actions] == [("c", "alert")]
        assert controller.evaluate(source.collect()) == []
    result = SchedulerExecutor("k8s", set(nodes), "owner").execute(actions, [])
    assert result[0]["status"] == "alert"
