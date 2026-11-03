import json

import pytest

from cluster.drain.controller import Action, SchedulerExecutor
from cluster.drain.metrics import StepTimeHook


def actions(nodes):
    return [
        Action(node, kind, "simultaneous fault") for node in nodes for kind in ("drain", "requeue")
    ]


class SlurmRunner:
    def __init__(self, jobs, fail_node=None, unowned=None):
        self.jobs, self.fail_node, self.unowned = jobs, fail_node, unowned
        self.calls, self.requeued = [], set()

    def __call__(self, argv, stdin=None):
        self.calls.append(argv)
        if argv[1] == "update":
            if argv[2] == f"nodename={self.fail_node}":
                raise RuntimeError("isolation refused")
            return ""
        if argv[1] == "requeue":
            self.requeued.add(argv[2])
            return ""
        if argv[2] == "hostnames":
            return argv[3].replace(",", "\n")
        job_id = argv[3]
        return json.dumps(
            {
                "jobs": [
                    {
                        "job_id": int(job_id),
                        "comment": "other-owner" if job_id == self.unowned else "crucible:owner",
                        "nodes": self.jobs[job_id],
                        "job_state": ["PENDING" if job_id in self.requeued else "RUNNING"],
                    }
                ]
            }
        )


class K8sRunner:
    def __init__(self, nodes, fail_node=None, unowned=None, pod_jobs=None):
        self.nodes, self.fail_node, self.unowned = nodes, fail_node, unowned
        self.pod_jobs = pod_jobs or {name: name for name in nodes}
        self.calls, self.evicted = [], set()
        self.eviction_bodies = []

    def pod(self, name, replacement=False):
        job = self.pod_jobs[name]
        return {
            "metadata": {
                "uid": f"{name}-{'new' if replacement else 'old'}",
                "ownerReferences": [
                    {"uid": f"{job}-job", "kind": "Job", "name": job, "controller": True}
                ],
            },
            "spec": {"nodeName": "healthy" if replacement else self.nodes[name]},
            "status": {"phase": "Running"},
        }

    def __call__(self, argv, stdin=None):
        self.calls.append(argv)
        if "cordon" in argv or "taint" in argv:
            if "taint" in argv and self.fail_node in argv:
                raise RuntimeError("isolation refused")
            return ""
        if "--raw" in argv:
            body = json.loads(stdin)
            self.eviction_bodies.append(body)
            self.evicted.add(body["metadata"]["name"])
            return ""
        kind, name = argv[4:6]
        if kind == "job":
            return json.dumps(
                {
                    "metadata": {
                        "uid": f"{name}-job",
                        "labels": {
                            "crucible/owner": "other-owner" if name == self.unowned else "owner"
                        },
                    },
                    "spec": {},
                }
            )
        if kind == "pod":
            return json.dumps(self.pod(name))
        name = argv[6].split("=", 1)[1].removesuffix("-job")
        return json.dumps(
            {
                "items": [
                    self.pod(p, p in self.evicted) for p, j in self.pod_jobs.items() if j == name
                ]
            }
        )


def k8s_jobs(nodes):
    return [{"namespace": "lab", "job": f"train-{n}", "pod": f"train-{n}"} for n in nodes]


@pytest.mark.parametrize("nodes", [("a", "b"), ("a", "b", "c")])
def test_slurm_batch_isolates_all_faults_before_first_requeue_and_routes_jobs(nodes):
    jobs = {str(42 + i): n for i, n in enumerate(nodes)}
    runner = SlurmRunner(jobs)
    results = SchedulerExecutor("slurm", set(nodes), "owner", dry_run=False, runner=runner).execute(
        actions(nodes), [{"id": job_id} for job_id in jobs]
    )
    mutation = [c[1] for c in runner.calls if c[1] in ("update", "requeue")]
    assert mutation == ["update"] * len(nodes) + ["requeue"] * len(nodes)
    assert all(r["status"] == "requeued" for r in results if r["kind"] == "requeue")
    for result in results:
        if result["kind"] == "requeue":
            assert len(result["jobs"]) == 1
            assert jobs[str(result["jobs"][0]["job"]["id"])] == result["node"]


def test_spanning_slurm_job_requeued_once_and_associated_with_both_nodes():
    runner = SlurmRunner({"42": "a,b"})
    results = SchedulerExecutor("slurm", {"a", "b"}, "owner", dry_run=False, runner=runner).execute(
        actions(("a", "b")), [{"id": "42"}, {"id": "42"}]
    )
    assert sum(c[:2] == ["scontrol", "requeue"] for c in runner.calls) == 1
    requeues = [r for r in results if r["kind"] == "requeue"]
    assert all(r["status"] == "requeued" for r in requeues)
    assert all(r["jobs"][0]["nodes"] == ["a", "b"] for r in requeues)


@pytest.mark.parametrize("scheduler", ["slurm", "k8s"])
def test_any_batch_isolation_failure_blocks_all_requeue_attempts(scheduler):
    nodes = ("a", "b")
    runner = (
        SlurmRunner({"42": "a", "43": "b"}, fail_node="b")
        if scheduler == "slurm"
        else K8sRunner({f"train-{n}": n for n in nodes}, fail_node="b")
    )
    jobs = [{"id": "42"}, {"id": "43"}] if scheduler == "slurm" else k8s_jobs(nodes)
    results = SchedulerExecutor(
        scheduler, set(nodes), "owner", dry_run=False, runner=runner
    ).execute(actions(nodes), jobs)
    assert not any(c[:2] == ["scontrol", "requeue"] or "--raw" in c for c in runner.calls)
    assert all(
        r["status"] == "failed" and "batch isolation" in r["error"]
        for r in results
        if r["kind"] == "requeue"
    )
    assert any(not e["ok"] for r in results for e in r["evidence"])
    assert all(
        j["status"] == "not_executed" for r in results if r["kind"] == "requeue" for j in r["jobs"]
    )


def test_k8s_batch_isolates_all_nodes_before_eviction_and_routes_explicit_pods():
    nodes = ("a", "b", "c")
    runner = K8sRunner({f"train-{n}": n for n in nodes})
    results = SchedulerExecutor("k8s", set(nodes), "owner", dry_run=False, runner=runner).execute(
        actions(nodes), k8s_jobs(nodes)
    )
    isolation = [i for i, c in enumerate(runner.calls) if "cordon" in c or "taint" in c]
    evictions = [i for i, c in enumerate(runner.calls) if "--raw" in c]
    assert len(isolation) == 6 and len(evictions) == 3 and max(isolation) < min(evictions)
    assert all(
        r["status"] == "requeued" and len(r["jobs"]) == 1 for r in results if r["kind"] == "requeue"
    )


@pytest.mark.parametrize("scheduler", ["slurm", "k8s"])
def test_job_partition_never_hides_true_ownership_failure(scheduler):
    nodes = ("a", "b")
    runner = (
        SlurmRunner({"42": "a", "43": "b"}, unowned="43")
        if scheduler == "slurm"
        else K8sRunner({f"train-{n}": n for n in nodes}, unowned="train-b")
    )
    jobs = [{"id": "42"}, {"id": "43"}] if scheduler == "slurm" else k8s_jobs(nodes)
    results = SchedulerExecutor(
        scheduler, set(nodes), "owner", dry_run=False, runner=runner
    ).execute(actions(nodes), jobs)
    assert any(
        j["status"] == "failed" and "owned" in j["error"]
        for r in results
        for j in r.get("jobs", [])
    )
    assert not any(c[:3] == ["scontrol", "requeue", "43"] for c in runner.calls)
    assert "train-b" not in getattr(runner, "evicted", set())


@pytest.mark.parametrize("scheduler", ["slurm", "k8s"])
def test_malformed_job_reference_is_visible_failure(scheduler):
    runner = SlurmRunner({}) if scheduler == "slurm" else K8sRunner({})
    results = SchedulerExecutor(scheduler, {"a"}, "owner", dry_run=False, runner=runner).execute(
        actions(("a",)), [{}]
    )
    assert results[-1]["status"] == "failed"
    assert results[-1]["jobs"][0]["status"] == "failed"


def test_prometheus_labels_escape_protocol_characters_and_keep_unicode(tmp_path):
    hook = StepTimeHook('节点\\"\n', '实验\\"\n', tmp_path / "step.prom", clock=lambda: 1000)
    assert hook.observe(50, 1)
    body = hook.path.read_text()
    assert 'node="节点\\\\\\"\\n",run_id="实验\\\\\\"\\n"' in body
    assert "\\u" not in body
    assert len(body.splitlines()) == 6


def test_multiple_explicit_pods_of_same_job_each_evicted_once_with_uid():
    runner = K8sRunner(
        {"train-1": "a", "train-2": "b"}, pod_jobs={"train-1": "train", "train-2": "train"}
    )
    jobs = [{"namespace": "lab", "job": "train", "pod": p} for p in runner.nodes]
    jobs.append(jobs[0].copy())
    results = SchedulerExecutor("k8s", {"a", "b"}, "owner", dry_run=False, runner=runner).execute(
        actions(("a", "b")), jobs
    )
    assert len([c for c in runner.calls if "--raw" in c]) == 2
    assert {b["deleteOptions"]["preconditions"]["uid"] for b in runner.eviction_bodies} == {
        "train-1-old",
        "train-2-old",
    }
    assert all(
        r["status"] == "requeued" and len(r["jobs"]) == 1 for r in results if r["kind"] == "requeue"
    )


def test_k8s_replacement_on_other_known_fault_node_is_not_confirmed():
    class WrongReplacement(K8sRunner):
        def pod(self, name, replacement=False):
            pod = super().pod(name, replacement)
            if replacement:
                pod["spec"]["nodeName"] = "b"
            return pod

    runner = WrongReplacement({"train-a": "a"})
    results = SchedulerExecutor("k8s", {"a", "b"}, "owner", dry_run=False, runner=runner).execute(
        [
            Action("a", "requeue", "fault"),
            Action("b", "drain", "fault"),
            Action("a", "drain", "fault"),
        ],
        k8s_jobs(("a",)),
    )
    assert results[0]["status"] == "failed"
    assert "replacement not observed" in results[0]["jobs"][0]["error"]


@pytest.mark.parametrize(
    "label", ["tab\t", "nul\0", "carriage\r", "del\x7f", "c1\x85", "surrogate\ud800", 42]
)
@pytest.mark.parametrize("field", ["node", "run_id"])
def test_prometheus_labels_reject_unsupported_controls_and_non_strings(tmp_path, label, field):
    kwargs = {"node": "a", "run_id": "run", "path": tmp_path / "step.prom", field: label}
    with pytest.raises(ValueError):
        StepTimeHook(**kwargs)
    assert not kwargs["path"].exists()
