"""Health policy and explicit, offline-by-default scheduler command adapters.

One controller process owns each state file. evaluate persists intent before returning;
callers must retain executor results and reconcile failures rather than replay blindly.
"""

import copy
import json
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Literal

import yaml


def atomic_write(path, body):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@dataclass(frozen=True)
class NodeMetrics:
    observed_at: float
    xid_errors: int | None
    link_up: bool | None
    step_time: float | None = None
    step_observed_at: float | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class Action:
    node: str
    kind: Literal["drain", "requeue", "alert"]
    reason: str


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class Controller:
    def __init__(self, *, clock=time.time, state_path=None, rules_path=None):
        self.clock = clock
        self.state_path = Path(state_path) if state_path else None
        self.rules = yaml.safe_load(
            Path(rules_path or Path(__file__).with_name("rules.yaml")).read_text()
        )
        for key in (
            "step_time_ratio",
            "consecutive_checks",
            "drain_cooldown_seconds",
            "sample_freshness_seconds",
            "min_comparison_nodes",
        ):
            if not finite(self.rules[key]) or self.rules[key] <= 0:
                raise ValueError(f"invalid rule {key}")
        if (
            self.rules["step_time_ratio"] <= 1
            or any(
                type(self.rules[k]) is not int
                for k in ("consecutive_checks", "min_comparison_nodes")
            )
            or self.rules["min_comparison_nodes"] < 3
        ):
            raise ValueError("invalid comparison policy")
        self.state = {}
        if self.state_path and self.state_path.exists():
            data = json.loads(self.state_path.read_text())
            if data.get("schema") != 1 or not isinstance(data.get("nodes"), dict):
                raise ValueError("invalid controller state")
            for s in data["nodes"].values():
                if (
                    not isinstance(s, dict)
                    or set(s) != set(self._empty())
                    or type(s["incident"]) is not bool
                    or type(s["alerted"]) is not bool
                    or type(s["count"]) is not int
                    or s["count"] < 0
                    or not all(finite(s[k]) for k in ("health_at", "step_at", "last_drain"))
                    or not isinstance(s["run_id"], str)
                ):
                    raise ValueError("invalid node state")
            self.state = data["nodes"]

    @staticmethod
    def _empty():
        return {
            "health_at": -1,
            "step_at": -1,
            "last_drain": -1e20,
            "incident": False,
            "count": 0,
            "alerted": False,
            "run_id": "",
        }

    def _save(self, state):
        if self.state_path:
            atomic_write(
                self.state_path, json.dumps({"schema": 1, "nodes": state}, allow_nan=False)
            )

    def _fresh(self, timestamp, now):
        return (
            finite(timestamp)
            and timestamp >= 0
            and 0 <= now - timestamp <= self.rules["sample_freshness_seconds"]
        )

    def _health_known(self, m, now):
        return (
            self._fresh(m.observed_at, now)
            and type(m.xid_errors) is int
            and 0 <= m.xid_errors < 1_000_000
            and type(m.link_up) is bool
        )

    def _step_known(self, m, now):
        return (
            self._health_known(m, now)
            and m.xid_errors == 0
            and m.link_up
            and finite(m.step_time)
            and m.step_time > 0
            and self._fresh(m.step_observed_at, now)
            and bool(m.run_id)
        )

    def evaluate(self, metrics: dict[str, NodeMetrics]) -> list[Action]:
        now = self.clock()
        pending = copy.deepcopy(self.state)
        actions = []
        # Missing nodes retain state; an inventory-aware source supplies unknown metrics.
        for node, m in sorted(metrics.items()):
            s = pending.setdefault(node, self._empty())
            if not self._health_known(m, now):
                actions.append(
                    Action(node, "alert", "health indeterminate: missing/stale/invalid sample")
                )
                continue
            reasons = [f"XID {m.xid_errors}"] if m.xid_errors else []
            if not m.link_up:
                reasons.append("link down")
            # observed_at is the oldest component's timestamp, so an unchanged
            # watermark must not hide a new fault on a different GPU/NIC.
            if reasons:
                if (
                    not s["incident"]
                    and now - s["last_drain"] >= self.rules["drain_cooldown_seconds"]
                ):
                    reason = "; ".join(reasons)
                    actions.extend([Action(node, "drain", reason), Action(node, "requeue", reason)])
                    s["incident"] = True
                    s["last_drain"] = now
            elif m.observed_at > s["health_at"]:
                s["incident"] = False
            s["health_at"] = max(s["health_at"], m.observed_at)
            if m.step_time is None and m.run_id is None:
                continue
            peers = [
                p.step_time
                for p in metrics.values()
                if self._step_known(p, now) and p.run_id == m.run_id
            ]
            if not self._step_known(m, now) or len(peers) < self.rules["min_comparison_nodes"]:
                actions.append(
                    Action(node, "alert", "step time indeterminate: need fresh comparable peers")
                )
                continue
            if s["run_id"] != m.run_id:
                s.update(run_id=m.run_id, step_at=-1, count=0, alerted=False)
            if m.step_observed_at <= s["step_at"]:
                continue
            s["step_at"] = m.step_observed_at
            if m.step_time > median(peers) * self.rules["step_time_ratio"]:
                s["count"] += 1
                if s["count"] >= self.rules["consecutive_checks"] and not s["alerted"]:
                    actions.append(
                        Action(
                            node,
                            "alert",
                            "step time > median × "
                            f"{self.rules['step_time_ratio']} for "
                            f"{self.rules['consecutive_checks']} checks",
                        )
                    )
                    s["alerted"] = True
            else:
                s.update(count=0, alerted=False)
        self._save(pending)  # Failure raises before returning actions or changing in-memory state.
        self.state = pending
        return actions


def command_runner(argv, stdin=None):
    return subprocess.run(
        argv, input=stdin, text=True, capture_output=True, check=True, timeout=30
    ).stdout


class SchedulerExecutor:
    """Explicit owned jobs only; never discover/evict every workload on a node.

    dry_run requires no scheduler. Live mode is a command adapter for future authorized
    deployment, not enabled by this task. Results include raw command evidence.
    """

    def __init__(self, scheduler, managed_nodes, owner, *, dry_run=True, runner=command_runner):
        if scheduler not in ("slurm", "k8s"):
            raise ValueError("scheduler must be slurm or k8s")
        self.scheduler, self.managed_nodes, self.owner = scheduler, set(managed_nodes), owner
        self.dry_run, self.runner = dry_run, runner
        self._name(owner)
        for node in self.managed_nodes:
            self._name(node)

    @staticmethod
    def _name(value):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError("invalid explicit resource name")
        return value

    def _call(self, argv, evidence, stdin=None):
        entry = {"argv": argv, "stdin": stdin}
        evidence.append(entry)
        try:
            output = self.runner(argv, stdin=stdin)
            entry.update(ok=True, stdout=output)
            return output
        except Exception as exc:
            entry.update(
                ok=False,
                error=str(exc),
                stdout=getattr(exc, "stdout", None),
                stderr=getattr(exc, "stderr", None),
            )
            raise

    def execute(self, actions: list[Action], jobs: list[dict]) -> list[dict]:
        """Isolate the entire fault batch before any job can be placed again.

        Result order follows input actions, but commands run in two phases. Any
        isolation failure blocks every requeue in this batch. Jobs are inspected
        once per explicit identity and associated with verified affected nodes;
        failures whose node cannot be verified remain visible batch-wide.
        """
        results, drained = [], set()
        for action in actions:
            evidence = []
            result = {
                "node": action.node,
                "kind": action.kind,
                "reason": action.reason,
                "evidence": evidence,
            }
            results.append(result)
            try:
                if action.node not in self.managed_nodes:
                    raise ValueError("node outside explicit managed inventory")
                if action.kind == "alert":
                    result["status"] = "alert"
                elif action.kind not in ("drain", "requeue"):
                    raise ValueError("invalid action")
                elif self.dry_run:
                    result.update(status="planned", scheduler=self.scheduler, jobs=jobs)
                elif action.kind == "drain":
                    if self.scheduler == "slurm":
                        self._call(
                            [
                                "scontrol",
                                "update",
                                f"nodename={action.node}",
                                "state=drain",
                                f"reason={action.reason}",
                            ],
                            evidence,
                        )
                    else:
                        self._call(["kubectl", "cordon", action.node], evidence)
                        self._call(
                            [
                                "kubectl",
                                "taint",
                                "nodes",
                                action.node,
                                "crucible/health=failed:NoSchedule",
                                "--overwrite",
                            ],
                            evidence,
                        )
                    drained.add(action.node)
                    result["status"] = "drained"
                # Requeue actions are deliberately deferred until all drains complete.
            except Exception as exc:  # noqa: BLE001 - independent actions retain failure evidence
                result.update(status="failed", error=str(exc))
        if self.dry_run:
            return results
        requeues = [r for r in results if r["kind"] == "requeue"]
        faults = {a.node for a in actions if a.kind in ("drain", "requeue")}
        isolation_failed = faults - drained
        isolation_failed.update(
            r["node"] for r in results if r["kind"] == "drain" and r["status"] == "failed"
        )
        if isolation_failed or not jobs:
            error = (
                f"batch isolation incomplete: {sorted(isolation_failed)}; no requeue attempted"
                if isolation_failed
                else "explicit affected job inventory is required"
            )
            for result in requeues:
                result.update(
                    status="failed",
                    error=error,
                    jobs=[{"job": job, "status": "not_executed", "error": error} for job in jobs],
                )
            return results
        outcomes, seen = [], set()
        targets = {r["node"] for r in requeues}
        if not targets:
            return results
        for job in jobs:
            outcome = {"job": job, "nodes": [], "evidence": []}
            try:
                identity = (
                    (str(job["id"]),)
                    if self.scheduler == "slurm"
                    else tuple(job[k] for k in ("namespace", "job", "pod"))
                )
                if identity in seen:
                    continue
                seen.add(identity)
                if self.scheduler == "slurm":
                    self._slurm_requeue(targets, job, outcome)
                else:
                    self._k8s_requeue(targets, faults, job, outcome)
                outcome["status"] = "requeued"
            except Exception as exc:  # noqa: BLE001 - preserve all explicit-reference failures
                outcome.update(status="failed", error=str(exc))
            outcomes.append(outcome)
        for result in requeues:
            related = [o for o in outcomes if not o["nodes"] or result["node"] in o["nodes"]]
            result["jobs"] = related
            result["evidence"] = [e for o in related for e in o["evidence"]]
            result["status"] = (
                "failed"
                if any(o["status"] == "failed" for o in related)
                else "requeued"
                if related
                else "no_affected_jobs"
            )
        return results

    def _slurm_requeue(self, targets, job, outcome):
        evidence = outcome["evidence"]
        job_id = str(job["id"])
        if not re.fullmatch(r"[1-9][0-9]*", job_id):
            raise ValueError("only explicit scalar Slurm job IDs are supported")
        show = ["scontrol", "show", "job", job_id, "--json"]

        def read():
            jobs = json.loads(self._call(show, evidence))["jobs"]
            if len(jobs) != 1 or str(jobs[0]["job_id"]) != job_id:
                raise ValueError("job ID mismatch")
            if jobs[0].get("comment") != f"crucible:{self.owner}":
                raise ValueError("job is not explicitly owned")
            return jobs[0]

        current = read()
        if "RUNNING" not in current["job_state"]:
            raise ValueError("affected job is not running")
        nodes = self._call(
            ["scontrol", "show", "hostnames", current["nodes"]], evidence
        ).splitlines()
        outcome["nodes"] = sorted(targets.intersection(nodes))
        if not outcome["nodes"]:
            raise ValueError("job is not on any affected node")
        self._call(["scontrol", "requeue", job_id], evidence)
        after = read()
        if "PENDING" not in after["job_state"]:
            raise ValueError("requeue not confirmed as pending; reconcile before retry")

    def _k8s_requeue(self, targets, faults, job, outcome):
        evidence = outcome["evidence"]
        ns, name, podname = [self._name(job[k]) for k in ("namespace", "job", "pod")]

        def get(kind, name):
            return json.loads(
                self._call(["kubectl", "-n", ns, "get", kind, name, "-o", "json"], evidence)
            )

        current, pod = get("job", name), get("pod", podname)
        uid = current["metadata"]["uid"]
        self._name(uid)
        if current["metadata"].get("labels", {}).get("crucible/owner") != self.owner:
            raise ValueError("job is not explicitly owned")
        if current.get("spec", {}).get("suspend") or any(
            c.get("type") in ("Complete", "Failed") and c.get("status") == "True"
            for c in current.get("status", {}).get("conditions", [])
        ):
            raise ValueError("job cannot replace an evicted pod")

        def owned(p):
            return any(
                o.get("kind") == "Job"
                and o.get("uid") == uid
                and o.get("name") == name
                and o.get("controller") is True
                for o in p.get("metadata", {}).get("ownerReferences", [])
            )

        node = pod.get("spec", {}).get("nodeName")
        if not owned(pod) or node not in targets:
            raise ValueError("pod ownership or affected node mismatch")
        outcome["nodes"] = [node]
        listing = [
            "kubectl",
            "-n",
            ns,
            "get",
            "pods",
            "-l",
            f"batch.kubernetes.io/controller-uid={uid}",
            "-o",
            "json",
        ]
        before = {p["metadata"]["uid"] for p in json.loads(self._call(listing, evidence))["items"]}
        eviction = {
            "apiVersion": "policy/v1",
            "kind": "Eviction",
            "metadata": {"name": podname, "namespace": ns},
            "deleteOptions": {"preconditions": {"uid": pod["metadata"]["uid"]}},
        }
        self._call(
            [
                "kubectl",
                "create",
                "--raw",
                f"/api/v1/namespaces/{ns}/pods/{podname}/eviction",
                "-f",
                "-",
            ],
            evidence,
            json.dumps(eviction),
        )
        # One bounded check: asynchronous replacement may need operator reconciliation.
        after = json.loads(self._call(listing, evidence))["items"]
        if not any(
            owned(p)
            and p["metadata"]["uid"] not in before
            and p.get("spec", {}).get("nodeName") not in faults
            and p.get("status", {}).get("phase") in ("Pending", "Running")
            for p in after
        ):
            raise ValueError("replacement not observed; eviction is not confirmed requeue")
