"""Fault timeline observations on a single coordinator clock; no hardware execution."""

import json
import os
import time
from pathlib import Path

from cluster.drain.controller import SchedulerExecutor, finite
from cluster.faults.plan import PAUSED


class FaultTimeline:
    def __init__(self, run_id, *, clock_id, emit=None, clock=time.monotonic):
        if not run_id or not clock_id:
            raise ValueError("run_id and explicit coordinator clock domain required")
        self.run_id, self.clock_id, self.clock, self.emit = run_id, clock_id, clock, emit
        self.events = []

    def record(self, kind, incident, node, *, evidence=None):
        at = self.clock()
        if not finite(at) or at < 0 or (self.events and at < self.events[-1]["at"]):
            raise ValueError("invalid or backwards coordinator clock")
        if not all(isinstance(v, str) and v for v in (kind, incident, node)):
            raise ValueError("explicit event kind, incident and node required")
        if kind in ("isolation_complete", "isolation_incomplete") and (
            not isinstance(evidence, dict)
            or not finite(evidence.get("started_at"))
            or not 0 <= evidence["started_at"] <= at
        ):
            raise ValueError("batch start must precede completion receipt")
        evidence = json.loads(json.dumps(evidence, allow_nan=False))
        event = {
            "schema": 1,
            "run_id": self.run_id,
            "clock_id": self.clock_id,
            "kind": kind,
            "incident": incident,
            "node": node,
            "at": at,
            "evidence": evidence,
        }
        if self.emit:
            self.emit(event)  # Failure persists no success in memory.
        self.events.append(event)
        return event

    def observe_controller(self, controller, metrics, incident_by_node):
        """Timestamp Task 3 action receipt, retaining raw source sample timestamps.

        No Action timestamp is invented. Indeterminate input alerts are not fault
        detection. First genuine alert or drain per incident is the detection event.
        """
        actions = controller.evaluate(metrics)
        for action in actions:
            if (
                action.node not in incident_by_node
                or action.kind == "requeue"
                or "indeterminate" in action.reason
            ):
                continue
            incident = incident_by_node[action.node]
            if any(e["kind"] == "detected" and e["incident"] == incident for e in self.events):
                continue
            sample = metrics[action.node]
            self.record(
                "detected",
                incident,
                action.node,
                evidence={
                    "action": vars(action),
                    "sample_observed_at": sample.observed_at,
                    "step_observed_at": sample.step_observed_at,
                    "source_clock": "Task3 source timestamp domain; not used as monotonic clock",
                },
            )
        return actions

    def scheduler_results(self, results, incidents, *, started_at):
        """Record caller-timed Task3 batch outcome; preserve raw per-command evidence.

        The caller captures started_at immediately before execute(). This method
        records actual receipt, never timestamps a planned drain as completed.
        Isolation is batch-wide: any failure leaves every incident incomplete.
        """
        if not finite(started_at) or started_at < 0:
            raise ValueError("explicit batch start in coordinator clock required")
        drains = [r for r in results if r.get("kind") == "drain"]
        expected = set(incidents)
        complete = {
            r["node"] for r in drains if r.get("status") == "drained"
        } == expected and not any(r.get("status") == "failed" for r in drains)
        finished = None
        if complete:
            timings = [entry for result in drains for entry in result.get("evidence", [])]
            if timings and all(
                e.get("clock_id") == self.clock_id
                and finite(e.get("started_at"))
                and finite(e.get("finished_at"))
                and started_at <= e["started_at"] <= e["finished_at"]
                for e in timings
            ):
                finished = max(e["finished_at"] for e in timings)
        for node, incident in incidents.items():
            kind = "isolation_complete" if complete else "isolation_incomplete"
            self.record(
                kind,
                incident,
                node,
                evidence={
                    "started_at": started_at,
                    "finished_at": finished,
                    "batch_results": results,
                },
            )
        return complete

    def metrics(self, incident):
        selected = [e for e in self.events if e["incident"] == incident]

        def one(kind):
            matches = [e for e in selected if e["kind"] == kind]
            if len(matches) > 1:
                raise ValueError(f"duplicate {kind} for incident")
            return matches[0] if matches else None

        injected, detected = one("injection_confirmed"), one("detected")
        isolated = one("isolation_complete")
        recovery_start, recovered = one("recovery_start"), one("recovery_complete")

        def delta(start, end):
            if start is None or end is None:
                return None
            if end < start:
                raise ValueError("incident phases are out of order")
            return end - start

        return {
            "detection_seconds": delta(
                injected["at"] if injected else None, detected["at"] if detected else None
            ),
            "drain_seconds": delta(
                isolated["evidence"]["started_at"] if isolated else None,
                isolated["evidence"]["finished_at"] if isolated else None,
            ),
            "detection_to_isolation_seconds": delta(
                detected["at"] if detected else None,
                isolated["evidence"]["finished_at"] if isolated else None,
            ),
            "recovery_seconds": delta(
                recovery_start["at"] if recovery_start else None,
                recovered["at"] if recovered else None,
            ),
        }


class ObservedSchedulerExecutor(SchedulerExecutor):
    """Task3 adapter that times the actual isolation commands before requeue.

    Only dry-run or an explicitly injected CPU command shim is supported here;
    this module provides no production command runner while the budget is paused.
    """

    def __init__(
        self,
        scheduler,
        managed_nodes,
        owner,
        *,
        clock_id,
        clock=time.monotonic,
        dry_run=True,
        runner=None,
    ):
        if not dry_run and runner is None:
            raise PermissionError(PAUSED)
        self.clock, self.clock_id = clock, clock_id
        super().__init__(
            scheduler, managed_nodes, owner, dry_run=dry_run, runner=runner or self._refuse
        )

    @staticmethod
    def _refuse(argv, stdin=None):
        raise PermissionError(PAUSED)

    def _call(self, argv, evidence, stdin=None):
        start = self.clock()
        try:
            return super()._call(argv, evidence, stdin)
        finally:
            evidence[-1].update(started_at=start, finished_at=self.clock(), clock_id=self.clock_id)


class JsonlSink:
    """Single-writer append journal; fsync each observation before returning."""

    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, event):
        body = json.dumps(event, allow_nan=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
