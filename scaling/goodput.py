"""Coordinator-clock event accounting for checkpoint-enabled fault experiments.

Only rank zero publishes, after all-rank synchronization. A completed run names an
actually saved full-state checkpoint; unsaved tail work and rolled-back attempts
are excluded. This measures productive wall-time fraction, not baseline-speed efficiency.
"""

import math
import time


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def goodput(useful_steps_time, wall_time) -> float:
    if (
        not _finite(useful_steps_time)
        or not _finite(wall_time)
        or not 0 <= useful_steps_time <= wall_time
        or wall_time <= 0
    ):
        raise ValueError("finite 0 <= useful_steps_time <= positive wall_time required")
    return useful_steps_time / wall_time


def _replay_events(events, *, require_terminal):
    """Validate one ordered run and replay checkpoint lineage, without guessing gaps.

    Timestamps and operation starts must be from one monotonic coordinator clock
    domain. Missing terminal events fail closed. A failed run may explicitly close
    against its last saved checkpoint; an interrupted log alone has no final result.
    """
    if not events or events[0].get("kind") != "run_start":
        raise ValueError("run_start required")
    identity = tuple(events[0].get(k) for k in ("run_id", "clock_id"))
    if not all(isinstance(v, str) and v for v in identity):
        raise ValueError("run and clock domain required")
    if type(events[0].get("world_size")) is not int or events[0]["world_size"] <= 0:
        raise ValueError("explicit positive world size required")
    current, checkpoints, lost = {}, {}, {}
    attempt, completed, failed, terminal = 0, 0, False, False
    recoveries = []
    previous = events[0].get("at")
    restore_pending = None
    worker_checkpoint = None
    work_finished_at = None
    for index, event in enumerate(events):
        at, kind = event.get("at"), event.get("kind")
        if (
            type(event.get("schema")) is not int
            or event.get("schema") != 1
            or type(event.get("rank")) is not int
            or event.get("rank") != 0
            or tuple(event.get(k) for k in ("run_id", "clock_id")) != identity
            or not _finite(at)
            or at < 0
            or at < previous
            or terminal
        ):
            raise ValueError("invalid event identity, clock, order or terminal boundary")
        previous = at
        if worker_checkpoint is not None and kind != "run_end":
            raise ValueError("work after worker completion is forbidden")
        if kind == "run_start":
            if index != 0:
                raise ValueError("duplicate run_start")
            continue
        if kind in ("step_complete", "checkpoint_saved", "restore_complete", "operation_failed"):
            start = event.get("started_at")
            if not _finite(start) or not events[index - 1]["at"] <= start <= at:
                # restore start is itself the preceding event; operations never overlap.
                raise ValueError("operation start must be ordered in coordinator clock")
        if kind == "step_complete":
            step = event.get("step")
            if failed or restore_pending or type(step) is not int or step != len(current) + 1:
                raise ValueError("contiguous global completed steps required")
            if at <= event["started_at"]:
                raise ValueError("positive synchronized step duration required")
            current[step] = (attempt, at - event["started_at"])
            completed += 1
        elif kind == "checkpoint_saved":
            checkpoint = event.get("checkpoint_id")
            if (
                failed
                or restore_pending
                or not isinstance(checkpoint, str)
                or not checkpoint
                or checkpoint in checkpoints
                or type(event.get("step")) is not int
                or event.get("step") != len(current)
                or event.get("full_state") is not True
            ):
                raise ValueError("unique full-state checkpoint of current completed state required")
            checkpoints[checkpoint] = dict(current)
        elif kind == "failure":
            if failed:
                raise ValueError("duplicate failure")
            failed = True
        elif kind == "restore_start":
            checkpoint = event.get("checkpoint_id")
            if not failed or restore_pending or checkpoint not in checkpoints:
                raise ValueError("recovery requires failed attempt and known saved checkpoint")
            restore_pending = (checkpoint, at)
        elif kind == "restore_complete":
            if restore_pending != (event.get("checkpoint_id"), event["started_at"]) or event.get(
                "step"
            ) != len(checkpoints[restore_pending[0]]):
                raise ValueError("restore completion must match saved state and start")
            saved = checkpoints[restore_pending[0]]
            for step, value in current.items():
                if saved.get(step) != value:
                    lost[(value[0], step)] = value[1]
            current = dict(saved)
            recoveries.append(at - restore_pending[1])
            attempt += 1
            failed, restore_pending = False, None
        elif kind == "operation_failed":
            # No success state is inferred from exceptions, even on save/load.
            failed = True
            restore_pending = None
        elif kind in ("worker_complete", "run_end"):
            checkpoint = event.get("checkpoint_id")
            if checkpoint not in checkpoints or restore_pending:
                raise ValueError("explicit actual final saved state required")
            saved = checkpoints[checkpoint]
            if any(current.get(step) != value for step, value in saved.items()):
                raise ValueError("terminal checkpoint must belong to current retained lineage")
            for step, value in current.items():
                if saved.get(step) != value:
                    lost[(value[0], step)] = value[1]
            current = dict(saved)
            if kind == "worker_complete":
                if failed:
                    raise ValueError("failed worker cannot claim successful completion")
                worker_checkpoint, work_finished_at = checkpoint, at
            else:
                if worker_checkpoint is not None and checkpoint != worker_checkpoint:
                    raise ValueError("coordinator must close the actual worker checkpoint")
                terminal = True
        else:
            raise ValueError(f"unknown training event {kind}")
    if not terminal and require_terminal:
        raise ValueError("missing run_end: final retained state is unknown")
    # A previously discarded checkpoint may be selected again; surviving attempts
    # must never remain in both useful work and final lost-work totals.
    for step, value in current.items():
        lost.pop((value[0], step), None)
    useful = sum(value[1] for value in current.values())
    wall = events[-1]["at"] - events[0]["at"]
    result = {
        "useful_steps_time": useful,
        "wall_time": wall,
        "goodput": goodput(useful, wall) if terminal else None,
        "lost_steps": len(lost),
        "lost_steps_time": sum(lost.values()),
        "completed_attempts": completed,
        "retained_steps": len(current),
        "recovery_seconds": recoveries,
        "final_checkpoint_id": events[-1].get("checkpoint_id") if terminal else None,
        "work_finished_at": work_finished_at,
        "observation_lateness_seconds": events[-1].get("observation_lateness_seconds"),
    }
    return result, current, checkpoints, failed, terminal


def account_events(events):
    return _replay_events(events, require_terminal=True)[0]


def worker_state(events):
    if not events or events[-1].get("kind") != "worker_complete":
        raise ValueError("successful worker_complete required before observation closure")
    result = _replay_events(events, require_terminal=False)[0]
    result["final_checkpoint_id"] = events[-1]["checkpoint_id"]
    return result


class FaultTrainingHook:
    """Wrap successful operations; synchronize must barrier AND wait for device work.

    All ranks call operations and synchronize; only reporting rank zero emits. The
    reporter's clock is the sole source of durations (no sum of per-rank times).
    Reuse the reporter state across trainer replacements, or replay its retained log.
    """

    def __init__(
        self,
        run_id,
        *,
        emit,
        clock=time.monotonic,
        clock_id,
        synchronize=lambda: None,
        rank=0,
        world_size=1,
    ):
        if (
            not run_id
            or not clock_id
            or type(rank) is not int
            or rank < 0
            or type(world_size) is not int
            or not 0 <= rank < world_size
        ):
            raise ValueError("run_id, clock_id and nonnegative rank required")
        self.run_id, self.emit, self.clock = run_id, emit, clock
        self.clock_id, self.synchronize, self.rank = clock_id, synchronize, rank
        self.world_size = world_size
        self.started, self.last_step, self.checkpoints, self.failed = False, 0, {}, False
        self.ended = False
        self.failed_at = None

    @classmethod
    def resume(cls, events, *, emit, clock, synchronize=lambda: None, rank=0):
        _, current, checkpoints, failed, terminal = _replay_events(events, require_terminal=False)
        if terminal:
            raise ValueError("cannot resume a terminal run")
        if events[-1]["kind"] in ("restore_start", "worker_complete"):
            raise ValueError("incomplete recovery must be reconciled before replay")
        hook = cls(
            events[0]["run_id"],
            clock_id=events[0]["clock_id"],
            emit=emit,
            clock=clock,
            synchronize=synchronize,
            rank=rank,
            world_size=events[0]["world_size"],
        )
        if clock() < events[-1]["at"]:
            raise ValueError("restart must retain the same coordinator clock domain")
        hook.started, hook.last_step, hook.failed = True, len(current), failed
        hook.checkpoints = {key: len(value) for key, value in checkpoints.items()}
        for event in events:
            if event["kind"] == "restore_complete":
                hook.failed_at = None
            elif event["kind"] in ("failure", "operation_failed") and hook.failed_at is None:
                hook.failed_at = event["at"]
        return hook

    def _event(self, kind, **fields):
        if self.ended:
            raise ValueError("run is terminal")
        at = self.clock()
        if not _finite(at) or at < 0:
            raise ValueError("invalid coordinator clock")
        if self.rank == 0:
            self.emit(
                dict(
                    schema=1,
                    run_id=self.run_id,
                    clock_id=self.clock_id,
                    rank=0,
                    kind=kind,
                    at=at,
                    **fields,
                )
            )
        if kind in ("failure", "operation_failed") and self.failed_at is None:
            self.failed_at = at

    def start(self):
        if self.started:
            raise ValueError("run already started")
        self._event("run_start", world_size=self.world_size)
        self.started = True

    def _operation(self, kind, operation, **fields):
        if not self.started:
            raise ValueError("run_start required")
        self.synchronize()
        start = self.clock()
        try:
            result = operation()
            self.synchronize()
        except Exception as exc:
            self.failed = True
            self._event("operation_failed", started_at=start, operation=kind, error=str(exc))
            raise
        self._event(kind, started_at=start, **fields)
        return result

    def step(self, step, operation):
        if self.failed or type(step) is not int or step != self.last_step + 1:
            raise ValueError("step must follow the retained global optimizer step")
        result = self._operation("step_complete", operation, step=step)
        self.last_step = step
        return result

    def save(self, checkpoint_id, step, operation):
        if self.failed or step != self.last_step or checkpoint_id in self.checkpoints:
            raise ValueError("save must describe current state with a unique checkpoint ID")
        result = self._operation(
            "checkpoint_saved", operation, step=step, checkpoint_id=checkpoint_id, full_state=True
        )
        self.checkpoints[checkpoint_id] = step
        return result

    def failure(self, reason):
        if self.failed or not self.started:
            raise ValueError("started nonfailed attempt required")
        self.failed = True
        self._event("failure", reason=reason)

    def restore(self, checkpoint_id, step, operation):
        if not self.failed or self.checkpoints.get(checkpoint_id) != step:
            raise ValueError("restore requires known actual saved checkpoint after failure")
        if not self.started:
            raise ValueError("run_start required")
        self.synchronize()
        self._restore_at = self.clock()
        # Avoid taking a second clock sample for the start event.
        if self.rank == 0:
            self.emit(
                {
                    "schema": 1,
                    "run_id": self.run_id,
                    "clock_id": self.clock_id,
                    "rank": 0,
                    "kind": "restore_start",
                    "at": self._restore_at,
                    "checkpoint_id": checkpoint_id,
                }
            )
        try:
            result = operation()
            self.synchronize()
        except Exception as exc:
            self._event(
                "operation_failed",
                started_at=self._restore_at,
                operation="restore_complete",
                error=str(exc),
            )
            raise
        self._event(
            "restore_complete", started_at=self._restore_at, checkpoint_id=checkpoint_id, step=step
        )
        self.last_step, self.failed, self.failed_at = step, False, None
        return result

    def complete_worker(self, checkpoint_id):
        if checkpoint_id not in self.checkpoints or self.failed:
            raise ValueError("successful final saved state required")
        self._event("worker_complete", checkpoint_id=checkpoint_id)
        self.ended = True

    def finish(self, checkpoint_id):
        if checkpoint_id not in self.checkpoints:
            raise ValueError("actual final checkpoint required")
        self._event("run_end", checkpoint_id=checkpoint_id)
        self.ended = True
