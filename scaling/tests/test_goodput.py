import math

import pytest

from scaling.goodput import FaultTrainingHook, account_events, goodput


def test_goodput_accounts_restart_and_lost_steps():
    events = []
    now = [0.0]
    hook = FaultTrainingHook("run", emit=events.append, clock=lambda: now[0], clock_id="cpu")
    hook.start()

    def advance(seconds):
        now[0] += seconds

    for step in range(1, 4):
        hook.step(step, lambda: advance(2))
        if step == 2:
            hook.save("c2", step, lambda: advance(1))
    hook.failure("node lost")
    advance(3)  # detection + scheduling downtime belongs in denominator
    hook.restore("c2", 2, lambda: advance(4))
    hook.step(3, lambda: advance(3))  # redo logical step, only this attempt survives
    hook.save("c3", 3, lambda: advance(1))
    hook.finish("c3")
    result = account_events(events)
    assert result["useful_steps_time"] == 7
    assert result["lost_steps_time"] == 2
    assert result["lost_steps"] == 1
    assert result["completed_attempts"] == 4
    assert result["retained_steps"] == 3
    assert result["recovery_seconds"] == [4]
    assert result["wall_time"] == 18
    assert result["goodput"] == pytest.approx(7 / 18)


@pytest.mark.parametrize(
    "useful,wall", [(1, 0), (-1, 2), (3, 2), (math.nan, 2), (1, math.inf), (True, 2)]
)
def test_goodput_rejects_invalid_times(useful, wall):
    with pytest.raises(ValueError):
        goodput(useful, wall)


def test_failed_operations_never_claim_completion_and_missing_end_is_rejected():
    events = []
    hook = FaultTrainingHook("run", emit=events.append, clock=lambda: 1, clock_id="cpu")
    hook.start()

    def fail():
        raise RuntimeError("disk failed")

    with pytest.raises(RuntimeError):
        hook.step(1, fail)
    assert not any(e["kind"] == "step_complete" for e in events)
    with pytest.raises(ValueError):
        account_events(events)


def test_accounting_rejects_duplicate_rank_events_clock_mix_and_false_checkpoint():
    events = []
    now = iter(range(10))
    hook = FaultTrainingHook("run", emit=events.append, clock=lambda: next(now), clock_id="cpu")
    hook.start()
    hook.step(1, lambda: None)
    hook.save("c1", 1, lambda: None)
    hook.finish("c1")
    for mutate in ("rank", "duplicate", "clock", "checkpoint"):
        altered = [dict(e) for e in events]
        if mutate == "rank":
            altered[1]["rank"] = 1
        elif mutate == "duplicate":
            altered.insert(2, dict(altered[1]))
        elif mutate == "clock":
            altered[1]["clock_id"] = "other-host"
        else:
            altered[-1]["checkpoint_id"] = "never-saved"
        with pytest.raises(ValueError):
            account_events(altered)


def test_unsaved_tail_is_lost_and_failed_partial_step_stays_in_wall_time():
    events, now = [], [0.0]
    hook = FaultTrainingHook("run", emit=events.append, clock=lambda: now[0], clock_id="cpu")
    hook.start()

    def work():
        now[0] += 1

    hook.step(1, work)
    hook.save("c1", 1, work)
    hook.step(2, work)

    def interrupted():
        now[0] += 3
        raise RuntimeError("partial optimizer attempt lost")

    with pytest.raises(RuntimeError):
        hook.step(3, interrupted)
    hook.finish("c1")
    result = account_events(events)
    assert result["goodput"] == pytest.approx(1 / 6)
    assert result["lost_steps"] == 1
    assert result["retained_steps"] == 1
    assert result["completed_attempts"] == 2


def test_hook_replays_journal_without_inventing_a_new_clock_or_run_start():
    events, now = [], [0.0]
    hook = FaultTrainingHook("run", emit=events.append, clock=lambda: now[0], clock_id="cpu")
    hook.start()
    hook.step(1, lambda: now.__setitem__(0, 1))
    hook.save("c1", 1, lambda: now.__setitem__(0, 2))
    hook.failure("lost worker")
    resumed = FaultTrainingHook.resume(events, emit=events.append, clock=lambda: now[0])
    resumed.restore("c1", 1, lambda: now.__setitem__(0, 3))
    resumed.finish("c1")
    assert account_events(events)["goodput"] == pytest.approx(1 / 3)
    assert sum(e["kind"] == "run_start" for e in events) == 1
    with pytest.raises(ValueError):
        FaultTrainingHook.resume(events, emit=events.append, clock=lambda: 4)
