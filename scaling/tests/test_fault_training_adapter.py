from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from scaling.fault_train import instrument_fault_trainer
from scaling.goodput import FaultTrainingHook, account_events


def config(load_step=-1):
    return SimpleNamespace(
        checkpoint=SimpleNamespace(
            enable=True,
            load_only=False,
            async_mode="disabled",
            last_save_model_only=False,
            last_save_in_hf=False,
            initial_load_model_only=False,
            initial_load_in_hf=False,
            initial_load_path=None,
            exclude_from_loading=[],
            create_seed_checkpoint=False,
            keep_latest_k=2,
            interval=2,
            enable_first_step_checkpoint=False,
            load_step=load_step,
        ),
        fault_tolerance=SimpleNamespace(enable=False),
    )


class FakeTrainer:
    storage: ClassVar[dict] = {}
    fail_save, fail_load, fail_step = False, False, 0

    def __init__(self, config):
        self.job_config, self.step = config, 0
        self.checkpointer = FakeManager(self)

    def train_step(self, data):
        if self.step == self.fail_step:
            raise RuntimeError("node lost")
        return data

    def train(self):
        self.checkpointer.load(step=self.job_config.checkpoint.load_step)
        while self.should_continue_training():
            self.step += 1
            self.train_step("data")
            self.checkpointer.save(self.step, last_step=self.step == 3)

    def should_continue_training(self):
        return self.step < 3

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, state):
        self.step = state["step"]


class FakeManager:
    folder = "/cpu-fixture/checkpoint"

    def __init__(self, trainer):
        self.trainer = trainer

    def save(self, curr_step, last_step=False):
        if self.trainer.fail_save:
            raise RuntimeError("storage failed")
        if curr_step % 2 == 0 or last_step:
            self.trainer.storage[curr_step] = self.trainer.state_dict()
            path = Path(self.folder) / f"step-{curr_step}"
            path.mkdir(parents=True, exist_ok=True)
            (path / ".metadata").write_text("CPU fake DCP metadata")

    def load(self, step=-1):
        if self.trainer.fail_load:
            return False
        if step == -1:
            return False
        self.trainer.load_state_dict(self.trainer.storage[step])
        return True

    def maybe_wait_for_staging(self):
        pass


@pytest.fixture(autouse=True)
def isolated_checkpoint_files(tmp_path, monkeypatch):
    monkeypatch.setattr(FakeManager, "folder", str(tmp_path / "checkpoint"))


class FakeRNG:
    def __init__(self):
        self.value = 42

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = state["value"]


def make_hook():
    events = []
    times = iter(range(100))
    hook = FaultTrainingHook("run", clock_id="cpu", clock=lambda: next(times), emit=events.append)
    return hook, events


def test_thin_checkpoint_adapter_uses_upstream_loop_and_restores_real_saved_state():
    FakeTrainer.storage = {}
    hook, events = make_hook()
    trainer_type = instrument_fault_trainer(FakeTrainer)
    first = trainer_type(config(), hook=hook, rng=FakeRNG())
    first.fail_step = 3
    with pytest.raises(RuntimeError, match="node lost"):
        first.train()
    assert hook.failed
    assert list(FakeTrainer.storage) == [2]
    rng = FakeRNG()
    rng.value = 99
    second = trainer_type(config(load_step=2), hook=hook, rng=rng)
    second.train()
    assert rng.value == 42
    assert second.step == 3
    result = account_events(events)
    assert result["retained_steps"] == 3
    assert len(result["recovery_seconds"]) == 1
    assert sum(e["kind"] == "step_complete" for e in events) == 3
    assert sum(e["kind"] == "checkpoint_saved" for e in events) == 2


@pytest.mark.parametrize("failure", ["save", "load"])
def test_failed_manager_operation_never_publishes_checkpoint_or_restore_success(failure):
    FakeTrainer.storage = {}
    hook, events = make_hook()
    trainer_type = instrument_fault_trainer(FakeTrainer)
    first = trainer_type(config(), hook=hook, rng=FakeRNG())
    first.fail_step = 3
    if failure == "save":
        first.fail_save = True
        with pytest.raises(RuntimeError, match="storage failed"):
            first.train()
        assert not any(e["kind"] == "checkpoint_saved" for e in events)
    else:
        with pytest.raises(RuntimeError):
            first.train()
        second = trainer_type(config(load_step=2), hook=hook, rng=FakeRNG())
        second.fail_load = True
        with pytest.raises(ValueError, match="load failed"):
            second.train()
        assert not any(e["kind"] == "restore_complete" for e in events)
    assert not any(e["kind"] == "run_end" for e in events)


@pytest.mark.parametrize(
    "field,value",
    [
        ("async_mode", "async"),
        ("last_save_model_only", True),
        ("load_step", 0),
        ("keep_latest_k", 3),
    ],
)
def test_adapter_rejects_false_durable_or_model_only_recovery_modes(field, value):
    c = config()
    setattr(c.checkpoint, field, value)
    hook, _ = make_hook()
    with pytest.raises(ValueError):
        instrument_fault_trainer(FakeTrainer)(c, hook=hook, rng=FakeRNG())


def test_rng_is_rank_keyed_and_world_size_change_is_rejected():
    trainer_type = instrument_fault_trainer(FakeTrainer)
    states = []
    for rank in (0, 1):
        hook = FaultTrainingHook(
            "run", clock_id="cpu", rank=rank, world_size=2, emit=lambda e: None
        )
        rng = FakeRNG()
        rng.value = rank + 10
        trainer = trainer_type(config(), hook=hook, rng=rng)
        states.append(trainer.state_dict())
    assert states[0]["crucible_rng_rank_0"]["state"]["value"] == 10
    assert states[1]["crucible_rng_rank_1"]["state"]["value"] == 11
    assert "crucible_rng_rank_0" not in states[1]
    merged = states[0] | states[1]  # DCP planner union of rank-local keys.
    trainer.crucible_rng.value = 99
    trainer.load_state_dict(merged)
    assert trainer.crucible_rng.value == 11
    trainer.crucible_fault_hook.world_size = 3
    with pytest.raises(ValueError, match="unchanged world size"):
        trainer.load_state_dict(merged)


def test_fault_training_cli_remains_closed_even_with_ordinary_approval_flags():
    import os
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [str(root / ".venv/bin/python"), "-m", "scaling.fault_train"],
        env=dict(os.environ, CRUCIBLE_MULTINODE_APPROVED="1"),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "resource budget" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_torch_hook_factory_waits_for_device_and_all_ranks_without_importing_torch():
    from scaling.fault_train import make_torch_hook

    calls, events = [], []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda d: calls.append(("device", d))),
        distributed=SimpleNamespace(
            barrier=lambda: calls.append(("barrier",)), get_rank=lambda: 0, get_world_size=lambda: 2
        ),
    )
    times = iter(range(20))
    hook = make_torch_hook(
        torch=torch,
        device="cpu-fake",
        run_id="run",
        clock_id="coordinator",
        clock=lambda: next(times),
        emit=events.append,
    )
    hook.start()
    hook.step(1, lambda: calls.append(("train",)))
    assert calls == [
        ("device", "cpu-fake"),
        ("barrier",),
        ("device", "cpu-fake"),
        ("train",),
        ("device", "cpu-fake"),
        ("barrier",),
        ("device", "cpu-fake"),
    ]
    assert events[0]["world_size"] == 2


def test_restore_rejects_checkpoint_file_generation_without_successful_save_event():
    FakeTrainer.storage = {}
    hook, events = make_hook()
    trainer_type = instrument_fault_trainer(FakeTrainer)
    first = trainer_type(config(), hook=hook, rng=FakeRNG())
    first.fail_step = 3
    with pytest.raises(RuntimeError):
        first.train()
    FakeTrainer.storage[2]["crucible_checkpoint_event_id"] = "unacknowledged-partial-overwrite"
    second = trainer_type(config(load_step=2), hook=hook, rng=FakeRNG())
    with pytest.raises(ValueError, match="load failed"):
        second.train()
    assert not any(e["kind"] == "restore_complete" for e in events)


def test_runtime_hook_factory_runs_after_upstream_initializes_distributed_state():
    from scaling.fault_train import create_fault_trainer

    initialized = []

    class Base(FakeTrainer):
        def __init__(self, config):
            super().__init__(config)
            self.device = "fake-cuda-device"
            initialized.append(True)

    def rank():
        assert initialized  # Real upstream creates its process group in __init__.
        return 0

    torch = SimpleNamespace(distributed=SimpleNamespace(get_rank=rank, get_world_size=lambda: 1))
    trainer = create_fault_trainer(
        Base,
        config(),
        torch=torch,
        numpy=object(),
        run_id="run",
        clock_id="cpu",
        clock=lambda: 1,
        emit=lambda e: None,
        wait_until=lambda deadline: None,
    )
    assert trainer.crucible_fault_hook.world_size == 1
    assert trainer.crucible_rng.device == "fake-cuda-device"


def test_fixed_campaign_stops_steps_for_final_save_then_accounts_full_two_hours():
    from scaling.fault_train import CampaignWindow

    FakeTrainer.storage = {}
    now, events = [0.0], []
    hook = FaultTrainingHook("run", clock_id="cpu", clock=lambda: now[0], emit=events.append)
    hook.start()

    class Timed(FakeTrainer):
        def train_step(self, data):
            now[0] += 10
            return data

    window = CampaignWindow(
        started_at=0, clock=lambda: now[0], wait_until=lambda deadline: now.__setitem__(0, deadline)
    )
    trainer = instrument_fault_trainer(Timed)(config(), hook=hook, rng=FakeRNG(), campaign=window)
    trainer.train()
    assert trainer.step == 708  # The three-step fake target is not the time-based campaign stop.
    assert events[-2]["kind"] == "checkpoint_saved"
    assert events[-2]["at"] == 7080
    assert events[-1]["kind"] == "worker_complete"
    assert events[-1]["at"] == 7080
    with pytest.raises(ValueError, match="missing run_end"):
        account_events(events)  # Workers release before coordinator observes the window.


def test_cooperative_deadline_breach_has_no_successful_terminal_and_recovery_budget_is_enforced():
    from scaling.fault_train import CampaignWindow

    now = [0.0]
    window = CampaignWindow(started_at=0, clock=lambda: now[0], wait_until=lambda d: None)
    now[0] = 7080
    assert not window.should_train()
    now[0] = 7201
    with pytest.raises(TimeoutError):
        window.finish()
    now[0] = 401
    with pytest.raises(TimeoutError, match="recovery budget"):
        window.check_recovery(100)


def test_supervisor_bounds_a_blocked_training_or_save_group_without_claiming_success():
    from scaling.fault_train import CampaignSupervisor

    now, calls = [0.0], []

    class BlockedGroup:
        def wait(self, timeout):
            calls.append(("wait", timeout))
            now[0] += timeout
            raise TimeoutError("fake blocked CUDA/save")

        def terminate_group(self):
            calls.append(("TERM", now[0]))

        def kill_group(self):
            calls.append(("KILL", now[0]))

    result = CampaignSupervisor().rehearse(
        BlockedGroup(),
        started_at=0,
        clock=lambda: now[0],
        terminate_grace_seconds=5,
    )
    assert result["status"] == "deadline_exceeded"
    assert calls[:3] == [("wait", 7200), ("TERM", 7200), ("wait", 5)]
    assert ("KILL", 7205) in calls
    assert result["valid"] is False


def test_owned_launcher_accounts_initialization_and_waits_for_descendants_not_leader():
    from cluster.faults.supervisor import CampaignSupervisor

    now, calls, launches, events = [0.0], [], [], []

    class Scope:
        live = False

        def populated(self):
            return self.live

        def attach(self, pid):
            assert pid == 43210
            self.live = True
            calls.append(("attach_before_exec", now[0]))

        def wait_empty(self, timeout, **kwargs):
            if self.live:
                now[0] += timeout
                raise TimeoutError("leader exited but descendant ignores TERM")
            return now[0]

        def terminate(self):
            calls.append(("pidfd_TERM", now[0]))

        def kill(self):
            calls.append(("cgroup_KILL", now[0]))
            self.live = False

    class Child:
        pid = 43210

        def wait(self, timeout):
            assert not scope.live  # Reaping leader before descendants finish is forbidden.
            calls.append(("reap", now[0]))
            return 0

    def popen(argv, **kwargs):
        assert "--gate" in argv and kwargs["pass_fds"]
        launches.append((argv, kwargs))
        now[0] += 30
        return Child()

    scope = Scope()
    result = CampaignSupervisor().rehearse_launch(
        ["python", "-m", "owned-training-entry"],
        run_id="run",
        clock_id="cpu",
        clock=lambda: now[0],
        emit=events.append,
        env={"WORLD_SIZE": "2"},
        popen=popen,
        scope=scope,
    )
    assert launches[0][1]["start_new_session"] is True
    assert launches[0][1]["shell"] is False
    assert calls == [
        ("attach_before_exec", 30),
        ("pidfd_TERM", 7200),
        ("cgroup_KILL", 7205),
        ("reap", 7205),
    ]
    assert result["valid"] is False
    assert "cleanup_error" not in result


def test_supervisor_stops_after_term_when_owned_group_exits_and_retains_reap_code():
    from cluster.faults.supervisor import CampaignSupervisor

    now, calls = [0.0], []

    class Responsive:
        def wait(self, timeout):
            if not calls:
                now[0] += timeout
                raise TimeoutError("deadline")
            return -15

        def terminate_group(self):
            calls.append("TERM")

        def kill_group(self):
            calls.append("KILL")

    result = CampaignSupervisor().rehearse(Responsive(), started_at=0, clock=lambda: now[0])
    assert calls == ["TERM"]
    assert result["exit_code"] == -15
    assert result["valid"] is False


@pytest.mark.parametrize("future_terminal", [False, True])
def test_exited_group_with_missing_or_future_journal_is_invalid_and_never_signaled(future_terminal):
    from cluster.faults.supervisor import CampaignSupervisor

    now, signals = [0.0], []
    events = []
    if future_terminal:
        source_now = [0.0]
        hook = FaultTrainingHook(
            "run", clock_id="cpu", clock=lambda: source_now[0], emit=events.append
        )
        hook.start()
        hook.step(1, lambda: source_now.__setitem__(0, 1))
        hook.save("c1", 1, lambda: source_now.__setitem__(0, 2))
        source_now[0] = 7200
        hook.finish("c1")

    class Exited:
        def wait(self, timeout):
            now[0] = 7000
            return 0

        def terminate_group(self):
            signals.append("TERM")

        def kill_group(self):
            signals.append("KILL")

    result = CampaignSupervisor().rehearse(
        Exited(), started_at=0, clock=lambda: now[0], read_events=lambda: events
    )
    assert result["valid"] is False
    assert signals == []


def test_supervisor_accepts_clean_exit_with_worker_checkpoint_and_coordinator_observation():
    from cluster.faults.supervisor import CampaignSupervisor

    now, events = [0.0], []
    hook = FaultTrainingHook("run", clock_id="cpu", clock=lambda: now[0], emit=events.append)
    hook.start()
    hook.step(1, lambda: now.__setitem__(0, 1))
    hook.save("c1", 1, lambda: now.__setitem__(0, 2))
    hook.complete_worker("c1")

    class Complete:
        def wait(self, timeout):
            now[0] = 3
            return 0

    result = CampaignSupervisor().rehearse(
        Complete(),
        started_at=0,
        clock=lambda: now[0],
        read_events=lambda: events,
        emit=events.append,
        wait_until=lambda deadline: now.__setitem__(0, deadline),
    )
    assert result["valid"] is True
    assert result["accounting"]["goodput"] == pytest.approx(1 / 7200)


def test_distributed_window_broadcasts_one_decision_despite_staggered_rank_clocks():
    from scaling.fault_train import CampaignWindow, torch_coordinator

    messages, rank = [], [0]

    def broadcast(packet, src, device):
        assert src == 0
        if rank[0] == 0:
            messages.append(packet[0])
        else:
            packet[0] = messages.pop(0)

    torch = SimpleNamespace(
        distributed=SimpleNamespace(get_rank=lambda: rank[0], broadcast_object_list=broadcast)
    )
    leader_time, follower_reads = [7079.999], []
    leader = CampaignWindow(
        started_at=0,
        clock=lambda: leader_time[0],
        wait_until=lambda d: None,
        coordinate=torch_coordinator(torch, "cpu"),
    )
    follower = CampaignWindow(
        started_at=0.01,
        clock=lambda: follower_reads.append(7080.001),
        wait_until=lambda d: None,
        coordinate=torch_coordinator(torch, "cpu"),
    )
    assert leader.should_train() is True
    rank[0] = 1
    assert follower.should_train() is True
    assert follower_reads == []
    rank[0] = 0
    leader_time[0] = 7080.002
    assert leader.should_train() is False
    rank[0] = 1
    assert follower.should_train() is False


def test_off_interval_final_save_never_retains_three_checkpoint_directories(tmp_path):
    from pathlib import Path

    FakeTrainer.storage = {}
    hook, _ = make_hook()
    trainer = instrument_fault_trainer(FakeTrainer)(config(), hook=hook, rng=FakeRNG())
    hook.start()
    max_count = 0
    for step in range(1, 6):
        trainer.step = step
        hook.step(step, lambda: None)
        trainer.checkpointer.save(step, last_step=step == 5)
        max_count = max(
            max_count, len(list(Path(trainer.checkpointer.manager.folder).glob("step-*")))
        )
    assert max_count == 2
    assert sorted(p.name for p in Path(trainer.checkpointer.manager.folder).glob("step-*")) == [
        "step-4",
        "step-5",
    ]


def test_coordinator_can_wake_late_after_workers_safely_exit_before_budget():
    from cluster.faults.supervisor import CampaignSupervisor

    now, events = [0.0], []
    hook = FaultTrainingHook("run", clock_id="cpu", clock=lambda: now[0], emit=events.append)
    hook.start()
    hook.step(1, lambda: now.__setitem__(0, 1))
    hook.save("c1", 1, lambda: now.__setitem__(0, 2))
    now[0] = 7081
    hook.complete_worker("c1")

    class AllExited:
        def wait(self, timeout):
            now[0] = 7090
            return 0

    result = CampaignSupervisor().rehearse(
        AllExited(),
        started_at=0,
        clock=lambda: now[0],
        read_events=lambda: events,
        emit=events.append,
        wait_until=lambda deadline: now.__setitem__(0, deadline + 0.000001),
    )
    assert result["valid"] is True
    assert events[-1]["at"] == 7200.000001
    assert events[-1]["workers_exited_at"] == 7090
    assert result["accounting"]["wall_time"] == 7200.000001
    assert result["observation_lateness_seconds"] == pytest.approx(0.000001)


@pytest.mark.parametrize("post_save_failure", [False, True])
def test_retention_is_rank_zero_only_and_result_reaches_every_rank(monkeypatch, post_save_failure):
    import threading
    import time

    from scaling.checkpoint_retention import CheckpointRetention
    from scaling.fault_train import torch_coordinator

    local = threading.local()
    broadcast_barrier, save_barrier = threading.Barrier(2), threading.Barrier(2)
    frame, writes, events, errors = [None], [], [], []
    original_begin, original_confirm = CheckpointRetention.begin, CheckpointRetention.confirm

    def broadcast(packet, src, device):
        if local.rank == 0:
            frame[0] = packet[0]
        broadcast_barrier.wait(timeout=2)
        packet[0] = frame[0]
        broadcast_barrier.wait(timeout=2)

    def begin(self, step, protected_checkpoint_id=None):
        writes.append(("begin", local.rank))
        return original_begin(self, step, protected_checkpoint_id=protected_checkpoint_id)

    def confirm(self, step, identity):
        writes.append(("confirm", local.rank))
        if post_save_failure:
            raise OSError("post-save retention verification failed")
        return original_confirm(self, step, identity)

    monkeypatch.setattr(CheckpointRetention, "begin", begin)
    monkeypatch.setattr(CheckpointRetention, "confirm", confirm)
    torch = SimpleNamespace(
        distributed=SimpleNamespace(get_rank=lambda: local.rank, broadcast_object_list=broadcast)
    )
    sync = threading.Barrier(2)

    class CollectiveManager(FakeManager):
        def save(self, curr_step, last_step=False):
            if local.rank == 0:
                super().save(curr_step, last_step)
            save_barrier.wait(timeout=2)

    class CollectiveTrainer(FakeTrainer):
        def __init__(self, c):
            super().__init__(c)
            self.checkpointer = CollectiveManager(self)

    def rank_worker(rank):
        local.rank = rank
        hook = FaultTrainingHook(
            "run",
            clock_id="thread-cpu",
            clock=time.monotonic,
            synchronize=lambda: sync.wait(timeout=2),
            emit=events.append,
            rank=rank,
            world_size=2,
        )
        hook.coordinate = torch_coordinator(torch, "cpu")
        trainer = instrument_fault_trainer(CollectiveTrainer)(config(), hook=hook, rng=FakeRNG())
        hook.start()
        trainer.step = 1
        try:
            hook.step(1, lambda: None)
            trainer.checkpointer.save(1, last_step=True)
        except Exception as exc:  # noqa: BLE001 - collect both ranks' actual failures
            errors.append((rank, str(exc)))

    threads = [threading.Thread(target=rank_worker, args=(rank,), daemon=True) for rank in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
    assert not any(thread.is_alive() for thread in threads)
    assert writes == [("begin", 0), ("confirm", 0)]
    if post_save_failure:
        assert sorted(rank for rank, _ in errors) == [0, 1]
        assert all("post-save retention" in error for _, error in errors)
        assert not any(event["kind"] == "checkpoint_saved" for event in events)
    else:
        assert errors == []
        assert sum(event["kind"] == "checkpoint_saved" for event in events) == 1


def test_fault_runtime_uses_config_copy_to_disable_only_upstream_asynchronous_purge():
    original = config()
    hook, _ = make_hook()
    trainer = instrument_fault_trainer(FakeTrainer)(original, hook=hook, rng=FakeRNG())
    assert original.checkpoint.keep_latest_k == 2
    assert trainer.job_config is not original
    assert trainer.job_config.checkpoint.keep_latest_k == 0


def test_observed_worker_exit_after_budget_cannot_be_closed_as_success():
    from cluster.faults.supervisor import CampaignSupervisor

    now, events = [0.0], []
    hook = FaultTrainingHook("run", clock_id="cpu", clock=lambda: now[0], emit=events.append)
    hook.start()
    hook.step(1, lambda: now.__setitem__(0, 1))
    hook.save("c1", 1, lambda: now.__setitem__(0, 2))
    hook.complete_worker("c1")

    class LateExit:
        def wait(self, timeout):
            now[0] = 7200.001
            return 0

    result = CampaignSupervisor().rehearse(
        LateExit(),
        started_at=0,
        clock=lambda: now[0],
        read_events=lambda: events,
        emit=events.append,
        wait_until=lambda d: None,
    )
    assert result["valid"] is False
    assert result["status"] == "deadline_exceeded"
    assert not any(event["kind"] == "run_end" for event in events)


@pytest.mark.parametrize("failure_mode", [None, "before_write", "partial_write"])
def test_explicit_old_checkpoint_replay_replaces_superseded_generation_but_keeps_anchor(
    failure_mode,
):
    FakeTrainer.storage = {}
    hook, events = make_hook()
    trainer_type = instrument_fault_trainer(FakeTrainer)
    first = trainer_type(config(), hook=hook, rng=FakeRNG())
    hook.start()
    for step in range(1, 5):
        first.step = step
        first.train_step("batch")
        first.checkpointer.save(step)
    root = Path(first.checkpointer.folder)
    assert sorted(p.name for p in root.glob("step-*")) == ["step-2", "step-4"]
    saved_anchor = (root / "step-2/.crucible-confirmed.json").read_text()
    hook.failure("replace trainer and restore the older retained checkpoint")
    second = trainer_type(config(load_step=2), hook=hook, rng=FakeRNG())
    second.checkpointer.load(step=2)
    assert sum(event["kind"] == "restore_complete" for event in events) == 1
    assert second.step == 2
    for step in (3, 4):
        second.step = step
        second.train_step("replayed batch")
    second.fail_save = failure_mode == "before_write"
    if failure_mode == "partial_write":

        def interrupted_save(curr_step, last_step=False):
            partial = root / f"step-{curr_step}"
            partial.mkdir()
            (partial / "partial-data").write_text("failed save evidence")
            assert len(list(root.glob("step-*"))) <= 2
            raise RuntimeError("storage failed during new generation")

        second.checkpointer.manager.save = interrupted_save
    if failure_mode is not None:
        with pytest.raises(RuntimeError, match="storage failed"):
            second.checkpointer.save(4)
        assert (root / "step-2/.crucible-confirmed.json").read_text() == saved_anchor
        assert (root / "step-2/.metadata").is_file()
        if failure_mode == "partial_write":
            assert (root / "step-4/partial-data").read_text() == "failed save evidence"
            assert not (root / "step-4/.crucible-confirmed.json").exists()
        else:
            assert not (root / "step-4").exists()
        assert sum(event["kind"] == "checkpoint_saved" for event in events) == 2
        third = trainer_type(config(load_step=2), hook=hook, rng=FakeRNG())
        third.checkpointer.load(step=2)
        assert sum(event["kind"] == "restore_complete" for event in events) == 2
        hook.finish(third.checkpointer.last_checkpoint)
        assert account_events(events)["retained_steps"] == 2
    else:
        second.checkpointer.save(4)
        assert second.checkpointer.last_checkpoint.endswith("step-4@generation-1")
        assert (
            FakeTrainer.storage[4]["crucible_checkpoint_event_id"]
            == second.checkpointer.last_checkpoint
        )
        hook.finish(second.checkpointer.last_checkpoint)
        result = account_events(events)
        assert result["retained_steps"] == 4
        assert result["lost_steps"] == 2
        assert sum(event["kind"] == "checkpoint_saved" for event in events) == 3
    assert len(list(root.glob("step-*"))) <= 2


def test_normal_final_state_reuses_existing_same_step_periodic_checkpoint():
    FakeTrainer.storage = {}
    hook, events = make_hook()

    class FourSteps(FakeTrainer):
        def should_continue_training(self):
            return self.step < 4

    trainer = instrument_fault_trainer(FourSteps)(config(), hook=hook, rng=FakeRNG())
    trainer.train()
    assert trainer.step == 4
    saved = [event for event in events if event["kind"] == "checkpoint_saved"]
    assert [event["step"] for event in saved] == [2, 3, 4]
    assert saved[-1]["checkpoint_id"].endswith("step-4@generation-0")
    assert events[-1]["checkpoint_id"] == saved[-1]["checkpoint_id"]
    assert len(list(Path(trainer.checkpointer.folder).glob("step-*"))) == 2
