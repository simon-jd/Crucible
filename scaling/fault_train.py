"""Checkpoint-enabled thin adapter for the pinned TorchTitan v0.2.0 Trainer.

No upstream train loop is copied. Wrap its public train_step/save/load calls;
only synchronous, full-state native checkpoints are accepted. CLI execution is
hard-closed while the $0 resource budget pauses multi-node fault experiments.
"""

import copy
import json
import math
import os
from pathlib import Path

from cluster.faults.plan import PAUSED
from cluster.faults.supervisor import CampaignSupervisor  # noqa: F401 - public re-export
from scaling.checkpoint_retention import CheckpointRetention


class CampaignWindow:
    """Fixed 7200-second campaign with explicit final-save and recovery budgets.

    All clocks/waits are supplied by the coordinator. No device commands, timers,
    sleeps or background threads are started by this object.
    """

    def __init__(
        self,
        *,
        started_at,
        clock,
        wait_until,
        final_checkpoint_reserve_seconds=120,
        recovery_budget_seconds=300,
        coordinate=lambda operation: operation(),
    ):
        if (
            type(started_at) not in (int, float)
            or not math.isfinite(started_at)
            or started_at < 0
            or type(final_checkpoint_reserve_seconds) is not int
            or not 0 < final_checkpoint_reserve_seconds < 7200
            or type(recovery_budget_seconds) is not int
            or recovery_budget_seconds <= 0
        ):
            raise ValueError(
                "valid coordinator origin and explicit checkpoint/recovery budgets required"
            )
        self.started_at, self.clock, self.wait_until = started_at, clock, wait_until
        self.deadline = started_at + 7200
        self.training_deadline = self.deadline - final_checkpoint_reserve_seconds
        self.recovery_budget = recovery_budget_seconds
        self.coordinate = coordinate

    def _check_deadline(self):
        now = self.clock()
        if not math.isfinite(now) or not self.started_at <= now <= self.deadline:
            raise TimeoutError("two-hour campaign deadline exceeded or clock invalid")
        return now

    def check_deadline(self):
        return self.coordinate(self._check_deadline)

    def should_train(self):
        return self.coordinate(lambda: self._check_deadline() < self.training_deadline)

    def check_recovery(self, failed_at):
        def check():
            now = self._check_deadline()
            if failed_at is None or not failed_at <= now <= failed_at + self.recovery_budget:
                raise TimeoutError("confirmed-failure-to-checkpoint-recovery budget exceeded")
            return now

        return self.coordinate(check)

    def finish(self):
        # Worker completion precedes teardown/exit. Only the surviving supervisor
        # waits for the observation window, after the entire owned scope is empty.
        return self.check_deadline()


def torch_coordinator(torch, device):
    """Every rank participates; only rank zero observes time or mutates shared files."""

    def coordinate(operation):
        packet = [None]
        if torch.distributed.get_rank() == 0:
            try:
                packet[0] = {"ok": True, "value": operation()}
            except Exception as exc:  # noqa: BLE001 - broadcast the same failure before any branch
                packet[0] = {
                    "ok": False,
                    "error": str(exc),
                    "timeout": isinstance(exc, TimeoutError),
                }
        torch.distributed.broadcast_object_list(packet, src=0, device=device)
        if not packet[0]["ok"]:
            exception = TimeoutError if packet[0]["timeout"] else ValueError
            raise exception(packet[0]["error"])
        return packet[0]["value"]

    return coordinate


def validate_config(config):
    c = config.checkpoint
    if (
        not c.enable
        or c.load_only
        or c.async_mode.lower() != "disabled"
        or c.last_save_model_only
        or c.last_save_in_hf
        or c.initial_load_model_only
        or c.initial_load_in_hf
        or c.initial_load_path
        or c.exclude_from_loading
        or c.create_seed_checkpoint
        or c.keep_latest_k != 2
        or type(c.interval) is not int
        or c.interval <= 0
        or config.fault_tolerance.enable
    ):
        raise ValueError(
            "fault training requires synchronous full-state native checkpoints, "
            "keep_latest_k=2, no TorchFT/seed/partial/external load"
        )
    if config.checkpoint.load_step == 0 or config.checkpoint.load_step < -1:
        raise ValueError("step-0 seed is model-only upstream; use a full checkpoint > 0")


class ObservedCheckpointer:
    def __init__(self, manager, trainer, hook, config):
        self.manager, self.trainer, self.hook, self.config = manager, trainer, hook, config
        self.last_checkpoint = None
        self.retention = CheckpointRetention(manager.folder, hook.run_id)

    def __getattr__(self, name):
        return getattr(self.manager, name)

    def identity(self, step):
        # Mirrors the pinned manager's native step-N layout; no private API call.
        return str(Path(self.manager.folder) / f"step-{step}")

    def save(self, curr_step, last_step=False):
        if self.trainer.crucible_campaign:
            self.trainer.crucible_campaign.check_deadline()
        c = self.config.checkpoint
        due = (
            last_step
            or curr_step % c.interval == 0
            or (curr_step == 1 and c.enable_first_step_checkpoint)
        )
        if not due:
            return self.manager.save(curr_step, last_step=last_step)
        if curr_step != self.hook.last_step or curr_step != self.trainer.step:
            raise ValueError("save cannot include an incomplete optimizer step")
        checkpoint = self.identity(curr_step)
        # A logical step re-saved after rollback needs a new generation identity.
        generation = sum(s == curr_step for s in self.hook.checkpoints.values())
        event_id = checkpoint + f"@generation-{generation}"
        self.trainer.crucible_checkpoint_event_id = event_id
        # Rank zero alone holds the filesystem lock across the collective save.
        # Followers only participate in broadcast; they never wait on that lock.
        coordinate = getattr(self.hook, "coordinate", None)
        if coordinate is None:
            if self.hook.world_size != 1:
                raise ValueError("distributed checkpoint retention requires rank-zero coordination")
            coordinate = lambda operation: operation()

        def save():
            coordinate(
                lambda: self.retention.begin(
                    curr_step, protected_checkpoint_id=self.last_checkpoint
                )
            )
            try:
                result = self.manager.save(curr_step, last_step=last_step)
                coordinate(lambda: self.retention.confirm(curr_step, event_id))
                return result
            finally:
                if self.hook.rank == 0:
                    self.retention.end()

        result = self.hook.save(event_id, curr_step, save)
        self.last_checkpoint = event_id
        return result

    def load(self, step=-1):
        if not self.hook.failed:
            if step != -1 or self.trainer.step != 0:
                raise ValueError("fresh run requires empty checkpoint directory")
            result = self.manager.load(step=step)
            if result is not False:
                raise ValueError("unexpected existing checkpoint in fresh run")
            return result
        if type(step) is not int or step <= 0:
            raise ValueError("restart must explicitly name an actual full checkpoint step")
        prefix = self.identity(step) + "@generation-"
        known = [key for key in self.hook.checkpoints if key.startswith(prefix)]
        if not known:
            raise ValueError("restart checkpoint has no successful save event")
        checkpoint = known[-1]

        failed_at = self.hook.failed_at

        def load():
            if self.trainer.crucible_campaign:
                self.trainer.crucible_campaign.check_recovery(failed_at)
            result = self.manager.load(step=step)
            if (
                result is not True
                or self.trainer.step != step
                or self.trainer.crucible_checkpoint_event_id != checkpoint
            ):
                raise ValueError("checkpoint load failed or restored trainer step mismatches")
            if self.trainer.crucible_campaign:
                self.trainer.crucible_campaign.check_recovery(failed_at)
            return result

        result = self.hook.restore(checkpoint, step, load)
        if self.trainer.crucible_campaign:
            self.trainer.crucible_campaign.check_recovery(failed_at)
        self.last_checkpoint = checkpoint
        return result


def instrument_fault_trainer(base):
    """Caller supplies a persistent coordinator hook; all ranks participate in sync.

    Construct the Trainer normally with the fixed upstream config. Replacements
    reuse/replay the same run journal and set an explicit checkpoint.load_step.
    The hook must synchronize CUDA AND all ranks before and after each operation.
    """

    class FaultTrainer(base):
        def __init__(self, config, *, hook=None, rng=None, runtime_factory=None, campaign=None):
            if runtime_factory is None and (hook is None or rng is None):
                raise ValueError(
                    "explicit hook/RNG or post-initialization runtime factory required"
                )
            if runtime_factory is not None and (hook is not None or rng is not None):
                raise ValueError("runtime factory and prebuilt hook/RNG are mutually exclusive")
            self.crucible_rng, self.crucible_fault_hook = rng, hook
            self.crucible_checkpoint_event_id = None
            self.crucible_campaign = campaign
            validate_config(config)  # Reject unsupported modes before upstream initializes GPUs.
            runtime_config = copy.deepcopy(config)
            # The supported zero setting disables upstream's asynchronous purge.
            # This adapter owns strict, synchronous two-checkpoint retention.
            runtime_config.checkpoint.keep_latest_k = 0
            super().__init__(runtime_config)
            if runtime_factory is not None:
                try:
                    hook, rng = runtime_factory(self)
                except Exception:
                    self.close()
                    raise
                self.crucible_rng = rng
            self.crucible_fault_hook = hook
            self.checkpointer = ObservedCheckpointer(self.checkpointer, self, hook, config)

        def state_dict(self):
            state = super().state_dict()
            state["crucible_checkpoint_event_id"] = self.crucible_checkpoint_event_id
            state[f"crucible_rng_rank_{self.crucible_fault_hook.rank}"] = {
                "world_size": self.crucible_fault_hook.world_size,
                "state": self.crucible_rng.state_dict(),
            }
            return state

        def load_state_dict(self, state):
            key = f"crucible_rng_rank_{self.crucible_fault_hook.rank}"
            local = state[key]
            if local["world_size"] != self.crucible_fault_hook.world_size:
                raise ValueError("fault restart requires unchanged world size and rank mapping")
            self.crucible_rng.load_state_dict(local["state"])
            self.crucible_checkpoint_event_id = state["crucible_checkpoint_event_id"]
            super().load_state_dict(state)

        def should_continue_training(self):
            if self.crucible_campaign is not None:
                return self.crucible_campaign.should_train()
            return super().should_continue_training()

        def train_step(self, data_iterator):
            return self.crucible_fault_hook.step(
                self.step, lambda: super(FaultTrainer, self).train_step(data_iterator)
            )

        def train(self):
            hook = self.crucible_fault_hook
            if not hook.started:
                hook.start()
            try:
                result = super().train()
                if hook.last_step != self.step:
                    raise ValueError("upstream canceled final step; terminal state is incomplete")
                # Upstream saves last step on normal target completion. On early clean
                # return, force synchronous full-state save via its public save API.
                if self.checkpointer.last_checkpoint is None or (
                    hook.checkpoints[self.checkpointer.last_checkpoint] != hook.last_step
                ):
                    self.checkpointer.save(hook.last_step, last_step=True)
                if self.crucible_campaign:
                    self.crucible_campaign.finish()
                if self.crucible_campaign:
                    hook.complete_worker(self.checkpointer.last_checkpoint)
                else:
                    hook.finish(self.checkpointer.last_checkpoint)
                return result
            except Exception as exc:
                if not hook.failed:
                    hook.failure(str(exc))
                raise

    return FaultTrainer


class RNGState:
    """Rank-local Python, NumPy and Torch CPU/current-CUDA RNG checkpoint state.

    Dependencies are supplied by the future runtime; this module never imports
    Torch or initializes CUDA. Distinct rank keys avoid saving rank-zero RNG for all.
    """

    def __init__(self, *, torch, numpy, device):
        import random

        self.torch, self.numpy, self.device, self.random = torch, numpy, device, random

    def state_dict(self):
        return {
            "python": self.random.getstate(),
            "numpy": self.numpy.random.get_state(),
            "torch_cpu": self.torch.get_rng_state(),
            "torch_cuda": self.torch.cuda.get_rng_state(self.device),
        }

    def load_state_dict(self, state):
        self.random.setstate(state["python"])
        self.numpy.random.set_state(state["numpy"])
        self.torch.set_rng_state(state["torch_cpu"])
        self.torch.cuda.set_rng_state(state["torch_cuda"], self.device)


def make_torch_hook(*, torch, device, run_id, clock_id, clock, emit, prior_events=None):
    """Explicit future runtime wiring; the clock must survive reporter replacement.

    No imports or device operations occur until the caller invokes this factory.
    The barrier makes rank-zero completion include all participating training ranks.
    """
    from scaling.goodput import FaultTrainingHook

    def synchronize():
        torch.cuda.synchronize(device)
        torch.distributed.barrier()
        torch.cuda.synchronize(device)

    kwargs = {
        "emit": emit,
        "clock": clock,
        "synchronize": synchronize,
        "rank": torch.distributed.get_rank(),
    }
    if prior_events is not None:
        if (
            prior_events[0].get("world_size") != torch.distributed.get_world_size()
            or prior_events[0].get("run_id") != run_id
            or prior_events[0].get("clock_id") != clock_id
        ):
            raise ValueError("restart must preserve world size, run ID and coordinator clock")
        hook = FaultTrainingHook.resume(prior_events, **kwargs)
        hook.coordinate = torch_coordinator(torch, device)
        return hook
    hook = FaultTrainingHook(
        run_id, clock_id=clock_id, world_size=torch.distributed.get_world_size(), **kwargs
    )
    hook.coordinate = torch_coordinator(torch, device)
    return hook


def create_fault_trainer(
    base, config, *, torch, numpy, run_id, clock_id, clock, emit, wait_until, prior_events=None
):
    """Start coordinator accounting before upstream distributed/device initialization.

    Torchrun RANK/WORLD_SIZE identify the reporter before process-group creation;
    make_torch_hook later checks actual initialized world size against this origin.
    """
    if prior_events is None and os.environ.get("CRUCIBLE_CAMPAIGN_ORIGIN_JSON"):
        origin = json.loads(os.environ["CRUCIBLE_CAMPAIGN_ORIGIN_JSON"])
        if origin.get("run_id") != run_id or origin.get("clock_id") != clock_id:
            raise ValueError("launcher origin must match the persistent run and clock")
        prior_events = [origin]
    if prior_events is None:
        origin = {
            "schema": 1,
            "rank": 0,
            "kind": "run_start",
            "run_id": run_id,
            "clock_id": clock_id,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "at": clock(),
        }
        if int(os.environ.get("RANK", "0")) == 0:
            emit(origin)
        prior_events = [origin]
    campaign = CampaignWindow(started_at=prior_events[0]["at"], clock=clock, wait_until=wait_until)

    def runtime(trainer):
        hook = make_torch_hook(
            torch=torch,
            device=trainer.device,
            run_id=run_id,
            clock_id=clock_id,
            clock=clock,
            emit=emit,
            prior_events=prior_events,
        )
        campaign.coordinate = hook.coordinate
        return hook, RNGState(torch=torch, numpy=numpy, device=trainer.device)

    return instrument_fault_trainer(base)(config, runtime_factory=runtime, campaign=campaign)


def main():
    raise SystemExit(PAUSED)


if __name__ == "__main__":
    main()
