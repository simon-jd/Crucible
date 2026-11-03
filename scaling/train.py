"""Thin, deferred-import adapter for pinned TorchTitan v0.2.0 Trainer.train_step.

Importing this module does not import torch, initialize CUDA, or launch training.
Training is paused under the $0 resource budget; main fails before importing torch unless
an operator explicitly acknowledges a separate future multi-node authorization.
"""

import importlib.metadata
import importlib.util
import json
import math
import os
import socket
import subprocess
import time
from pathlib import Path

from cluster.drain.metrics import StepTimeHook


class CompletedStepTimer:
    """Synchronized train-step windows, not asynchronous CUDA submission timings.

    Includes dataloading, gradient accumulation and step logging in train_step;
    excludes checkpoint/validation/GC outside that call. Explicit per-step sync
    adds instrumentation overhead; use this same wrapper in every comparison.
    One hook per node; every rank must participate in global_max at each window.
    """

    def __init__(
        self,
        *,
        hook,
        synchronize,
        tokens_per_step,
        global_max,
        emit,
        clock=time.perf_counter,
        warmup_steps=50,
        profile=None,
    ):
        if type(tokens_per_step) is not int or tokens_per_step <= 0:
            raise ValueError("positive distinct global tokens per optimizer step required")
        if type(warmup_steps) is not int or warmup_steps < 0 or warmup_steps % 50:
            raise ValueError("warmup_steps must be a nonnegative multiple of 50")
        self.hook, self.synchronize, self.clock = hook, synchronize, clock
        self.profile = profile
        self.tokens_per_step, self.global_max, self.emit = tokens_per_step, global_max, emit
        self.warmup_steps, self.last_step, self.seconds = warmup_steps, 0, 0.0

    def measure(self, completed_step, operation):
        if type(completed_step) is not int or completed_step != self.last_step + 1:
            raise ValueError("fresh training must complete contiguous steps from 1")
        self.synchronize()
        if self.profile is not None and completed_step == 51:
            self.profile(True)
        start = self.clock()
        result = operation()  # Failed/cancelled steps never publish.
        self.synchronize()
        elapsed = self.clock() - start
        if self.profile is not None and completed_step == 55:
            self.profile(False)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("invalid synchronized wall time")
        self.seconds += elapsed
        self.last_step = completed_step
        if completed_step % 50 == 0:
            if self.hook is not None:
                self.hook.observe(completed_step, self.seconds / 50)
            max_seconds = self.global_max(self.seconds)
            if not math.isfinite(max_seconds) or max_seconds <= 0:
                raise ValueError("invalid global window wall time")
            if completed_step > self.warmup_steps:
                self.emit(
                    {
                        "completed_steps": completed_step,
                        "window_steps": 50,
                        "window_max_rank_seconds": max_seconds,
                        "global_tokens_per_s": 50 * self.tokens_per_step / max_seconds,
                        "timing_scope": "synchronized_train_step",
                    }
                )
            self.seconds = 0.0
        return result


def instrument_trainer(base):
    """Subclass only train_step; preserve upstream initialization and train loop."""

    class InstrumentedTrainer(base):
        def train_step(self, data_iterator):
            return self.crucible_timer.measure(
                self.step, lambda: super(InstrumentedTrainer, self).train_step(data_iterator)
            )

    return InstrumentedTrainer


def main():
    if os.environ.get("CRUCIBLE_MULTINODE_APPROVED") != "1":
        raise SystemExit("multi-node training paused: separate user budget authorization required")
    source = Path(os.environ["TORCHTITAN_SOURCE"]).resolve()
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(source), "diff", "HEAD", "--exit-code"], check=False, capture_output=True
    ).returncode
    from scaling.experiment import COMMIT

    spec = importlib.util.find_spec("torchtitan")
    if commit != COMMIT or dirty or spec is None or spec.origin is None:
        raise ValueError("exact clean pinned TorchTitan source checkout required")
    if not Path(spec.origin).resolve().is_relative_to(source / "torchtitan"):
        raise ValueError("installed TorchTitan must resolve to the pinned source checkout")
    for package, version in [
        ("torch", "2.10.0.dev20251019+cu126"),
        ("torchao", "0.15.0.dev20251015+cu126"),
    ]:
        if importlib.metadata.version(package) != version:
            raise ValueError(f"{package} must match candidate release dependency {version}")
    # Lazy imports: offline tests and plan generation need neither Torch nor TorchTitan.
    import torch
    from torchtitan.config import ConfigManager
    from torchtitan.tools.logging import init_logger
    from torchtitan.train import Trainer

    init_logger()
    config = ConfigManager().parse_args()
    if config.checkpoint.enable or config.training.global_batch_size <= 0:
        raise ValueError(
            "scaling timing requires checkpoint-disabled fresh training and explicit global batch"
        )
    run_id = os.environ["CRUCIBLE_RUN_ID"]
    output = Path(os.environ["CRUCIBLE_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    trainer = None
    try:
        trainer = instrument_trainer(Trainer)(config)
        rank = torch.distributed.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        node = socket.gethostname()
        hook = (
            StepTimeHook(node=node, run_id=run_id, path=output / f"{node}.prom")
            if local_rank == 0
            else None
        )

        def global_max(seconds):
            value = torch.tensor(seconds, device=trainer.device, dtype=torch.float64)
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
            return value.item()

        def emit(record):
            if rank == 0:
                record["run_id"] = run_id
                with (output / "throughput.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")

        trainer.crucible_timer = CompletedStepTimer(
            hook=hook,
            synchronize=lambda: torch.cuda.synchronize(trainer.device),
            tokens_per_step=config.training.global_batch_size * config.training.seq_len,
            global_max=global_max,
            emit=emit,
            profile=(
                lambda start: (
                    torch.cuda.cudart().cudaProfilerStart()
                    if start
                    else torch.cuda.cudart().cudaProfilerStop()
                )
            )
            if os.environ.get("CRUCIBLE_PROFILE") == "1"
            else None,
        )
        trainer.train()
    finally:
        if trainer is not None:
            trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
