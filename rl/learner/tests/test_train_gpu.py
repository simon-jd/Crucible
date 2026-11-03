"""Opt-in real neutral-rollout -> joined worker -> Qwen FSDP2 learner acceptance."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("CRUCIBLE_LEARNER_GPU") != "1",
        reason="use scripts/test_learner_gpu.sh under the fixed shared lock",
    ),
]
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL = str(
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots"
    / REVISION
)


def test_real_rollout_then_fsdp2_learner():
    # This is before heavy imports in this process. Rollout also configures its
    # actual spawned worker's cache paths before importing vLLM/Torch.
    from rl.learner.runtime import configure_runtime

    environment = configure_runtime()
    from rl.rollout.engine import PolicyCheckpoint, RolloutEngine
    from rl.rollout.sampler import Prompt, SamplingConfig

    config = SamplingConfig(seed=41, max_tokens=8, max_model_len=128)
    prompts = [Prompt("p", "The capital of France is"), Prompt("q", "One plus one is")]
    with RolloutEngine(PolicyCheckpoint(MODEL, REVISION, 7), config) as engine:
        worker = engine._backend._process
        samples = engine.generate(prompts, 2, 7)
    assert not worker.is_alive() and worker.exitcode == 0
    print(
        json.dumps(
            {
                "phase": "rollout_closed_before_learner",
                "pid": worker.pid,
                "exitcode": worker.exitcode,
            }
        ),
        flush=True,
    )

    import resource
    import tempfile
    from importlib import metadata

    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor

    from rl.learner.train import (
        LearnerConfig,
        RolloutGroup,
        _batch,
        _score,
        encode_group,
        load_fsdp2_learner,
    )

    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    rendezvous = tempfile.mktemp(prefix="nccl-", dir=environment["TMPDIR"])
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
        device_id=torch.device("cuda:0"),
    )
    try:
        learner = load_fsdp2_learner(
            MODEL, LearnerConfig(group_size=2, max_length=128), policy_version=7
        )
        assert isinstance(learner.policy, FSDPModule)
        assert all(isinstance(layer, FSDPModule) for layer in learner.policy.model.layers)
        assert all(
            isinstance(p, DTensor) and p.dtype == torch.float32 for p in learner.policy.parameters()
        )
        assert all(
            p.dtype == torch.bfloat16 and not p.requires_grad
            for p in learner.reference.parameters()
        )
        groups = [
            RolloutGroup(
                "actual-1", str(i), prompt, tuple(samples[i * 2 : i * 2 + 2]), (1.0, 0.0), config
            )
            for i, prompt in enumerate(prompts)
        ]
        stale = replace(
            groups[0],
            batch_id="old",
            samples=tuple(replace(sample, policy_version=5) for sample in groups[0].samples),
        )
        encoded = encode_group(groups[0], learner.tokenizer, 128)
        ids, attention, positions, targets, old, mask, _ = _batch(encoded, torch.device("cuda:0"))
        with torch.no_grad():
            initial_logps = _score(learner.policy, ids, attention, positions, targets)
        raw_error = float((initial_logps[mask] - old[mask]).abs().max())
        assert raw_error < 0.15, raw_error
        parameter = learner.policy.model.layers[0].self_attn.q_proj.weight
        before = parameter.to_local().detach().cpu().clone()
        ref_parameter = learner.reference.model.layers[0].self_attn.q_proj.weight
        reference_before = ref_parameter.detach().cpu().clone()
        grad_dtypes = set()

        def check_gradients(optimizer, _args, _kwargs):
            for parameter_group in optimizer.param_groups:
                for p in parameter_group["params"]:
                    if p.grad is not None:
                        grad_dtypes.add(str(p.grad.dtype))

        hook = learner.optimizer.register_step_pre_hook(check_gradients)
        metrics = learner.step(groups + [stale])
        hook.remove()
        assert metrics["dropped_stale"] == 2
        assert metrics["accepted_samples"] == 4
        assert metrics["effective_tokens"] == sum(len(s.token_ids) for s in samples)
        assert metrics["optimizer_steps"] == 1 and metrics["policy_version"] == 8
        change = float((parameter.to_local().detach().cpu() - before).abs().max())
        assert change > 0
        torch.testing.assert_close(ref_parameter.detach().cpu(), reference_before, rtol=0, atol=0)
        assert all(p.grad is None for p in learner.reference.parameters())
        assert grad_dtypes == {"torch.float32"}
        state_bytes = sum(
            value.numel() * value.element_size()
            for state in learner.optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        )
        state_dtypes = {
            str(value.dtype)
            for state in learner.optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        }
        assert state_dtypes == {"torch.float32"}
        second = learner.step(groups + [stale])
        assert second["policy_version"] == 9 and second["dropped_stale"] == 2
        all_stale = learner.step(groups)
        assert all_stale["optimizer_steps"] == 0 and all_stale["dropped_stale"] == 4
        print(
            json.dumps(
                {
                    "phase": "learner_complete",
                    "model": MODEL,
                    "revision": REVISION,
                    "versions": {
                        p: metadata.version(p) for p in ("torch", "triton", "transformers")
                    },
                    "cuda": torch.version.cuda,
                    "nccl": torch.cuda.nccl.version(),
                    "world_size": dist.get_world_size(),
                    "device": torch.cuda.get_device_name(),
                    "fully_sharded_layers": len(learner.policy.model.layers),
                    "parameter_count": sum(p.numel() for p in learner.policy.parameters()),
                    "parameter_dtype": str(parameter.dtype),
                    "gradient_dtypes": sorted(grad_dtypes),
                    "reference_dtype": str(ref_parameter.dtype),
                    "optimizer_dtypes": sorted(state_dtypes),
                    "optimizer_state_bytes": state_bytes,
                    "raw_logprob_max_error": raw_error,
                    "parameter_max_change": change,
                    "step1": metrics,
                    "step2": second,
                    "all_stale_step": all_stale,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                    "runtime_environment": environment,
                    "checkpoints_written": 0,
                }
            ),
            flush=True,
        )
        # A real AdamW update followed by a caller hook failure is uncertain state.
        # Accept the original rollout at age2 for this isolated failure exercise.
        learner.config = replace(learner.config, max_staleness=2)
        optimizer = learner.optimizer

        def fail_after_update(_optimizer, _args, _kwargs):
            raise RuntimeError("injected failure after actual AdamW update")

        hook = optimizer.register_step_post_hook(fail_after_update)
        try:
            with pytest.raises(RuntimeError, match="after actual AdamW update"):
                learner.step(groups)
        finally:
            hook.remove()
        assert learner.phase_status == "failed"
        assert learner.last_committed_version == 9
        for attribute in ("policy", "optimizer", "policy_version"):
            with pytest.raises(RuntimeError, match="phase"):
                getattr(learner, attribute)
        with pytest.raises(RuntimeError, match="phase"):
            learner.step([])
        assert all(p.grad is None for group in optimizer.param_groups for p in group["params"])
        print(
            json.dumps(
                {
                    "phase": "failed_learner_guarded",
                    "phase_status": learner.phase_status,
                    "last_committed_version_diagnostic_only": learner.last_committed_version,
                    "failure_reason": learner.failure_reason,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                }
            ),
            flush=True,
        )
        del learner
    finally:
        dist.destroy_process_group()
        Path(rendezvous).unlink(missing_ok=True)
    assert not dist.is_initialized()
    print(
        "learner NCCL group destroyed; process exit returns GPU ownership to fixed guard",
        flush=True,
    )
