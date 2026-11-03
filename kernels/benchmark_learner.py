"""Matched real FSDP2 learner fixture; run only via scripts/test_logprob_gpu.sh.

This is a short performance fixture, not rollout/heldout quality evaluation.
Each variant reloads the immutable base and a fresh AdamW, warms up one full step,
then measures one full step. No weights, optimizer state or downloads are written.
"""

from rl.learner.runtime import configure_runtime

ENVIRONMENT = configure_runtime()  # before heavy imports/compiler initialization

import gc
import hashlib
import json
import os
import resource
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from rl.learner import train
from rl.rollout.sampler import Prompt, Sample, SamplingConfig

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL = (
    Path(ENVIRONMENT["HF_HUB_CACHE"]) / "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / REVISION
)
SEED = 1010
PEAK_LIMIT = 18_000_000_000  # observed allocator threshold, not a physical hard cap


def gpu_preflight():
    if os.environ.get("CRUCIBLE_LOGPROB_GPU") != "1":
        raise RuntimeError("run the fixed shared-lock GPU wrapper")
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    if free < 19_000_000_000:
        raise RuntimeError(f"need 19GB free for 18GB observed peak plus margin; available {free}")
    print(
        json.dumps(
            {
                "phase": "gpu_preflight",
                "free_bytes": free,
                "total_bytes": total,
                "device": torch.cuda.get_device_name(),
                "cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(),
                "versions": {
                    p: metadata.version(p)
                    for p in ("torch", "triton", "transformers", "vllm", "numpy")
                },
                "driver": subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
                ).strip(),
                "environment": ENVIRONMENT,
                "seed": SEED,
            }
        ),
        flush=True,
    )


def fixture(learner):
    """Two distinct autoregressive completions with actual neutral raw old scores."""
    prompt = Prompt("perf", "The capital of France is")
    prefix = learner.tokenizer.encode(prompt.text)
    samples = []
    for branch in range(2):
        ids, generated, old = list(prefix), [], []
        for index in range(8):
            batch = torch.tensor([ids], device="cuda")
            with torch.no_grad():
                logits = (
                    learner.policy(
                        input_ids=batch, attention_mask=torch.ones_like(batch), use_cache=False
                    )
                    .logits[0, -1]
                    .float()
                )
                # Only the first token branches; all subsequent tokens are greedy.
                token = int(logits.topk(2).indices[branch if index == 0 else 0])
                logprob = float(logits.log_softmax(-1)[token])
            ids.append(token)
            generated.append(token)
            old.append(logprob)
            if token == learner.tokenizer.eos_token_id:
                break
        samples.append(
            Sample("perf", learner.tokenizer.decode(generated), tuple(generated), tuple(old), 0)
        )
    assert samples[0].token_ids != samples[1].token_ids
    return [
        train.RolloutGroup(
            "perf-batch",
            "perf-group",
            prompt,
            tuple(samples),
            (1.0, 0.0),
            SamplingConfig(seed=SEED, max_tokens=8, max_model_len=128),
        )
    ]


def _one_variant(variant, groups):
    torch.manual_seed(SEED)
    learner = train.load_fsdp2_learner(
        str(MODEL),
        train.LearnerConfig(group_size=2, microbatch_size=2, max_length=128, max_staleness=2),
        learning_rate=1e-5,
    )
    if groups is None:
        groups = fixture(learner)
        print(
            json.dumps(
                {
                    "phase": "fixed_performance_fixture",
                    "groups": [asdict(g) for g in groups],
                    "revision": REVISION,
                    "fixture_sha256": hashlib.sha256(repr(groups).encode()).hexdigest(),
                }
            ),
            flush=True,
        )
    scorer = train._score_reference if variant == "baseline" else train._score
    with patch.object(train, "_score", scorer):
        warmup = learner.step(groups)  # same real optimizer/finite checks/cleanup in both variants
        parameter = learner.policy.model.layers[0].self_attn.q_proj.weight
        before = parameter.to_local().detach().clone()
        observed = {}

        def capture_gradient(_optimizer, _args, _kwargs):
            grad = parameter.grad.to_local()
            observed["gradient_max_abs"] = float(grad.abs().max())

        hook = learner.optimizer.register_step_pre_hook(capture_gradient)
        torch.cuda.synchronize()
        start_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        metrics = learner.step(groups)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        hook.remove()
        changed = float((parameter.to_local().detach() - before).abs().max())
        assert observed["gradient_max_abs"] > 0 and changed > 0
        assert learner.phase_status == "ready" and metrics["policy_version"] == 2
        assert warmup["policy_version"] == 1 and warmup["optimizer_steps"] == 1
        assert metrics["optimizer_steps"] == 1 and metrics["accepted_samples"] == 2
        assert metrics["dropped_stale"] == 0
        assert metrics["effective_tokens"] == sum(len(s.token_ids) for s in groups[0].samples)
        assert peak < PEAK_LIMIT, f"observed allocator peak exceeds planned threshold: {peak}"
        result = {
            "variant": variant,
            "seconds": elapsed,
            "peak_allocated_bytes": peak,
            "start_allocated_bytes": start_allocated,
            "peak_increment_bytes": peak - start_allocated,
            "peak_reserved_bytes": reserved,
            "gradient_max_abs": observed["gradient_max_abs"],
            "parameter_max_change": changed,
            "warmup_metrics": warmup,
            "measured_metrics": metrics,
        }
    parameter = None
    del hook, before, learner, capture_gradient
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    result["after_cleanup_allocated_bytes"] = torch.cuda.memory_allocated()
    return groups, result


def main():
    gpu_preflight()
    rendezvous = tempfile.mktemp(prefix="task10-nccl-", dir=ENVIRONMENT["TMPDIR"])
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
        device_id=torch.device("cuda:0"),
    )
    groups, results = None, []
    try:
        for round_index in range(3):
            order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
            for variant in order:
                groups, result = _one_variant(variant, groups)
                result["round"] = round_index + 1
                results.append(result)
                print(json.dumps({"phase": "learner_measurement", **result}), flush=True)
        summary = {}
        for variant in ("baseline", "candidate"):
            rows = [row for row in results if row["variant"] == variant]
            summary[variant] = {
                key: {
                    "median": statistics.median(row[key] for row in rows),
                    "min": min(row[key] for row in rows),
                    "max": max(row[key] for row in rows),
                }
                for key in ("seconds", "peak_allocated_bytes", "peak_increment_bytes")
            }
        print(
            json.dumps(
                {
                    "phase": "learner_summary",
                    "summary": summary,
                    "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                    "checkpoints": 0,
                    "resident_services": 0,
                }
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group()
        Path(rendezvous).unlink(missing_ok=True)
    assert not dist.is_initialized()
    print(
        "Task10 learner NCCL group destroyed; process exit releases GPU through fixed guard",
        flush=True,
    )


if __name__ == "__main__":
    main()
