"""Opt-in real cached model acceptance, invoked only through with_gpu.sh."""

import json
import math
import os
from pathlib import Path

import pytest

from rl.rollout.engine import PolicyCheckpoint, RolloutEngine
from rl.rollout.sampler import Prompt, SamplingConfig

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("CRUCIBLE_ROLLOUT_GPU") != "1",
        reason="opt in only inside scripts/with_gpu.sh",
    ),
]
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL = str(
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots"
    / REVISION
)


def test_real_local_rollout():
    from importlib import metadata

    config = SamplingConfig(seed=123, temperature=0.8, top_p=0.9, max_tokens=16)
    prompts = [
        Prompt("a", "List several unusual names for a friendly robot:"),
        Prompt("b", "The best way to learn mathematics is"),
    ]
    checkpoint = PolicyCheckpoint(MODEL, REVISION, 7)
    with RolloutEngine(checkpoint, config) as engine:
        worker = engine._backend._process
        cache_environment = engine._backend.cache_environment
        cache_root = Path.cwd() / ".cache/task7/runtime"
        for key, value in cache_environment.items():
            assert Path(value).is_relative_to(cache_root), (key, value)
            assert Path(value).is_dir()
        assert "CUDA_CACHE_PATH" in cache_environment
        assert "XDG_CACHE_HOME" in cache_environment
        first = engine.generate(prompts, 3, 7)
        again = engine.generate(prompts, 3, 7)
        assert first == again, "stochastic generation must repeat for the same seed"
        assert [s.prompt_id for s in first] == ["a"] * 3 + ["b"] * 3
        assert {s.policy_version for s in first} == {7}
        assert len({s.token_ids for s in first[:3]}) > 1, "must exercise stochastic draws"
        assert all(0 < len(s.token_ids) == len(s.logprobs) <= 16 for s in first)
        assert all(math.isfinite(lp) and lp <= 0 for s in first for lp in s.logprobs)
        with pytest.raises(ValueError, match="loaded policy"):
            engine.generate(prompts, 3, 8)
        print(f"owned live worker pid={worker.pid}", flush=True)
    assert not worker.is_alive()
    assert worker.exitcode == 0
    # Independent chosen-token raw-model scoring after the rollout worker releases
    # CUDA; a BF16 vLLM vs FP32 transformers tolerance is explicit, not bit equality.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    sample = first[0]
    prompt_ids = tokenizer.encode(prompts[0].text)
    chosen = list(sample.token_ids[:8])
    ids = torch.tensor([prompt_ids + chosen])
    with torch.inference_mode():
        logits = model(ids).logits[0, len(prompt_ids) - 1 : len(prompt_ids) + len(chosen) - 1]
        reference = (
            torch.log_softmax(logits.float(), -1)
            .gather(1, torch.tensor(chosen).unsqueeze(1))
            .squeeze(1)
            .tolist()
        )
    errors = [abs(a - b) for a, b in zip(reference, sample.logprobs, strict=False)]
    print(
        json.dumps(
            {
                "model": MODEL,
                "revision": REVISION,
                "policy_version": 7,
                "versions": {p: metadata.version(p) for p in ["vllm", "transformers", "torch"]},
                "sampling": vars(config),
                "group_size": 3,
                "samples": len(first),
                "chosen_ids": chosen,
                "vllm_raw_logprobs": list(sample.logprobs[:8]),
                "transformers_fp32_logprobs": reference,
                "max_abs_logprob_error": max(errors),
                "worker_cache_environment": cache_environment,
                "worker_pid": worker.pid,
                "worker_join_exitcode": worker.exitcode,
            }
        ),
        flush=True,
    )
    assert max(errors) < 0.15
