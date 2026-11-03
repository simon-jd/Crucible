"""Held-out T11 eval of a saved checkpoint, identical to grpo_run's eval.

Same sets (data["gsm8k_eval"], data["math500_eval"]), same prompts, same
sampling (n=4, temperature 0.6, top_p 0.95, max_tokens 512, per-problem seed
10_000 + i) and the same vLLM engine arguments as rl.local.grpo_run (engine
seed 17, generation_config="vllm", bf16), via grpo_run.run_eval itself.

Use it for the "after" numbers of the verl run (checkpoint saved by verl as an
fp32 HF model; loaded here as bf16, like grpo_run's in-memory bf16 sync).

Sanity check before trusting a comparison: on the base snapshot,
    --model <Qwen2.5-0.5B-Instruct snapshot>
must reproduce grpo_run's "before" for the T11 base run
(docs/evidence/rl-grpo-p4/eos-sync-k0-s17/summary.json): gsm8k 0.42625
(truncated 50), math500 0.2875 (truncated 162). A mismatch means the eval
itself differs (vLLM build, engine args, tokenizer) and the after-numbers are
not comparable.

usage (GPU; under the flock):
  bash scripts/with_gpu.sh -- env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \\
    .venv-ml/bin/python -m rl.local.eval_checkpoint --model CKPT_HF_DIR --output OUT
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rl.local import grpo_run as gr

BASE_MODEL = "/home/postedism/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path(".cache/t11/data.json"))
    p.add_argument("--model", required=True, help="HF model dir (e.g. OUT/ckpt/global_step_100/actor/huggingface)")
    # The base tokenizer, not the one verl re-saved with its own transformers version,
    # so the chat template and special tokens match grpo_run byte for byte.
    p.add_argument("--tokenizer", default=BASE_MODEL)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--eval-n", type=int, default=4)
    p.add_argument("--eval-max-tokens", type=int, default=512)
    p.add_argument("--vllm-mem", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from rl.learner.runtime import configure_runtime

    configure_runtime()
    os.environ.update({"VLLM_USE_V1": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"})
    import torch
    from vllm import LLM

    args.output.mkdir(parents=True, exist_ok=False)
    data = json.loads(args.data.read_text())
    started = time.monotonic()
    summary = {"args": {k: str(v) for k, v in vars(args).items()}, "data_sha256": data["sha256"]}
    llm = LLM(
        model=args.model, tokenizer=args.tokenizer, dtype="bfloat16", seed=args.seed,
        gpu_memory_utilization=args.vllm_mem, max_model_len=1024, enforce_eager=True,
        distributed_executor_backend="uni", enable_prefix_caching=True, max_num_seqs=128,
        generation_config="vllm", logprobs_mode="raw_logprobs", max_logprobs=1,
    )  # fmt: skip
    tokenizer = llm.get_tokenizer()
    pool = ThreadPoolExecutor(16)
    sets = {"gsm8k": data["gsm8k_eval"], "math500": data["math500_eval"]}
    summary["after"] = gr.run_eval(llm, tokenizer, sets, args.eval_n, args.eval_max_tokens, pool, args.output / "eval-after.jsonl.gz")
    summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    summary["wall_s"] = time.monotonic() - started
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({"after": {k: (v["pass_at_1"], v["truncated"]) for k, v in summary["after"].items()}}), flush=True)


if __name__ == "__main__":
    main()
