"""E1: multi-GPU NCCL weight-sync correctness, learner GPU -> vLLM on another GPU.

Two processes, each with its own GPU (CUDA_VISIBLE_DEVICES set by the launcher):

  --role learner  FSDP2 learner at policy version 7 (the frozen Task 9 protocol);
                  exports BF16 v7, broadcasts it over the pair group (NCCL) with
                  WeightSync; one controlled optimizer step to v8; broadcasts v8.
                  Then builds the HF BF16 SDPA reference (batch 1, TF32 off, highest
                  FP32 matmul precision) from exactly the BF16 tensors it sent and
                  compares it with the vLLM logits the rollout returns.
  --role rollout  in-process vLLM 0.10.2 V1 (BF16, eager, raw_logits, batch 2) on
                  the base snapshot; receives each version into a MemoryStore,
                  loads it with model.load_weights, returns full-vocabulary logits
                  at the last prompt position for the 6 protocol prompts.

Acceptance is the frozen protocol docs/evidence/raw/task9-confirmation-protocol.json
(per prompt and version: every one of 151,936 coordinates within
0.375 + 0.01*|ref|, RMS <= 0.10, mean abs <= 0.08, finite, identical token IDs).
WeightSync uses sha256 digests here, so both ranks also prove the received bytes
equal the sent bytes before the candidate commits. With --bucket-mb N both roles use
BucketedSync (rl/sync/bucketed.py) instead: the packed BF16 state in N MiB broadcasts,
committed only when both ranks' SHA-256 manifests agree. The learner writes result.json,
logits.npz and prints one line "ROLE_RESULT <json>".
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/evidence/raw/task9-confirmation-protocol.json"
VOCAB = 151936
VERSIONS = (7, 8)


def compare(actual, reference, limits):
    """Per-row protocol metrics for two [rows, vocab] float tensors."""
    import torch

    rows = []
    for index in range(reference.shape[0]):
        observed, expected = actual[index].double(), reference[index].double()
        error = (observed - expected).abs()
        finite = bool(torch.isfinite(observed).all() and torch.isfinite(expected).all())
        bound = limits["absolute_tolerance"] + limits["relative_tolerance"] * expected.abs()
        row = {
            "prompt_index": index,
            "coordinates": error.numel(),
            "finite": finite,
            "max_abs": float(error.max()),
            "mean_abs": float(error.mean()),
            "rms": float(error.square().mean().sqrt()),
            "failed_coordinates": int((error > bound).sum()),
        }
        row["passed"] = (
            finite
            and row["failed_coordinates"] == 0
            and row["rms"] <= limits["maximum_rms_error"]
            and row["mean_abs"] <= limits["maximum_mean_absolute_error"]
        )
        rows.append(row)
    return rows


def _barrier(group, device):
    import torch
    import torch.distributed as dist

    t = time.monotonic()
    flag = torch.zeros(1, device=device)
    dist.all_reduce(flag, group=group)
    torch.cuda.synchronize()
    return time.monotonic() - t


def learner(args):
    from rl.learner.runtime import configure_runtime

    configure_runtime()
    import tempfile

    import numpy as np
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM

    from rl.learner.train import LearnerConfig, RolloutGroup, load_fsdp2_learner
    from rl.rollout.sampler import Prompt, Sample, SamplingConfig
    from rl.sync.acceptance import load_protocol
    from rl.sync.export import base_schema, ready_bf16_tensors
    from rl.sync.memory_store import MemoryStore
    from rl.sync.pair_group import destroy_pair_group, init_pair_group

    protocol = load_protocol(PROTOCOL)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    rendezvous = tempfile.mktemp(prefix="e1-default-")
    dist.init_process_group("nccl", init_method="file://" + rendezvous, rank=0, world_size=1)
    pair, kv = init_pair_group("nccl", "0.0.0.0", args.port, 0, timeout=args.timeout)
    result = {"role": "learner", "host": os.uname().nodename, "env": _nccl_env()}
    try:
        learner_ = load_fsdp2_learner(
            args.model,
            LearnerConfig(group_size=2, max_length=128),
            policy_version=7,
            learning_rate=1e-4,
        )
        schema = base_schema(args.model, torch.bfloat16)
        store = MemoryStore(schema, assets=args.model, keep=False, digest="sha256")
        sync = _make_sync(store, pair, args)
        sent, syncs = {}, []
        for version in VERSIONS:
            if version == 8:
                prompt = Prompt("train", "The capital of France is")
                ids = learner_.tokenizer(prompt.text, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = learner_.policy(**ids).logits[0, -1].float()
                logps = logits.log_softmax(-1)
                chosen = logits.topk(2).indices.tolist()
                samples = tuple(Sample("train", "", (t,), (float(logps[t]),), 7) for t in chosen)
                group = RolloutGroup(
                    "controlled", "0", prompt, samples, (1.0, 0.0),
                    SamplingConfig(max_tokens=1, max_model_len=128),
                )  # fmt: skip
                step = learner_.step([group])
                assert step["policy_version"] == 8, step
                result["controlled_step"] = step
            t = time.monotonic()
            got_version, generator = ready_bf16_tensors(learner_, schema, device="cuda")
            tensors = list(generator)
            torch.cuda.synchronize()
            export_s = time.monotonic() - t
            assert got_version == version
            sent[version] = {name: value.cpu() for name, value in tensors}
            wait_s = _barrier(pair, device)
            stats = sync.push(iter(tensors), version)
            torch.cuda.synchronize()
            syncs.append({"version": version, "export_s": export_s, "barrier_wait_s": wait_s,
                          "push_s": stats.seconds, "bytes": stats.bytes, "digest": store.digest,
                          "GBps": stats.bytes / stats.seconds / 1e9,
                          "detail": getattr(sync, "last", None)})  # fmt: skip
            del tensors
            print(json.dumps({"pushed": syncs[-1]}), flush=True)
        # Return channel: rollout broadcasts its logits (src=1) and posts token IDs.
        actual = {}
        for version in VERSIONS:
            buffer = torch.empty(
                (len(protocol["prompts"]), VOCAB), dtype=torch.float32, device=device
            )
            dist.broadcast(buffer, src=1, group=pair)
            actual[version] = buffer.cpu()
        rollout_info = json.loads(kv.get("e1/rollout"))
        result["rollout"] = rollout_info
        result["learner_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        del learner_
        torch.cuda.empty_cache()

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        reference, input_ids = {}, []
        for version in VERSIONS:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            missing, unexpected = model.load_state_dict(sent[version], strict=False)
            assert not unexpected and set(missing) <= {"lm_head.weight"}, (missing, unexpected)
            model.tie_weights()
            model = model.to(device).eval()
            vectors, input_ids = [], []
            for text in protocol["prompts"]:
                inputs = tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    vectors.append(model(**inputs).logits[0, -1].float().cpu())
                input_ids.append(inputs["input_ids"][0].cpu().tolist())
            reference[version] = torch.stack(vectors)
            del model
            torch.cuda.empty_cache()
        limits = protocol["acceptance_per_prompt_per_version"]
        rows = []
        for version in VERSIONS:
            for row in compare(actual[version], reference[version], limits):
                rows.append(
                    {"version": version, "prompt": protocol["prompts"][row["prompt_index"]], **row}
                )
        ids_match = rollout_info["prompt_token_ids"] == {str(v): input_ids for v in VERSIONS}
        arrays = {f"vllm_v{v}": actual[v].numpy() for v in VERSIONS}
        arrays |= {f"hf_bf16_sdpa_v{v}": reference[v].numpy() for v in VERSIONS}
        np.savez_compressed(out / "logits.npz", **arrays)
        result |= {
            "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "syncs": syncs,
            "receiver_syncs": rollout_info["syncs"],
            "digests_agree": [s["digest"] for s in syncs]
            == [s["digest"] for s in rollout_info["syncs"]],
            "token_ids_match": ids_match,
            "rows": rows,
            "failed_rows": sum(not r["passed"] for r in rows),
            "failed_coordinates": sum(r["failed_coordinates"] for r in rows),
            "coordinates": sum(r["coordinates"] for r in rows),
            "reference_logits_change_v7_v8": float((reference[8] - reference[7]).abs().max()),
            "vllm_logits_change_v7_v8": float((actual[8] - actual[7]).abs().max()),
            "logits_npz_sha256": hashlib.sha256((out / "logits.npz").read_bytes()).hexdigest(),
            "sync_mode": _sync_mode(args),
            "torch": torch.__version__,
            "nccl": ".".join(map(str, torch.cuda.nccl.version())),
            "device": torch.cuda.get_device_name(0),
        }
        result["passed"] = bool(
            result["failed_rows"] == 0 and ids_match and result["digests_agree"]
            and result["vllm_logits_change_v7_v8"] > 0.01
        )  # fmt: skip
        kv.set("e1/done", b"1")
    finally:
        destroy_pair_group(pair)
        dist.destroy_process_group()
        Path(rendezvous).unlink(missing_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print("ROLE_RESULT " + json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)
    return 0 if result["passed"] else 1


def rollout(args):
    from rl.learner.runtime import configure_runtime

    configure_runtime()
    os.environ.update(
        {"VLLM_USE_V1": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_NO_USAGE_STATS": "1"}
    )
    import torch
    import torch.distributed as dist
    from vllm import LLM, SamplingParams

    from rl.sync.acceptance import load_protocol
    from rl.sync.export import base_schema
    from rl.sync.memory_store import MemoryStore
    from rl.sync.pair_group import destroy_pair_group, init_pair_group

    protocol = load_protocol(PROTOCOL)
    # Same diagnostic engine profile as the Task 9 confirmation (batch 2, eager, raw logits).
    llm = LLM(
        model=args.model, dtype="bfloat16", seed=0, gpu_memory_utilization=0.80, max_model_len=128,
        enforce_eager=True, distributed_executor_backend="uni", enable_prefix_caching=False,
        max_num_seqs=16, generation_config="vllm", logprobs_mode="raw_logits", max_logprobs=-1,
        max_num_batched_tokens=128 * 16,
    )  # fmt: skip
    device = torch.device("cuda", torch.cuda.current_device())
    pair, kv = init_pair_group("nccl", args.peer, args.port, 1, timeout=args.timeout)
    info = {"role": "rollout", "host": os.uname().nodename, "env": _nccl_env(), "syncs": [],
            "prompt_token_ids": {}}  # fmt: skip
    try:
        schema = base_schema(args.model, torch.bfloat16)
        store = MemoryStore(schema, assets=args.model, keep=True, digest="sha256")
        sync = _make_sync(store, pair, args)
        params = SamplingParams(max_tokens=1, temperature=1, logprobs=-1, detokenize=False, seed=0)
        prompts = protocol["prompts"]
        vectors_by_version = {}
        for version in VERSIONS:
            wait_s = _barrier(pair, device)
            stats = sync.push(None, version)
            t = time.monotonic()
            tensors = store.take(version)
            llm.apply_model(lambda model, w=tensors: model.load_weights(w))
            llm.reset_prefix_cache()
            torch.cuda.synchronize()
            load_s = time.monotonic() - t
            del tensors
            info["syncs"].append({"version": version, "barrier_wait_s": wait_s, "push_s": stats.seconds,
                                  "bytes": stats.bytes, "load_s": load_s, "digest": store.digest,
                                  "detail": getattr(sync, "last", None)})  # fmt: skip
            vectors, ids = [], []
            for offset in range(0, len(prompts), 2):
                for output in llm.generate(prompts[offset : offset + 2], params, use_tqdm=False):
                    mapping = output.outputs[0].logprobs[0]
                    if set(mapping) != set(range(VOCAB)):
                        raise ValueError("vLLM output omits vocabulary coordinates")
                    vectors.append([float(mapping[i].logprob) for i in range(VOCAB)])
                    ids.append(list(output.prompt_token_ids))
            vectors_by_version[version] = torch.tensor(vectors, dtype=torch.float32)
            info["prompt_token_ids"][str(version)] = ids
        info["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        info["device"] = torch.cuda.get_device_name(device)
        kv.set("e1/rollout", json.dumps(info).encode())
        for version in VERSIONS:
            dist.broadcast(vectors_by_version[version].to(device), src=1, group=pair)
        torch.cuda.synchronize()
        kv.wait(["e1/done"])
    finally:
        destroy_pair_group(pair)
    print(
        "ROLE_RESULT " + json.dumps({k: v for k, v in info.items() if k != "prompt_token_ids"}),
        flush=True,
    )
    return 0


def _make_sync(store, pair, args):
    from rl.sync.bucketed import BucketedSync
    from rl.sync.nccl_broadcast import WeightSync

    if not args.bucket_mb:
        return WeightSync(store, group=pair, source=0, timeout=args.timeout)
    sync = BucketedSync(store, group=pair, source=0, bucket_bytes=int(args.bucket_mb * (1 << 20)),
                        timeout=args.timeout)  # fmt: skip
    sync.warm()
    return sync


def _sync_mode(args):
    if args.bucket_mb:
        return {"kind": "bucketed", "bucket_mb": args.bucket_mb, "digest": "sha256 manifest"}
    return {"kind": "per-tensor", "digest": "sha256"}


def _nccl_env():
    return {
        k: v for k, v in os.environ.items() if k.startswith("NCCL_") or k == "CUDA_VISIBLE_DEVICES"
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--role", choices=("learner", "rollout"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--peer", default="127.0.0.1", help="learner host (rollout role)")
    p.add_argument("--output", default="e1-out")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--bucket-mb", type=float, default=0, help="BucketedSync bucket size (0: WeightSync)")
    args = p.parse_args(argv)
    return learner(args) if args.role == "learner" else rollout(args)


if __name__ == "__main__":
    sys.exit(main())
