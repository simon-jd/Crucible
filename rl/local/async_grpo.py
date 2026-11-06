"""Two-GPU async GRPO: FSDP2 learner and vLLM rollout as separate processes on separate GPUs.

Same science as rl.local.grpo_run --staleness 1 (Qwen2.5-0.5B-Instruct, 100 steps x 8
GSM8K prompts x 8 samples, the EOS rule, same prompts, sampling seeds, learner and
held-out eval), but rollout and learning overlap instead of taking turns:

  rollout (GPU r): generate B_g with the weights it holds, grade on CPU, post B_g;
                   after posting B_g (g >= 1) receive version g over NCCL and load it.
  learner (GPU l): for step t: wait for B_t, train (version t -> t+1), export BF16
                   on the GPU and broadcast version t+1 with rl.sync.nccl_broadcast.

B_0 and B_1 come from version 0 and B_g from version g-1, so step t trains on a batch
one version old (step 0: zero), exactly the k=1 data of grpo_run. While the learner
trains B_t the rollout generates B_{t+1}; they meet at the broadcast of version t+1.
After step 99 the rollout receives version 100 and runs the held-out eval.

Transport: a separate two-rank process group (rl.sync.pair_group) on a TCPStore at the
learner; the same store carries the batches (JSON) between the roles. WeightSync
pushes into an in-memory store (rl.sync.memory_store) with sum64 content digests;
with --bucket-mb N, BucketedSync (rl.sync.bucketed) sends the packed state in N MiB
broadcasts with a SHA-256 manifest instead. The rollout loads the received GPU tensors
with model.load_weights. Every role
writes <output>/<role>.jsonl and <role>-summary.json; `merge` turns a learner and a
rollout directory into grpo_run-compatible steps.jsonl and summary.json plus
per-GPU idle shares, overlap, staleness histogram and sync costs.

  python -m rl.local.async_grpo --role learner --data D --model M --output O --port P
  python -m rl.local.async_grpo --role rollout --data D --model M --output O --port P --peer HOST
  python -m rl.local.async_grpo merge LEARNER_DIR ROLLOUT_DIR OUT_DIR [BEFORE_SUMMARY_JSON]
"""

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path


def generation_version(g):
    """Policy version the rollout holds when it generates batch g (k=1 pipeline)."""
    return 0 if g < 2 else g - 1


def receives_after(g, steps):
    """Versions the rollout receives after posting batch g (the last one also gets the final)."""
    out = [g] if g >= 1 else []
    if g == steps - 1:
        out.append(steps)
    return out


class Channel:
    """JSON messages over the pair TCPStore; each key is read once and deleted."""

    def __init__(self, kv, timeout=1800):
        self.kv, self.timeout = kv, timedelta(seconds=timeout)

    def post(self, key, value):
        self.kv.set(key, json.dumps(value, separators=(",", ":")).encode())

    def get(self, key):
        self.kv.wait([key], self.timeout)
        value = json.loads(self.kv.get(key))
        self.kv.delete_key(key)
        return value


def encode_group(prompt, samples, rewards, batch_id, group_id, sampling):
    return {
        "batch_id": batch_id, "group_id": group_id,
        "prompt": {"prompt_id": prompt.prompt_id, "text": prompt.text},
        "samples": [{"prompt_id": s.prompt_id, "text": s.text, "token_ids": list(s.token_ids),
                     "logprobs": list(s.logprobs), "policy_version": s.policy_version} for s in samples],
        "rewards": list(rewards), "sampling": dict(vars(sampling)),
    }  # fmt: skip


def decode_groups(rows):
    from rl.learner.train import RolloutGroup
    from rl.rollout.sampler import Prompt, Sample, SamplingConfig

    return [
        RolloutGroup(
            r["batch_id"],
            r["group_id"],
            Prompt(**r["prompt"]),
            tuple(
                Sample(
                    s["prompt_id"],
                    s["text"],
                    tuple(s["token_ids"]),
                    tuple(s["logprobs"]),
                    s["policy_version"],
                )
                for s in r["samples"]
            ),
            tuple(r["rewards"]),
            SamplingConfig(**r["sampling"]),
        )
        for r in rows
    ]


def _barrier(group, device):
    import torch
    import torch.distributed as dist

    t = time.monotonic()
    flag = torch.zeros(1, device=device)
    dist.all_reduce(flag, group=group)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.monotonic() - t


def _sync(device):
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize()


def run_learner(backend, channel, sync, group, steps, log, device):
    """Learner loop. backend: .version, .step(groups) -> metrics, .export() -> [(name, tensor)]."""
    stale = {}
    for t in range(steps):
        t0, w0 = time.monotonic(), time.time()
        batch = channel.get(f"batch/{t}")
        wait_s = time.monotonic() - t0
        groups = decode_groups(batch["groups"])
        version_before = backend.version
        for g in groups:
            for s in g.samples:
                age = version_before - s.policy_version
                stale[age] = stale.get(age, 0) + 1
        t1, l0 = time.monotonic(), time.time()
        metrics = backend.step(groups)
        _sync(device)
        learn_s, l1 = time.monotonic() - t1, time.time()
        t2 = time.monotonic()
        tensors = backend.export()
        _sync(device)
        export_s = time.monotonic() - t2
        barrier_s = _barrier(group, device)
        s0 = time.time()
        stats = sync.push(iter(tensors), backend.version)
        _sync(device)
        del tensors
        s1 = time.time()
        detail = getattr(sync, "last", None)
        entry = {
            "step": t, "generation": batch["generation"], "trained_version": batch["version"],
            "policy_version": backend.version, "wait_batch_s": wait_s, "learn_s": learn_s,
            "export_s": export_s, "barrier_s": barrier_s, "sync_s": stats.seconds, "sync_bytes": stats.bytes,
            "wall_s": time.monotonic() - t0, "learn_interval": [l0, l1], "export_sync_interval": [l1, s1],
            "push_interval": [s0, s1], "step_interval": [w0, s1],
            **{k: metrics[k] for k in ("loss", "dropped_stale", "accepted_samples", "effective_tokens")},
            **({"sync_detail": detail} if detail else {}),
        }  # fmt: skip
        log.write(json.dumps(entry) + "\n")
        log.flush()
        if t % 10 == 0:
            print(
                json.dumps(
                    {k: entry[k] for k in ("step", "wait_batch_s", "learn_s", "sync_s", "wall_s")}
                ),
                flush=True,
            )
    return {"staleness_histogram": {str(k): v for k, v in sorted(stale.items())}}


def run_rollout(backend, channel, sync, store, group, steps, log, device):
    """Rollout loop. backend: .generate(g, version) -> (rows, meta), .load(tensors)."""
    version, loaded = 0, []
    for g in range(steps):
        held = generation_version(g)
        if held != version:
            raise RuntimeError(f"rollout holds version {version}, schedule expects {held}")
        g0 = time.time()
        rows, meta = backend.generate(g, version)
        channel.post(
            f"batch/{g}", {"generation": g, "version": version, "groups": rows, "meta": meta}
        )
        entry = {"generation": g, "version": version, "post_time": time.time(), "cycle_start": g0, **meta,
                 "receives": []}  # fmt: skip
        for target in receives_after(g, steps):
            barrier_s = _barrier(group, device)
            r0 = time.time()
            stats = sync.push(None, target)
            t = time.monotonic()
            backend.load(store.take(target))
            _sync(device)
            load_s = time.monotonic() - t
            version = target
            loaded.append(target)
            detail = getattr(sync, "last", None)
            entry["receives"].append({"version": target, "barrier_s": barrier_s, "recv_s": stats.seconds,
                                      "bytes": stats.bytes, "load_s": load_s, "interval": [r0, time.time()],
                                      **({"sync_detail": detail} if detail else {})})  # fmt: skip
        log.write(json.dumps(entry) + "\n")
        log.flush()
    return {"loaded_versions": loaded, "final_version": version}


# ---------------------------------------------------------------------------- merge


def _union(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _clip_total(intervals, lo, hi):
    return sum(max(0.0, min(b, hi) - max(a, lo)) for a, b in _union(intervals))


def _intersect_total(xs, ys, lo, hi):
    xs, ys, total, i, j = _union(xs), _union(ys), 0.0, 0, 0
    while i < len(xs) and j < len(ys):
        a, b = max(xs[i][0], ys[j][0], lo), min(xs[i][1], ys[j][1], hi)
        total += max(0.0, b - a)
        if xs[i][1] < ys[j][1]:
            i += 1
        else:
            j += 1
    return total


def _range(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def merge(learner_dir, rollout_dir, out_dir, before_from=None):
    """Combine both roles' logs into steps.jsonl / summary.json (grpo_run format plus extras).

    before_from: a summary.json (e.g. the time-shared run of the same seed) whose base-model
    eval is reused when the rollout did not run --eval-before; its path is recorded.
    """
    learner_dir, rollout_dir, out_dir = Path(learner_dir), Path(rollout_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lsteps = [json.loads(x) for x in (learner_dir / "learner.jsonl").read_text().splitlines()]
    rgens = [json.loads(x) for x in (rollout_dir / "rollout.jsonl").read_text().splitlines()]
    lsum = json.loads((learner_dir / "learner-summary.json").read_text())
    rsum = json.loads((rollout_dir / "rollout-summary.json").read_text())
    by_gen = {r["generation"]: r for r in rgens}
    steps = []
    for e in lsteps:
        r = by_gen[e["generation"]]
        busy = e["learn_s"] + e["export_s"] + e["sync_s"]
        steps.append({
            "step": e["step"], "trained_version": e["trained_version"], "policy_version": e["policy_version"],
            "reward": r["reward"], "shaped_reward": r["shaped_reward"], "mean_completion_tokens": r["len"],
            "flags": r["flags"], "rollout_s": r["rollout_s"], "learn_s": e["learn_s"], "sync_s": e["sync_s"],
            "reward_s": r["reward_s"], "wall_s": e["wall_s"], "gpu_idle_share": max(0.0, e["wall_s"] - busy) / e["wall_s"],
            **{k: e[k] for k in ("loss", "dropped_stale", "accepted_samples", "effective_tokens", "wait_batch_s",
                                 "export_s", "barrier_s", "sync_bytes")},
        })  # fmt: skip
    with open(out_dir / "steps.jsonl", "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in steps)
    lo = min(r["cycle_start"] for r in rgens)
    hi = lsteps[-1]["push_interval"][1]
    window = hi - lo
    learner_busy = [e["learn_interval"] for e in lsteps] + [
        e["export_sync_interval"] for e in lsteps
    ]
    rollout_gen = [r["gen_interval"] for r in rgens]
    rollout_busy = rollout_gen + [x["interval"] for r in rgens for x in r["receives"]]
    receives = [x for r in rgens for x in r["receives"]]
    trained = sum(e["accepted_samples"] for e in lsteps)
    pipeline = {
        "training_window_s": window,
        "learner_gpu_busy_s": _clip_total(learner_busy, lo, hi),
        "rollout_gpu_busy_s": _clip_total(rollout_busy, lo, hi),
        "learner_gpu_idle_share": 1 - _clip_total(learner_busy, lo, hi) / window,
        "rollout_gpu_idle_share": 1 - _clip_total(rollout_busy, lo, hi) / window,
        "learn_generate_overlap_s": _intersect_total(
            [e["learn_interval"] for e in lsteps], rollout_gen, lo, hi
        ),
        "trained_samples_per_s": trained / window,
        "generated_samples_per_s": sum(r["samples"] for r in rgens) / window,
        "step_wall_s": _range([e["wall_s"] for e in lsteps]),
        "wait_batch_s": _range([e["wait_batch_s"] for e in lsteps]),
        "learn_s": _range([e["learn_s"] for e in lsteps]),
        "export_s": _range([e["export_s"] for e in lsteps]),
        "barrier_s": _range([e["barrier_s"] for e in lsteps]),
        "learner_push_s": _range([e["sync_s"] for e in lsteps]),
        "rollout_generate_s": _range([r["rollout_s"] for r in rgens]),
        "rollout_grade_s": _range([r["reward_s"] for r in rgens]),
        "rollout_recv_s": _range([x["recv_s"] for x in receives]),
        "rollout_load_s": _range([x["load_s"] for x in receives]),
        "rollout_barrier_s": _range([x["barrier_s"] for x in receives]),
        "sync_bytes_per_version": sorted({x["bytes"] for x in receives}),
        "learner_push_GBps": _range([e["sync_bytes"] / e["sync_s"] / 1e9 for e in lsteps]),
        "staleness_histogram": lsum["staleness_histogram"],
        "dropped_stale": sum(e["dropped_stale"] for e in lsteps),
        "versions_loaded": len(receives),
    }
    flags = {}
    for s in steps:
        for k, v in s["flags"].items():
            flags[k] = flags.get(k, 0) + v
    summary = {
        "args": rsum["args"] | {"layout": rsum["args"].get("layout")},
        "data_sha256": rsum["data_sha256"],
        **(
            {
                "before": rsum["before"],
                "eval_before_s": rsum["eval_before_s"],
                "before_source": "this run",
            }
            if "before" in rsum
            else {}
        ),
        "after": rsum["after"],
        "eval_after_s": rsum["eval_after_s"],
        "totals": {
            "generated": sum(r["samples"] for r in rgens),
            "trained": trained,
            "dropped_stale": pipeline["dropped_stale"],
            "flags": flags,
        },
        "pipeline": pipeline,
        "learner": {k: v for k, v in lsum.items() if k != "staleness_histogram"},
        "rollout": {k: v for k, v in rsum.items() if k not in ("before", "after", "args")},
        "peak_cuda_allocated_bytes": max(
            lsum["peak_allocated_bytes"], rsum["peak_allocated_bytes"]
        ),
        "wall_s": window,
    }
    if "before" not in summary and before_from is not None:
        other = json.loads(Path(before_from).read_text())
        if (
            other["data_sha256"] != summary["data_sha256"]
            or other["args"]["seed"] != summary["args"]["seed"]
        ):
            raise ValueError("base-model eval must come from the same data and seed")
        summary["before"], summary["before_source"] = other["before"], str(before_from)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


# ---------------------------------------------------------------------------- GPU roles


class VllmRollout:
    def __init__(self, args, llm, tokenizer, pool):
        from rl.rollout.sampler import SamplingConfig

        self.args, self.llm, self.tokenizer, self.pool = args, llm, tokenizer, pool
        self.data = json.loads(Path(args.data).read_text())
        self.train = self.data["train"]
        self.answers = {r["id"]: r["answer"] for r in self.train}
        self.sampling = SamplingConfig(seed=args.seed, temperature=1.0, top_p=1.0, top_k=-1,
                                       max_tokens=args.max_new_tokens, max_model_len=1024)  # fmt: skip

    def generate(self, g, version):
        from vllm import SamplingParams

        from rl.local import grpo_run as gr
        from rl.rollout.sampler import Prompt, samples_from_outputs

        a = self.args
        rows = [
            self.train[(g * a.prompts_per_step + i) % len(self.train)]
            for i in range(a.prompts_per_step)
        ]
        prompts = [Prompt(r["id"], self.tokenizer.apply_chat_template(gr.prompt_messages(r["question"]), tokenize=False,
                                                                       add_generation_prompt=True)) for r in rows]  # fmt: skip
        params = [
            SamplingParams(n=a.group, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=a.max_new_tokens, logprobs=0,
                           seed=(a.seed * 1_000_003 + g * 1009 + i) % 2**31)
            for i in range(len(prompts))
        ]  # fmt: skip
        t, t_wall = time.monotonic(), time.time()
        outputs = self.llm.generate([p.text for p in prompts], params, use_tqdm=False)
        rollout_s, gen_interval = time.monotonic() - t, [t_wall, time.time()]
        samples = samples_from_outputs(prompts, outputs, a.group, version)
        finish = [
            c.finish_reason for o in outputs for c in sorted(o.outputs, key=lambda c: c.index)
        ]
        t = time.monotonic()
        correct, _ = gr.grade(
            self.pool, [s.text for s in samples], [self.answers[s.prompt_id] for s in samples]
        )
        reward_s = time.monotonic() - t
        flags = {}
        for s, f in zip(samples, finish, strict=True):
            for flag in gr.math_flags(s.text, f == "length"):
                flags[flag] = flags.get(flag, 0) + 1
        groups = []
        for gi, pr in enumerate(prompts):
            sl = slice(gi * a.group, (gi + 1) * a.group)
            rewards = tuple(
                gr.shaped_reward(c, f, a.reward)
                for c, f in zip(correct[sl], finish[sl], strict=True)
            )
            groups.append(
                encode_group(pr, samples[sl], rewards, f"g{g}", f"g{g}/{gi}", self.sampling)
            )
        meta = {
            "reward": sum(correct) / len(correct), "shaped_reward": sum(sum(x["rewards"]) for x in groups) / len(samples),
            "flags": flags, "len": sum(len(s.token_ids) for s in samples) / len(samples), "samples": len(samples),
            "rollout_s": rollout_s, "reward_s": reward_s, "gen_interval": gen_interval,
        }  # fmt: skip
        return groups, meta

    def load(self, tensors):
        self.llm.apply_model(lambda model, w=tensors: model.load_weights(w))
        self.llm.reset_prefix_cache()


class FsdpLearner:
    def __init__(self, learner, schema):
        self.learner, self.schema = learner, schema

    @property
    def version(self):
        return self.learner.policy_version

    def step(self, groups):
        return self.learner.step(groups)

    def export(self):
        from rl.sync.export import ready_bf16_tensors

        _, tensors = ready_bf16_tensors(self.learner, self.schema, device="cuda")
        return list(tensors)


class MemoryPeak:
    """Sample device memory in use (all processes on this GPU) every 0.5 s."""

    def __init__(self):
        import threading

        self.peak, self._stop = 0, threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import torch

        while not self._stop.wait(0.5):
            free, total = torch.cuda.mem_get_info()
            self.peak = max(self.peak, total - free)

    def stop(self):
        self._stop.set()
        self._thread.join(2)
        return self.peak


def _common(args):
    from rl.learner.runtime import configure_runtime

    configure_runtime()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    return out


def learner_main(args):
    out = _common(args)
    import torch
    import torch.distributed as dist

    from rl.learner.train import LearnerConfig, load_fsdp2_learner
    from rl.sync.bucketed import make_weight_sync
    from rl.sync.export import base_schema
    from rl.sync.pair_group import destroy_pair_group, init_pair_group

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    rendezvous = tempfile.mktemp(prefix="async-grpo-default-")
    dist.init_process_group("nccl", init_method="file://" + rendezvous, rank=0, world_size=1)
    peak = MemoryPeak()
    config = LearnerConfig(
        group_size=args.group, microbatch_size=args.microbatch, max_length=1024,
        max_staleness=1, clip_eps=args.clip, kl_coef=args.kl, scale_by_std=True,
    )  # fmt: skip
    learner = load_fsdp2_learner(args.model, config, learning_rate=args.lr)
    pair, kv = init_pair_group("nccl", "0.0.0.0", args.port, 0, timeout=args.timeout)
    try:
        schema = base_schema(args.model, torch.bfloat16)
        _, sync = make_weight_sync(schema, assets=args.model, keep=False, group=pair, timeout=args.timeout,
                                   bucket_mb=args.bucket_mb)  # fmt: skip
        channel = Channel(kv, timeout=args.timeout)
        started = time.monotonic()
        with open(out / "learner.jsonl", "w") as log:
            result = run_learner(
                FsdpLearner(learner, schema), channel, sync, pair, args.steps, log, device
            )
        result |= {
            "train_wall_s": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_device_used_bytes": peak.stop(),
            "device": torch.cuda.get_device_name(0), "host": os.uname().nodename, "nccl_env": _nccl_env(),
            "sync_mode": _sync_mode(args),
        }  # fmt: skip
        (out / "learner-summary.json").write_text(json.dumps(result, indent=1) + "\n")
        print("ROLE_RESULT " + json.dumps(result), flush=True)
        Channel(kv, timeout=args.eval_timeout).get(
            "rollout/done"
        )  # keep the store up for the rollout
    finally:
        destroy_pair_group(pair)
        dist.destroy_process_group()
        Path(rendezvous).unlink(missing_ok=True)
    return 0


def rollout_main(args):
    out = _common(args)
    os.environ.update({"VLLM_USE_V1": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_NO_USAGE_STATS": "1",
                       "DO_NOT_TRACK": "1"})  # fmt: skip
    from concurrent.futures import ThreadPoolExecutor

    import torch
    from vllm import LLM

    from rl.local import grpo_run as gr
    from rl.sync.bucketed import make_weight_sync
    from rl.sync.export import base_schema
    from rl.sync.pair_group import destroy_pair_group, init_pair_group

    # Same engine arguments as grpo_run, so the held-out eval is the same computation.
    llm = LLM(
        model=args.model, dtype="bfloat16", seed=args.seed, gpu_memory_utilization=args.vllm_mem,
        max_model_len=1024, enforce_eager=True, distributed_executor_backend="uni",
        enable_prefix_caching=True, max_num_seqs=128, generation_config="vllm",
        logprobs_mode="raw_logprobs", max_logprobs=1,
    )  # fmt: skip
    device = torch.device("cuda", torch.cuda.current_device())
    peak = MemoryPeak()
    tokenizer = llm.get_tokenizer()
    pool = ThreadPoolExecutor(16)
    backend = VllmRollout(args, llm, tokenizer, pool)
    summary = {"args": {k: str(v) for k, v in vars(args).items()} | {"staleness": "1"},
               "data_sha256": backend.data["sha256"]}  # fmt: skip
    sets = {"gsm8k": backend.data["gsm8k_eval"], "math500": backend.data["math500_eval"]}
    if args.eval_before:
        t = time.monotonic()
        summary["before"] = gr.run_eval(llm, tokenizer, sets, args.eval_n, args.eval_max_tokens, pool,
                                        out / "eval-before.jsonl.gz")  # fmt: skip
        summary["eval_before_s"] = time.monotonic() - t
        print(
            json.dumps({"before": {k: v["pass_at_1"] for k, v in summary["before"].items()}}),
            flush=True,
        )
    pair, kv = init_pair_group("nccl", args.peer, args.port, 1, timeout=args.timeout)
    try:
        schema = base_schema(args.model, torch.bfloat16)
        store, sync = make_weight_sync(schema, assets=args.model, keep=True, group=pair,
                                       timeout=args.timeout, bucket_mb=args.bucket_mb)  # fmt: skip
        channel = Channel(kv, timeout=args.timeout)
        started = time.monotonic()
        with open(out / "rollout.jsonl", "w") as log:
            summary |= run_rollout(backend, channel, sync, store, pair, args.steps, log, device)
        summary["train_wall_s"] = time.monotonic() - started
        t = time.monotonic()
        summary["after"] = gr.run_eval(llm, tokenizer, sets, args.eval_n, args.eval_max_tokens, pool,
                                       out / "eval-after.jsonl.gz")  # fmt: skip
        summary["eval_after_s"] = time.monotonic() - t
        summary |= {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_device_used_bytes": peak.stop(),
            "device": torch.cuda.get_device_name(device), "host": os.uname().nodename, "nccl_env": _nccl_env(),
            "sync_mode": _sync_mode(args),
        }  # fmt: skip
        (out / "rollout-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(
            json.dumps({"after": {k: v["pass_at_1"] for k, v in summary["after"].items()}}),
            flush=True,
        )
        channel.post("rollout/done", True)
    finally:
        destroy_pair_group(pair)
    return 0


def _sync_mode(args):
    if args.bucket_mb:
        return {"kind": "bucketed", "bucket_mb": args.bucket_mb, "digest": "sha256"}
    return {"kind": "per-tensor", "digest": "sum64"}


def _nccl_env():
    return {
        k: v for k, v in os.environ.items() if k.startswith("NCCL_") or k == "CUDA_VISIBLE_DEVICES"
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "merge":
        summary = merge(*argv[1:5])
        print(json.dumps(summary["pipeline"], indent=1))
        return 0
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--role", choices=("learner", "rollout"), required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--peer", default="127.0.0.1", help="learner host, for the rollout role")
    p.add_argument("--layout", default="", help="label recorded in the summary")
    p.add_argument("--kl", type=float, default=0.001)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--eval-n", type=int, default=4)
    p.add_argument("--eval-max-tokens", type=int, default=512)
    p.add_argument(
        "--eval-before", action="store_true", help="also run the base-model eval (rollout role)"
    )
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--vllm-mem", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--reward", choices=("none", "eos"), default="eos")
    p.add_argument("--timeout", type=float, default=1800, help="per collective/message wait (s)")
    p.add_argument(
        "--bucket-mb", type=float, default=0,
        help="BucketedSync with buckets of this many MiB (0: per-tensor WeightSync)",
    )  # fmt: skip
    p.add_argument(
        "--eval-timeout", type=float, default=7200, help="learner waits this long for the eval"
    )
    args = p.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise SystemExit("timeout must be positive")
    if not math.isfinite(args.bucket_mb) or args.bucket_mb < 0:
        raise SystemExit("bucket-mb must be >= 0")
    return learner_main(args) if args.role == "learner" else rollout_main(args)


if __name__ == "__main__":
    sys.exit(main())
