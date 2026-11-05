"""E5 payload: first multi-node runs of Crucible's cluster code on Snowflake GPU_NV_SM nodes (1x A10 24 GB each),
NCCL over TCP (NET/Socket). Reuses the pure logic of cluster/drain, cluster/health, cluster/provision and scaling/;
none of their apply/launch paths (Slurm, K8s, TorchTitan, mpirun, ssh) is used.

  driver:  --phases health collectives scaling drain fault   (default: all; --quick = smoke sizes)
  rank:    --mode rank --cfg <json>   one process of a distributed world, started by a node actor

Phases:
- health       per node: cluster/provision/node_probe.py (host + workload probes), tool presence (dcgmi, nccl-tests
               all_reduce_perf, ibstat, nsys), /sys/class/infiniband, nvidia-smi facts, disk write rate, a bf16
               matmul (achieved TFLOPS), a CUPTI probe (torch.profiler kernel records), and on the head node a pip
               probe for TorchTitan / torch nightly.
- collectives  all-reduce and broadcast (torch.distributed, E0's run_collectives) on 2, 3 and 4 nodes, E0's tuned
               NCCL settings (3 interleaved passes) plus the default once and an NCCL_IB_DISABLE=0 probe.
- scaling      DDP training of Qwen2.5-0.5B-Instruct (bf16 autocast, fp32 master weights and fp32 gradient
               all-reduce, AdamW) on 1, 2, 4 nodes: weak (16 sequences per GPU per step) and strong (16 sequences
               per step), 3 interleaved rounds; synchronized step timing split into communication-free micro-steps,
               the last micro-step (backward + gradient all-reduce) and the optimizer; plus the gradient volume
               all-reduced alone in DDP-sized buckets. CUPTI is unavailable on these vGPUs, so kernel traces (and
               scaling.analyze_nsys.summarize_kernels) cannot be used for the communication share.
- drain        4-node DDP run; every rank publishes window step times through cluster.drain.metrics.StepTimeHook
               (synchronized step time and local no-communication compute time, two run_ids); the driver parses
               the textfiles into NodeMetrics and runs cluster.drain.controller.Controller + SchedulerExecutor
               (dry-run). One node gets a competing CUDA matmul process for a fixed span of samples.
- fault        4-node DDP run with full-state DCP checkpoints (rank 0, head node's disk; CheckpointRetention keeps
               two), scaling.goodput.FaultTrainingHook via scaling.fault_train.make_torch_hook on every rank.
               The victim is the last rank, on another node than the checkpoint.
               The driver SIGKILLs the last rank's process, tears the world down, records the failure through the
               same hook API, restarts every rank from the last checkpoint and replays the log with account_events.

One Ray actor per node (STRICT_SPREAD); rank 0 runs on the head node next to the driver, so driver and rank 0 share
one CLOCK_MONOTONIC (the goodput coordinator clock). Every NCCL world runs in fresh rank processes (NCCL reads its
environment once). Rendezvous ports cycle through E5's band 12950-12999. Prints one "RESULT <json>" per phase.
"""
import argparse
import json
import os
import pickle
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))  # repository root inside the job
PAYLOAD = Path(__file__).resolve()

PORTS = tuple(range(12950, 13000))
MODEL_ASSET = "models/Qwen2.5-0.5B-Instruct"
DATA_ASSET = "data/gsm8k"
SEQ = 1024
MICRO = 2
WEAK_PER_GPU_SEQ = 16
STRONG_GLOBAL_SEQ = 16
A10_BF16_DENSE_PEAK = 125e12  # NVIDIA A10 datasheet: BF16 tensor core 125 TFLOPS dense (250 with sparsity)
BASE = Path("/tmp/e5")

SOCKET = {"NCCL_SOCKET_NTHREADS": "8", "NCCL_NSOCKS_PERTHREAD": "8", "NCCL_MIN_NCHANNELS": "4"}
VARIANTS = {
    "default": {},
    "tuned": dict(SOCKET),  # E0: best broadcast
    "tuned_buff16m": {**SOCKET, "NCCL_BUFFSIZE": "16777216"},  # E0: best all-reduce
    "ib_enabled": {"NCCL_IB_DISABLE": "0"},  # probe only: does NCCL find a usable IB/RoCE device (mlx5_0 in sysfs)?
}
TRAIN_NCCL = VARIANTS["tuned_buff16m"]  # DDP is all-reduce only


def progress(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# ---------------------------------------------------------------- pure helpers (CPU-tested)

def bus_factor(op, n):
    """nccl-tests convention: all-reduce busbw = algbw * 2(n-1)/n, broadcast busbw = algbw."""
    if n < 1:
        raise ValueError("world size must be positive")
    return 2 * (n - 1) / n if op == "all_reduce" else 1.0


def with_busbw(rows, n):
    return [{**r, "busGBps": round(r["GBps"] * bus_factor(r["op"], n), 3)} for r in rows]


def batch_plan(kind, world, micro=MICRO, per_gpu_seq=WEAK_PER_GPU_SEQ, global_seq=STRONG_GLOBAL_SEQ, seq=SEQ):
    """weak: fixed sequences per GPU per step; strong: fixed global sequences per step, split across ranks."""
    if kind == "weak":
        per_rank = per_gpu_seq
    elif kind == "strong":
        if global_seq % world:
            raise ValueError("global batch must split evenly across ranks")
        per_rank = global_seq // world
    else:
        raise ValueError(f"unknown scaling kind {kind}")
    if per_rank % micro or per_rank <= 0:
        raise ValueError("per-rank batch must be a positive multiple of the micro-batch")
    return {"kind": kind, "world": world, "micro": micro, "accum": per_rank // micro, "per_rank_seq": per_rank,
            "global_seq": per_rank * world, "seq": seq, "tokens_per_step": per_rank * world * seq}


def sequence_ids(step, rank, plan, n_seq):
    """Deterministic data order: global step s uses sequences [(s-1)G, sG) mod n_seq, rank r takes its slice.
    Strong-scaling runs therefore see the same global batch for every world size."""
    if step < 1:
        raise ValueError("steps count from 1")
    per = plan["per_rank_seq"]
    start = (step - 1) * plan["global_seq"] + rank * per
    return [(start + i) % n_seq for i in range(per)]


def pack_tokens(docs, seq, eos):
    flat = []
    for d in docs:
        flat.extend(d)
        flat.append(eos)
    n = len(flat) // seq
    return [flat[i * seq:(i + 1) * seq] for i in range(n)]


def parse_prom(text):
    """Parse a StepTimeHook textfile into {metric: (labels, value)}."""
    out = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        head, value = line.rsplit(" ", 1)
        name, _, labels = head.partition("{")
        pairs = re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', labels)
        out[name] = ({k: v.replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\") for k, v in pairs},
                     float(value))
    return out


def node_metrics(sample, *, now, offset, link_up, xid_errors=0):
    """StepTimeHook sample -> cluster.drain NodeMetrics. offset = driver clock - node clock (seconds).
    XID comes from DCGM in the original design; without DCGM the caller passes 0 and says so."""
    from cluster.drain.controller import NodeMetrics
    labels, step_time = sample["crucible_step_time_seconds"]
    stamp = sample["crucible_step_sample_timestamp_seconds"][1] + offset
    return NodeMetrics(observed_at=now, xid_errors=xid_errors, link_up=link_up, step_time=step_time,
                       step_observed_at=stamp, run_id=labels["run_id"])


def step_alerts(actions):
    return sorted(a.node for a in actions if a.kind == "alert" and a.reason.startswith("step time >"))


def verify_drain(log, slow_node, ratio=1.2, consecutive=3, min_nodes=3):
    """Recompute the controller's rule independently from the logged samples.

    log: [{"sample": i, "times": {node: seconds}, "alerts": [node, ...]}] in evaluation order. A node is slow in a
    sample when its time > median(all nodes) * ratio; an alert is expected when its run of slow samples reaches
    `consecutive` (once per run)."""
    runs, expected, actual = {}, [], []
    slow_marks = {}
    for entry in log:
        times = entry["times"]
        for node in entry.get("alerts", []):
            actual.append((entry["sample"], node))
        if len(times) < min_nodes:
            continue
        med = statistics.median(times.values())
        for node, t in times.items():
            slow = t > med * ratio
            slow_marks.setdefault(node, []).append(slow)
            runs[node] = runs.get(node, 0) + 1 if slow else 0
            if slow and runs[node] == consecutive:
                expected.append((entry["sample"], node))
    first_slow = next((i for i, s in enumerate(slow_marks.get(slow_node, [])) if s), None)
    first_alert = next((s for s, n in actual if n == slow_node), None)
    return {"expected_alerts": expected, "actual_alerts": actual, "match": expected == actual,
            "healthy_flagged": sorted({n for _, n in actual if n != slow_node}),
            "slow_node_alerted": first_alert is not None,
            "slow_node_first_slow_sample": None if first_slow is None else log[first_slow]["sample"],
            "slow_node_first_alert_sample": first_alert,
            "slow_flags": {n: [int(x) for x in v] for n, v in slow_marks.items()}}


def kernel_rows(trace):
    """torch.profiler chrome trace -> (device, start ns, end ns, name) rows for analyze_nsys.summarize_kernels."""
    rows, skipped = [], 0
    for e in trace.get("traceEvents", []):
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        start = round(float(e["ts"]) * 1000)
        end = round((float(e["ts"]) + float(e.get("dur", 0))) * 1000)
        if end <= start or not e.get("name"):
            skipped += 1
            continue
        rows.append((str(e.get("args", {}).get("device", 0)), start, end, e["name"]))
    return rows, skipped


def step_split(steps, accum, comm_alone=None):
    """Wall-time communication share of synchronized DDP steps (kernel traces are unavailable without CUPTI).

    steps: [{"t", "local", "last", "opt"}] from train_step. `local` is one communication-free micro-step (no_sync),
    `last` is the final micro-step whose backward launches the gradient all-reduce, measured up to the point where
    the all-reduce has finished. Exposed communication = last - local (the part of the all-reduce not hidden behind
    that backward); compute = accum * local + opt. comm_alone: seconds to all-reduce the same gradient volume in
    DDP-sized buckets with nothing else running; hidden = comm_alone - exposed."""
    med = {k: statistics.median(s[k] for s in steps) for k in ("t", "last", "opt")}
    local = statistics.median(s["local"] for s in steps if s["local"] is not None) if accum > 1 else None
    out = {"step_s": med["t"], "local_micro_s": local, "last_micro_s": med["last"], "opt_s": med["opt"]}
    if local is not None:
        exposed = max(0.0, med["last"] - local)
        out.update(compute_s=accum * local + med["opt"], exposed_comm_s=exposed,
                   exposed_comm_share=exposed / med["t"])
        if comm_alone is not None:
            out.update(comm_alone_s=comm_alone, comm_alone_share=comm_alone / med["t"],
                       hidden_comm_s=max(0.0, comm_alone - exposed))
    return out


def qwen_structure(config):
    """analyze_nsys.ModelStructure inputs from a HF config dict: N = parameters minus the (tied) embedding."""
    h, inter, layers = config["hidden_size"], config["intermediate_size"], config["num_hidden_layers"]
    heads, kv = config["num_attention_heads"], config["num_key_value_heads"]
    d = h // heads
    attn = h * heads * d + heads * d + 2 * (h * kv * d + kv * d) + heads * d * h  # q,k,v with bias; o no bias
    mlp = 3 * h * inter
    norms = 2 * h
    n = layers * (attn + mlp + norms) + h  # + final norm
    return {"active_parameters": n, "layers": layers, "heads": heads, "head_dim": d,
            "embedding_parameters": config["vocab_size"] * h}


def jsonl_appender(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def emit(record):
        with open(path, "a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    return emit


def read_jsonl(path):
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break  # a torn last line from a killed writer ends the log
    return out


def repair_jsonl(path):
    """Cut the log after its last complete record, so an append after a killed writer starts on a clean line.
    Returns the number of bytes dropped."""
    path = Path(path)
    try:
        raw = path.read_text()
    except OSError:
        return 0
    keep = []
    for line in raw.split("\n"):
        try:
            json.loads(line)
        except json.JSONDecodeError:
            break
        keep.append(line)
    body = "".join(line + "\n" for line in keep)
    if body != raw:
        path.write_text(body)
    return len(raw) - len(body)


def redact(text, secrets=()):
    """Remove the PyPI proxy token and the account identifier (also when the job does not know it by name)."""
    text = re.sub(r"//[^/@\s'\"]+@", "//<credentials>@", text)  # index URLs carry a token as user:password
    text = re.sub(r"eyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}", "<token>", text)  # any JWT
    text = re.sub(r"[\w-]+(?=\.registry-local\.snowflakecomputing\.com)", "<account>", text)
    text = re.sub(r"(?<=//)[\w-]+(?=\.(?:[\w-]+\.)*snowflakecomputing\.com)", "<account>", text)
    for s in secrets:
        if s:
            text = re.sub(re.escape(s), "<account>", text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------- rank processes

def init_pg(cfg):
    import datetime

    import torch
    import torch.distributed as dist
    cuda = cfg.get("device", "cuda") == "cuda"
    kw = {}
    if cuda:
        torch.cuda.set_device(0)
        kw["device_id"] = torch.device("cuda", 0)
    dist.init_process_group(cfg.get("backend", "nccl"), init_method=f"tcp://{cfg['master']}:{cfg['port']}",
                            rank=cfg["rank"], world_size=cfg["world"],
                            timeout=datetime.timedelta(seconds=cfg.get("pg_timeout", 900)), **kw)
    return torch.device("cuda", 0) if cuda else torch.device("cpu")


def nccl_rank(cfg):
    import torch.distributed as dist

    from scripts.snowflake.payloads.e0_network import ITERS_NET, nccl_log_lines, run_collectives
    t0 = time.perf_counter()
    init_pg(cfg)
    iters = {int(k): tuple(v) for k, v in cfg.get("iters", {}).items()} or ITERS_NET
    bw, ok = run_collectives(cfg["rank"], cfg["sizes"], iters)
    dist.destroy_process_group()
    return {"rank": cfg["rank"], "bw": with_busbw(bw, cfg["world"]), "check_ok": ok,
            "seconds": round(time.perf_counter() - t0, 1), "nccl_log": nccl_log_lines(os.environ.get("NCCL_DEBUG_FILE", ""))}


class TinyLM:
    """CPU-test stand-in with the same call contract as the HF wrapper: module(ids) -> mean LM loss."""

    @staticmethod
    def build(torch, vocab=64, dim=16):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(vocab, dim)
                self.out = torch.nn.Linear(dim, vocab)

            def forward(self, ids):
                logits = self.out(self.emb(ids[:, :-1]))
                return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                                         ids[:, 1:].reshape(-1))
        torch.manual_seed(0)
        return M()


def build_model(cfg, torch, device):
    if cfg.get("model") == "tiny":
        return TinyLM.build(torch).to(device), None
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32, attn_implementation="sdpa")
    hf.config.use_cache = False

    class LossWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, ids):
            return self.model(input_ids=ids, labels=ids, use_cache=False).loss
    return LossWrapper(hf).to(device), hf.config.to_dict()


def cpu_torch_shim(torch):
    """make_torch_hook calls torch.cuda.synchronize(device); on the CPU (gloo) test path that is a no-op."""
    from types import SimpleNamespace
    return SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device=None: None), distributed=torch.distributed)


def train_rank(cfg):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    rank, world, kind = cfg["rank"], cfg["world"], cfg["kind"]
    cuda = cfg.get("device", "cuda") == "cuda"
    out = {"rank": rank, "world": world, "kind": kind, "host": socket.gethostname(), "attempt": cfg.get("attempt", 0)}
    fault = kind == "fault"
    emit, prior = (lambda e: None), None
    if fault and rank == 0:
        emit = jsonl_appender(cfg["events"])
        if cfg.get("resume"):
            prior = read_jsonl(cfg["events"])
        else:
            # Same origin record as scaling.fault_train.create_fault_trainer, before any distributed/device init.
            prior = [{"schema": 1, "rank": 0, "kind": "run_start", "run_id": cfg["run_id"], "clock_id": cfg["clock_id"],
                      "world_size": world, "at": time.monotonic()}]
            emit(prior[0])
    t_init = time.perf_counter()
    device = init_pg(cfg)
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model, hf_config = build_model(cfg, torch, device)
    ddp = DDP(model, device_ids=[0] if cuda else None, gradient_as_bucket_view=True,
              bucket_cap_mb=cfg.get("bucket_cap_mb", 25))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-5), weight_decay=0.0, fused=cuda)
    tokens = torch.load(cfg["tokens"])
    plan = cfg["plan"]
    accum, micro = plan["accum"], plan["micro"]
    out["init_seconds"] = round(time.perf_counter() - t_init, 2)
    if hf_config is not None:
        out["structure"] = qwen_structure(hf_config)
    out["parameters"] = sum(p.numel() for p in model.parameters())

    def sync():
        if cuda:
            torch.cuda.synchronize(device)

    def micro_step(ids):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            loss = ddp(ids)
        (loss / accum).backward()
        return loss.detach()

    def train_step(step):
        """One synchronized optimizer step. t = whole step; local = mean communication-free micro-step (no_sync);
        last = final micro-step until its gradient all-reduce has finished; opt = AdamW step."""
        ids = sequence_ids(step, rank, plan, tokens.shape[0])
        chunks = [ids[i * micro:(i + 1) * micro] for i in range(accum)]
        sync()
        t0 = time.perf_counter()
        losses = []
        t1 = t0
        for i, c in enumerate(chunks):
            batch = tokens[c].to(device, dtype=torch.long, non_blocking=True)
            if i < accum - 1:
                with ddp.no_sync():
                    losses.append(micro_step(batch))
            else:
                sync()  # everything so far ran without communication
                t1 = time.perf_counter()
                losses.append(micro_step(batch))
        sync()
        t2 = time.perf_counter()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync()
        t3 = time.perf_counter()
        return {"t": t3 - t0, "local": (t1 - t0) / (accum - 1) if accum > 1 else None, "last": t2 - t1,
                "opt": t3 - t2, "loss": float(torch.stack(losses).mean())}

    def comm_alone(reps=3, bucket_bytes=None):
        """All-reduce the gradient volume (fp32, one value per parameter) in DDP-sized buckets, nothing else running.
        First call is a warm-up. Returns the slowest rank's seconds per pass."""
        bucket = (bucket_bytes or cfg.get("bucket_cap_mb", 25) * 2**20) // 4
        n = sum(p.numel() for p in model.parameters())
        buf = torch.zeros(n, dtype=torch.float32, device=device)
        out = []
        for i in range(reps + 1):
            sync()
            dist.barrier()
            sync()
            t = time.perf_counter()
            for off in range(0, n, bucket):
                dist.all_reduce(buf[off:off + bucket])
            sync()
            if i:
                out.append(global_max(time.perf_counter() - t))
        del buf
        return {"bytes": 4 * n, "bucket_bytes": 4 * bucket, "seconds": [round(x, 4) for x in out]}

    def global_max(value):
        t = torch.tensor(float(value), device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()

    try:
        if kind == "scale":
            out.update(scale_loop(cfg, train_step, global_max, plan, comm_alone))
        elif kind == "drain":
            out.update(drain_loop(cfg, train_step))
        elif kind == "fault":
            out.update(fault_loop(cfg, torch, dist, device, cuda, model, opt, train_step, emit, prior))
        else:
            raise ValueError(f"unknown kind {kind}")
        if cuda:
            out["peak_mem_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return out


def scale_loop(cfg, train_step, global_max, plan, comm_alone):
    warm, timed = cfg["warmup_steps"], cfg["timed_steps"]
    losses, steps = [], []
    step = 0
    for _ in range(warm):
        step += 1
        losses.append(train_step(step)["loss"])
    for _ in range(timed):
        step += 1
        s = train_step(step)
        steps.append(s)
        losses.append(s["loss"])
    total = sum(s["t"] for s in steps)
    max_total = global_max(total)  # like scaling.train.CompletedStepTimer: the slowest rank's window counts
    res = {"plan": plan, "step_seconds": [round(s["t"], 4) for s in steps],
           "local_micro_seconds": [None if s["local"] is None else round(s["local"], 4) for s in steps],
           "last_micro_seconds": [round(s["last"], 4) for s in steps],
           "opt_seconds": [round(s["opt"], 4) for s in steps],
           "window_seconds": round(total, 3), "window_max_rank_seconds": round(max_total, 3),
           "global_tokens_per_s": timed * plan["tokens_per_step"] / max_total,
           "timing_scope": "synchronized_train_step", "losses": [round(x, 4) for x in losses]}
    if cfg["world"] > 1 and cfg.get("comm_alone_reps"):
        res["comm_alone"] = comm_alone(cfg["comm_alone_reps"])
    alone = statistics.median(res["comm_alone"]["seconds"]) if "comm_alone" in res else None
    res["split"] = step_split(steps, plan["accum"], alone)
    return res


def drain_loop(cfg, train_step):
    from cluster.drain.metrics import StepTimeHook
    every = cfg["report_every"]
    d = Path(cfg["prom_dir"])
    d.mkdir(parents=True, exist_ok=True)
    hooks = {"sync": StepTimeHook(cfg["node"], cfg["run_id"] + "-sync", d / "sync.prom", every=every),
             "local": StepTimeHook(cfg["node"], cfg["run_id"] + "-local", d / "local.prom", every=every)}
    win = {"sync": 0.0, "local": 0.0}
    steps = []
    for step in range(1, cfg["steps"] + 1):
        if (d / "stop").exists():
            break
        s = train_step(step)
        win["sync"] += s["t"]
        win["local"] += s["local"]
        steps.append({"step": step, "t": round(s["t"], 4), "local": round(s["local"], 4), "last": round(s["last"], 4),
                      "at": round(time.time(), 3)})
        if step % every == 0:
            for k, hook in hooks.items():
                hook.observe(step, win[k] / every)
            win = {"sync": 0.0, "local": 0.0}
    return {"steps": steps}


def fault_loop(cfg, torch, dist, device, cuda, model, opt, train_step, emit, prior):
    import numpy
    import torch.distributed.checkpoint as dcp

    from scaling.checkpoint_retention import CheckpointRetention
    from scaling.fault_train import RNGState, make_torch_hook

    rank = cfg["rank"]
    # Coordinator clock = rank 0's CLOCK_MONOTONIC. Other ranks never emit; they use an offset estimate of it so the
    # hook's own ordering checks hold.
    packet = [prior, time.monotonic()] if rank == 0 else [None, None]
    dist.broadcast_object_list(packet, src=0, device=device if cuda else None)
    prior, r0_now = packet
    offset = 0.0 if rank == 0 else r0_now - time.monotonic()

    def clock():
        return time.monotonic() + offset

    tmod = torch if cuda else cpu_torch_shim(torch)
    hook = make_torch_hook(torch=tmod, device=device, run_id=cfg["run_id"], clock_id=cfg["clock_id"], clock=clock,
                           emit=emit, prior_events=prior)
    rng = RNGState(torch=torch, numpy=numpy, device=device) if cuda else None
    root = Path(cfg["ckpt_dir"])
    retention = CheckpointRetention(root, cfg["run_id"])
    progress_log = jsonl_appender(cfg["progress"]) if rank == 0 else (lambda r: None)
    params = list(model.parameters())

    def optim_tensors():
        return {str(i): {k: v for k, v in sorted(opt.state[p].items()) if torch.is_tensor(v)}
                for i, p in enumerate(params) if p in opt.state}

    def init_optim_state():
        # Materialize AdamW state so DCP can load into it: one zero-gradient step with lr 0 leaves weights unchanged.
        lrs = [g["lr"] for g in opt.param_groups]
        for g in opt.param_groups:
            g["lr"] = 0.0
        for p in params:
            p.grad = torch.zeros_like(p)
        opt.step()
        opt.zero_grad(set_to_none=True)
        for g, lr in zip(opt.param_groups, lrs):
            g["lr"] = lr

    def rng_state():
        return pickle.dumps(rng.state_dict()) if rng is not None else b""

    timings = {"save": [], "load": [], "save_bytes": []}
    state = {"last": None}

    def save(step):
        generation = sum(s == step for s in hook.checkpoints.values())
        event_id = f"{root}/step-{step}@generation-{generation}"

        def op():
            t0 = time.perf_counter()
            gathered = [None] * cfg["world"] if rank == 0 else None
            dist.gather_object(rng_state(), gathered, dst=0)
            hook.coordinate(lambda: retention.begin(step, protected_checkpoint_id=state["last"]))
            try:
                if rank == 0:
                    extra = pickle.dumps({"step": step, "checkpoint_id": event_id, "world": cfg["world"],
                                          "param_groups": opt.state_dict()["param_groups"], "rng": gathered})
                    writer = dcp.FileSystemWriter(str(root / f"step-{step}"), sync_files=True,
                                                  thread_count=cfg.get("ckpt_threads", 1))
                    dcp.save({"model": model.state_dict(), "optim": optim_tensors(), "extra": extra},
                             storage_writer=writer, no_dist=True)
                    timings["save_bytes"].append(sum(p.stat().st_size for p in (root / f"step-{step}").iterdir()))
                hook.coordinate(lambda: retention.confirm(step, event_id))
            finally:
                if rank == 0:
                    retention.end()
            timings["save"].append(round(time.perf_counter() - t0, 2))

        hook.save(event_id, step, op)
        state["last"] = event_id

    def load(event_id, step):
        t0 = time.perf_counter()
        path = event_id.split("@", 1)[0]
        if rank == 0:
            target = {"model": model.state_dict(), "optim": optim_tensors(), "extra": b""}
            dcp.load(target, checkpoint_id=path, no_dist=True)
            extra = pickle.loads(target["extra"])
            if extra["checkpoint_id"] != event_id or extra["step"] != step or extra["world"] != cfg["world"]:
                raise ValueError("checkpoint identity mismatch")
            rngs = extra["rng"]
        else:
            rngs = None
        for t in list(model.state_dict().values()) + [v for d in optim_tensors().values() for v in d.values()]:
            if t.device == device:
                dist.broadcast(t, src=0)
            else:  # e.g. a CPU AdamW step counter on the NCCL path
                tmp = t.to(device)
                dist.broadcast(tmp, src=0)
                t.copy_(tmp.to(t.device))
        mine = [None]
        dist.scatter_object_list(mine, rngs, src=0)
        if rng is not None and mine[0]:
            rng.load_state_dict(pickle.loads(mine[0]))
        timings["load"].append(round(time.perf_counter() - t0, 2))

    total, every = cfg["steps"], cfg["ckpt_every"]
    start = 1
    if cfg.get("resume"):
        saved = [e for e in prior if e["kind"] == "checkpoint_saved"]
        last = saved[-1]
        if not hook.failed:
            hook.failure("restart without a recorded failure")
        init_optim_state()
        hook.restore(last["checkpoint_id"], last["step"], lambda: load(last["checkpoint_id"], last["step"]))
        state["last"] = last["checkpoint_id"]
        start = last["step"] + 1
    for step in range(start, total + 1):
        s = hook.step(step, lambda s=step: train_step(s))
        progress_log({"attempt": cfg.get("attempt", 0), "step": step, "loss": round(s["loss"], 5),
                      "t": round(s["t"], 4), "at": round(time.monotonic(), 3)})
        if cfg.get("exit_after_step") == step:  # CPU tests only: die abruptly, like a SIGKILLed rank
            os._exit(137)
        if step % every == 0:
            save(step)
    if hook.checkpoints.get(state["last"]) != total:
        save(total)
    hook.finish(state["last"])
    return {"timings": timings, "final_checkpoint": state["last"], "resumed_from": start - 1}


def disk_rate(path, nbytes=2 << 30, block=64 << 20):
    """Sequential write with fsync (what a synchronous checkpoint save needs), then delete. GB/s."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = os.urandom(block)
    t = time.perf_counter()
    try:
        with open(path, "wb") as f:
            f.writelines(buf for _ in range(nbytes // block))
            f.flush()
            os.fsync(f.fileno())
        return {"path": str(path.parent), "bytes": nbytes, "seconds": round(time.perf_counter() - t, 2),
                "GBps": round(nbytes / (time.perf_counter() - t) / 1e9, 3)}
    except OSError as e:
        return {"path": str(path.parent), "error": repr(e)}
    finally:
        path.unlink(missing_ok=True)


def gpu_load(cfg):
    """Competing CUDA load: back-to-back bf16 matmuls on the same GPU until killed."""
    import torch
    torch.cuda.set_device(0)
    n = cfg.get("n", 4096)
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    Path(cfg["ready"]).write_text("1")
    while True:
        for _ in range(20):
            a = (a @ b).clamp_(-1, 1)
        torch.cuda.synchronize()


def matmul_bench(cfg):
    import torch
    torch.cuda.set_device(0)
    out = {}
    for n in (4096, 8192):
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        reps = []
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(20):
                a @ b
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 20
            reps.append(round(2 * n**3 / dt / 1e12, 1))
        out[f"bf16_{n}"] = reps
    return {"tflops": out, "gpu": torch.cuda.get_device_name(0), "cupti_probe": cupti_probe(cfg, torch, a, b)}


def cupti_probe(cfg, torch, a, b):
    """Can torch.profiler (CUPTI) record CUDA kernels here? nsys/ncu need the same vGPU profiling support."""
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                a @ b
            torch.cuda.synchronize()
        path = Path(cfg["out_dir"]) / "cupti.json"
        prof.export_chrome_trace(str(path))
        rows, skipped = kernel_rows(json.loads(path.read_text()))
        path.unlink(missing_ok=True)
        res = {"kernel_records": len(rows), "zero_length": skipped}
    except Exception as e:  # noqa: BLE001 - record and go on
        res = {"error": repr(e)[:300]}
    log = Path(cfg["out_dir"]) / "log.txt"
    res["log"] = [line.strip()[:200] for line in (log.read_text(errors="replace").splitlines() if log.exists() else [])
                  if "CUPTI" in line][:4]
    return res


def rank_main(cfg_path):
    cfg = json.loads(Path(cfg_path).read_text())
    task = cfg["task"]
    fn = {"nccl": nccl_rank, "train": train_rank, "load": gpu_load, "matmul": matmul_bench}[task]
    res = fn(cfg)
    Path(cfg["result"]).write_text(json.dumps(res))


# ---------------------------------------------------------------- node actor (driver side)

def sh(*cmd, timeout=60, env=None):
    try:
        r = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, check=False, env=env)
        return {"rc": r.returncode, "out": (r.stdout.strip() + ("\n" + r.stderr.strip() if r.stderr.strip() else ""))[-3000:]}
    except Exception as e:  # noqa: BLE001 - record any failure and go on
        return {"rc": None, "out": repr(e)}


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as e:
        return repr(e)


def make_actor():
    import ray

    @ray.remote(num_gpus=1, num_cpus=1, max_concurrency=8)
    class Node:
        def __init__(self):
            sys.path.insert(0, str(ROOT))
            self.procs = {}
            BASE.mkdir(parents=True, exist_ok=True)

        def info(self):
            return {"ip": ray.util.get_node_ip_address(), "host": socket.gethostname(), "pid": os.getpid(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "time": time.time(),
                    "boot_id": read("/proc/sys/kernel/random/boot_id")}

        def clock(self):
            return time.time()

        def setup(self, bundle):
            """Unpack the driver's model + tokens tar (Snowpark sessions exist only in the driver process)."""
            import io
            import tarfile
            if (BASE / "model" / "config.json").exists() and (BASE / "tokens.pt").exists():
                return {"host": socket.gethostname(), "unpacked": False}
            t = time.time()
            with tarfile.open(fileobj=io.BytesIO(bundle)) as tar:
                tar.extractall(BASE, filter="data")
            return {"host": socket.gethostname(), "unpacked": True, "seconds": round(time.time() - t, 1)}

        def health(self, pip_probe, secrets):
            return node_health(pip_probe, secrets)

        def launch(self, name, cfg, env):
            d = BASE / "runs" / name
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
            cfg = {**cfg, "result": str(d / "result.json"), "out_dir": str(d)}
            (d / "cfg.json").write_text(json.dumps(cfg))
            log = open(d / "log.txt", "w")  # noqa: SIM115 - closed with the process
            p = subprocess.Popen([sys.executable, str(PAYLOAD), "--mode", "rank", "--cfg", str(d / "cfg.json")],
                                 stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env, "PYTHONUNBUFFERED": "1"},
                                 cwd=str(ROOT), start_new_session=True)
            self.procs[name] = (p, log, d)
            return p.pid

        def poll(self, name):
            return self.procs[name][0].poll()

        def kill(self, name, sig=signal.SIGKILL):
            p = self.procs[name][0]
            if p.poll() is None:
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    pass
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            return p.poll()

        def result(self, name, tail=40):
            p, log, d = self.procs[name]
            rc = p.poll()
            log.flush()
            res = {"rc": rc}
            try:
                res.update(json.loads((d / "result.json").read_text()))
            except (OSError, json.JSONDecodeError):
                res["log_tail"] = (d / "log.txt").read_text(errors="replace").splitlines()[-tail:]
            if rc is not None:
                log.close()
            return res

        def read(self, path):
            try:
                return Path(path).read_text()
            except OSError:
                return None

        def touch(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("1")
            return True

        def carrier(self):
            return read("/sys/class/net/eth0/carrier")

        def disk(self):
            return sh("df", "-h", "/tmp", "/dev/shm")["out"]

    return Node


def tokenize(data_dir, model_dir, out_path):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    files = sorted(Path(data_dir).rglob("train-*.parquet"))
    try:
        import pyarrow.parquet as pq
        rows = [r for f in files for r in pq.read_table(f).to_pylist()]
    except ImportError:
        import pandas as pd
        rows = [r for f in files for r in pd.read_parquet(f).to_dict("records")]
    docs = tok([f"Question: {r['question']}\nAnswer: {r['answer']}" for r in rows], add_special_tokens=False)["input_ids"]
    seqs = pack_tokens(docs, SEQ, tok.eos_token_id)
    torch.save(torch.tensor(seqs, dtype=torch.int32), out_path)
    return {"examples": len(rows), "sequences": len(seqs), "tokens": len(seqs) * SEQ, "files": [f.name for f in files]}


def node_health(pip_probe, secrets):
    from types import SimpleNamespace

    from cluster.provision import node_probe
    out = {"host": socket.gethostname()}
    for kind in ("host", "workload"):
        args = SimpleNamespace(kind=kind, host_root=False, shared_root="/tmp", token="e5-token", scheduler="none",
                               image=None)
        try:
            data = node_probe.probe(args)
            for c in data.get("commands", []):
                c["stdout"] = (c.get("stdout") or "")[:1500]
                c["stderr"] = (c.get("stderr") or "")[:500]
            out[f"node_probe_{kind}"] = data
        except Exception as e:  # noqa: BLE001 - record and go on
            out[f"node_probe_{kind}"] = {"error": repr(e)}
    out["tools"] = {t: shutil.which(t) for t in ("dcgmi", "nv-hostengine", "all_reduce_perf", "ibstat",
                                                 "ib_write_bw", "nsys", "ncu", "mpirun", "nvcc")}
    out["libs"] = {"libdcgm": sorted(str(p) for d in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/lib")
                                     for p in Path(d).glob("libdcgm.so*"))[:4],
                   "nccl_tests_on_disk": sorted(str(p) for d in ("/opt", "/usr/local")
                                                for pat in ("*/build/all_reduce_perf", "*/all_reduce_perf")
                                                for p in Path(d).glob(pat))[:4]}
    if out["tools"]["dcgmi"]:
        out["dcgmi_diag_r1"] = sh("dcgmi", "diag", "-r", "1", "-j", timeout=300)
    if out["tools"]["ncu"]:
        out["ncu_version"] = sh("ncu", "--version", timeout=30)["out"][-300:]
    ib = Path("/sys/class/infiniband")
    out["infiniband_sysfs"] = sorted(p.name for p in ib.iterdir()) if ib.exists() else None
    if ib.exists():
        out["infiniband_ports"] = {f"{dev.name}/{port.name}": {k: read(port / k) for k in ("state", "phys_state", "rate",
                                                                                         "link_layer")}
                                   for dev in ib.iterdir() for port in sorted((dev / "ports").glob("*"))}
    out["dev_infiniband"] = sorted(os.listdir("/dev/infiniband")) if os.path.isdir("/dev/infiniband") else None
    out["ibv_devinfo"] = sh("ibv_devinfo")["out"][:1500] if shutil.which("ibv_devinfo") else None
    out["nvidia_smi"] = sh("nvidia-smi", "--query-gpu=index,name,pci.bus_id,memory.total,driver_version,"
                           "clocks.max.sm,clocks.sm,power.limit,ecc.mode.current,pstate",
                           "--format=csv,noheader")["out"]
    fields = ("temperature.gpu", "utilization.gpu", "memory.used", "ecc.errors.uncorrected.volatile.total",
              "ecc.errors.uncorrected.aggregate.total", "retired_pages.pending", "clocks_event_reasons.active",
              "pcie.link.gen.current", "pcie.link.width.current")
    out["nvidia_smi_health"] = {f: sh("nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader")["out"][:200]
                                for f in fields}
    q = sh("nvidia-smi", "-q")["out"].splitlines()
    out["nvidia_smi_q_excerpt"] = [line.strip() for line in q
                                   if any(k in line for k in ("Product Name", "vGPU", "Virtualization", "Host VGPU",
                                                              "License", "Driver Model", "GPU Virtualization"))][:20]
    out["nvidia_smi_topo"] = sh("nvidia-smi", "topo", "-m")["out"][:600]
    out["disk_write"] = [disk_rate(BASE / "disktest" / "w.bin"), disk_rate(Path("/dev/shm/e5-disktest/w.bin"))]
    out["numa"] = {p.name: read(p / "numa_node") for p in Path("/sys/bus/pci/devices").glob("*")
                   if read(p / "vendor") == "0x10de"}
    out["disk"] = sh("df", "-h", "/tmp", "/dev/shm")["out"]
    out["mem"] = read("/proc/meminfo").splitlines()[:3]
    out["cpus"] = {"count": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)),
                   "cgroup_cpu_max": read("/sys/fs/cgroup/cpu.max")}
    out["eth0"] = {"carrier": read("/sys/class/net/eth0/carrier"), "mtu": read("/sys/class/net/eth0/mtu"),
                   "speed": read("/sys/class/net/eth0/speed")}
    if pip_probe:
        pip = [sys.executable, "-m", "pip", "--timeout", "15", "--retries", "0"]
        probes = {"torchtitan_0.2.0": sh(*pip, "download", "--no-deps", "-d", str(BASE / "pip"), "torchtitan==0.2.0",
                                         timeout=180),
                  "torch_nightly_2.10.0.dev20251019": sh(*pip, "download", "--no-deps", "-d", str(BASE / "pip"),
                                                         "--pre", "torch==2.10.0.dev20251019", timeout=180),
                  "torch_nightly_index": sh(*pip, "download", "--no-deps", "-d", str(BASE / "pip"), "--pre",
                                            "--index-url", "https://download.pytorch.org/whl/nightly/cu126",
                                            "torch==2.10.0.dev20251019+cu126", timeout=120),
                  "torchao_nightly_0.15.0.dev20251015": sh(*pip, "download", "--no-deps", "-d", str(BASE / "pip"),
                                                           "--pre", "torchao==0.15.0.dev20251015", timeout=120),
                  "pip_env": {k: v for k, v in os.environ.items() if k.startswith("PIP_")}}
        out["pip_probe"] = json.loads(redact(json.dumps(probes), secrets))
    return out


# ---------------------------------------------------------------- driver

class Cluster:
    """The node actors, head first, plus port and run-name bookkeeping."""

    def __init__(self, ray, nodes, infos, head):
        self.ray, self.nodes, self.infos, self.head = ray, nodes, infos, head
        self.k = 0

    def port(self):
        p = PORTS[self.k % len(PORTS)]
        self.k += 1
        return p

    def get(self, refs, timeout=None):
        return self.ray.get(refs, timeout=timeout)

    def run_world(self, world, name, cfg, env, timeout, cfg_for_rank=None):
        """Start `world` rank processes on nodes[0..world-1], wait for all, return their results."""
        port = self.port()
        master = self.infos[0]["ip"]
        names = [f"{name}-r{r}" for r in range(world)]
        for r in range(world):
            c = {**cfg, "rank": r, "world": world, "master": master, "port": port}
            if cfg_for_rank:
                c.update(cfg_for_rank(r))
            self.get(self.nodes[r].launch.remote(names[r], c, env))
        deadline = time.time() + timeout
        while time.time() < deadline:
            rcs = self.get([self.nodes[r].poll.remote(names[r]) for r in range(world)])
            if all(rc is not None for rc in rcs):
                break
            if any(rc not in (None, 0) for rc in rcs):  # one rank failed: give the others a moment, then stop
                time.sleep(20)
                break
            time.sleep(2)
        for r in range(world):
            self.get(self.nodes[r].kill.remote(names[r]))
        return {"port": port, "ranks": self.get([self.nodes[r].result.remote(names[r]) for r in range(world)])}


def nccl_env(extra):
    return {"NCCL_IB_DISABLE": "1", "NCCL_DEBUG": "INFO", "NCCL_DEBUG_FILE": "/tmp/nccl.%h.%p.log", **extra}


def phase_health(c, a):
    secrets = [os.environ.get("SNOWFLAKE_ACCOUNT", "")]
    health = c.get([n.health.remote(i == 0, secrets) for i, n in enumerate(c.nodes)], timeout=900)
    mm = c.get([n.launch.remote("matmul", {"task": "matmul"}, {}) for n in c.nodes])
    for _ in range(120):
        if all(rc is not None for rc in c.get([n.poll.remote("matmul") for n in c.nodes])):
            break
        time.sleep(2)
    matmul = c.get([n.result.remote("matmul") for n in c.nodes])
    for h, m in zip(health, matmul):
        h["matmul"] = m
    progress("health: " + json.dumps([{"host": h["host"], "tools": h["tools"], "tflops": h["matmul"].get("tflops")}
                                      for h in health])[:1500])
    return {"nodes": health, "launched": mm}


def phase_collectives(c, a):
    from scripts.snowflake.payloads.e0_network import QUICK_SIZES, SIZES, summarize_runs
    sizes = list(QUICK_SIZES if a.quick else SIZES)
    worlds = [w for w in (2, 3, 4) if w <= len(c.nodes)]
    runs = []
    for p in range(1 if a.quick else a.passes):
        for world in worlds:
            plan = ([] if a.quick or p else ["default"]) + (["ib_enabled"] if p == 0 and world in (2, 4) else []) \
                + ["tuned", "tuned_buff16m"]
            for variant in plan:
                t0 = time.time()
                # The IB probe only asks whether NCCL picks a non-socket transport; small sizes suffice.
                vsizes = list(QUICK_SIZES) if variant == "ib_enabled" else sizes
                r = c.run_world(world, f"nccl-{world}-{variant}-{p}", {"task": "nccl", "sizes": vsizes},
                                nccl_env(VARIANTS[variant]), timeout=300 if variant == "ib_enabled" else 900)
                ok = all("bw" in x for x in r["ranks"])
                run = {"world": world, "variant": variant, "pass": p, "env": VARIANTS[variant], "port": r["port"],
                       "seconds": round(time.time() - t0, 1)}
                if ok:
                    run["ranks"] = r["ranks"]
                else:
                    run["error"] = r["ranks"]
                runs.append(run)
                brief = {f"{b['op']}@{b['bytes'] >> 20}M": b["GBps"] for b in r["ranks"][0].get("bw", [])} if ok \
                    else str(r["ranks"])[:300]
                progress(f"nccl pass {p} world {world} {variant}: {run['seconds']} s {brief}")
    keyed = [{**x, "variant": f"{x['world']}n/{x['variant']}"} for x in runs]
    return {"runs": runs, "summary_algbw": summarize_runs(keyed),
            "summary_busbw": summarize_runs([{**x, "ranks": [{"bw": [{**b, "GBps": b["busGBps"]} for b in rk["bw"]]}
                                                             for rk in x["ranks"]]} for x in keyed if "ranks" in x])}


def train_cfg(kind, plan, **extra):
    return {"task": "train", "kind": kind, "plan": plan, "model": str(BASE / "model"), "tokens": str(BASE / "tokens.pt"),
            **extra}


def phase_scaling(c, a):
    n = len(c.nodes)
    worlds = (2, 4) if n >= 4 else tuple(range(2, n + 1))
    configs = [("weak", 1)] + [(k, w) for w in worlds for k in ("weak", "strong") if STRONG_GLOBAL_SEQ % w == 0
                                or k == "weak"]
    rounds = 1 if a.quick else a.rounds
    warm, timed, reps = (2, 3, 1) if a.quick else (3, 10, 3)
    runs = []
    for rnd in range(1, rounds + 1):
        for kind, world in configs:
            plan = batch_plan(kind, world)
            t0 = time.time()
            r = c.run_world(world, f"scale-{kind}-{world}-{rnd}",
                            train_cfg("scale", plan, warmup_steps=warm, timed_steps=timed, comm_alone_reps=reps),
                            nccl_env(TRAIN_NCCL), timeout=1200)
            run = {"kind": kind, "world": world, "round": rnd, "plan": plan, "seconds": round(time.time() - t0, 1),
                   "ranks": r["ranks"]}
            runs.append(run)
            r0 = r["ranks"][0]
            progress(f"scale {kind} x{world} round {rnd}: {run['seconds']} s tokens/s "
                     f"{r0.get('global_tokens_per_s')} split {r0.get('split')} comm_alone "
                     f"{r0.get('comm_alone', {}).get('seconds')} "
                     f"{'' if 'global_tokens_per_s' in r0 else str(r0)[:600]}")
    return {"runs": runs, "peak_flops": A10_BF16_DENSE_PEAK, "nccl_env": TRAIN_NCCL, "bucket_cap_mb": 25}


def clock_offsets(c):
    out = []
    for n in c.nodes:
        t0 = time.time()
        remote = c.get(n.clock.remote())
        t1 = time.time()
        out.append({"offset": (t0 + t1) / 2 - remote, "rtt": t1 - t0})
    return out


def phase_drain(c, a):
    from cluster.drain.controller import Controller, SchedulerExecutor
    n = len(c.nodes)
    plan = batch_plan("weak", n, per_gpu_seq=8)
    every = 3 if a.quick else 4
    steps = 30 if a.quick else 56
    load_from, load_to = (3, 6) if a.quick else (4, 10)  # samples after which the load starts / stops
    slow = n - 1
    offsets = clock_offsets(c)
    hosts = [i["host"] for i in c.infos]
    port = c.port()
    master = c.infos[0]["ip"]
    names = [f"drain-r{r}" for r in range(n)]
    prom = [str(BASE / "drain" / f"node{r}") for r in range(n)]
    for r in range(n):
        cfg = {**train_cfg("drain", plan, steps=steps, report_every=every, run_id="e5-drain", node=hosts[r],
                           prom_dir=prom[r]), "rank": r, "world": n, "master": master, "port": port}
        c.get(c.nodes[r].launch.remote(names[r], cfg, nccl_env(TRAIN_NCCL)))
    ctrls = {"sync": Controller(clock=time.time), "local": Controller(clock=time.time)}
    executor = SchedulerExecutor("slurm", set(hosts), owner="e5", dry_run=True)
    logs = {"sync": [], "local": []}
    seen, sample, load = 0, 0, {}
    deadline = time.time() + (900 if a.quick else 1800)
    while time.time() < deadline:
        rcs = c.get([c.nodes[r].poll.remote(names[r]) for r in range(n)])
        texts = {k: c.get([c.nodes[r].read.remote(f"{prom[r]}/{k}.prom") for r in range(n)]) for k in ("sync", "local")}
        parsed = {k: [parse_prom(t) if t else None for t in v] for k, v in texts.items()}
        idx = [int(p["crucible_step_index"][1]) if p else 0 for p in parsed["sync"]]
        if min(idx) > seen and all(p for p in parsed["local"]):
            seen = min(idx)
            sample += 1
            now = time.time()
            carriers = c.get([nd.carrier.remote() for nd in c.nodes])
            for k in ("sync", "local"):
                samples = parsed[k]
                if any(int(s["crucible_step_index"][1]) != seen for s in samples):
                    progress(f"drain sample {sample}: nodes at different windows {idx}, evaluating latest")
                metrics = {hosts[r]: node_metrics(samples[r], now=now, offset=offsets[r]["offset"],
                                                  link_up=carriers[r] == "1") for r in range(n)}
                actions = ctrls[k].evaluate(metrics)
                results = executor.execute(actions, jobs=[])
                logs[k].append({"sample": sample, "step": seen, "at": now,
                                "times": {hosts[r]: samples[r]["crucible_step_time_seconds"][1] for r in range(n)},
                                "steps": {hosts[r]: int(samples[r]["crucible_step_index"][1]) for r in range(n)},
                                "alerts": step_alerts(actions),
                                "actions": [{"node": x.node, "kind": x.kind, "reason": x.reason} for x in actions],
                                "executor": [{k2: v for k2, v in x.items() if k2 != "evidence"} for x in results],
                                "load_on": bool(load) and "stopped_at" not in load})
            progress(f"drain sample {sample} step {seen}: local "
                     f"{[round(x, 3) for x in logs['local'][-1]['times'].values()]} alerts {logs['local'][-1]['alerts']}"
                     f" | sync {[round(x, 3) for x in logs['sync'][-1]['times'].values()]} alerts {logs['sync'][-1]['alerts']}")
            if sample == load_from and not load:
                c.get(c.nodes[slow].launch.remote("load", {"task": "load", "ready": str(BASE / "load.ready")}, {}))
                load = {"node": hosts[slow], "started_after_sample": sample, "started_at": time.time()}
            elif sample == load_to and load and "stopped_at" not in load:
                c.get(c.nodes[slow].kill.remote("load"))
                load.update(stopped_after_sample=sample, stopped_at=time.time())
        if all(rc is not None for rc in rcs):
            break
        time.sleep(1)
    if load and "stopped_at" not in load:
        c.get(c.nodes[slow].kill.remote("load"))
        load.update(stopped_after_sample=sample, stopped_at=time.time())
    for r in range(n):
        c.get(c.nodes[r].kill.remote(names[r]))
    ranks = c.get([c.nodes[r].result.remote(names[r]) for r in range(n)])
    return {"plan": plan, "report_every": every, "slow_node": hosts[slow], "slow_method": "competing CUDA process "
            "(bf16 4096x4096 matmul loop) on the same GPU", "load": load, "clock_offsets": offsets,
            "xid_source": "none (no DCGM on the node); xid_errors supplied as 0", "link_source": "/sys/class/net/eth0/carrier",
            "rules": ctrls["local"].rules, "logs": logs,
            "verify": {k: verify_drain(logs[k], hosts[slow]) for k in logs},
            "ranks": ranks}


def record_failure(events_path, reason):
    """Supervisor side: record the failure through the repo's hook on the coordinator clock (the driver shares the
    head node, hence CLOCK_MONOTONIC, with rank 0). If rank 0 already logged operation_failed, nothing is added."""
    from scaling.goodput import FaultTrainingHook
    repair_jsonl(events_path)  # rank 0 may have died mid-line
    hook = FaultTrainingHook.resume(read_jsonl(events_path), emit=jsonl_appender(events_path), clock=time.monotonic)
    if hook.failed:
        return "rank0 operation_failed"
    hook.failure(reason)
    return "driver"


FAULT_STORES = {"disk": BASE / "fault", "shm": Path("/dev/shm/e5-fault")}  # both on the head node (rank 0)


def fault_plan(a):
    """(round, checkpoint store) in run order: every round on disk, the first --fault-shm-rounds also on /dev/shm."""
    rounds = 1 if a.quick else a.fault_rounds
    return [(rnd, store) for rnd in range(1, rounds + 1)
            for store in ("disk", "shm") if store == "disk" or rnd <= a.fault_shm_rounds]


def phase_fault(c, a):
    if c.infos[0]["ip"] != c.head:
        raise RuntimeError("rank 0 is not on the head node: no shared coordinator clock with the driver")
    runs = []
    for rnd, store in fault_plan(a):
        t0 = time.time()
        try:
            r = fault_run(c, a, store, rnd)
        except Exception as e:  # noqa: BLE001 - one failed round must not hide the others
            import traceback
            r = {"error": repr(e)[:2000], "traceback": traceback.format_exc()[-3000:]}
        r = {"round": rnd, "store": store, "seconds": round(time.time() - t0, 1), **r}
        runs.append(r)
        progress(f"fault round {rnd} {store}: {r['seconds']} s accounting {r.get('accounting', r.get('error'))}")
    for d in FAULT_STORES.values():
        shutil.rmtree(d, ignore_errors=True)
    return {"runs": runs}


def fault_run(c, a, store, rnd):
    from scaling.goodput import account_events
    n = len(c.nodes)
    plan = batch_plan("weak", n, per_gpu_seq=8)
    total, every, kill_at = (12, 4, 6) if a.quick else (30, 10, 25)
    victim = n - 1
    run_id = f"e5-fault-{store}-{rnd}"
    clock_id = f"monotonic:{c.infos[0]['boot_id']}"
    fdir = FAULT_STORES[store] / f"round{rnd}"
    for d in FAULT_STORES.values():  # leave no earlier round's checkpoints behind (disk and RAM)
        if d.exists():
            shutil.rmtree(d)
    events_path, progress_path = fdir / "events.jsonl", fdir / "progress.jsonl"
    base_cfg = train_cfg("fault", plan, steps=total, ckpt_every=every, run_id=run_id, clock_id=clock_id,
                         events=str(events_path), progress=str(progress_path), ckpt_dir=str(fdir / "ckpt"),
                         ckpt_threads=a.ckpt_threads)
    env = nccl_env(TRAIN_NCCL)
    timeline = {}

    def launch(attempt, resume):
        port, master = c.port(), c.infos[0]["ip"]
        names = [f"fault-{store}{rnd}-a{attempt}-r{r}" for r in range(n)]
        for r in range(n):
            c.get(c.nodes[r].launch.remote(names[r], {**base_cfg, "rank": r, "world": n, "master": master, "port": port,
                                                      "attempt": attempt, "resume": resume}, env))
        return names

    names = launch(0, False)
    timeline["attempt0_launched"] = time.monotonic()
    killed = None
    deadline = time.time() + (900 if a.quick else 1800)
    while time.time() < deadline:
        ev = read_jsonl(events_path)
        done = [e["step"] for e in ev if e["kind"] == "step_complete"]
        rcs = c.get([c.nodes[r].poll.remote(names[r]) for r in range(n)])
        if done and max(done) >= kill_at:
            c.get(c.nodes[victim].kill.remote(names[victim], signal.SIGKILL))
            killed = {"rank": victim, "host": c.infos[victim]["host"], "after_step": max(done),
                      "at": time.monotonic(), "rc": c.get(c.nodes[victim].poll.remote(names[victim]))}
            break
        if any(rc is not None for rc in rcs):
            break
        time.sleep(0.5)
    if killed is None:
        for r in range(n):
            c.get(c.nodes[r].kill.remote(names[r]))
        return {"error": "fault attempt 0 ended before the kill point",
                "ranks": c.get([c.nodes[r].result.remote(names[r]) for r in range(n)]), "events": read_jsonl(events_path)}
    # Supervisor role (what torchrun's elastic agent does): the killed rank is gone, so stop the survivors, which are
    # blocked in (or failing out of) the gradient all-reduce, before restarting the whole world.
    survivors = {}
    for r in range(n):
        if r != victim:
            t0 = time.monotonic()
            rc_before = c.get(c.nodes[r].poll.remote(names[r]))
            survivors[r] = {"rc_before_teardown": rc_before, "rc": c.get(c.nodes[r].kill.remote(names[r])),
                            "teardown_s": round(time.monotonic() - t0, 2)}
    timeline["teardown_done"] = time.monotonic()
    attempt0 = c.get([c.nodes[r].result.remote(names[r], 15) for r in range(n)])
    recorded_by = record_failure(events_path, f"rank {victim} process SIGKILLed by the E5 driver after step "
                                              f"{killed['after_step']}")
    timeline["failure_recorded"] = time.monotonic()
    names = launch(1, True)
    timeline["attempt1_launched"] = time.monotonic()
    deadline = time.time() + (900 if a.quick else 1800)
    while time.time() < deadline:
        rcs = c.get([c.nodes[r].poll.remote(names[r]) for r in range(n)])
        if all(rc is not None for rc in rcs) or any(rc not in (None, 0) for rc in rcs):
            break
        time.sleep(1)
    if any(rc not in (None, 0) for rc in rcs):
        time.sleep(15)
    for r in range(n):
        c.get(c.nodes[r].kill.remote(names[r]))
    timeline["attempt1_done"] = time.monotonic()
    attempt1 = c.get([c.nodes[r].result.remote(names[r]) for r in range(n)])
    events = read_jsonl(events_path)
    try:
        accounting = account_events(events)
    except ValueError as e:
        accounting = {"error": str(e)}
    t0 = events[0]["at"] if events else 0
    return {"plan": plan, "steps": total, "ckpt_every": every, "kill_after_step": kill_at, "killed": killed,
            "survivors": survivors, "failure_recorded_by": recorded_by,
            "timeline_rel_s": {k: round(v - t0, 2) for k, v in timeline.items()} | {"kill": round(killed["at"] - t0, 2)},
            "checkpoint_location": f"{fdir / 'ckpt'} on {c.infos[0]['host']} (rank 0, head node); victim on "
                                   f"{c.infos[victim]['host']}",
            "events": events, "progress": read_jsonl(progress_path), "accounting": accounting,
            "attempt0": attempt0, "attempt1": attempt1, "disk_after": c.get(c.nodes[0].disk.remote())}


PHASES = {"health": phase_health, "collectives": phase_collectives, "scaling": phase_scaling, "drain": phase_drain,
          "fault": phase_fault}


def driver(a):
    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False)
    t_start = time.time()
    gpu_nodes = [x for x in ray.nodes() if x["Alive"] and x["Resources"].get("GPU", 0) >= 1]
    # Wait briefly for late instances to join Ray.
    for _ in range(60):
        if len(gpu_nodes) >= a.nodes:
            break
        time.sleep(5)
        gpu_nodes = [x for x in ray.nodes() if x["Alive"] and x["Resources"].get("GPU", 0) >= 1]
    n = min(a.nodes, len(gpu_nodes))
    pg = placement_group([{"GPU": 1, "CPU": 1}] * n, strategy="STRICT_SPREAD")
    ray.get(pg.ready(), timeout=600)
    Node = make_actor()
    actors = [Node.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg, placement_group_bundle_index=i)).remote() for i in range(n)]
    infos = ray.get([x.info.remote() for x in actors], timeout=300)
    head = ray.util.get_node_ip_address()
    order = sorted(range(n), key=lambda i: (infos[i]["ip"] != head, infos[i]["ip"]))
    c = Cluster(ray, [actors[i] for i in order], [infos[i] for i in order], head)
    meta = {"phase": "setup", "nodes_requested": a.nodes, "ray_gpu_nodes": len(gpu_nodes), "nodes_used": n,
            "head_ip": head, "infos": c.infos, "rank0_on_head": c.infos[0]["ip"] == head}
    progress(f"nodes {[(i['host'], i['ip']) for i in c.infos]} rank0_on_head={meta['rank0_on_head']}")
    meta["setup"] = stage_inputs(ray, c)
    meta["versions"] = versions()
    meta["seconds"] = round(time.time() - t_start, 1)
    print("RESULT " + json.dumps(meta), flush=True)
    for name in a.phases:
        t0 = time.time()
        progress(f"phase {name} start")
        try:
            res = PHASES[name](c, a)
        except Exception as e:  # noqa: BLE001 - one phase failing must not hide the others
            import traceback
            res = {"error": repr(e)[:2000], "traceback": traceback.format_exc()[-4000:]}
        res = {"phase": name, "quick": a.quick, "nodes": n, "seconds": round(time.time() - t0, 1), **res}
        progress(f"phase {name} done in {res['seconds']} s")
        out = BASE / "results" / a.run
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.json").write_text(json.dumps(res, default=str))
        up = upload_results(a.run) if a.run else {"upload": "skipped (no --run)"}
        print("RESULT " + json.dumps({**compact(res), "full_result": up}, default=str), flush=True)


def upload_results(run):
    """Pack every phase result of this job so far and put it on the asset stage (results.py get e5/<run>), because
    the job log fetch truncates long output (a 139 KB RESULT line was cut in fullA)."""
    import tarfile
    archive = BASE / "results" / f"{run}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(BASE / "results" / run, arcname=run)
    try:
        from snowflake.snowpark import Session

        from scripts.snowflake.assets import STAGE
        s = Session.builder.getOrCreate()
        s.file.put(str(archive), f"@{STAGE}/results/e5", auto_compress=False, overwrite=True)
        return {"uploaded": f"results/e5/{archive.name}", "bytes": archive.stat().st_size}
    except Exception as e:  # noqa: BLE001 - the compact RESULT line still carries the key numbers
        return {"upload_error": repr(e)[:300], "bytes": archive.stat().st_size}


RANK0_KEYS = ("rc", "rank", "host", "world", "kind", "init_seconds", "parameters", "structure", "plan", "step_seconds",
              "local_micro_seconds", "last_micro_seconds", "opt_seconds", "window_seconds", "window_max_rank_seconds",
              "global_tokens_per_s", "losses", "comm_alone", "split", "peak_mem_gib", "timings", "final_checkpoint",
              "resumed_from")


def slim_rank(x, rank0):
    keep = RANK0_KEYS if rank0 else ("rc", "rank", "host", "global_tokens_per_s", "window_max_rank_seconds",
                                     "peak_mem_gib", "timings", "check_ok", "bw")
    out = {k: x[k] for k in keep if k in x}
    if "log_tail" in x:
        out["log_tail"] = x["log_tail"][-4:]
    if "nccl_log" in x:
        out["nccl_log"] = [line for line in x["nccl_log"] if "NET/" in line or "Using network" in line][:3]
    if "check_ok" in x and "bw" in x:
        out.update(check_ok=x["check_ok"], bw=x["bw"])
    return out


def compact(res):
    """The phase result without bulky per-rank detail, small enough for one log line; e5_results reads either."""
    drop = {"node_probe_host", "node_probe_workload"}
    out = {}
    for k, v in res.items():
        if k == "ranks" and isinstance(v, list):
            out[k] = [slim_rank(x, i == 0) for i, x in enumerate(v)]
        elif k == "nodes" and isinstance(v, list):
            out[k] = [{kk: vv for kk, vv in x.items() if kk not in drop} for x in v]
        elif k in ("runs", "attempt0", "attempt1") and isinstance(v, list):
            out[k] = [compact(x) if isinstance(x, dict) and "rc" not in x else slim_rank(x, i == 0)
                      for i, x in enumerate(v)]
        elif k in ("summary_algbw", "launched"):
            continue
        else:
            out[k] = v
    return out


def stage_inputs(ray, c):
    """Fetch model and data on the head (driver), tokenize once, ship one tar to every node through Ray."""
    import io
    import tarfile

    from scripts.snowflake.assets import fetch
    t = time.time()
    got = {"model": fetch(MODEL_ASSET, str(BASE / "model")), "data": fetch(DATA_ASSET, str(BASE / "data"))}
    got["tokens"] = tokenize(BASE / "data", BASE / "model", BASE / "tokens.pt")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(BASE / "model", arcname="model")
        tar.add(BASE / "tokens.pt", arcname="tokens.pt")
    ref = ray.put(buf.getvalue())
    got["bundle_bytes"] = buf.tell()
    del buf
    got["nodes"] = ray.get([x.setup.remote(ref) for x in c.nodes], timeout=1800)
    got["seconds"] = round(time.time() - t, 1)
    progress(f"inputs staged in {got['seconds']} s: {got['tokens']}")
    return got


def versions():
    import importlib.metadata as md
    out = {}
    for p in ("torch", "torchvision", "transformers", "numpy", "ray", "nvidia-nccl-cu12", "pyarrow", "pyyaml"):
        try:
            out[p] = md.version(p)
        except Exception:  # noqa: BLE001
            out[p] = None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("driver", "rank"), default="driver")
    ap.add_argument("--cfg")
    ap.add_argument("--phases", nargs="*", choices=sorted(PHASES), default=list(PHASES))
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--run", default="", help="tag: full results go to @ASSETS/results/e5/<run>.tar.gz")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--ckpt-threads", type=int, default=4, help="DCP FileSystemWriter threads for the fault phase")
    ap.add_argument("--fault-rounds", type=int, default=3, help="fault runs with checkpoints on the head node's disk")
    ap.add_argument("--fault-shm-rounds", type=int, default=1, help="of those rounds, also run with /dev/shm")
    a = ap.parse_args(argv)
    if a.mode == "rank":
        rank_main(a.cfg)
    else:
        driver(a)


if __name__ == "__main__":
    main()
