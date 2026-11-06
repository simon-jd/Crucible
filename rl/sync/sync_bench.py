"""Weight-sync microbenchmark between two GPUs: per-tensor WeightSync, BucketedSync, raw broadcasts.

Two processes, one GPU each, over a separate pair group (rl.sync.pair_group), as in E1/E2:

  --role source    holds the base model's BF16 tensors on its GPU (Qwen2.5-0.5B-Instruct:
                   290 tensors, 988,065,536 bytes) and pushes them
  --role receiver  receives them into its GPU memory

Variants, run in interleaved passes (pass 0 is a warm-up and is not counted):

  per_tensor_sum64   WeightSync + MemoryStore(digest="sum64"): the E2 path
  per_tensor_sha256  WeightSync + MemoryStore(digest="sha256"): the E1 path
  bucket_<N>M        BucketedSync with N MiB buckets and a SHA-256 manifest (rl/sync/bucketed.py)
  raw_<N>M           the packed buffer as plain dist.broadcast calls of N MiB: no control frames,
                     no hashing, no checks; the link floor for that call size

Every timed push: both ranks meet at a barrier (all-reduce + cuda synchronize), then each times
push + cuda synchronize. In the warm-up pass the receiver also compares every received tensor
with the base weights it reads itself. The source writes result.json with both ranks' records
and prints one line "ROLE_RESULT <json>".
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path


def _sync(device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()


def _barrier(group, device):
    import torch
    import torch.distributed as dist

    t = time.monotonic()
    flag = torch.zeros(1, device=device)
    dist.all_reduce(flag, group=group)
    _sync(device)
    return time.monotonic() - t


def load_base(model, device, dtype):
    """{name: tensor} of every base safetensors entry, floating ones cast to dtype."""
    from safetensors import safe_open

    out = {}
    for file in sorted(Path(model).glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as reader:
            for name in reader.keys():  # noqa: SIM118 - safetensors is not iterable
                tensor = reader.get_tensor(name)
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype)
                out[name] = tensor.to(device).contiguous()
    return out


def variant_names(buckets):
    return (
        ["per_tensor_sum64", "per_tensor_sha256"]
        + [f"bucket_{b}M" for b in buckets]
        + [f"raw_{b}M" for b in buckets]
    )


def summarize(records, tensor_bytes):
    """Per variant: timed passes (pass >= 1) of both ranks, medians and effective GB/s."""
    rows = {}
    for r in records:
        if r["pass"] == 0:
            continue
        row = rows.setdefault(r["variant"], {"source_s": [], "receiver_s": [], "details": []})
        row["source_s"].append(r["seconds"])
        row["receiver_s"].append(r["receiver_seconds"])
        if r.get("detail"):
            row["details"].append(r["detail"])
    for row in rows.values():
        slowest = [max(a, b) for a, b in zip(row["source_s"], row["receiver_s"], strict=True)]
        row["median_s"] = statistics.median(slowest)
        row["min_s"], row["max_s"] = min(slowest), max(slowest)
        row["GBps"] = tensor_bytes / row["median_s"] / 1e9
        details = row.pop("details")
        for key in ("collectives", "buckets", "chunks"):
            if details and key in details[0]:
                row[key] = details[0][key]
        for key in ("pack_s", "transfer_s", "hash_wait_s", "verify_commit_s"):
            if details and all(key in d for d in details):
                row[f"median_{key}"] = statistics.median(d[key] for d in details)
    return rows


def run(args):
    from rl.learner.runtime import configure_runtime

    configure_runtime()
    import torch
    import torch.distributed as dist

    from rl.sync.bucketed import BucketedSync, pack, plan_buckets, unpack
    from rl.sync.export import base_schema
    from rl.sync.memory_store import MemoryStore
    from rl.sync.nccl_broadcast import WeightSync
    from rl.sync.pair_group import destroy_pair_group, init_pair_group

    source = args.role == "source"
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cuda = args.device == "cuda"
    backend = "nccl" if cuda else "gloo"  # gloo/CPU: a local dry run of the same code
    if cuda:
        torch.cuda.set_device(0)
    device = torch.device("cuda", 0) if cuda else torch.device("cpu")
    rendezvous = tempfile.mktemp(prefix="sync-bench-default-")
    dist.init_process_group(backend, init_method="file://" + rendezvous, rank=0, world_size=1)
    host = "0.0.0.0" if source else args.peer
    pair, kv = init_pair_group(backend, host, args.port, 0 if source else 1, timeout=args.timeout)
    buckets = [int(b) for b in args.buckets.split(",")]
    names = variant_names(buckets)
    owners = []
    try:
        schema = base_schema(args.model, torch.bfloat16)
        base = load_base(args.model, device if source else "cpu", torch.bfloat16)
        shards = sorted(base.items()) if source else None
        plan = plan_buckets(schema, 1 << 20)
        pinned = torch.empty(plan.total_bytes, dtype=torch.uint8, pin_memory=True) if cuda else None
        syncs = {}
        for mode in ("sum64", "sha256"):
            store = MemoryStore(schema, assets=args.model, keep=not source, digest=mode)
            syncs[f"per_tensor_{mode}"] = (
                store,
                WeightSync(store, group=pair, source=0, timeout=args.timeout),
            )
        for b in buckets:
            store = MemoryStore(schema, assets=args.model, keep=not source, digest="sha256")
            sync = BucketedSync(store, group=pair, source=0, bucket_bytes=b << 20, timeout=args.timeout,
                                host=pinned)  # fmt: skip
            sync.warm()
            owners.append(sync)
            syncs[f"bucket_{b}M"] = (store, sync)
        flat = torch.zeros(plan.total_bytes, dtype=torch.uint8, device=device)
        if source:
            pack(plan, shards, flat)
        _sync(device)
        records, version = [], 0
        for p in range(args.passes + 1):
            for name in names:
                version += 1
                barrier_s = _barrier(pair, device)
                t = time.monotonic()
                record = {"pass": p, "variant": name, "version": version, "barrier_s": barrier_s}
                if name.startswith("raw_"):
                    step = int(name[4:-1]) << 20
                    calls = 0
                    for a in range(0, plan.total_bytes, step):
                        dist.broadcast(flat[a : min(a + step, plan.total_bytes)], src=0, group=pair)
                        calls += 1
                    _sync(device)
                    record |= {"seconds": time.monotonic() - t, "detail": {"collectives": calls},
                               "bytes": plan.total_bytes}  # fmt: skip
                    tensors = unpack(plan, flat) if not source and p == 0 else None
                else:
                    store, sync = syncs[name]
                    stats = sync.push(iter(shards) if source else None, version)
                    _sync(device)
                    record |= {"seconds": time.monotonic() - t, "bytes": stats.bytes, "digest": store.digest,
                               "detail": getattr(sync, "last", None)}  # fmt: skip
                    tensors = store.take(version) if not source else None
                if tensors is not None and p == 0:
                    got = {k: v.cpu() for k, v in tensors}
                    record["equal_to_base"] = set(got) == set(base) and all(
                        torch.equal(got[k], base[k]) for k in base
                    )
                del tensors
                records.append(record)
            print(json.dumps({"pass": p, "done": len(records)}), flush=True)
        info = {"host": os.uname().nodename, "env": _nccl_env(), "records": records,
                "device": torch.cuda.get_device_name(0) if cuda else "cpu"}  # fmt: skip
        if not source:
            kv.set("bench/receiver", json.dumps(info).encode())
            kv.wait(["bench/done"])
            print(
                "ROLE_RESULT " + json.dumps({"role": "receiver", "host": info["host"]}), flush=True
            )
            return 0
        peer = json.loads(kv.get("bench/receiver"))
        theirs = {r["version"]: r for r in peer["records"]}
        for r in records:
            other = theirs[r["version"]]
            r["receiver_seconds"] = other["seconds"]
            r["receiver_detail"] = other.get("detail")
            r["receiver_equal_to_base"] = other.get("equal_to_base")
            if "digest" in r:
                r["digests_agree"] = r["digest"] == other.get("digest")
        tensor_bytes = plan.tensor_bytes
        rows = summarize(records, tensor_bytes)
        result = {
            "variants": names, "passes": args.passes, "tensor_bytes": tensor_bytes,
            "packed_bytes": plan.total_bytes, "tensors": len(schema), "rows": rows,
            "digests_agree": all(r.get("digests_agree", True) for r in records),
            "warmup_equal_to_base": all(r.get("receiver_equal_to_base") for r in records if r["pass"] == 0),
            "source": {k: info[k] for k in ("host", "env", "device")},
            "receiver": {k: peer[k] for k in ("host", "env", "device")},
            "torch": torch.__version__, "backend": backend,
            "nccl": ".".join(map(str, torch.cuda.nccl.version())) if cuda else None,
            "records": records,
        }  # fmt: skip
        (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
        kv.set("bench/done", b"1")
        brief = {k: v for k, v in result.items() if k != "records"}
        print("ROLE_RESULT " + json.dumps(brief), flush=True)
        return 0 if result["digests_agree"] and result["warmup_equal_to_base"] else 1
    finally:
        for sync in owners:
            sync.close()
        destroy_pair_group(pair)
        dist.destroy_process_group()
        Path(rendezvous).unlink(missing_ok=True)


def _nccl_env():
    return {
        k: v for k, v in os.environ.items() if k.startswith("NCCL_") or k == "CUDA_VISIBLE_DEVICES"
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--role", choices=("source", "receiver"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--peer", default="127.0.0.1", help="source host (receiver role)")
    p.add_argument(
        "--buckets", default="16,64,256,1024", help="bucket sizes in MiB, comma-separated"
    )
    p.add_argument("--passes", type=int, default=3, help="timed passes after one warm-up pass")
    p.add_argument("--output", default="bench-out")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="cpu: gloo dry run")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
