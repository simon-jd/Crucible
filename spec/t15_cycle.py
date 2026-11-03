"""T15 phase 7: offline EAGLE-3 training in feature batches inside one GPU job.

SpecForge's online mode cannot run here (managed-local needs disjoint capture and
trainer GPUs; external online resume needs the capture server and Mooncake kept alive
between jobs). So features are made offline, one batch at a time, because all of them
do not fit the disk share. Each batch is one epoch of a single SpecForge run:
the config has num_epochs = number of batches and total_steps = batches x batch size,
every batch has the same sample count (SpecForge's resume contract checks dataset_size),
and features always go to the same directory with the same file names. Per batch:
  1. prepare_hidden_states.py writes the batch's features,
  2. `specforge train` resumes from {run_id}-latest and stops at max_steps = (k+1) x B,
  3. spec.t15_cleanup deletes the feature directory.
The job stops before a batch that would not finish inside --budget-s.
Run under the GPU lock: bash scripts/run_t15_gpu.sh cycle ...
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from spec.t15_cleanup import delete_feature_batch

SHARE_BYTES = 40 * 1000**3  # scripts/resource_preflight.py PROJECT_SHARE_BYTES


def latest_step(train_dir, run_id):
    link = Path(train_dir) / f"{run_id}-latest"
    if not link.is_symlink():
        return None
    name = os.readlink(link).rstrip("/").rsplit("/", 1)[-1]
    return int(name.rsplit("-step", 1)[1])


def next_batch(step, batch_size, n_batches):
    if step is None:
        return 0
    if step % batch_size:
        raise ValueError(f"checkpoint step {step} is not at a batch boundary of {batch_size}")
    k = step // batch_size
    return None if k >= n_batches else k


def stage_batch(base, batch_file):
    """prepare_hidden_states caches the loaded rows (Dataset.from_generator fingerprint)
    and the processed dataset by the data *path*, not its contents. So copy the batch to a
    content-addressed path; a rewritten batch file then can never reuse old rows.
    Returns (cache_dir, data_path)."""
    data = Path(batch_file).read_bytes()
    cache_dir = Path(base) / f"batch-{hashlib.sha256(data).hexdigest()[:16]}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    staged = cache_dir / "batch.jsonl"
    if not staged.exists():
        staged.write_bytes(data)
    return cache_dir, staged


def should_start(elapsed_s, per_batch_s, budget_s):
    return elapsed_s + per_batch_s <= budget_s


def train_overrides(k, batch_size, train_dir, run_id):
    out = [f"training.max_steps={(k + 1) * batch_size}"]
    if k > 0:
        out.append(f"training.resume_from={train_dir}/{run_id}-latest")
    return out


def _load():
    uptime = subprocess.run(["uptime"], capture_output=True, text=True, check=False).stdout.strip()
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "nproc": os.cpu_count(), "uptime": uptime}


def _project_bytes():
    out = subprocess.run(["du", "-sb", "."], capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


def _count_features(path):
    return sum(1 for p in Path(path).rglob("data_*.ckpt"))


def _du(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def _run(cmd, log):
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(map(str, cmd))}\n")
        fh.flush()
        subprocess.run(list(map(str, cmd)), stdout=fh, stderr=subprocess.STDOUT, check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--draft-config", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--specforge-src", required=True)
    p.add_argument("--batches-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--n-batches", type=int, required=True)
    p.add_argument("--features", type=Path, default=Path(".cache/t15/features-full"))
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--capture-batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--budget-s", type=float, default=3.5 * 3600)
    p.add_argument("--first-batch-s", type=float, default=900)
    p.add_argument("--feature-bytes", type=float, required=True, help="expected bytes per batch")
    p.add_argument("--max-batches", type=int, default=None, help="stop after this many in this job")
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--stdout-log", type=Path, required=True)
    args = p.parse_args(argv)
    if args.batch_size % args.capture_batch_size:
        # prepare_hidden_states drops the last partial capture batch, so a batch of 200
        # rows at capture batch 16 yields 192 features and breaks the equal-batch contract.
        raise SystemExit(f"t15_cycle: --batch-size {args.batch_size} must be a multiple of the "
                         f"capture batch size {args.capture_batch_size}")

    start, per_batch, done_here = time.monotonic(), args.first_batch_s, 0
    while True:
        k = next_batch(latest_step(args.train_dir, args.run_id), args.batch_size, args.n_batches)
        if k is None:
            print(json.dumps({"event": "all batches trained"}))
            return
        elapsed = time.monotonic() - start
        if not should_start(elapsed, per_batch, args.budget_s) or done_here == args.max_batches:
            print(json.dumps({"event": "stop", "next_batch": k, "elapsed_s": elapsed}))
            return
        if args.features.exists():
            raise SystemExit(f"t15_cycle: {args.features} exists; delete it before batch {k}")
        used = _project_bytes()
        if used + args.feature_bytes > SHARE_BYTES:
            raise SystemExit(f"t15_cycle: {used} + {args.feature_bytes:.0f} bytes exceeds the share")
        row = {"batch": k, "load_before": _load(), "project_bytes_before": used}
        t0 = time.monotonic()
        cache_dir, data_path = stage_batch(".cache/t15/sf-cache", args.batches_dir / f"batch_{k:03d}.jsonl")
        _run(
            [
                ".venv-specforge/bin/torchrun", "--standalone", "--nproc_per_node=1",
                f"{args.specforge_src}/scripts/prepare_hidden_states.py",
                "--strategy", "eagle3", "--target-model-path", args.target,
                "--draft-model-config", args.draft_config,
                "--data-path", data_path, "--output-path", args.features, "--cache-dir", cache_dir,
                "--chat-template", "qwen", "--max-length", args.max_length,
                "--tp-size", "1", "--batch-size", args.capture_batch_size,
                "--num-workers", "2", "--num-io-threads", "4",
                "--sglang-mem-fraction-static", "0.6",
            ],
            args.stdout_log,
        )
        t1 = time.monotonic()
        row["features"] = _count_features(args.features)
        row["feature_bytes"] = _du(args.features)
        if row["features"] != args.batch_size:
            raise SystemExit(f"t15_cycle: batch {k} has {row['features']} features, need {args.batch_size}")
        _run(
            [".venv-specforge/bin/specforge", "train", "--config", args.config,
             *train_overrides(k, args.batch_size, args.train_dir, args.run_id)],
            args.stdout_log,
        )
        t2 = time.monotonic()
        step = latest_step(args.train_dir, args.run_id)
        if step != (k + 1) * args.batch_size:
            raise SystemExit(f"t15_cycle: after batch {k} the latest checkpoint is step {step}")
        row["deleted"] = delete_feature_batch(args.features)
        row.update(
            features_s=round(t1 - t0, 1),
            train_s=round(t2 - t1, 1),
            total_s=round(time.monotonic() - t0, 1),
            step=step,
            load_after=_load(),
        )
        with open(args.log, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        per_batch = max(per_batch, row["total_s"]) if done_here == 0 else row["total_s"] * 1.1
        done_here += 1


if __name__ == "__main__":
    main()
