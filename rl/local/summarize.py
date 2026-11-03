"""Summarize rl.local.grpo_run outputs: before/after pass@1 and per-step costs.

The pass@1 change is paired by problem: each held-out problem's c/n after minus
before, with the standard error of that per-problem mean.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def paired_delta(before, after, n):
    diffs = [(after[k] - before[k]) / n for k in before]
    mean = sum(diffs) / len(diffs)
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    return {"delta": mean, "stderr": sd / math.sqrt(len(diffs)), "improved": sum(d > 0 for d in diffs), "worse": sum(d < 0 for d in diffs)}


def pooled_delta(runs, n):
    """Pool seeds of one config: per-problem mean over seeds of (after - before) / n.

    runs: [(before_per_problem_correct, after_per_problem_correct), ...]. The SE is
    over problems of the seed-averaged difference, so it shrinks with sampling noise
    but not below the problem-set spread.
    """
    keys = set(runs[0][0])
    if any(set(b) != keys or set(a) != keys for b, a in runs):
        raise ValueError("runs cover different problem sets")
    diffs = [sum(a[k] - b[k] for b, a in runs) / (n * len(runs)) for k in sorted(keys)]
    mean = sum(diffs) / len(diffs)
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    per_seed = [sum(a[k] - b[k] for k in keys) / (n * len(keys)) for b, a in runs]
    return {"delta": mean, "stderr": sd / math.sqrt(len(diffs)), "seeds": len(runs), "per_seed": per_seed}


def _range(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def step_stats(steps):
    out = {k: _range([s[k] for s in steps]) for k in ("wall_s", "rollout_s", "learn_s", "sync_s", "gpu_idle_share")}
    out["dropped_stale"] = sum(s["dropped_stale"] for s in steps)
    out["reward_first10"] = statistics.mean(s["reward"] for s in steps[:10])
    out["reward_last10"] = statistics.mean(s["reward"] for s in steps[-10:])
    out["truncated_first10"] = statistics.mean(s["flags"].get("truncated", 0) for s in steps[:10])
    out["truncated_last10"] = statistics.mean(s["flags"].get("truncated", 0) for s in steps[-10:])
    out["steps"] = len(steps)
    return out


def summarize(run_dir):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines()]
    n = int(summary["args"]["eval_n"])
    evals = {}
    for name in summary["before"]:
        b, a = summary["before"][name], summary["after"][name]
        evals[name] = {"before": b["pass_at_1"], "after": a["pass_at_1"], "problems": b["problems"], "samples_per_problem": n,
                       "truncated": {"before": b["truncated"], "after": a["truncated"]},
                       **paired_delta(b["per_problem_correct"], a["per_problem_correct"], n)}  # fmt: skip
    return {
        "run": run_dir.name,
        "staleness": int(summary["args"]["staleness"]),
        "kl": float(summary["args"]["kl"]),
        **{k: summary["args"].get(k) for k in ("group", "clip", "seed", "reward", "prompts_per_step")},
        "eval": evals,
        "steps": step_stats(steps),
        "totals": summary["totals"],
        "peak_cuda_allocated_bytes": summary["peak_cuda_allocated_bytes"],
        "wall_s": summary["wall_s"],
    }


CONFIG_KEYS = ("staleness", "kl", "group", "clip", "reward", "max_new_tokens")


def pool_runs(run_dirs):
    """Group runs that differ only in seed and pool each group with pooled_delta."""
    groups = {}
    for run_dir in run_dirs:
        summary = json.loads((Path(run_dir) / "summary.json").read_text())
        a = summary["args"]
        key = tuple((k, a.get(k, "none" if k == "reward" else None)) for k in CONFIG_KEYS)
        groups.setdefault(key, []).append((Path(run_dir).name, summary))
    out = []
    for key, runs in groups.items():
        n = int(runs[0][1]["args"]["eval_n"])
        evals = {}
        for name in runs[0][1]["before"]:
            pairs = [(s["before"][name]["per_problem_correct"], s["after"][name]["per_problem_correct"]) for _, s in runs]
            evals[name] = {"before": runs[0][1]["before"][name]["pass_at_1"],
                           "after_mean": statistics.mean(s["after"][name]["pass_at_1"] for _, s in runs),
                           "truncated_after": [s["after"][name]["truncated"] for _, s in runs],
                           **pooled_delta(pairs, n)}  # fmt: skip
        out.append({"config": dict(key), "runs": [r for r, _ in runs], "eval": evals})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--pool", action="store_true", help="pool seeds of identical configs")
    args = p.parse_args(argv)
    print(json.dumps(pool_runs(args.runs) if args.pool else [summarize(r) for r in args.runs], indent=1))


if __name__ == "__main__":
    main()
