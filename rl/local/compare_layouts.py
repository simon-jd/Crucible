"""Compare GRPO layouts run with the same seeds: accuracy (paired, pooled) and pipeline costs.

  python -m rl.local.compare_layouts --layout a=DIR_s17,DIR_s18,DIR_s19 --layout b=... [--out FILE]

Each DIR holds a grpo_run-format summary.json and steps.jsonl (time-shared runs from
rl.local.grpo_run, two-GPU runs merged by rl.local.async_grpo). Per layout: the pooled
before -> after change (rl.local.summarize.pooled_delta, the 2 SE bar) and per-step costs.
Between layouts: per problem, the seed-averaged after(x) - after(y), with its SE over
problems; seeds are paired by position, so list them in the same order.
"""

import argparse
import json
import statistics
from pathlib import Path

from rl.local.summarize import pooled_delta


def _load(run):
    run = Path(run)
    summary = json.loads((run / "summary.json").read_text())
    steps = [json.loads(x) for x in (run / "steps.jsonl").read_text().splitlines()]
    return summary, steps


def _med(values):
    return statistics.median(values) if values else None


def step_costs(summary, steps):
    """Median per-step costs; for time-shared runs the pipeline block is derived from steps."""
    out = {
        k: _med([s[k] for s in steps])
        for k in ("wall_s", "rollout_s", "learn_s", "sync_s", "reward_s")
    }
    trained = sum(s["accepted_samples"] for s in steps)
    ages = {}
    for s in steps:
        age = s["policy_version"] - 1 - s["trained_version"]
        ages[str(age)] = ages.get(str(age), 0) + s["accepted_samples"] + s["dropped_stale"]
    if "pipeline" in summary:
        p = summary["pipeline"]
        out |= {
            "training_s": p["training_window_s"],
            "trained_samples_per_s": p["trained_samples_per_s"],
            "learner_gpu_idle_share": p["learner_gpu_idle_share"],
            "rollout_gpu_idle_share": p["rollout_gpu_idle_share"],
            "learn_generate_overlap_s": p["learn_generate_overlap_s"],
            "sync_s_learner_push": p["learner_push_s"]["median"],
            "sync_s_rollout_recv_plus_load": p["rollout_recv_s"]["median"]
            + p["rollout_load_s"]["median"],
            "export_s": p["export_s"]["median"],
            "staleness_histogram": p["staleness_histogram"],
        }
    else:
        # One GPU: idle = time outside rollout, learn and sync (grpo_run's definition).
        # Training time is the sum of step walls; the single warm-up generation is not logged.
        total = sum(s["wall_s"] for s in steps)
        busy = sum(s["rollout_s"] + s["learn_s"] + s["sync_s"] for s in steps)
        out |= {
            "training_s": total,
            "trained_samples_per_s": trained / total,
            "gpu_idle_share": 1 - busy / total,
            "staleness_histogram": dict(sorted(ages.items())),
        }
    out["dropped_stale"] = sum(s["dropped_stale"] for s in steps)
    out["trained_samples"] = trained
    out["peak_cuda_allocated_bytes"] = summary.get("peak_cuda_allocated_bytes")
    return out


def compare(layouts):
    """layouts: {name: [run_dir, ...]} with seeds in the same order for every layout."""
    loaded = {name: [_load(r) for r in runs] for name, runs in layouts.items()}
    report = {"layouts": {}, "pairs": {}}
    for name, runs in loaded.items():
        n = int(runs[0][0]["args"]["eval_n"])
        evals = {}
        for set_name in runs[0][0]["after"]:
            pairs = [(s["before"][set_name]["per_problem_correct"], s["after"][set_name]["per_problem_correct"])
                     for s, _ in runs]  # fmt: skip
            d = pooled_delta(pairs, n)
            evals[set_name] = {
                "before": [s["before"][set_name]["pass_at_1"] for s, _ in runs],
                "after": [s["after"][set_name]["pass_at_1"] for s, _ in runs],
                "truncated_after": [s["after"][set_name]["truncated"] for s, _ in runs],
                **d, "se_multiple": d["delta"] / d["stderr"] if d["stderr"] else None,
                "clears_2se": d["delta"] >= 2 * d["stderr"],
            }  # fmt: skip
        costs = [step_costs(s, st) for s, st in runs]
        report["layouts"][name] = {
            "runs": [str(r) for r in layouts[name]],
            "seeds": [s["args"]["seed"] for s, _ in runs],
            "eval": evals,
            "per_seed_costs": costs,
        }
    names = list(loaded)
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            rx, ry = loaded[x], loaded[y]
            if [s["args"]["seed"] for s, _ in rx] != [s["args"]["seed"] for s, _ in ry]:
                raise ValueError(f"layouts {x} and {y} have different seed lists")
            n = int(rx[0][0]["args"]["eval_n"])
            report["pairs"][f"{y}-{x}"] = {
                set_name: pooled_delta(
                    [
                        (
                            sx["after"][set_name]["per_problem_correct"],
                            sy["after"][set_name]["per_problem_correct"],
                        )
                        for (sx, _), (sy, _) in zip(rx, ry, strict=True)
                    ],
                    n,
                )
                for set_name in rx[0][0]["after"]
            }
    return report


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--layout", action="append", required=True, help="NAME=DIR,DIR,DIR")
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    layouts = {}
    for item in a.layout:
        name, dirs = item.split("=", 1)
        layouts[name] = dirs.split(",")
    report = compare(layouts)
    text = json.dumps(report, indent=1)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
