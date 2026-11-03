#!/usr/bin/env bash
# Phase-5 T11 batch: same training as phase 4 (--reward eos, 100 steps, 6,400 samples,
# identical 1,024 training prompts), evaluated on the full held-out pools: all 1,319
# GSM8K test problems and all 377 gradable MATH-500 problems (phase 4: 200 / 100).
# Fixed before the runs: the primary result is the async k=1 change pooled over seeds
# 17/18/19 (rl/local/summarize.py --pool); "significant" means pooled delta >= 2 SE,
# the same rule as phase 4. Group 16 gets the same 3 seeds as a secondary comparison.
# Each run takes the GPU lock separately (scripts/with_gpu.sh) and releases it.
# usage: scripts/run_grpo_phase5.sh [run-name-glob]   (default: all runs)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
out=docs/evidence/rl-grpo-p5
data=.cache/t11/data-full.json
only="${1:-*}"
mkdir -p "$out" .cache/t11/phase5/logs
run() {
  name="$1"; shift
  # shellcheck disable=SC2053
  [[ "$name" == $only ]] || return 0
  if [ -f "$out/$name/summary.json" ]; then return 0; fi
  { echo "== $name $(date -Is) nproc=$(nproc) $(uptime)"; } >> "$out/machine-load.txt"
  bash scripts/run_grpo_gpu.sh "$out/$name" --data "$data" --reward eos "$@" > ".cache/t11/phase5/logs/$name.log" 2>&1
  echo "== $name exit=$? $(date -Is) $(uptime)" >> "$out/machine-load.txt"
}
for seed in 17 18 19; do
  run "full-async-k1-s$seed" --staleness 1 --seed "$seed"
done
for seed in 17 18 19; do
  run "full-async-k1-g16-s$seed" --staleness 1 --group 16 --prompts-per-step 4 --seed "$seed"
done
