#!/usr/bin/env bash
# Phase-4 T11 batch: each run takes the GPU lock separately (scripts/with_gpu.sh) and releases it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
out=docs/evidence/rl-grpo-p4
mkdir -p "$out" .cache/t11/phase4/logs
run() {
  name="$1"; shift
  if [ -f "$out/$name/summary.json" ]; then return 0; fi
  { echo "== $name $(date -Is) nproc=$(nproc) $(uptime)"; } >> "$out/machine-load.txt"
  bash scripts/run_grpo_gpu.sh "$out/$name" --reward eos "$@" > ".cache/t11/phase4/logs/$name.log" 2>&1
  echo "== $name exit=$? $(date -Is) $(uptime)" >> "$out/machine-load.txt"
}
for seed in 17 18 19; do
  run "eos-async-k1-s$seed" --staleness 1 --seed "$seed"
  run "eos-sync-k0-s$seed" --staleness 0 --seed "$seed"
done
run eos-async-k1-g4 --staleness 1 --group 4 --prompts-per-step 16
run eos-async-k1-g16 --staleness 1 --group 16 --prompts-per-step 4
run eos-async-k1-clip0.1 --staleness 1 --clip 0.1
run eos-async-k1-clip0.3 --staleness 1 --clip 0.3
echo "== done $(date -Is)" >> "$out/machine-load.txt"
