#!/usr/bin/env bash
# Task 11 single-GPU math GRPO run under the shared GPU flock.
# usage: scripts/run_grpo_gpu.sh OUTPUT_DIR [rl.local.grpo_run flags...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
out="${1:?output directory}"
shift
model=/home/postedism/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
exec bash scripts/with_gpu.sh -- env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  .venv-ml/bin/python -m rl.local.grpo_run --data .cache/t11/data.json --model "$model" \
  --output "$out" "$@"
