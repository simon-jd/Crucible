#!/usr/bin/env bash
# Task 12 single-GPU baseline for one trace, under the shared GPU flock.
# usage: scripts/run_serving_baseline_gpu.sh azure|burst OUTPUT_DIR
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "${1:-}" in
  azure) trace=serving/baseline/traces/azure-60s format=azure base=0.25 ;;
  burst) trace=serving/baseline/traces/burst-300s format=burstgpt base=1.0 ;;
  *) echo "usage: $0 azure|burst OUTPUT_DIR" >&2; exit 2 ;;
esac
out="${2:?output directory}"
rev=4da05a8edb55c6046cce958586c33b61da07bb79
rt="$PWD/.cache/t12/runtime"
mkdir -p "$rt"/{vllm,triton,torch,torch-home,xdg,cuda,tmp}
exec bash scripts/with_gpu.sh -- env -u HUGGINGFACE_HUB_CACHE -u TRANSFORMERS_CACHE \
  HF_HOME=/home/postedism/.cache/huggingface \
  HF_HUB_CACHE=/home/postedism/.cache/huggingface/hub \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  CUDA_VISIBLE_DEVICES=0 \
  CUDA_CACHE_PATH="$rt/cuda" XDG_CACHE_HOME="$rt/xdg" TORCH_HOME="$rt/torch-home" \
  VLLM_CACHE_ROOT="$rt/vllm" TRITON_CACHE_DIR="$rt/triton" \
  TORCHINDUCTOR_CACHE_DIR="$rt/torch" TMPDIR="$rt/tmp" OMP_NUM_THREADS=4 \
  .venv-ml/bin/python -m serving.baseline.vllm_baseline \
  --trace-dir "$trace" --format "$format" --base-mult "$base" \
  --snapshot "/home/postedism/.cache/huggingface/hub/models--Qwen--Qwen3-8B-AWQ/snapshots/$rev" \
  --model-name Qwen/Qwen3-8B-AWQ --token-ids .cache/t12/qwen3-8b-awq-ids.json \
  --replay-bin .cache/bin/replay --cache-root "$rt" --output "$out"
