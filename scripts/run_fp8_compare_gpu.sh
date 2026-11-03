#!/usr/bin/env bash
# Task 13 single-GPU BF16 vs FP8 session under the shared GPU flock.
# usage: scripts/run_fp8_compare_gpu.sh OUTPUT_DIR [fp8_session flags...]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
out="${1:?output directory}"
shift
rev=b968826d9c46dd6066d109eabc6255188de91218
rt="$PWD/.cache/t13/runtime"
mkdir -p "$rt"/{vllm,triton,torch,torch-home,xdg,cuda,tmp}
exec bash scripts/with_gpu.sh -- env -u HUGGINGFACE_HUB_CACHE -u TRANSFORMERS_CACHE \
  HF_HOME=/home/postedism/.cache/huggingface \
  HF_HUB_CACHE=/home/postedism/.cache/huggingface/hub \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  CUDA_VISIBLE_DEVICES=0 \
  CUDA_CACHE_PATH="$rt/cuda" XDG_CACHE_HOME="$rt/xdg" TORCH_HOME="$rt/torch-home" \
  VLLM_CACHE_ROOT="$rt/vllm" TRITON_CACHE_DIR="$rt/triton" \
  TORCHINDUCTOR_CACHE_DIR="$rt/torch" TMPDIR="$rt/tmp" OMP_NUM_THREADS=4 \
  .venv-ml/bin/python -m serving.eval.fp8_session \
  --snapshot "/home/postedism/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/$rev" \
  --model-name Qwen/Qwen3-8B --token-ids .cache/t13/qwen3-8b-ids.json \
  --replay-bin .cache/bin/replay --heldout .cache/t13/heldout.json \
  --cache-root "$rt" --output "$out" "$@"
