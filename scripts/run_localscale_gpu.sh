#!/usr/bin/env bash
# Task 14 local router/autoscaler and cold-start experiment, under the shared GPU flock.
# usage: scripts/run_localscale_gpu.sh scale|cold OUTPUT_DIR [extra serving.localscale.run args]
# Build the binaries first: make t14-bins
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:?scale|cold}"
out="${2:?output directory}"
shift 2
rev=7ae557604adf67be50417f59c2c2f167def9a775
rt="$PWD/.cache/t14/runtime"
mkdir -p "$rt"/{vllm,triton,torch,torch-home,xdg,cuda,tmp}
exec bash scripts/with_gpu.sh -- env -u HUGGINGFACE_HUB_CACHE -u TRANSFORMERS_CACHE \
  HF_HOME=/home/postedism/.cache/huggingface \
  HF_HUB_CACHE=/home/postedism/.cache/huggingface/hub \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
  CUDA_VISIBLE_DEVICES=0 \
  CUDA_CACHE_PATH="$rt/cuda" XDG_CACHE_HOME="$rt/xdg" TORCH_HOME="$rt/torch-home" \
  VLLM_CACHE_ROOT="$rt/vllm" TRITON_CACHE_DIR="$rt/triton" \
  TORCHINDUCTOR_CACHE_DIR="$rt/torch" TMPDIR="$rt/tmp" OMP_NUM_THREADS=4 \
  .venv-ml/bin/python -m serving.localscale.run --mode "$mode" \
  --snapshot "/home/postedism/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/$rev" \
  --revision "$rev" --cache-root "$rt" --output "$out" "$@"
