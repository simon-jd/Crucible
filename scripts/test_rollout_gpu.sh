#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p .cache/task7/runtime/{vllm,triton,torch,torch-home,cupy,numba,tmp,cuda,xdg}
exec bash scripts/with_gpu.sh -- env -u HUGGINGFACE_HUB_CACHE -u TRANSFORMERS_CACHE \
  HF_HOME=/home/postedism/.cache/huggingface \
  HF_HUB_CACHE=/home/postedism/.cache/huggingface/hub \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CRUCIBLE_ROLLOUT_GPU=1 \
  CUDA_CACHE_PATH="$PWD/.cache/task7/runtime/cuda" \
  XDG_CACHE_HOME="$PWD/.cache/task7/runtime/xdg" \
  TORCH_HOME="$PWD/.cache/task7/runtime/torch-home" \
  VLLM_CACHE_ROOT="$PWD/.cache/task7/runtime/vllm" \
  TRITON_CACHE_DIR="$PWD/.cache/task7/runtime/triton" \
  TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/task7/runtime/torch" \
  CUPY_CACHE_DIR="$PWD/.cache/task7/runtime/cupy" \
  NUMBA_CACHE_DIR="$PWD/.cache/task7/runtime/numba" \
  TMPDIR="$PWD/.cache/task7/runtime/tmp" OMP_NUM_THREADS=2 \
  .venv-ml/bin/python -m pytest -q -s rl/rollout/tests/test_rollout_gpu.py
