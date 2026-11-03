# Sourced by spec/train_eagle3.sh and spec/t15_gpu.sh: shared HF cache, offline hub,
# and every compiler/JIT cache under .cache/t15 (never the user's ~/.cache).
unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_CACHE="$HOME/.cache/huggingface/hub" HF_HUB_OFFLINE=1
export TMPDIR="$PWD/.cache/t15/tmp" XDG_CACHE_HOME="$PWD/.cache/t15/xdg"
export TRITON_CACHE_DIR="$PWD/.cache/t15/xdg/triton" TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/t15/xdg/inductor"
export FLASHINFER_WORKSPACE_BASE="$PWD/.cache/t15/xdg" CUDA_CACHE_PATH="$PWD/.cache/t15/xdg/nv"
export WANDB_MODE=disabled PYTHONNOUSERSITE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
