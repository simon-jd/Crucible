#!/usr/bin/env bash
# EAGLE-3 draft training with SpecForge, pinned to main e3ab28d (no upstream tags).
# usage: spec/train_eagle3.sh CONFIG.yaml
# CONFIG starts from upstream examples/configs/offline/colocated/qwen3-8b-eagle3.yaml
# with the target model swapped for a small one (T15: spec/configs/qwen3-4b-eagle3-t15.yaml)
# and paths under .cache/t15/. Runs `specforge train --config CONFIG` under the shared GPU lock.
# The environment is .venv-specforge (scripts/bootstrap_specforge.sh, wheels from pypi.org).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
config="${1:?usage: spec/train_eagle3.sh CONFIG.yaml}"
commit=e3ab28d681df457f2a04e59dfac8117c2c4cbfef
env_dir=.venv-specforge
# One retained checkpoint for the T15 Qwen3-4B draft (150M trainable: bf16 weights, fp32
# master, two fp32 Adam moments, about 2.1 GB) plus the SGLang export (0.3 GB) and logs.
need_bytes=$((2800 * 1000 * 1000))
share_bytes=$((40 * 1000 * 1000 * 1000))  # project disk share (scripts/resource_preflight.py)
if [[ ! -x "$env_dir/bin/specforge" ]]; then
  echo "train_eagle3: blocked: $env_dir is not installed (run scripts/bootstrap_specforge.sh)" >&2
  exit 3
fi
if [[ "$(cat "$env_dir/SPECFORGE_COMMIT" 2>/dev/null)" != "$commit" ]]; then
  echo "train_eagle3: $env_dir is not SpecForge $commit" >&2
  exit 3
fi
used=$(du -sb . | cut -f1)
if (( used + need_bytes > share_bytes )); then
  echo "train_eagle3: blocked: project uses $used bytes; +$need_bytes exceeds the 40 GB share" >&2
  exit 3
fi
[[ -f "$config" ]] || { echo "train_eagle3: no config $config" >&2; exit 2; }
source spec/t15_env.sh
echo "load $(date -Is) nproc=$(nproc) $(uptime)"
bash scripts/with_gpu.sh -- "$env_dir/bin/specforge" train --config "$config"
echo "load $(date -Is) nproc=$(nproc) $(uptime)"
