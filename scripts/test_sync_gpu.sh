#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode=fresh
planned=2100000000
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 1 || "$1" != --reuse ]]; then
    echo 'usage: bash scripts/test_sync_gpu.sh [--reuse]' >&2
    exit 2
  fi
  mode=reuse
  planned=50000000
fi
# Fresh: at most two BF16 snapshots + assets/cache/evidence, no third save copy.
# Reuse: exact owned manifests + source-exit/protocol binding are checked inside guard.
env -u HUGGINGFACE_HUB_CACHE -u TRANSFORMERS_CACHE \
  HF_HOME=/home/postedism/.cache/huggingface HF_HUB_CACHE=/home/postedism/.cache/huggingface/hub \
  .venv/bin/python scripts/resource_preflight.py --planned-project-bytes "$planned" \
  --download-bytes 0 --checkpoints 2 --service-memory-bytes 0
exec bash scripts/with_gpu.sh -- env CRUCIBLE_SYNC_GPU=1 CRUCIBLE_SYNC_MODE="$mode" OMP_NUM_THREADS=2 \
  .venv-ml/bin/python -c 'from rl.learner.runtime import configure_runtime; configure_runtime(); import pytest; raise SystemExit(pytest.main(["-q", "-s", "rl/sync/tests/test_sync_gpu.py::test_fresh_or_explicit_reuse_acceptance"]))'
