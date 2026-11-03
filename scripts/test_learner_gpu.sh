#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# 0 downloads/checkpoints/resident services; reserve compiler/cache/evidence growth.
env -u HUGGINGFACE_HUB_CACHE -u TRANSFORMERS_CACHE \
  HF_HOME=/home/postedism/.cache/huggingface HF_HUB_CACHE=/home/postedism/.cache/huggingface/hub \
  .venv/bin/python scripts/resource_preflight.py --planned-project-bytes 100000000 \
  --download-bytes 0 --checkpoints 0 --service-memory-bytes 0
# The stdlib entry point configures all paths before importing pytest or Torch.
exec bash scripts/with_gpu.sh -- env CRUCIBLE_LEARNER_GPU=1 OMP_NUM_THREADS=2 \
  .venv-ml/bin/python -c 'from rl.learner.runtime import configure_runtime; configure_runtime(); import pytest; raise SystemExit(pytest.main(["-q", "-s", "rl/learner/tests/test_train_gpu.py"]))'
