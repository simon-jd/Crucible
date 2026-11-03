#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# All compiler/temp/Git/evidence growth, zero downloads/checkpoints/services.
env -u HUGGINGFACE_HUB_CACHE -u TRANSFORMERS_CACHE \
  HF_HOME=/home/postedism/.cache/huggingface HF_HUB_CACHE=/home/postedism/.cache/huggingface/hub \
  .venv/bin/python scripts/resource_preflight.py --planned-project-bytes 1000000000 \
  --download-bytes 0 --checkpoints 0 --service-memory-bytes 0
exec bash scripts/with_gpu.sh -- env CRUCIBLE_LOGPROB_GPU=1 OMP_NUM_THREADS=2 \
  bash -euc '
    .venv-ml/bin/python -c '\''
from kernels.benchmark_learner import gpu_preflight
gpu_preflight()
import os
import pytest
selection = "cuda or peak_memory"
if os.environ.get("CRUCIBLE_LOGPROB_FOCUSED") == "1":
    selection = "cuda and not compile and not registration"
raise SystemExit(pytest.main(["-q", "-s", "kernels/tests/test_ops.py", "-k", selection]))
'\''
    if [[ "${CRUCIBLE_LOGPROB_FOCUSED:-0}" != "1" ]]; then
      .venv-ml/bin/python -m kernels.benchmark_learner
    fi
  '
