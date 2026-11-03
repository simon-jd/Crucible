#!/usr/bin/env bash
# Project-local Torch environment. No model download, GPU task, or host installation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Count conservatively: reserve growth to a 10 decimal GB environment and allow
# 6 GB downloads. Existing environment files must not be counted twice on reapply.
env -u HUGGINGFACE_HUB_CACHE -u TRANSFORMERS_CACHE \
  HF_HOME="$HOME/.cache/huggingface" HF_HUB_CACHE="$HOME/.cache/huggingface/hub" \
  .venv/bin/python - <<'PY'
import subprocess
from pathlib import Path
from scripts.resource_preflight import project_bytes

environment = Path('.venv-ml')
present = project_bytes(environment) if environment.exists() else 0
growth = max(0, 10_000_000_000 - present)
subprocess.run([
    '.venv/bin/python', 'scripts/resource_preflight.py',
    '--planned-project-bytes', str(growth), '--download-bytes', '6000000000',
    '--checkpoints', '0', '--service-memory-bytes', '0',
], check=True)
PY

export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-ml"
export UV_NO_CACHE=1
export UV_LINK_MODE=copy
uv sync --locked --extra ml
.venv-ml/bin/python -c 'import torch, triton; print("torch=" + torch.__version__, "build_cuda=" + str(torch.version.cuda), "triton=" + triton.__version__)'
