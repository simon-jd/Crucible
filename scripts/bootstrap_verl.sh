#!/usr/bin/env bash
# Separate verl environment (.venv-verl) for the T11 comparison: verl 0.9.1 with its vLLM
# rollout extra and math extra, wheels from pypi.org / files.pythonhosted.org only.
# A dry run (2026-11-01) resolved 257 packages: torch 2.11.0, vllm 0.24.0,
# transformers 5.9.0, numpy 2.3.5, ray 2.59.0. Interpreter: the project's .tools/python
# CPython 3.12.13. No host installation, no GPU.
# usage: scripts/bootstrap_verl.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python=.tools/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12
# The environment is about 10 GB (like .venv-ml and .venv-specforge); downloads about 5 GB.
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
.venv/bin/python scripts/resource_preflight.py --planned-project-bytes 12000000000 \
  --download-bytes 5000000000 --checkpoints 0 --service-memory-bytes 0

export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_NO_CACHE=1
export UV_LINK_MODE=copy
export TMPDIR="$PWD/.cache/t11/tmp"
export UV_INDEX_URL=https://pypi.org/simple
mkdir -p "$TMPDIR"
[[ -x .venv-verl/bin/python ]] || uv venv .venv-verl --python "$python"
uv pip install --python .venv-verl/bin/python --index-strategy first-index "verl[vllm,math]==0.9.1"
.venv-verl/bin/python -c 'import torch, vllm, transformers, ray, verl; print("torch=" + torch.__version__, "cuda=" + str(torch.version.cuda), "vllm=" + vllm.__version__, "transformers=" + transformers.__version__, "ray=" + ray.__version__, "verl=" + verl.__version__)'
du -sb .venv-verl
