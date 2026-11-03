#!/usr/bin/env bash
# Separate SpecForge environment (.venv-specforge) for T15, pinned to SpecForge main e3ab28d.
# Source: api.github.com tarball (extracted under .cache/t15/src); wheels: pypi.org and
# files.pythonhosted.org only (user decision 2026-10-29). Interpreter: the project's
# .tools/python CPython 3.12.13. No host installation, no GPU.
# usage: scripts/bootstrap_specforge.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
commit=e3ab28d681df457f2a04e59dfac8117c2c4cbfef
src=.cache/t15/src/x/sgl-project-SpecForge-e3ab28d
python=.tools/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12
[[ -f "$src/pyproject.toml" ]] || {
  echo "bootstrap_specforge: missing $src (fetch api.github.com/repos/sgl-project/SpecForge/tarball/$commit)" >&2
  exit 2
}
unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export HF_HOME="$HOME/.cache/huggingface" HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
# The environment is about 10 GB (estimate plus margin); downloads about 5 GB.
.venv/bin/python scripts/resource_preflight.py --planned-project-bytes 12000000000 \
  --download-bytes 5000000000 --checkpoints 0 --service-memory-bytes 0

export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_NO_CACHE=1
export UV_LINK_MODE=copy
export TMPDIR="$PWD/.cache/t15/tmp"
export UV_INDEX_URL=https://pypi.org/simple
mkdir -p "$TMPDIR"
[[ -x .venv-specforge/bin/python ]] || uv venv .venv-specforge --python "$python"
uv pip install --python .venv-specforge/bin/python --index-strategy first-index "$src"
echo "$commit" > .venv-specforge/SPECFORGE_COMMIT
.venv-specforge/bin/python -c 'import torch, sglang, transformers, specforge; print("torch=" + torch.__version__, "cuda=" + str(torch.version.cuda), "sglang=" + sglang.__version__, "transformers=" + transformers.__version__)'
du -sb .venv-specforge
