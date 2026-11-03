#!/usr/bin/env bash
# Pin the project interpreter inside .tools/python and point existing venvs at it,
# so tests never depend on the host uv install (whose 3.12 alias moves on upgrade).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
version=$(tr -d '[:space:]' < .python-version)
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
export UV_PYTHON_INSTALL_BIN=0  # never write shims into ~/.local/bin
command -v uv >/dev/null || { echo 'uv is required' >&2; exit 1; }
uv python install "$version"
python=$(echo "$UV_PYTHON_INSTALL_DIR"/cpython-"$version"-linux-x86_64-gnu/bin/python3.12)
[[ -x "$python" ]] || { echo "pinned interpreter missing: $python" >&2; exit 1; }
home=$(dirname "$python")
for venv in .venv .venv-ml; do
  [[ -f "$venv/pyvenv.cfg" ]] || continue
  sed -i -e "s|^home = .*|home = $home|" -e "s|^version_info = .*|version_info = $version|" \
    "$venv/pyvenv.cfg"
  ln -sfn "$python" "$venv/bin/python"
  echo "$venv -> $python"
done
