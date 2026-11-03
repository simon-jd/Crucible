#!/usr/bin/env bash
# 安装仅在本项目内的开发依赖；不会配置节点或启动云资源。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
export GOCACHE="$PWD/.cache/go-build"
export GOPATH="$PWD/.cache/go"
export GOENV=off
export GOTOOLCHAIN=local
command -v uv >/dev/null || { echo '请先提供 uv 命令' >&2; exit 1; }
uv sync --locked
python3 scripts/install_go.py
"$PWD/.tools/go/bin/go" version
