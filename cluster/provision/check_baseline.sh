#!/usr/bin/env bash
set -euo pipefail
script_dir="${BASH_SOURCE[0]%/*}"
exec "${BASELINE_PYTHON:-python3}" "$script_dir/baseline.py" "$@"
