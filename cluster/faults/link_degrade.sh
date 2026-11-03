#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
exec .venv/bin/python -m cluster.faults.plan --method ib_width "$@"
