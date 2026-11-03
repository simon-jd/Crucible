#!/usr/bin/env bash
# Offline entry only. The $0 resource budget pauses all real Task4 multi-node work.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
python_bin=${CRUCIBLE_PYTHON:-"$root/.venv/bin/python"}
case "${1:-plan}" in
  plan) exec "$python_bin" -m scaling.experiment plan ;;
  summarize) shift; exec "$python_bin" -m scaling.experiment summarize "$@" ;;
  analyze) shift; exec "$python_bin" -m scaling.analyze_nsys "$@" ;;
  *) echo 'Offline only: plan | summarize RECORDS.json | analyze TRACE.sqlite. Multi-node execution paused.' >&2; exit 2 ;;
esac
