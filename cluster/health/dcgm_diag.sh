#!/usr/bin/env bash
set -euo pipefail
# Does not start/stop nv-hostengine or change GPU state outside the diagnostic.
if [[ "${1:-}" != "--execute" || "$#" -ne 1 ]]; then
  printf '%s\n' '{"status":"planned-unverified","argv":["dcgmi","diag","-r","3","-j"],"note":"未在真实多机上验证"}'
  exit 0
fi
exec dcgmi diag -r 3 -j
