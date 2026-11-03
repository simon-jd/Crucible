#!/usr/bin/env bash
# T15 EAGLE-3 entry point. Every stage is one GPU job under the shared lock.
# usage: scripts/run_t15_gpu.sh regen [IN OUT] | features N | train | cycle ARGS | bench OUT_DIR
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
stage="${1:?usage: scripts/run_t15_gpu.sh regen|features N|train|cycle ARGS|bench OUT_DIR}"
if [[ $stage == train ]]; then
  exec bash spec/train_eagle3.sh spec/configs/qwen3-4b-eagle3-t15.yaml  # takes the lock itself
fi
exec bash scripts/with_gpu.sh -- bash spec/t15_gpu.sh "$@"
