#!/usr/bin/env bash
# Reruns TestReplayTimingAccuracy under CPU contention: the test binary and
# HOGS busy loops are pinned to one core (taskset), so the load stays on that
# core instead of spreading over the shared machine.
# Usage: bash scripts/replay_contention_repro.sh <out-dir> [count] [hogs] [cores] [gomaxprocs]
# cores is a taskset list (e.g. 30,31); hogs are spread over it.
set -euo pipefail
out=${1:?out dir}
count=${2:-30}
hogs=${3:-3}
cores=${4:-31}
cpu=${5:-1}
root=$(cd "$(dirname "$0")/.." && pwd)
export GOCACHE=$root/.cache/go-build GOPATH=$root/.cache/go GOENV=off GOTOOLCHAIN=local
mkdir -p "$out" "$root/.cache/replay-repro"
bin=$root/.cache/replay-repro/replay.test
"$root/.tools/go/bin/go" test -c -o "$bin" ./serving/replay
git -C "$root" rev-parse HEAD >"$out/commit.txt"
git -C "$root" diff --stat -- serving/replay >>"$out/commit.txt"
{ echo "nproc $(nproc)"; echo "start $(uptime)"; echo "count=$count hogs=$hogs cores=$cores gomaxprocs=$cpu"; } >"$out/machine-load.txt"
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
for _ in $(seq "$hogs"); do
  taskset -c "$cores" bash -c 'while :; do :; done' &
  pids+=($!)
done
set +e
(cd "$root/serving/replay" && taskset -c "$cores" "$bin" -test.run '^TestReplayTimingAccuracy$' \
  -test.cpu "$cpu" -test.count "$count" -test.v) >"$out/go-test.txt" 2>&1
status=$?
set -e
echo "end $(uptime)" >>"$out/machine-load.txt"
pass=$(grep -c '^--- PASS' "$out/go-test.txt" || true)
fail=$(grep -c '^--- FAIL' "$out/go-test.txt" || true)
missing=$(grep -c 'missing emission timestamp' "$out/go-test.txt" || true)
echo "exit=$status pass=$pass fail=$fail missing_emission_lines=$missing" | tee "$out/summary.txt"
