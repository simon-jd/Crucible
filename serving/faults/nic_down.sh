#!/usr/bin/env bash
# Multi-node fault: take one node's NIC down for a fixed time, then bring it back.
# usage: CRUCIBLE_MULTINODE_APPROVED=1 serving/faults/nic_down.sh NODE IFACE SECONDS
# PAUSED: this needs the rented multi-node cluster (outside Crucible's $0 budget) and
# root on that node; it is not run on the single-4090 host. Never run it against a
# node you do not own: it cuts the node's traffic on IFACE.
set -euo pipefail
if [[ "${CRUCIBLE_MULTINODE_APPROVED:-}" != 1 ]]; then
  echo "nic_down: paused until a multi-node GPU budget is approved (未在真实多机上验证)" >&2
  exit 3
fi
node="${1:?NODE}" iface="${2:?IFACE}" seconds="${3:?SECONDS}"
if ! [[ "$seconds" =~ ^[0-9]+$ ]] || (( seconds < 1 || seconds > 300 )); then
  echo "nic_down: SECONDS must be 1..300" >&2
  exit 2
fi
# The link comes back even if this script is interrupted after the down step.
ssh "$node" "sudo ip link set dev $iface down; sleep $seconds; sudo ip link set dev $iface up" &
echo "down $(date +%s.%N) node=$node iface=$iface seconds=$seconds"
wait
echo "up $(date +%s.%N)"
