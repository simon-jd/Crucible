#!/usr/bin/env bash
# Single-node decode fault: SIGKILL one local vLLM server's whole process group.
# usage: serving/faults/kill_decode.sh PGID
# Refuses anything whose group leader is not a vLLM OpenAI server, so it cannot hit
# another project's process. Prints the kill time (epoch seconds) on success.
# The multi-node version (kill a decode Pod) is paused with the rest of Task 13 P/D.
set -euo pipefail
pgid="${1:?usage: kill_decode.sh PGID}"
if ! [[ "$pgid" =~ ^[0-9]+$ ]] || (( pgid <= 1 )); then
  echo "kill_decode: invalid PGID $pgid" >&2
  exit 2
fi
cmdline="/proc/$pgid/cmdline"
if [[ ! -r "$cmdline" ]] || ! tr '\0' ' ' < "$cmdline" | grep -q "vllm.entrypoints.openai.api_server"; then
  echo "kill_decode: $pgid is not a vLLM OpenAI server group leader; refusing" >&2
  exit 3
fi
if [[ "$(ps -o pgid= -p "$pgid" | tr -d ' ')" != "$pgid" ]]; then
  echo "kill_decode: $pgid is not a process-group leader; refusing" >&2
  exit 3
fi
kill -KILL -- "-$pgid"
date +%s.%N
