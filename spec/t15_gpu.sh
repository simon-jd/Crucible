#!/usr/bin/env bash
# T15 GPU stages for the Qwen3-4B EAGLE-3 draft (SpecForge e3ab28d, .venv-specforge).
# Run each stage as its own GPU job: bash scripts/with_gpu.sh -- bash spec/t15_gpu.sh STAGE
#   regen     SGLang serves the target; SpecForge regenerate_train_data.py writes the
#             target's own greedy, non-thinking answers for the training prompts.
#   features  SpecForge prepare_hidden_states.py (EAGLE-3 offline features), N samples.
#   cycle     phase 7 full-size draft: spec/t15_cycle.py (features per batch, resumed
#             specforge train, delete the batch's features), within one job's budget.
#   (training is spec/train_eagle3.sh spec/configs/qwen3-4b-eagle3-t15.yaml; it takes the lock)
#   bench    baseline vs EAGLE-3 servers, ABAB rounds, batch 1/4/16/64, same prompts.
# SGLang uses --mem-fraction-static 0.75 (the project rule is 0.80 or lower).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
stage="${1:?usage: spec/t15_gpu.sh regen|features N|bench OUT_DIR}"
py=.venv-specforge/bin/python
src=.cache/t15/src/x/sgl-project-SpecForge-e3ab28d
snap=/home/postedism/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
draft=.cache/t15/draft-sglang
port=30000
source spec/t15_env.sh

server_pid=
start_server() {  # start_server LOG [extra sglang args...]
  local log=$1; shift
  "$py" -m sglang.launch_server --model-path "$snap" --host 127.0.0.1 --port "$port" \
    --mem-fraction-static 0.75 --context-length 2048 --cuda-graph-max-bs 64 \
    --max-running-requests 64 --log-level warning "$@" > "$log" 2>&1 &
  server_pid=$!
  for _ in $(seq 600); do
    if curl -fsS "http://127.0.0.1:$port/health" > /dev/null 2>&1; then return 0; fi
    kill -0 "$server_pid" 2> /dev/null || { echo "t15: server exited; see $log" >&2; return 1; }
    sleep 1
  done
  echo "t15: server not healthy after 600 s" >&2
  return 1
}
stop_server() {
  [[ -n $server_pid ]] || return 0
  kill "$server_pid" 2> /dev/null || true
  wait "$server_pid" 2> /dev/null || true
  server_pid=
}
trap stop_server EXIT
load() { echo "load $(date -Is) nproc=$(nproc) $(uptime)"; }

case $stage in
  regen)  # regen [IN OUT]; phase 6 used the defaults
    in="${2:-.cache/t15/data/train_prompts.jsonl}"
    out="${3:-.cache/t15/data/train_regen.jsonl}"
    [[ ! -e $out ]] || { echo "t15: $out exists" >&2; exit 2; }
    load
    start_server "${out%.jsonl}-server.log"
    "$py" "$src/scripts/regenerate_train_data.py" --model "$snap" --reasoning disable \
      --temperature 0 --max-tokens 640 --concurrency 64 \
      --input-file-path "$in" --output-file-path "$out" \
      --server-address "127.0.0.1:$port"
    load
    ;;
  cycle)  # phase 7: feature batch -> resumed training -> delete, until the job budget ends
    shift
    "$py" -m spec.t15_cycle --target "$snap" --specforge-src "$src" "$@"
    ;;
  features)
    n="${2:?features needs the number of samples}"
    load
    .venv-specforge/bin/torchrun --standalone --nproc_per_node=1 "$src/scripts/prepare_hidden_states.py" \
      --strategy eagle3 --target-model-path "$snap" --draft-model-config spec/configs/qwen3-4b-eagle3.json \
      --data-path .cache/t15/data/train_regen.jsonl --output-path .cache/t15/features \
      --cache-dir .cache/t15/sf-cache --chat-template qwen --max-length 1024 --num-samples "$n" \
      --tp-size 1 --batch-size 16 --num-workers 2 --num-io-threads 4 --sglang-mem-fraction-static 0.6
    load
    ;;
  bench)  # bench OUT_DIR [BATCHES MODE...]; MODE is base or NAME=DRAFT_DIR:STEPS:TOPK:DRAFT_TOKENS.
    # Phase 6 used the defaults: 1,4,16,64 base eagle3=.cache/t15/draft-sglang:3:1:4.
    out="${2:?bench needs OUT_DIR}"
    batches="${3:-1,4,16,64}"
    modes=("${@:4}")
    [[ ${#modes[@]} -gt 0 ]] || modes=(base "eagle3=$draft:3:1:4")
    mkdir -p "$out"
    for mode in "${modes[@]}"; do
      if [[ $mode != base ]]; then
        IFS=: read -r d _ <<< "${mode#*=}"
        [[ -f $d/config.json ]] || { echo "t15: no exported draft at $d" >&2; exit 2; }
      fi
    done
    for round in 1 2; do
      for mode in "${modes[@]}"; do
        name=base extra=()
        if [[ $mode != base ]]; then
          name=${mode%%=*}
          IFS=: read -r d steps topk ntok <<< "${mode#*=}"
          extra=(--speculative-algorithm EAGLE3 --speculative-draft-model-path "$d"
            --speculative-num-steps "$steps" --speculative-eagle-topk "$topk"
            --speculative-num-draft-tokens "$ntok")
        fi
        load
        start_server ".cache/t15/bench-$(basename "$out")-$name-$round-server.log" "${extra[@]}"
        "$py" -m spec.bench_tpot --url "http://127.0.0.1:$port" --prompts .cache/t15/data/eval_prompts.jsonl \
          --tokenizer "$snap" --batches "$batches" --max-tokens 256 --label "$name-r$round" \
          --out "$out/$name-r$round.json"
        stop_server
      done
    done
    load
    ;;
  *)
    echo "t15: unknown stage $stage" >&2
    exit 2
    ;;
esac
