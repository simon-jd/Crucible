# E3: autoscaling onto a second GPU, Snowflake GPU_NV_2M — 2026-11-04

Hardware: Snowflake GPU_NV_2M: 2× A10 24 GB (vGPU "A10-24Q", driver 570.211.01) in one node, no NVLink/P2P
(`nvidia-smi topo -m`: GPU0–GPU1 `SYS`, GPU0 on NUMA node 0 with CPUs 0–17, GPU1 on NUMA node 2 with CPUs
36–53), 72 vCPUs visible in the container. One ML Job on `CRUCIBLE_2XA10_POOL`, vLLM 0.10.2 / torch 2.8.0 /
transformers 4.55.4 / numpy 2.2.6 from the pins.

Result: with the second replica on its own GPU, scaling out no longer slows the first replica, and it helps.
On the same 120 s burst at 4x load, burst goodput was 0.042 req/s with one fixed replica, 0.308 req/s with
autoscaling 1 → 2, and 0.525 req/s with two fixed replicas (medians of 3). The new replica served its first
request a median 38.5 s after the burst began. p50 TPOT in the burst stayed at 9.45–9.50 ms in all three
conditions (it rose 3.7 → 5.6 ms on the shared RTX 4090). Failed requests fell from 210–213 per trial (one
replica) to 29–40 (autoscaling) and 0 (two replicas). But one A10 replica is much slower than the 4090 replica
was (p50 TPOT 9.1 vs 3.7 ms at 1x), so the 4x burst overloads it badly, and goodput in every condition is
capped by the TPOT SLO: p50 burst TPOT is above the 9.37 ms SLO even with two replicas. The absolute numbers
are not comparable with [autoscaling.md](../../autoscaling.md): different GPU, CPU and SLO.

## Setup

Same workload and definitions as [autoscaling.md](../../autoscaling.md), except where listed under Deviations.

| | |
|---|---|
| Replicas | vLLM 0.10.2, Qwen2.5-0.5B-Instruct rev `7ae557604adf67be50417f59c2c2f167def9a775` from asset `models/Qwen2.5-0.5B-Instruct` (`model.safetensors` sha256 `fdf756fa…fb7fe`, identical to the 4090 run's snapshot), each `--gpu-memory-utilization 0.25 --max-num-seqs 8 --max-model-len 8192 --enforce-eager --no-enable-prefix-caching --seed 17`. `a` on GPU0 (`CUDA_VISIBLE_DEVICES=0`), `b` on GPU1 (`CUDA_VISIBLE_DEVICES=1`), set per replica by `serving/localscale/run.py --gpus a=0,b=1` |
| Router, autoscaler, replayer | the repository's Go code built with `CGO_ENABLED=0 -trimpath` on home-ubuntu (`make snowflake-bins`), shipped with the payload; sha256 in `main/node.json`. Router and autoscaler flags unchanged (least queue over fresh snapshots; 2 s interval, min 1 / max 2, queue 4 / 1, p95 TTFT 0.8 / 0.2 s, 60 s up / 300 s down cooldown, dry-run with the actuator acting) |
| Load | the same `trace.csv` (1,123 requests: 60 s at 1.25 req/s, 120 s at 5 req/s, 360 s at 1.25 req/s; Azure conv lengths; seed 17), Go replayer `-api openai-text -account`, 60 s deadline, the same word list (sha256 `54e33c43…6c28`) |
| Conditions | `static`: fixed 1 replica on GPU0. `auto`: 1 → 2 → 1, `b` spawned on GPU1 on the autoscaler's proposal. `static2`: fixed 2 replicas, `b` started on GPU1 before the replay (28.3–28.7 s start → ready) and routable throughout. Order static, auto, static2, static2, auto, static, static, auto, static2 (3 trials each) |
| SLO | pooled p99 of the 1x phase of the 3 static trials, as before, re-derived on this node: TTFT ≤ 85.7 ms, TPOT ≤ 9.37 ms (183 samples) |

## Results

Median [min, max] over 3 trials. All 9 trials are sender-valid, and in every trial
`Completed + Failed + Retried == Planned` (1,123).

| | static (1 replica, GPU0) | auto (1 → 2, b on GPU1) | static2 (GPU0 + GPU1) |
|---|---|---|---|
| 4x burst goodput (req/s) | 0.042 [0.017, 0.050] | 0.308 [0.025, 0.433] | 0.525 [0.375, 0.958] |
| 1x phase goodput (req/s) | 1.000 [1.000, 1.017] | 0.950 [0.933, 1.000] | 1.017 [0.983, 1.017] |
| after-drop 1x goodput (req/s) | 0.792 [0.697, 0.808] | 0.967 [0.964, 1.008] | 1.006 [0.939, 1.022] |
| burst p50 TPOT (ms) | 9.45 [9.41, 9.46] | 9.45 [9.44, 9.55] | 9.50 [9.50, 9.53] |
| burst p50 TTFT (ms) | 16,924 [16,743, 17,172] | 3,673 [3,412, 3,802] | 32.3 [31.9, 32.9] |
| burst p99 TTFT (ms) | 21,394 [21,350, 21,453] | 19,443 [19,194, 19,583] | 1,970 [1,818, 2,099] |
| 1x phase p50 TPOT (ms) | 9.10 [9.07, 9.15] | 9.13 [9.09, 9.19] | 8.54 [8.50, 8.54] |
| completed / failed / retried, per trial (of 1,123) | 913/210/0, 910/213/0, 912/211/0 | 1083/40/0, 1094/29/0, 1089/34/0 | 1123/0/0 in all 3 |
| burst requests ok / TTFT within SLO / TPOT within SLO (of 637) | 425–429 / 10–11 / 73–136 | 597–608 / 127–145 / 58–185 | 637 / 411–424 / 66–171 |
| requests served by `b` | — | 342 / 353 / 349 (two starts, see below) | 493 / 491 / 494 |
| GPU1 mean utilization in the burst (nvidia-smi, %) | 0.0 | 29.3 / 31.5 / 31.1 | 43.8 / 44.0 / 44.1 |
| scale-up proposal after burst start (s) | 6.0 (logged, not acted on) | 6.0 [6.0, 6.0] | — |
| new replica ready after burst start (s) | — | 34.5 [34.3, 34.7] | — |
| first request done on new replica (s) | — | 38.5 [36.5, 41.5] | — |
| spawn → ready (s) | — | 28.5 [28.3, 28.7] | — |
| scale-down proposal after load drop (s) | — | 188.0 [188.0, 188.0] | — |
| new replica stopped after load drop (s) | — | 237.8 [228.8, 238.0] | — |

- **Placement.** GPU1 memory rose to about 6.4–6.5 GiB only while `b` ran, and stayed at 0 in static trials
  (`main/gpu-samples.csv`, 1 s samples of both GPUs). GPU0 averaged 45–46% utilization in the burst in every
  condition, so `a` was not slowed by `b`. Low utilization with a full batch suggests the eager-mode replicas
  are limited by the host side (kernel launches), not by the A10.
- **Why goodput stays low with two GPUs.** TPOT does not improve with a second GPU: each replica still decodes
  up to 8 sequences in eager mode, and its p50 TPOT under a full batch (9.45–9.50 ms) is above the SLO taken
  from the light 1x phase (9.37 ms). What the second GPU removes is queueing: burst p50 TTFT 16.9 s → 32 ms and
  411–424 of 637 burst requests within the TTFT SLO, against 10–11 with one replica. Goodput needs both.
- **Failures.** Every failure is an HTTP 503 from the router during the burst (static: scheduled 27.7–120.7 s
  after burst start). The router answers 503 when its 64-request admission limit is full ("router full") or when no
  backend can be selected; the replay record keeps only the status, so the split was not recorded. The
  backlog of one replica passes 64 in-flight requests, so overload is shed rather than queued past the 60 s
  deadline (0 timeouts in all trials). With autoscaling, the 29–40 failures are scheduled 27.7–42.5 s after
  burst start, ending about 8 s after `b` became routable (34.3–34.7 s).
- **Second scale-up.** In all three auto trials the autoscaler proposed 1 → 2 again at 482 s (302 s after the
  load drop) and `b` was started a second time; it was still serving when the replay ended and was stopped
  at trial end. The trigger is a short cluster of Poisson arrivals at about 477 s that filled `a`'s 8 slots:
  windowed p95 TTFT jumped from 0.04 s to 1.4–1.5 s, above the 0.8 s threshold (`trial-*-auto/timeline.json`;
  the static trials logged the same proposal). The 4090 run did not show this. The "requests served by b"
  figures cover both starts (access-log `POST` count minus one readiness completion per start).
- **Spawn to ready** is 28.5 s on this node against 13.25 s on the 4090, and the readiness wait is the bulk
  of the 38.5 s from burst start to the new replica's first request; the proposal came after 6 s.
- Peak owned RSS 6.34 GB (two replicas plus driver). Replica `a` start → ready 32.7 s.

## Deviations from autoscaling.md

- Different hardware (A10 vGPU instead of RTX 4090, a Snowflake container instead of the shared workstation), so
  the SLO values differ (re-derived with the same rule: TTFT 85.7 vs 55.7 ms, TPOT 9.37 vs 4.86 ms).
- A third condition, `static2`. Replicas are pinned with `CUDA_VISIBLE_DEVICES` per replica; the 4090 runs set
  `CUDA_VISIBLE_DEVICES=0` for all.
- Replicas start without `PROMETHEUS_MULTIPROC_DIR`. The ML Job image sets
  `PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus-multi-dir`; inherited by both vLLM processes, it made each
  replica's `/metrics` report the sum over both (a fresh `b` showed 658 completed requests, identical to `a`;
  `smoke/smoke2-static2-timeline.json`), so the router's per-replica queues and the autoscaler's queue were
  wrong. Dropping it restores one in-process registry per replica, which is what the 4090 runs had
  (`smoke/smoke3-static2-timeline.json` shows separate counts). Fix: commit d17579f.
- The replayer now runs with `-account` in scale mode (it already did since the fault work), so the accounting
  row is new; T14 had no accounting.
- Replicas are not pinned to the CPUs of their GPU's NUMA node.

## Runs and credits

Credits are the driver's upper-bound estimates (wall time from submit to pool stop × 3.50 credits/h), from
`~/.cache/crucible-snowflake-ledger.jsonl`; total 7.66.

| Run | What happened | Credits |
|---|---|---|
| smoke 1 (20/60/40 s phases, 3 conditions) | failed at the first trial: the stage mount drops the executable bit of the shipped Go binaries (`smoke/smoke1-exec-bit-error.txt`); the payload now copies them to local disk | 0.439 |
| smoke 2 (same) | ran end to end, found the shared-metrics problem above; its numbers are not used | 0.908 |
| main, first launch | cancelled by me 62 s after submit, before it ran, once smoke 2 showed the metrics problem | 0.061 |
| smoke 3 (20/40/20 s, static + static2) | confirmed separate per-replica metrics | 0.730 |
| main (9 trials) | 5,215 s inside the job, 5,683 s submit → pool stop; job `CRUCIBLE_E3_MAIN_16M5QAIIMN11N` | 5.525 |

## Limits

- Both replicas are on one node, so this is scale-out onto a second device of the same host (shared CPU, PCIe
  and host memory), not onto a second machine; no data moves between the GPUs. Kubernetes was not used; the
  actuator stands in for the scale write, as before.
- The workload is the 4090 one, unchanged. On this node it overloads one replica far more than on the 4090,
  so the static baseline sheds about 210 burst requests; a lower burst level would give a different picture.
- Goodput is sensitive to the TPOT SLO, which comes from the 1x phase of the same session's static trials and
  sits just below the full-batch decode time.
- One session, 3 trials per condition, deterministic workload: the ranges show system noise only.

## Raw outputs and reproduction

- `main/`: `summary.json` (driver summary), `e3-table.json` (the table above, from `e3_table.py main`),
  `node.json` (GPUs, topology, CPUs, packages, binary and model hashes), `gpu-samples.csv`, `trace.csv`,
  `driver.log`, per trial `replay.json`, `events.json`, `timeline.json`, `decisions.json`, `router.log`,
  `autoscaler.log`, `router.json`; vLLM logs as excerpts (`server-*.excerpt.log`, first line gives the
  request count); `mljob-report.json`, `instance0.excerpt.log`. `smoke/`: reports and the timelines named
  above.
- Command (from the worktree root, after `make snowflake-bins` and copying `qwen25-05b-words.json` to
  `.cache/snowflake/`):
  `.venv-mljobs/bin/python scripts/snowflake/mljob.py --pool CRUCIBLE_2XA10_POOL --entry
  scripts/snowflake/payloads/e3_autoscale.py --include serving/localscale serving/baseline scripts/snowflake
  .cache/snowflake/bin .cache/snowflake/qwen25-05b-words.json --pip vllm==0.10.2 torch==2.8.0
  transformers==4.55.4 numpy==2.2.6 --out .cache/snowflake/e3-main --max-minutes 150 --name e3_main --args
  e3-main --phases 60:1,120:4,360:1 --base-rps 1.25 --conditions static,auto,static2 --pairs 3`, then
  `python scripts/snowflake/assets.py get e3/runs/e3-main.tar.gz <dir>` (sha256 `752144a9…0def`).
