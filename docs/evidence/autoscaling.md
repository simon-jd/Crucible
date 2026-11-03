# Task 14: local router + autoscaler on one RTX 4090 — 2026-10-27

Result: the Go router and Go autoscaler ran against real vLLM replicas for the first time. On a
4x step burst, the autoscaler's scale-up was acted on and the new replica served traffic a median
35.5 s after the burst started (3 trials). It was stopped 223 s after the load dropped. **On this
one GPU, autoscaling lowered goodput**: the burst-phase SLO goodput was 4.00 req/s with a fixed
single replica and 0.675 req/s with autoscaling (medians of 3). Both replica processes share the
same GPU, so the second replica adds batch slots but slows every decode step (p50 TPOT 3.7 → 5.6 ms,
SLO 4.86 ms). Cold start from an evicted page cache adds about 0.2 s over a warm start (11.26 vs
11.05 s). Fetching the weights from huggingface.co (23.9 s) would roughly triple start time.
Other projects were running on the host (32 cores, load average 5.8 at the start and 4.2 at the end).

## Setup

| | |
|---|---|
| Replicas | vLLM 0.10.2, `Qwen/Qwen2.5-0.5B-Instruct` rev `7ae557604adf67be50417f59c2c2f167def9a775`, each `--gpu-memory-utilization 0.25 --max-num-seqs 8 --max-model-len 8192 --enforce-eager --no-enable-prefix-caching --seed 17`; `a` on port 31101 for every trial, `b` on 31102 only when scaled up |
| Router | `serving/router/cmd/router` with both backends configured; it routes only to backends whose adapter snapshot is fresh (≤15 s, polled every 5 s), choosing the lowest queue (no prefix claims) |
| Autoscaler | `serving/autoscaler`, dry-run, 2 s interval, min 1 / max 2, queue-high 4 / queue-low 1 per desired replica, p95 TTFT high 0.8 s / low 0.2 s, 60 s up / 300 s down cooldown |
| Actuator | `serving/localscale/run.py`: on a proposal 1→2 it reports desired 2, spawns `b`, and makes it routable when `/health` is 200 and one completion succeeds. On 2→1 it reports desired 1, removes `b` from routing, waits 21 s plus an empty queue, then stops `b` |
| Adapter | `serving/localscale/adapter.py`: per-replica router snapshots from vLLM `/metrics` (waiting queue), and autoscaler metrics with p95 TTFT from the summed TTFT-histogram deltas over the last 10 s (null when no samples) |
| Load | `trace.csv`: 1,123 requests with Azure conv request lengths (the 456 T12 rows, cycled) and seeded Poisson arrivals: 60 s at 1.25 req/s (1x), 120 s at 5 req/s (4x), 360 s at 1.25 req/s. The Go replayer (`-api openai-text`, 60 s deadline) sends string prompts of single-token words ([check](t14-autoscaling/word-allowlist-check.json): 4,096 words, 0 / 200 mismatched tokenizations) |
| Plan | static / auto alternating over 3 pairs: order static, auto, auto, static, static, auto. SLO = pooled p99 TTFT and TPOT of the 1x phase of the 3 static trials: TTFT ≤ 55.7 ms, TPOT ≤ 4.86 ms (183 samples) |

The pilot (one replica, 1–6 req/s ramp) showed TTFT tails from 4 req/s and p50 TTFT 312 ms at
6 req/s, so the 4x level (5 req/s) overloads one replica.

## Results

Median [min, max] over 3 trials. All 6 trials are sender-valid with 0 failed requests.

| | static (1 replica) | auto (1→2→1) |
|---|---|---|
| 1x phase goodput (req/s) | 1.017 [0.983, 1.017] | 0.950 [0.867, 1.017] |
| 4x burst goodput (req/s) | 4.000 [3.967, 4.150] | 0.675 [0.658, 1.342] |
| 4x burst p99 TTFT (ms) | 970 [943, 990] | 1355 [1113, 2520] |
| 4x burst p50 TPOT (ms) | 3.70–3.75 | 5.44–5.69 |
| after-drop 1x goodput (req/s) | 1.139 [1.136, 1.169] | 0.897 [0.694, 0.914] |
| scale-up proposal after burst start (s) | 54.0 [30.0, 56.0] (logged, not acted on) | 20.0 [14.0, 44.0] |
| new replica ready after burst start (s) | — | 33.3 [28.1, 56.5] |
| first request done on new replica (s) | — | 35.5 [30.5, 61.7] |
| spawn → ready (s) | — | 13.25 [12.45, 14.05] |
| scale-down proposal after load drop (s) | — | 202.0 [196.0, 228.0] |
| new replica stopped after load drop (s) | — | 223.4 [217.5, 249.4] |

- Offered burst load is 637 requests in 120 s (5.3 req/s). The static replica completed all of
  them; 476–498 met the SLO. With autoscaling, only 79–161 did.
- Why autoscaling lost: two vLLM processes on one GPU take turns on the device, so the decode
  step of every running request gets slower (p50 TPOT +50%), and the SLO TPOT (from the light 1x
  phase) is tight. `b`'s startup (profiling, first completion) also competes with `a` during
  the burst, which shows in the higher p99 TTFT. After the drop, `b` stays up until the 300 s down
  cooldown ends, and the shared GPU keeps TPOT high, so after-drop goodput is also lower.
- The scale-down timing follows the policy: the down cooldown counts from the scale-up, so the
  proposal comes about 300 s after scale-up. Draining (21 s router freshness wait) adds the rest.
- Peak owned RSS was 5.77 GB (two replicas plus driver), under the 16 GB cap.

## Cold start

Start-to-ready of one replica (spawn → `/health` 200 and one successful completion), 3
alternating warm/cold pairs ([raw](t14-autoscaling/cold/cold-summary.json)):

| | warm page cache | cold (snapshot evicted with `posix_fadvise(DONTNEED)`) |
|---|---|---|
| start → ready (s) | 11.05 [10.84, 11.07] | 11.26 [11.05, 11.64] |
| vLLM "Loading weights took" (s) | 0.08–0.11 | 0.36–0.39 |

Weight loading is a small part of startup for a 0.99 GB model from local NVMe; process start, CUDA
init and memory profiling dominate. As a remote "object store" stand-in, fetching
`model.safetensors` (988,097,824 bytes, sha256 checked) from huggingface.co took 20.07 / 24.06 /
23.89 s ([raw](t14-autoscaling/hf-fetch-timings.json)). Fetch plus start is therefore about 35 s.
That figure is a sum of two separate measurements, not one end-to-end run.

## Limits

- One GPU: the second replica is a second process on the same card, not new hardware. These
  results measure the control loop and its cost on a shared device. They do not show what
  scaling to a separate GPU would give. A multi-GPU node would be needed for that (paused,
  未在真实多机上验证).
- Kubernetes was not used: no Deployment, no scale subresource, no kind cluster. The actuator
  stands in for the Kubernetes scale write. Kubernetes and object-storage cold start are not done.
- The workload is small (0.5B model, 8 sequence slots per replica) so that one 4090 can run two
  replicas. The SLO comes from the same trials' 1x phase.
- p95 TTFT comes from vLLM's coarse histogram buckets (interpolated). Proposal times depend on
  where the 2 s autoscaler tick falls, which explains the 14–44 s spread.
- Raw outputs: `t14-autoscaling/scale/` (per trial: `replay.json`, `events.json`, `timeline.json`,
  `decisions.json`, router/autoscaler/server logs; `summary.json`), `t14-autoscaling/cold/`.
  Command: `make t14-bins`, then `bash scripts/run_localscale_gpu.sh scale <out> --base-rps 1.25
  --phases 60:1,120:4,360:1 --pairs 3` and `bash scripts/run_localscale_gpu.sh cold <out> --pairs 3`.

# Earlier CPU-only evidence (2026-10-24)

Implementation scope: CPU-only stdlib Go router, streaming proxy, SLO-aware policy, explicit metrics adapters, dry-run controller, conditional future Kubernetes scale client, verified weight-cache staging and experiment runners. There was no available `cache-aware-routing-lab` source to port; this is an independent implementation preserving the requested behavior. No neighboring private code was used.

The four required tests were written before implementation and failed on undefined router/policy types. The RED output is [task14-red.log](raw/task14-red.log); focused GREEN is [task14-green-focused.log](raw/task14-green-focused.log). Additional loopback coverage exercises SSE delivery before completion, cancellation propagation, overload rejection, response forwarding, body bounds, no POST replay on connection failure, cache bounds/freshness/scope, concurrent updates/reservations, exact cooldown boundaries, invalid/missing metrics, dry-run, conditional scale conflicts and uncertain ACK reconciliation. A built-router test checks startup, actual forwarding, SIGTERM exit and closed listener.

The policy uses queue depth and p95 TTFT, deadband hysteresis, exact >=60-second up and >=300-second down cooldowns. It scales one desired replica per decision within configured bounds. First up is immediate; first down waits 300 seconds from construction. Cooldown advances only when desired replica count changes observably or a checked scale ACK confirms it. `cur` means desired count, never ready count. State is local to one controller process; restart and HA limitations are documented.

Routing uses longest trusted observed byte-prefix match, with model/revision/adapter/tenant scope. Selection pressure is the maximum of waiting queue and local outstanding reservations; overlap is not double-counted by addition. Pressure above the threshold spills to the least-pressure fresh backend. Cache observations are bounded and expire. They are not inferred from successful requests or supplied by client hit headers; no GPU KV hit rate has been measured.

## Reproduction and contract boundaries

See [the deployment and adapter guide](../../serving/deploy/README.md) for exact JSON schemas, build commands, flags, bounds and failure behavior. Three runnable binaries are provided: router, autoscaler and experiments. The custom adapter contracts are not claims that SGLang supplies native cache-status or cold-start endpoints. Backend membership is explicitly configured; future orchestration must provision per-backend URLs and update/restart router configuration when topology changes. Automatic Kubernetes endpoint discovery is not implemented.

All Go/TMP caches are project-local, with `GOENV=off GOTOOLCHAIN=local GOPROXY=off GOMAXPROCS=2` and `-p=1`. Go 1.27.3 was preinstalled; no external dependency was downloaded. The approved final gate builds the binaries, checks formatting, runs `go test -count=1 -p=1 -v ./serving/...`, `go vet -p=1 ./serving/...`, builds a race-instrumented router and runs `go test -race -count=1 -p=1 -v ./serving/...`. `TASK14_ROUTER_BINARY` enables the actual process lifecycle test. `scripts/task14_check.py` records resource samples around that gate. Unchanged unrelated Python tests were not rerun.

## Experiments and meaning of results

The injected-clock 4x burst alternates baseline/candidate order across three matched pairs. The queue signal changes from 2 to 8 per replica at logical second 10 and returns to idle at second 190. Each burst trial proposes desired replicas 3/4/5 at logical seconds 10/70/130 and first down to 4 at second 430. First up reaction is 0 logical seconds (median/range 0/[0,0]); first down after load drop is 240 logical seconds (240/[240,240]). These measure policy arithmetic, not elapsed GPU replica activation or scheduling latency. Raw runnable output: [task14-burst.json](raw/task14-burst.json).

The cold-start fixture alternates object/NVMe source labels over three matched pairs and reports measured HTTP start-to-ready elapsed times with median/min/max. Both source labels use the same self-owned loopback protocol fixture. Those timing values **do not compare actual NVMe to object storage**. The fixture checks source/content/trial binding and cleanup on success, mismatch and timeout. Small HTTP downloads test SHA256 integrity, byte bounds, atomic cache population, cache-hit reuse and temporary-file cleanup. Raw fixture timings are in [task14-experiments-focused.log](raw/task14-experiments-focused.log) and the final Go log. One intermediate missing-import build failure was corrected and retained as [task14-experiments-build-failure.log](raw/task14-experiments-build-failure.log).

The future cold runner requires explicit endpoints, immutable content hash, model/image/hardware metadata, matched reset/cache conditions and >=3 alternating pairs. Every trial attempts an explicit stop, including cancellation/failure; failed cleanup is reported as unknown rather than claimed successful. Actual model readiness and storage behavior require an implemented trusted adapter and real measurement.

## Acceptance limits

The offline Kubernetes manifest parses as five objects and defaults to dry-run with maximum one desired replica. Its placeholders intentionally prevent claiming a verified deployment. The init container is specified to populate an immutable node-local NVMe artifact read/write; serving mounts that cache read-only. Neither manifest application nor a real model/NVMe/object-store experiment was executed. Real GPU scale reaction, physical cold-start comparison, held-out SLO calibration, native backend cache instrumentation and Kubernetes deployment remain **unverified**. This task spent $0, used no GPU queue, launched no model and accessed no cloud/remote API.

## Final gates and resource closure

Final gate exit **0**: formatting/build passed; **24/24 top-level tests passed** normally and **24/24 under the race detector**, including the built-router lifecycle test; vet produced no diagnostics. Logs: [normal](raw/task14-final-go.log), [race](raw/task14-final-race.log), [vet](raw/task14-final-vet.log), [gate resource result](raw/task14-final-resources.json). The small measurement helper also passed targeted Ruff checking after import/noqa cleanup; no behavioral code changed after the Go gate.

The fresh preflight recorded 12,404,087,067 project bytes and reserved 700,000,000 additional bytes against the 14,000,000,000 limit. The final gate lasted 37.43 seconds. Across 124 samples (requested interval 0.25 seconds plus scan cost), the wrapper/descendant RSS peak was **275,496,960 bytes**, Task 14 cache peak **357,445,194 bytes**, and temporary-file peak **105,311,844 bytes** (temporary files are included in cache bytes). These are sampled peaks, not guaranteed continuous maxima. The project measured 12,687,637,689 bytes at cleanup, with 283,440,066 Task 14 cache bytes retained. No checkpoint was removed or changed. [Cleanup evidence](raw/task14-cleanup.json) records **zero Task 14 resident processes** and **zero leftover Task 14 temporary files**. No service remains running.

## Fix round 1 — bounded downstream I/O and signal cleanup

Independent review found two lifecycle gaps in the initial implementation. Both now have RED regressions and focused normal/race coverage: [RED](raw/task14-fix1-red.log), [focused GREEN](raw/task14-fix1-focused.log), [race GREEN](raw/task14-fix1-race.log). The original evidence is preserved.

The router starts its lifetime budget before reading the body and installs read/write socket deadlines. Context cancellation interrupts blocked socket I/O. Cancellation callbacks are stopped or joined before handler return, so a late callback cannot corrupt a later keepalive exchange. Deadlines remain through net/http's post-handler request-body drain and final response flush; the Go HTTP server resets them before normal keepalive reuse. Expired exchanges abort. Wrappers must preserve ResponseController deadline capabilities directly or through Unwrap; unsupported/erroring writers fail closed before admission/forwarding. Real TCP tests leave a client open without reading, verify handler completion/zero leases within the budget, cover earlier parent cancellation and a stalled request body, and verify the same connection remains usable after a successful request's old deadline.

The experiments executable now handles SIGINT/SIGTERM through a signal-derived context. Actual normal and race binaries are started against self-owned loopback adapters, signalled after the start acknowledgment, observed attempting the source/hash/trial-bound stop, and reaped. A failed stop continues to emit **remote cleanup unknown**. These guarantees do not extend to SIGKILL/host failure or to untested real model processes.

The scoped gate passed **10/10 top-level tests normally and 10/10 under race**, changed-package vet, formatting, and normal/race executable builds. An expected reverse-proxy `context deadline exceeded` diagnostic appears in the normal intentional timeout test; no tests failed and no race was detected. No unrelated Go/Python suite or GPU work was repeated. [Fresh preflight](raw/task14-fix1-preflight.json): project 10,687,109,710 bytes plus 700 MB reserve fits the 14 GB cap. [Resource samples](raw/task14-fix1-resources.json): exit 0, 9.33 seconds, 31 samples, peak wrapper/descendant RSS 199,499,776 bytes, Task 14 cache 317,038,656 bytes, temp 13,776,452 bytes (included in cache). Sampling is not a strict continuous peak. [Cleanup](raw/task14-fix1-cleanup.json) confirms zero Task 14 processes and zero temporary files. Real deployment/GPU/storage acceptance and the source-provenance gap remain unchanged.
