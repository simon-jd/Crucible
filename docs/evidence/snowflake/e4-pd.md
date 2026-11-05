# E4: prefill/decode disaggregation 1P:1D on Snowflake A10s (vLLM P2pNcclConnector) — 2026-11-04/05

Result: 1P:1D disaggregated serving ran and was measured for the first time in this repository, on
a different stack from the manifests: vLLM 0.10.2's NCCL KV connector (`P2pNcclConnector`, V1
engine) moving KV over NCCL's SHM transport inside one node or NCCL NET/Socket (TCP) between two
nodes. The SGLang + RDMA manifests in `serving/deploy/pd-*.yaml` remain untested.

On this workload the fair two-GPU baseline wins. Two data-parallel replicas reach 2.09 / 2.52 req/s
goodput at 2x / 4x. Same-node 1P:1D matches them at 1x and 2x (1.07 / 2.04 req/s) with a lower
decode tail (p99 TPOT 22.8 vs 40.5 ms at 2x), but collapses at 4x (0.20 req/s): the single prefill
GPU cannot keep up, because the connector makes every prefill step wait until its KV has been sent
(median prefill phase 0.29 s at 1x, 9.1 s at 4x). Cross-node 1P:1D over TCP degrades already at 2x
(1.50 req/s): KV moves at about 0.4 GB/s, so a median prompt's 151 MB of KV takes 0.34–0.97 s. Making
the connector survive 4x at all took two fixes to vLLM 0.10.2 (listed below).

## Hardware and stack

| | |
|---|---|
| Layout 1 `colo` | one Snowflake GPU_NV_SM node (1× A10 24 GB, vGPU "A10-24Q"), `SYSTEM_COMPUTE_POOL_GPU` |
| Layout 2 `dp2`, layout 3 `pd-shm-ce` | Snowflake GPU_NV_2M: 2× A10 24 GB (vGPU) in one node, no NVLink/P2P (`nvidia-smi topo`: SYS, GPUs on different NUMA nodes), `CRUCIBLE_2XA10_POOL` |
| Layout 4 `pd-tcp` | 2 Snowflake GPU_NV_SM nodes (1× A10 24 GB each) over TCP, one two-instance ML Job |
| Software | vLLM 0.10.2, torch 2.8.0, transformers 4.55.4, NCCL 2.27.3 (`nvidia-nccl-cu12`), aiohttp 3.12.15, Python 3.12 (job image), installed from Snowflake's PyPI proxy |
| Model | `Qwen/Qwen3-8B-AWQ` rev `4da05a8e…` (stage asset `models/Qwen3-8B-AWQ`), served as `awq_marlin` |

## Setup

- **Workload**: the Task 12 harness unchanged (`serving/baseline/vllm_baseline.py` functions, Go
  open-loop replayer `serving/replay` built static, `-api openai`, 60 s deadline, 192 in flight,
  seed 17): Azure LLM inference 2023 conv, 191-request calibration window and 265-request
  evaluation window; "1x" = 0.25× the source rate, 2x and 4x scale arrivals (4x = the trace's real
  rate); 3 evaluation rounds per level in the baseline's rotated order; the same 4 warmup requests.
- **SLO**: the 4090 SLOs do not apply. They were **recalibrated on layout 1** (colocated A10) with
  3 sender-valid 1x calibration runs, same rule (pooled nearest-rank p99 over successful ≥2-token
  requests): **TTFT ≤ 2110.7 ms, TPOT ≤ 90.27 ms** (573 samples). That SLO is applied to all four
  layouts offline (`serving/pd/analyze.py`); the other layouts ran no calibration windows.
- **Servers**: Task 12 arguments for every server (`--gpu-memory-utilization 0.8 --max-model-len
  8192 --max-num-seqs 64 --no-enable-prefix-caching --seed 17`, CUDA graphs, chunked prefill with
  vLLM's default 2048 batched tokens). P/D decode servers use 0.7 (see deviations). KV cache:
  colocated / each replica / prefill 87,824 / 87,824 / 87,664 tokens; decode 74,672 (2M node) and
  69,488 (SM node).
- **Proxy** (`serving/pd/proxy.py`, aiohttp, on the head node for every layout): `colo` = pass-through
  to one server; `dp2` = least-outstanding-requests routing to two replicas (the Go router's
  least-queue rule; the Go router itself rejects token-ID prompts); `pd` = the connector's protocol
  (request id `___prefill_addr_…___decode_addr_…` in `X-Request-Id`, the prefill copy with
  `max_tokens=1` and no streaming, then the original request to the decode server, streamed back).
  At most 96 requests sit between prefill start and decode finish; later ones wait in the proxy
  (counted in TTFT).
- **Connector**: `P2pNcclConnector`, `send_type` PUT_ASYNC, 8 NCCL channels, ZMQ/KV ports 12980
  (prefill) and 12985 (decode), HTTP 12971/12972, proxy 12970, model copy 12975 (all in E4's band).
  Decode: `kv_buffer_size` 2e9 bytes of received KV on the GPU, then a 64 GiB pinned host pool.
  NCCL settings from E0: `NCCL_SHM_USE_CUDA_MEMCPY=1` on one node (transport logged as
  `SHM/CE/direct`), `NCCL_SOCKET_NTHREADS=8 NCCL_NSOCKS_PERTHREAD=8 NCCL_BUFFSIZE=16777216` across
  nodes (`NET/Socket`, "Using 8 threads and 8 sockets per thread"; E0's `NCCL_MIN_NCHANNELS=4` is
  overridden by the connector, which pins the channel count to 8).
- **KV measurement** (`serving/pd/kvprobe.py`): vLLM exposes no KV transfer metrics for this
  connector, so a vLLM general plugin (`serving/pd/plugin/`) wraps `P2pNcclEngine` methods and logs
  every (request, layer) tensor: bytes and wall time of each send (ZMQ handshake + `ncclSend` +
  stream sync), receive time, and how long the decode forward pass waited for it.
- **Launch**: `scripts/snowflake/payloads/e4_pd.py` via `scripts/snowflake/mljob.py`; the second
  instance of `pd-tcp` is driven through a Ray actor pinned to that node, and gets the model from
  the head over HTTP (the worker container has no Snowpark session). Results go to the asset stage
  (`e4-results/<tag>.tar`, fetched with `scripts/snowflake/e4_results.py`).

## Results

Goodput in req/s, median [min, max] over 3 rounds; "Level − 1x" is the paired difference within a
round. Latency cells are medians over rounds. "—": no sender-valid round.

| Layout | Level | Goodput | Level − 1x | Good / sent | OK | Timeouts | p50 / p99 TTFT ms | p50 / p99 TPOT ms | Output tok/s |
|---|---|---|---|---|---|---|---|---|---|
| 1 colo (1 GPU) | 1x | 1.057 [1.057, 1.057] | — | 261/265 | 265 | 0 | 340 / 1969 | 23.6 / 68.3 | 311 |
| | 2x | 0.216 [0.201, 0.222] | −0.842 [−0.856, −0.835] | 30–33/265 | 253–254 | 11–12 | 3760 / 15917 | 112.3 / 183.2 | 478 |
| | 4x | — (voided) | — | 0/265 | 116–119 | 142–146 | 6088 / 48713 | 114.3 / 664.2 | 221 |
| 2 dp2 (2 GPUs) | 1x | 1.086 [1.085, 1.086] | — | 265/265 | 265 | 0 | 292 / 1315 | 15.9 / 29.8 | 315 |
| | 2x | 2.089 [2.064, 2.090] | +1.003 [+0.978, +1.005] | 264–265/265 | 265 | 0 | 312 / 1644 | 21.2 / 40.5 | 606 |
| | 4x | **2.517** [2.441, 2.588] | +1.431 [+1.356, +1.503] | 188–197/265 | 265 | 0 | 423 / 2483 | 73.5 / 131.0 | 1009 |
| 3 pd-shm-ce (1P:1D, 1 node) | 1x | 1.070 [1.070, 1.070] | — | 263/265 | 265 | 0 | 314 / 1734 | 15.6 / 18.0 | 313 |
| | 2x | 2.043 [2.043, 2.059] | +0.973 [+0.973, +0.989] | 261–263/265 | 265 | 0 | 356 / 2119 | 19.2 / 22.8 | 601 |
| | 4x | 0.200 [0.200, 0.200] | −0.870 [−0.871, −0.870] | 17/265 | 265 | 0 | 9146 / 17315 | 28.1 / 32.8 | 905 |
| 4 pd-tcp (1P:1D, 2 nodes) | 1x | 1.041 [1.041, 1.045] | — | 257–258/265 | 265 | 0 | 381 / 2677 | 16.1 / 19.3 | 311 |
| | 2x | 1.497 [1.480, 1.536] | +0.452 [+0.439, +0.495] | 192–199/265 | 265 | 0 | 1079 / 4823 | 21.2 / 24.2 | 593 |
| | 4x | 0.059 [0.059, 0.059] | −0.982 [−0.987, −0.982] | 7/265 | 264–265 | 0–1 | 26743 / 52673 | 23.0 / 25.6 | 640 |

KV transfer per request (P/D layouts, medians over rounds of the per-trial p50 unless noted; every
request of every trial has probe rows, 36 layer tensors each):

| Layout | Level | KV bytes p50 (mean) | Send span p50 / p90 s | Effective send GB/s | Decode wait p50 / max ms | Prefill phase p50 s | Proxy admission wait p90 s | Tensors parked in host pool |
|---|---|---|---|---|---|---|---|---|
| pd-shm-ce | 1x | 151.0 MB (140.8) | 0.261 / 0.416 | 2.81 | 0.0 / 0.0 | 0.286 | 0 | 0 |
| | 2x | 151.0 MB | 0.275 / 0.530 | 1.91 | 0.0 / 11 | 0.316 | 0 | ~1,230 of 9,540 |
| | 4x | 151.0 MB | 0.601 / 0.833 | 0.70 | 3 / 28 | 9.07 | 0 | ~4,600 |
| pd-tcp | 1x | 151.0 MB (140.8) | 0.339 / 0.720 | 0.48 | 0.0 / 19 | 0.350 | 0 | 0 |
| | 2x | 151.0 MB | 0.566 / 1.119 | 0.41 | 0.0 / 21 | 1.04 | 0 | ~1,080 |
| | 4x | 151.0 MB | 0.974 / 1.421 | 0.36 | 1 / 27 | 26.8 | 10.5 | ~1,950 |

"Send span" runs from the first to the last layer tensor of a request on the prefill side; the
effective rate is total bytes over summed per-tensor send time. A median prompt (1,017 tokens,
rounded up to 16-token blocks) carries 151 MB of fp16 KV (36 layers × 2 × 8 heads × 128 × 2 B =
147,456 B/token, matching `layouts.kv_bytes`); one evaluation window moves 37.3 GB. "Prefill
phase" is proxy-measured from admission to the prefill response, so it includes queueing at the
prefill server, compute and the KV send.

Reading the numbers:

- **Colocated A10** saturates between 1x and 2x: at 2x p50 TPOT is 112 ms against a 90 ms SLO and 11–12
  requests time out. The 4x trials are voided by the late-send rule (5–6 of 265 late per round; in
  round 0, 4 requests hit the replayer's 192-in-flight bound and 2 writes were 22–29 ms late); 0 of 265
  met the SLO in any round.
- **dp2 vs pd-shm-ce** (same node, same two GPUs): equal goodput at 1x and 2x, and P/D has the
  tighter decode tail (p99 TPOT 18.0 / 22.8 ms vs 29.8 / 40.5 ms), as expected when decode never
  shares a step with prefill. At 4x dp2 still serves 2.52 req/s, while P/D collapses to 0.20: with
  one prefill GPU each step waits in `wait_for_save` until its layers' KV is pushed (PUT with a ZMQ
  handshake and a stream sync per tensor), so prefill throughput is capped by compute plus transfer
  and the prefill queue grows (TTFT p50 9.1 s). The decode side stays healthy (p50 TPOT 28 ms, decode
  wait for KV ≤ 28 ms). A 1P:1D split is the wrong ratio for this prompt-heavy trace (≈950 input vs
  ≈290 output tokens per request); a 2P:1D layout would need a third GPU.
- **Same-node transport**: the copy-engine SHM path delivers 2.8 GB/s per request at 1x, far below
  E0's 9.9 GB/s all-reduce, because the connector sends 36 separate ~4 MB tensors per request with a
  handshake each. NCCL's default host-copy SHM is slower still (0.67 GB/s at 1x, host-pool run below).
- **Cross-node over TCP**: 0.36–0.48 GB/s effective, the order of E0's measured socket rates. At
  2x (≈310 MB/s of KV demanded on average, in bursts) the prefill phase median rises to 1.0 s and
  goodput to 1.50 req/s; at 4x (≈620 MB/s demanded) the link is saturated and almost nothing meets
  the SLO.
- Decode-side KV waits are negligible in every layout: the proxy sends the decode request only after
  the prefill response, by which time the tensors have arrived.

## What had to change to get here (all runs kept)

| Run (tag) | Configuration | Outcome |
|---|---|---|
| `e4-smoke1b`, `e4-smoke2` | each P/D server saw only its own GPU (`CUDA_VISIBLE_DEVICES=0` / `1`) | every P/D request timed out: NCCL `transport/shm.cc:590 … Cuda failure 217 'peer access is not supported between these two devices'` on the first send, for both `SHM/direct` and `SHM/CE` (the attempted `NCCL_CUMEM_ENABLE=0` override did not reach the engine process in that run: the patches were not yet loaded as a vLLM plugin) |
| `e4-smoke4`, `e4-smoke5` | both GPUs visible to both processes (`0,1` / `1,0`; each uses its first) | SHM works, default and copy-engine, with the connector's own cuMem setting (E0's processes also saw both GPUs); cross-node smoke works. The KV probe recorded nothing yet, hence the plugin |
| `e4-full-pd-tcp` | first full cross-node run | prefill engine died during 4x: `P2pNcclConnector.build_connector_meta` indexes `new_block_ids[0]` of a continuing chunked-prefill request, which is None when that step allocated no block (`TypeError: 'NoneType' object is not subscriptable`) |
| `e4-full-pd-shm`, `e4-full-pd-tcp2` | + fix 1 (None treated as no new blocks), `kv_buffer_size` 2e9 | decode GPU out of memory at 4x in all three P/D layouts. Cause: the engine subtracts a received tensor from its buffer count when the model loads it, but keeps the GPU tensor until the request finishes, so `kv_buffer_size` never bounds GPU copies (0 tensors went to the host pool) |
| `e4-full-pd-shm3`, `e4-full-pd-tcp3` (`hostpool/`) | every received tensor parked in the pinned host pool (`kv_buffer_size` 1 byte) | stable at every level, but the decode listener's extra device-to-host copy per tensor halves the effective transfer rate: goodput at 2x pd-shm-ce 1.993, default SHM 1.837, pd-tcp 0.045 (prefill phase 16 s) |
| `e4-full-pd-shm4`, `e4-full-pd-tcp4` (main rows) | + fix 2 (bytes stay counted until the request finishes), `kv_buffer_size` 2e9 | stable; KV beyond 2 GB on the GPU goes to the host pool as designed |

Both fixes are applied at run time by the plugin (`serving/pd/vllm_entry.py`, switches
`CRUCIBLE_P2P_FIX` and `CRUCIBLE_P2P_BUFFER_FIX`, CPU tests in `serving/pd/tests/`); no installed
file is modified. Host-pool variant table: `e4/hostpool/table.md`.

## Deviations and limits

- Different stack from the manifests: vLLM `P2pNcclConnector` over NCCL SHM/TCP, not SGLang with
  KV over RDMA. No RDMA, InfiniBand or NVLink exists on these nodes. NixlConnector was not tried:
  it needs the NIXL/UCX packages outside Crucible's pins, which was not quick.
- vLLM 0.10.2 is run with two run-time patches (above) and, for `pd-shm-ce`, an override of the
  connector's forced `NCCL_CUMEM_ENABLE=1` to 0 (from the failed workaround attempt; smoke4 shows it
  is not needed once both GPUs are visible, but the main pd-shm-ce run has it).
- SLO recalibrated on the colocated A10 (above); the 4090 numbers in serving-baseline.md are not
  comparable. Layouts 2–4 ran no calibration windows.
- Decode servers use `--gpu-memory-utilization 0.7` (others 0.8) to leave room for the connector's
  2 GB of received KV, so decode has 15–21% fewer KV cache tokens than a colocated server.
- The P/D proxy admits at most 96 requests between prefill start and decode finish (bounds the
  decode's received-KV copies). It mattered only at 4x (pd-tcp admission wait p90 10.5 s).
- All layouts run through the Python proxy, not the Go router; `colo` sends through it too, so the
  client path is the same everywhere.
- Layouts ran on two node types: `colo` and `pd-tcp` on GPU_NV_SM nodes, `dp2` and `pd-shm-ce` on the
  GPU_NV_2M node. Both are A10-24Q vGPUs; colo vs dp2 therefore also changes the node.
- The final P/D configuration (fix 2, `e4-full-pd-shm4` / `-tcp4`) had no separate smoke run; each
  run's first 1x trial served as one, and earlier smokes covered the same code path.
- Each layout ran once (one server start, 3 rounds); rounds are near-identical because the
  workload is deterministic, so ranges show run-to-run noise only. One trace, one model, one
  evaluation window.
- `pd-shm` with NCCL's default SHM was measured only in the host-pool configuration; its 4x
  trials are voided by the late-send rule (3 of 265 late).

## Cost

Ledger sum for E4: 23.16 credits (upper-bound estimates from submit to pool stop; includes the
smokes and the failed and superseded runs; time waiting for the pool lock is not included). Main rows: colo 1.37, dp2 1.96, pd-shm-ce 1.88,
pd-tcp 2.26.

## Raw outputs

`docs/evidence/snowflake/e4/`: `summary.json` and `table.md` (analysis of the four main rows);
per layout (`colo/`, `dp2/`, `pd-shm-ce/`, `pd-tcp/`) the payload's `summary.json`, every replay
JSON (gzipped), `proxy.jsonl.gz`, per-request KV rows `kv-requests.jsonl.gz`, server log excerpts
and the job's `report.json`, `driver.log` and RESULT lines; `environment.json` (nodes, packages,
model fetch and copy); `hostpool/` (the host-pool variant, with `table.md`); `failed/` (summaries,
log excerpts and job records of the failed runs above); `smoke/` (smoke job records). Full tars
(per-tensor probe events, complete server logs) are on the stage under `e4-results/`.
Reproduce the table with
`python -m serving.pd.analyze --slo-from docs/evidence/snowflake/e4/colo --layout colo=docs/evidence/snowflake/e4/colo --layout dp2=docs/evidence/snowflake/e4/dp2 --layout pd-shm-ce=docs/evidence/snowflake/e4/pd-shm-ce --layout pd-tcp=docs/evidence/snowflake/e4/pd-tcp --output /tmp/s.json`.
