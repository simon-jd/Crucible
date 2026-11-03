# Task 12 serving baseline (single RTX 4090) — 2026-10-27

Result: a valid single-GPU baseline. Both traces ran 3 sender-valid 1x calibrations
and 3 rotated rounds of 1x/2x/4x evaluation. No request timed out, and no trial was
voided by the late-send rule. On Azure conv, goodput doubles from 1x to 2x and
collapses at 4x (= the trace's source rate): every request still finishes, but
decode slows past the SLO. BurstGPT stays inside the SLO up to 4x.

## Setup

| | |
|---|---|
| Server | vLLM 0.10.2 OpenAI server, V1 engine, CUDA graphs, chunked prefill, prefix cache off, `--gpu-memory-utilization 0.8 --max-model-len 8192 --max-num-seqs 64 --seed 17` |
| Model | `Qwen/Qwen3-8B-AWQ` rev `4da05a8edb55c6046cce958586c33b61da07bb79` (4-bit AWQ, run by vLLM as `awq_marlin`); KV cache 11.90 GiB = 86,624 tokens |
| Client | Go open-loop replayer (`serving/replay`, `-api openai`), 60 s absolute deadline from schedule, 192 in-flight max, seed 17; prompts are synthetic token IDs from the model's tokenizer allowlist at the trace's lengths; `ignore_eos` + `min_tokens` force the trace's output length |
| Traces | Azure LLM inference 2023 conv: two adjacent 60 s source windows (191 calibration / 265 evaluation requests). BurstGPT_1: first two disjoint 300 s windows with ≥100 requests (105 / 123). Selected with `serving/baseline/select_traces.py` from the pinned prefixes in `serving/baseline/traces/manifest.json` |
| Load | "1x" = Azure at 0.25× source rate, BurstGPT at 1.0× source rate; 2x and 4x scale arrivals. Azure 4x is therefore the trace's real rate |
| SLO | from the three 1x calibration runs: pooled nearest-rank p99 TTFT and p99 TPOT over successful ≥2-token requests. Goodput = requests meeting both / (first send → last completion) |
| Driver | `serving/baseline/vllm_baseline.py` via `scripts/run_serving_baseline_gpu.sh {azure,burst}` (one GPU-lock hold per trace) |

Why this load: the earlier Qwen3-8B BF16 attempt left about 2 GB for KV and timed out about
75% of Azure requests at the source rate. AWQ frees 11.9 GiB for KV. The Azure prefix averages
5.1 req/s, 6.0k prefill tok/s and 1.3k decode tok/s, about the decode limit of one 4090.
That makes the source rate a stress point (4x), not a base. BurstGPT is about 15 times lighter,
so its 1x is the source rate.

## Results

SLOs: Azure TTFT ≤ 460.7 ms and TPOT ≤ 21.51 ms (573 samples). BurstGPT TTFT ≤ 203.5 ms and
TPOT ≤ 11.37 ms (312 samples; 3 calibration requests failed, see below).

Goodput in req/s: median [min, max] over 3 rounds. The pair column is the paired difference
from 1x in the same round.

| Trace | Level | Goodput | Level − 1x | Good / sent | OK | Timeouts | p50 / p99 TTFT ms | p50 / p99 TPOT ms | Output tok/s |
|---|---|---|---|---|---|---|---|---|---|
| Azure | 1x | 1.089 [1.089, 1.089] | — | 263/265 | 265 | 0 | 116 / 453 | 9.5 / 16.1 | 318 |
| Azure | 2x | 2.041 [2.040, 2.041] | +0.952 [+0.951, +0.952] | 252/265 | 265 | 0 | 122 / 602 | 12.3 / 23.1 | 622 |
| Azure | 4x | 0.104 [0.104, 0.104] | −0.985 [−0.985, −0.985] | 7/265 | 265 | 0 | 155 / 827 | 33.2 / 60.6 | 1141 |
| BurstGPT | 1x | 0.401 [0.401, 0.405] | — | 119–120/123 | 123 | 0 | 67 / 243 | 8.0 / 9.0 | 27 |
| BurstGPT | 2x | 0.812 [0.805, 0.812] | +0.411 [+0.400, +0.411] | 120–121/123 | 123 | 0 | 67 / 249 | 8.0 / 9.9 | 53 |
| BurstGPT | 4x | 1.502 [1.502, 1.529] | +1.101 [+1.098, +1.128] | 113–115/123 | 123 | 0 | 65 / 235 | 9.1 / 14.0 | 105 |

Latency cells come from round 0. Every round is in
`summary.json`, and rounds differ by at most a few ms.

- Azure 4x is saturated: output throughput reaches 1,141 tok/s, p50 TPOT rises to 33 ms
  (SLO 21.5 ms), and 258 of 265 requests miss the SLO even though all complete. The knee is
  between 0.5× and 1.0× of the Azure source rate on this GPU.
- Failures: none in evaluation. Each BurstGPT calibration run has one HTTP 400: source row
  `554140,ChatGPT,0,0,…` has zero input tokens, and vLLM rejects the empty prompt. It stays
  in the counts and was excluded from the SLO samples as a failure. Other zero-output rows
  (none in these windows) would be sent as 1-token requests, because vLLM rejects
  `max_tokens=0`.
- Sender validity: 0 or 1 late send (>5 ms) per trial. The worst case is 1/191 = 0.5%,
  under the 1% void line. With ≥105 requests per trial, the rule tolerates one late send.
- Resources: peak owned RSS 5.62 GB (Azure) and 4.84 GB (Burst), under the 16 GB cap. The
  runtime compile cache peaked at 163 MB. Startup took 38 s and 20 s; wall time was 34.6 and 41.6 min.

## Limits

- The rounds are almost identical because the workload is deterministic: same arrivals,
  seeded synthetic prompts, greedy decoding with forced lengths, and an otherwise idle
  server. The [min, max] range measures run-to-run noise of this system, not variation
  between different trace samples. Each condition has one evaluation window per trace.
- p99 SLOs from 191 and 105 calibration requests are close to the sample maximum.
- Deviations from the plan, chosen to fit one 4090: vLLM instead of SGLang (the SGLang
  native overlay never got past a 16 GB-RSS startup; vLLM was already validated here), and
  Qwen3-8B AWQ instead of Qwen3-30B-A3B. The GPU is shared: another tenant held about 3.2 GB.
- 235B, TP+EP and multi-node are paused (未在真实多机上验证).

Raw outputs: `serving-baseline/{azure,burst}-qwen3-8b-awq/`, containing replay JSON for
every trial, `server.log` and `summary.json`.

## Earlier attempts (SGLang native, Qwen3-8B BF16)

These are kept because the W3 draft cites them.

### Native ABI probe and model acquisition preparation

An import-only probe of the SGLang overlay recorded Torch 2.8.0+cu128, Transformers 4.57.1,
sgl-kernel 0.3.16.post5, FlashInfer 0.5.0 and SGLang 0.5.5. It started no model server.
Raw: `raw/task12-baseline-abi-01/abi.json`.

### Full baseline attempt01: stopped by the service-memory guard

SGLang startup reached 16,277,946,368B aggregate owned RSS, above the 16,000,000,000B service
cap, and was stopped before readiness. Raw: `raw/task12-baseline-full-01-guard.log`.

### Diagnostic02 actual result — startup readiness, no generation

A startup-only diagnostic (BF16/context4096/static.80/running8/seed17,
TORCHINDUCTOR_COMPILE_THREADS=1, server warmup skipped, no HTTP requests) reached readiness
with sampled peak RSS 7409397760 bytes. Raw: `raw/task12-startup-memory-02-root-metrics.json`.
