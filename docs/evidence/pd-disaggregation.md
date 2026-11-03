# Task 13: FP8 vs BF16 on one RTX 4090 (P/D paused) — 2026-10-27

Result: on the same Qwen3-8B checkpoint and the same load, vLLM online FP8 cuts p50 TPOT from
24.6 to 15.5 ms (−37%) and p50 TTFT from 128 to 89 ms, and triples KV capacity. Held-out
accuracy changes by −1.0 pp on GSM8K and +0.3 pp on MMLU-Pro, both within sampling noise.
At this load, throughput and SLO goodput barely move, because BF16 already serves almost
every request inside the SLO.

## Setup

| | |
|---|---|
| Model | `Qwen/Qwen3-8B` rev `b968826d9c46dd6066d109eabc6255188de91218` for both variants |
| Baseline / candidate | vLLM 0.10.2 BF16 weights / the same checkpoint with `--quantization fp8` (online dynamic FP8 of weights and activations; Ada FP8 GEMM) |
| Common server args | `--gpu-memory-utilization 0.8 --max-model-len 8192 --max-num-seqs 64 --seed 17 --no-enable-prefix-caching --enforce-eager` (eager for both: BF16 weights leave little memory, and the project is about 0.2 GB under its disk share, so no compile cache) |
| Load | Task 12 Azure conv evaluation window (265 requests) at 0.25× source rate, open loop, 60 s deadline |
| Plan | AB, BA, AB (6 runs), with a server restart at each precision change. SLO fixed from 3 BF16 calibration runs on the disjoint calibration window: TTFT ≤ 1500.4 ms, TPOT ≤ 40.08 ms (p99) |
| Quality | greedy, `enable_thinking=False`. GSM8K: the 200 Task 11 held-out test problems, `numeric-final-v1`. MMLU-Pro: 300 seeded test questions (rev `b189ec76…`), `choice-letter-v1`, output limited to the valid option letters via vLLM `guided_choice` |
| Tools | `serving/eval/fp8_session.py` via `scripts/run_fp8_compare_gpu.sh`; `serving/eval/fp8_bundle.py` builds the schema-2 bundles; `serving/eval/accuracy.py` produces the reports |

## Results

Median [min, max] over 3 runs per precision. All 6 runs are sender-valid, with 265/265 OK
and 0 timeouts in each.

| | BF16 | FP8 | FP8 − BF16 |
|---|---|---|---|
| Output throughput (tok/s) | 311.02 [311.00, 311.04] | 315.35 [315.33, 315.37] | +1.4% |
| SLO goodput (req/s) | 1.065 [1.065, 1.065] | 1.088 [1.088, 1.088] | +2.2% |
| p50 / p99 TTFT (ms, run 1) | 128 / 485 | 89 / 329 | −30% / −32% |
| p50 / p99 TPOT (ms, run 1) | 24.6 / 31.8 | 15.5 / 20.9 | −37% / −34% |
| KV cache (tokens) | 22,192 | 68,016 | ×3.1 |
| GSM8K accuracy (200) | 0.920 | 0.910 | −0.010 |
| MMLU-Pro accuracy (300) | 0.377 | 0.380 | +0.003 |

- Throughput follows the offered load here (0.25× source rate). Both variants finish every
  request; BF16 misses the BF16-derived SLO on 2 of 265 requests per run. The latency gap is
  the real difference; it would turn into a goodput gap at higher load (Task 12 shows the
  knee between 0.5× and 1.0× of source for AWQ with CUDA graphs).
- For accuracy, one standard error is about ±1.9 pp on GSM8K and ±2.8 pp on MMLU-Pro
  (binomial). Neither delta is distinguishable from zero. The accuracy denominator includes
  every item; unparseable answers count as wrong (GSM8K: 5 for BF16, 8 for FP8, mostly a
  trailing extra `####`).

## Quality attempt 1 (kept, not used)

The first quality pass, in `raw/quality-*.jsonl`, used an unconstrained MMLU-Pro prompt with
max 8 tokens. Qwen3 answered like `C. <option text>`, so 256 of 300 BF16 answers were
truncated and could not be parsed (accuracy 0.057). Its GSM8K template `'#### <number>'` was
sometimes echoed literally. I fixed the prompts (MMLU-Pro `guided_choice`, a concrete
`#### 42` example), applied them identically to both precisions, and reran quality alone in
`quality-v2/`. The converter checks that the rerun's server arguments equal the performance
session's.

## Limits and paused parts

- One GPU, one model size, one load level; eager mode for both variants, so absolute latency is
  higher than with CUDA graphs (compare Task 12's AWQ baseline).
- Online FP8 uses per-tensor dynamic scales, not a calibrated FP8 checkpoint. A pre-quantized
  FP8 checkpoint was not downloaded: the shared model cache has about 8 GB left of 40 GB.
- P/D disaggregation (1P:1D, 2P:1D, cross-node, KV over RDMA) and 235B are paused until
  multi-node GPUs are budgeted: 未在真实多机上验证. The inert P/D layouts remain in
  `serving/deploy/pd-*.yaml`.

Raw outputs: `fp8-vs-bf16/raw/` (calibration, six run replays, server logs, session.json),
`fp8-vs-bf16/quality-v2/`, `fp8-vs-bf16/bundle-{gsm8k,mmlu_pro}.json`, and
`fp8-vs-bf16/report-{gsm8k,mmlu_pro}.json` (accuracy.py output, `comparison_eligible: true`).
