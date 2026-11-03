# Report checker and reproduction provenance

This file adds checker/reproduction facts only. It does not create scientific results.
The [bounded extraction receipt](raw/task16-provenance.json) records selected source hashes,
stdlib-only metadata/JSON extraction and last-path Git commits. Original evidence is unchanged.

## Checker runtime

Actual CPU execution environment: Python 3.12.13, pytest 8.4.2, Ruff 0.17.2.
These are measured local tool identities, not the model-execution runtime.
Source: version commands and stdlib distribution metadata in the extraction receipt.

## Accepted functional source lineage

The custom-op implementation/evidence last-path commit is
`41c5f12a24c2c7cd1dfe1a8f61d6b90c574a9d38`;
[original evidence](fused-logprob.md) and [frozen source hashes](raw/task10-final-source.sha256).
The weight-sync evidence last-path commit is
`8d3c2817559be48ebf2a1badcb14932c22863f51`;
[original evidence](weight-sync.md) retains separate source/rollback/confirmation runs.
The CPU comparison fix last-path commit is
`cbe766cda35f3715119069e3a5d53b09aeab754a`;
[original evidence and scoped approval](pd-disaggregation.md).
These commit identities identify source/evidence lineage, not full experiment acceptance.

## Learner fixture configuration

The [original guarded output](raw/task10-gpu-final.txt) records the actual fixed fixture.
Its JSON sampling values are seed `1010`, temperature `1.0`, top_p `1.0`, top_k `-1`,
max_tokens `8`, max_model_len `128`. The extraction receipt preserves these exact JSON values.
The original model revision is `7ae557604adf67be50417f59c2c2f167def9a775`.
This is a short controlled performance fixture, not held-out quality evaluation.
The [original measurement prose](fused-logprob.md#actual-learner-matched-measurements)
separately records seed1010, LR1e-5, microbatch2, max_length128 and world-size1 FSDP2.
The original timing table contains already-reported millisecond values; reports reuse those
literals and do not invent rounding or convert raw seconds to new unsupported claims.

## Serving identity and pending gates

The [verified acquisition manifest](raw/task12-model-acquisition-02.json) identifies
`Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`.
Its model plan digest is `067ca7df8c06e738b38db2e73634e4d1a4ac411c9ebf59cb10db68e557d06a37`;
runtime-lock digest is `2183ac96fead291b5ece21dfe04a4da0e68b28daad5cc97acb7adb7ecd0b6081`.
The rejection-evidence last-path commit is
`0def66fde399123e9d7d48a40a3ddb2267e820dc`.
[Original launch-block record](serving-baseline.md#full02-v3-launch-approval-block--no-native-process)
records refusal before CreateProcess; it is not an executed native trial.
[Original startup result](serving-baseline.md#diagnostic02-actual-result--startup-readiness-no-generation)
is readiness only; the complete BF16 serving baseline is still unaccepted.
[Original CPU comparison evidence](pd-disaggregation.md) retains pending actual FP8/BF16,
quality, P/D, cross-node and 235B gates. Physical work remains paused: 未在真实多机上验证.
The task brief retains missing async training/held-out/synchronous-baseline and EAGLE/fault
results; no result for those experiments is introduced here.

## Checker command

The actual CPU report-check command is:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/check_report_numbers.py reports/w2-async-grpo.md reports/w3-disaggregated-serving.md reports/repro/README.md
```

This is a checker command identity only, never a native experiment command.

## Startup literal provenance

The [original diagnostic summary](raw/task12-startup-memory-02-summary.json) records
`max_sampled_aggregate_stat_rss_bytes` exactly as JSON integer `7409397760`.
The [original failed attempt](serving-baseline.md#full-baseline-attempt01-stopped-by-the-service-memory-guard)
reports the service ceiling with literal `16,000,000,000B`, attached to the word unchanged
in its original prose. This section separates typography only and introduces no new
measurement, conversion, cap or result. The existing failure remains a failure.

## Single-GPU run identities

Facts for the phase-2 single-GPU runs (T11, T12, T13). Each commit added the driver code
together with its raw outputs; the runs record no commit of their own.

| Task | Commit | Command | Seed |
|---|---|---|---|
| T12 baseline | `bc749482dfc1a4a92a01c79cdf20165784cb6349` (bc74948) | `bash scripts/run_serving_baseline_gpu.sh azure\|burst <out>` | 17 |
| T11 GRPO | `b49326bee0f7e0afd17f46e84995e1bf50d4dba4` (b49326b) | `bash scripts/run_grpo_gpu.sh <out> [--staleness 0\|1 --kl X]` | 17 |
| T13 FP8/BF16 | `4f834c7babd964bfe6588276616549afd3635219` (4f834c7) | `bash scripts/run_fp8_compare_gpu.sh <out> [--quality-only]` | 17 |

- Models (shared HF cache, fixed revisions): `Qwen/Qwen2.5-0.5B-Instruct` rev
  `7ae557604adf67be50417f59c2c2f167def9a775`; `Qwen/Qwen3-8B-AWQ` rev
  `4da05a8edb55c6046cce958586c33b61da07bb79`; `Qwen/Qwen3-8B` rev
  `b968826d9c46dd6066d109eabc6255188de91218`.
- Hardware and runtime: one NVIDIA GeForce RTX 4090 (24 GB, shared with another project),
  driver 595.99.02, CUDA 12.8, Torch 2.8.0, Triton 3.4.0, Transformers 4.55.4, vLLM 0.10.2,
  Python 3.12.13, Go 1.27.3. Host-native, no container, so no image digest.
- Environment hashes (`scripts/env_identity.py`, captured 2026-10-27 after the runs, with no
  installs since): `.venv-ml` 146 distributions, sha256
  `e4feddb69cdcb3572341a4edacd14410bb3f4e6053329ecb3c1924db774b8bef`
  ([list](t16-reports/venv-ml-distributions.txt)); `.venv` 10 distributions, sha256
  `71a9efac2614738fe2f110a67bd0e8854a71a1bbdb660ac6d8cb40fa8a701e04`
  ([list](t16-reports/venv-distributions.txt)).
- Configs: vLLM server args are in each run's `summary.json` / `session.json`
  (`--gpu-memory-utilization 0.8 --max-model-len 8192 --max-num-seqs 64 --seed 17`); GRPO
  arguments are in `rl-grpo/<run>/summary.json` under `args`.
- Data prep for T11: `python -m rl.local.prepare_data --root .cache/datasets --output
  .cache/t11/data.json`; dataset hashes in `rl-grpo/data-identity.json`.
- PDF build: `scripts/build_report_pdf.py` (Pillow 11.3.0, markdown-it-py 4.0.0, Noto Sans CJK).
  Commands: `.venv-ml/bin/python scripts/build_report_pdf.py reports/w2-async-grpo.md
  reports/pdf/w2-async-grpo.pdf` and the same for `reports/w3-disaggregated-serving.md` →
  `reports/pdf/w3-disaggregated-serving.pdf`.
