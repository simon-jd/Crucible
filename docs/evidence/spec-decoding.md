# Task 15 speculative decoding: SpecForge EAGLE-3 draft for Qwen3-4B on one RTX 4090 (2026-10-29)

## Setup

- Environment: `.venv-specforge`, built by `scripts/bootstrap_specforge.sh` from the project's CPython
  3.12.13 (`.tools/python`) with uv, SpecForge source at e3ab28d (`e3ab28d681df457f2a04e59dfac8117c2c4cbfef`)
  (api.github.com tarball, sha256 `a0908c06616fa5f6dc9f87ad29bb66cbcd1b6532185a754e8740442e9d961752`)
  and its pinned pyproject: torch 2.13.0+cu130, transformers 5.12.1, sglang 0.5.18. All 226
  packages came from pypi.org / files.pythonhosted.org; no other index was needed
  ([freeze](spec-decoding/env/specforge-freeze.txt)). Size: 9,079,611,662 bytes (`du -sb`).
- Target: Qwen/Qwen3-4B BF16 at revision `1cfa9a7208912126459214e8b04321603b3df60c` (shared HF cache,
  every file checked against the revision's tree hashes by `scripts/fetch_qwen3_4b.sh`). The plan's
  Qwen3-30B-A3B does not fit one 4090 in BF16, and Qwen3-8B BF16 leaves about 3 GB for draft, KV and
  CUDA graphs at `--mem-fraction-static` 0.75, too little for batch 64.
- Data: UltraChat 200k (HuggingFaceH4/ultrachat_200k at `8049631c405ae6576f93f445c6b8166f76f5505a`,
  shard sha256 checked). 1,200 first-turn prompts from `train_sft`; Qwen3-4B regenerated its own
  answers (SpecForge `regenerate_train_data.py`, greedy, thinking disabled, max 640 tokens):
  1,200/1,200 succeeded ([summary](spec-decoding/data/regen-summary.txt)). 128 held-out prompts
  from `test_sft`, disjoint by normalized-text hash from the whole train shard
  ([manifest](spec-decoding/data/prompt-split-manifest.json), [prompts](spec-decoding/data/eval_prompts.jsonl)).
- Features: SpecForge `prepare_hidden_states.py` (EAGLE-3, max length 1024) for the first 240
  regenerated conversations: 4,206,741,801 bytes, because records are padded to the capture batch
  (16) ([summary](spec-decoding/data/features-summary.txt)).
- Draft: upstream `configs/qwen3-4b-eagle3.json` with `draft_vocab_size` 32000 → 16000 and
  `intermediate_size` 9728 → 6144 (150M instead of 218M trainable parameters). The reason is disk:
  after the features the project had 3.11 GB of its 28 GB share left, and one checkpoint costs
  about 14 bytes per trainable parameter (bf16 weights, fp32 master, two Adam moments).
- Training: `specforge train` (`spec/train_eagle3.sh spec/configs/qwen3-4b-eagle3-t15.yaml`), offline
  colocated, 10 epochs, batch 1, lr 1e-4, TTT length 7, flex attention: 2,400 steps in one GPU job
  15:12:03–15:17:56 ([steps](spec-decoding/train/train-steps.log)). Checkpoint 2,096,951,625 bytes;
  SGLang export (`specforge export --to sglang`) 299,801,440 bytes.
- Benchmark (`spec/bench_tpot.py`, one GPU job 15:18:47–15:45:53): SGLang 0.5.18 server,
  `--mem-fraction-static 0.75`, CUDA graphs to batch 64. EAGLE-3 runs add `--speculative-num-steps 3
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`. 2 rounds: base, eagle3, base, eagle3,
  with a server restart each time. Each run sends the same 128 held-out prompts, greedy, max 256
  new tokens, closed loop with 1, 4, 16 and 64 requests in flight. TPOT per request = (last
  chunk − first chunk) / (tokens − 1); the table gives the median over 128 requests per run.
  Acceptance length = 1 + Σ`spec_num_correct_drafts` / Σ`spec_verify_ct` (SGLang per-request counters).
- Machine: 32 cores, load average 2.1–5.7 during the benchmark; SponsorAtlas and Keel sessions
  were running at the same time and shared the CPU (the GPU lock was ours for each job)
  ([load](spec-decoding/bench/machine-load.txt)).

## Results

Two rounds per mode; values are round 1 / round 2. All 1,024 requests per mode completed, 0 failed
([summary](spec-decoding/bench/summary.json), raw per-request rows in `spec-decoding/bench/*-r*.json`).

| Batch | Base TPOT (ms) | EAGLE-3 TPOT (ms) | TPOT speedup | Output tok/s ratio | Acceptance length |
|---|---|---|---|---|---|
| 1 | 9.34 / 9.34 | 8.28 / 8.26 | 1.128 / 1.130 | 1.144 / 1.144 | 1.423 / 1.421 |
| 4 | 10.20 / 10.18 | 8.89 / 8.73 | 1.147 / 1.166 | 1.170 / 1.169 | 1.420 / 1.427 |
| 16 | 11.05 / 11.05 | 10.37 / 10.34 | 1.066 / 1.069 | 1.091 / 1.088 | 1.419 / 1.424 |
| 64 | 13.55 / 13.84 | 17.56 / 17.25 | 0.772 / 0.803 | 0.736 / 0.756 | 1.421 / 1.424 |

- Acceptance length is about 1.42 at every batch size: on average 0.42 of the 3 drafted tokens
  are accepted per verification step.
- Speculative decoding helps at batch 1–16 (TPOT 6–14% lower) and hurts at batch 64 (TPOT 25–30%
  higher, throughput 24–26% lower): at 64 the target is no longer memory bound, so verifying 4 tokens
  per step costs more than 1.42 accepted tokens save.
- Output token totals per run (128 requests) are 30,505–30,649 in both modes, within 0.5% of each
  other: greedy outputs are nearly but not exactly identical between the two servers.

## Phase 7 (2026-11-01): full-size draft, 60,697 regenerated conversations

Details and raw outputs: [phase7/README.md](spec-decoding/phase7/README.md).

- **Draft and data:** SpecForge's standard draft config (vocab 32000, intermediate 9728). It was
  trained offline in 105 batches of 576 conversations, with features deleted after each batch,
  in one SpecForge run of 60,480 steps. The data is 60,697 UltraChat conversations regenerated by
  Qwen3-4B (thinking off, greedy). The same 128 held-out prompts were used; all training rows
  are disjoint from them by normalized-text hash.
- **Same harness as above:** 2 rounds, a server restart per run, 0 failed requests.

| Batch | Base TPOT (ms) | EAGLE-3 TPOT (ms) | TPOT speedup | Tok/s ratio | Acceptance | Public AngelSlim draft: speedup / acceptance |
|---|---|---|---|---|---|---|
| 1 | 9.31 / 9.32 | 5.66 / 5.65 | 1.646 / 1.649 | 1.657 / 1.657 | 2.159 / 2.159 | 1.629 / 1.633; 2.11 |
| 4 | 10.15 / 10.16 | 6.07 / 6.07 | 1.670 / 1.673 | 1.680 / 1.683 | 2.152 / 2.152 | 1.675 / 1.669; 2.11 |
| 16 | 11.02 / 11.03 | 7.26 / 7.22 | 1.518 / 1.528 | 1.537 / 1.551 | 2.156 / 2.160 | 1.507 / 1.509; 2.11 |
| 64 | 13.62 / 13.45 | 11.78 / 11.87 | 1.157 / 1.133 | 1.062 / 1.062 | 2.165 / 2.164 | 1.141 / 1.118; 2.11 |

- **Acceptance:** 1.42 → 2.16. Speculative decoding now helps at batch 64 too (TPOT 12–14%
  lower, throughput 6% higher), where phase 6 was 20–23% slower than no-spec.
- **Public draft:** [AngelSlim/Qwen3-4B_eagle3](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3)
  (Apache-2.0) is within 1–3% of ours on these UltraChat prompts.
- **Declared grid** (batch 1 and 64 only):
  - Tree drafting (topk 4, 8 draft tokens) is best at batch 1, with a 1.88–1.90× TPOT speedup.
  - A chain of 2 steps is best at batch 64, with a 1.29× speedup.
  - Longer drafts are slower than no-spec at batch 64.
- **GPU time:** about 9.3 h on one 4090, with other sessions sharing the machine (32 cores).

## Limits

These are phase 6's limits; phase 7 above removed the first one.


- The draft saw 240 conversations (about 0.15M tokens), not the 100k+ samples of upstream recipes;
  that is why acceptance is 1.42. Disk set both numbers: 4.2 GB of features and a reduced draft
  fit the 28 GB share. More data needs the feature directory `.cache/t15/features` deleted (SpecForge
  skips existing feature files and `rm` is not in this project's tool permissions), or a larger share.
- Training accuracy in the step log is on the training data; there is no held-out offline eval.
- One target model, one dataset family (UltraChat), one draft shape (chain of 3), 256 output tokens.
- The plan's Qwen3-30B-A3B target and multi-GPU serving are out of scope for one 4090.

## Earlier CPU phase

The counter convention in `spec/eval_acceptance.py` and its synthetic CLI receipts are unchanged:
[CPU log](raw/task15-cpu-python-green.log), [CLI receipts](raw/task15-cpu-cli-commands.json),
[schema/API](../../spec/README.md). `spec/bench_tpot.py` uses its `acceptance_length` for the real runs.
