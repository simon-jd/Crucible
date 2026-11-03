# Task 11: math GRPO on Qwen2.5-0.5B-Instruct, one RTX 4090 — 2026-10-27

## Update (phase 5, 2026-10-28): full held-out set passes 2 SE

Result: the same async k=1, `--reward eos`, group 8 training (100 steps, 6,400 samples,
seeds 17/18/19), scored on the full held-out pools instead of the phase-4 subsets.
Pooled GSM8K pass@1 moves 0.4503 → 0.4874, **+3.71 ± 0.67 pp (5.51 SE)**.
Pooled MATH-500 moves 0.2467 → 0.2832, **+3.65 ± 0.98 pp (3.72 SE)**.
Both clear the pre-registered bar (pooled delta ≥ 2 SE). Every seed is positive.
This is the phase-4 configuration on a larger problem set, not a new training recipe.

Group 16 (4 prompts/step, same 6,400 samples, seeds 17/18/19) also clears 2 SE:
GSM8K **+4.45 ± 0.71 pp (6.27 SE)**, per seed +4.62 / +3.98 / +4.74 pp;
MATH-500 **+2.61 ± 0.92 pp (2.82 SE)**, per seed +3.91 / +2.92 / +0.99 pp.
GSM8K truncations after training are 78 / 61 / 71. MATH-500 truncations are
511 / 526 / 467. Train reward stays nearly flat (0.467 → 0.431, 0.436 → 0.444,
0.461 → 0.486), unlike group 8. Training truncations per 64 samples fall from
14.0 / 15.7 / 15.5 to 4.1 / 5.0 / 6.8. Run wall is 1,216.4 / 1,227.7 / 1,200.3 s.
Peak CUDA allocation is 14,810,452,480 / 14,806,077,952 / 14,803,452,416 bytes.
All three exited 0.

Held-out size: all 1,319 GSM8K test problems and all 377 gradable MATH-500 problems
(phase 4 used 200 and 100). Pre-training eval is identical across the three seeds.
Pooling is `rl/local/summarize.py --pool` on `docs/evidence/rl-grpo-p5/full-async-k1-s{17,18,19}`.
Per-seed GSM8K deltas are +3.45 / +5.10 / +2.58 pp. Per-seed MATH-500 deltas are
+3.58 / +4.51 / +2.85 pp. GSM8K eval truncations after training are 79 / 53 / 95
of 5,276 samples (before: 267). MATH-500 truncations after training are 507 / 542 / 475
of 1,508 samples (before: 715). Train reward, first 10 steps to last 10, is
0.409 → 0.633, 0.405 → 0.655 and 0.412 → 0.656. Training truncations per 64 samples
fall from 16.3 / 14.4 / 14.3 to 2.1 / 2.4 / 4.0. Median step wall is 5.49 / 5.59 / 5.47 s,
rollout 1.76 / 1.82 / 1.77 s, learner 2.63 / 2.66 / 2.62 s, weight sync 0.018 s,
GPU-idle share 0.19, stale drops 0. Run wall is 1,193.3 / 1,184.1 / 1,183.6 s.
Peak CUDA allocation is 14,795,768,320 / 14,799,377,408 / 14,795,910,656 bytes.
Each run exited 0 (`rl-grpo-p5/machine-load.txt`). Machine load average at the
boundaries was 5.81–10.81. Raw outputs and `summary.json` / `pooled.json` are in
`docs/evidence/rl-grpo-p5/`.

## Update (phase 4, 2026-10-27/28): EOS penalty, 3 seeds, group and clip ablations

Result: an EOS penalty (`--reward eos`: a completion that hits max tokens without EOS
scores 0) removes the "box, then run on" exploit in all 10 runs. Pooled over 3 seeds,
async k=1 improves held-out pass@1 by +3.3 ± 2.0 pp on GSM8K and +3.0 ± 2.2 pp on MATH-500.
All 3 seeds improve on both sets. Neither change reaches 2 SE (1.6 and 1.4 SE), so a
significant improvement is still not shown. Sync k=0 with the same penalty pools to
+0.8 ± 1.8 and +0.75 ± 2.3 pp. The group and clip ablations (one seed each) all fall
within the seed-to-seed spread of the reference configuration.

Setup is as below (100 steps, 6,400 trained completions per run, same data, same fixed
eval seeds), plus `--reward eos`. Seeds 17/18/19 change rollout sampling and vLLM
seeding only. Pre-training eval is identical across runs (GSM8K 0.426, MATH-500 0.2875).
Pooling: per problem, average (after − before)/n over seeds. The SE is over the 200 (or
100) problems of that average (`rl/local/summarize.py --pool`), so it removes sampling
noise but not problem-set spread. Command: `bash scripts/run_grpo_phase4.sh` (each run takes
and releases the GPU lock itself). Machine: 32 cores, load average 1.8–16.1 at run
boundaries (`rl-grpo-p4/machine-load.txt`), with Keel's GPU matrix and PathWatch running.

| Config (all `--reward eos`) | Seeds | GSM8K Δ pp (pooled ± SE) | per seed | MATH-500 Δ pp (pooled ± SE) | per seed | GSM8K eval truncated (of 800) |
|---|---|---|---|---|---|---|
| async k=1, group 8, clip 0.2 | 17/18/19 | +3.3 ± 2.0 | +3.1 / +2.0 / +4.75 | +3.0 ± 2.2 | +1.25 / +5.25 / +2.5 | 17 / 16 / 11 |
| sync k=0, group 8, clip 0.2 | 17/18/19 | +0.8 ± 1.8 | +2.1 / −0.75 / +1.1 | +0.75 ± 2.3 | +0.75 / +2.25 / −0.75 | 9 / 9 / 1 |
| async k=1, group 4 (16 prompts/step) | 17 | +0.5 ± 2.2 | — | +0.75 ± 2.5 | — | 9 |
| async k=1, group 16 (4 prompts/step) | 17 | +4.9 ± 2.5 | — | +1.75 ± 2.6 | — | 16 |
| async k=1, clip 0.1 | 17 | +1.1 ± 2.4 | — | +3.75 ± 2.8 | — | 17 |
| async k=1, clip 0.3 | 17 | +1.25 ± 2.2 | — | +0.5 ± 2.6 | — | 7 |
| phase 3, no penalty: async k=1 | 17 | +1.75 ± 2.2 | — | +2.25 ± 3.0 | — | 7 |
| phase 3, no penalty: sync k=0 | 17 | −1.9 ± 2.5 | — | 0.0 ± 2.6 | — | **703** |

- The exploit: with the penalty, sync k=0 ends with 1–9 truncated GSM8K eval completions,
  against 703 without it. Training truncations per 64 samples fall from 13.5–15.7 (first 10
  steps) to 1.8–2.3 (last 10) in every sync seed. The penalized async seeds go from
  14.6–16.6 to 2.2–5.0.
- Train correctness reward rises from 0.38–0.43 to 0.58–0.63 (first/last 10 steps) in
  every group-8 run, and from 0.42 to 0.60 at group 4. Group 16 is the exception: it stays flat (0.455 → 0.456) and still
  has the largest single GSM8K gain. With one seed, that cannot be separated from noise.
- Ablations are single seeds. The reference config's own seeds span +2.0 to +4.75 pp on
  GSM8K, so no group or clip setting is shown to be better or worse.
- Cost per step is unchanged by the penalty: median wall 5.46–6.07 s, rollout 1.74–1.92 s,
  learner 2.60–2.83 s, weight sync 0.018–0.019 s, GPU-idle share 0.19–0.20, 0 stale drops,
  peak CUDA allocation 14.8–14.9 GB, 669–757 s per run.
- The first group-4 attempt was stopped mid-run by this session's tool time limit, not by
  a fault. Its partial output was discarded and the run was repeated from scratch.
- Raw outputs: `rl-grpo-p4/<run>/` (`steps.jsonl`, `summary.json`, `eval-{before,after}.jsonl.gz`),
  `rl-grpo-p4/summary.json` (per run), and `rl-grpo-p4/pooled.json` (per config).

## Phase 2 runs (no EOS penalty)

Result: end-to-end GRPO training runs on one GPU. Rollout (in-process vLLM) and the FSDP2
learner time-share the card, and weights move learner → vLLM in memory every step. On three
6,400-sample runs, training reward rises in every run. Held-out pass@1 moves by at most
+3.3 pp, and no run's change is distinguishable from zero (|Δ| < 1.3 SE). The sync k=0 run
learned a reward exploit: it writes the boxed answer and then keeps generating without EOS.

## Setup

| | |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` rev `7ae557604adf67be50417f59c2c2f167def9a775` |
| Data | GSM8K train (HF parquet rev `740312ad…`, LFS SHA256 checked) and 1,024 seeded training prompts. Held-out: 200 GSM8K test problems and 100 MATH-500 problems (rev `6e4ed1a2…`, git-blob SHA1 checked), restricted to the 377/500 whose reference the bounded verifier can grade. 0 duplicates, 0 train/held-out overlap ([identity](rl-grpo/data-identity.json)) |
| Reward | `rl.verifiers.math` on the last `\boxed{}` (1/0); no format, length or EOS terms |
| Training | 100 steps × 8 prompts × group 8 = 6,400 trained completions per run; AdamW lr 3e-6, clip 0.2, std-scaled advantages, temperature 1.0, max 384 new tokens, seed 17 |
| Engine | `rl/local/grpo_run.py` via `scripts/run_grpo_gpu.sh`: vLLM 0.10.2 (V1, uni executor, 15% of GPU memory, eager) plus the `rl.learner.train` FSDP2 learner (FP32 master weights, BF16 compute, frozen BF16 reference), on vLLM's world-1 NCCL group |
| Eval | pass@1 = mean c/n with n=4 samples per problem (temperature 0.6, top-p 0.95, max 512 tokens). Fixed per-problem seeds, identical before/after and across runs. SE is over per-problem paired differences |

## Results

| Run | GSM8K before → after (Δ ± SE) | MATH-500 before → after (Δ ± SE) | Train reward first/last 10 steps | GSM8K eval truncated (of 800) |
|---|---|---|---|---|
| async k=1, KL 0.001 (base) | 0.426 → 0.444 (+1.8 ± 2.2) | 0.288 → 0.310 (+2.3 ± 3.0) | 0.41 → 0.55 | 50 → 7 |
| sync k=0, KL 0.001 | 0.426 → 0.408 (−1.9 ± 2.5) | 0.288 → 0.288 (0.0 ± 2.6) | 0.40 → 0.59 | 50 → **703** |
| async k=1, KL 0 | 0.426 → 0.438 (+1.1 ± 2.4) | 0.288 → 0.320 (+3.3 ± 3.1) | 0.38 → 0.65 | 50 → 9 |

Per-step cost (median [min, max] over 100 steps; the three runs are nearly the same):

| | async k=1 | sync k=0 | async KL 0 |
|---|---|---|---|
| Step wall (s) | 5.93 [5.33, 8.84] | 5.84 [5.58, 6.41] | 5.74 [5.25, 6.45] |
| Rollout / learner (s) | 1.84 / 2.95 | 1.91 / 2.82 | 1.85 / 2.76 |
| Weight sync (s) | 0.019 [0.018, 0.202] | 0.018 [0.017, 0.022] | 0.018 [0.017, 0.021] |
| GPU-idle share | 0.18 [0.11, 0.22] | 0.18 [0.16, 0.23] | 0.19 [0.17, 0.23] |
| Stale samples dropped | 0 | 0 | 0 |
| Run wall (s) | 752 | 731 | 705 |

- GPU-idle share means step time outside rollout, learner and sync phases, which is mostly
  math grading in CPU subprocesses. Another tenant's GPU use is not visible to this process.
  Peak CUDA allocation is 14.8 GB.
- Weight sync means a BF16 copy of every policy parameter (about 494M) into vLLM's model plus a
  prefix-cache reset, all on the GPU, with no disk. Proof that it took effect: with fixed
  eval seeds, post-training samples differ from pre-training samples.
- async k=1 trains every batch exactly one version stale (k=1 admits it), so 0 drops is
  expected. On one serial GPU, async cannot overlap rollout with learning; it only changes
  data staleness, so step times match sync.

## Reward hacking caught

The `truncated` flag in the sync k=0 run climbs from 15.3 to 55.4 per 64 samples
(first to last 10 steps), and mean completion length rises from 297 to 370 tokens. After
training, 703/800 GSM8K eval completions hit the 512-token limit, and 685 of those already
contain a `\boxed{}` answer. The policy states an answer and then runs on, often off-topic.
The reward grades only the last box and never penalizes that. Pass@1 barely moved
(the verifier accepts a run-on whose last box is right), but the outputs are unusable. The
two async runs went the other way: truncation fell (16.6 → 3.1 and 17.2 → 1.9 per 64).
Each schedule ran with one seed, so this does not show that sync causes the exploit.
"multiple_boxed_answers" (hedging) stayed rare: 11 / 74 / 17 per 6,400.

## Not done / limits

- One seed per configuration, and a small budget (6,400 samples); the improvement is not
  established.
- The group-size and clip ablations were not run.
- No verl baseline: verl is not installed, and its environment would exceed the 14 GB
  project disk share (the project is at 13.8 GB).
- Code reward is blocked on a working sandbox (see STATUS blockers).
- MATH-500 is the gradable subset (377/500), a disclosed selection.

Raw outputs: `rl-grpo/{async-k1,sync-k0,async-k1-kl0}/` (`steps.jsonl`, `summary.json`,
`eval-{before,after}.jsonl.gz` with every eval completion), plus `rl-grpo/summary.json` and
`rl-grpo/data-identity.json`.
