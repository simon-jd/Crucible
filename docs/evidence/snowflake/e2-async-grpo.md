# E2: async GRPO with rollout and learner on separate GPUs (Snowflake A10, 2026-11-04)

Result: with the learner on one A10 and vLLM rollout on another, really overlapped
and synced over NCCL every step, a 100-step run trains its 6,400 samples in
**741 s on one node (8.64 samples/s)** and **933 s across two nodes (6.86 samples/s)**,
against **1,207 s (5.30 samples/s)** for the unchanged time-shared loop on one A10:
**1.63×** and **1.29×**. The two-GPU loops are now learner-bound: the learner GPU is
idle 1% of the training window, the rollout GPU 21–27%. Held-out accuracy: GSM8K
clears the 2 SE bar in all three layouts; MATH-500 clears it in the two-GPU layouts
(2.22 and 3.08 SE) but **not in the time-shared A10 reference (+1.79 ± 1.01 pp,
1.76 SE)**. No pairwise layout difference in final accuracy reaches 2 SE. A rerun of (c) with
E0's tuned socket settings ([below](#layout-c-rerun-with-tuned-sockets)) syncs in 1.9–2.2 s
instead of 2.8–3.2 s and trains in 860 s on average (1.40× over (a)).

Hardware: (a) one A10 24 GB (vGPU "A10-24Q") of a Snowflake GPU_NV_SM node. (b)
Snowflake GPU_NV_2M: 2× A10 24 GB (vGPU) in one node, no NVLink/P2P; learner on GPU0,
vLLM on GPU1, NCCL `SHM/direct/direct` with `NCCL_SHM_USE_CUDA_MEMCPY=1`. (c) 2 Snowflake
GPU_NV_SM nodes (1× A10 24 GB each) over TCP, NCCL `NET/Socket` with
`NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4`. torch 2.8.0, vLLM 0.10.2,
transformers 4.55.4 from Snowflake's PyPI proxy. The 4090 numbers in the README are
context only (different GPU, shared host).

## Setup (kept from the README)

Qwen2.5-0.5B-Instruct (rev `7ae5576…`), 100 steps × 8 GSM8K prompts × group 8 = 6,400
trained completions, the EOS rule (`--reward eos`), async k=1, AdamW lr 3e-6, clip 0.2,
KL 0.001, max 384 new tokens, microbatch 2, vLLM at 15% memory, seeds 17/18/19, the same
data file (`data_sha256 dbdf1a1b…`, the 4090 phase-5 file). Held-out: all 1,319 GSM8K
test and the 377 gradable MATH-500 problems, pass@1 with 4 samples, temperature 0.6,
top-p 0.95, 512 tokens, fixed per-problem seeds; paired per-problem SE pooled over
seeds (`rl/local/summarize.pooled_delta`), bar = pooled delta ≥ 2 SE.

- (a) `rl/local/grpo_run.py` unchanged (vLLM and the FSDP2 learner time-share one GPU).
- (b), (c) `rl/local/async_grpo.py` (new): learner and rollout processes. The rollout
  generates B_g with the weights it holds (B_0, B_1 from version 0, B_g from g−1), grades
  on CPU, posts the batch; the learner trains B_t while the rollout generates B_{t+1},
  exports BF16 on its GPU and broadcasts version t+1 with `WeightSync` over a two-rank
  pair group into the rollout's GPU memory (`rl/sync/{pair_group,memory_store}.py`,
  on-GPU `sum64` content digests agreed by both ranks), which `model.load_weights` them.
  Same prompts, sampling seeds, learner, reward and eval code as grpo_run, so every
  step trains on data exactly one version old (step 0: zero).
- The base-model eval is run once per seed in (a) and reused for (b)/(c); (b) seed 17
  also ran it: its per-problem results equal (a)'s bit for bit, as do (a)'s three seeds.

## Results

Per-step costs, median over 100 steps; per layout the three seeds' values
(`comparison.json`).

| | (a) time-shared, 1× A10 | (b) 2× A10, one node | (c) 2 nodes, TCP |
|---|---|---|---|
| Training time, 100 steps (s) | 1,217 / 1,205 / 1,199 | 742 / 742 / 740 | 910 / 960 / 930 |
| Trained samples/s | 5.26 / 5.31 / 5.34 | 8.63 / 8.63 / 8.65 | 7.03 / 6.67 / 6.89 |
| Median step wall (s) | 11.74 / 11.61 / 11.56 | 7.00 / 6.97 / 6.87 | 8.58 / 9.01 / 8.77 |
| Rollout generate (s) | 4.50 / 4.38 / 4.38 | 4.42 / 4.29 / 4.40 | 4.31 / 4.36 / 4.32 |
| Grading, CPU (s) | 1.37 / 1.39 / 1.32 | 1.17 / 1.14 / 1.17 | 1.23 / 1.30 / 1.31 |
| Learner step (s) | 5.73 / 5.77 / 5.68 | 5.77 / 5.79 / 5.63 | 5.58 / 5.69 / 5.67 |
| BF16 export on learner GPU (s) | in sync | 0.064 / 0.064 / 0.062 | 0.063 / 0.066 / 0.064 |
| Weight sync per step (s) | 0.056 / 0.053 / 0.055 (in-process copy) | 1.055 / 1.051 / 1.055 | 2.80 / 3.17 / 2.90 |
| Bytes per sync | 988,065,536 | 988,065,536 | 988,065,536 |
| Receiver `load_weights` (s) | — | 0.007 | 0.008 |
| Learn ∥ generate overlap (s) | 0 | 443 / 427 / 438 | 426 / 437 / 430 |
| GPU idle share | 0.113 / 0.116 / 0.110 (one GPU) | learner 0.011, rollout 0.25–0.27 | learner 0.009, rollout 0.21–0.22 |
| Staleness (samples by age) | 0: 64, 1: 6,336 | 0: 64, 1: 6,336 | 0: 64, 1: 6,336 |
| Stale samples dropped | 0 | 0 | 0 |
| Peak torch allocation, learner / rollout GPU (GB) | 14.85 (one process) | 11.69–11.77 / 4.17 | 11.76–11.78 / 4.17 |
| Peak nvidia-smi used (MiB), learner / rollout GPU | 18,066–18,260 | 15,236–15,422 / 4,652–5,104 | 15,228–15,358 / 4,644 |

Idle share: (a) is grpo_run's definition (step time outside rollout, learn and sync).
(b)/(c) are per GPU over the training window (first generation to last broadcast),
counting learn + export + broadcast as learner-GPU busy and generate + receive + load
as rollout-GPU busy; grading, posting and waiting for the learner are rollout idle.
(a)'s training time is the sum of its 100 step walls (its one extra warm-up generation,
about 6 s, is not logged); (b)/(c) generate 100 batches, grpo_run 101 (the last unused).

Held-out pass@1 (base: GSM8K 0.4526, MATH-500 0.2580 for every seed on the A10;
the 4090 base was 0.4503 / 0.2467):

| Layout | GSM8K after (s17/18/19) | GSM8K Δ pooled | MATH-500 after | MATH-500 Δ pooled |
|---|---|---|---|---|
| (a) time-shared | 0.4956 / 0.4860 / 0.4901 | **+3.80 ± 0.68 pp (5.58 SE)** | 0.2792 / 0.2686 / 0.2798 | +1.79 ± 1.01 pp (**1.76 SE, below the bar**) |
| (b) 2× A10 one node | 0.4977 / 0.4871 / 0.4864 | **+3.78 ± 0.65 pp (5.80 SE)** | 0.2812 / 0.2759 / 0.2832 | **+2.21 ± 0.99 pp (2.22 SE)** |
| (c) 2 nodes TCP | 0.4820 / 0.4774 / 0.4882 | **+2.99 ± 0.67 pp (4.47 SE)** | 0.2918 / 0.2911 / 0.2845 | **+3.12 ± 1.01 pp (3.08 SE)** |

Every seed improves on both sets in every layout. Between layouts (after − after,
per problem, seed-paired): (b) − (a) GSM8K −0.02 ± 0.49 pp, MATH-500 +0.42 ± 0.68 pp;
(c) − (a) −0.80 ± 0.50 and +1.33 ± 0.72 pp; (c) − (b) −0.78 ± 0.50 and +0.91 ± 0.71 pp.
None reaches 2 SE. The layouts are not bit-identical replays: per-step training rewards
of the same seed already differ at steps 0–2 (vLLM and learner numerics differ between
a shared and a dedicated GPU), so these differences include run-to-run noise. Train
reward first → last 10 steps: (a) 0.40–0.42 → 0.60–0.66, (b) 0.39–0.41 → 0.63–0.66,
(c) 0.40–0.41 → 0.62–0.63.

## Reading the numbers

- Overlap works: about 430–440 s of each run has the learner training while vLLM
  generates. The step is then max(learn ≈ 5.7 s, generate + grade ≈ 5.6 s) plus the sync,
  which nothing overlaps: the broadcast is a rendezvous between the two loops.
- Sync is the remaining cost. In (b) it is 1.06 s per step (0.94 GB/s effective, versus
  8.4 GB/s raw SHM all-reduce measured by the probe); in (c) 2.8–3.2 s (0.31–0.35 GB/s,
  near the 0.40 GB/s NCCL reached with 4×4 sockets in the probe). E0, measured after
  these runs, reached 2.6 GB/s broadcast with 8×8 sockets and 4 channels, so (c) did not
  run with the best socket settings and its sync time is not the link's limit. The (b) overhead is WeightSync's per-tensor protocol
  (290 tensors, each with control frames, acknowledgements and a digest), not the link.
- `NCCL_SHM_USE_CUDA_MEMCPY=1` helps: an 8-step smoke without it synced in 1.47 s per
  step versus 1.06 s with it (`smokes/e2b-smoke*-mljob-report.json`).
- Learner time is the same in every layout (5.6–5.8 s): the time-shared GPU did not slow
  the learner; (a) loses its time by running generation and learning one after the other.
- Memory: separating the roles moves vLLM's ~3.6 GB (15% of the card) and its
  activations off the learner GPU (torch peak 14.85 → 11.7 GB).

## Layout (c) rerun with tuned sockets

Run 2026-11-05 as layout (c2), after E0 found that broadcast between two GPU_NV_SM nodes reaches
2.6 GB/s with `NCCL_SOCKET_NTHREADS=8 NCCL_NSOCKS_PERTHREAD=8 NCCL_MIN_NCHANNELS=4`. The (c)
numbers above are kept as they were.

What changed:
- The NCCL environment of both roles. (c2) used E0's settings, without `NCCL_BUFFSIZE`, which
  made broadcast unstable in E0. (c) used `NCCL_SOCKET_NTHREADS=4 NCCL_NSOCKS_PERTHREAD=4`.
- `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,ENV` was added. These subsystems log only while
  communicators are set up.

What stayed the same:
- Hardware: 2 Snowflake GPU_NV_SM nodes (1× A10 24 GB each) over TCP, pool `SYSTEM_COMPUTE_POOL_GPU`.
- Code, pins, data file (`dbdf1a1b…`), seeds 17/18/19, steps, hyperparameters and held-out sets.
- The base eval is reused from (a), as before.

Deviations that do not touch training:
- Payload role timeout: 3,000 s instead of 4,800 s.
- Rendezvous ports: 12927–12929 instead of 12937–12939.

An 8-step smoke ran first (`smokes/e2c2-smoke-*`). It synced in 2.28 s per step, against 2.96 s in
`e2c-smoke`. All three full runs exited 0.

Per seed (17 / 18 / 19), medians over 100 steps (`e2/comparison-c2.json`):

| | (c) 4×4 sockets | (c2) 8×8 sockets, 4 channels |
|---|---|---|
| Training time, 100 steps (s) | 910 / 960 / 930 | **889 / 838 / 853** |
| Trained samples/s | 7.03 / 6.67 / 6.89 | 7.20 / 7.63 / 7.50 |
| Speedup over (a), mean training time | 1.29× | **1.40×** (per seed 1.37 / 1.44 / 1.41) |
| Median step wall (s) | 8.58 / 9.01 / 8.77 | 8.29 / 7.80 / 7.91 |
| Weight sync per step (s) | 2.80 / 3.17 / 2.90 | **2.22 / 1.90 / 1.95** |
| Effective sync rate, 988 MB / median (GB/s) | 0.35 / 0.31 / 0.34 | 0.45 / 0.52 / 0.51 |
| Sync time summed over 100 steps (s) | 280 / 316 / 290 | 223 / 193 / 197 |
| Learner step (s) | 5.58 / 5.69 / 5.67 | 5.78 / 5.57 / 5.74 |
| Rollout generate (s) | 4.31 / 4.36 / 4.32 | 4.54 / 4.44 / 4.43 |
| Grading, CPU (s) | 1.23 / 1.30 / 1.31 | 1.28 / 1.33 / 1.33 |
| Learn ∥ generate overlap (s) | 426 / 437 / 430 | 451 / 443 / 444 |
| GPU idle share, learner / rollout | 0.009 / 0.21–0.22 | 0.010 / 0.23–0.24 |
| Staleness (samples by age), stale dropped | 0: 64, 1: 6,336; 0 | 0: 64, 1: 6,336; 0 |
| Peak nvidia-smi used (MiB), learner / rollout GPU | 15,228–15,358 / 4,644 | 15,240–15,390 / 4,656 |

Mean training time:
- (a) 1,207 s.
- (b) 741 s, 1.63× faster than (a).
- (c) 933 s, 1.29×.
- (c2) 860 s, 1.40×.

So (c2) trains 7.8% faster than (c), but still takes 16% longer than (b) on one node. The sync
fell by a third (mean 2.96 → 2.02 s). The learner was still the bottleneck (idle 1%), so the saved
sync time went straight into training time.

Held-out pass@1 (base: GSM8K 0.4526, MATH-500 0.2580):

| Layout | GSM8K after (s17/18/19) | GSM8K Δ pooled | MATH-500 after | MATH-500 Δ pooled |
|---|---|---|---|---|
| (c) 2 nodes TCP, 4×4 | 0.4820 / 0.4774 / 0.4882 | **+2.99 ± 0.67 pp (4.47 SE)** | 0.2918 / 0.2911 / 0.2845 | **+3.12 ± 1.01 pp (3.08 SE)** |
| (c2) 2 nodes TCP, 8×8 + 4 channels | 0.4865 / 0.4936 / 0.4653 | **+2.92 ± 0.68 pp (4.28 SE)** | 0.3004 / 0.2878 / 0.2752 | **+2.98 ± 1.04 pp (2.87 SE)** |

- (c2) clears the 2 SE bar on both sets.
- Differences between layouts in final accuracy (after − after):

  | | GSM8K | MATH-500 |
  |---|---|---|
  | (c2) − (c) | −0.08 ± 0.53 pp | −0.13 ± 0.73 pp |
  | (c2) − (a) | −0.88 ± 0.53 pp | +1.19 ± 0.67 pp |
  | (c2) − (b) | −0.86 ± 0.53 pp | +0.77 ± 0.72 pp |

  None reaches 2 SE.
- Train reward, first → last 10 steps: 0.39–0.41 → 0.59–0.62.
- The broadcast delivers the same bytes whatever the socket count, and digests were checked every
  step. So the accuracy differences are run-to-run noise, as for the other layouts.

### Why the sync stays far above the link time

**The tuned transport was in effect.** The NCCL INFO lines in `e2c2-s*/{learner,rollout}-log-excerpt.txt`
(both ranks, all three seeds) show:
- `NCCL version 2.27.3+cuda12.9`, `NET/IB : No device found` and `Using network Socket`.
- `NCCL_MIN_NCHANNELS set by environment to 4`. The pair communicator reports
  `4 coll channels … 2 p2p channels per peer` and `Channel 00/04` to `03/04`.
- `NET/Socket: Using 8 threads and 8 sockets per thread` four times per rank, and every
  connection `via NET/Socket/0`.
- `Connected all rings, use ring PXN 0 GDR 0`: no GPU Direct RDMA, so data passes through
  host buffers.

This is the configuration with which E0 measured 2.6 GB/s broadcast, so no setting was lost on
the way to the job.

**The link time went down, and the rest did not.** WeightSync sends in chunks of up to 16 MiB:
- At E0's 16 MiB broadcast rate (2.22 GB/s with these settings, 0.84 GB/s with 4×4), 988 MB
  needs 0.45 s of link time; with 4×4 it needs 1.17 s.
- The time above the link time is 1.45–1.77 s in (c2) and 1.63–2.00 s in (c), about the same.
- The new sockets removed most of the link time. Most of the remaining sync time is fixed cost
  that does not depend on the link rate.

**That fixed cost is WeightSync's per-tensor protocol** (`rl/sync/nccl_broadcast.py`):
- One push of the 290 tensors makes 1,478 collectives, as counted from the code and confirmed by
  running `WeightSync.push` over gloo with the Qwen2.5-0.5B BF16 schema:
  - 5 at the start, 5 per tensor, 16 more for the 17-chunk embedding, and 7 at the end.
  - 306 of them are data broadcasts.
  - 1,172 are tiny: 586 control-frame broadcasts and 586 four-byte status all-reduces.
- Each control frame and status flag is read on the host (`.item()` / `.cpu()`) before the next
  call, so the collectives run strictly one after another.
- Both GPUs also run a finite check and a sum64 checksum on every tensor.
- 169 of the 290 tensors are 1 MiB or smaller (121 are 4 KiB or smaller). At that size E0 found
  broadcast latency-bound.

**Estimated split.** The same protocol takes 1.05 s on one node in (b), where moving the data over
SHM takes only about 0.05 s. Over TCP, each of the ~1,170 sequential round trips costs roughly
0.5 ms more: (1.58 − 1.00) s / 1,172.

This split is an estimate from E0's rates, the count above and (b). The INIT/NET/ENV subsystems
log only setup, so the logs do not contain per-collective times.

**Next step (not tried).** A bucketed sync, with a few large flat broadcasts and one digest, would
take about 0.4–0.5 s per step at E0's rates.

### Raw outputs and credits

- Raw outputs: `e2/e2c2-s{17,18,19}/`, with the same files as (c).
- `e2/comparison-c2.json`: `python -m rl.local.compare_layouts --layout a=… --layout b=… --layout c=…
  --layout c2=…`. Its a/b/c entries and pairs equal `comparison.json`.
- Eval completions are stored in `@CRUCIBLE_LAB.PROBE.ASSETS/results/e12/e2c2-s<seed>.tar.gz`.
- Credits (ledger upper bounds): smoke 0.720, s17 1.977, s18 1.892, s19 1.885; total 6.47.

## Smokes and failures

Smokes first (8 steps for (b)/(c), 5 for (a), first 16 problems of each held-out set):
`e2a-smoke`, `e2b-smoke` (memcpy on), `e2b-smoke-shm-default`, `e2c-smoke`; all exited 0
(reports in `e2/smokes/`). No full run failed. The same-node NCCL failure and its fix
(both processes must see both GPUs) are in [e1-weight-sync.md](e1-weight-sync.md).

## Deviations

- A10 instead of the README's RTX 4090; the base eval differs by GPU (above), so the A10
  runs are compared with each other only.
- (b)/(c) reuse (a)'s base eval per seed (allowed by the brief; checked equal for seed 17).
- (b)/(c) generate 100 batches instead of grpo_run's 101 (its last batch is never trained).
- E2 syncs use on-GPU `sum64` digests (64-bit position-weighted sums, not cryptographic);
  E1 used SHA-256.
- GPU utilization from nvidia-smi in the reports is averaged over the whole process,
  including model load and eval; the idle shares above come from phase timestamps.
- In two-node runs, clocks of two hosts are compared for the window and overlap
  (same-node runs use one clock).

## Code and tests

`rl/local/async_grpo.py` (two-role loop, `merge` into grpo_run-format `steps.jsonl` /
`summary.json`), `rl/local/compare_layouts.py`, the E1 sync pieces, and the payload
`scripts/snowflake/payloads/e12.py` (+ `scripts/snowflake/results.py` to download job
outputs). CPU tests: `rl/local/tests/test_async_grpo.py` runs the learner and rollout loops
as two real processes over a gloo pair group with WeightSync into a MemoryStore and
checks the version schedule, that each batch comes from the expected weights, the
staleness histogram, overlap and the merge; `test_compare_layouts.py` checks the paired
statistics.

## Raw outputs

`e2/e2{a,b,c}-s{17,18,19}/`: `summary.json` (per-problem correct counts before/after,
pipeline block), `steps.jsonl`, `mljob-report.json`; for (b)/(c) also `learner.jsonl`,
`rollout.jsonl` (timestamps of every phase), `learner-summary.json`, log excerpts.
`e2/comparison.json` is `python -m rl.local.compare_layouts --layout a=… --layout b=…
--layout c=…`. Eval completions (`eval-*.jsonl.gz`, about 1 MB each) are not committed;
they are in `@CRUCIBLE_LAB.PROBE.ASSETS/results/e12/<run>.tar.gz`.

Credits (ledger upper bounds): (a) 2.824 + 1.326; (b) 2.234 + 1.613 + 1.584; (c) 1.910 +
1.969 + 1.965; smokes 0.708 + 0.523 + 0.711 + 0.554. E2 total 17.92; E1 + E2 19.92 of 81.
