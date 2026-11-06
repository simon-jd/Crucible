# E2b: bucketed weight sync between two nodes (Snowflake A10, 2026-11-06)

Result: sending the 988 MB BF16 state as 15 broadcasts of 64 MiB, with a SHA-256 manifest, instead
of WeightSync's 290-tensor protocol (1,478 collectives) cuts a two-node push from **2.02 s to
0.45 s** in the microbenchmark. The raw link needs **0.38 s**. The E1 correctness gate passes
unchanged: **0 of 1,823,232 coordinates out of bound**, and the logits are bit-identical to E1's.
In one 100-step layout (c) run (seed 17), sync took **0.48 s per step** (median) against 2.22 s in
the E2c rerun of the same seed. Training took **689 s** against 889 s, which is **1.75× faster than
(a)** (mean 1,207 s), against 1.40× for the E2c rerun. This is one run with one seed.

Hardware: 2 Snowflake GPU_NV_SM nodes (1× A10 24 GB each) over TCP, pool `SYSTEM_COMPUTE_POOL_GPU`,
NCCL `NET/Socket` with E0's settings `NCCL_SOCKET_NTHREADS=8 NCCL_NSOCKS_PERTHREAD=8
NCCL_MIN_NCHANNELS=4` (no `NCCL_BUFFSIZE`), as in the E2c rerun. torch 2.8.0, NCCL 2.27.3, vLLM
0.10.2, transformers 4.55.4 from Snowflake's PyPI proxy. Each job got its own node pair.

## What changed

`rl/sync/bucketed.py` (new), `BucketedSync`. It has the same interface as `WeightSync`: every
rank calls `push(shards, version)`, and the receiver `take`s the committed tensors from its
`MemoryStore`.

- **Layout.** Both ranks derive the same plan from the schema. Tensors go in sorted-name order,
  each at a 256-byte aligned offset of one flat byte buffer. For Qwen2.5-0.5B the padding is
  zero bytes: 988,065,536 bytes in total.
- **Pack.** The source copies each exported BF16 tensor into its slot. There is no cast, and
  every schema name must appear exactly once.
- **Transfer.** One broadcast per bucket goes straight into the receiver's flat buffer. Unpacking
  makes zero-copy views with the schema's shapes and dtypes, which `model.load_weights` copies
  into vLLM.
- **Integrity, equivalent to WeightSync with `digest="sha256"`:**
  - Before any data moves, the receiver checks the version, the contract SHA-256 (schema and
    asset hashes) and the layout SHA-256 against its own.
  - Both ranks SHA-256 the packed bytes in 59 chunks of up to 16 MiB. A side CUDA stream copies
    each finished bucket into a pinned host buffer, and 8 threads hash it while later buckets
    are still in flight.
  - Floating tensors must be finite on both ranks (`MemoryStage.adopt`, new).
  - The receiver commits only if the source's manifest (version, contract, layout, chunk hashes,
    sizes) equals the manifest of the bytes it received. SHA-256 of the canonical manifest is
    the version digest.
  - Collectives per push: number of buckets + 10 (6 status all-reduces and 2 control frames).
    That is 25 at 64 MiB, against 1,478 for WeightSync.
- **Where it is used.** `--bucket-mb N` in `rl/sync/nccl_check.py` (E1),
  `rl/local/async_grpo.py` (E2) and `scripts/snowflake/payloads/e12.py`. Without the flag the
  old path runs unchanged. `rl.sync.sync_bench` (new, `e12.py --task bench`) is the
  microbenchmark. `async_grpo --link-probe N` times N plain broadcasts of the weight bytes before
  and after training. This happens outside the training window.

## 1. CPU tests

- `rl/sync/tests/test_bucketed.py` (15 tests):
  - The plan: order, alignment, buckets covering the buffer, hash chunks that never cross a
    bucket edge, and a layout digest that depends on the schema and the bucket size.
  - Exact zero-copy pack/unpack for BF16, FP32, int64, bool, 0-d and empty tensors, with zero
    padding.
  - Rejection of unexpected, duplicate or missing names and of shape/dtype changes.
  - Manifest checks name the first differing chunk.
  - `adopt` enforces the full schema and finite values.
  - Two-process gloo runs over the pair group:
    - A clean push delivers the exact tensors in 1 bucket + 10 collectives.
    - Integrity failure: the receiver flips one bit after the wire. Both ranks raise, the cause
      reads "content differs in 1 of … chunks", the previous version stays committed, and the
      next clean push succeeds.
    - Different bucket sizes on the two ranks are rejected before any data moves.
    - A non-finite source tensor is rejected.
- `rl/sync/tests/test_sync_bench.py` checks the benchmark bookkeeping.
- `rl/local/tests/test_async_grpo.py`: the two-process pipeline test now runs with both sync paths
  and checks the link probe.
- `rl/sync` + `rl/local`: 123 passed, 1 skipped (GPU opt-in) in `.venv-ml`, CPU only.
- A gloo dry run of the whole benchmark ran on the host with the real model (CPU only).

## 2. Sync microbenchmark

`e12.py --task bench`: two processes on two nodes, the source holding the base model's 290 BF16
tensors on its GPU. Each variant ran one warm-up pass, then 3 timed passes, with the variants
interleaved. A barrier precedes every push. The time of a push is the slower rank's push plus
`cuda.synchronize`. In the warm-up pass the receiver compared every received tensor with the base
weights it read itself: all equal. Both ranks' digests agreed on every push.

| Variant | Collectives | Push, 3 passes (s) | Median (s) | GB/s |
|---|---|---|---|---|
| per-tensor WeightSync, sum64 (E2 path) | 1,478 (counted in E2c) | 1.915 / 2.021 / 2.020 | 2.020 | 0.49 |
| per-tensor WeightSync, SHA-256 (E1 path) | 1,478 | 2.659 / 2.672 / 2.857 | 2.672 | 0.37 |
| bucketed 4 MiB (236 buckets) | 246 | 0.593 / 0.573 / 0.569 | 0.573 | 1.72 |
| bucketed 16 MiB (59) | 69 | 0.455 / 0.463 / 0.457 | 0.457 | 2.16 |
| **bucketed 64 MiB (15)** | 25 | 0.434 / 0.495 / 0.451 | **0.451** | 2.19 |
| bucketed 256 MiB (4) | 14 | 0.452 / 0.465 / 0.472 | 0.465 | 2.13 |
| bucketed 1 GiB (1 bucket) | 11 | 0.543 / 0.583 / 0.539 | 0.543 | 1.82 |
| raw broadcast, 4 / 16 / 64 / 256 / 1,024 MiB calls | 236 … 1 | 0.36–0.39 | 0.385 / 0.380 / 0.386 / 0.380 / 0.386 | 2.56–2.60 |

- **Link floor.** The raw rows are plain `dist.broadcast` calls of the packed buffer, with no
  control frames and no hashing. They reach 2.6 GB/s whatever the call size, as E0 found
  (2.59 GB/s at 1 GiB).
- **Bucketed cost above the floor.** At 64 MiB a push costs about 0.07 s more than the raw floor.
  Packing takes 0.010 s. After the last bucket lands, the receiver needs 0.011 s to finish
  hashing and 0.024 s to check and commit.
- **The edges.**
  - With 4 MiB buckets each bucket waits on the host before the next is issued, which adds about
    0.7 ms per bucket. The 236 buckets cost 0.12 s more than 64 MiB. This is inferred from
    receiver transfer times of 0.54 s for 4 MiB buckets against 0.39 s for raw 4 MiB calls.
  - With one 1 GiB bucket, hashing can only start once everything has landed. The receiver's
    hash tail grows from 0.011 s to 0.108 s.
  - Between 16 and 256 MiB the medians differ by less than the pass-to-pass spread. 64 MiB was
    used below.
- **Per-tensor and bucketed against the same link.** Per-tensor sum64 here (2.02 s) matches the
  E2c rerun (1.90–2.22 s per step). Above the link floor, per-tensor costs 1.64 s and bucketed
  0.07 s.

## 3. E1 correctness gate, bucketed

`e12.py --task e1 --layout two-node --bucket-mb 64`: the same code, prompts, versions (7, then 8
after the controlled step), frozen protocol and bound as E1. Only the transport changed.

| | E1 two nodes (per-tensor, SHA-256) | E2b gate (bucketed 64 MiB, SHA-256 manifest) |
|---|---|---|
| Coordinates within bound | 1,823,232 / 1,823,232 | **1,823,232 / 1,823,232** |
| Rows passed (12) | 12 | **12** (0 failed coordinates) |
| Max abs error v7 / v8 | 0.375 / 0.2578 | 0.375 / 0.2578 |
| Max RMS v7 / v8 | 0.0775 / 0.0547 | 0.0775 / 0.0547 |
| Max mean abs v7 / v8 | 0.0605 / 0.0453 | 0.0605 / 0.0453 |
| Token IDs identical | yes | yes |
| Both ranks' digests agree | yes | yes (manifest digests `87632208…`, `407efb22…`) |
| `logits.npz` SHA-256 | `0992b52e…` | **`0992b52e…` (identical)** |
| vLLM logit change v7 → v8 (max) | 19.97 | 19.97 |
| Push v7 / v8 (s) | 5.884 / 5.929 (default sockets) | 0.824 / 0.599 |
| Receiver `load_weights` v7 / v8 (s) | 0.0105 / 0.0105 | 0.0101 / 0.0097 |

- **Same bytes.** The vLLM and HF logit arrays are bit-identical to both E1 runs, so the
  transport delivered exactly the bytes E1 delivered.
- **Digests.** The manifest digests have a different format from E1's per-tensor digests, so they
  cannot be compared with E1's directly.
- **Push times.** The E1 column mixes two changes: E1 used default sockets. The microbenchmark
  separates the two (same link: per-tensor SHA-256 2.67 s, bucketed 0.45 s). The v7 push includes
  the first broadcast on a new communicator. The v8 push (0.60 s, of which 0.54 s transfer) was
  slower than in the microbenchmark, on a different node pair.

## 4. E2 layout (c), bucketed, one run

`e12.py --task pair --layout two-node --seed 17 --timeout 3000 --bucket-mb 64 --link-probe 3`.
Code, pins, data file (`dbdf1a1b…`), seed 17, 100 steps, hyperparameters, held-out sets and NCCL
settings are those of the E2c rerun (c2). The base eval is reused from (a) seed 17, as before. An
8-step smoke ran first (`smokes/e2c-smoke-*`).

Seed 17, medians over 100 steps (`comparison-s17.json`; (a) and (c2) from `../e2/`):

| | (a) time-shared, 1× A10 | (c2) E2c rerun, per-tensor sum64 | **(c3) bucketed 64 MiB, SHA-256** |
|---|---|---|---|
| Training time, 100 steps (s) | 1,216.8 | 889.1 | **688.7** |
| Trained samples/s | 5.26 | 7.20 | **9.29** |
| Speedup over (a) seed 17 / over (a) mean 1,207 s | — | 1.37× / 1.36× | **1.77× / 1.75×** |
| Median step wall (s) | 11.74 | 8.29 | 6.38 |
| Weight sync per step (s), median | 0.056 (in-process copy) | 2.220 | **0.476** (p90 0.81, max 0.95) |
| Sync summed over 100 steps (s) | — | 223 | 55 |
| of which transfer, median (s) | — | — | 0.402 |
| Plain-broadcast link probe, same processes (s) | — | — | 0.68 / 0.40 / 0.37 before, 0.38 / 0.34 / 0.35 after |
| Learner step (s) | 5.73 | 5.78 | 5.60 |
| Rollout generate / grade (s) | 4.50 / 1.36 | 4.54 / 1.28 | 4.34 / 1.30 |
| Learn ∥ generate overlap (s) | 0 | 451 | 430 |
| GPU idle share, learner / rollout | 0.113 (one GPU) | 0.010 / 0.233 | 0.012 / 0.285 |
| Collectives per sync | — | 1,478 | 25 |
| Staleness (samples by age), stale dropped | 0: 64, 1: 6,336; 0 | 0: 64, 1: 6,336; 0 | 0: 64, 1: 6,336; 0 |
| Peak nvidia-smi used (MiB), learner / rollout | 18,066 (one GPU) | 15,390 / 4,656 | 15,628 / 5,334 |

Mean training time of the earlier layouts: (a) 1,207 s, (b) one node 741 s (1.63×), (c) 933 s
(1.29×), (c2) 860 s (1.40×, seeds 838–889 s).

- **Sync.** The sync fell from 2.22 to 0.48 s per step (−1.74 s). The 100 syncs took 55 s
  instead of 223 s. The first plain broadcast probably includes connection setup; the warm probes
  show a link floor of 0.34–0.40 s. So sync sits about 0.1 s above the link floor in this run, as
  in the microbenchmark (0.07 s).
- **Training time.** It fell by 200 s; the sync accounts for 168 s of that. The rest is run-to-run
  variation: the median learner step was 5.60 s here against 5.78 s in c2 seed 17, and 5.57–5.78
  s across c2's seeds.
- **Against (b).** (c3) on two nodes trained faster than (b) on one node. (b) still uses the
  per-tensor protocol, 1.06 s per step over SHM. A bucketed (b) was not run.
- **Bottleneck.** The loop is still learner-bound (learner idle 1.2%). The rollout GPU idles
  more, 28.5%, because its cycle got shorter.
- **The smoke was slower.** It synced in 1.30 s per step, with a transfer of 1.22 s, on a
  different node pair. It had no link probe, so I cannot say whether its link was slower. The
  full run's probe shows the floor for the pair it ran on. Link speed differs between node pairs
  and jobs, and nothing here controls it.

Held-out pass@1 (base: GSM8K 0.4526, MATH-500 0.2580; one seed, so the SE is over problems only):

| Layout, seed 17 | GSM8K after | GSM8K Δ | MATH-500 after | MATH-500 Δ |
|---|---|---|---|---|
| (a) time-shared | 0.4956 | +4.30 ± 0.84 pp (5.12 SE) | 0.2792 | +2.12 ± 1.23 pp (1.73 SE) |
| (c2) E2c rerun | 0.4865 | +3.39 ± 0.87 pp (3.90 SE) | 0.3004 | +4.24 ± 1.28 pp (3.33 SE) |
| **(c3) bucketed** | 0.4839 | **+3.13 ± 0.88 pp (3.57 SE)** | 0.2812 | +2.32 ± 1.19 pp (**1.95 SE, below the bar**) |

- **The 2 SE bar.** GSM8K clears it; MATH-500 misses it narrowly in this one run. For comparison,
  the three-seed (c2) cleared it at 2.87 SE, and the seed-17 (a) run missed it as well, at 1.73 SE.
- **Between layouts, seed 17 (after − after, per problem).** None reaches 2 SE:
  - (c3) − (c2): GSM8K −0.27 ± 0.85 pp, MATH-500 −1.92 ± 1.20 pp.
  - (c3) − (a): −1.18 ± 0.92 and +0.20 ± 1.08 pp.
- **Bytes and data.** Every step's received bytes matched the learner's SHA-256 manifest, and the
  data schedule (staleness histogram) is identical. So the accuracy differences come from training
  nondeterminism, as between the earlier layouts.
- **Train reward**, first → last 10 steps: 0.395 → 0.616 (c2 seed 17: 0.412 → 0.620).
- **Truncation.** The after eval truncated more completions than the other seed-17 runs: GSM8K 142
  of 5,276 against 40 in (c2) and 64 in (a); MATH-500 566 against 418 and 449. One seed; not
  explained here.

## Deviations

- **Integrity check.** The bucketed path uses a SHA-256 manifest over 16 MiB chunks. E2's
  per-tensor path used on-GPU sum64 and E1 per-tensor SHA-256. The new check is cryptographic and
  covers every byte, padding included.
- **E1 gate settings.** It ran with the tuned sockets and `NCCL_DEBUG_SUBSYS=INIT,NET,ENV`; E1
  used default sockets.
- **E2 run additions.** `--link-probe 3` adds 6 plain broadcasts outside the training window. It
  was CPU-tested and uses the same broadcast code as the benchmark's raw rows, but was not
  smoke-tested on GPU before the run. Rendezvous ports: 12921–12925, instead of 12927.
- **Memory.** The learner holds a 988 MB packed copy next to the exported tensors during a push.
  Each rank also keeps a 988 MB pinned host mirror. Peak GPU memory (nvidia-smi) was +238 MiB
  (learner) and +678 MiB (rollout, including the probe buffer) over c2 seed 17.
- **Benchmark weights.** The microbenchmark sends the base model's weights, not trained ones.
  Timing does not depend on the content.
- **E1 job log.** The E1 job's `RESULT` line was over 16 KB and was split in the job log, so its
  `mljob-report.json` keeps it as `unparsed`. The job's `result.json`, from the stage, is the
  source of the table above.

## Limits

- **One run.** The E2 comparison is one run with one seed: no spread for (c3). (c2) varied 838–889
  s and 1.90–2.22 s per sync across its seeds.
- **One node pair per measurement.** The microbenchmark has 3 timed passes on one node pair. The
  smoke's slower transfer shows that the link differs between node pairs.
- **What the SHA-256 proves.** It shows that the receiver holds the source's bytes. It is not
  protection against a malicious host.
- **What was not run.** A bucketed one-node (b) layout, more seeds of (c3), and the disk
  `VersionStore` lifecycle.

## Raw outputs and credits

- **Raw outputs.**
  - `e2b/bench/`: `result.json` with every push of both ranks, `mljob-report.json`, nvidia-smi
    samples and NCCL log excerpts.
  - `e2b/e1-two-node/`: `result.json` with all 12 rows.
  - `e2b/e2c-s17/`: `summary.json` merged with (a) seed 17's base eval, `steps.jsonl`,
    `learner.jsonl` and `rollout.jsonl` (per-step `sync_detail`), `learner-summary.json` (link
    probe) and log excerpts.
  - `e2b/comparison-s17.json`: `python -m rl.local.compare_layouts --layout a=… --layout b=…
    --layout c=… --layout c2=… --layout c3=e2b/e2c-s17` (seed 17 of each layout).
  - `e2b/smokes/`.
- **Not committed.** `logits.npz` (6.6 MB) and the eval completions are in
  `@CRUCIBLE_LAB.PROBE.ASSETS/results/e12/e2b-{e1-two-node,e2c-s17}.tar.gz`.
- **Credits** (ledger upper bounds): bench smoke 0.756, bench 0.624, E1 gate 0.888, E2 smoke 0.711,
  E2 run 1.766. Total **4.745**.
