# E1: multi-GPU NCCL weight sync correctness (Snowflake A10, 2026-11-04)

Result: the learner process broadcast its BF16 weights over NCCL with
`rl/sync/nccl_broadcast.py` to vLLM on another GPU, and vLLM's logits after the sync
passed the frozen Task 9 acceptance in both layouts: **all 1,823,232 coordinates
(6 prompts × 2 weight versions × 151,936 vocabulary entries) within the bound, 0
failures**, in (1) GPU0 → GPU1 on one node and (2) across two nodes. Both ranks'
SHA-256 content digests agreed for both versions before commit. One 988,065,536-byte
version took **2.21–2.22 s** inside one node and **5.88–5.93 s** across two nodes,
including a per-tensor SHA-256 on both sides.

Hardware: (1) Snowflake GPU_NV_2M: 2× A10 24 GB (vGPU "A10-24Q") in one node, no
NVLink/P2P; NCCL used `SHM/direct/direct`. (2) 2 Snowflake GPU_NV_SM nodes (1× A10 24 GB
each) over TCP; NCCL used `NET/Socket`, default socket settings. Software: torch
2.8.0+cu128, NCCL 2.27.3, vLLM 0.10.2, transformers 4.55.4, installed from Snowflake's
PyPI proxy in each job.

## What ran

`rl/sync/nccl_check.py`, launched by `scripts/snowflake/payloads/e12.py --task e1`
(two processes, one GPU each):

- **Learner** (rank 0 of a separate two-rank pair group, `rl/sync/pair_group.py`):
  the FSDP2 learner (`load_fsdp2_learner`, FP32 masters) at policy version 7 exports
  BF16 on its GPU and broadcasts it with `WeightSync(MemoryStore(..., digest="sha256"),
  group=pair, source=0)`; then one controlled AdamW step (lr 1e-4, the Task 9 source
  step) to version 8, exported and broadcast the same way.
- **Rollout** (rank 1): in-process vLLM 0.10.2 V1, BF16, eager, `raw_logits`, the Task 9
  diagnostic profile (batch 2, utilization 0.80, max length 128). It started from the
  base snapshot, received each version into GPU memory (`rl/sync/memory_store.py`),
  loaded it with `model.load_weights`, and computed the full logits at the last prompt
  position of the six protocol prompts.
- **Reference**: after both syncs the learner built Transformers 4.55.4 BF16 SDPA
  models from exactly the BF16 tensors it had sent (batch 1, TF32 off, highest FP32
  matmul precision) and compared them with the vLLM logits returned over the pair group.

Acceptance, unchanged from [the frozen protocol](../raw/task9-confirmation-protocol.json)
(SHA-256 `a9d94ba3…`, checked by `load_protocol`): for every prompt and version, every
coordinate `|vLLM − ref| ≤ 0.375 + 0.01·|ref|`, RMS ≤ 0.10, mean absolute ≤ 0.08,
all finite, identical input token IDs. Added here: the run fails unless both ranks'
digests agree and the vLLM logits change between v7 and v8.

## Results

| | Same node (GPU0 → GPU1) | Two nodes (TCP) |
|---|---|---|
| Coordinates within bound | 1,823,232 / 1,823,232 | 1,823,232 / 1,823,232 |
| Rows passed (12) | 12 | 12 |
| Max abs error v7 / v8 | 0.375 / 0.2578 | 0.375 / 0.2578 |
| Max RMS v7 / v8 | 0.0775 / 0.0547 | 0.0775 / 0.0547 |
| Max mean abs v7 / v8 | 0.0605 / 0.0453 | 0.0605 / 0.0453 |
| Token IDs identical | yes | yes |
| Digests agree (v7, v8) | yes (`046549e4…`, `84a854a6…`) | yes (same digests) |
| vLLM logit change v7 → v8 (max) | 19.97 | 19.97 |
| Bytes per version | 988,065,536 | 988,065,536 |
| Push v7 / v8 (s) | 2.224 / 2.207 | 5.884 / 5.929 |
| Effective GB/s | 0.44 | 0.17 |
| Export on learner GPU v7 / v8 (s) | 0.229 / 0.061 | 0.191 / 0.060 |
| Receiver `load_weights` (s) | 0.0097 / 0.0098 | 0.0105 / 0.0105 |
| Peak GPU memory (nvidia-smi), learner / rollout GPU | 11,762 / 20,854 MiB | 11,762 / 20,854 MiB |

- The two layouts produced bitwise identical logit arrays (`logits.npz` SHA-256
  `0992b52e…` in both runs) and the same weight digests: the controlled step is
  deterministic and the transport did not change a byte.
- The largest error, 0.375 on v7 prompt 1 ("A triangle has three"), sits exactly at the
  absolute part of the bound and passes only through the relative term. On the 4090 the
  v7 maximum was 0.3398. The bound was not changed.
- Sync times are a protocol cost, not link bandwidth: each push sends 290 tensors with
  per-tensor control frames and acknowledgements, and here hashes every tensor with
  SHA-256 on the host on both ranks. E2 uses the on-GPU `sum64` digest instead (see
  [e2-async-grpo.md](e2-async-grpo.md)). The probe's raw links are 2.3 GB/s (SHM
  all-reduce, default) and 0.19 GB/s (TCP, default).

## Failures kept

Two same-node attempts failed before any weight moved, both in NCCL's first
collective with `transport/shm.cc:590 NCCL WARN Cuda failure 217 'peer access is not
supported between these two devices'`: attempt 1 with defaults, attempt 2 with
`NCCL_P2P_DISABLE=1`. Each process then saw only its own GPU
(`CUDA_VISIBLE_DEVICES=0` / `1`). The fix: both processes see both GPUs
(`CUDA_VISIBLE_DEVICES=0,1` for the learner and `1,0` for the rollout, each uses its
`cuda:0`); NCCL then chose `SHM/direct/direct`. Reports:
`e1/failed-e1-same-node-attempt{1,2}-*/mljob-report.json`.

## Code and tests

- `rl/sync/pair_group.py`: a two-rank NCCL/gloo group on its own TCPStore beside each
  process's own world-1 default group (the FSDP2 learner's and in-process vLLM's),
  built with Torch 2.8's private `_new_process_group_helper` (pinned runtime).
- `rl/sync/memory_store.py`: the VersionStore staging interface held in memory, with
  `sha256` or `sum64` digests; the sender keeps only checksums.
- `rl/sync/nccl_broadcast.py`: the source rank is now the caller's rank in the group
  (`get_global_rank(group, group.rank())`), not the default-group rank, which is 0 in
  both processes here.
- `rl/sync/export.py`: `ready_bf16_tensors(..., device=)`, the same guarded export kept
  on the GPU.
- CPU tests: `rl/sync/tests/test_memory_store.py` (schema/digest rules; actual
  two-process gloo WeightSync over the pair group between two processes whose default
  rank is 0; a corrupted receiver tensor is rejected before commit). `rl/sync`,
  `rl/local` and `rl/learner/tests/test_grpo.py`: 114 passed, 1 skipped (GPU opt-in) in
  `.venv-ml`, CPU only; ruff clean.

## Limits

- One run per layout; the timing has no spread. The logits are deterministic, so a
  repeat would only add timing samples.
- The acceptance is the Task 9 empirical bound for this model, runtime and six inputs;
  it is not a general BF16 guarantee.
- The NCCL path writes into memory, not into the disk `VersionStore`; the disk
  publication/rollback lifecycle of Task 9 was not rerun.
- `logits.npz` (6.6 MB per run) is not committed (5 MB evidence budget); it is in
  `@CRUCIBLE_LAB.PROBE.ASSETS/results/e12/e1-{same,two}-node.tar.gz`
  (`python scripts/snowflake/results.py get e12/<run> <dir>`), SHA-256 recorded in
  `result.json`.

Raw outputs: `e1/e1-same-node/`, `e1/e1-two-node/` (`result.json` with all 12 rows,
`mljob-report.json`, nvidia-smi samples, NCCL log excerpts). Credits: 0.435 + 0.465 +
0.470 (same node, two failed + one passed) + 0.625 (two nodes) = 1.995 (ledger upper bound).
