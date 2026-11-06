# Versioned learner → rollout synchronization

`WeightSync.push(shards: Iterable[tuple[str, Tensor]], version: int) -> SyncStats(seconds, bytes)`
prepares a complete candidate. It does **not** switch a running engine. Use
`RolloutEngine(..., weight_store=store).load_weights(version)` to activate it.

`VersionStore(root, schema, assets)` owns at most two snapshots, counting partial
staging. The schema contains every unique parameter/persistent buffer name, shape,
and dtype; `base_schema(pinned_base_path, torch.bfloat16)` derives the Qwen schema
from all base safetensors. Qwen0.5B has 290 unique tensors, with its input embedding
and output head tied. Each rank needs the same immutable base config/tokenizer
assets; their copied hashes are part of the transport contract. Generic push
rejects missing/extra/duplicate names, shape/dtype changes, and nonfinite values.
It never silently casts, truncates, or skips parameters. The actual vLLM loader
supports BF16 floating state only and rejects incompatible owned schemas before
spawning a worker, then rechecks in the child before constructing vLLM. CPU injected
backends may test other exact dtypes; they do not establish real-loader support.

The store writes one tensor at a time, hashes every weight and copied asset,
creates a complete safetensors index and content manifest, fsyncs, and renames the
staging directory. There is no additional whole-checkpoint temporary copy.
Published files are read-only; the SHA256 manifest is the checkpoint revision.
Verification checks the exact file set and every content hash before/after actual
worker loading. This is a single-owner contract protecting against accidental
changes, **not** attestation against a malicious host with the same privileges.
Concurrent independent writers to one store are unsupported. The process-local
store lock protects one store object; reopening the same root in another object or
process does not provide a cross-process lease.

Only engine load acknowledgement plus content verification changes the active
identity. Loading closes and joins the old worker before starting its candidate.
On a candidate failure with confirmed cleanup, it reloads the verified previous
snapshot. Unknown startup/cleanup ownership fails closed and retains the worker
handle; it never starts another worker while termination is uncertain. The store
reserves an engine identity before initial worker construction. Its `active_version`
protection pin also covers that initial reservation; it is not a worker-ready ACK.
A constructor failure preserves the original exception and attaches
`error.recovery_engine`; the store also retains that failed owner. Retry
`error.recovery_engine.close()` to drain a retained partial worker. Only confirmed
cleanup releases that engine's reservation. Unknown ownership without a usable
handle stays blocked until the outer supervisor establishes a fresh owner phase.
A confirmed-clean constructor failure permits retry. Successful close releases
only its own pin, so repeated close cannot unpin a successor. `discard(version)` rejects the
active snapshot, and a persistent high-water mark prevents reusing discarded
version numbers. There is no automatic pruning. A third candidate is rejected
until the caller explicitly discards an inactive version. All snapshots, partial
staging, references, caches, and temporary files count toward the project budget.

For transport, omit `group` for a local disk handoff; otherwise every rank calls
`WeightSync(store, group=group, source=0).push(...)`, and only the source supplies
tensors. CPU gloo and future CUDA NCCL use the same protocol: schema/asset agreement,
16 MiB tensor chunks, receiver write acknowledgement, complete-manifest agreement,
then catalog acknowledgement. Control frames are bounded to 64 KiB, names to 1,024
characters, and individual tensors to 1 GiB. This version requires caller-owned
process groups and distinct store roots per rank. Destroy the group in `finally`.

The timeout is one absolute transaction deadline, including all collective waits,
not a fresh timeout per tensor. Blocking producer/filesystem calls are checked
before/after; they cannot be forcibly interrupted by this synchronous API. Process
or transport loss makes the owner/store in doubt and forbids reuse. Final commit
I/O and the successful push return both recheck the absolute deadline. If time
expires after publication, committed bytes/high-water identity remain intact and
the owner/store becomes in doubt; no successful statistics are returned. Published
candidate catalogs can differ after a lost final acknowledgement: this is **not**
a distributed atomic-commit guarantee. The previous active engine is untouched by
push. Tear down uncertain groups/processes and explicitly recover in a new owner
phase; never infer activation from a candidate directory alone.

`BucketedSync(MemoryStore(schema, digest="sha256"), group=pair, bucket_bytes=N)`
(`rl/sync/bucketed.py`) is the same interface for a running vLLM: both ranks derive one
layout of the schema in a flat byte buffer (sorted names, 256-byte aligned), the source
packs its tensors into it without casting, and the buffer moves in N-byte broadcasts
straight into the receiver's buffer, which is unpacked as zero-copy views. A push is
n_buckets + 10 collectives instead of WeightSync's ~5 per tensor. Integrity: version,
contract and layout hashes are agreed before data moves; both ranks SHA-256 the packed
bytes in ≤16 MiB chunks (on CUDA through a pinned host mirror, hashed by a thread pool
while later buckets are on the wire), check finiteness, and commit only if the source's
manifest equals the receiver's. `make_weight_sync(..., bucket_mb=)` picks either path;
`rl.sync.sync_bench` measures both against raw broadcasts.

`export_ready_learner_bf16(learner, sync)` is the explicit inference conversion.
It freshly obtains the guarded public policy/version, validates the full state and
Qwen tied alias, materializes DTensors individually, and converts floating tensors
to BF16. Public guards/version are rechecked through exhaustion before publication.
The caller owns the entire learner phase and must not call `step` concurrently.
Failed/executing learners cannot publish; cached state or diagnostic committed
versions are not accepted. FP32 master parameters are not numerically identical to
BF16 publication. No optimizer state is saved.

Real GPU calls require `scripts/with_gpu.sh`. The learner process must exit before
vLLM starts; one context at a time, fixed utilization 0.80, no resident service.
`_diagnostic_logits=True` creates an explicit diagnostic worker using actual
vLLM0.10.2 V1 `raw_logits`, `max_logprobs=-1`, and one generated token. `full_logits`
returns every model-vocabulary coordinate at the last input position and records
exact input token IDs. This worker refuses `generate`, so these values cannot be
misinterpreted as normal `Sample.logprobs`.

```bash
.venv-ml/bin/python -m pytest -q rl/sync/tests/test_sync.py
bash scripts/test_sync_gpu.sh
```

The default GPU entrypoint creates a fresh source phase, requires an empty
`.cache/task9/acceptance`, joins the source, and writes `binding.json` containing
this run's actual snapshot digests, source-report hash and exit0 proof, base
revision, and frozen protocol hash. A fresh run need not reproduce the historical
snapshot digests. `bash scripts/test_sync_gpu.sh --reuse` accepts only that exact
owned run with an unchanged binding, source report, protocol and verified manifests.
Missing or changed provenance fails closed. This is trusted local experiment
provenance, not malicious-host authentication.

Both modes use all six protocol inputs, HF BF16 SDPA batch1 against vLLM BF16 batch2,
all model-vocabulary coordinates and the same predeclared per-row gates. The source
and reference children exit before rollout starts. Fresh mode also exercises actual
candidate failure/bitwise rollback; reuse performs only version7→8. New results
stay inside the owned acceptance directory and cannot overwrite historical evidence.
The run retains at most two snapshots for explicit reuse; after the guard returns,
cleanup requires confirmed owner exit and verified inactive snapshot identities.
Never treat a missing binding as permission to infer a source exit or retag content.

The unified final entrypoint's routing and provenance checks have CPU coverage;
it has **not** been rerun end to end on GPU. Source/export, rollback and numerical
confirmation were separately executed against the same two exact snapshot digests.
See [the evidence](../../docs/evidence/weight-sync.md) for those combined results,
the empirical numerical scope, resource accounting and unverified NCCL boundary.
# Anonymous inference phase bridge (source prepared; GPU UNVERIFIED)

`PhaseBridge` owns one accepted serial training snapshot and at most one anonymous
inference snapshot, including pending/unknown acquisitions. Its independent
`BridgeBudget` reserves future weights plus source, CPU staging, metadata, runtime
baseline and declared GPU occupancy. These declarations are not measured full-model
fit or kernel quotas. No disk checkpoint is created and the disk VersionStore API
is unchanged. The coordinator can remain Torch-free.

The caller supplies an independently trusted training/schema/config contract,
immutable base revision and exact config/tokenizer asset hashes. `describe_source`
only describes metadata; it does not make a source trustworthy. The exporter
reconstructs that description and compares its digest to the caller's pinned
contract before validating actual records. Source acceptance must come from this
bridge's original trusted non-spawning producer and its confirmed exit/reap.

Bootstrap uses the existing verified base rollout path for version0 evaluation
and first samples. After that rollout exits/reaps, the first actual learner update
initializes AdamW; only then capture S1. This bridge requires complete initialized
AdamW state and does not support an empty-optimizer S0 or fabricated optimizer step.

Sequence: capture_initial -> export_inference -> start_rollout/generate ->
retire_rollout -> advance_learner -> export_inference. All commands are fixed trusted
worker entrypoints supplied by the caller. `advance_learner(sample_versions=...)`
requires complete group headers bound by the caller to its actual worker batch;
known empty/all-stale headers return unchanged receipt before spawning or consuming
source. An unexpected actual worker no-op fails version acceptance after source
consumption, leaving the existing explicit unrecoverable gap. No Adam reset or
automatic rollback is supported. The initial science scheduler remains separate.

The CPU exporter validates finite policy/reference/moments, integral nonnegative
Adam step, Python RNG and a local CPU generator state. CUDA RNG validation here is
limited to pinned dtype/size/structure and trusted producer lineage; no CUDA query,
initialization or semantic device-state validation is claimed. Floating buffers
retain their pinned source dtype; inference FP32 floating state is explicitly BF16.

On acquisition/registration/start/reap/close uncertainty, the entire bridge is
poisoned and retains the original process, identity and reservation. No later learner
or engine may start. Retire old weights only after original rollout exit/reap and
identity-aware parent last-close. All RAM payloads, pending/quarantine and existing
disk checkpoints count toward two slots. This narrow bridge privately creates its
training owner with no disk checkpoints and rejects additional retained snapshots.

Prepared CPU fixtures use actual sealed FDs, original subprocesses, BF16 reference,
AdamW continuity and a packed tensor receiver. They have not yet been run. Actual
vLLM/CUDA/FSDP, full-model16GB host/24GB GPU eligibility, full Task11 science and the
original Goal remain open; separate exact authorization is required for execution.


## Bridge CPU status after bounded validation

This status supersedes the bridge preparation-only execution statements above.
The [original bridge run](../../docs/evidence/raw/task11-ram-handoff-bridge.log)
recorded 20 individual passing weight-export cases before its supervisor failed
(exit125) on Pytest's numbered temporary-directory symlink. That failure remains
preserved. After local unnumbered fixtures were added, the separately authorized
[bridge-v2 run](../../docs/evidence/raw/task11-ram-handoff-bridge-v2.log) passed the
remaining46 cases in39.09s, including two real tiny AdamW/BF16-reference cycles and
a packed CPU receiver. This is cross-attempt20+46 PASS evidence; the original
66-case command did not finish successfully. The v2 log retains one Python3.12
multi-threaded-fork warning from the existing non-RAM RPC cleanup control.

The [v2 receipt](../../docs/evidence/raw/task11-ram-handoff-bridge-v2.json) confirms
owned-child cleanup, no remaining children and peak sampled charge1,502,236,672B
including8MiB reserve. Independent review is pending. The actual RAM vLLM loader,
full-model CUDA/FSDP/RNG behavior and host/GPU fit remain **UNVERIFIED**; CPU injected
receiver results establish none of those. Task11 science and the original Goal
remain open, with no new GPU authorization.

## Bridge stale-drop accounting

`PhaseBridge.audit["noop_dropped_stale"]` is a lifetime cumulative sample count
for valid submissions handled by the bridge's all-stale no-op path. The entire
header list is validated before the counter changes. Empty input adds zero;
invalid headers/target versions add nothing. Age exactly `max_staleness` is eligible.
An all-stale submission returns the unchanged receipt without consuming its source,
spawning a learner or changing the optimizer/version. The counter adds the number
of rejected samples across its complete groups.

The single caller records the counter before/after each serialized call and adds
that delta once to scheduler metrics. If any group is eligible, this bridge counter
adds zero; the actual learner's `dropped_stale` metric owns all rejected samples in
that accepted step. Aggregate that worker metric once, separately from the bridge
counter delta. Do not add mixed-batch header estimates as another drop count. The
bridge does not parse arbitrary worker stdout into trusted learner metrics.

Headers have no batch/sample identities, so this measures submissions, not unique
samples. Repeating a valid all-stale submission counts again. A mixed step retry
with its already committed target version is invalid and adds nothing; a valid
new target version is another submission, whose eligibility uses the current source
version. The caller owns any identity-based deduplication before submission.

The attempted focused RED run failed during initial memfd capture because of its
process file-size limit, before the counter assertions. It is **not meaningful
RED evidence**. ROOT authorized this narrow fix from source review; the three
unchanged focused tests and one separately gated GREEN run remain pending.
Independent review and actual vLLM/full-model/GPU validation remain open.

The single authorized focused GREEN run passed3 tests in12.47s with no warnings;
see [GREEN evidence](../../docs/evidence/raw/task11-ram-handoff-bridge-fix1-green.log).
Independent rereview remains pending. The earlier resource-failed run still provides
no effective assertionRED evidence; all failure records remain preserved.

Scoped CPU bridge review approved after I1 fix (9989494). Actual vLLM/full-model/GPU remains UNVERIFIED.

**Current (2026-10-26):** the CPU supervisor and its per-run execution gates are
retired; these tests run in `make test-ml` with the GPU hidden. The historical
supervisor logs above remain as evidence.
