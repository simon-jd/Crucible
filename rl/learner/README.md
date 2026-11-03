# Single-card GRPO learner

`group_advantages(rewards)` normalizes along the final (group) dimension with
`(r - mean) / (std + 1e-6)`. The std is population std (`correction=0`); singleton
and equal-reward groups produce zero. `scale_by_std=False` is the mean-only ablation.

`grpo_loss(logp_new, logp_old, logp_ref, adv, mask, clip_eps=0.2, kl_coef=0.001)`
uses the clipped ratio surrogate and k3 KL `exp(ref-new) - (ref-new) - 1`, averaged
across valid generated tokens. `adv` may be one scalar per completion or a token
matrix. `mask` is boolean; padding is excluded before exponentiation. Old/ref logps
and advantages are detached targets. KL ablations are `0`, `0.001`, and `0.04`;
zero disables the KL computation. Nonfinite valid inputs/losses/gradients fail.

`filter_stale(samples, current_version, k)` preserves order and returns
`(accepted_samples, dropped_count)`. Age exactly `k` is allowed. Negative/noninteger
versions and future versions are errors.

## Data and training API

`RolloutGroup(batch_id, group_id, prompt, samples, rewards, sampling)` represents
one complete group from one `RolloutEngine.generate` call. Preserve completion
order when attaching rewards. `batch_id` identifies that generate call;
`group_id` identifies the prompt occurrence within it. Do not merge repeated
prompt IDs from separate calls. Construct each group by slicing the original
ordered output into the **same G used for generation**, and configure that G via
`LearnerConfig(group_size=G)`. Truncated/mixed-version groups and duplicate group
identities within one step are rejected. Prompt IDs must match each Sample.
Sampling metadata is required; only temperature1/top_p1/top_k-1 is supported.
Non-neutral behavior probabilities require a later explicit algorithmic choice.

`Learner.step(groups)` accepts one optimizer-step batch, removes stale groups,
computes advantages on each full accepted group, scores prompt+generated IDs with
the pinned tokenizer, applies causal shift, and accumulates gradients. Sample text
is never re-tokenized; emitted EOS IDs are included. Overlong sequences are
rejected rather than truncated. Empty/all-stale batches return zero optimizer
steps without advancing the policy version. Each microbatch's loss is weighted
by its valid-token count divided by **all accepted tokens in the step**. The
optimizer updates once, then `policy_version` increments. Metrics include loss,
accepted samples, dropped stale samples, effective tokens and optimizer steps.

`Learner` also accepts ordinary PyTorch modules for CPU correctness tests. Policy
and reference must be separate, and the optimizer must own exactly the policy's
parameters. Both models use eval mode for deterministic scores; policy autograd
remains enabled and reference parameters are frozen. Reference stays fixed across
all steps. Supported optimizer structures are SGD and AdamW (including subclasses
with the same tensor/numeric state structure); unknown optimizers/state objects
are rejected. Initial and every pre/post-step policy/reference parameters and
buffers, optimizer state, and numerical options must be finite. DTensors are
checked using local tensors without gathering or copying the whole state.

The phase is `ready`, then `executing`, then `ready` only after validation, update,
and all cleanup succeed. Any execution exception, nonfinite gradient/result,
initial gradient-clear failure, or cleanup/FSDP-flag restore failure makes it
permanently `failed`. All cleanup actions are attempted independently; multiple
errors are preserved in a BaseExceptionGroup. No failed update advances the
committed version or returns success. There is no rollback or reset of a failed
phase: the caller must tear down the process and recover through a separate phase.
Input/group rejection before model/optimizer execution is reusable; empty batches
still validate state and cannot hide prior external corruption.

## Actual FSDP2 entry and lifecycle

The GPU process must execute `rl.learner.runtime.configure_runtime()` **before**
importing Torch, Transformers or CUDA consumers. Package import itself is inert.
The helper sets all compiler/driver/temp paths inside `.cache/task8/runtime`, and
uses only the shared offline Hugging Face cache. Do not run `uv sync`: preserve
`.venv-ml` and its separately locked rollout extension.

Under `scripts/with_gpu.sh`, first finish/close the rollout worker and prove its
process joined exit0. Then create one caller-owned NCCL process group (world1,
rank0, CUDA0; local file rendezvous). With this initialized, call:

```python
learner = load_fsdp2_learner(
    immutable_local_model_path,
    LearnerConfig(group_size=2, microbatch_size=1, max_staleness=1),
    policy_version=loaded_checkpoint.policy_version,
    learning_rate=1e-5,
)
for groups in prepared_optimizer_batches:
    metrics = learner.step(groups)
```

The factory uses the offline path for both model and tokenizer; caller must bind
that immutable path/revision to the starting policy version. It requires Torch
2.8.0, shards each Qwen decoder layer then root with actual `fully_shard`, and creates
AdamW **after** parameters become DTensors. FP32 master parameters and optimizer
moments, BF16 compute, FP32 gradient reduction; the reference is separate BF16.
Gradient sync is enabled only on the last microbatch and restored on exit.
Multi-rank is explicitly rejected: this task validates single RTX4090 only.

`learner.policy`, `learner.optimizer`, `learner.policy_version` are guarded, read-only
accessors for the in-memory Task9 boundary. Each raises if the phase is executing
or failed, as does any further `step` call. `phase_status` and `failure_reason`
remain available for diagnosis. `last_committed_version` is **diagnostic only**:
failed/executing weights may have changed and must never be published under that
number. Task9 must acquire the public state/version afresh in a ready phase and
must not bypass these guards via private fields. Python cannot revoke tensor or
optimizer references a caller cached before failure; those references must be
discarded, never reused for publication. This API is a single-caller phase owner,
not a concurrent snapshot/publication transaction.
Model state_dict tensors are DTensors. Task 9 provides an explicit, streamed
BF16 inference export with fresh ready-phase guards and content verification; see
[weight synchronization](../sync/README.md). This is not an optimizer checkpoint.
Do not full-materialize/serialize other states without a new peak-memory/disk budget.
At phase end destroy the owned process group in `finally`, then exit the owned
learner process before a new rollout phase; the fixed GPU guard reaps descendants
before releasing its lock. Merely deleting a Python learner variable does not
prove GPU context cleanup. Task9 implements state handoff across phases through owned versioned snapshots.

## Validation

```bash
.venv-ml/bin/python -m pytest -q rl/learner/tests/test_grpo.py rl/learner/tests/test_train.py
bash scripts/test_learner_gpu.sh
```

The GPU script has a read-only resource preflight and fixed shared lock. The
acceptance consumes actual neutral Qwen0.5B rollouts, checks worker exit, runs two
FSDP2/AdamW steps, verifies a parameter update and immutable reference, checks raw
logp agreement and staleness, and measures GPU/host memory. It uses controlled
fixture rewards; it is not held-out quality evaluation. Default `.venv` collection
skips Torch-dependent tests and all GPU execution. Results and scope are in
[learner evidence](../../docs/evidence/learner.md).

## Explicit anonymous RAM training-state handoff

`rl.learner.state_handoff` adds a Linux parent-owned memfd boundary for a trusted,
non-spawning learner worker. `SnapshotOwner.capture(command, version=..., binding=...)`
creates an empty descriptor, passes it to one child, and accepts its immutable
artifact only after the original `Popen` child has exited successfully and been
reaped. Acceptance checks descriptor device/inode, length, all four seals and
streamed SHA256. `capture(..., source=owned_snapshot)` reserves a second output
slot and passes the input and output descriptors to a fresh child, allowing
restore → real optimizer step → new sealed capture → exit/reap. The parent can
then explicitly release the old snapshot. `consume(snapshot, command)` supports
bounded inspection or other owned read consumers. Commands receive explicit
`CRUCIBLE_HANDOFF_*` fd, receipt, binding and budget environment values.

Only the owner may pass its descriptors to consumers. Do not duplicate them,
map them in another caller, spawn descendants, or retain external fd aliases.
Closing an owner fd does not establish that a checkpoint was released: every
owned consumer must be reaped first, and uncertain cleanup poisons the owner while
retaining its reservation. Pending/accepted RAM snapshots and declared existing
disk checkpoints share a maximum of two slots. A failed child, timeout, signal,
invalid seal/digest or oversized output never creates an accepted resume.

The child calls `capture_ready_learner(learner, fd, binding=..., budget=...)` while
it exclusively owns a ready phase. Capture uses fresh guarded policy/optimizer/
version access, validates exact named state, checks unchanged state before each
serialization pass and before sealing, and never substitutes diagnostic committed
versions. It serializes detached local tensors, preserving policy parameters,
all named buffers including nonpersistent buffers, tied aliases, fixed reference,
AdamW moments/step or SGD momentum, exact parameter group order/options, config,
version and supported RNG. The binding names immutable model/tokenizer revisions
and must declare `fixed-lr:no-scheduler:no-scaler:no-loader:no-numpy`. Other external
state is unsupported; optimizer subclasses/hooks, unknown fields, tensor-valued
options and differentiable/fused/capturable modes are rejected.

Restore uses a caller-constructed fresh `Learner` with matching config, optimizer
options, named parameter group order and expected policy version:
`learner.restore_handoff(fd, receipt, expected_version=..., binding=...)`. It enters
`restoring` before loading, blocks guarded access/step, validates the complete
schema before mutation, copies into owned tensors without dtype conversion, runs
state validation/cleanup, restores RNG last, and publishes `ready` only on complete
success. Any error permanently fails that learner and retains the original cause.
Built-in optimizer `load_state_dict` is deliberately unused because its parameter
names do not control positional loading and it can cast state. Loaded mmap views
are not returned or retained as optimizer tensors; the final process exit/reap
establishes that its mappings are gone.

Pinned Torch 2.8 loading remains `weights_only=True`, `mmap=True`,
`map_location="cpu"`, with private mapping of `/proc/self/fd/N`. There is no custom
unpickler, safe-global allowlist, disk checkpoint or whole-archive BytesIO. Saving
uses Torch's storage streaming and a save-only Pickler sink wrapper limiting
`data.pkl` before buffer growth; Torch retains its own storage persistent-ID logic.
The first serialization counts/discards all ZIP bytes; the second checks every
write against that exact count before writing. Storage-floor and metadata/item
bounds precede serialization. Default tiny limits are 131072 archive bytes in the
test owner, 32768 metadata bytes and 8192 bytes for one storage. Larger callers
must declare their own checked limits; the complete archive cap is not charged
as metadata. Planning charges retained payloads, each live mapped length again,
caller-declared live models/moments/process baseline, two largest-storage staging
buffers. Consume and advance share the restore-stage reserve, including two
storage-cap staging units before consumer launch. Sixteen metadata-cap units plus
135168 bytes conservatively cover pickle/hash buffers, bounded ZIP tails, record
objects and parser temporaries. These declarations are not allocator quotas.

Before any eager ZIP reader, a streaming preflight bounds the central directory
independently by metadata_cap bytes and min(100000, metadata_cap//512) records.
Names are ASCII archive/ names of at most128 bytes; central extras are at most28
bytes and ZIP64-only; archive/entry comments, split/concatenated archives and
ZIP64 extensible sectors are unsupported. Fixed ZIP64 end records and per-record
size/offset sentinels are checked. The save counting pass keeps only a bounded
directory tail and validates it before writing the output memfd; an earlier
storage-count bound limits writer directory construction. Default restricted
Torch loading and private mmap remain unchanged.

DTensor serialization stores plain local views and named global/local shape,
dtype, stride, exact rank0/world1 mesh and Replicate/Shard placement metadata.
Restore constructs moments on the destination parameter's matching mesh. Partial,
async local tensors, noncontiguous/partial-storage views and multi-rank are rejected.
Python and Torch CPU RNG are supported. A CUDA RNG path exists only when CUDA is
already initialized with exactly one generator and policy device `cuda:0`; CPU
execution checks only the safe initialized-context flag. CUDA execution, FSDP
resume and real-model memory fit remain **unverified**. CPU/Gloo DTensor checks
would establish only their narrow local path, never FSDP/GPU behavior.

These CPU tests run in `make test-ml` (GPU hidden via `CUDA_VISIBLE_DEVICES=`).
No real-model checkpoint, optimizer-lifecycle readiness, updated rollout export,
heldout quality, async/sync comparison, ablation or external baseline result is
established by them.

For an advance, `CaptureBudget` takes the maximum of two serial-stage charges:
restore retains the source payload plus its mapping and bounded restore overhead;
capture retains the source payload plus the new output payload and bounded staging.
It does not assume all three complete payloads coexist. This reduction is enforced:
restore's inner load/copy frame returns only owned RNG copies, then the production
path checks `/proc/self/maps` for the receipt's exact device/inode. Any remaining
mapping or inability to inspect mappings fails restore. Before sizing, writing and
sealing the next snapshot, capture requires the source fd/receipt and repeats that
proof. The parent still retains the source fd/slot through child exit/reap. This
process-local proof changes the charged stage peak; it does not establish actual
full-model fit or another process's mapping release.

### Explicit consuming stream continuation (tiny CPU verified; review pending)

`capture_stream(..., budget=SerialBudget(...))` and
`advance_consuming(source, ..., budget=...)` use a separate versioned tagged JSON
manifest followed by raw contiguous tensor records. The retained ZIP/mmap API
above keeps its behavior. The new path does not unpickle or mmap. Bounded JSON
bytes, nesting, node counts, integers, tensor shapes/strides/devices and complete
raw extents are checked before candidate tensor allocation. Exact destination
schema/options and aggregate CPU/GPU allocations are checked next; full candidate
finite values, Adam steps, RNG and aliases are validated before destination writes.
Primitive None/bool/int/finitefloat/string/list/tuple/dict distinctions are preserved.
The supported integer magnitude is at most 2**63-1; unsupported metadata is rejected.

A fresh trusted worker initially inherits only source and local control descriptors.
After successful restoration and completed copies it closes source aliases, sends
one bounded retirement ACK and blocks. The parent retains the original unreaped
child as its ownership anchor, verifies live identity and actual source descriptor
and mapping absence in that child, then proves it holds the sole remaining source
FD. Only after its final close retires the source does it create and transfer an
empty output memfd with SCM_RIGHTS. The worker then steps and captures. Acceptance
requires the same original child exit0/reap, identity/seals/size/SHA and bounded
stream format/version/binding validation. No ACK is an accepted checkpoint.

Consumption is irreversible: failures after retirement fail the run with no accepted
replacement and no optimizer reset, retry or rollback. Cleanup uncertainty poisons
the owner and retains tracked handles/reservations. A period with zero accepted
training snapshots is intentional. Disk, retained RAM, pending output and explicitly
declared future inference weights share the maximum of two checkpoint slots.
Workers must not spawn, externally transfer or reacquire source capabilities.

SerialBudget requires nonzero aggregate host baseline; explicit CPU candidate and
CPU destination/clone allowances; GPU baseline/candidate/destination/transient
allowances; retained payload and future-weight bytes plus weight slot. Actual
candidate CPU/GPU byte totals must fit BOTH candidate and destination allowances,
conservatively charging a full duplicate including scalar steps and RNG. Existing
model destination tensors may therefore be excluded from the separately declared
baseline, but cannot be omitted from both. Metadata reserves64*metadata_cap+262144B.
Host restore charges baseline+retained+weights+input+CPUcandidate+CPUdestination+
2*storage_cap+metadata; capture replaces input/candidate terms by output cap and retains destination
allowance. Their maximum is valid only across the enforced retirement barrier. VRAM
separately charges baseline+candidate+destination+transient against declared actual
single-card capacity, at most24GB. Full model fit still requires future measurement.

CPU transfers use bounded readinto chunks and one storage staging tensor; CUDA
transfer requires an already initialized single cuda:0 context and completed
synchronization. No CPU test queries CUDA availability/device count or initializes
CUDA. CUDA/FSDP/full-model behavior remains unverified. The supervisor
`stream` mode selects new tests plus one retained real AdamW control,120+5s,2GB,
4MiB anonymous reserve,1MB scratch and262144B file cap. Full Task11/Goal completion is not claimed. Minor M1 remains deferred to final review.


The single authorized [stream CPU run](../../docs/evidence/raw/task11-ram-handoff-stream.log)
passed44 new cases plus one affected retained AdamW control in13.64s. Its actual
28,830B version2 stream state and next loss matched uninterrupted training after
original producer exit/reap, fresh restore, verified source FD/maps retirement,
output grant, next-step capture/seal, child exit/reap and fresh inspection. The
[terminal receipt](../../docs/evidence/raw/task11-ram-handoff-stream.json) records
cleanup confirmed, no errors or remaining owned children, sampled aggregate RSS
plus4MiB reserve peak1,419,063,296B and terminal scratch files0B. Sampling is not a
hard memory quota or full-model fit evidence. Independent review is pending;
CUDA/FSDP, actual0.5B16GB fit, inference export/rollout and scientific experiments
remain open. Original static failure and subsequent successful checks are retained.
# Serial metadata reader for the inference bridge

`handoff_stream.read_metadata` shares structural validation with read_manifest but
does not admit tensor reconstruction. The original read_manifest still checks
source budget before parsing and actual CPU/GPU aggregate before decoding tagged
state. The new phase bridge supplies independent export/rollout budgets and keeps
the original serial input-retirement/output-grant and unrecoverable-gap semantics.
Source preparation is not an executed CPU or full-model/GPU result.


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

Scoped CPU bridge review approved after I1 fix (9989494). Actual vLLM/full-model/GPU remains UNVERIFIED.

**Current (2026-10-26):** the CPU supervisor and its per-run execution gates are
retired; these tests run in `make test-ml` with the GPU hidden. The historical
supervisor logs above remain as evidence.
