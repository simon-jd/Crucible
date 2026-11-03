# Atomic versioned weight synchronization

Task 9's actual single-RTX4090 evidence verifies complete learner weight export,
version activation, rollback and full-vocabulary numerical compatibility for the
fixed Qwen2.5-0.5B/runtime/input profile. Source, rollback and independent numerical
confirmation were **separate runs bound to the same two content digests**, not one
fresh execution of the final combined entrypoint. Multi-rank NCCL and multi-node
operation remain unverified; actual two-process gloo is CPU transport evidence.

Baseline: `e1978f7`, approved Task 7/Task 8 interfaces. The immutable numerical protocol
was committed before confirmation in `a999bf6`. No packages, Torch/NVIDIA ABI, paid
resources, remote/second GPU or resident service were introduced.

## Complete state and active-version boundary

The real source uses 494,032,768 parameters in 290 unique tensors, including the
Qwen tied embedding/head mapping. `WeightSync.push` transfers the complete expected
name/shape/dtype set, with no implicit casting or omission. Explicit guarded learner
export converts FP32 masters to BF16 inference weights. All weights and copied
config/tokenizer assets are hashed into an owned manifest and safetensors index.
Readonly files and pre/post-load hashes protect against accidental changes in the
single-owner lifecycle; this is not malicious-host attestation.

Push prepares a candidate, while `engine.load_weights(version)` activates only
after actual worker acknowledgement and content checks. It closes/joins the old
worker before creating the next. Confirmed candidate cleanup permits verified
rollback; unknown startup/cleanup ownership retains handles and fails closed.
Lost distributed final ACK can leave different candidate catalogs, so no distributed
atomic-commit guarantee is claimed. The previous active engine is untouched by push.

## CPU evidence

The single final default Python gate passed **579 tests, with 6 explicit skips**,
in 66.18 seconds. Three module skips reflect no Torch in `.venv`; three GPU tests
require explicit opt-in. Actual Torch/gloo and GPU evidence below is separate.

The three exact brief-named tests were written before implementation:
`test_logits_match_after_push`, `test_version_increments`, and
`test_failed_push_keeps_previous_version`. RED failed with the expected missing
implementation module. Actual two-process gloo checks normal complete payload/ACK,
receiver schema/write rejection, interrupted source, and receiver-process death
after catalog commit; the latter poisons the owner and preserves old bytes.

Torch-focused coverage totals 25 distinct cases across a 21-case run, a four-case
targeted run and a two-case targeted run (overlaps; not one 25-test execution).
Other boundaries include full schema/dtype/nonfinite checks, content corruption,
unknown worker ownership, fresh learner phase guards, a whole-transaction deadline,
two-snapshot limit, active pruning, discarded-version reuse, contract changes,
cleanup error preservation and mismatched reopened store contracts.

Finalization added 11 pure-stdlib tests for fresh content binding, exact reuse
rejection and real shell routing with external effects stubbed. Their own RED was
the missing acceptance helper, followed by GREEN. These tests do not exercise GPU.

- [Original RED](raw/task9-cpu-red.log), [21-case GREEN](raw/task9-cpu-focused.log)
- [Lifecycle targeted](raw/task9-cpu-lifecycle-focused.log), [reopen contract](raw/task9-cpu-contract-focused.log)
- [Binding RED](raw/task9-binding-red.log), [11 binding/routing tests](raw/task9-binding-green.log)
- [Final default Python CPU gate](raw/task9-cpu-final.log)

## Actual GPU lineage and preserved failures

Every GPU command used the fixed `scripts/with_gpu.sh` shared lock and owned-worker
cleanup. Another program can ignore that lock; it does not guarantee exclusive
use of every GPU byte. Attempt1 hit CUDA OOM during AdamW initialization with an
unrelated process using 12.32 GiB. Its source exited and guard returned 1; no vLLM
started. Only the inactive owned snapshot was cleaned after confirmed exit. An
in-guard 82%-free-memory check was then added before model work.

Attempt2 completed a controlled two-completion FSDP2/AdamW step 7→8. Parameter max
change was 0.00010120868682861328; published BF16 reference logits changed by 18.53125.
Both complete exports succeeded. An actual post-update exception then failed the
learner; fresh publication access rejected that phase, with no failed version saved.
The source process joinedexit 0 before actual vLLM0.10.2 V1 startup.

The initial BF16 eager reference comparison at `atol=.25, rtol=.01` **failed**:
2,135/303,872 coordinates, maximum error 0.46875. This was retained, not reclassified
as passing. The worker joined 0 and guard returned 1. Source mixed BF16 compute/FP32
masters versus that exact published BF16 eager reference also differed by maxima
.3125/.4765625. Full FSDP vectors were not saved, so later full pairwise FSDP metrics
are unavailable. No unrounded-master/BF16-publication equality is claimed.

The approved diagnostic reused those exact snapshots. Separate exiting processes
computed HF BF16 SDPA and FP32-on-exact-BF16-weights (TF32 off, highest matmul
precision), followed by actual vLLM. It verified load order 7→8(injected failure)→7→8,
bitwise equality of **all** rollback logits, and all four workers joiningexit 0;
outer guard returned 0. Diagnostic pass meant lifecycle success, not the original
numerical threshold passing. Complete eight-array dump and pairwise metrics remain.

HF eager versus SDPA, fused projections and residual/norm paths differ: installed
HF Qwen rounds residual addition before norm, whereas vLLM's fused add/RMSNorm
normalizes the FP32 sum before rounding. Higher-precision computation was diagnostic
and did not automatically meet the original tolerance. Root selected BF16 SDPA,
matching the learner factory, then fixed a new empirical bound **before** testing
six independent new inputs. No source training or passed rollback was repeated.

- [Attempt1 original OOM](raw/task9-gpu-attempt1.log), [attempt2 original failure/source facts](raw/task9-gpu-attempt2.log)
- [Diagnostic log](raw/task9-gpu-diagnostic.log), [full matrix and process records](raw/task9-diagnostic.json)
- [Eight complete diagnostic arrays](raw/task9-logits.npz), [lifecycle summary](raw/task9-attempt-lifecycle.json)

## Independent numerical confirmation — accepted scoped gate

[Frozen protocol](raw/task9-confirmation-protocol.json), SHA256
`a9d94ba329dddb322156549bd1492a26d8387062d7585fb292b3ff0bd7455b4c`, contains six new
fixed prompts and versions7/8. HF exact-published-BF16 SDPA uses batch 1; actual
vLLM0.10.2 V1 BF16 raw_logits uses batch 2. All151,936 model-vocabulary coordinates,
including padding beyond tokenizer entries, are compared at the exact final
input position. Token IDs/positions must match, all values must be finite, and
every prompt/version row must satisfy:

- `abs(actual-reference) <= .375 + .01*abs(reference)` for every coordinate;
- RMS error <= 0.10 and mean absolute error <= 0.08.

This empirical compatibility bound was calibrated on two prior inputs; it is not
a universal BF16 guarantee, arbitrary-batch guarantee or held-out task-quality claim.
The new inputs confirmed it without further threshold changes. All 12 rows and
1,823,232 coordinates passed; zero coordinate failures. Maximum row errors:

| Version | Max absolute | Max RMS | Max mean absolute |
|---|---:|---:|---:|
| 7 | .33984375 | .082173147 | .066404797 |
| 8 | .21875 | .041885413 | .032092274 |

Reference PID3812780 and actual workers3814235/3817609 joined 0; the original private
session confirmed **outer guard exit 0**, `1 passed in 37.16s`. Root independently
recomputed all thresholds from the saved arrays in float64. Confirmation followed
source and rollback as separate runs, linked by exact digests:

- v7: `9eee1e3b599f57c1368bb29a336a132e913b22c70de714a47e7355671a9d3f51`
- v8: `4d3edc9fbfe4b97581ed36a523b4a1e48a1283f08ed539e281f1ba4c8b640e27`

[Confirmation JSON](raw/task9-confirmation.json) retains all per-row metrics, exact
IDs/positions, profile and exits. [Complete four arrays](raw/task9-confirmation-logits.npz)
are 6,637,677 bytes, SHA256
`93728bd0104655460bca61d70ac6abd2439cf87b94b390c5d2d6d5bbcb8c3f00`.
[Original confirmation log](raw/task9-gpu-confirmation.log) is preserved.

## Reproduction and limits

```bash
.venv-ml/bin/python -m pytest -q rl/sync/tests/test_sync.py
.venv/bin/python -m pytest -q rl/sync/tests/test_acceptance.py
bash scripts/test_sync_gpu.sh          # fresh source, empty owned run directory
bash scripts/test_sync_gpu.sh --reuse  # only an unchanged bound owned run
```

The final default selects learner-compatible SDPA/fullV/six-input gates. Fresh
runs write their **own** content binding to the frozen profile; they need not match
the historical snapshot digests. Reuse requires the exact source-report hash,
confirmed source exit 0, base/protocol identity and currently verified manifests.
New run outputs stay under `.cache/task9/acceptance` and cannot overwrite historical
raw evidence. The final unified GPU orchestration was CPU-covered but **not rerun
end to end**. Actual component evidence above is combined, not a claim about an
unexecuted fresh command. NCCL multi-rank, multi-node, throughput and quality remain
unverified; no Go/Task14 gate was rerun for this Python-only change.

## Resources

Each BF16 payload is 988,065,536 bytes; both total 1,976,131,072 bytes plus copied assets.
Largest embedding:151936 × 896 × 2=272,269,312bytes. Per-tensor save and directory rename
avoid a third whole-copy temporary peak. At most two snapshot/staging directories
are allowed; no automatic pruning. No optimizer state or snapshots enter Git or
the shared immutable HF base cache.

Measured source peak GPU allocated/reserved:10,089,074,176/12,924,747,776bytes;
host peak RSS4,121,722,880bytes. Initial project10,377,210,799bytes plus2.1GB planned
fit14GB. Diagnostic and confirmation incremental preflights reserved30/50MB with
the same two snapshots. Resource snapshots and final verified cleanup are recorded
with this task's raw evidence. No unrelated or Task14 cache is removed.


After all owned guards/children were confirmed terminal, the two snapshots were
reverified against the frozen protocol digests and their exact manifest text and
source report archived. Only `.cache/task9/acceptance` was removed: 2,013,921,599
logical bytes, including the two inactive snapshots and owned reference tensors.
No shared HF or Task14 cache was touched. Retained checkpoints: **0**; owned
resident services: **0**. Project bytes changed from **12,687,864,826** to
**10,674,046,769**; a further 25 MB Git/evidence reserve passed preflight.

- [Before cleanup](raw/task9-resource-before-cleanup.json), [after cleanup](raw/task9-resource-after-cleanup.json)
- [Verified cleanup boundary](raw/task9-cleanup.json), [exact owned manifest contents](raw/task9-owned-manifests.json)
- [Original source report](raw/task9-source.json)

## Review fix round 1 (CPU only)

The four Important review findings are fixed in the existing sync/rollout files:

- A store reserves an engine identity before initial construction. Release checks
  that identity, so repeated close of an old engine cannot clear a successor's pin.
  The original constructor exception exposes `recovery_engine`, also retained by
  the store. Confirmed-clean failure permits retry; uncertain cleanup retains the
  partial worker and reservation until `close()` confirms exit. Missing cleanup
  handles remain blocked. This coordinates one store object in one process; it is
  not a filesystem or cross-process lease.
- Actual vLLM rejects owned non-BF16 floating schemas before spawning a worker and
  checks again before LLM construction. Generic publication and injected CPU
  backends still retain exact FP32/FP16 dtypes. This narrows the actual loader to
  the already accepted BF16 profile and introduces no new inference dtype claim.
- Final directory fsync and the return from push both check the shared absolute
  deadline. A late committed publication remains on disk with its persistent
  version identity; the store/transport fails closed in doubt, preserving old
  active bytes. Synchronous I/O still cannot be forcibly interrupted.

[New regression RED](raw/task9-fix1-red.log): 7 failed, 25 deselected.
[Initial focused GREEN](raw/task9-fix1-focused.log): 7 passed, 25 deselected.
After three additional boundaries and self-review, the [affected CPU gate](raw/task9-fix1-cpu.log)
passed **82 tests in 7.19 s**: the entire sync module (35 cases), rollout contracts,
RPC/cleanup and worker environment tests. This includes actual two-process CPU
**gloo**, not NCCL. Self-review then moved child schema validation after cache
configuration, still before vLLM import/construction; the
[loader/environment follow-up](raw/task9-fix1-worker-focused.log) passed **4 tests,
32 deselected in 1.34 s**. These overlap the 82 cases; they are not 86 distinct tests.
[Ruff and format checks](raw/task9-fix1-lint.log) pass for all five changed Python files.
The unchanged 579-test repository gate and GPU components were not rerun. Historical
BF16 numerical results and immutable protocol evidence remain unchanged; the final
unified GPU entrypoint remains unexecuted end to end.

The successful GPU logs include three known, **non-blocking warnings for this
correctness profile**: a deprecated NCCL environment setting; FlashInfer sampling
unavailable, with the logged fallback sampler used; and eager execution disabling
cudagraphs. No throughput claim depends on those paths, and no package/ABI upgrade
or extra GPU run was made to suppress the warnings.

The first [resource preflight](raw/task9-fix1-resources-env-rejected.json) rejected
inherited HF environment paths; the [explicit shared-cache preflight](raw/task9-fix1-resources-before.json)
passed at 10,707,516,873 project bytes with a 10 MB incremental reserve. The initial
RED used pytest's default `/tmp` directory (6,315 logical bytes); subsequent tests
used project-local temporary storage. [Exact owned cleanup](raw/task9-fix1-cleanup.json)
removed both that RED directory and the fix-round project temporary directory
(33,668 logical bytes), after all test processes returned. No unrelated/Task14
cache or shared HF content was touched. [Post-cleanup resources](raw/task9-fix1-resources-after.json):
10,707,557,038 project bytes, zero model checkpoints and resident services; 10 MB
additional Git/evidence reserve still passes the 14 GB budget. All work was local,
CPU-only and $0.
