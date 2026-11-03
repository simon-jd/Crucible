# Task 8 single-card FSDP2 learner

## Functional acceptance checklist (2026-10-23, before execution)

- Cost ceiling $0; local RTX4090 only. Run `bash scripts/test_learner_gpu.sh`.
- Wrapper uses the fixed `/var/tmp/rtx4090.lock`, descendant reaper, 4h hard limit.
- No installations/downloads; fixed `.venv-ml` Torch2.8.0+cu128, Triton3.4.0,
  transformers4.55.4, vLLM0.10.2. Shared cached Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- First generate four neutral completions (G=2); close and join the actual owned
  rollout worker and assert exit0 before initializing learner NCCL/CUDA.
- Shard all decoder layers then root using actual Torch2.8 FSDP2. FP32 masters,
  BF16 compute, FP32 reduce, FP32 AdamW states; frozen BF16 reference.
- Two optimizer steps, accumulation microbatch1, real parameter updates, raw
  logprob agreement, fixed reference, stale rejection and successful-step versions.
- Fail on worker cleanup failure, initialization/training exception, nonfinite
  loss/gradients, unchanged parameters, incorrect stale metrics, resource overflow.
  No performance/quality experiment, no retry loop, no remote resource fallback.
- Expected persistent learner state ~9GB plus short-sequence activations/casts;
  measure actual peak allocated/reserved GPU memory and process peak host RSS.
- Reserve 100MB project-local caches/evidence. Zero checkpoint/snapshot files,
  zero resident services. No full optimizer serialization or atomic-save temp file:
  approximately 6GB policy+optimizer state alone would exceed the current ~3.62GB
  project headroom. In-memory state is the Task9 integration boundary.
- Native host execution (no container image digest). Driver/CUDA/NCCL and fixed
  model/runtime versions recorded; no OFED/multi-node claim.

## Actual local GPU result

`bash scripts/test_learner_gpu.sh` passed on its first run: **1 passed in 14.58s**.
The approved fixed guard returned exit0. [Complete output](raw/task8-gpu-attempt1.txt)
includes the actual worker PID1906807, joined exit0 before learner initialization,
and explicit NCCL teardown. The guard's process-exit/reaper boundary completed
before the post-run [GPU snapshot](raw/task8-gpu-after.csv): 2900MiB in use,
driver595.99.02, RTX4090 24564MiB. No own learner service remains.

| Observed value | Result |
| --- | --- |
| Torch / CUDA / NCCL | 2.8.0 / 12.8 / 2.27.3 |
| Triton / Transformers | 3.4.0 / 4.55.4 |
| Model parameters | 494,032,768 |
| Actual FSDP2 | 24 decoder layers then root; world1 DTensors |
| Policy / gradient / Adam dtype | FP32 / FP32 / FP32 |
| Forward / frozen reference dtype | BF16 / BF16 |
| Adam tensor state | 3,952,263,304 bytes |
| Initial rollout-vs-learner max raw-logp error | 0.09025097 (limit0.15) |
| First-step inspected parameter max change | 0.0000101327896 |
| Optimizer steps | 2, versions7→8→9 |
| Each step | 4 accepted samples, 32 generated tokens, 2 stale drops |
| Following all-stale step | 4 drops, no optimizer step, version stays9 |
| Learner peak GPU allocated / reserved | 10,091,039,232 / 12,876,513,280 bytes |
| Learner process peak host RSS | 4,127,862,784 bytes |
| Checkpoints / persistent services | 0 / 0 |

Each step accumulated four real forward/backward microbatches. Optimizer pre-hook
checked FP32 DTensor gradients; optimizer state was inspected after actual update.
Reference parameters were frozen/excluded from optimizer, with an inspected
reference tensor bitwise unchanged. No inference from fake FSDP flags is used.
The rollout phase separately requested vLLM0.80 utilization and released its
worker before the measured learner phase. Its actual runtime cache setup remains
Task7's reviewed worker setup; learner actual process uses Task8's explicit helper.

## CPU and resource evidence

Initial [formula RED](raw/task8-red.txt) and [trainer RED](raw/task8-train-red.txt)
failed because modules did not yet exist. First focused GREEN was12 tests.
Self-review added a regression for KL0 avoiding unused exponential overflow;
its [RED](raw/task8-kl-ablation-red.txt) failed before that fix. Final focused
[Torch CPU run](raw/task8-cpu-final.txt) passed18 tests in1.11s, including
the explicit fp32 reference and gradients (atol1e-6), nonuniform-length token
accumulation versus a whole-batch optimizer update, causal/EOS alignment,
clipping, KL ablations, stale/future/complete-group checks, fixed targets,
optimizer ownership and cancellation on nonfinite gradients. These use tiny
fixture models and do not substitute for the actual Qwen run above.

The default full CPU gate and resource totals are recorded separately in
[full CPU output](raw/task8-default-cpu-final.txt) and
[final resource preflight](raw/task8-resource-final.json). Initial inherited
HF environment was correctly rejected by preflight; the
[failure record](raw/task8-preflight-inherited-failure.txt) documents correction
to the approved shared cache without any reinstall.

The once-only final default CPU gate passed **568 tests, 4 skips in 65.66s**;
two Torch-dependent modules and two explicitly opted-in GPU tests were skipped.
The dependency-light runtime checks ran here. Ruff and shell syntax passed.
Final project allocation was **10,376,639,959 bytes**, with **3,623,360,041 bytes**
remaining under the decimal 14GB cap; free filesystem space359,101,939,712 bytes.
Task8 runtime cache directories contained zero file bytes after exit; NCCL file
rendezvous was removed. Task7 runtime cache files remained619,468 bytes, counted
in the project total. Relative to preflight, total project growth was187,434 bytes,
well below the100MB reserve (the total also includes concurrent root evidence).

No model/runtime downloads, no training state written into shared model cache,
and no checkpoint or atomic replacement temp state were created. A full FP32
policy plus Adam state would be about5.93GB before serialization overhead,
exceeding the remaining project allocation; this implementation retains state
in process memory and leaves persistence/transport to the later Task9 budget.
Compiler, driver, temporary and evidence files count toward the14GB allocation.

Scope: functional local native-host evidence only. Rewards in the acceptance
are controlled fixtures, not verifier/held-out evaluation. No speedup, held-out
quality, multi-rank synchronization, multi-node, or Task9 publication claim.
Multi-rank execution is explicitly rejected; no container digest/OFED claim
applies to this native single-card run.

## Review fix 1: failed-phase and finite-update boundary

Before this fix, an optimizer could partially update and raise while leaving the
learner reusable under the old version, and finite-gradient SGD overflow was not
caught after the update. These review findings supersede the initial fatal-phase
claim; initial success evidence did not exercise those cases.

Fix acceptance checklist (before the additional GPU run): keep the same fixed
model/runtime/shared lock and $0 budget; no new installation/checkpoint/snapshot.
Validate initial and pre/post-update local DTensor parameters/buffers, reference,
and actual AdamW state/options for finite values. Retain the two successful steps,
then inject a post-hook exception **after one real additional AdamW update** and
require failed-phase guards, unchanged diagnostic committed version, gradient
cleanup, NCCL teardown and process exit. This third update is intentionally invalid
and never reported as a successful version. Stop on any unexpected result.
One additional GPU run is justified by changed real DTensor/AdamW validation and
failure cleanup; no performance claim. Existing GPU evidence remains preserved.

Fix result: [exact RED](raw/task8-fix1-red.txt) reproduced nine failures before
implementation, including real SGD mutation-then-throw, unmodified SGD finite
`lr=1e20`/gradient1e20 overflow, and nonfinite optimizer momentum state. The
[final focused Torch CPU run](raw/task8-fix1-cpu-final.txt) passed **33 tests in
1.22s**; the dependency-light learner-only collection passed **2 tests, 3 skips**.
The earlier568-test full suite was not repeated. [Ruff](raw/task8-fix1-lint.txt)
and shell syntax passed. CPU fault injections also cover initial gradient-clear
failure, FSDP-control cleanup callbacks, preserving both execution/cleanup errors,
refusing nested step/publication during execution, and safe prevalidation retries.
The control callback mocks are state-machine tests, not FSDP evidence.

The single additional [actual GPU run](raw/task8-fix1-gpu.txt) passed **1 test in
14.55s**, fixed guard exit0. Rollout worker1973023 joined exit0 before learner
initialization. Actual local FP32 DTensor/master/Adam state validation passed for
two updates with versions7→8→9. The injected post-AdamW hook failure then produced
`phase_status=failed`, `last_committed_version_diagnostic_only=9`; all further
step and public policy/optimizer/version access were rejected. The third changed
state was not returned as a successful policy. Gradients were cleared, the owned
NCCL group destroyed, and the owned process exited before the guard completed.
The original GPU result is retained separately; no performance comparison is made.

Whole fix-run peak GPU allocated/reserved: **10,086,451,712 / 12,922,650,624 bytes**;
peak host RSS **4,128,903,168 bytes**. The [final resource check](raw/task8-fix1-resource-final.json)
passed with project10,376,889,315 bytes and3,623,110,685 bytes remaining under14GB;
free filesystem358,665,793,536 bytes. Task8 runtime had0 file bytes after exit;
no checkpoint/state snapshots, installs, downloads or ABI changes.

The public API now enforces the failed-phase boundary: guarded policy/optimizer/
policy_version accessors require `ready`; last_committed_version is explicitly
historical diagnostic data. Pre-execution group validation can be retried; every
uncertain execution/cleanup failure is fatal to the phase. Supported finite-state
inspection is SGD/AdamW tensor/numeric structure using local DTensor tensors,
without full-state gathering/copying. Task9 must obtain state through guarded
accessors afresh and discard previously cached references on failure; Python
cannot revoke aliases already held by a caller. No reset/rollback is provided.
