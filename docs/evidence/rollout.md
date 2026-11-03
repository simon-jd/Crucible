# Task 7 local vLLM rollout

Date: 2026-10-23 (America/Los_Angeles). Functional acceptance on the local RTX 4090 only.
No training, held-out quality, throughput comparison, paid/remote GPU or multi-node claim.

## Environment and resource boundary

The existing `.venv-ml` was extended, with vLLM 0.10.2, Transformers 4.55.4, Torch
runtime 2.8.0+cu128 (distribution metadata 2.8.0), Triton 3.4.0, NumPy 2.2.6 and all
existing NVIDIA libraries unchanged. [Installed inventory](raw/task7-installed.txt);
[full hash lock](../../requirements/rollout.lock). Public sources: PyPI metadata and
`files.pythonhosted.org` wheel URLs/SHA256 in the [wheel inventory](raw/task7-wheel-inventory.json).
Resolution cutoff was 2025-10-01; no Torch wheel was downloaded.

The prerequisite [local runtime record](ml-runtime/cuda-smoke.json) identifies RTX 4090
SM8.9, driver595.99.02, CUDA build12.8 and NCCL2.27.3 on this same host. This acceptance
uses the bare host rather than a container, so image digest/OFED are not applicable;
no cross-node networking was exercised. Model path is the shared offline snapshot:
`/home/postedism/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`.
Model/revision also appear in the actual loading logs; no model copy/download was made.

Before installation, [preflight](raw/task7-install-preflight.json) passed with project
7,386,764,561 bytes and 6,200,000,000 additional planned peak bytes, below 14e9.
120 added/replacement wheels: compressed982,279,908 / unpacked2,927,823,165 bytes,
verified from each wheel's ZIP central directory. Sequential no-cache installation
bounds extra final + one unpacked/copy staging + one compressed wheel +800MB compiler
reserve at5,504,806,519 bytes, inside the conservative6.2GB plan. This is an estimated
upper bound, not a continuously measured peak. No shared-path cache was used to escape
project accounting. [Installation log](raw/task7-install.txt),
[idempotent zero-install reapply](raw/task7-bootstrap-reapply.txt).
No checkpoints or resident services were retained. Post-task preflight is linked below.

An initial bootstrap check aborted before installation because it expected the runtime
local suffix from distribution metadata: `AssertionError: ('torch', '2.8.0', '2.8.0+cu128')`.
It was corrected to compare metadata2.8.0 and actual runtime2.8.0+cu128 separately.
No package was installed by the aborted check. The preserved install log is the successful run.

## RED and CPU contract path

Before engine implementation:
`.venv/bin/python -m pytest -q rl/rollout/tests/test_rollout.py`
failed collection with `ModuleNotFoundError: No module named 'rl.rollout.engine'`.
[Original RED output](raw/task7-red.txt). After implementation, the same focused command
passed17 tests; self-review added failure closure, immutable identity, invalid settings
and EOS/index ordering cases. [Focused GREEN](raw/task7-cpu-focused.txt):28 passed.
These CPU fake-backend fixtures verify contracts; they are not model evidence.

## Actual functional GPU acceptance and cleanup

All three actual runs used `bash scripts/test_rollout_gpu.sh`, whose foreground command is
`bash scripts/with_gpu.sh -- env ... .venv-ml/bin/python -m pytest -q -s rl/rollout/tests/test_rollout_gpu.py`.
The fixed guard acquires `/var/tmp/rtx4090.lock`, bounds the whole lifecycle to4h and
reaps its descendants before handoff; no neighboring GPU process was inspected or signaled.
The planned experiment was this single small-context functional test. Stop conditions:
any assertion/load/OOM/version/logprob/cleanup error, >300s initialization, >120s generation,
or guard4h limit; no scaling or benchmark sweep. Owned process termination is bounded
in the worker and guard retains isolation until descendant cleanup is confirmed.

Parameters: BF16; V1 in-process core, uni executor, eager; utilization**0.80**;
max_model_len256, max_tokens16, seed123, temperature0.8, top_p0.9, top_k-1;
2 ordered prompts × group_size3; loaded policy_version7. Same-call seeds produced
identical samples on two calls while n-way outputs exhibited stochastic diversity.
All six samples carried version7, mismatch8 was rejected, and each generated token
had a finite selected-token logprob. Eight chosen IDs from sample0 were independently
scored with FP32 Transformers after the rollout worker released CUDA. Maximum absolute
raw-logprob difference**0.0552303791**, below explicit tolerance0.15.

The first [actual run](raw/task7-gpu-attempt1.txt) passed1 test in19.59s, worker1592320
joined exit0, but upstream uni shutdown left an undestroyed NCCL process-group warning.
Self-review added explicit vLLM distributed-group cleanup, then the
[focused final run](raw/task7-gpu-final.txt) passed1 test in12.10s, same functional result:
`rollout distributed groups destroyed`, worker1595550 joined exitcode0, guard command exit0.
The fixed guard's successful return requires its kernel ECHILD cleanup proof, covering
helper/descendant processes beyond the explicit worker join. There is no resident server.
Durations are test runtime observations only; they are not performance comparisons.

Residual, unfiltered upstream/expected warnings in both post-cleanup runs: Torch2.8 NCCL
`TORCH_NCCL_AVOID_RECORD_STREAMS` deprecation (variable was not inherited and no setter
was found in installed vLLM source); optional FlashInfer absent, using the PyTorch-native
stochastic top-p/top-k sampler; eager-mode cudagraph-dispatch notice. None caused assertion
or teardown failure. The initial first-compile `_POSIX_C_SOURCE` redefinition warning
is preserved in attempt1 and did not recur with the populated project-local Triton cache.
No extra dependency or host patch was introduced merely to suppress warnings.

The explicit raw-logprob mode and shutdown APIs were checked against the fixed official
[V1 sampler](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/v1/sample/sampler.py),
[core client](https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/v1/engine/core_client.py)
and installed0.10.2 source. Returned logps are raw model log-softmax before non-neutral
sampling transforms; Task8 must use matching raw model/reference scoring. The
[API contract](../../rl/rollout/README.md) documents EOS/text differences and reproducibility.
Task9 runtime weight replacement/rollback remains unimplemented by explicit scope agreement.

## Final CPU gate

Before the subsequent cache-only boundary correction, frozen code: `env CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -m 'not gpu'`.
[Final CPU output](raw/task7-cpu-final.txt): **543 passed, 1 deselected in65.76s**.
Real model tests require explicit opt-in and are excluded from that CPU gate.

## Cache-boundary correction after the CPU gate

Self-review found CUDA_CACHE_PATH/XDG_CACHE_HOME were not explicitly set in the first
two actual runs. The default driver cache may have been written; its prior contents
and any increment were unmeasured and cannot be attributed to this task. The existing
shared default cache was not inspected or deleted. Those earlier runs therefore cannot
prove complete driver-cache accounting. The controller approved a narrow resource fix
after the full CPU gate, retaining that gate and using focused coverage for the delta.

The worker now configures CUDA/XDG/vLLM/Triton/Torch/CuPy/Numba/TMP caches before
heavy imports even for public API callers, and resets Python tempfile's cached directory.
The GPU wrapper sets the same paths before starting Python.
`.venv/bin/python -m pytest -q rl/rollout/tests/test_worker_environment.py`: **1 passed**;
[CPU output](raw/task7-cache-fix-cpu.txt), with inherited unapproved defaults deliberately
overridden. No second full CPU suite was run.

`bash scripts/test_rollout_gpu.sh` then passed **1 real-model test in12.16s**, guard exit0;
[cache-fix actual run](raw/task7-gpu-cache-fix.txt) prints every actual child cache path,
confirms the same functional/logprob results and worker1624450 joined exit0 after
distributed-group destruction. All cache paths are inside the project.
[Cache inventory](raw/task7-cache-final.json):619,468 bytes total runtime cache, including
zero bytes in the newly redirected CUDA/XDG directories.
[Post-task preflight](raw/task7-resource-final.json): project10,375,369,094 bytes,
free359,679,148,032 bytes, no checkpoints/resident services, within14GB.

## Review fix round 1 — absolute RPC deadlines and reused prerequisites

Review baseline `dd959d2`; this fix addresses Important I1 and Minor M1. M2 remains
explicitly nonblocking and disclosed. The controller approved a focused `_rpc.py`
responsibility split and one real-GPU integration for the changed wire. No installations,
stack replacement, full CPU rerun, benchmark, cloud action or warning-suppression patch.

**I1 RED:** the review's CPU-only `multiprocessing.Pipe` partial-frame reproduction was
repeated against the original `_receive(timeout=.05)`, with an external 0.500 s safety
alarm: `RuntimeError external safety timer fired elapsed= 0.5`.
[Original regression output](raw/task7-fix1-red-partial.txt). Initial readability followed
by blocking `Connection.recv()` bypassed the operation deadline.

**I1 fix:** application messages now use nonblocking owned socketpairs and an 8-byte
unsigned length header. Each complete frame is bounded to 64 MiB before reading its
body; primitive-only JSON, finite values, 65,536-character strings and depth12 bound
codec work. Validation/encoding checks an absolute deadline incrementally, and bounded
JSON decoding checks the same deadline before/after decoding. Startup has one300 s
parent deadline; each generation has one120 s deadline covering the whole request send
and response receive, including serialization and partial transfers. No deadline is
reset by a new chunk or initial readability. Late decoded responses are rejected.
The worker retains the existing cache containment, loaded-policy identity and raw
chosen-token semantics; JSON token-key strings are restored to integer token IDs.

Graceful close control has at most0.1 s, after which the transport is closed regardless
of send success. Nested `finally` cleanup ensures even a transport-close exception
cannot skip the owned Process join/TERM/KILL sequence. Cleanup uses the existing
30/10/10 s grace windows. Timeout/error workers reject new work; owner handles remain
available when cleanup fails. Abnormal exits remain visible errors with the original
operation/transport failure in exception context. The outer fixed guard still owns
all descendant cleanup and GPU handoff.

Focused CPU commands and results:

- `.venv/bin/python -m pytest -q rl/rollout/tests/test_rpc.py rl/rollout/tests/test_bootstrap_pins.py rl/rollout/tests/test_rollout.py rl/rollout/tests/test_worker_environment.py`
  → **50 passed in0.45 s**. [Output](raw/task7-fix1-cpu.txt).
- Self-review additionally hardened the transport-close exception path, then ran
  `.venv/bin/python -m pytest -q rl/rollout/tests/test_rpc.py`
  → **18 passed in0.47 s**. [Final RPC output](raw/task7-fix1-rpc-final.txt).
  Real sockets cover partial headers/payloads, slow-drip absolute deadlines, full write
  buffers, invalid/oversized lengths and late codec results. Actual CPU-owned children
  verify blocked generation/partial responses reach cleanup, and a TERM-ignoring child
  is KILLed/reaped despite a blocked close write. Test-only short grace windows and a
  1 s external safety alarm keep failures bounded. A transport-close exception still
  reaps the actual child; an unconfirmed-owner fixture retains its handle/refuses work.
  The tests never load a model or use GPU hardware.
- `.venv/bin/python -m ruff check rl/rollout` → **All checks passed!**;
  `bash -n scripts/bootstrap_rollout.sh` → exit0. [Lint](raw/task7-fix1-lint.txt).

**M1 RED/fix:** four metadata-only drift fixtures previously reached a blocked
preflight/install sentinel instead of rejecting Jinja2/mpmath/PyYAML/SymPy drift:
[4 failed](raw/task7-fix1-red-pins.txt). Exact pins3.1.6/1.3.0/6.0.3/1.14.0 were added
to the pre-mutation prerequisite gate; the same four fixtures then passed:
`.venv/bin/python -m pytest -q rl/rollout/tests/test_bootstrap_pins.py`
→ **4 passed in0.02 s**, [output](raw/task7-fix1-pins.txt), also included in the50-test run.
Actual metadata-only checking of all21 protected prerequisite pins matched exactly:
[output](raw/task7-fix1-prerequisites.txt). No alternative dependency was installed and
no ML-stack reinstall occurred.

**Changed-wire actual GPU integration:** `bash scripts/test_rollout_gpu.sh`, using the
same fixed `scripts/with_gpu.sh` guard, cached revision,0.80 utilization, sampling values
and project-local cache paths. [Preflight](raw/task7-fix1-gpu-preflight.json) reserved
100 MB new runtime cache, no download. [Actual output](raw/task7-fix1-gpu.txt):
**1 passed in12.73 s**, command/guard exit0; six samples reproduce stochastically,
version mismatch is rejected, token/logprob correspondence passes and independent
raw-model comparison remains0.0552303791 below0.15 tolerance. Actual worker1762533
reported distributed groups destroyed, joined exitcode0; successful guard completion
also proves all owned descendants drained. This is one functional wire/lifecycle
integration, not a speed comparison.

**M2:** the same three known Torch deprecation / optional FlashInfer fallback / eager
cudagraph notices remain visible in the new raw output. No logs are described as pristine,
no warning was filtered and no cosmetic dependency/host change was made.

[Post-fix resource preflight](raw/task7-fix1-resource-final.json): project10,376,307,878 bytes,
free359,392,059,392 bytes, checkpoints0/services0, within14 GB. Historical unmeasured
first-two-run default-driver-cache caveat remains unchanged. The earlier543-test CPU
result belongs to the original implementation snapshot; the focused CPU and actual
GPU outputs above cover this fix. No full CPU suite was repeated.
