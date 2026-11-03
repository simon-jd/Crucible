# Fused chunked token log-probabilities — Task 10

Validated locally on 2026-10-24. The custom op reduces isolated forward/backward peak allocation by **66.67%** at V=152,000. The actual Qwen learner's median peak allocation is **unchanged**. Its median step time is 2.59% lower in this short measurement, with overlapping ranges; this does not establish a repeatable whole-learner speedup.

## Implementation and contract

`torch.ops.crucible.token_logprobs(logits[B,T,V], targets[B,T]) -> [B,T]` is registered through `torch.library.custom_op`, `register_fake`, and `register_autograd`. Import `kernels.ops` to register it. Production `rl/learner/train.py::_score` now invokes it for both policy and frozen reference. `_score_reference` preserves the previous selected-logits FP32 `log_softmax + gather` for measurement.

Supported logits are FP32/BF16 on CPU and CUDA, plus FP64 on CPU for mathematical gradcheck. Output is FP32 for FP32/BF16 and FP64 for FP64; gradient has input dtype. Targets must be int64, same device, exact `[B,T]` shape, and in `[0,V)`. V must be positive; empty B/T and arbitrary PyTorch strided/expanded layouts are supported. Unsupported dtypes/shapes raise. CUDA target validation synchronizes a small boolean reduction to host before dispatch and raises ValueError without a device assertion. This synchronization cost is included in measurements. First-order autograd is supported; higher derivatives are outside this implementation's contract.

CUDA processes vocabulary tiles of 1,024 elements. Each tile produces a maximum and a shifted exponential sum, then row reduction combines these without forming `max + log(sum)` and subtracting large nearly equal values. Output is `(target - maximum) - log(shifted_sum)`. Backward recomputes the same statistics from saved logits and writes `(one_hot(target) - probability) * upstream_gradient` directly to the required dense logits gradient. It never materializes a full softmax, nor reconstructs normalization from target minus logprob. Scratch is `8*B*T*ceil(V/1024) + 8*B*T` bytes, excluding output, input, and required gradient; CPU similarly uses bounded vocabulary slices. Tests cover large common finite offsets, extreme finite differences, noncontiguous and expanded layouts, and nonfinite propagation including all-negative-infinity tiles.

Learner causal completion positions, EOS handling, padding mask, token-weighted microbatch accumulation, clipping/KL, detached old/reference scores, staleness checks, FP32 master parameters/AdamW, BF16 compute/reference, finite-state checks, cleanup, and permanent failed-phase guards remain unchanged.

## Reproduction and environment

Run from the repository using the existing pinned environment:

```bash
OMP_NUM_THREADS=2 .venv-ml/bin/python -c 'from rl.learner.runtime import configure_runtime; configure_runtime(); import pytest; raise SystemExit(pytest.main(["-q", "kernels/tests/test_ops.py", "rl/learner/tests/test_train.py", "rl/learner/tests/test_grpo.py", "rl/learner/tests/test_runtime.py"]))'
bash scripts/test_logprob_gpu.sh
```

The GPU script performs canonical project/cache/disk preflight with a 1 GB additional-growth reservation, then uses existing `scripts/with_gpu.sh`, fixed `/var/tmp/rtx4090.lock`, and four-hour maximum ownership. It checks at least 19 GB driver-visible free memory inside the lock before compilation or model load. Runtime configuration happens before heavy imports, placing CUDA/XDG/Triton/Inductor/Torch/temp caches under `.cache/task8/runtime`. The immutable model is reused offline from shared `~/.cache/huggingface/hub`; conflicting inherited HF variables are removed. No download, environment upgrade, remote GPU, cloud/API call, checkpoint, or resident service is involved.

Final hardware/software: local NVIDIA GeForce RTX4090; driver 595.99.02; CUDA 12.8; NCCL 2.27.3; Torch 2.8.0; Triton 3.4.0; Transformers 4.55.4; vLLM 0.10.2; NumPy 2.2.6. The existing pinned environment is unchanged. Qwen2.5-0.5B-Instruct is loaded by explicit local snapshot revision `7ae557604adf67be50417f59c2c2f167def9a775`, previously upstream-hash verified; this task did not redownload it. No container was introduced and no image/OFED/multinode claim is made.

[Source hashes](raw/task10-final-source.sha256) were recorded before the final job and verified unchanged after its exit. Final guard-wrapped session63723 exited0. A preliminary queued iteration had source edits after launch; its 13 passing CUDA tests are retained as iteration evidence, not used as the frozen final gate.

## Correctness and registration

- **CPU:** 51 passed, 1 intentionally skipped GPU-only memory test. Includes FP64 mathematical gradcheck, FP32/BF16 forward/backward comparisons, actual Inductor `torch.compile(fullgraph=True)` forward/backward, schema/FakeTensor/autograd/compile opcheck, learner op dispatch and original-score update parity, and existing learner/GRPO/runtime tests. One sandbox “Can't initialize NVML” warning is preserved; no CUDA correctness is inferred from this CPU run. [CPU log](raw/task10-cpu-final.txt)
- **Actual CUDA:** 16 passed, 18 CPU-only cases deselected. FP32/BF16 gradients are compared against the reference independently of CPU gradcheck; actual Inductor fullgraph forward/backward and opcheck pass. Invalid targets raise and a subsequent valid invocation succeeds in the same CUDA context. [Final GPU log](raw/task10-gpu-final.txt)
- **TDD:** required named tests existed before the op and failed with missing `kernels.ops`. The initial CPU implementation's boolean-subtraction failure was fixed and preserved. [RED](raw/task10-red.txt), [initial CPU failure](raw/task10-cpu-focused.txt), [CPU iteration](raw/task10-cpu-iteration2.txt), [preliminary CUDA iteration](raw/task10-gpu-focused.txt)
- Ruff check/format and shell syntax pass. [Lint](raw/task10-lint.txt)

Opcheck establishes registration behavior; FP64 gradcheck checks mathematical derivatives; CUDA reference comparisons establish actual kernel derivatives. These are distinct checks. Existing unrelated Task9 synchronization, Task14 Go, and Task12 suites were not repeated.

## Isolated vocabulary-memory result

FP32 logits `[2,32,152000]`, seed15, forward plus backward, both paths warmed. Inputs remain live; each sample clears input gradient, synchronizes, resets allocator peaks, records current allocation, then executes and synchronizes forward/backward. Incremental peak is maximum allocated bytes minus starting allocated bytes, including the required dense logits gradient.

| Path | Incremental peak bytes |
|---|---:|
| Previous FP32 log_softmax + gather | 119,538,688 |
| Fused chunked op | 39,847,936 |

This is an isolated memory acceptance check, not an isolated timing claim or a whole-learner memory claim.

## Actual learner matched measurements

The fixture is a short deterministic performance workload, **not heldout quality evaluation or long RL training**. One prompt produces two distinct eight-token completions: the first uses the highest-ranked first token, the second the second-ranked first token, then both continue greedily. Every old logprob is computed from the actual initial BF16 policy logits with FP32 raw normalization. Fixed rewards `(1,0)` produce nonzero opposite advantages. Exact IDs/scores and fixture hash `f05480b1c9bd72559c684c981bfe502bdec7cf8ba42d1705b773c388ce2477e1` are in the raw log; all six variants reuse these same records.

Three matched pairs run **baseline/candidate, candidate/baseline, baseline/candidate**. Each variant reloads the same immutable model, seed1010, and fresh AdamW (LR1e-5), BF16 frozen reference, world-size1 FSDP2, microbatch2, max_length128. One unmeasured complete learner warmup initializes optimizer state and compilation; one subsequent complete `Learner.step` is measured using `perf_counter` and CUDA synchronization immediately before/after. The measured step includes all common state validation, scoring, GRPO backward, optimizer work, cleanup, and a common q_proj-gradient observation hook. Model loads, fixture construction, warmup, post-step parameter comparison, and teardown are excluded. An equally sized q_proj clone is live for both paths and included in their starting allocation.

Record policy version0 remains valid after warmup at version1 with max_staleness2. Every measured step asserts accepted2, dropped0, effective_tokens16, optimizer_steps1, ready phase, and version1→2. All six measured steps have q_proj gradient max magnitude **0.052490234375** and parameter max update **0.000010132789611816406**; none is a zero-gradient or stale-record surrogate. Candidate/reference measured losses differ by approximately 6e-8. The unchanged finite-state and cleanup work is included equally.

| Round | Execution order | Baseline step ms | Candidate step ms |
|---|---|---:|---:|
| 1 | B, C | 228.129 | 228.380 |
| 2 | C, B | 267.974 | 228.713 |
| 3 | B, C | 234.800 | 248.127 |
| Median | | **234.800** | **228.713** |
| Range | | 228.129–267.974 | 228.380–248.127 |

| Learner allocation metric | Baseline median [min,max] bytes | Candidate median [min,max] bytes |
|---|---|---|
| Total measured-step Torch peak | 10,020,235,776 [10,020,235,776; 10,020,661,760] | 10,020,235,776 [10,020,235,776; 10,020,235,776] |
| Increment above step start | 3,067,372,544 [3,067,372,544; 3,067,372,544] | 3,067,372,544 [3,067,372,544; 3,067,372,544] |

Whole-step median peak allocation is unchanged despite the isolated kernel reduction. Peak reserved memory is 11,177,820,160–11,720,982,528B for baseline and 11,720,982,528B for candidate. The first baseline's starting allocation differs by 425,984B from later rounds; the medians and incremental peak show no learner memory improvement. A 2.59% lower median time with overlapping ranges across only three pairs is a small observed result, not strong evidence of repeatable acceleration.

Allocated/reserved peaks describe the Torch allocator during the measured step; they exclude NCCL/driver/external allocations and are not a physical hard cap. The 18 GB observed allocation threshold is checked after each step; the 19 GB free preflight adds margin but does not guarantee an instantaneous limit. Host peak RSS was 5,426,044,928B. Between variants, owned models/optimizer are deleted, garbage collection and empty_cache/synchronize run; 17,039,360B of framework allocation remained while the NCCL process was alive. The NCCL group was explicitly destroyed and the process exited under the fixed guard, which returned0.

## Resource closure and limits

Final pre-execution project use was10,710,567,038B with292,841,889,792B free; driver-visible GPU free was22,073,049,088B for kernel tests and22,072,918,016B for learner. Post-run canonical preflight reports project10,711,701,109B and294,019,153,920B free, still passing the1GB additional-growth reservation within the14GB project cap. Small final documentation/Git growth fits that reservation. [Resource record](raw/task10-resources-final.json)

Spend $0; downloads0; retained checkpoints0; resident services0. No other GPU processes were touched. This evidence covers one local4090, one small performance fixture, first-order custom-op differentiation, and the actual guarded learner path. No multi-GPU, heldout-quality, long-run stability, CUDA FP64, FP16, or higher-order-autograd result is claimed.
