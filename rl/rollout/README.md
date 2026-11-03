# Local rollout boundary

`RolloutEngine.generate(prompts: list[Prompt], group_size: int, policy_version: int) -> list[Sample]`
returns samples in input-prompt order, then completion-index order. `Prompt(prompt_id, text)`
contains a nonempty string ID and the exact model input text; callers format chat templates.
`Sample(prompt_id, text, token_ids, logprobs, policy_version)` is immutable, with tuples for
IDs and logprobs. No fake backend is selected automatically.

Construct one `PolicyCheckpoint(absolute_local_path, immutable_revision, policy_version)`
and use the engine as a context manager. The caller assigns a version to a checkpoint,
then the worker must actually load it and acknowledge that identity before generation.
`generate` checks the requested version against this loaded, immutable identity before
any model call. It cannot relabel old weights by supplying a new argument. Paths must remain immutable while an engine is running. For owned, content-verified
versions, Task 9 provides `weight_store=store` and `load_weights(version)` with an
exclusive close → load → rollback transaction. See [weight synchronization](../sync/README.md).
Legacy checkpoints without a store retain the caller's immutability responsibility.
Never stage two engines with 0.80 utilization. An owned store reserves the engine
identity before initial startup; a failed constructor exposes its retained owner as
`error.recovery_engine` on the original exception. Call that owner's `close()` to
retry known partial-worker cleanup. Unknown ownership keeps the store reserved;
confirmed cleanup releases only that owner, so stale repeated close cannot unpin
a later engine. This is process-local protection for a single shared store object,
not coordination between independently opened stores. Actual vLLM accepts only
BF16 floating state in owned snapshots, rejecting FP32/FP16 before worker spawn.
Generic publication and injected CPU test backends retain their exact dtype contracts.

The engine owns one spawned worker, using vLLM **0.10.2 V1**, an in-process core and
single-process (`uni`) executor. Actual GPU calls must run under `scripts/with_gpu.sh`.
Use `close()` or a context manager for every batch. Close shuts down the core,
explicitly destroys distributed groups and joins the worker; timeouts escalate only
against that owned multiprocessing process and surface nonzero exits. The outer fixed
GPU guard reaps all descendants before releasing its shared flock. Generation errors
close the engine so that a late RPC response cannot be consumed as a later request.
Initialization has a 300 s absolute deadline; each generation has one 120 s absolute
deadline covering request encoding/send and complete response receive/decoding. Application
RPC uses nonblocking Unix sockets, an 8-byte length header and primitive finite JSON,
with a 64 MiB frame limit, 65,536-character string limit and nesting limit 12. Partial
headers/bodies and write backpressure consume the same deadline. Codec validation/encoding
checks that deadline incrementally; bounded JSON decoding checks it before and after.
Close spends at most 0.1 s sending graceful control, then closes the transport before
30 s normal exit, 10 s TERM and 10 s KILL joins. Transport errors cannot skip owned-process
cleanup. Failed/unconfirmed workers retain their owner handle and reject further work;
nonzero exits or unconfirmed cleanup remain errors. No resident inference service.
Scripts using multiprocessing should place entry-point code under `if __name__ == '__main__':`.

## Sampling and learner logps

Defaults are stochastic `seed=0, temperature=1, top_p=1, top_k=-1, max_tokens=32,
max_model_len=256`. Each call resets per-prompt seed to `seed + prompt_index`; n-way
sampling uses vLLM's child-request seed handling. Repeated calls with identical ordered
prompts, engine/configuration and group size reproduce; changing order/batch shape is
outside that reproducibility guarantee. Greedy sampling and seed=-1 are rejected.

`logprobs=0` requests each chosen token's logprob, extracted by **chosen token ID**, never
by dictionary iteration order or cumulative sequence score. Explicit `raw_logprobs` mode
returns FP32 log-softmax of the BF16 model logits **before temperature/top-p/top-k or
penalties**. These are raw model logps, and are not the behavior logps of a non-neutral
sampler. Task 8 must score policy/reference tokens using this same raw-model distribution;
neutral default sampling aligns the behavior and model distributions. Non-neutral
sampling needs an explicit algorithmic decision about behavior importance correction.

`generation_config='vllm'` disables checkpoint generation-config overrides. Stops are
model-config EOS and max_tokens only, without custom string stops or extra stop IDs.
Emitted EOS IDs remain in token_ids and have logprobs. Text skips special tokens and
omits stop-token text, so re-tokenizing text is not a safe substitute for token_ids.
Prompt tokens are excluded from both generated-token arrays. vLLM may generate fewer
than max_tokens on EOS; count correspondence remains exact.

## Reproduce this local acceptance

Prerequisites: existing `.venv`, ABI-pinned `.venv-ml` from `bash scripts/bootstrap_ml.sh`,
and shared cached `Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775` under `/home/postedism/.cache/huggingface/hub`.
The test is offline and never downloads a duplicate model.

```bash
bash scripts/bootstrap_rollout.sh
.venv/bin/python -m pytest -q rl/rollout/tests/test_rollout.py
bash scripts/test_rollout_gpu.sh
```

The bootstrap checks existing Torch/Triton/NumPy/NVIDIA and exact reused
Jinja2/mpmath/PyYAML/SymPy versions before mutation, runs resource preflight,
and installs only missing/mismatched added wheels sequentially using explicit public
PyPI wheel URLs and SHA256. The complete hash lock includes the existing ML ABI;
`requirements/rollout.in` and `rollout-abi.txt` document inputs. Resolution used uv with
`--exclude-newer 2025-10-01`; the lock header records the regeneration command. Do not
run `uv sync` afterwards expecting it to retain this separate rollout environment extension.
The wheel inventory contains compressed and ZIP-central-directory unpacked sizes.
All transient/install/compiler caches remain project-local and count toward 14 GB.
The backend itself sets CUDA/XDG/vLLM/Triton/Torch/CuPy/Numba/TMP paths before heavy
imports, so direct API callers also receive cache containment.

Default CPU tests skip the real backend. The GPU script uses the fixed shared lock,
0.80 memory utilization, eager execution, project-local compiler caches and corrected
HF cache environment. It verifies stochastic reproducibility, diversity, group/version
contracts, full token/logprob correspondence, worker exit, and eight independent FP32
Transformers chosen-token raw logps (absolute tolerance 0.15 against BF16 vLLM).
This is functional evidence, not a throughput benchmark or held-out quality evaluation.
See [recorded evidence](../../docs/evidence/rollout.md).
# Explicit RAM checkpoint branch (actual vLLM UNVERIFIED)

The new PhaseBridge passes RamPolicyCheckpoint plus a separate owned capability
and explicit project runtime root to VLLMWorker. POSIX spawn reduction calls DupFd
only inside `__reduce__` while get_spawning_popen is active. Pinned Python3.12
duplicate_for_child records the existing FD in pass_fds, without a parent dup or
resource_sharer broker. Ambiguous Process.start failure retains the original
Process/capability even when pid is None; it cannot be declared cleaned.

Pinned vLLM0.10.2's custom BaseModelLoader loads the sealed iterator before profiling.
Base directory/revision is only the config/tokenizer source, separately bound into
the sealed artifact. The branch requires BF16 unquantized world1 V1/in-process/uni,
eager/no prefix caching/no adapters or offload. Existing disk loading remains intact.
Before ready/version ACK, the actual model is inspected through LLM.apply_model:
all qkv weights/biases, gate/up shards, direct parameters, persistent buffers, tied
head and embedding padding are checked against exact BF16 source coordinates.
Returned loader names or an echoed digest alone are insufficient.

The shared coordinate adapter can be tested using CPU packed tensors without
importing vLLM. Such tests do not validate the actual runtime or GPU memory fit.
No new GPU execution has been authorized by preparing this source.


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
