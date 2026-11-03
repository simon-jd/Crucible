# Task 12 deployment and reproduction boundary

Both YAML files are offline templates, never applied. Single-card template targets
user-authorized Qwen3-8B; the original 30B colocated target was superseded by the
local GPU budget rule. 235B FP8/8xH100 TP+EP is paused pending budget approval;
未在真实多机上验证. `replicas: 0` and `REPLACE_*` make readiness explicit. No image
or model digest is invented. Existing Task 14 deployments/adapters are unchanged.

Before actual local execution (later phase, not performed here):

1. End/freeze competing project work and acquire `flock /var/tmp/rtx4090.lock`.
   Record an experiment checklist and <=4-hour stop condition first. Never create
   another GPU queue job during the current Task 10 run. Budget remains $0.
2. Inventory the shared immutable HF cache and available disk read-only. Pin exact
   Qwen/Qwen3-8B revision and tokenizer/config hashes; run the resource guard before
   any approved download (>5 GB needs free >=50 GB + size). No duplicate model copy.
   Establish actual BF16 weight/KV/headroom feasibility on 24 GB with current shared
   occupancy. The 4096 context/0.80 memory values are starting limits, not measured
   fit. Any quantization/model substitution requires a documented plan adjustment.
3. Build an isolated runtime after Task 10 releases its frozen environment. Verify
   the v0.5.5 source contract against an actual image digest and exact dependencies;
   record driver/CUDA/NCCL/OFED/runtime versions, CPU/GPU clocks and process command.
   Never install into the live shared ML environment. Root verified this release
   requires Torch 2.8.0, Transformers 4.57.1, FlashInfer python/cubin 0.5.0,
   sgl-kernel 0.3.16.post5 and torchao 0.9.0; existing rollout dependencies differ.
   Design an isolated environment/overlay with explicit ABI validation and peak
   installation/cache accounting before any installation. Resolve torch/CUDA/ABI support
   and package/cache size under a fresh resource plan before installation.
4. Launch one transient loopback SGLang server under the shared lock, directly with
   native token-ID API; disable radix cache for the length-only baseline. No K8s is
   needed locally. Own its PID/process group and use bounded startup/request timeouts
   and a cleanup trap. Stop the server on success/error/signal; verify exit and freed
   GPU memory before releasing lock. SIGKILL/host crash require explicit cleanup audit.
5. Generate the allowlist using the documented stdlib helper, then verify native
   `prompt_tokens`, delta output IDs and requested output count on real zero/one/multi
   token smoke requests. Inspect zero-input and zero-output failures without filtering.
   This runtime/tokenizer smoke is a mandatory gate still unexecuted.
6. Replay both official traces (initial bounded prefixes are parser fixtures; choose
   explicit disjoint bounded calibration/evaluation windows with new provenance for
   representative measurements). Run 1x/2x/4x with >=3 matched alternating rounds,
   reject invalid senders, derive real SLO only from calibration baseline, then report
   held-out goodput/latencies and sample/failure/coalescing counts. Capacity exhaustion
   is a failed experiment, never a reason to smooth the trace. FP8 comparisons later
   require held-out quality deltas beside performance.
7. Save manifests, commands, raw records, source hashes, distributions and cleanup
   evidence in docs/evidence; report median/range. Whole Task 12 remains unfinished
   until the authorized real single-GPU baseline is verified.

The K8s local PV binds the existing shared cache read-only; it does not populate
models or establish node identity/storage provisioning. Native server flags were
checked at source commit `0c006b8809cd99e1f95926401a2823dd952641c8` only. The 8-GPU
expert backend and FP8 numerical/hardware behavior are not verified by YAML parsing.
