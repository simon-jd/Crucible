# Task13 offline P/D layouts and evidence CLI

CPU implementation only. **未在真实多机上验证**. No native FP8/BF16 comparison, real SLO goodput,
held-out quality, or per-request RDMA KV-transfer latency has been measured by this task.
The native Task12 BF16 baseline remains unaccepted. No YAML here authorizes execution.

## Inert layouts

Every Deployment, including the router, has `replicas: 0`. Unknown image digests, model
revision/local path, node names and RDMA identities remain `REPLACE_*`. Do not apply these
files: resolving placeholders is insufficient without a separately reviewed physical proposal.

| File | Prefill engines | Decode engines | Total engine GPUs | Placement |
| --- | --- | --- | --- | --- |
| pd-1p1d.yaml | one, TP=2 | one, TP=1 | 3 | same unresolved node |
| pd-2p1d.yaml | two, TP=1 each | one, TP=1 | 3 | same unresolved node |
| pd-cross-node.yaml | one, TP=2 | one, TP=1 | 3 | distinct unresolved P/D nodes |

All three require a matched colocated control with **three independent one-GPU replicas**,
identical model lineage, prompt order, sampling, arrival load, SLO, warmup/cache-reset policy
and observation window. P/D ratios count engine instances, not GPUs. Different TP is an
explicit topology difference; the source allows TP=2/1 because the larger divides by the
smaller, but actual model-head/KV-head compatibility and memory must be verified. These
three-GPU physical templates cannot run on the authorized single RTX4090.

Each P/D worker has a unique headless Service. Prefill HTTP port 30000 and bootstrap port
8998 belong to that same worker; multiple prefill instances never share a bootstrap Service.
Decode exposes HTTP 30000. Router HTTP port 30080 uses a repeated `--prefill URL 8998` for
each prefill worker and `--decode URL` for each decoder. Router policy is round_robin.
Request handoff preserves `bootstrap_host`, `bootstrap_port`, and `bootstrap_room`; the
decoder connects to the selected prefill bootstrap address. Mooncake uses P2PHANDSHAKE/RDMA.
The pod network must additionally carry dynamically chosen native engine ports. None of
network reachability, GPU-direct RDMA, HCA mapping, device plugins or real handoff is proven.

The source identity is SGLang 0.5.5 commit `0c006b8809cd99e1f95926401a2823dd952641c8`.
Source hashes and official source URLs are in
`docs/evidence/raw/task13-source-contract.json`; the hashed files are vendored next to it as
`task13-sglang-*.txt` (re-fetched from the GitHub contents API at that commit on 2026-10-27,
same sha256 as the deleted installed overlay). The separately inspected router sources
are pinned to that same commit, with exact URL/byte/hash receipts in
`docs/evidence/raw/task13-router-source-research.json`. They establish parser/entrypoint
contracts, not an installed router or working Rust ABI. `sglang.srt.disaggregation.mini_lb`
is a relocation error in this revision; the templates use `sglang_router.launch_router`.
The Rust router must be built/pinned/validated later; the debugging Python mini-LB is not
silently substituted. Tests parse source AST/text without importing any native package.

Required later prerequisites: verified image digests and driver/CUDA/NCCL/OFED identities;
model revision/files/head topology; router runtime/digest; local model mount; node/HCA
identity and device/network mapping; real per-request transfer instrumentation and failure
checks; same-card colocated controls. No TensorRT-LLM or alternative quantizer was added.

From the repository root, using the existing Python environment:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m serving.deploy.task13_validate serving/deploy/pd-1p1d.yaml
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m serving.deploy.task13_validate serving/deploy/pd-2p1d.yaml
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m serving.deploy.task13_validate serving/deploy/pd-cross-node.yaml
```

The validator checks the actual role/TP/card budget, zero replicas, commands, endpoints,
bootstrap ports, Services/selectors, node split, immutable placeholders and pending gates.
`--execution-ready` always rejects these offline templates. This CLI is not a Kubernetes
schema validator, operator, deployer, or permission to replace/increase replicas.

## Raw accuracy/performance evidence schema 2

Run from repository root; no inference, network, native imports or model loading occurs:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m serving.eval.accuracy path/to/measured-bundle.json > path/to/report.json
```

The complete illustrative input is `docs/evidence/raw/task13-round1-fixture-v2-input.json`, explicitly
`kind: fixture`. Its output `task13-round1-fixture-v2-report.json` contains **synthetic test data only**:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m serving.eval.accuracy docs/evidence/raw/task13-round1-fixture-v2-input.json --allow-fixture
```

The CLI reads one self-contained UTF-8 JSON raw-evidence bundle and writes JSON to stdout.
Schema 1 (including the preserved original task13-fixture-input/report.json artifacts) is
rejected: its missing plan and terminal observation evidence cannot be inferred safely.
Produce schema 2 only from the predeclared plan and actual observation records; do not
invent a plan from surviving runs or mark an incomplete run complete during migration.
Duplicate JSON keys, missing required fields, nonfinite values, duplicate/missing source IDs,
wrong chronology, impossible timing/counts, config/provenance mismatches and unmeasured
required successful-request timings are errors (exit 1). Keep the original invalid input;
the CLI never rewrites or removes it. It does not silently drop incomplete pairs or examples.
Unknown extra request/quality metadata is retained, but cannot replace computed correctness.

Top-level fields:

- `schema`: integer 2; `kind`: `measured` or `fixture` (fixtures require CLI opt-in).
- `conditions`: the common frozen model, source/runtime/protocol identities, GPU count,
  sampling, fixed BF16 SLO, scheduled performance workload, held-out dataset definition,
  required planned_rounds and explicit run_plan.
- `configurations`: exactly `baseline` and `candidate`, each with exactly `precision`,
  `backend`, `settings`, `model_sha256`, `runtime_sha256`. Precisions are bf16/fp8.
  Backend must name the actual implementation, not an intended fallback. All non-precision
  options go in identical `settings` dictionaries, including `layout` (`colocated` or `pd`).
  Precision-specific backend is explicit. Actual settings must include server arguments,
  precision/kernel attestation identities and warmup/cache policy when native data exists.
- `runs`: chronological raw performance runs, **at least three complete pairs**, in the
  predeclared sequence AB, BA, AB, BA…; A=baseline, B=candidate. Every planned member
  must appear exactly once, in order. No reorder/filter/retry-to-pass or omitted trailing pair.
- `quality`: exactly baseline/candidate held-out records, bound to the same configuration
  and conditions hashes as their performance runs. A fast config cannot borrow another's quality.

`conditions.model` has nonempty `name`, immutable `revision`, and SHA256 of the actual model
file-manifest identity. `source_sha256`, `runtime_sha256`, `protocol_sha256` identify retained
source, runtime and experiment protocol manifests. `gpu_count` is a positive integer.
`sampling` includes seed, nonnegative temperature and positive max_new_tokens; retain all
other generation options in this common object. These are identity assertions to be checked
against native artifacts during review, not cryptographic proof that measurements occurred.

`conditions.slo` includes `calibration_precision: bf16`, calibration evidence SHA256,
`ttft_ms`, and `tpot_ms`. Both variants use this exact same calibration identity and thresholds.
`conditions.workload` is an ordered nonempty list of unique `{id, prompt_sha256, scheduled_s,
input_tokens, requested_tokens}` records. The first scheduled timestamp is zero and later
arrivals are nondecreasing. Hash the actual ordered prompts, not synthetic replacements in a
real run. `conditions.heldout` has dataset, subset, immutable revision, normalization, and
ordered nonempty `examples` containing unique id, prompt_sha256 and correct label.

`conditions.planned_rounds` is a required integer >=3. `conditions.run_plan` is a required
ordered list of exactly twice that many objects, each containing exactly id, round, variant.
IDs are nonempty and unique. Rounds start at1 and form AB, BA, AB… pairs. Freeze this plan
before execution along with the other conditions/protocol; retain it even if later runs
abort or fail. Actual `runs` must match its exact IDs, rounds, variants and order. Removing
one final member or an entire final pair is rejected, even when three earlier pairs remain.
An unattempted planned run cannot be silently deleted or synthesized as measured evidence:
retain the incomplete input and the plan; the CLI rejects it. Attempted aborted runs retain
all request rows and suppress aggregates. The report's rounds comes from planned_rounds,
not the surviving list length, and includes a separate run_plan_sha256 identity.

Canonical identity hashes are SHA256 of UTF-8 `json.dumps(value, sort_keys=True,
separators=(',', ':'), allow_nan=False)` with Python's default ensure_ascii=True. `digest`
is available in the module. Every run/quality record carries `conditions_sha256` and
`config_sha256` equal to those exact frozen objects. The report includes these hashes and
separate run-plan/SLO/workload/heldout hashes. The CLI additionally emits the original bundle's path,
byte count and byte SHA256; archive the original raw bundle and referenced native manifests.
No external artifact is fetched or silently trusted as proof of hardware execution.

Each run has a unique id, 1-based round, variant, both identity hashes, explicit boolean
aborted, and integer `started_at_ns`/`finished_at_ns` from one common monotonic clock. Runs
cannot overlap or reverse. The start anchors the first scheduled request (relative zero);
finish brackets the required `observation_end_s` measured relative to that same start.
The bound must be finite/nonnegative and cannot exceed the integer nanosecond clock bracket
(decimal Fraction comparison, without a tolerance that admits out-of-bracket observations).
`requests` must match all workload IDs in exact order.
Each request contains id, status, send_s, end_s, ttft_ms, tpot_ms, output_tokens,
kv_transfer_ms, and error. Times are relative to run start. Status is success/error/truncated/
missing. Error/missing require a reason; success/truncated require error=null. Missing means
no send/end and zero output. Attempted requests, including failures, require terminal time.
All missing rows have an explicit terminal observation at their run's observation_end_s;
they retain end_s=null because no request completed. That bound is an observed classification
of unsent requests, not an invented completion. It must cover every observed completion and,
for non-aborted runs, the latest declared scheduled request. Without missing rows it must
equal the final request completion exactly, excluding cleanup. Aborted runs may finish before
the remaining schedule, but must retain all missing future rows and always suppress aggregate
performance. Impossible observation/clock bounds are rejected even for aborted runs.
Every measurement must be finite/nonnegative and consistent with actual token count and
request duration. Tokens cannot exceed requested_tokens. TTFT is null for zero tokens;
TPOT is null for fewer than two tokens. Successful multi-token responses require both.
For P/D successful requests KV transfer must be measured; colocated KV transfer is null
(not applicable). Failures may carry explicit null timing; undefined counts remain visible.

Sender validity: more than 1% missing send timestamps or send lag strictly above 5 ms
invalidates the whole performance comparison; no invalid pair is excluded. An aborted run
or zero measurement interval also invalidates it. A 1e-12-second floating boundary tolerance
is used only for the 5 ms comparison. Nonfinite timestamps are rejected instead of coerced.
Throughput is **successful output tokens / seconds from first schedule through final terminal
observation**: observation_end_s is the final completion when no rows are missing; with
missing rows it also includes their terminal observation bound. For example,99 successful
requests completing by0.101s and one unsent request observed at0.2s use0.2s, not0.101s.
A non-aborted observation ending at0.101s cannot cover an unsent request scheduled at100s.
Failed/truncated observed tokens are counted separately. SLO goodput is successful
requests satisfying both defined TTFT and TPOT thresholds per same interval. A one-token
response has undefined TPOT and contributes no goodput. Each run preserves all source,
success/error/truncated/missing, sender-invalid, token, SLO and undefined-timing counts.
Valid complete pairs get median/min/max per variant. Invalid performance returns null
aggregate throughput/goodput with every raw run and count retained and quality still reported.
Only the common helper's numeric `_median` is reused; its fixed-AB chronological validation
API is never invoked or fed reordered BA measurements.

Each quality record's `items` must match all declared held-out IDs/order. Each item repeats
its exact prompt_sha256 and correct label and includes status, answer (text/null), error.
Accuracy denominator is **all source examples**, including errors, missing responses,
truncations and unparseable predictions. Only successful correct answers score. Paired raw
outcomes and counts are retained. Two exact normalization policies are supported:

- `numeric-final-v1`: strip whitespace; if present use text after the last `####`; parse a
  signed integer or decimal with valid optional thousands grouping. Compare exact rational
  values. No units, arbitrary prose extraction, exponent, rounding or numeric tolerance.
- `choice-letter-v1`: strip whitespace and uppercase one letter A–J. Reasoning/prose or
  multiple letters does not parse. Retain the original model response in answer; no hidden
  answer extraction or manually repaired predictions.

`accuracy_delta` is candidate minus baseline in fractions; `absolute_accuracy_difference`
is its absolute magnitude, reported alongside throughput, not as relative percent change.
Run separate bundles for distinct dataset/subset/normalization policies. Held-out means held
out from this experiment's calibration/tuning; no pretraining contamination exclusion is claimed.

`comparison_eligible` indicates measured-labeled, structurally valid complete performance
pairs, not accepted hardware evidence. `benchmark_accepted` is deliberately always false:
a parser cannot attest origin or accept the future native benchmark. Fixture output remains
explicitly fixture and ineligible. Native acceptance requires the independent artifact/runtime
review and resource gates documented in `docs/evidence/pd-disaggregation.md`.

## CPU checks

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.cache/task13-cpu" .venv/bin/python -B -m pytest -p no:cacheprovider serving/eval/tests/test_accuracy.py serving/eval/tests/test_task13_deploy.py
.venv/bin/ruff check --no-cache serving/eval/accuracy.py serving/deploy/task13_validate.py serving/eval/tests/test_accuracy.py serving/eval/tests/test_task13_deploy.py
```

The installed-source test skips if the pinned local source is absent; a skip is not proof.
This delivery's complete CPU run must contain no skips. Tests use only the owned
`.cache/task13-cpu` fixture directory and disable pytest/bytecode caches. They do not use
Kubernetes, download models, import SGLang/Torch, or exercise GPUs.
