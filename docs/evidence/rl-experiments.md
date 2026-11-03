# RL evaluation and configuration phase

## Scope and authenticity

This is an independent CPU implementation phase. No training, model import,
inference, trusted math grading, untrusted code execution, dataset acquisition,
native runtime, GPU, baseline execution or installation occurred. Every supplied
question and correctness value in the retained CLI examples is explicitly a
synthetic fixture. Schema consistency is not measurement authenticity or benchmark
eligibility. Original end-to-end quality, timing, idle, synchronization,
reward-hack, stale/synchronous, ablation and equal-budget external experiments
remain required. Code experiments remain blocked on public tested code data and
an actual isolated sandbox; there is no host subprocess verifier substitute.

## CPU checks and CLI results

The initial named RED run exited 4 with 3 missing-module collection errors before
implementation; see [raw RED](raw/task11-red-01.txt). Named tests were
`test_pass_at_1_unbiased_estimator`,
`test_heldout_split_disjoint_from_train`, and
`test_actual_config_family_complete_and_equal_budget`.

The first full new suite passed 39 tests, then additional content-binding and
strict-budget regressions brought the final suite to 42 passing tests in 0.16s;
see [first GREEN](raw/task11-green-01.txt) and
[final GREEN](raw/task11-green-03.txt). These tests exercise the actual YAML files,
planned unequal denominators, strict types/statuses, overlap anywhere in the full
candidate pool, input-order independence, and changed gold/reference binding.
Ruff check passed; Ruff format check reports 7 files already formatted. See
[lint](raw/task11-ruff-final.txt) and [format](raw/task11-format-check-final.txt).
No unrelated historical/native/model suites were run.

Real CPU CLI calls used the retained `task11-fixture-*` JSON inputs. Selection,
verification and reversed-input selection exited 0; `cmp` between original and
reversed-input manifest outputs exited 0. See
[manifest](raw/task11-fixture-manifest.json),
[verified manifest](raw/task11-cli-heldout-verify.json), and
[reversed-input result](raw/task11-cli-heldout-reversed.json).

The complete synthetic report retains planned counts n=1 and n=3, correct counts
c=1 and c=0, total planned 4, attempted 4, per-problem pass@1 values 1.0 and 0.0,
and macro pass@1 0.5; coverage is true. This is not the pooled 0.25. The incomplete
report retains total planned 4, attempted 3, the explicit missing slot, the actual
failed slot and macro 0.5; coverage is false. Both valid reports exit 0. See
[complete report](raw/task11-cli-eval-complete.txt) and
[incomplete report](raw/task11-cli-eval-incomplete.txt).

Duplicate outcomes, a changed completion plan, a tampered manifest seed, changed
gold content with the old manifest, and modified config budget each exit 2. Raw
failures remain in [duplicate](raw/task11-cli-eval-duplicate.txt),
[changed plan](raw/task11-cli-eval-modified-plan.txt),
[tampered manifest](raw/task11-cli-heldout-tampered.txt),
[changed gold](raw/task11-cli-heldout-changed-gold.txt), and
[config budget](raw/task11-cli-config-invalid-final.txt). The final invalid base
input is retained as [YAML](raw/task11-fixture-invalid-config-base.yaml); it changes
only planned train completions to 1024 from the final accepted proposal.

## Accounting and content contract

`pass_at_1(n,c)` implements c/n, the unbiased standard k=1 estimator. It rejects
booleans/nonintegers, n<=0 and c outside [0,n]; the combination-formula test includes
counts up to 10**100 without evaluating factorials in the implementation.
Reports preserve every planned slot. `observed` requires exact boolean correctness;
`failed` and `not_run` forbid it. Omitted slots are `missing`. Failed attempts count
incorrect; missing/not_run keep their denominators and make coverage incomplete.
The final score is the unweighted mean of each problem's c/n.

Records have id/question/source and optional string answer. The question hash is
SHA256 over Unicode NFC, CRLF/CR-to-LF, outer-stripped question text, preserving
internal whitespace/indentation and ignoring answers/source labels. Full supplied
train/candidate pools must have unique IDs/questions and no train/heldout overlap
before selection. It is exact project-data exclusion only, not semantic or
pretraining contamination evidence.

Ranking uses SHA256 of canonical JSON containing seed/id/question_sha256/source,
with ID tie breaking. Canonical JSON sorts keys, uses compact separators, UTF-8 and
ensure_ascii=False. A separate record_sha256 binds the entire supported raw record,
including optional gold. The canonical manifest binds complete input identity
pools, selected stable IDs/question hashes/record hashes/source identities,
normalization/algorithm/seed/count and the strict boolean synthetic_fixture flag.
Its own sha256 field is excluded from its digest. Verification recomputes from
full supplied content; self-consistency alone cannot verify upstream acquisition.

Evaluation plans require schema_version, strict synthetic_fixture,
heldout_manifest_sha256, and canonically ID-sorted problems containing selected
id/question_sha256/record_sha256/source/planned_completions. Outcomes bind the
complete plan_sha256, schema_version and matching fixture flag. The consumer checks
all selected identities against the supplied manifest, rejecting rebinding. No
real-data subset or output was selected. The API/schema and commands are documented
in [the evaluation README](../../rl/eval/README.md).

## Fixed pre-output proposal

The actual YAML family contains 11 configurations: the base, 9 one-factor ablations
and 1 separately identified external baseline proposal. Base: async k=1, group8,
KL0.001, clip0.2, scale_by_std=true. Required family coverage includes sync k=0,
async k=1/2, groups4/8/16, KL0/0.001/0.04, clip0.1/0.2/0.3 and no-std. The strict
safe_load resolver validates these concrete files, not just a hypothetical schema.
It rejects unknown/type-invalid values, arbitrary inheritance, incomplete families,
and model/data/budget/runtime/resource drift except exact declared factors. See
[config CLI](raw/task11-cli-config-valid-final.json) and
[base YAML](../../rl/configs/qwen25_05b_math_code.yaml).

Every proposal uses Qwen/Qwen2.5-0.5B-Instruct revision
7ae557604adf67be50417f59c2c2f167def9a775, seed17, LR1e-5, context2048,
temperature1/top_p1/top_k-1, raw model distribution. Train:128 unique GSM8K main/train
prompt IDs, 16 completions per prompt, 2048 planned completions, optimizer batch16,
max new tokens256. Groups4/8/16 all divide the same exposure and optimizer batch.
No actual selected IDs are known.

Before and after heldout proposal:64 GSM8K main/test plus64 MATH-500 test problems,
1 planned completion per problem, 128 planned per phase, max new tokens512, the
same data/seed/manifest/plan across variants. Selection and plan digests remain
null until reviewed acquisition and pre-output commitment; the loader intentionally
rejects filling them silently. These are proposals, not statistical-power,
improvement, model-fit or four-hour-feasibility claims. Do not retune after outputs
to force improvement.

Timing predeclares 3 matched pairs in chronological AB/BA/AB order, preserving
data/seed/model/budget/runtime/resources inside each pair. One GPU is time shared
between rollout and learner; maximum staleness is a planned schedule, not measured
concurrent GPU overlap. Proposed session ceiling14400 seconds, project cap
14000000000 bytes, at most2 checkpoints, resident services cap16000000000 bytes and
paid budget0 USD do not establish feasibility or authorize execution.

The separately identified synchronous external proposal is official verl v0.4.1
commit8d9e350ea58c7ad4b50dd14d9dcb50577242c55f, HF rollout/FSDP strategy,
runtime_verified=false. It shares the model/data/train/eval sample budget. The
source-only candidate manifest is
`.superpowers/sdd/5-crucible-gpu-ml-plan/task11-verl-v041-source/manifest.json`.
It is neither executed baseline evidence nor a replacement for project FSDP2.

## Expected provenance and readiness

Metadata was supplied by the root as expected upstream identity only. GSM8K
revision740312add88f781978c0658806c59bc2815b9866 has expected train asset
main/train-00000-of-00001.parquet,2306545B,LFS SHA256
ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d and test asset
main/test-00000-of-00001.parquet,419088B,LFS SHA256
ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59.
MATH-500 revision6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be test.jsonl expected446564B,
Git blob SHA1 2376b9a194b46c0790e197c91b7249e5f88ac09b. That is not SHA256.
Actual SHA256 remains unknown for all dataset assets. Exact metadata-response
hashes and asset Git identities are copied into the base proposal. No contents
were fetched or verified here.

Readiness is explicitly unverified for acquisition, tokenizer lengths, trusted
grading, optimizer/RAM/snapshot lifecycle, external dependency/ABI/Ray/time/memory
and bounded-session feasibility. Public tested code data has no fabricated pin;
actual isolated sandbox remains unavailable. No launch path exists. Original
actual math/code training, quality improvement, all equal-budget ablations,
external baseline, step timing, idle, synchronization and reward-hack observations
remain open.

## Resource boundary

Initial inherited-environment preflight exited 1 because HF aliases resolved away
from the required canonical shared cache, not because of quota or data failure.
The failed JSON is retained in
[inherited environment](raw/task11-resources-inherited-failed.json). The command-local
canonical rerun exited 0 at project13287940276B plus690000000B planned addition.
Before final tests the fresh canonical check exited 0 at project13288172088B,
plus690000000B, with no downloads/checkpoints/resident services. See
[initial pass](raw/task11-resources-before-tests.json) and
[final-test preflight](raw/task11-resources-before-final-tests.json).
Task11 has an entire10000000B CPU allowance replacing the closed future Task16
allowance; no project cap or combined planned reserve was increased. Owned fixture
hashes/cleanup and final accounting are recorded separately in raw evidence.
