# CPU evaluation and experiment configs

These modules account for caller-trusted correctness observations. They do not
generate completions, grade math, execute code, launch training, authenticate a
measurement, or establish benchmark eligibility.

`heldout.select_heldout(train, candidates, count=..., seed=...,
synthetic_fixture=...)` accepts nonempty lists of JSON records with exact fields
`id`, `question`, `source`, and optional string `answer`. IDs and source identities
are nonempty strings. Use a source identity that includes its upstream revision
and asset identity. IDs must be unique across the full supplied inputs. No actual
benchmark content is included here.

Question hashes normalize Unicode NFC, replace CRLF and CR with LF, and strip only
outer whitespace. Internal whitespace and indentation remain unchanged. Answers
and source labels cannot hide question duplication or train/heldout overlap.
Duplicates and overlap in the complete candidate pools fail before selection.
This is exact project-data exclusion, not semantic matching or evidence about
pretraining contamination.

Ranking sorts ascending SHA256 of canonical JSON containing `seed`, `id`,
`question_sha256`, and `source`, with ID as tie breaker. Count is a positive integer
bounded by candidate count; seed is a nonnegative integer; booleans are rejected
as integers. Input order cannot affect selection. Each identity also carries
`record_sha256` over the complete supported raw record, including optional answer.
The manifest binds complete train/candidate identity pools, selected identities,
seed/count/normalization/algorithm, and the strict boolean fixture flag. Canonical
JSON uses sorted keys, separators `(',', ':')`, UTF-8 and `ensure_ascii=False`.
The manifest SHA256 excludes only its own `sha256` field. `validate_manifest`
checks self-consistency; `verify_manifest(manifest, train, candidates)` additionally
recomputes hashes from full supplied content. Neither authenticates upstream data.

`pass_at_1.build_plan(manifest, {id: positive_count, ...})` binds every selected ID,
question hash, record hash, source identity, and the complete manifest SHA256.
It returns `{schema_version: 1, synthetic_fixture: bool,
heldout_manifest_sha256: hex, problems: [...]}`. Problem rows contain `id`,
`question_sha256`, `record_sha256`, `source`, `planned_completions`; they are sorted
by ID. Commit the manifest and plan before real model outputs; current real-data
selection identities remain unknown.

`evaluate(plan, outcomes, manifest)` accepts an outcomes object with exact fields
`schema_version: 1`, matching boolean `synthetic_fixture`, `plan_sha256`, and
`outcomes`. Each row has `problem_id`, zero-based integer `sample_index`, and
`status`: `observed` requires an exact boolean `correct`; `failed` and `not_run`
forbid that field. Duplicate/unknown/extra IDs, indices or fields fail. Omitted
slots become explicit `missing` rows. All planned slots count in the denominator;
attempted failures count incorrect. Missing/not_run make coverage incomplete.
Reports preserve each problem's `n`, `c`, `pass_at_1` and slots, and compute the
unweighted macro mean of `c/n`, never pooled length weighting. Fixture flags remain
visible. An incomplete report exits successfully because it is a valid report,
while invalid input exits with status 2.

Run the retained synthetic fixtures from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m rl.eval.heldout --train docs/evidence/raw/task11-fixture-train.json --candidates docs/evidence/raw/task11-fixture-candidates.json --count 2 --seed 17 --synthetic-fixture
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m rl.eval.heldout --train docs/evidence/raw/task11-fixture-train.json --candidates docs/evidence/raw/task11-fixture-candidates.json --verify docs/evidence/raw/task11-fixture-manifest.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m rl.eval.pass_at_1 --manifest docs/evidence/raw/task11-fixture-manifest.json --plan docs/evidence/raw/task11-fixture-plan.json --outcomes docs/evidence/raw/task11-fixture-outcomes-incomplete.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m rl.configs.experiments
```

The config loader resolves the base YAML plus nine one-factor ablations and the
separately identified external synchronous verl HF/FSDP variant, using
`yaml.safe_load`. The base YAML is editable (fill in pins, hashes, budgets). The
loader rejects unknown or missing top-level keys, family omissions/additions,
ablations that change anything beyond their declared factor, and resources beyond
the resource budget (one GPU, $0, 4 h, 2 checkpoints, 16 GB services, 14 GB disk).

Group size divides both fixed prompt exposure and optimizer batch. The base and
variants share all sampling, data, seed and resource conditions except their
declared factor; external framework plus fresh schedule is explicitly separate.
Timing is predeclared as matched AB/BA/AB pairs; `k` describes planned bounded
staleness under time sharing, never measured concurrent GPU overlap. Expected
dataset hashes are upstream metadata only: GSM LFS SHA256 is not locally verified,
and MATH Git blob SHA1 is not SHA256. Actual SHA256 and selected IDs are unknown.

These are planned budgets, not statistical-power, improvement, model-fit or
four-hour-feasibility claims. All real training, trusted grading, timing/idle/sync/
reward-hack measurements, ablations, equal-budget external execution and original
code experiments remain required. Code data pin and actual isolated sandbox are
unavailable. Acquisition, tokenizer lengths, optimizer/RAM/snapshot lifecycle,
external ABI/Ray/memory and time feasibility are still open. Do not retune after
seeing held-out results to force an improvement.
