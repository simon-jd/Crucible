# Speculative decoding counter evaluator

`eval_acceptance.py` is an independent CPU utility. It does not train a draft,
run inference, collect timings, grade answers, or authenticate measurements.
All checked-in Task15 examples are explicitly synthetic.

`acceptance_length(traces)` accepts a nonempty list of terminal cumulative rows,
exactly once per request:

```json
{"request_id":"synthetic-r0","spec_accepted_tokens":6,"spec_verify_ct":2}
```

It computes `1 + sum(spec_accepted_tokens) / sum(spec_verify_ct)`: counter-based
acceptance length including the guaranteed verification token. Accepted draft
tokens exclude that token. With no accepted draft tokens a verification contributes one. The
result is weighted by verification steps, not a mean of request ratios, and is
neither an acceptance probability nor evidence of speedup. All-zero verification
is unavailable and raises `ValueError`; a report instead uses JSON `null`.
Upstream verification counters do not establish exact post-EOS/stop-trimmed
client output or equality with `completion_tokens`. Native counter provenance
and stop behavior require later attestation.

The meaning follows [SGLang v0.5.5 per-request counter updates](https://github.com/sgl-project/sglang/blob/v0.5.5/python/sglang/srt/managers/scheduler_output_processor_mixin.py).
The server-lifetime aggregate read by its serving benchmark cannot isolate
heldout requests. Completion token counts, including prefill, cannot substitute
for the exact verification counters.

`build_plan(manifest, selected_records, *, protocol, target, draft, requests,
synthetic_fixture)` creates a predeclared v1 plan. `validate_plan(plan)` verifies
it and `evaluate(plan, outcomes)` creates the report. Both return independent
JSON data. Canonical SHA256 uses the existing `rl.eval.heldout.digest` helper.

A plan has exactly `schema_version`, `synthetic_fixture`, `heldout_manifest`,
`selected_records`, `protocol`, `target`, `draft`, `requests`, and `sha256`.
The full manifest is validated through `rl.eval.heldout.validate_manifest`.
Every selected full record must match its selected ID, normalized question hash,
source, and full-record hash (including any answer/gold). The complete supplied
manifest, full selected records, and all protocol/model/batch identities enter
the digest. This establishes internal identity consistency; it cannot prove the
manifest's acquisition history, actual heldout eligibility, or native execution.

Each of protocol/target/draft is exactly `{"name": "...", "sha256": "..."}`;
the hash is lowercase hex with 64 digits. No actual target or draft was selected
in this phase. Requests contain exactly `request_id`, `prompt_id`, `batch_id`,
`batch_size`, `slot`. IDs are nonblank, request IDs and batch slots are unique,
sizes are 1–64, and every batch has all slots 0 through size minus one. Every
selected prompt must occur. Multiple batches may share a size. Requests and full
selected records are sorted by ID; changing any bound content invalidates old
outcomes. `build_plan` canonicalizes input order; validation rejects an already
sealed plan with a changed order/digest.

Outcomes have exactly `schema_version: 1`, `plan_sha256`, the matching strict
boolean `synthetic_fixture`, and `rows`. Each row has `request_id` and `status`:

- `completed` requires integer nonnegative `spec_accepted_tokens` and
  `spec_verify_ct`. Zero verifications requires zero accepted tokens.
- `failed` requires a nonblank `reason`. Counters may be both absent, both null,
  or a valid observed prefix pair. Unknown counters remain unknown. Prefixes
  survive in `failed_prefixes` and never enter the completed-row statistic.
- `not_run` requires a nonblank `reason` and forbids counter fields.

Omitted planned rows become `missing`. Unknown fields/IDs, duplicate terminal
rows (including streamed cumulative snapshots), booleans as counts, and malformed
counter pairs are errors. JSON duplicate fields and nonfinite constants are
rejected. Reports preserve every slot, all reasons, missing/failure counts, and
completed-only conditional statistics overall and per batch.
`coverage_complete` excludes missing/not_run but permits recorded failures;
`all_requests_completed` also excludes failures. `complete_counter_report`
requires all requests completed and an available statistic. It never establishes
scientific, heldout, native, or benchmark acceptance. Every report retains the
fixture flag and explicitly unverified authenticity.

Consume an explicit predeclared plan without generating a new selection:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m spec.eval_acceptance \
  --plan docs/evidence/raw/task15-cpu-synthetic-plan.json \
  --outcomes docs/evidence/raw/task15-cpu-synthetic-valid-input.json
```

CLI exit codes are 0 for a complete usable counter report, 1 for incomplete,
failed or unavailable counters, and 2 for malformed input. Invalid diagnostics
use `synthetic_fixture: null` because rejected input cannot establish provenance.

Actual SpecForge training, heldout measurements, batch 1–64 end-to-end TPOT,
feature/checkpoint integration and real faults remain required and open.
