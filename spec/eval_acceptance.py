"""CPU counter reports; identity consistency does not prove authentic measurement."""

import argparse
import json
import math
import sys
from pathlib import Path

from rl.eval.heldout import digest, identities, integer, keys, sha256, text, validate_manifest


def _counters(row):
    accepted = integer(row["spec_accepted_tokens"])
    verifies = integer(row["spec_verify_ct"])
    if verifies == 0 and accepted != 0:
        raise ValueError("zero verifications require zero accepted draft tokens")
    return accepted, verifies


def acceptance_length(traces):
    """Counter-based acceptance length including the guaranteed verification token.

    Rows are terminal cumulative counters exactly once per request. This is not
    draft acceptance probability, training accuracy, or an inferred speedup.
    Upstream verification counters do not establish exact post-EOS/stop-trimmed
    client output or equality with completion_tokens. Native counter provenance
    and stop behavior require later attestation.
    """
    if type(traces) is not list or not traces:
        raise ValueError("terminal traces must be a nonempty list")
    seen, accepted, verifies = set(), 0, 0
    for row in traces:
        keys(row, {"request_id", "spec_accepted_tokens", "spec_verify_ct"})
        request_id = text(row["request_id"])
        if request_id in seen:
            raise ValueError("duplicate terminal request; streamed snapshots are invalid")
        seen.add(request_id)
        a, v = _counters(row)
        accepted += a
        verifies += v
    if verifies == 0:
        raise ValueError("acceptance length unavailable without verification steps")
    try:
        result = 1 + accepted / verifies
    except OverflowError as exc:
        raise ValueError("acceptance length is not finite") from exc
    if not math.isfinite(result):
        raise ValueError("acceptance length is not finite")
    return result


def build_plan(manifest, selected_records, *, protocol, target, draft, requests, synthetic_fixture):
    """Bind a predeclared complete manifest, selected full records, and batch slots.

    The manifest's structural consistency is checked; its acquisition provenance
    and full pool contents cannot be authenticated by this API.
    """
    validate_manifest(manifest)
    if type(synthetic_fixture) is not bool or synthetic_fixture != manifest["synthetic_fixture"]:
        raise ValueError("strict synthetic_fixture must match heldout manifest")
    selected = identities(selected_records)
    if selected != sorted(manifest["selected"], key=lambda row: row["id"]):
        raise ValueError("selected full records do not match manifest identities")
    for identity in (protocol, target, draft):
        keys(identity, {"name", "sha256"})
        text(identity["name"])
        sha256(identity["sha256"])
    if type(requests) is not list or not requests:
        raise ValueError("planned requests must be a nonempty list")
    prompts = {row["id"] for row in selected}
    seen, used, groups = set(), set(), {}
    for row in requests:
        keys(row, {"request_id", "prompt_id", "batch_id", "batch_size", "slot"})
        request_id, prompt_id, batch_id = [
            text(row[k]) for k in ("request_id", "prompt_id", "batch_id")
        ]
        size, slot = integer(row["batch_size"], 1), integer(row["slot"])
        if size > 64 or slot >= size or request_id in seen or prompt_id not in prompts:
            raise ValueError("invalid/duplicate planned request, prompt, or batch slot")
        seen.add(request_id)
        used.add(prompt_id)
        previous_size, slots = groups.setdefault(batch_id, (size, set()))
        if previous_size != size or slot in slots:
            raise ValueError("conflicting batch size or duplicate batch slot")
        slots.add(slot)
    if used != prompts or any(slots != set(range(size)) for size, slots in groups.values()):
        raise ValueError("every selected prompt and every declared batch slot must be planned")
    body = {
        "schema_version": 1,
        "synthetic_fixture": synthetic_fixture,
        "heldout_manifest": manifest,
        "selected_records": sorted(selected_records, key=lambda row: row["id"]),
        "protocol": protocol,
        "target": target,
        "draft": draft,
        "requests": sorted(requests, key=lambda row: row["request_id"]),
    }
    # Return independent JSON data, never aliases to caller-owned dictionaries.
    return json.loads(json.dumps({**body, "sha256": digest(body)}, ensure_ascii=False))


def validate_plan(plan):
    keys(
        plan,
        {
            "schema_version",
            "synthetic_fixture",
            "heldout_manifest",
            "selected_records",
            "protocol",
            "target",
            "draft",
            "requests",
            "sha256",
        },
    )
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("unsupported plan schema_version")
    sha256(plan["sha256"])
    expected = build_plan(
        plan["heldout_manifest"],
        plan["selected_records"],
        protocol=plan["protocol"],
        target=plan["target"],
        draft=plan["draft"],
        requests=plan["requests"],
        synthetic_fixture=plan["synthetic_fixture"],
    )
    if plan != expected:
        raise ValueError("plan digest/order/content mismatch")
    return expected


def _statistic(rows):
    completed = [row for row in rows if row["status"] == "completed"]
    traces = [
        {k: row[k] for k in ("request_id", "spec_accepted_tokens", "spec_verify_ct")}
        for row in completed
    ]
    accepted = sum(row["spec_accepted_tokens"] for row in completed)
    verifies = sum(row["spec_verify_ct"] for row in completed)
    return {
        "scope": "conditional on completed terminal rows only; includes guaranteed token",
        "completed_requests": len(completed),
        "spec_accepted_tokens": accepted,
        "spec_verify_ct": verifies,
        "acceptance_length": acceptance_length(traces) if verifies else None,
    }


def evaluate(plan, outcomes):
    """Preserve every planned slot; failed prefixes never enter the main statistic."""
    plan = validate_plan(plan)
    keys(outcomes, {"schema_version", "plan_sha256", "synthetic_fixture", "rows"})
    if type(outcomes["schema_version"]) is not int or outcomes["schema_version"] != 1:
        raise ValueError("unsupported outcomes schema_version")
    if (
        type(outcomes["synthetic_fixture"]) is not bool
        or outcomes["synthetic_fixture"] != plan["synthetic_fixture"]
        or outcomes["plan_sha256"] != plan["sha256"]
    ):
        raise ValueError("outcomes do not bind this complete plan and fixture flag")
    if type(outcomes["rows"]) is not list:
        raise ValueError("outcome rows must be a list")
    planned = {row["request_id"] for row in plan["requests"]}
    observed = {}
    counter_keys = {"spec_accepted_tokens", "spec_verify_ct"}
    for row in outcomes["rows"]:
        keys(row, {"request_id", "status"}, counter_keys | {"reason"})
        request_id = text(row["request_id"])
        status = row["status"]
        if request_id not in planned or request_id in observed:
            raise ValueError("unknown or duplicate terminal outcome")
        if status == "completed":
            keys(row, {"request_id", "status"} | counter_keys)
            _counters(row)
        elif status == "failed":
            text(row.get("reason"))
            supplied = counter_keys & row.keys()
            if supplied and supplied != counter_keys:
                raise ValueError("failed prefix counters require a complete pair")
            if supplied and not all(row[key] is None for key in counter_keys):
                _counters(row)
        elif status == "not_run":
            keys(row, {"request_id", "status", "reason"})
            text(row["reason"])
        else:
            raise ValueError("unknown terminal outcome status")
        observed[request_id] = row
    rows = [
        {**request, **observed.get(request["request_id"], {"status": "missing"})}
        for request in plan["requests"]
    ]
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("completed", "failed", "not_run", "missing")
    }
    counts.update(planned=len(rows), observed=len(observed))
    statistic = _statistic(rows)
    result = {
        "schema_version": 1,
        "synthetic_fixture": plan["synthetic_fixture"],
        "authenticity": "unverified; schema/hash consistency does not establish measurement authenticity",
        "plan_sha256": plan["sha256"],
        "counts": counts,
        "coverage_complete": counts["missing"] == 0 and counts["not_run"] == 0,
        "all_requests_completed": counts["completed"] == len(rows),
        "complete_counter_report": counts["completed"] == len(rows)
        and statistic["acceptance_length"] is not None,
        "counter_statistic": statistic,
        "per_batch": [
            {
                "batch_id": batch_id,
                "counter_statistic": _statistic([r for r in rows if r["batch_id"] == batch_id]),
            }
            for batch_id in sorted({r["batch_id"] for r in rows})
        ],
        "failed_prefixes": [row for row in rows if row["status"] == "failed"],
        "requests": rows,
    }
    return json.loads(json.dumps(result, ensure_ascii=False))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object field")
        result[key] = value
    return result


def _load(path):
    def invalid_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)

    return json.loads(
        path.read_text(), object_pairs_hook=_unique_object, parse_constant=invalid_constant
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = evaluate(_load(args.plan), _load(args.outcomes))
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0 if result["complete_counter_report"] else 1
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(
            json.dumps({"synthetic_fixture": None, "authenticity": "unverified", "error": str(exc)})
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
