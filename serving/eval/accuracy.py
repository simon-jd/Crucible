"""Validate self-contained, paired BF16/FP8 raw evidence; never run inference.

Run from repository root: python -B -m serving.eval.accuracy BUNDLE.json.
JSON schema, normalization and evidence/acceptance limits: deploy/task13-README.md.
"""

import argparse
import hashlib
import json
import math
import re
from fractions import Fraction
from pathlib import Path

from scripts.experiment_results import _median

VARIANTS = ("baseline", "candidate")
STATUSES = ("success", "error", "truncated", "missing")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def number(value, name, minimum=0):
    check(
        type(value) in (int, float) and math.isfinite(value) and value >= minimum,
        f"{name} must be finite and >= {minimum}",
    )
    return value


def integer(value, name, minimum=0):
    check(type(value) is int and value >= minimum, f"{name} must be integer >= {minimum}")
    return value


def identity(value, name):
    check(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{name} must be a lowercase SHA256",
    )


def text(value, name):
    check(
        isinstance(value, str) and bool(value.strip()) and "REPLACE_" not in value,
        f"{name} must be a resolved nonempty string",
    )


def normalize(answer, policy):
    """Exact final number (optionally after last ####), or a single A-J letter.

    Numeric grouping must be valid; no units, prose guessing, rounding or tolerance.
    Decimal strings are compared as exact rational numbers. Invalid predictions lose.
    """
    if not isinstance(answer, str):
        return None
    answer = answer.strip()
    if policy == "choice-letter-v1":
        return answer.upper() if re.fullmatch("[A-Ja-j]", answer) else None
    check(policy == "numeric-final-v1", "unsupported normalization")
    answer = answer.rsplit("####", 1)[-1].strip()
    if not re.fullmatch(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", answer):
        return None
    return Fraction(answer.replace(",", ""))


def series(values):
    check(bool(values), "cannot summarize empty series")
    return {"median": _median(values), "min": min(values), "max": max(values)}


def ordered(items, expected, name):
    check(
        isinstance(items, list) and all(isinstance(x, dict) for x in items),
        f"{name} must be a list of objects",
    )
    ids = [x["id"] for x in items]
    check(
        ids == expected and len(set(ids)) == len(ids),
        f"{name} IDs/order must exactly match declared source; no missing/duplicate items",
    )


def conditions_valid(c):
    rounds = integer(c["planned_rounds"], "planned_rounds", 3)
    plan = c["run_plan"]
    check(
        isinstance(plan, list) and len(plan) == 2 * rounds, "run plan must cover all planned rounds"
    )
    for i, member in enumerate(plan):
        check(
            isinstance(member, dict) and set(member) == {"id", "round", "variant"},
            "run plan members require exactly id, round and variant",
        )
        text(member["id"], "planned run ID")
        rnd = i // 2 + 1
        expected = VARIANTS[i % 2] if rnd % 2 else VARIANTS[1 - i % 2]
        check(
            type(member["round"]) is int
            and member["round"] == rnd
            and member["variant"] == expected,
            "run plan must declare chronological AB/BA pairs",
        )
    ordered(plan, [member["id"] for member in plan], "run plan")
    for name in ("source_sha256", "runtime_sha256", "protocol_sha256"):
        identity(c[name], name)
    model = c["model"]
    for name in ("name", "revision"):
        text(model[name], f"model {name}")
    identity(model["sha256"], "model")
    # Optional: the candidate is a separately published quantized checkpoint of
    # the baseline model. It must name that base model, use the identical
    # tokenizer (so prompts are identical) and be a different artifact.
    if "candidate_model" in c:
        cand = c["candidate_model"]
        check(
            isinstance(cand, dict)
            and set(cand) == {"name", "revision", "sha256", "base_model", "tokenizer_sha256"},
            "candidate_model requires name, revision, sha256, base_model, tokenizer_sha256",
        )
        for name in ("name", "revision"):
            text(cand[name], f"candidate model {name}")
        identity(cand["sha256"], "candidate model")
        identity(cand["tokenizer_sha256"], "candidate tokenizer")
        identity(model["tokenizer_sha256"], "baseline tokenizer")
        check(cand["base_model"] == model["name"], "candidate checkpoint must derive from the baseline model")
        check(cand["tokenizer_sha256"] == model["tokenizer_sha256"], "candidate tokenizer differs from baseline")
        check(cand["sha256"] != model["sha256"], "candidate_model must be a different checkpoint")
    integer(c["gpu_count"], "gpu_count", 1)
    check(isinstance(c["sampling"], dict) and bool(c["sampling"]), "sampling required")
    integer(c["sampling"]["seed"], "sampling seed")
    number(c["sampling"]["temperature"], "temperature")
    integer(c["sampling"]["max_new_tokens"], "max_new_tokens", 1)
    slo = c["slo"]
    check(slo["calibration_precision"] == "bf16", "SLO must come from fixed BF16 calibration")
    identity(slo["calibration_sha256"], "calibration")
    number(slo["ttft_ms"], "SLO TTFT")
    number(slo["tpot_ms"], "SLO TPOT")
    workload = c["workload"]
    check(isinstance(workload, list) and bool(workload), "nonempty workload required")
    previous = -1
    for x in workload:
        text(x["id"], "request ID")
        identity(x["prompt_sha256"], "prompt")
        number(x["scheduled_s"], "scheduled_s")
        check(x["scheduled_s"] >= previous, "workload must be scheduled chronologically")
        previous = x["scheduled_s"]
        integer(x["input_tokens"], "input_tokens", 1)
        integer(x["requested_tokens"], "requested_tokens", 1)
    check(workload[0]["scheduled_s"] == 0, "relative clock must start at first scheduled request")
    ordered(workload, [x["id"] for x in workload], "workload")
    heldout = c["heldout"]
    for name in ("dataset", "subset", "revision"):
        text(heldout[name], f"heldout {name}")
    check(
        heldout["normalization"] in ("numeric-final-v1", "choice-letter-v1"),
        "unsupported normalization",
    )
    examples = heldout["examples"]
    check(isinstance(examples, list) and bool(examples), "nonempty heldout examples required")
    for x in examples:
        text(x["id"], "heldout ID")
        identity(x["prompt_sha256"], "heldout prompt")
        check(normalize(x["label"], heldout["normalization"]) is not None, "invalid correct label")
    ordered(examples, [x["id"] for x in examples], "heldout")


def status_valid(item):
    check(item["status"] in STATUSES, "unknown outcome status")
    if item["status"] in ("error", "missing"):
        text(item["error"], "failure reason")
    else:
        check(item["error"] is None, "successful/truncated item cannot carry an error")


def performance(run, c, layout):
    requests = run["requests"]
    ordered(requests, [x["id"] for x in c["workload"]], "performance")
    check(type(run["aborted"]) is bool, "aborted must be explicit boolean")
    counts = {s: 0 for s in STATUSES}
    counts.update(
        source=len(requests),
        sender_invalid=0,
        output_tokens=0,
        successful_output_tokens=0,
        slo_successes=0,
        undefined_ttft=0,
        undefined_tpot=0,
        undefined_kv_transfer=0,
    )
    ends, kv = [], []
    for req, planned in zip(requests, c["workload"], strict=True):
        status_valid(req)
        counts[req["status"]] += 1
        n = integer(req["output_tokens"], "output_tokens")
        check(n <= planned["requested_tokens"], "output_tokens exceeds requested bound")
        counts["output_tokens"] += n
        send, end = req["send_s"], req["end_s"]
        if send is not None:
            number(send, "send_s")
            check(send >= planned["scheduled_s"], "send precedes schedule")
        if end is not None:
            number(end, "end_s")
            check(
                end >= (send if send is not None else planned["scheduled_s"]),
                "completion precedes send/schedule",
            )
            ends.append(end)
        if send is None or send - planned["scheduled_s"] > 0.005 + 1e-12:
            counts["sender_invalid"] += 1
        for key, count in [
            ("ttft_ms", "undefined_ttft"),
            ("tpot_ms", "undefined_tpot"),
            ("kv_transfer_ms", "undefined_kv_transfer"),
        ]:
            if req[key] is None:
                counts[count] += 1
            else:
                number(req[key], key)
        if req["kv_transfer_ms"] is not None:
            check(layout == "pd", "colocated KV transfer must be null (not applicable)")
            kv.append(req["kv_transfer_ms"])
        if req["status"] == "missing":
            check(send is None and end is None and n == 0, "missing request has observations")
        else:
            check(end is not None, "attempted request needs terminal time even on failure")
        if n == 0:
            check(
                req["ttft_ms"] is None and req["tpot_ms"] is None,
                "zero output cannot have token timings",
            )
        if n < 2:
            check(req["tpot_ms"] is None, "TPOT undefined for fewer than two output tokens")
        if send is not None and end is not None:
            duration_ms = (end - send) * 1000
            for key in ("ttft_ms", "kv_transfer_ms"):
                check(
                    req[key] is None or req[key] <= duration_ms + 1e-6,
                    f"{key} exceeds request duration",
                )
            if req["ttft_ms"] is not None and req["tpot_ms"] is not None:
                check(
                    req["ttft_ms"] + (n - 1) * req["tpot_ms"] <= duration_ms + 1e-6,
                    "token timing exceeds duration",
                )
        if req["status"] == "success":
            check(
                send is not None and n > 0 and req["ttft_ms"] is not None,
                "successful request needs send, tokens and TTFT",
            )
            check(n < 2 or req["tpot_ms"] is not None, "successful multi-token TPOT missing")
            if layout == "pd":
                check(req["kv_transfer_ms"] is not None, "P/D KV transfer is unmeasured")
            counts["successful_output_tokens"] += n
            if (
                req["ttft_ms"] <= c["slo"]["ttft_ms"]
                and req["tpot_ms"] is not None
                and req["tpot_ms"] <= c["slo"]["tpot_ms"]
            ):
                counts["slo_successes"] += 1
    duration = number(run["observation_end_s"], "observation_end_s")
    last_completion = max(ends, default=0)
    check(duration >= last_completion, "observation ends before an observed completion")
    if not counts["missing"]:
        check(
            duration == last_completion,
            "without missing rows observation must end at final completion",
        )
    if not run["aborted"]:
        check(
            duration >= c["workload"][-1]["scheduled_s"],
            "non-aborted observation must cover the entire declared schedule",
        )
    valid = not run["aborted"] and counts["sender_invalid"] / len(requests) <= 0.01
    valid = valid and duration > 0
    throughput = counts["successful_output_tokens"] / duration if duration else None
    goodput = counts["slo_successes"] / duration if duration else None
    for metric in (throughput, goodput):
        if metric is not None:
            number(metric, "derived performance metric")
    return {
        "id": run["id"],
        "round": run["round"],
        "variant": run["variant"],
        "valid": valid,
        "counts": counts,
        "duration_s": duration,
        "last_completion_s": last_completion,
        "missing_terminal_observation_s": duration if counts["missing"] else None,
        "schedule_covered": duration >= c["workload"][-1]["scheduled_s"],
        "throughput_tokens_s": throughput,
        "slo_goodput_requests_s": goodput,
        "kv_transfer_ms": series(kv) if kv else None,
    }


def report(bundle):
    """Pure schema/accounting validation; input data is evidence, not an attestation."""
    # Canonicalization also rejects NaN/Infinity anywhere, including unused metadata.
    digest(bundle)
    check(
        bundle["schema"] == 2 and type(bundle["schema"]) is int,
        "schema 2 required: explicit run plan and observation bounds; schema 1 is not inferred",
    )
    check(bundle["kind"] in ("fixture", "measured"), "kind must be fixture or measured")
    c, configs = bundle["conditions"], bundle["configurations"]
    conditions_valid(c)
    check(set(configs) == set(VARIANTS), "both configurations required")
    for v, precision in zip(VARIANTS, ("bf16", "fp8"), strict=True):
        config = configs[v]
        check(
            set(config) == {"precision", "backend", "settings", "model_sha256", "runtime_sha256"},
            "config requires precision/backend/identity and all other options in settings",
        )
        check(config["precision"] == precision, f"{v} must be {precision}")
        text(config["backend"], "actual backend")
        lineage = c["candidate_model"] if v == "candidate" and "candidate_model" in c else c["model"]
        check(config["model_sha256"] == lineage["sha256"], "model lineage mismatch")
        check(config["runtime_sha256"] == c["runtime_sha256"], "runtime mismatch")
        check(config["settings"]["layout"] in ("colocated", "pd"), "unknown layout")
    check(
        configs["baseline"]["settings"] == configs["candidate"]["settings"],
        "precision comparison requires identical non-precision settings",
    )
    runs = bundle["runs"]
    plan = c["run_plan"]
    ordered(runs, [member["id"] for member in plan], "run plan")
    results = []
    previous_end = -1
    for member, run in zip(plan, runs, strict=True):
        rnd, expected = member["round"], member["variant"]
        check(
            type(run["round"]) is int and run["round"] == rnd and run["variant"] == expected,
            "runs must retain predeclared chronological AB/BA counterbalanced order",
        )
        text(run["id"], "run ID")
        check(run["config_sha256"] == digest(configs[expected]), "performance config mismatch")
        check(run["conditions_sha256"] == digest(c), "performance conditions mismatch")
        start = integer(run["started_at_ns"], "started_at_ns")
        end = integer(run["finished_at_ns"], "finished_at_ns")
        check(start >= previous_end and end > start, "run clocks overlap or reverse chronology")
        previous_end = end
        measured = performance(run, c, configs[expected]["settings"]["layout"])
        check(
            Fraction(str(measured["duration_s"])) * 1_000_000_000 <= end - start,
            "observation duration exceeds run clock bracket",
        )
        results.append(measured)
    check(set(bundle["quality"]) == set(VARIANTS), "both heldout quality records required")
    quality, paired = {}, []
    examples = c["heldout"]["examples"]
    for v in VARIANTS:
        q = bundle["quality"][v]
        check(q["config_sha256"] == digest(configs[v]), "quality config mismatch")
        check(q["conditions_sha256"] == digest(c), "quality conditions mismatch")
        ordered(q["items"], [x["id"] for x in examples], "quality")
        counts = {s: 0 for s in STATUSES}
        counts.update(source=len(examples), correct=0, unparseable=0)
        outcomes = []
        for item, ex in zip(q["items"], examples, strict=True):
            status_valid(item)
            check(
                item["prompt_sha256"] == ex["prompt_sha256"] and item["label"] == ex["label"],
                "quality prompt/correct label mismatch",
            )
            check(
                item["answer"] is None or isinstance(item["answer"], str),
                "answer must be text/null",
            )
            counts[item["status"]] += 1
            pred = normalize(item["answer"], c["heldout"]["normalization"])
            correct = item["status"] == "success" and pred == normalize(
                ex["label"], c["heldout"]["normalization"]
            )
            counts["correct"] += int(correct)
            counts["unparseable"] += int(pred is None)
            outcomes.append(correct)
        quality[v] = {
            "counts": counts,
            "accuracy": counts["correct"] / len(examples),
            "correct_by_item": outcomes,
        }
    for i, ex in enumerate(examples):
        paired.append(
            {
                **ex,
                **{
                    v: {
                        **bundle["quality"][v]["items"][i],
                        "correct": quality[v]["correct_by_item"][i],
                    }
                    for v in VARIANTS
                },
            }
        )
    valid = all(r["valid"] for r in results)
    aggregate = {}
    for field in ("throughput_tokens_s", "slo_goodput_requests_s"):
        aggregate[field] = (
            {v: series([r[field] for r in results if r["variant"] == v]) for v in VARIANTS}
            if valid
            else None
        )
    delta = quality["candidate"]["accuracy"] - quality["baseline"]["accuracy"]
    return {
        "schema": 2,
        "kind": bundle["kind"],
        "input_sha256_canonical": digest(bundle),
        "comparison_eligible": valid and bundle["kind"] == "measured",
        "candidate_checkpoint": c["candidate_model"]["name"] if "candidate_model" in c else None,
        "benchmark_accepted": False,
        "acceptance_note": "Schema validation is not native evidence review or authenticity proof.",
        "conditions": c,
        "configurations": configs,
        "identities": {
            "conditions_sha256": digest(c),
            "slo_sha256": digest(c["slo"]),
            "heldout_sha256": digest(c["heldout"]),
            "workload_sha256": digest(c["workload"]),
            "run_plan_sha256": digest(plan),
            "config_sha256": {v: digest(configs[v]) for v in VARIANTS},
        },
        "rounds": c["planned_rounds"],
        "performance_valid": valid,
        "throughput": aggregate["throughput_tokens_s"],
        "slo_goodput": aggregate["slo_goodput_requests_s"],
        "runs": results,
        "raw_runs": runs,
        "quality": quality,
        "paired_outcomes": paired,
        "accuracy_delta": delta,
        "absolute_accuracy_difference": abs(delta),
        "accuracy_unit": "fraction; delta = candidate - baseline; all source items in denominator",
        "pretraining_contamination_excluded": False,
    }


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        check(key not in out, f"duplicate JSON key: {key}")
        out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--allow-fixture", action="store_true", help="emit visibly synthetic report"
    )
    args = parser.parse_args()
    try:
        raw = args.bundle.read_bytes()
        bundle = json.loads(raw, object_pairs_hook=unique_object)
        check(args.allow_fixture or bundle["kind"] != "fixture", "fixture requires --allow-fixture")
        result = report(bundle)
        result["raw_evidence"] = {
            "path": str(args.bundle),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
        parser.exit(1, f"invalid evidence (retain original input): {error}\n")


if __name__ == "__main__":
    main()
