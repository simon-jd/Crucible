"""CPU pass@1 accounting over predeclared slots and caller-trusted outcomes.

No inference, grading, code execution, benchmark eligibility or authenticity is
established here. Missing/not_run slots count in n and make coverage incomplete.
"""

import argparse
import json
import sys
from pathlib import Path

from rl.eval.heldout import digest, integer, keys, sha256, text, validate_manifest


def pass_at_1(n, c):
    """Unbiased standard k=1 estimator, c/n, without unstable combinations."""
    integer(n, 1)
    integer(c)
    if c > n:
        raise ValueError("correct count exceeds planned count")
    return c / n


def plan_digest(plan):
    return digest(plan)


def build_plan(manifest, completions):
    validate_manifest(manifest)
    selected = {row["id"]: row for row in manifest["selected"]}
    if type(completions) is not dict or set(completions) != set(selected):
        raise ValueError("completion budgets must cover exactly selected IDs")
    problems = []
    for identity in sorted(selected):
        integer(completions[identity], 1)
        problems.append({**selected[identity], "planned_completions": completions[identity]})
    return {
        "schema_version": 1,
        "synthetic_fixture": manifest["synthetic_fixture"],
        "heldout_manifest_sha256": manifest["sha256"],
        "problems": problems,
    }


def _validate_plan(plan, manifest):
    validate_manifest(manifest)
    keys(plan, {"schema_version", "synthetic_fixture", "heldout_manifest_sha256", "problems"})
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("unsupported plan schema")
    sha256(plan["heldout_manifest_sha256"])
    if type(plan["problems"]) is not list or not plan["problems"]:
        raise ValueError("problems must be a nonempty list")
    budgets = {}
    for row in plan["problems"]:
        keys(row, {"id", "question_sha256", "source", "record_sha256", "planned_completions"})
        identity = text(row["id"])
        if identity in budgets:
            raise ValueError("duplicate problem ID")
        sha256(row["question_sha256"])
        sha256(row["record_sha256"])
        text(row["source"])
        budgets[identity] = integer(row["planned_completions"], 1)
    expected = build_plan(manifest, budgets)
    if type(plan["synthetic_fixture"]) is not bool or plan != expected:
        raise ValueError("plan does not bind selected manifest identities/contract")
    return budgets


def evaluate(plan, outcomes, manifest):
    budgets = _validate_plan(plan, manifest)
    keys(outcomes, {"schema_version", "synthetic_fixture", "plan_sha256", "outcomes"})
    if type(outcomes["schema_version"]) is not int or outcomes["schema_version"] != 1:
        raise ValueError("unsupported outcomes schema")
    if (
        type(outcomes["synthetic_fixture"]) is not bool
        or outcomes["synthetic_fixture"] != plan["synthetic_fixture"]
    ):
        raise ValueError("synthetic fixture flags must match")
    if sha256(outcomes["plan_sha256"]) != plan_digest(plan):
        raise ValueError("outcomes bind a different plan")
    if type(outcomes["outcomes"]) is not list:
        raise ValueError("outcomes must be a list")
    slots = {}
    for row in outcomes["outcomes"]:
        keys(row, {"problem_id", "sample_index", "status"}, {"correct"})
        identity, index = text(row["problem_id"]), integer(row["sample_index"])
        if identity not in budgets or index >= budgets[identity] or (identity, index) in slots:
            raise ValueError("duplicate/unknown problem ID or sample index")
        status = row["status"]
        if status == "observed":
            if "correct" not in row or type(row["correct"]) is not bool:
                raise ValueError("observed outcome requires boolean correctness")
        elif status in ("failed", "not_run"):
            if "correct" in row:
                raise ValueError("failed/not_run outcomes forbid correctness field")
        else:
            raise ValueError("invalid outcome status")
        slots[identity, index] = dict(row)
    problems, attempted, complete = [], 0, True
    for identity, n in budgets.items():
        problem_slots = []
        for index in range(n):
            row = slots.get(
                (identity, index),
                {"problem_id": identity, "sample_index": index, "status": "missing"},
            )
            complete &= row["status"] in ("observed", "failed")
            attempted += row["status"] in ("observed", "failed")
            problem_slots.append(row)
        c = sum(row.get("correct", False) for row in problem_slots)
        problems.append(
            {"id": identity, "n": n, "c": c, "pass_at_1": pass_at_1(n, c), "slots": problem_slots}
        )
    return {
        "schema_version": 1,
        "synthetic_fixture": plan["synthetic_fixture"],
        "plan_sha256": plan_digest(plan),
        "heldout_manifest_sha256": manifest["sha256"],
        "coverage_complete": complete,
        "planned_completions": sum(budgets.values()),
        "attempted_completions": attempted,
        "macro_pass_at_1": sum(row["pass_at_1"] for row in problems) / len(problems),
        "problems": problems,
        "measurement_authenticity": "caller responsibility; schema is not execution evidence",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan, outcomes, manifest = [
            json.loads(path.read_text()) for path in (args.plan, args.outcomes, args.manifest)
        ]
        print(json.dumps(evaluate(plan, outcomes, manifest), indent=2))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"pass_at_1: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
