"""CPU accounting tests; every completion here is explicitly synthetic."""

import copy
from math import comb

import pytest

from rl.eval.heldout import select_heldout
from rl.eval.pass_at_1 import build_plan, evaluate, pass_at_1, plan_digest


def fixture():
    manifest = select_heldout(
        [{"id": "train", "question": "train question", "source": "fixture/train"}],
        [
            {"id": "a", "question": "question a", "source": "fixture/test"},
            {"id": "b", "question": "question b", "source": "fixture/test"},
        ],
        count=2,
        seed=17,
        synthetic_fixture=True,
    )
    plan = build_plan(manifest, {"a": 1, "b": 3})
    outcomes = {
        "schema_version": 1,
        "plan_sha256": plan_digest(plan),
        "synthetic_fixture": True,
        "outcomes": [
            {"problem_id": "a", "sample_index": 0, "status": "observed", "correct": True},
            {"problem_id": "b", "sample_index": 0, "status": "failed"},
            {"problem_id": "b", "sample_index": 1, "status": "observed", "correct": False},
            {"problem_id": "b", "sample_index": 2, "status": "not_run"},
        ],
    }
    return manifest, plan, outcomes


def test_pass_at_1_unbiased_estimator():
    for n, c in [(1, 0), (1, 1), (9, 4), (100, 99), (10**100, 10**99)]:
        assert pass_at_1(n, c) == pytest.approx(1 - comb(n - c, 1) / comb(n, 1))
    for n, c in [(True, 1), (1, False), (0, 0), (-1, 0), (2, 3), (2, -1), (2.0, 1)]:
        with pytest.raises(ValueError):
            pass_at_1(n, c)


def test_planned_macro_denominators_and_explicit_missing():
    manifest, plan, outcomes = fixture()
    report = evaluate(plan, outcomes, manifest)
    assert report["macro_pass_at_1"] == 0.5  # Not the pooled 1/4.
    assert not report["coverage_complete"]
    assert report["planned_completions"] == 4
    assert report["attempted_completions"] == 3
    assert report["problems"][1]["n"] == 3
    assert report["problems"][1]["slots"][0]["status"] == "failed"
    outcomes["outcomes"].pop()
    assert evaluate(plan, outcomes, manifest)["problems"][1]["slots"][2]["status"] == "missing"
    outcomes["outcomes"].append({"problem_id": "b", "sample_index": 2, "status": "failed"})
    assert evaluate(plan, outcomes, manifest)["coverage_complete"]


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "unknown",
        "index",
        "bool_index",
        "status",
        "correct",
        "failed_correct",
        "flag",
        "extra",
    ],
)
def test_invalid_outcomes_rejected(change):
    manifest, plan, outcomes = fixture()
    row = outcomes["outcomes"][0]
    if change == "duplicate":
        outcomes["outcomes"].append(copy.deepcopy(row))
    elif change == "unknown":
        row["problem_id"] = "unknown"
    elif change == "index":
        row["sample_index"] = 1
    elif change == "bool_index":
        row["sample_index"] = False
    elif change == "status":
        row["status"] = "missing"
    elif change == "correct":
        row["correct"] = 1
    elif change == "failed_correct":
        row["status"] = "failed"
    elif change == "flag":
        outcomes["synthetic_fixture"] = False
    else:
        row["extra"] = 1
    with pytest.raises(ValueError):
        evaluate(plan, outcomes, manifest)


@pytest.mark.parametrize(
    "field", ["planned_completions", "question_sha256", "record_sha256", "source", "id"]
)
def test_rebound_or_modified_plan_rejected(field):
    manifest, plan, outcomes = fixture()
    plan["problems"][0][field] = 2 if field == "planned_completions" else "changed"
    with pytest.raises(ValueError):
        evaluate(plan, outcomes, manifest)
    outcomes["plan_sha256"] = plan_digest(plan)
    if field != "planned_completions":
        with pytest.raises(ValueError):
            evaluate(plan, outcomes, manifest)


def test_manifest_rebinding_and_unknown_plan_fields_rejected():
    manifest, plan, outcomes = fixture()
    plan["heldout_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        evaluate(plan, outcomes, manifest)
    _, plan, outcomes = fixture()
    plan["unexpected"] = True
    with pytest.raises(ValueError):
        evaluate(plan, outcomes, manifest)


def test_plan_duplicate_and_boolean_budget_rejected():
    manifest, plan, outcomes = fixture()
    plan["problems"].append(copy.deepcopy(plan["problems"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(plan, outcomes, manifest)
    with pytest.raises(ValueError):
        build_plan(manifest, {"a": True, "b": 3})
    with pytest.raises(ValueError):
        build_plan(manifest, {"a": 1})
