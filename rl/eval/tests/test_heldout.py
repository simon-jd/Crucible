"""Selection uses made-up questions, never benchmark content."""

import copy

import pytest

from rl.eval.heldout import question_hash, select_heldout, validate_manifest, verify_manifest


def records():
    return (
        [{"id": "t", "question": "train", "source": "fixture/train"}],
        [{"id": str(i), "question": f"question {i}", "source": "fixture/test"} for i in range(5)],
    )


def test_heldout_split_disjoint_from_train():
    train, candidates = records()
    candidates[4]["question"] = " \r\ntrain\r\n "
    candidates[4]["answer"] = "changed answer"
    candidates[4]["source"] = "other/source"
    with pytest.raises(ValueError, match="overlap"):
        select_heldout(train, candidates, count=1, seed=17, synthetic_fixture=True)


def test_normalization_preserves_internal_indentation_and_ignores_answer():
    assert question_hash(" e\u0301\r\n x ") == question_hash("é\n x")
    assert question_hash("a\n x") != question_hash("a\nx")
    train, candidates = records()
    manifest = select_heldout(train, candidates, count=2, seed=17, synthetic_fixture=True)
    candidates[0]["answer"] = "irrelevant gold"
    with pytest.raises(ValueError, match="content/provenance"):
        verify_manifest(manifest, train, candidates)


def test_order_independence_and_tamper_rejection():
    train, candidates = records()
    manifest = select_heldout(train, candidates, count=2, seed=17, synthetic_fixture=True)
    assert manifest == select_heldout(
        train[::-1], candidates[::-1], count=2, seed=17, synthetic_fixture=True
    )
    for field in ["seed", "count", "selected", "sha256"]:
        altered = copy.deepcopy(manifest)
        altered[field] = [] if field == "selected" else 3 if field != "sha256" else "0" * 64
        with pytest.raises(ValueError):
            validate_manifest(altered)
    candidates[0]["question"] = "modified"
    with pytest.raises(ValueError):
        verify_manifest(manifest, train, candidates)


def test_full_record_binding_is_distinct_from_question_identity():
    train, candidates = records()
    manifest = select_heldout(train, candidates, count=2, seed=17, synthetic_fixture=True)
    candidates[0]["answer"] = "new gold"
    changed = select_heldout(train, candidates, count=2, seed=17, synthetic_fixture=True)
    assert [row["id"] for row in changed["selected"]] == [row["id"] for row in manifest["selected"]]
    assert (
        changed["candidates"][0]["question_sha256"] == manifest["candidates"][0]["question_sha256"]
    )
    assert changed["candidates"][0]["record_sha256"] != manifest["candidates"][0]["record_sha256"]
    assert changed["sha256"] != manifest["sha256"]


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "duplicate_id",
        "duplicate_question",
        "blank",
        "bool_count",
        "count_high",
        "negative_seed",
        "unknown",
    ],
)
def test_invalid_full_candidate_pool_rejected(kind):
    train, candidates = records()
    count, seed = 1, 17
    if kind == "empty":
        candidates = []
    elif kind == "duplicate_id":
        candidates[-1]["id"] = candidates[0]["id"]
    elif kind == "duplicate_question":
        candidates[-1]["question"] = candidates[0]["question"]
    elif kind == "blank":
        candidates[-1]["question"] = " \n"
    elif kind == "bool_count":
        count = True
    elif kind == "count_high":
        count = 6
    elif kind == "negative_seed":
        seed = -1
    else:
        candidates[0]["unknown"] = 1
    with pytest.raises(ValueError):
        select_heldout(train, candidates, count=count, seed=seed, synthetic_fixture=True)
