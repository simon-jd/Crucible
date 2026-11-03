"""Synthetic rows; checks T15 full-size feature batches and the held-out disjointness check."""

import pytest

from spec.t15_data import assert_disjoint, plan_batches, prompt_key


def _regen(n):
    return [
        {
            "id": f"t{i}",
            "conversations": [
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "content": f"answer {i}"},
            ],
        }
        for i in range(n)
    ]


def test_plan_batches_has_equal_sizes_and_drops_the_tail():
    batches = plan_batches(_regen(10), batch_size=3)
    assert [len(b) for b in batches] == [3, 3, 3]
    assert [r["id"] for r in batches[0]] == ["t0", "t1", "t2"]
    assert batches[2][-1]["id"] == "t8"


def test_plan_batches_rejects_too_few_rows():
    with pytest.raises(ValueError):
        plan_batches(_regen(2), batch_size=3)


def test_assert_disjoint_accepts_fresh_eval_prompts():
    eval_rows = [{"prompt_id": "e0", "prompt": "a fresh question"}]
    assert assert_disjoint(_regen(5), eval_rows) == 5


def test_assert_disjoint_catches_normalized_duplicates():
    eval_rows = [{"prompt_id": "e0", "prompt": "QUESTION   3"}]
    assert prompt_key("question 3") == prompt_key(eval_rows[0]["prompt"])
    with pytest.raises(ValueError, match="e0"):
        assert_disjoint(_regen(5), eval_rows)
