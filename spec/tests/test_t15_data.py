"""Synthetic rows; checks the T15 train/held-out prompt split."""

import pytest

from spec.t15_data import build_split, prompt_key, train_record


def _rows(prefix, n):
    return [{"prompt_id": f"{prefix}{i}", "prompt": f"{prefix} question number {i}"} for i in range(n)]


def test_prompt_key_normalizes_case_and_whitespace():
    assert prompt_key("Hello   World\n") == prompt_key("hello world")
    assert prompt_key("hello world") != prompt_key("hello worlds")


def test_split_is_deterministic_and_sized():
    train_rows, test_rows = _rows("t", 50), _rows("e", 30)
    a = build_split(train_rows, test_rows, n_train=10, n_eval=5, max_chars=1000)
    b = build_split(list(reversed(train_rows)), list(reversed(test_rows)), 10, 5, 1000)
    assert a == b
    assert len(a["train"]) == 10 and len(a["eval"]) == 5


def test_heldout_disjoint_from_whole_train_shard():
    train_rows = _rows("t", 20)
    # Every test prompt duplicates a train prompt (up to whitespace/case) except two.
    test_rows = [{"prompt_id": f"d{i}", "prompt": r["prompt"].upper()} for i, r in enumerate(train_rows)]
    test_rows += [{"prompt_id": "x0", "prompt": "fresh one"}, {"prompt_id": "x1", "prompt": "fresh two"}]
    split = build_split(train_rows, test_rows, n_train=3, n_eval=2, max_chars=1000)
    assert {r["prompt_id"] for r in split["eval"]} == {"x0", "x1"}
    with pytest.raises(ValueError):
        build_split(train_rows, test_rows, n_train=3, n_eval=3, max_chars=1000)


def test_split_drops_duplicates_and_long_prompts():
    rows = _rows("t", 5) + [{"prompt_id": "dup", "prompt": "T question number 0"}]
    rows.append({"prompt_id": "long", "prompt": "x" * 2000})
    split = build_split(rows, _rows("e", 3), n_train=5, n_eval=3, max_chars=1000)
    ids = [r["prompt_id"] for r in split["train"]]
    assert "long" not in ids and len({prompt_key(r["prompt"]) for r in split["train"]}) == 5


def test_train_record_is_specforge_conversation():
    record = train_record({"prompt_id": "a", "prompt": "hi"})
    assert record == {"id": "a", "conversations": [{"role": "user", "content": "hi"}]}
