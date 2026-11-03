import hashlib
import json

import pytest

from rl.eval import math_data as md


def test_gsm8k_records_use_final_answer_without_commas():
    cols = {"question": ["Q1?", "Q2?"], "answer": ["work <<1+1=2>>\n#### 1,000", "x\n#### -3"]}
    rows = md.gsm8k_records(cols, "test")
    assert rows == [
        {"id": "gsm8k/test/00000", "question": "Q1?", "source": "openai/gsm8k:main/test", "answer": "1000"},
        {"id": "gsm8k/test/00001", "question": "Q2?", "source": "openai/gsm8k:main/test", "answer": "-3"},
    ]
    with pytest.raises(ValueError):
        md.gsm8k_records({"question": ["Q"], "answer": ["no marker"]}, "test")


def test_math500_records_keep_latex_answer_and_unique_id():
    line = json.dumps({"problem": "P", "answer": "\\frac{1}{2}", "unique_id": "test/algebra/1.json"})
    assert md.math500_records([line]) == [
        {"id": "math500/test/algebra/1.json", "question": "P", "source": "HuggingFaceH4/MATH-500:test", "answer": "\\frac{1}{2}"}
    ]


def test_dedupe_keeps_first_question_and_reports_dropped_ids():
    rows = [
        {"id": "a", "question": "Same  ", "source": "s", "answer": "1"},
        {"id": "b", "question": "Same", "source": "s", "answer": "1"},
        {"id": "c", "question": "Other", "source": "s", "answer": "2"},
    ]
    kept, dropped = md.dedupe(rows)
    assert [r["id"] for r in kept] == ["a", "c"] and dropped == ["b"]


def test_drop_overlap_removes_train_rows_whose_question_is_heldout():
    train = [{"id": "t1", "question": "Shared", "source": "s"}, {"id": "t2", "question": "Only", "source": "s"}]
    heldout = [{"id": "h1", "question": "Shared ", "source": "s"}]
    kept, dropped = md.drop_overlap(train, heldout)
    assert [r["id"] for r in kept] == ["t2"] and dropped == ["t1"]


def test_select_prompts_is_deterministic_and_seed_dependent():
    rows = [{"id": f"r{i}", "question": f"q{i}", "source": "s", "answer": "1"} for i in range(50)]
    a = md.select(rows, 10, seed=17)
    assert a == md.select(list(reversed(rows)), 10, seed=17)
    assert a != md.select(rows, 10, seed=18) and len({r["id"] for r in a}) == 10


def test_git_blob_sha1_matches_git_object_format():
    data = b"hello\n"
    assert md.git_blob_sha1(data) == hashlib.sha1(b"blob 6\0hello\n").hexdigest()
