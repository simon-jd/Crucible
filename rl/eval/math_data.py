"""GSM8K / MATH-500 records in the rl.eval.heldout record shape.

Records are {id, question, source, answer}. Pools are de-duplicated by the same
normalized question hash heldout.py uses, and training rows that share a
question with a held-out row are dropped before any selection.
"""

import hashlib
import json

from rl.eval.heldout import digest, question_hash


def gsm8k_records(columns, split):
    rows = []
    for i, (question, answer) in enumerate(zip(columns["question"], columns["answer"], strict=True)):
        if "####" not in answer:
            raise ValueError(f"gsm8k row {i} has no final-answer marker")
        final = answer.rsplit("####", 1)[1].strip().replace(",", "")
        rows.append({"id": f"gsm8k/{split}/{i:05d}", "question": question, "source": f"openai/gsm8k:main/{split}", "answer": final})
    return rows


def math500_records(lines):
    rows = []
    for line in lines:
        if not line.strip():
            continue
        item = json.loads(line)
        rows.append(
            {"id": "math500/" + item["unique_id"], "question": item["problem"], "source": "HuggingFaceH4/MATH-500:test", "answer": item["answer"]}
        )
    return rows


def dedupe(rows):
    seen, kept, dropped = set(), [], []
    for row in rows:
        h = question_hash(row["question"])
        if h in seen:
            dropped.append(row["id"])
        else:
            seen.add(h)
            kept.append(row)
    return kept, dropped


def drop_overlap(train, heldout):
    held = {question_hash(r["question"]) for r in heldout}
    kept = [r for r in train if question_hash(r["question"]) not in held]
    return kept, [r["id"] for r in train if question_hash(r["question"]) in held]


def select(rows, count, seed):
    """Deterministic seeded subset, independent of input order (same rule as heldout.py)."""
    ranked = sorted(rows, key=lambda r: (digest({"seed": seed, "id": r["id"], "question_sha256": question_hash(r["question"])}), r["id"]))
    if count > len(ranked):
        raise ValueError("count exceeds pool")
    return ranked[:count]


def git_blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
