"""T15 prompts: UltraChat 200k train_sft for draft training, test_sft for held-out serving.

Held-out prompts are disjoint (by normalized-text hash) from the whole train shard read,
not only from the selected training prompts. Selection is a deterministic sha256 order.
main() reads parquet, so it runs in .venv-specforge (pyarrow); the helpers are pure.
Phase 7: `batches` splits the regenerated conversations into equal feature batches after
re-checking that no held-out prompt appears in them.
"""

import argparse
import hashlib
import json
from pathlib import Path


def prompt_key(text):
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def _select(rows, n, max_chars, exclude):
    picked, seen = [], set(exclude)
    for row in sorted(rows, key=lambda r: (prompt_key(r["prompt"]), r["prompt_id"])):
        key = prompt_key(row["prompt"])
        if key in seen or not row["prompt"].strip() or len(row["prompt"]) > max_chars:
            continue
        seen.add(key)
        picked.append({"prompt_id": row["prompt_id"], "prompt": row["prompt"]})
        if len(picked) == n:
            return picked
    raise ValueError(f"only {len(picked)} eligible prompts, need {n}")


def build_split(train_rows, test_rows, n_train, n_eval, max_chars):
    train_keys = {prompt_key(r["prompt"]) for r in train_rows}
    return {
        "train": _select(train_rows, n_train, max_chars, ()),
        "eval": _select(test_rows, n_eval, max_chars, train_keys),
    }


def train_record(row):
    return {"id": row["prompt_id"], "conversations": [{"role": "user", "content": row["prompt"]}]}


def plan_batches(regen_rows, batch_size, n_batches=None):
    """Equal-size feature batches in file order; the tail that does not fill one is dropped.

    Every batch must hold the same number of samples: SpecForge's resume contract
    checks dataset_size, and each batch is one epoch of the same run (phase 7).
    """
    available = len(regen_rows) // batch_size
    n = available if n_batches is None else n_batches
    if n < 1 or n > available:
        raise ValueError(f"{len(regen_rows)} rows make {available} batches of {batch_size}, need {n}")
    return [regen_rows[i * batch_size : (i + 1) * batch_size] for i in range(n)]


def assert_disjoint(regen_rows, eval_rows):
    """Raise if any held-out prompt matches a training user turn by normalized-text hash."""
    train_keys = {
        prompt_key(turn["content"])
        for row in regen_rows
        for turn in row["conversations"]
        if turn["role"] == "user"
    }
    clash = [r["prompt_id"] for r in eval_rows if prompt_key(r["prompt"]) in train_keys]
    if clash:
        raise ValueError(f"held-out prompts found in training data: {clash}")
    return len(regen_rows)


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def batches_main(argv):
    """python -m spec.t15_data batches REGEN.jsonl EVAL.jsonl BATCH_SIZE N_BATCHES OUT_DIR"""
    regen_path, eval_path, batch_size, n_batches, out_dir = argv
    regen = _jsonl(regen_path)
    checked = assert_disjoint(regen, _jsonl(eval_path))
    batches = plan_batches(regen, int(batch_size), int(n_batches))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for k, rows in enumerate(batches):
        path = out / f"batch_{k:03d}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        outputs[path.name] = _sha256(path)
    manifest = {
        "regen": {regen_path: _sha256(regen_path)},
        "eval": {eval_path: _sha256(eval_path)},
        "regen_rows_checked_disjoint": checked,
        "batch_size": int(batch_size),
        "n_batches": len(batches),
        "batches": outputs,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "batches"}))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["prompt", "prompt_id"])
    return table.to_pylist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--test-parquet", required=True)
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--n-eval", type=int, required=True)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    split = build_split(
        _read(args.train_parquet), _read(args.test_parquet), args.n_train, args.n_eval, args.max_chars
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path, eval_path = args.out_dir / "train_prompts.jsonl", args.out_dir / "eval_prompts.jsonl"
    train_path.write_text("".join(json.dumps(train_record(r)) + "\n" for r in split["train"]))
    eval_path.write_text("".join(json.dumps(r) + "\n" for r in split["eval"]))
    manifest = {
        "inputs": {p: _sha256(p) for p in (args.train_parquet, args.test_parquet)},
        "n_train": args.n_train,
        "n_eval": args.n_eval,
        "max_chars": args.max_chars,
        "outputs": {str(p): _sha256(p) for p in (train_path, eval_path)},
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["batches"]:
        batches_main(sys.argv[2:])
    else:
        main()
