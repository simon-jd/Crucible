"""Build the pinned GSM8K / MATH-500 train and held-out sets for rl.local.grpo_run.

Inputs are the pinned files named in rl/configs/qwen25_05b_math_code.yaml; their
hashes are checked before use. Held-out rows come from GSM8K test and MATH-500;
training rows from GSM8K train, minus duplicates and any question that also
appears in a held-out pool. MATH-500 candidates are limited to problems whose
reference answer the bounded verifier can grade (reference == reference), so a
wrong grade is never caused by an unparseable reference; that restriction is a
disclosed selection. Prints an identity summary; writes the full set to --output.
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rl.eval import math_data as md
from rl.eval.heldout import digest, select_heldout
from rl.eval.parquet_lite import read_string_columns

EXPECTED = {
    "gsm8k/train-00000-of-00001.parquet": ("sha256", "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"),
    "gsm8k/test-00000-of-00001.parquet": ("sha256", "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"),
    "math500/test.jsonl": ("git_blob_sha1", "2376b9a194b46c0790e197c91b7249e5f88ac09b"),
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-count", type=int, default=1024)
    p.add_argument("--gsm8k-eval", type=int, default=200)
    p.add_argument("--math500-eval", type=int, default=100)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args(argv)
    hashes = {}
    for name, (kind, want) in EXPECTED.items():
        data = (args.root / name).read_bytes()
        got = hashlib.sha256(data).hexdigest() if kind == "sha256" else md.git_blob_sha1(data)
        if got != want:
            raise SystemExit(f"{name}: {kind} {got} != pinned {want}")
        hashes[name] = {kind: got, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}

    train, train_dupes = md.dedupe(md.gsm8k_records(read_string_columns(args.root / "gsm8k/train-00000-of-00001.parquet"), "train"))
    test, test_dupes = md.dedupe(md.gsm8k_records(read_string_columns(args.root / "gsm8k/test-00000-of-00001.parquet"), "test"))
    math, math_dupes = md.dedupe(md.math500_records((args.root / "math500/test.jsonl").read_text().splitlines()))

    from rl.verifiers.math import verify_math

    with ThreadPoolExecutor(16) as pool:
        ok = list(pool.map(lambda r: verify_math("\\boxed{" + r["answer"] + "}", r["answer"]).correct, math))
    gradable = [r for r, good in zip(math, ok, strict=True) if good]

    train, overlap = md.drop_overlap(train, test + gradable)
    strip = [{k: r[k] for k in ("id", "question", "source")} for r in train]
    manifests = {}
    for name, pool_rows, count in (("gsm8k", test, args.gsm8k_eval), ("math500", gradable, args.math500_eval)):
        cand = [{k: r[k] for k in ("id", "question", "source")} for r in pool_rows]
        manifests[name] = select_heldout(strip, cand, count=count, seed=args.seed, synthetic_fixture=False)
    by_id = {r["id"]: r for r in test + gradable}
    body = {
        "train": md.select(train, args.train_count, seed=args.seed),
        "gsm8k_eval": [by_id[r["id"]] for r in manifests["gsm8k"]["selected"]],
        "math500_eval": [by_id[r["id"]] for r in manifests["math500"]["selected"]],
    }
    full = {**body, "sha256": digest(body)}
    args.output.write_text(json.dumps(full) + "\n")
    print(
        json.dumps(
            {
                "inputs": hashes,
                "rows": {"gsm8k_train": len(train) + len(overlap), "gsm8k_test": len(test), "math500": len(math)},
                "dropped": {"train_duplicates": train_dupes, "test_duplicates": test_dupes, "math500_duplicates": math_dupes,
                            "train_overlapping_heldout": overlap, "math500_ungradable_reference": len(math) - len(gradable)},
                "heldout_manifest_sha256": {k: v["sha256"] for k, v in manifests.items()},
                "selected": {k: len(v) for k, v in body.items()},
                "data_sha256": full["sha256"],
                "seed": args.seed,
            },
            indent=1,
        )
    )  # fmt: skip


if __name__ == "__main__":
    main()
