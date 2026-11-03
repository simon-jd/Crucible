"""Task 13 held-out sets: GSM8K (Task 11's held-out test subset) and MMLU-Pro test.

Inputs are pinned: GSM8K from the rl.local.prepare_data output, MMLU-Pro test
parquet at TIGER-Lab/MMLU-Pro revision b189ec765aa7ed75c8acfea42df31fdae71f97be
(SHA256 checked). MMLU-Pro rows are a seeded order-independent subset.
"""

import argparse
import hashlib
import json
from pathlib import Path

from rl.eval import math_data as md
from rl.eval.parquet_lite import read_columns

MMLU_PRO_SHA256 = "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8"
MMLU_PRO_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--t11-data", type=Path, required=True)
    p.add_argument("--mmlu-pro", type=Path, required=True)
    p.add_argument("--count", type=int, default=300)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    raw = args.mmlu_pro.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MMLU_PRO_SHA256:
        raise SystemExit("MMLU-Pro parquet hash mismatch")
    cols = read_columns(args.mmlu_pro)
    rows = [
        {"id": f"mmlu-pro/test/{qid}", "question": q, "source": "TIGER-Lab/MMLU-Pro:test", "options": o, "answer": a, "category": c}
        for qid, q, o, a, c in zip(cols["question_id"], cols["question"], cols["options"], cols["answer"], cols["category"], strict=True)
    ]
    t11 = json.loads(args.t11_data.read_text())
    out = {
        "gsm8k": t11["gsm8k_eval"],
        "mmlu_pro": md.select(rows, args.count, seed=args.seed),
        "provenance": {
            "gsm8k": {"revision": "740312add88f781978c0658806c59bc2815b9866", "t11_data_sha256": t11["sha256"]},
            "mmlu_pro": {"revision": MMLU_PRO_REVISION, "sha256": MMLU_PRO_SHA256, "rows": len(rows), "seed": args.seed},
        },
    }
    args.output.write_text(json.dumps(out) + "\n")
    print(json.dumps({"counts": {"gsm8k": len(out["gsm8k"]), "mmlu_pro": len(out["mmlu_pro"])}, "provenance": out["provenance"]}))


if __name__ == "__main__":
    main()
