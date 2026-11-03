"""Exact project-data exclusion and deterministic selection; never fetches data.

Question normalization is Unicode NFC, CRLF/CR -> LF, then outer strip only.
Internal whitespace/indentation survives. Answers and source labels do not affect
question hashes. This is neither semantic matching nor pretraining exclusion.
"""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

NORMALIZATION = "NFC;CRLF/CR->LF;outer-strip;preserve-internal-whitespace"
ALGORITHM = "sha256-canonical-seed-id-question-source-v1"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def keys(value, required, optional=()):
    if (
        type(value) is not dict
        or set(value) - set(required) - set(optional)
        or set(required) - set(value)
    ):
        raise ValueError(f"expected fields {sorted(required)}, optional {sorted(optional)}")


def text(value):
    if type(value) is not str or not value.strip():
        raise ValueError("expected nonempty string")
    return value


def sha256(value):
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected lowercase SHA256")
    return value


def integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"expected integer >= {minimum}")
    return value


def question_hash(question):
    normalized = unicodedata.normalize(
        "NFC", text(question).replace("\r\n", "\n").replace("\r", "\n")
    ).strip()
    if not normalized:
        raise ValueError("empty normalized question")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def identities(records):
    if type(records) is not list or not records:
        raise ValueError("records must be a nonempty list")
    result = []
    for record in records:
        keys(record, {"id", "question", "source"}, {"answer"})
        if "answer" in record and type(record["answer"]) is not str:
            raise ValueError("answer must be a string when supplied")
        result.append(
            {
                "id": text(record["id"]),
                "question_sha256": question_hash(record["question"]),
                "source": text(record["source"]),
                "record_sha256": digest(record),
            }
        )
    return sorted(result, key=lambda row: row["id"])


def check_pool(rows):
    if type(rows) is not list or not rows:
        raise ValueError("identity pool must be nonempty")
    ids, hashes = set(), set()
    for row in rows:
        keys(row, {"id", "question_sha256", "source", "record_sha256"})
        identity, question = text(row["id"]), sha256(row["question_sha256"])
        text(row["source"])
        sha256(row["record_sha256"])
        if identity in ids or question in hashes:
            raise ValueError("duplicate identity/question in full pool")
        ids.add(identity)
        hashes.add(question)
    if rows != sorted(rows, key=lambda row: row["id"]):
        raise ValueError("pool must be canonically sorted by ID")
    return ids, hashes


def _manifest(train, candidates, *, count, seed, synthetic_fixture):
    train_ids, train_hashes = check_pool(train)
    candidate_ids, candidate_hashes = check_pool(candidates)
    if train_ids & candidate_ids or train_hashes & candidate_hashes:
        raise ValueError("train/heldout overlap in full supplied pools")
    integer(count, 1)
    integer(seed)
    if count > len(candidates) or type(synthetic_fixture) is not bool:
        raise ValueError("invalid count or synthetic_fixture flag")
    ranked = sorted(
        candidates,
        key=lambda row: (
            digest(
                {"seed": seed, **{key: row[key] for key in ("id", "question_sha256", "source")}}
            ),
            row["id"],
        ),
    )
    body = {
        "schema_version": 1,
        "synthetic_fixture": synthetic_fixture,
        "normalization": NORMALIZATION,
        "algorithm": ALGORITHM,
        "seed": seed,
        "count": count,
        "train": train,
        "candidates": candidates,
        "selected": ranked[:count],
    }
    return {**body, "sha256": digest(body)}


def select_heldout(train, candidates, *, count, seed, synthetic_fixture):
    """Reject all full-pool duplicates/overlaps before deterministic selection."""
    return _manifest(
        identities(train),
        identities(candidates),
        count=count,
        seed=seed,
        synthetic_fixture=synthetic_fixture,
    )


def validate_manifest(manifest):
    """Check self-consistency, not acquisition authenticity; use verify for content."""
    keys(
        manifest,
        {
            "schema_version",
            "synthetic_fixture",
            "normalization",
            "algorithm",
            "seed",
            "count",
            "train",
            "candidates",
            "selected",
            "sha256",
        },
    )
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported schema_version")
    expected = _manifest(
        manifest["train"],
        manifest["candidates"],
        count=manifest["count"],
        seed=manifest["seed"],
        synthetic_fixture=manifest["synthetic_fixture"],
    )
    if manifest != expected:
        raise ValueError("manifest digest/selection/contract mismatch")
    return manifest


def verify_manifest(manifest, train, candidates):
    validate_manifest(manifest)
    expected = select_heldout(
        train,
        candidates,
        count=manifest["count"],
        seed=manifest["seed"],
        synthetic_fixture=manifest["synthetic_fixture"],
    )
    if manifest != expected:
        raise ValueError("manifest does not bind supplied question content/provenance")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--verify", type=Path, help="verify a predeclared manifest against full inputs"
    )
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--synthetic-fixture", action="store_true")
    args = parser.parse_args(argv)
    try:
        train, candidates = [json.loads(path.read_text()) for path in (args.train, args.candidates)]
        if args.verify:
            if args.count is not None or args.seed is not None or args.synthetic_fixture:
                raise ValueError("verification takes its contract only from the manifest")
            result = verify_manifest(json.loads(args.verify.read_text()), train, candidates)
        else:
            result = select_heldout(
                train,
                candidates,
                count=args.count,
                seed=args.seed,
                synthetic_fixture=args.synthetic_fixture,
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"heldout: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
