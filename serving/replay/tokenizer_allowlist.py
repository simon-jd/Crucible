"""Build a reproducible input-ID allowlist from a trusted pinned local HF snapshot.

Stdlib only; never downloads, imports model code, or writes into the snapshot.
This validates vocabulary metadata, not runtime/model compatibility.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

LIMIT = 32 * 1024 * 1024


def build(snapshot, model, revision, words=False):
    snapshot = Path(snapshot)
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or snapshot.name != revision:
        raise ValueError("snapshot basename must equal immutable revision")
    if not model or snapshot.parent.name != "snapshots":
        raise ValueError("expected named model and HF snapshots/revision path")
    if snapshot.parent.parent.name != "models--" + model.replace("/", "--"):
        raise ValueError("snapshot model path mismatch")
    docs, hashes = {}, {}
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json"):
        with (snapshot / name).open("rb") as source:
            data = source.read(LIMIT + 1)
        if len(data) > LIMIT:
            raise ValueError("metadata exceeds 32MiB bound")
        docs[name] = json.loads(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    size = docs["config.json"]["vocab_size"]
    if type(size) is not int or not 0 < size <= 10_000_000:
        raise ValueError("invalid model vocab_size")
    vocab = docs["tokenizer.json"]["model"]["vocab"]
    if not isinstance(vocab, dict):
        raise TypeError("only explicit vocabulary dictionaries supported")
    ids = list(vocab.values())
    if any(type(i) is not int or i < 0 or i >= size for i in ids):
        raise ValueError("invalid vocabulary ID")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate vocabulary IDs")
    excluded = set()
    for token in docs["tokenizer.json"].get("added_tokens", []):
        value = token["id"]
        if type(value) is not int or value < 0:
            raise ValueError("invalid added token ID")
        excluded.add(value)
    for doc in (docs["config.json"], docs["tokenizer_config.json"]):
        for key, value in doc.items():
            if key.endswith(("_token_id", "_token_ids")):
                values = value if isinstance(value, list) else [value]
                for item in values:
                    if item is None:
                        continue
                    if type(item) is not int or item < 0:
                        raise ValueError("invalid special ID")
                    excluded.add(item)
    for key, value in docs["tokenizer_config.json"].items():
        if key.endswith(("_token", "_tokens")):
            values = value if isinstance(value, list) else [value]
            for item in values:
                content = item.get("content") if isinstance(item, dict) else item
                if isinstance(content, str) and content in vocab:
                    excluded.add(vocab[content])
    for key in docs["tokenizer_config.json"].get("added_tokens_decoder", {}):
        if not key.isdigit():
            raise ValueError("invalid added-token decoder ID")
        excluded.add(int(key))
    texts = None
    if words:
        # Byte-level BPE writes a leading space as "Ġ". A space plus 3-10 lowercase
        # ASCII letters is one pre-token, so concatenated words keep one token each.
        ids = [vocab[k] for k in vocab if re.fullmatch("Ġ[a-z]{3,10}", k) and vocab[k] not in excluded]
        ids = sorted(ids)[:4096]
        names = {v: k for k, v in vocab.items()}
        texts = [" " + names[i][1:] for i in ids]
    else:
        ids = sorted(set(ids) - excluded)[:4096]
    if len(ids) < 2:
        raise ValueError("insufficient non-special IDs")
    result = {
        "Model": model,
        "Revision": revision,
        "TokenizerSHA256": hashes["tokenizer.json"],
        "file_sha256": hashes,
        "token_ids": ids,
        "policy": "ascending first <=4096 vocab IDs excluding added/special IDs",
    }
    if words:
        result["token_texts"] = texts
        result["policy"] = "ascending first <=4096 non-special space+lowercase words (3-10 letters)"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--words", action="store_true", help="single-token words + token_texts")
    args = parser.parse_args()
    print(json.dumps(build(args.snapshot, args.model, args.revision, args.words), sort_keys=True))


if __name__ == "__main__":
    main()
