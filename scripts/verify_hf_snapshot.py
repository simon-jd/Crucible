"""Verify a plainly downloaded HF snapshot against its pinned tree listing.

LFS files are checked by sha256 (tree `lfs.oid`), others by git blob SHA1 (tree
`oid`). --preflight instead checks the shared-cache share and the free-space rule
of the resource budget for the bytes still missing, before any download.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from scripts.resource_preflight import GB, SHARED_MODEL_CACHE_SHARE_BYTES, project_bytes


def git_blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def file_digest(path, lfs):
    if not lfs:
        return git_blob_sha1(path.read_bytes())
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(entries, dest):
    """{path: problem} for every listed file that is missing, short or mismatched."""
    problems = {}
    for e in entries:
        path = dest / e["path"]
        lfs = e.get("lfs")
        want = lfs["oid"] if lfs else e["oid"]
        if not path.is_file():
            problems[e["path"]] = "missing"
        elif path.stat().st_size != e["size"]:
            problems[e["path"]] = f"size {path.stat().st_size} != {e['size']}"
        elif file_digest(path, bool(lfs)) != want:
            problems[e["path"]] = "digest mismatch"
    return problems


def preflight(entries, dest, cache, free):
    missing = sum(e["size"] for e in entries if not (dest / e["path"]).is_file())
    shared = project_bytes(cache) if cache.exists() else 0
    errors = []
    if shared + missing > SHARED_MODEL_CACHE_SHARE_BYTES:
        errors.append(f"shared cache {shared} + {missing} exceeds {SHARED_MODEL_CACHE_SHARE_BYTES // GB} GB")
    if free < missing + (50 * GB if missing > 5 * GB else 0):
        errors.append("free space below download + 50 GB reserve")
    return {"missing_bytes": missing, "shared_cache_bytes": shared, "free_bytes": free, "errors": errors}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tree", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--cache", type=Path, default=Path.home() / ".cache/huggingface")
    args = p.parse_args(argv)
    entries = [e for e in json.loads(args.tree.read_text()) if e["type"] == "file"]
    if args.preflight:
        existing = args.dest
        while not existing.exists():
            existing = existing.parent
        result = preflight(entries, args.dest, args.cache, shutil.disk_usage(existing).free)
        print(json.dumps(result))
        return 1 if result["errors"] else 0
    problems = verify(entries, args.dest)
    print(json.dumps({"files": len(entries), "bytes": sum(e["size"] for e in entries), "problems": problems}))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
