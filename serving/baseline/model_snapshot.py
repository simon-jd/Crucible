"""Prepare only the reviewed Qwen3-8B snapshot; no model/native imports."""

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import signal
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from scripts.resource_preflight import (
    PROJECT_SHARE_BYTES,
    SHARED_MODEL_CACHE_SHARE_BYTES,
    project_bytes,
)

ROOT = Path(__file__).resolve().parents[2]
REPO = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
PLAN_SHA256 = "067ca7df8c06e738b38db2e73634e4d1a4ac411c9ebf59cb10db68e557d06a37"
TOTAL_BYTES = 16_397_459_696
OFFICIAL_HOSTS = {
    "huggingface.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.hf.co",
    "cdn-lfs-eu-1.hf.co",
    "cas-bridge.xethub.hf.co",
    "us.aws.cdn.hf.co",
}


def load_plan(path):
    with Path(path).open("rb") as source:
        data = source.read((64 << 10) + 1)
    if len(data) > 64 << 10:
        raise ValueError("pinned model plan exceeds metadata bound")
    if hashlib.sha256(data).hexdigest() != PLAN_SHA256:
        raise ValueError("model plan differs from reviewed pinned bytes")
    plan = json.loads(data)
    if plan["repo"] != REPO or plan["revision"] != REVISION or plan["total_bytes"] != TOTAL_BYTES:
        raise ValueError("invalid pinned model identity")
    return plan


def cache_paths(cache, item):
    name, blob = item["name"], item["blob"]
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("unsafe model filename")
    if len(blob) not in (40, 64) or any(c not in "0123456789abcdef" for c in blob):
        raise ValueError("unsafe model blob identity")
    hub = Path(cache).resolve() / "hub"
    storage = hub / "models--Qwen--Qwen3-8B"
    return (
        storage / "blobs" / blob,
        storage / "snapshots" / REVISION / name,
        hub / ".locks" / "models--Qwen--Qwen3-8B" / (blob + ".lock"),
    )


def _mkdir(directory, cache):
    root = Path(cache).resolve()
    for parent in [root, *reversed(directory.relative_to(root).parents), directory]:
        # Relative parents above are resolved explicitly below, not against cwd.
        candidate = parent if parent.is_absolute() else root / parent
        if candidate.is_symlink():
            raise ValueError("model cache directory is a symlink")
        candidate.mkdir(exist_ok=True)


@contextlib.contextmanager
def blob_lock(path, timeout=300):
    """Same Linux flock operation/inode/path as HF's installed UnixFileLock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o644)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("blob lock must be a regular file")
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("shared HF blob lock deadline exceeded") from None
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        # Never unlink a shared lock: another participant may already hold its inode.


def verify_file(path, item, deadline=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"]:
        raise ValueError("model file type/size mismatch")
    actual = hashlib.sha256()
    upstream = hashlib.sha256() if item["algorithm"] == "sha256" else hashlib.sha1()
    if item["algorithm"] == "git-sha1":
        upstream.update(f"blob {item['bytes']}\0".encode())
    elif item["algorithm"] != "sha256":
        raise ValueError("unsupported model hash")
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("model verification deadline exceeded")
            actual.update(block)
            upstream.update(block)
    if upstream.hexdigest() != item["upstream_hash"]:
        raise ValueError("model upstream hash mismatch")
    return {
        "bytes": path.stat().st_size,
        "sha256": actual.hexdigest(),
        "algorithm": item["algorithm"],
        "upstream_hash": upstream.hexdigest(),
    }


def canonical_bytes(root, rescans=None):
    """Restart the entire canonical walk only for a disappearing entry."""
    for attempt in range(1, 4):
        try:
            return project_bytes(root)
        except FileNotFoundError as error:
            if rescans is not None:
                rescans.append(
                    {
                        "root": str(root),
                        "attempt": attempt,
                        "error": repr(error),
                        "message": str(error),
                        "errno": error.errno,
                        "filename": error.filename,
                        "filename2": error.filename2,
                    }
                )
            if attempt == 3:
                raise
    raise AssertionError("unreachable canonical scan state")


def resource_sample(cache, remaining, full_download=TOTAL_BYTES):
    cache = Path(cache)
    existing = cache
    while not existing.exists():
        existing = existing.parent
    rescans = []
    shared = canonical_bytes(cache, rescans) if cache.exists() else 0
    project = canonical_bytes(ROOT, rescans)
    free = shutil.disk_usage(existing).free
    reserve = 50_000_000_000 if full_download > 5_000_000_000 else 0
    sample = {
        "monotonic": time.monotonic(),
        "project_bytes": project,
        "shared_cache_bytes": shared,
        "remaining_download_bytes": remaining,
        "free_bytes": free,
        "canonical_rescans": rescans,
    }
    if (
        project + 600_000_000 + 50_000_000 > PROJECT_SHARE_BYTES
        or shared + remaining > SHARED_MODEL_CACHE_SHARE_BYTES
        or free < reserve + full_download
    ):
        raise RuntimeError("model resource preflight/high-water stop: " + json.dumps(sample))
    return sample


def _open_download(url, deadline, test_origin):
    from serving.baseline.deadline_http import open_response

    def allowed(parsed):
        origin = (parsed.scheme, parsed.hostname, parsed.port)
        return (
            origin == test_origin
            if test_origin is not None
            else parsed.scheme == "https"
            and parsed.hostname in OFFICIAL_HOSTS
            and parsed.port in (None, 443)
        )

    return open_response(url, deadline, allowed)


def _download(item, partial, *, deadline, test_url, observe, progress):
    url = item["url"]
    test_origin = None
    if test_url is not None:
        parsed = urllib.parse.urlparse(test_url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
            raise ValueError("test URL must be literal loopback HTTP")
        url = test_url
        test_origin = (parsed.scheme, parsed.hostname, parsed.port)
    elif not url.startswith(f"https://huggingface.co/{REPO}/resolve/{REVISION}/"):
        raise ValueError("model URL is not the pinned repository/revision")
    count, last_sample = 0, 0.0
    with _open_download(url, deadline, test_origin) as response:
        if (
            response.status != 200
            or response.headers.get("Content-Length") != str(item["bytes"])
            or response.headers.get("Content-Encoding", "identity") != "identity"
        ):
            raise ValueError("model response status/size/encoding mismatch")
        parsed = urllib.parse.urlparse(response.geturl())
        response_identity = {
            "scheme": parsed.scheme,
            "host": parsed.hostname,
            "path": parsed.path,
            "query_omitted": bool(parsed.query),
        }
        with partial.open("xb") as destination:
            owned = os.fstat(destination.fileno())
            progress["partial_identity"] = {"device": owned.st_dev, "inode": owned.st_ino}
            while count < item["bytes"]:
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError("model download deadline exceeded")
                if now - last_sample >= 1:
                    observe(item["bytes"] - count)
                    last_sample = now
                block = response.read1(min(1 << 20, item["bytes"] - count))
                if not block:
                    raise ValueError("truncated model response")
                destination.write(block)
                count += len(block)
                progress["downloaded_bytes"] = count
            destination.flush()
            os.fsync(destination.fileno())
    observe(0)
    if time.monotonic() >= deadline:
        raise TimeoutError("model transfer deadline exceeded")
    return response_identity


def _ensure_file(
    item, cache, *, download=False, test_url=None, observe=None, deadline=None, progress=None
):
    """Verify/reuse one planned blob, or publish one bounded owned partial."""
    cache = Path(cache).resolve()
    if test_url is None and cache != (Path.home() / ".cache/huggingface").resolve():
        raise ValueError("production model cache must be the shared HF cache")
    blob, pointer, lock = cache_paths(cache, item)
    deadline = deadline or time.monotonic() + 900
    progress = progress if progress is not None else {}
    observe = observe or (lambda remaining: resource_sample(cache, remaining, item["bytes"]))
    observe(0 if blob.exists() else item["bytes"])
    if not download and (not blob.exists() or not os.path.lexists(pointer)):
        raise FileNotFoundError("pinned model is incomplete; download mode is not enabled")
    for directory in (blob.parent, pointer.parent, lock.parent):
        _mkdir(directory, cache)
    with blob_lock(lock, timeout=min(300, max(0, deadline - time.monotonic()))):
        if os.path.lexists(pointer) and (not pointer.is_symlink() or pointer.resolve() != blob):
            raise ValueError("existing snapshot entry differs; refusing overwrite")
        if os.path.lexists(blob):
            proof = verify_file(blob, item, deadline)
            result = {**proof, "downloaded_bytes": 0, "reused": True}
        else:
            if not download:
                raise FileNotFoundError("pinned model blob missing")
            partials = list(blob.parent.glob(f".{blob.name}.task12-*.incomplete"))
            if os.path.lexists(str(blob) + ".incomplete") or partials:
                raise ValueError("existing partial must be reviewed; refusing overwrite/retry")
            partial = blob.with_name(f".{blob.name}.task12-{uuid.uuid4().hex}.incomplete")
            progress.update(partial=str(partial), downloaded_bytes=0)
            response = _download(
                item,
                partial,
                deadline=deadline,
                test_url=test_url,
                observe=observe,
                progress=progress,
            )
            proof = verify_file(partial, item, deadline)
            if time.monotonic() >= deadline:
                raise TimeoutError("model publication deadline exceeded")
            # Atomic no-clobber publication on one filesystem. Unlike rename(),
            # link() cannot overwrite a competing blob; unlink only our own partial.
            os.link(partial, blob)
            partial.unlink()
            result = {
                **proof,
                "downloaded_bytes": item["bytes"],
                "reused": False,
                "response": response,
                "publication": "no-clobber hardlink then owned unlink",
            }
        if time.monotonic() >= deadline:
            raise TimeoutError("model snapshot publication deadline exceeded")
        if not os.path.lexists(pointer):
            pointer.symlink_to(os.path.relpath(blob, pointer.parent))
        if pointer.resolve() != blob:
            raise ValueError("snapshot publication identity mismatch")
        if time.monotonic() >= deadline:
            raise TimeoutError("model snapshot completion deadline exceeded")
        return {
            **result,
            "name": item["name"],
            "blob_path": str(blob),
            "snapshot_path": str(pointer),
        }


def ensure_file(
    item, cache, *, download=False, test_url=None, observe=None, deadline=None, progress=None
):
    from serving.baseline.deadline_http import absolute_deadline

    deadline = deadline if deadline is not None else time.monotonic() + 900
    with absolute_deadline(deadline):
        return _ensure_file(
            item,
            cache,
            download=download,
            test_url=test_url,
            observe=observe,
            deadline=deadline,
            progress=progress,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("model-files.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--download",
        action="store_true",
        help="fetch missing pinned public model files (up to 16.4 GB)",
    )
    args = parser.parse_args()

    def terminated(signum, frame):
        raise TimeoutError("model preparation terminated")

    signal.signal(signal.SIGTERM, terminated)
    plan = load_plan(args.plan)
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("model evidence must be fresh and project-local")
    cache = Path.home() / ".cache/huggingface"
    remaining = sum(x["bytes"] for x in plan["files"] if not cache_paths(cache, x)[0].exists())
    report = {
        "schema": 1,
        "repo": REPO,
        "revision": REVISION,
        "plan_sha256": PLAN_SHA256,
        "runtime_lock_sha256": plan["runtime_lock_sha256"],
        "preparer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "expected_total_bytes": TOTAL_BYTES,
        "status": "running",
        "files": [],
        "samples": [],
        "pending": None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 3600
    try:
        for item in plan["files"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("whole model preparation deadline exceeded")
            absent = not cache_paths(cache, item)[0].exists()
            after_current = remaining - (item["bytes"] if absent else 0)
            report["pending"] = {"name": item["name"]}

            def observe(current_remaining, after_current=after_current):
                report["samples"].append(
                    resource_sample(
                        cache,
                        after_current + current_remaining,
                        TOTAL_BYTES if args.download else 0,
                    )
                )
                output.write_text(json.dumps(report, indent=2) + "\n")

            result = ensure_file(
                item,
                cache,
                download=args.download,
                observe=observe,
                deadline=min(deadline, time.monotonic() + 900),
                progress=report["pending"],
            )
            report["files"].append(result)
            remaining = after_current
        if time.monotonic() >= deadline:
            raise TimeoutError("whole model preparation completion deadline exceeded")
        report["pending"] = None
        report["status"] = "verified"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
