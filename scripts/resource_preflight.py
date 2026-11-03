"""Read-only resource planning; GB means decimal 1,000,000,000 bytes."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

GB = 1_000_000_000
# Disk shares: doubled by the user on 2026-10-27, project share 40 GB from 2026-10-29.
PROJECT_SHARE_BYTES = 40 * GB
SHARED_MODEL_CACHE_SHARE_BYTES = 80 * GB
ROOT = Path(__file__).resolve().parents[1]


def project_bytes(root):
    """Count all project files, including development caches; never follow symlinks."""

    def unreadable(error):
        raise error

    size, seen = 0, set()
    for directory, _subdirs, files in os.walk(root, followlinks=False, onerror=unreadable):
        for filename in files:
            path = Path(directory) / filename
            if path.is_symlink():
                continue
            stat = path.stat()
            inode = (stat.st_dev, stat.st_ino)
            if inode not in seen:
                size += stat.st_size
                seen.add(inode)
    return size


def check_limits(
    *,
    project_size,
    planned_project_bytes,
    download_bytes,
    free_bytes,
    checkpoints,
    service_memory_bytes,
    model_cache,
):
    for value in (
        project_size,
        planned_project_bytes,
        download_bytes,
        free_bytes,
        checkpoints,
        service_memory_bytes,
    ):
        if type(value) is not int or value < 0:
            raise ValueError("resource declarations must be nonnegative integer bytes/counts")
    errors = []
    if project_size + planned_project_bytes > PROJECT_SHARE_BYTES:
        errors.append(f"project allocation exceeds {PROJECT_SHARE_BYTES // GB} GB")
    if checkpoints > 2:
        errors.append("retain at most 2 checkpoints; remove only this project's old checkpoints")
    minimum = planned_project_bytes + download_bytes + (50 * GB if download_bytes > 5 * GB else 0)
    if free_bytes < minimum:
        errors.append(
            "download + project files require free space >= size "
            "(and 50 GB reserve when download size > 5 GB)"
        )
    if service_memory_bytes > 16 * GB:
        errors.append("project resident service memory exceeds 16 GB")
    expected = Path.home() / ".cache/huggingface"
    if Path(model_cache).expanduser().absolute() != expected.absolute():
        errors.append(f"model cache must be shared: {expected}")
    return errors


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def effective_model_cache():
    """Apply HF_HOME > XDG_CACHE_HOME/huggingface > ~/.cache/huggingface.

    Explicit empty values are invalid here, rather than silently treated as unset.
    An unused XDG_CACHE_HOME does not override an explicit HF_HOME.
    """
    if "HF_HOME" in os.environ:
        value = os.environ["HF_HOME"]
        if not value:
            raise ValueError("HF_HOME must not be empty; unset it to use cache defaults")
    else:
        base = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        if not base:
            raise ValueError("XDG_CACHE_HOME must not be empty when HF_HOME is unset")
        value = os.path.join(base, "huggingface")
    return Path(os.path.expandvars(os.path.expanduser(value))).absolute()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planned-project-bytes",
        type=nonnegative,
        required=True,
        help="all additional project files, including planned checkpoints",
    )
    parser.add_argument("--download-bytes", type=nonnegative, required=True)
    parser.add_argument(
        "--checkpoints",
        type=nonnegative,
        required=True,
        help="total retained checkpoints after the task",
    )
    parser.add_argument(
        "--service-memory-bytes",
        type=nonnegative,
        required=True,
        help="aggregate resident services' planned peak memory",
    )
    args = parser.parse_args()
    try:
        cache = effective_model_cache()
        shared_cache = Path.home() / ".cache/huggingface"
        # No mkdir: inspect the effective cache's filesystem, never an assumed default.
        disk = cache
        while not disk.exists():
            disk = disk.parent
        free = shutil.disk_usage(disk).free
        size = project_bytes(ROOT)
        errors = check_limits(
            project_size=size,
            planned_project_bytes=args.planned_project_bytes,
            download_bytes=args.download_bytes,
            free_bytes=free,
            checkpoints=args.checkpoints,
            service_memory_bytes=args.service_memory_bytes,
            model_cache=cache,
        )
        for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
            if key in os.environ and Path(os.environ[key]).expanduser() != shared_cache / "hub":
                errors.append(f"{key} must use {shared_cache / 'hub'}")
        if args.planned_project_bytes > shutil.disk_usage(ROOT).free:
            errors.append("insufficient free space for planned project files")
        print(
            json.dumps(
                {
                    "ok": not errors,
                    "units": "decimal GB = 1,000,000,000 bytes",
                    "project_bytes": size,
                    "free_bytes": free,
                    "model_cache": str(cache),
                    "errors": errors,
                },
                indent=2,
            )
        )
        return 1 if errors else 0
    except (OSError, ValueError) as exc:
        print(f"Resource preflight failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
