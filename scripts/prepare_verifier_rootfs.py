"""Prepare a small dedicated CPython rootfs from this trusted local interpreter.

No network, image daemon, host policy changes, candidate execution, or HOME copy.
Pin the interpreter SHA256, and retain the emitted full dependency manifest.
This prepares files only; it neither grants runtime privileges nor runs gVisor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAX_ROOTFS_BYTES = 200 * 1024**2


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan_files():
    executable = Path(sys.executable).resolve()
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    files = {"usr/bin/python3": executable}
    excluded = {
        "site-packages",
        "__pycache__",
        "test",
        "tests",
        "ensurepip",
        "idlelib",
        "tkinter",
        "turtledemo",
        "venv",
    }
    for directory, subdirs, filenames in os.walk(stdlib, followlinks=False):
        subdirs[:] = sorted(name for name in subdirs if name not in excluded)
        for name in sorted(filenames):
            source = Path(directory) / name
            if source.suffix in (".pyc", ".pyo") or source.name.startswith("_tkinter."):
                continue
            if not source.resolve().is_relative_to(stdlib) or not source.is_file():
                raise ValueError("stdlib_symlink_or_special_file")
            relative = source.relative_to(stdlib)
            files[f"usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/{relative}"] = (
                source.resolve()
            )
    # ldd is used only on trusted local CPython ELF files, never untrusted input.
    elfs = [executable] + [path for path in files.values() if ".so" in path.name]
    for path in elfs:
        result = subprocess.run(
            ["/usr/bin/ldd", str(path)], text=True, capture_output=True, check=True, timeout=5
        )
        if "not found" in result.stdout:
            raise ValueError("missing_elf_dependency")
        for library in re.findall(r"(?:=>\s+|^\s*)(/[^\s()]+)", result.stdout, re.MULTILINE):
            source = Path(library).resolve()
            if source.name.startswith("libpython"):
                destination = "usr/lib/" + Path(library).name
            elif (
                source.is_relative_to("/usr/lib")
                or source.is_relative_to("/lib")
                or source.is_relative_to("/lib64")
            ):
                destination = library.lstrip("/")
            else:
                raise ValueError("unexpected_elf_dependency_location")
            if destination in files and files[destination] != source:
                raise ValueError("dependency_collision")
            files[destination] = source
    files["verifier/_candidate.py"] = REPO / "rl/verifiers/_candidate.py"
    return files


def prepare(
    destination: Path,
    expected_python_sha256: str,
    *,
    dry_run=False,
    expected_manifest: Path | None = None,
):
    destination = destination.absolute()
    if destination.is_symlink() or not (
        destination.resolve().is_relative_to(REPO / ".tools")
        or destination.resolve().is_relative_to(REPO / ".cache")
        or destination.resolve().is_relative_to("/tmp")
    ):
        raise ValueError("destination_must_be_project_tools_or_tmp")
    if destination.exists() or destination.with_name(destination.name + ".manifest.json").exists():
        raise ValueError("destination_already_exists")
    executable = Path(sys.executable).resolve()
    if sha256(executable) != expected_python_sha256:
        raise ValueError("python_digest_mismatch")
    files = plan_files()
    total = sum(source.stat().st_size for source in files.values())
    if total > MAX_ROOTFS_BYTES:
        raise ValueError("rootfs_size_limit")
    existing = destination.parent
    while not existing.exists():
        existing = existing.parent
    # Bound construction + retained destination at 2x and preserve 50 GiB headroom.
    peak = 2 * total
    if shutil.disk_usage(existing).free < 50 * 1024**3 + peak:
        raise ValueError("insufficient_disk_headroom")
    summary = {
        "files": len(files) + 1,
        "bytes": total,
        "peak_additional_bytes": peak,
        "python_version": sys.version,
        "python_sha256": expected_python_sha256,
        "destination": str(destination),
    }
    if dry_run:
        return summary
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="verifier-rootfs-build-", dir=destination.parent))
    try:
        manifest = {
            "format": "crucible-rootfs-v1",
            "python_version": sys.version,
            "python_sha256": expected_python_sha256,
            "files": {},
            "sources": {},
        }
        for relative, source in sorted(files.items()):
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            target.chmod(0o555 if relative == "usr/bin/python3" or ".so" in target.name else 0o444)
            manifest["files"][relative] = {
                "sha256": sha256(target),
                "mode": target.stat().st_mode & 0o777,
            }
            manifest["sources"][relative] = str(source)
        for directory in ("tmp", "proc", "dev", "tests"):
            (stage / directory).mkdir(exist_ok=True)
        public = stage / "tests/public.json"
        public.write_text("[]")
        public.chmod(0o444)
        manifest["files"]["tests/public.json"] = {"sha256": sha256(public), "mode": 0o444}
        # All directories are traversable by the sandbox's unprivileged uid.
        for directory in stage.rglob("*"):
            if directory.is_dir():
                directory.chmod(0o755)
        stage.chmod(0o755)
        if expected_manifest is not None:
            expected = json.loads(expected_manifest.read_text())
            if manifest["files"] != expected["files"]:
                raise ValueError("dependency_lock_mismatch")
        stage.rename(destination)
        raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        manifest_path = destination.with_name(destination.name + ".manifest.json")
        manifest_path.write_bytes(raw)
        summary.update(manifest=str(manifest_path), manifest_sha256=hashlib.sha256(raw).hexdigest())
        return summary
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-python-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--expected-manifest",
        type=Path,
        help="Require exact pinned file hashes/modes before publishing rootfs",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.destination,
                args.expected_python_sha256,
                dry_run=args.dry_run,
                expected_manifest=args.expected_manifest,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
