"""Rootfs assembly tests copy synthetic trusted files; no Python is executed."""

import hashlib
import sys
from pathlib import Path

import pytest

from scripts import prepare_verifier_rootfs as prep


def test_rootfs_pin_and_destination_required(tmp_path):
    with pytest.raises(ValueError, match="python_digest"):
        prep.prepare(tmp_path / "root", "bad", dry_run=True)
    with pytest.raises(ValueError, match="project_tools_or_tmp"):
        prep.prepare(Path("/etc/crucible-unapproved"), "bad", dry_run=True)


def test_rootfs_fixture_plan_manifest_and_dry_run(tmp_path, monkeypatch):
    source = tmp_path / "trusted-file"
    source.write_bytes(b"trusted fixture")
    monkeypatch.setattr(
        prep, "plan_files", lambda: {"usr/bin/python3": source, "verifier/_candidate.py": source}
    )
    expected = hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest()
    target = tmp_path / "rootfs"
    planned = prep.prepare(target, expected, dry_run=True)
    assert not target.exists() and planned["peak_additional_bytes"] == 2 * planned["bytes"]
    result = prep.prepare(target, expected)
    assert target.joinpath("tests/public.json").read_text() == "[]"
    manifest = Path(result["manifest"])
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == result["manifest_sha256"]
    assert not any(path.is_symlink() for path in target.rglob("*"))
    with pytest.raises(ValueError, match="already_exists"):
        prep.prepare(target, expected)


def test_rootfs_rejects_oversized_plan(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"x")
    monkeypatch.setattr(prep, "plan_files", lambda: {"usr/bin/python3": source})
    monkeypatch.setattr(prep, "MAX_ROOTFS_BYTES", 0)
    expected = hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="size_limit"):
        prep.prepare(tmp_path / "root", expected)


def test_rootfs_dependency_lock_rejects_changed_files(tmp_path, monkeypatch):
    import json

    source = tmp_path / "trusted"
    source.write_bytes(b"trusted")
    monkeypatch.setattr(prep, "plan_files", lambda: {"usr/bin/python3": source})
    expected = hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest()
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"files": {}}))
    target = tmp_path / "new-root"
    with pytest.raises(ValueError, match="dependency_lock_mismatch"):
        prep.prepare(target, expected, expected_manifest=lock)
    assert not target.exists() and not list(tmp_path.glob("verifier-rootfs-build-*"))
