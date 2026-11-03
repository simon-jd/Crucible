"""The project's own disk-share limits follow the resource budget (doubled 2026-10-27; Crucible 40 GB from 2026-10-29)."""

import subprocess
from pathlib import Path

import pytest

from scripts.resource_preflight import (
    GB,
    PROJECT_SHARE_BYTES,
    SHARED_MODEL_CACHE_SHARE_BYTES,
    check_limits,
)

ROOT = Path(__file__).resolve().parents[2]
# Old literals that would silently keep an earlier project share (14 GB, then 28 GB).
STALE = (
    "14_000_000_000",
    "14 * 1000 * 1000 * 1000",
    "14 * GB",
    "14 GB share",
    "14 GB project",
    "14 GB cap",
    "28_000_000_000",
    "28,000,000,000",
    "28 * 1000 * 1000 * 1000",
    "28 * GB",
    "28 GB share",
    "28 GB project",
    "28 GB cap",
    "28 GB** 上限",
)


def test_shares_match_the_budget():
    assert PROJECT_SHARE_BYTES == 40 * GB
    assert SHARED_MODEL_CACHE_SHARE_BYTES == 80 * GB


def test_preflight_uses_the_project_share():
    base = {
        "planned_project_bytes": 0, "download_bytes": 0, "free_bytes": 60 * GB, "checkpoints": 0,
        "service_memory_bytes": 0, "model_cache": Path.home() / ".cache/huggingface",
    }
    assert check_limits(project_size=PROJECT_SHARE_BYTES, **base) == []
    errors = check_limits(project_size=PROJECT_SHARE_BYTES + 1, **base)
    assert errors and "40 GB" in errors[0]


def test_runtime_and_task14_guards_use_the_project_share():
    from scripts import task14_check
    from serving.baseline import prepare_runtime

    assert prepare_runtime.CAP == PROJECT_SHARE_BYTES
    assert task14_check.PROJECT_SHARE_BYTES == PROJECT_SHARE_BYTES
    prepare_runtime.check_peak(PROJECT_SHARE_BYTES - 8 * GB, 8 * GB, 0)
    with pytest.raises(ValueError):
        prepare_runtime.check_peak(PROJECT_SHARE_BYTES - 8 * GB, 8 * GB, 1)


def test_rl_resource_limits_use_the_project_share():
    pytest.importorskip("yaml")
    from rl.configs import experiments

    assert experiments.RESOURCE_LIMITS["project_cap_bytes"] == PROJECT_SHARE_BYTES


def test_model_snapshot_high_water_uses_both_shares(tmp_path, monkeypatch):
    from serving.baseline import model_snapshot as model

    counts = {tmp_path: 70 * GB, model.ROOT: 20 * GB}
    monkeypatch.setattr(model, "project_bytes", lambda path: counts[path])
    monkeypatch.setattr(model.shutil, "disk_usage", lambda path: type("D", (), {"free": 500 * GB})())
    # Over the old 14/40 GB shares, inside the doubled ones.
    model.resource_sample(tmp_path, 9 * GB)
    with pytest.raises(RuntimeError, match="high-water"):
        model.resource_sample(tmp_path, 10 * GB + 1)
    counts[model.ROOT] = PROJECT_SHARE_BYTES - 650_000_000 + 1
    with pytest.raises(RuntimeError, match="high-water"):
        model.resource_sample(tmp_path, 0)


def test_train_eagle3_checks_the_project_share():
    text = (ROOT / "spec/train_eagle3.sh").read_text()
    assert "share_bytes=$((40 * 1000 * 1000 * 1000))" in text
    assert "used + need_bytes > share_bytes" in text


def test_no_stale_share_literals_in_tracked_sources():
    files = subprocess.run(
        ["git", "ls-files", "*.py", "*.sh", "*.go", "*.md"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    skip = ("docs/evidence/", "scripts/tests/test_disk_shares.py")
    stale = []
    for name in files:
        if name.startswith(skip):
            continue
        text = (ROOT / name).read_text(errors="replace")
        stale += [f"{name}: {s}" for s in STALE if s in text]
    assert stale == []
