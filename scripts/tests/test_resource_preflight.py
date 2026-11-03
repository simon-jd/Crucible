import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.resource_preflight import GB, PROJECT_SHARE_BYTES, check_limits, project_bytes


def check(**changes):
    values = {
        "project_size": 10 * GB,
        "planned_project_bytes": GB,
        "download_bytes": 0,
        "free_bytes": 60 * GB,
        "checkpoints": 2,
        "service_memory_bytes": 16 * GB,
        "model_cache": Path.home() / ".cache/huggingface",
    }
    return check_limits(**(values | changes))


def test_exact_limits_and_shared_cache():
    assert check(project_size=PROJECT_SHARE_BYTES, planned_project_bytes=0) == []
    assert check(download_bytes=6 * GB, free_bytes=56 * GB, planned_project_bytes=0) == []


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"project_size": PROJECT_SHARE_BYTES}, "40 GB"),
        ({"checkpoints": 3}, "2 checkpoints"),
        ({"download_bytes": 6 * GB, "free_bytes": 56 * GB - 1}, "50 GB"),
        ({"download_bytes": 5 * GB, "free_bytes": 5 * GB - 1}, "download"),
        ({"service_memory_bytes": 16 * GB + 1}, "16 GB"),
        ({"model_cache": Path("/tmp/duplicate-models")}, "huggingface"),
    ],
)
def test_rejects_overages(changes, expected):
    assert expected in " ".join(check(**changes))


@pytest.mark.parametrize(
    "key",
    [
        "project_size",
        "planned_project_bytes",
        "download_bytes",
        "free_bytes",
        "checkpoints",
        "service_memory_bytes",
    ],
)
def test_rejects_negative_declarations(key):
    with pytest.raises(ValueError):
        check(**{key: -1})


def test_disk_measurement_does_not_follow_external_symlinks(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (tmp_path / "shared").write_bytes(b"x" * 100)
    (project / "data").write_bytes(b"x" * 7)
    (project / "external").symlink_to(tmp_path / "shared")
    assert project_bytes(project) == 7


def test_project_growth_also_reserves_space_during_shared_cache_download():
    assert check(download_bytes=6 * GB, free_bytes=56 * GB, planned_project_bytes=GB)


CACHE_VARIABLES = (
    "HF_HOME",
    "XDG_CACHE_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
)


def run_preflight(overrides):
    env = {key: value for key, value in os.environ.items() if key not in CACHE_VARIABLES}
    env.update(overrides)
    script = Path(__file__).resolve().parents[1] / "resource_preflight.py"
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--planned-project-bytes",
            "0",
            "--download-bytes",
            "0",
            "--checkpoints",
            "0",
            "--service-memory-bytes",
            "0",
        ],
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize("xdg", ["/tmp/crucible-review-xdg", ""])
def test_cli_rejects_nonshared_xdg_default_when_hf_home_unset(xdg):
    result = run_preflight({"XDG_CACHE_HOME": xdg})
    assert result.returncode != 0, result.stdout


@pytest.mark.parametrize("xdg", ["/tmp/crucible-review-xdg", ""])
def test_cli_explicit_shared_hf_home_takes_priority_over_xdg(xdg):
    cache = Path.home() / ".cache/huggingface"
    result = run_preflight({"HF_HOME": str(cache), "XDG_CACHE_HOME": xdg})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["model_cache"] == str(cache)


@pytest.mark.parametrize("overrides", [{}, {"XDG_CACHE_HOME": str(Path.home() / ".cache")}])
def test_cli_default_shared_cache_remains_valid(overrides):
    result = run_preflight(overrides)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True


@pytest.mark.parametrize(
    "variable", ["HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"]
)
def test_cli_empty_cache_values_are_not_treated_as_unset(variable):
    result = run_preflight({variable: ""})
    assert result.returncode != 0, result.stdout


@pytest.mark.parametrize(
    "variable", ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"]
)
def test_cli_conflicting_cache_override_still_rejected(variable):
    result = run_preflight(
        {
            "HF_HOME": str(Path.home() / ".cache/huggingface"),
            variable: "/tmp/crucible-nonshared-hub",
        }
    )
    assert result.returncode != 0, result.stdout


def test_cli_rejected_xdg_reports_effective_path_instead_of_shared_default():
    result = run_preflight({"XDG_CACHE_HOME": "/tmp/crucible-review-xdg"})
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["model_cache"] == "/tmp/crucible-review-xdg/huggingface"


def test_capacity_query_uses_effective_xdg_filesystem(monkeypatch, tmp_path):
    from scripts.resource_preflight import main

    for key in CACHE_VARIABLES:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "sys.argv",
        [
            "resource_preflight.py",
            "--planned-project-bytes",
            "0",
            "--download-bytes",
            "0",
            "--checkpoints",
            "0",
            "--service-memory-bytes",
            "0",
        ],
    )
    measured = []

    def capacity(path):
        measured.append(path)
        return SimpleNamespace(free=100 * GB)

    monkeypatch.setattr("scripts.resource_preflight.shutil.disk_usage", capacity)
    monkeypatch.setattr("scripts.resource_preflight.project_bytes", lambda root: 0)
    assert main() == 1
    assert measured[0] == tmp_path
