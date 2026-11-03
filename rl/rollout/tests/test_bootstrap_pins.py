"""Metadata-only reproducibility gate regression; never install an alternate stack."""

import re
import subprocess
from importlib import metadata
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "package,drift",
    [("jinja2", "3.1.5"), ("mpmath", "1.2.1"), ("pyyaml", "6.0.2"), ("sympy", "1.13.3")],
)
def test_reused_pin_drift_fails_before_preflight_or_install(monkeypatch, package, drift):
    root = Path(__file__).resolve().parents[3]
    script = (root / "scripts/bootstrap_rollout.sh").read_text()
    code = script.split(".venv-ml/bin/python - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
    pins = {
        n.lower(): v
        for n, v in re.findall(
            r"^([\w-]+)==([^\s\\]+)", (root / "requirements/rollout.lock").read_text(), re.MULTILINE
        )
    }
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        metadata, "version", lambda name: drift if name.lower() == package else pins[name.lower()]
    )

    def mutation(*_args, **_kwargs):
        pytest.fail("reached preflight/install before rejecting exact reused-pin drift")

    monkeypatch.setattr(subprocess, "run", mutation)
    with pytest.raises(AssertionError, match=package, check=lambda e: package in str(e).lower()):
        # Execute only our checked-in bootstrap, with all subprocess calls blocked.
        exec(compile(code, "bootstrap_rollout.sh:metadata-gate", "exec"), {})  # noqa: S102
