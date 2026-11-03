"""Run provenance and entrypoint routing without Torch or a GPU."""

import hashlib
import json
from pathlib import Path

import pytest

from rl.sync.acceptance import load_protocol, validate_reuse, write_binding

PROTOCOL = (
    Path(__file__).resolve().parents[3] / "docs/evidence/raw/task9-confirmation-protocol.json"
)
DIGEST = "a9d94ba329dddb322156549bd1492a26d8387062d7585fb292b3ff0bd7455b4c"


def fixture_run(tmp_path):
    source = {
        "base_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "source_pid": 123,
        "source_step": {"policy_version": 8},
        "failed_learner_publication_rejected": True,
    }
    (tmp_path / "source.json").write_text(json.dumps(source))
    snapshots = {"7": "a" * 64, "8": "b" * 64}
    return snapshots


def test_fresh_binds_own_content_not_historical_snapshots(tmp_path):
    snapshots = fixture_run(tmp_path)
    binding = write_binding(tmp_path, PROTOCOL, snapshots, source_pid=123, source_exitcode=0)
    protocol = load_protocol(PROTOCOL)
    assert binding["snapshots"] == snapshots != protocol["snapshot_digests"]
    assert binding["protocol_sha256"] == DIGEST
    assert (
        binding["source_sha256"]
        == hashlib.sha256((tmp_path / "source.json").read_bytes()).hexdigest()
    )
    assert validate_reuse(tmp_path, PROTOCOL, snapshots) == binding


@pytest.mark.parametrize(
    "failure",
    [
        "missing_binding",
        "changed_source",
        "changed_snapshots",
        "changed_protocol",
        "failed_source",
        "changed_pid",
    ],
)
def test_reuse_fails_closed_on_changed_provenance(tmp_path, failure):
    snapshots = fixture_run(tmp_path)
    write_binding(tmp_path, PROTOCOL, snapshots, source_pid=123, source_exitcode=0)
    protocol_path = PROTOCOL
    if failure == "missing_binding":
        (tmp_path / "binding.json").unlink()
    elif failure == "changed_source":
        (tmp_path / "source.json").write_text("{}")
    elif failure == "changed_snapshots":
        snapshots["8"] = "c" * 64
    elif failure == "changed_protocol":
        protocol_path = tmp_path / "protocol.json"
        protocol_path.write_text("{}")
    else:
        data = json.loads((tmp_path / "binding.json").read_text())
        data["source_process"]["exitcode" if failure == "failed_source" else "pid"] = 1
        (tmp_path / "binding.json").write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_reuse(tmp_path, protocol_path, snapshots)


def test_binding_requires_confirmed_source_exit_and_does_not_overwrite(tmp_path):
    snapshots = fixture_run(tmp_path)
    with pytest.raises(ValueError, match="exit0"):
        write_binding(tmp_path, PROTOCOL, snapshots, source_pid=123, source_exitcode=1)
    write_binding(tmp_path, PROTOCOL, snapshots, source_pid=123, source_exitcode=0)
    with pytest.raises(FileExistsError):
        write_binding(tmp_path, PROTOCOL, snapshots, source_pid=123, source_exitcode=0)


@pytest.mark.parametrize(
    ("arguments", "mode", "planned"),
    [([], "fresh", "2100000000"), (["--reuse"], "reuse", "50000000")],
)
def test_shell_entrypoint_routes_only_frozen_protocol(tmp_path, arguments, mode, planned):
    import shutil
    import subprocess

    root = Path(__file__).resolve().parents[3]
    (tmp_path / "scripts").mkdir()
    (tmp_path / ".venv/bin").mkdir(parents=True)
    shutil.copyfile(root / "scripts/test_sync_gpu.sh", tmp_path / "scripts/test_sync_gpu.sh")
    # Execute the actual shell entrypoint, replacing only external side effects.
    preflight = tmp_path / ".venv/bin/python"
    preflight.write_text('#!/bin/sh\nprintf "preflight:%s\\n" "$@"\n')
    preflight.chmod(0o755)
    guard = tmp_path / "scripts/with_gpu.sh"
    guard.write_text('#!/bin/sh\nprintf "guard:%s\\n" "$@"\n')
    completed = subprocess.run(
        ["bash", str(tmp_path / "scripts/test_sync_gpu.sh"), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "preflight:" + planned in completed.stdout
    assert "guard:CRUCIBLE_SYNC_MODE=" + mode in completed.stdout
    assert "test_fresh_or_explicit_reuse_acceptance" in completed.stdout
    assert "test_actual_learner_to_vllm_full_logits" not in completed.stdout


def test_shell_entrypoint_rejects_unknown_mode_before_any_gpu():
    import subprocess

    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["bash", str(root / "scripts/test_sync_gpu.sh"), "--unknown"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2 and "usage:" in completed.stderr
