"""Trusted executable fixtures, never real isolation or candidate execution."""

import hashlib
import json
from pathlib import Path

import pytest

from rl.verifiers.runsc import LocalRunsc, RunscConfig, build_spec, verify_rootfs


def config(tmp_path):
    root = tmp_path / "rootfs"
    root.mkdir()
    files = {}
    for name, content in [
        ("usr/bin/python3", b"fixture only"),
        ("verifier/_candidate.py", Path("rl/verifiers/_candidate.py").read_bytes()),
        ("tests/public.json", b"[]"),
    ]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        target.chmod(0o444)
        files[name] = {"sha256": hashlib.sha256(content).hexdigest(), "mode": 0o444}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": files}))
    runtime = tmp_path / "runsc-fixture"
    runtime.write_text('#!/usr/bin/python3\nimport sys\nif "run" in sys.argv: print("2")\n')
    runtime.chmod(0o755)
    return RunscConfig(
        runtime,
        root,
        manifest,
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
        tmp_path / "work",
        Path("/sys/fs/cgroup/crucible-fixture-unprovisioned"),
    )


def test_oci_spec_is_dedicated_readonly_and_bounded(tmp_path):
    cfg = config(tmp_path)
    public = tmp_path / "public.json"
    public.write_text("[]")
    spec = build_spec(cfg, "encoded-request", public, "crucible-fixture", 4)
    assert spec["root"] == {"path": str(cfg.rootfs), "readonly": True}
    assert spec["process"]["user"] == {"uid": 65534, "gid": 65534}
    assert spec["process"]["noNewPrivileges"] is True
    assert all(not value for value in spec["process"]["capabilities"].values())
    assert spec["process"]["args"][:4] == [
        "/usr/bin/python3",
        "-I",
        "-S",
        "/verifier/_candidate.py",
    ]
    mounts = spec["mounts"]
    bind = [mount for mount in mounts if mount["type"] == "bind"]
    assert len(bind) == 1 and bind[0]["source"] == str(public) and "ro" in bind[0]["options"]
    assert any(ns["type"] == "network" for ns in spec["linux"]["namespaces"])
    assert spec["linux"]["resources"]["pids"]["limit"] == 128
    assert spec["linux"]["resources"]["memory"]["limit"] == 512 * 1024**2
    assert "size=16777216" in mounts[1]["options"]


def test_rootfs_manifest_hash_and_tampering(tmp_path):
    cfg = config(tmp_path)
    verify_rootfs(cfg.rootfs, cfg.manifest, cfg.manifest_sha256)
    with pytest.raises(ValueError, match="manifest_digest"):
        verify_rootfs(cfg.rootfs, cfg.manifest, "0" * 64)
    (cfg.rootfs / "usr/bin/python3").chmod(0o644)
    (cfg.rootfs / "usr/bin/python3").write_bytes(b"changed")
    with pytest.raises(ValueError, match="contents_mismatch"):
        verify_rootfs(cfg.rootfs, cfg.manifest, cfg.manifest_sha256)


def test_rootfs_rejects_extra_files_and_escaping_links(tmp_path):
    cfg = config(tmp_path)
    extra = cfg.rootfs / "extra"
    extra.write_text("unexpected")
    with pytest.raises(ValueError, match="contents_mismatch"):
        verify_rootfs(cfg.rootfs, cfg.manifest, cfg.manifest_sha256)
    extra.unlink()
    extra.symlink_to("/etc")
    with pytest.raises(ValueError, match="escaping_symlink"):
        verify_rootfs(cfg.rootfs, cfg.manifest, cfg.manifest_sha256)


def test_missing_preconditions_launch_nothing(tmp_path, monkeypatch):
    cfg = config(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("must not launch")

    monkeypatch.setattr("rl.verifiers.runsc.subprocess.Popen", forbidden)
    result = LocalRunsc(cfg).run("def solve(x): return x", 1, "[]", 1)
    assert result.error == "local_runtime_prerequisites_missing"
    assert result.cleanup_confirmed and not result.completed


def test_fixture_lifecycle_success_cleanup_and_no_host_candidate(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    adapter = LocalRunsc(cfg)
    monkeypatch.setattr(adapter, "_preflight", lambda: None)  # Fixture ONLY.
    marker = tmp_path / "candidate-ran-on-host"
    candidate = f'open({str(marker)!r}, "w").write("BAD")'
    result = adapter.run(candidate, 1, "[]", 2)
    assert result.completed and result.cleanup_confirmed and result.stdout == "2\n"
    assert not marker.exists() and not adapter.last_bundle.exists()


@pytest.mark.parametrize(
    "behavior,flag",
    [("import time; time.sleep(5)", "timed_out"), ('print("x"*100000)', "output_truncated")],
)
def test_fixture_timeout_and_output_limit_cleanup(tmp_path, monkeypatch, behavior, flag):
    cfg = config(tmp_path)
    cfg.runsc.write_text(
        '#!/usr/bin/python3\nimport sys\nif "run" in sys.argv:\n ' + behavior + "\n"
    )
    adapter = LocalRunsc(cfg)
    monkeypatch.setattr(adapter, "_preflight", lambda: None)
    result = adapter.run("never executed", 1, "[]", 1)
    assert getattr(result, flag) and result.cleanup_confirmed
    assert not adapter.last_bundle.exists()


def test_fixture_failed_delete_retains_bundle_and_zero_success(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg.runsc.write_text(
        '#!/usr/bin/python3\nimport sys\nif "run" in sys.argv: print("2")\nif "delete" in sys.argv: sys.exit(1)\n'
    )
    adapter = LocalRunsc(cfg)
    monkeypatch.setattr(adapter, "_preflight", lambda: None)
    result = adapter.run("never executed", 1, "[]", 1)
    assert not result.cleanup_confirmed and adapter.last_bundle.exists()


def test_fixture_invalid_utf8_is_not_repaired_into_answer(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg.runsc.write_text(
        '#!/usr/bin/python3\nimport sys\nif "run" in sys.argv: sys.stdout.buffer.write(b"\\xff")\n'
    )
    adapter = LocalRunsc(cfg)
    monkeypatch.setattr(adapter, "_preflight", lambda: None)
    result = adapter.run("never executed", 1, "[]", 1)
    assert result.error == "invalid_utf8_output" and result.cleanup_confirmed
