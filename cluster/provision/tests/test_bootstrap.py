"""Offline bootstrap tests; synthetic artifacts are never installed on this host."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_default_bootstrap_plan_does_not_run_installers(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bootstrap.py"),
            "--manifest",
            str(ROOT / "bootstrap.example.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    plan = json.loads(result.stdout)
    assert plan["mode"] == "plan"
    assert plan["ready"] is False
    assert "node_image_id" in " ".join(plan["errors"])


def test_synthetic_bootstrap_is_idempotent_without_host_changes(tmp_path):
    spec = importlib.util.spec_from_file_location("bootstrap", ROOT / "bootstrap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    artifact = tmp_path / "synthetic.deb"
    artifact.write_bytes(b"synthetic, never dpkg installed")
    secret = tmp_path / "munge.key"
    secret.write_bytes(b"NEVER-PRINT-THIS-SECRET")
    secret.chmod(0o600)
    config = tmp_path / "slurm.conf"
    config.write_text("ClusterName=crucible\n")
    manifest = {
        "schema": 1,
        "role": "slurm-worker",
        "node_image_id": "synthetic-image",
        "artifacts": [
            {
                "file": artifact.name,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "package": "synthetic",
                "version": "1.2.3",
            }
        ],
        "files": [
            {"source": config.name, "target": "/etc/slurm/slurm.conf", "mode": "0644"},
            {
                "source": secret.name,
                "target": "/etc/munge/munge.key",
                "mode": "0400",
                "secret": True,
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    plan = module.plan(path)
    assert plan["ready"]
    assert b"NEVER-PRINT" not in json.dumps(plan).encode()
    calls = []
    installed = {}

    def fake_execute(argv):
        calls.append(argv)
        if argv[0] == "dpkg-query":
            return installed.get("synthetic", "")
        if argv[0] == "dpkg-deb":
            return "synthetic\n1.2.3"
        if argv[0] == "dpkg":
            installed["synthetic"] = "1.2.3\tinstalled"
        return ""

    root = tmp_path / "synthetic-root"
    module.apply(path, root=root, execute=fake_execute)
    module.apply(path, root=root, execute=fake_execute)
    assert sum(c[0] == "dpkg" for c in calls) == 1
    assert sum(c[:2] == ["systemctl", "restart"] for c in calls) == 2  # munge + slurmd once
    assert (root / "etc/munge/munge.key").stat().st_mode & 0o777 == 0o400


def test_deployment_default_is_offline_and_unprovided_artifacts_fail_closed():
    result = subprocess.run(
        [sys.executable, str(ROOT / "deploy.py"), "--inventory", str(ROOT / "deploy.example.json")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data["mode"] == "plan"
    assert data["ready"] is False
    assert not data["applied"]


def load(name):
    sys.path.insert(0, str(ROOT))
    try:
        spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def synthetic_site(tmp_path):
    """Synthetic local bytes/versions only; never real Debian or node evidence."""
    artifact = tmp_path / "synthetic.deb"
    artifact.write_bytes(b"synthetic package")
    secret = tmp_path / "munge.key"
    secret.write_bytes(b"DO-NOT-LEAK-SECRET")
    secret.chmod(0o600)
    lock = json.loads((ROOT / "versions.lock").read_text())
    lock.update(status="locked", node_image={"id": "synthetic-image", "sha256": "c" * 64})
    lock["image"].update(digest="sha256:" + "a" * 64, slurm_sqsh_sha256="b" * 64)
    (tmp_path / "site.lock").write_text(json.dumps(lock))
    nodes = []
    for index in range(5):
        role = "slurm-controller" if index == 0 else "slurm-worker"
        manifest = {
            "schema": 1,
            "role": role,
            "node_image_id": "synthetic-image",
            "artifacts": [
                {
                    "file": artifact.name,
                    "package": "synthetic",
                    "version": "1.2.3",
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            ],
            "files": [
                {
                    "source": secret.name,
                    "target": "/etc/munge/munge.key",
                    "mode": "0400",
                    "secret": True,
                }
            ],
        }
        name = f"node-{index}.json"
        (tmp_path / name).write_text(json.dumps(manifest))
        nodes.append({"host": f"operator@node{index}.example", "manifest": name})
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"scheduler": "slurm", "lock": "site.lock", "nodes": nodes}))
    return inventory


def test_complete_deployment_plan_and_injected_transport_never_expose_secrets(tmp_path):
    module = load("deploy")
    inventory = synthetic_site(tmp_path)
    preview = module.inspect(inventory)
    assert preview["ready"], preview
    assert "DO-NOT-LEAK" not in json.dumps(preview)
    calls, staged = [], []

    def fake_transport(argv):
        calls.append(argv)
        if argv[0] == "scp":
            payload = Path(argv[-2])
            manifest = json.loads((payload / "manifest.json").read_text())
            staged.append(manifest["role"])
            key = payload / manifest["files"][0]["source"]
            assert key.stat().st_mode & 0o777 == 0o600
            assert key.read_bytes() == b"DO-NOT-LEAK-SECRET"
        return ""

    result = module.apply(inventory, run=fake_transport)
    assert result["applied"]
    assert staged == ["slurm-controller"] + ["slurm-worker"] * 4
    assert sum("--apply" in str(call) for call in calls) == 5
    assert sum("rm -rf" in str(call) for call in calls) == 5
    assert "DO-NOT-LEAK" not in json.dumps(calls)


def test_artifact_drift_and_insecure_secret_fail_before_transport(tmp_path):
    module = load("deploy")
    inventory = synthetic_site(tmp_path)
    (tmp_path / "munge.key").chmod(0o644)
    assert not module.inspect(inventory)["ready"]
    (tmp_path / "munge.key").chmod(0o600)
    (tmp_path / "synthetic.deb").write_bytes(b"tampered")
    preview = module.inspect(inventory)
    assert not preview["ready"]
    assert "checksum" in " ".join(preview["errors"])

    def no_transport(argv):
        raise AssertionError("transport must not run for invalid input")

    import pytest

    with pytest.raises(ValueError, match="validation"):
        module.apply(inventory, run=no_transport)


def test_default_complete_plan_never_invokes_ssh(tmp_path):
    inventory = synthetic_site(tmp_path)
    # PATH is empty: a hidden ssh/scp/dpkg call would fail this complete plan.
    import os

    result = subprocess.run(
        [sys.executable, str(ROOT / "deploy.py"), "--inventory", str(inventory)],
        env=dict(os.environ, PATH=""),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ready"]
    assert "DO-NOT-LEAK" not in result.stdout + result.stderr


def test_k8s_bootstrap_second_apply_preserves_runtime_and_skips_join(tmp_path):
    module = load("bootstrap")
    synthetic_site(tmp_path)
    path = tmp_path / "node-1.json"
    manifest = json.loads(path.read_text())
    manifest["role"] = "k8s-worker"
    join = tmp_path / "join.yaml"
    join.write_text("SYNTHETIC-TOKEN-DO-NOT-PRINT")
    join.chmod(0o600)
    (tmp_path / "containerd.toml").write_bytes((ROOT / "k8s/containerd.toml").read_bytes())
    manifest["files"] = [
        {
            "source": "join.yaml",
            "target": "/etc/kubernetes/join.yaml",
            "mode": "0600",
            "secret": True,
        },
        {"source": "containerd.toml", "target": "/etc/containerd/config.toml", "mode": "0644"},
    ]
    path.write_text(json.dumps(manifest))
    root = tmp_path / "root"
    calls, installed = [], {}

    def fake_execute(argv):
        calls.append(argv)
        if argv[0] == "dpkg-query":
            return installed.get("version", "")
        if argv[0] == "dpkg-deb":
            return "Package: synthetic\nVersion: 1.2.3"
        if argv[0] == "dpkg":
            installed["version"] = "1.2.3\tinstalled"
        if argv[0] == "kubeadm":
            marker = root / "etc/kubernetes/kubelet.conf"
            marker.write_text("SYNTHETIC")
        return ""

    assert module.apply(path, root=root, execute=fake_execute)["changed"]
    assert not module.apply(path, root=root, execute=fake_execute)["changed"]
    assert sum(c[0] == "kubeadm" for c in calls) == 1
    assert "SYNTHETIC-TOKEN" not in json.dumps(calls)
    assert (root / "etc/containerd/config.toml").read_bytes() == (
        ROOT / "k8s/containerd.toml"
    ).read_bytes()


@pytest.mark.parametrize(
    "initial_source,initial_fstype,should_pass,activation_source",
    [
        ("storage-a:/crucible", "nfs4", False, "storage-b:/crucible"),
        ("storage-b:/crucible", "ext4", False, "storage-b:/crucible"),
        ("storage-b:/crucible", "nfs4", True, "storage-b:/crucible"),
        (None, None, True, "storage-b:/crucible"),
        (None, None, False, "storage-a:/crucible"),
    ],
)
def test_shared_writes_require_verified_mount_source_and_type(
    tmp_path, initial_source, initial_fstype, should_pass, activation_source
):
    """All mount/package commands are injected CPU fixtures; no real mount occurs."""
    module = load("bootstrap")
    synthetic_site(tmp_path)
    path = tmp_path / "node-1.json"
    manifest = json.loads(path.read_text())
    (tmp_path / "shared.mount").write_text(
        "[Mount]\nWhat=storage-b:/crucible\nWhere=/shared\nType=nfs4\n"
    )
    (tmp_path / "probe.py").write_text("SYNTHETIC PROBE")
    (tmp_path / "fixture.sqsh").write_bytes(b"SYNTHETIC IMAGE")
    manifest["files"] += [
        {"source": "shared.mount", "target": "/etc/systemd/system/shared.mount", "mode": "0644"},
        {
            "source": "probe.py",
            "target": "/shared/crucible/cluster/provision/node_probe.py",
            "mode": "0644",
        },
        {"source": "fixture.sqsh", "target": "/shared/images/crucible.sqsh", "mode": "0644"},
    ]
    path.write_text(json.dumps(manifest))
    root = tmp_path / "root"
    (root / "shared").mkdir(parents=True)
    sentinel = root / "shared/existing-file"
    sentinel.write_bytes(b"KEEP CURRENT MOUNT DATA")
    calls = []
    mounted = {"source": initial_source, "fstype": initial_fstype}

    def fake_execute(argv):
        calls.append(argv)
        if argv[0] == "dpkg-query":
            return "1.2.3\tinstalled"
        if argv[0] == "dpkg-deb":
            return "Package: synthetic\nVersion: 1.2.3"
        if argv[0] == "findmnt":
            records = [{"target": str(root / "shared"), **mounted}] if mounted["source"] else []
            return json.dumps({"filesystems": records})
        if argv[:3] == ["systemctl", "enable", "--now"]:
            mounted.update(source=activation_source, fstype="nfs4")
        return ""

    if should_pass:
        result = module.apply(path, root=root, execute=fake_execute)
        assert result["applied"]
        assert (root / "shared/images/crucible.sqsh").read_bytes() == b"SYNTHETIC IMAGE"
        assert (root / "shared/crucible/cluster/provision/node_probe.py").is_file()
        if initial_source is None:
            assert sum(call[0] == "findmnt" for call in calls) == 3
            assert any(call[:3] == ["systemctl", "enable", "--now"] for call in calls)
        else:
            assert not any("--now" in call for call in calls)
    else:
        with pytest.raises(ValueError, match="shared mount"):
            module.apply(path, root=root, execute=fake_execute)
        assert not (root / "shared/images").exists()
        assert not (root / "shared/crucible").exists()
        if initial_source is not None:
            assert not any("--now" in call for call in calls)
    assert sentinel.read_bytes() == b"KEEP CURRENT MOUNT DATA"
    assert not any(call[0] in ("mount", "umount") for call in calls)


@pytest.mark.parametrize(
    "table",
    [
        "not-json",
        '{"filesystems":[{"target":"/shared","source":"storage-b:/crucible"}]}',
        (
            '{"filesystems":[{"target":"/shared","source":"B","fstype":"nfs4"},'
            '{"target":"/shared","source":"A","fstype":"nfs4"}]}'
        ),
    ],
)
def test_unreadable_or_ambiguous_mount_identity_is_not_treated_as_absent(table):
    module = load("bootstrap")
    with pytest.raises(module.SharedMountError, match="identity unavailable"):
        module.observed_shared_mount(Path("/"), lambda argv: table)


def test_mount_mismatch_cli_reports_pending_and_never_applied(tmp_path, monkeypatch, capsys):
    module = load("bootstrap")
    synthetic_site(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["bootstrap.py", "--manifest", str(tmp_path / "node-1.json"), "--apply"]
    )
    monkeypatch.setattr(module.os, "geteuid", lambda: 0)

    def fake_apply(path):
        raise module.SharedMountError("shared mount SOURCE/FSTYPE mismatch; no shared writes")

    monkeypatch.setattr(module, "apply", fake_apply)
    assert module.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["applied"] is False
    assert report["pending"] == "shared-mount-verification"
    assert "SOURCE/FSTYPE" in report["error"]
