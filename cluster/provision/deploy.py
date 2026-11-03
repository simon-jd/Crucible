#!/usr/bin/env python3
"""Plan by default; explicitly deploy reviewed bundles to one controller/four workers."""

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import bootstrap
from baseline import BaselineError, validate_lock

HERE = Path(__file__).resolve().parent


def checked_file(base, item):
    path = bootstrap.source(base, item["file"])
    if not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256") or ""):
        raise ValueError("deployment artifact SHA256 is missing/unlocked")
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError("deployment artifact checksum mismatch")
    return path


def inspect(inventory):
    result = {"mode": "plan", "ready": False, "applied": False, "nodes": [], "errors": []}
    try:
        data = json.loads(inventory.read_text())
        scheduler = data["scheduler"]
        if scheduler not in ("slurm", "k8s"):
            raise ValueError("scheduler must be slurm or k8s")
        validate_lock(
            json.loads(bootstrap.source(inventory.parent, data["lock"]).read_text()), scheduler
        )
        nodes = data["nodes"]
        if len(nodes) != 5 or len({node["host"] for node in nodes}) != 5:
            raise ValueError("inventory requires one controller and four distinct worker hosts")
        roles = []
        for node in nodes:
            if not re.fullmatch(r"[a-z_][a-z0-9_-]*@[A-Za-z0-9][A-Za-z0-9.-]*", node["host"]):
                raise ValueError("host must be user@hostname")
            manifest = bootstrap.source(inventory.parent, node["manifest"])
            preview = bootstrap.plan(manifest)
            if not preview["ready"]:
                raise ValueError(
                    "node artifacts/config/secrets not ready: " + "; ".join(preview["errors"])
                )
            if scheduler == "k8s" and not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9.-]*", node.get("node_name", "")
            ):
                raise ValueError("Kubernetes inventory requires each kubelet node_name")
            roles.append(preview["role"])
            result["nodes"].append(
                {
                    "host": node["host"],
                    "role": preview["role"],
                    "packages": preview["packages"],
                    "files": preview["files"],
                }
            )
        if roles.count(scheduler + "-controller") != 1 or roles.count(scheduler + "-worker") != 4:
            raise ValueError("node roles must match the scheduler: one controller, four workers")
        if scheduler == "k8s":
            config = bootstrap.source(inventory.parent, data["kubeconfig"])
            if config.stat().st_mode & 0o077:
                raise ValueError("kubeconfig must exclude group/other access")
            lock = json.loads(bootstrap.source(inventory.parent, data["lock"]).read_text())
            releases = data["helm_releases"]
            expected = {
                "gpu-operator": "gpu_operator",
                "network-operator": "network_operator",
                "kueue": "kueue",
            }
            if {r["name"] for r in releases} != set(expected) or len(releases) != 3:
                raise ValueError("three pinned operator Helm releases are required")
            for release in releases:
                if release["version"] != lock["versions"][expected[release["name"]]]:
                    raise ValueError("Helm version differs from versions.lock")
                checked_file(inventory.parent, release["chart"])
                for values in release["values"]:
                    checked_file(inventory.parent, values)
            if not data.get("manifests_before_helm") or not data.get("manifests_after_helm"):
                raise ValueError("pinned CNI/namespace and baseline manifests are required")
            for item in data["manifests_before_helm"] + data["manifests_after_helm"]:
                checked_file(inventory.parent, item)
        result["ready"] = True
    except (OSError, ValueError, KeyError, TypeError, AttributeError, BaselineError) as error:
        result["errors"].append(str(error))
    return result


def ssh(host, command):
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", host, command]


def stage(manifest, directory):
    """Package only this node's reviewed inputs; preserve secrets without printing them."""
    data = bootstrap.read_manifest(manifest)
    for index, artifact in enumerate(data["artifacts"]):
        source = bootstrap.source(manifest.parent, artifact["file"])
        name = f"package-{index}.deb"
        shutil.copyfile(source, directory / name)
        artifact["file"] = name
    for index, item in enumerate(data.get("files", [])):
        source = bootstrap.source(manifest.parent, item["source"])
        name = f"config-{index}"
        shutil.copyfile(source, directory / name)
        (directory / name).chmod(0o600 if item.get("secret") else 0o644)
        item["source"] = name
    shutil.copyfile(HERE / "bootstrap.py", directory / "bootstrap.py")
    (directory / "manifest.json").write_text(json.dumps(data))
    (directory / "manifest.json").chmod(0o600)


def apply(inventory, run=bootstrap.execute):
    preview = inspect(inventory)
    if not preview["ready"]:
        raise ValueError("deployment input validation failed")
    data = json.loads(inventory.read_text())
    # Controller is configured first; join configs must contain preselected bootstrap tokens.
    nodes = sorted(
        data["nodes"],
        key=lambda n: bootstrap.read_manifest(bootstrap.source(inventory.parent, n["manifest"]))[
            "role"
        ].endswith("worker"),
    )
    for node in nodes:
        remote = "/var/tmp/crucible-bootstrap-" + uuid.uuid4().hex
        try:
            with tempfile.TemporaryDirectory(prefix="crucible-bootstrap-") as temporary:
                directory = Path(temporary)
                stage(bootstrap.source(inventory.parent, node["manifest"]), directory)
                run(ssh(node["host"], "umask 077; mkdir " + shlex.quote(remote)))
                run(
                    [
                        "scp",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "StrictHostKeyChecking=yes",
                        "-r",
                        str(directory) + "/.",
                        node["host"] + ":" + remote + "/",
                    ]
                )
                run(
                    ssh(
                        node["host"],
                        shlex.join(
                            [
                                "sudo",
                                "-n",
                                "python3",
                                remote + "/bootstrap.py",
                                "--manifest",
                                remote + "/manifest.json",
                                "--apply",
                            ]
                        ),
                    )
                )
        finally:
            # The random directory was created by this invocation and contains credentials.
            run(ssh(node["host"], shlex.join(["sudo", "-n", "rm", "-rf", "--", remote])))
    if data["scheduler"] == "k8s":
        kube = ["--kubeconfig", str(bootstrap.source(inventory.parent, data["kubeconfig"]))]
        for node in data["nodes"]:
            if (
                bootstrap.read_manifest(bootstrap.source(inventory.parent, node["manifest"]))[
                    "role"
                ]
                == "k8s-worker"
            ):
                run(
                    [
                        "kubectl",
                        *kube,
                        "label",
                        "node",
                        node["node_name"],
                        "crucible.ai/worker=true",
                        "--overwrite",
                    ]
                )
        for item in data["manifests_before_helm"]:
            run(["kubectl", *kube, "apply", "-f", str(checked_file(inventory.parent, item))])
        for release in data["helm_releases"]:
            command = [
                "helm",
                *kube,
                "upgrade",
                "--install",
                release["name"],
                str(checked_file(inventory.parent, release["chart"])),
                "--version",
                release["version"],
                "--namespace",
                release["namespace"],
                "--create-namespace",
                "--wait",
                "--timeout",
                "10m",
            ]
            for values in release["values"]:
                command += ["-f", str(checked_file(inventory.parent, values))]
            run(command)
        for item in data["manifests_after_helm"]:
            run(["kubectl", *kube, "apply", "-f", str(checked_file(inventory.parent, item))])
    return {"mode": "apply", "applied": True, "acceptance": "UNVERIFIED; run check_baseline.sh"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="explicit SSH/node/cluster mutation")
    args = parser.parse_args()
    preview = inspect(args.inventory)
    if not args.apply:
        print(json.dumps(preview, indent=2))
        return 0 if preview["ready"] else 1
    if not preview["ready"]:
        print(json.dumps(preview, indent=2))
        return 1
    try:
        print(json.dumps(apply(args.inventory), indent=2))
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError):
        parser.exit(1, "FAIL: deployment failed; installer/credential output withheld\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
