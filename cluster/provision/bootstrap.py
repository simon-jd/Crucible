#!/usr/bin/env python3
"""Offline-first Ubuntu node bootstrap from a reviewed, checksummed Debian bundle."""

import argparse
import configparser
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

ROLES = {"slurm-controller", "slurm-worker", "k8s-controller", "k8s-worker"}
TARGETS = (
    "/etc/slurm/",
    "/etc/enroot/",
    "/etc/munge/",
    "/etc/kubernetes/",
    "/etc/containerd/",
    "/etc/crucible/",
    "/etc/systemd/system/",
    "/shared/crucible/cluster/provision/",
    "/shared/images/",
)


def source(base, name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("bundle source must be relative")
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError(f"bundle source missing/escapes bundle: {name}")
    return path


def read_manifest(path):
    data = json.loads(path.read_text())
    if data.get("schema") != 1 or data.get("role") not in ROLES:
        raise ValueError("unsupported bootstrap schema/role")
    if not isinstance(data.get("node_image_id"), str) or not data["node_image_id"]:
        raise ValueError("node_image_id is required")
    return data


class SharedMountError(ValueError):
    """Shared storage could not be proven safe; current mounts stay unchanged."""


def requested_shared_mount(path, data):
    units = [
        item
        for item in data.get("files", [])
        if item["target"] == "/etc/systemd/system/shared.mount"
    ]
    if not units:
        if any(item["target"].startswith("/shared/") for item in data.get("files", [])):
            raise ValueError("shared mount definition is required before shared file writes")
        return None
    try:
        unit = configparser.ConfigParser(interpolation=None)
        unit.read_string(source(path.parent, units[0]["source"]).read_text())
        mount = unit["Mount"]
        expected = {"source": mount["What"].strip(), "fstype": mount["Type"].strip()}
        if mount["Where"].strip() != "/shared" or not all(expected.values()):
            raise ValueError("shared mount must specify What, Where=/shared and Type")
    except (configparser.Error, KeyError) as error:
        raise ValueError("shared mount definition is invalid/incomplete") from error
    return expected


def observed_shared_mount(root, run):
    try:
        # Query the kernel table without a selector: an absent /shared is a valid
        # empty match, while command/read/JSON failures remain errors, not absence.
        table = json.loads(
            run(["findmnt", "--kernel", "--json", "--list", "--output", "TARGET,SOURCE,FSTYPE"])
        )
        entries = table["filesystems"]
        if not isinstance(entries, list) or any(
            not isinstance(row, dict) or not isinstance(row.get("target"), str) for row in entries
        ):
            raise ValueError("invalid mount table")
        matches = [row for row in entries if row["target"] == str(root / "shared")]
        if len(matches) > 1:
            raise ValueError("ambiguous stacked mounts")
        if not matches:
            return None
        observed = {key: matches[0][key] for key in ("source", "fstype")}
        if not all(isinstance(value, str) and value for value in observed.values()):
            raise ValueError("missing mount identity")
        return observed
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        raise SharedMountError("shared mount identity unavailable; no shared writes") from error


def plan(path):
    result = {"mode": "plan", "ready": False, "errors": [], "packages": [], "files": []}
    try:
        data = read_manifest(path)
        result["role"] = data["role"]
        if not data.get("artifacts"):
            raise ValueError("artifacts require a complete pinned Debian dependency closure")
        names = set()
        for artifact in data["artifacts"]:
            if not re.fullmatch(r"[a-z0-9][a-z0-9+.-]+", artifact["package"]):
                raise ValueError("invalid package name")
            if artifact["package"] in names:
                raise ValueError("duplicate package")
            names.add(artifact["package"])
            if not re.fullmatch(r"[A-Za-z0-9.+:~_-]+", artifact.get("version") or ""):
                raise ValueError("every package requires an exact version")
            result["packages"].append(
                {"package": artifact["package"], "version": artifact["version"]}
            )
            if not re.fullmatch(r"[a-f0-9]{64}", artifact.get("sha256") or ""):
                raise ValueError("artifacts contain an unlocked/missing SHA256")
            package = source(path.parent, artifact["file"])
            if hashlib.sha256(package.read_bytes()).hexdigest() != artifact["sha256"]:
                raise ValueError(f"artifact checksum mismatch: {artifact['package']}")
        targets = set()
        for item in data.get("files", []):
            target = item["target"]
            if not target.startswith(TARGETS) or ".." in Path(target).parts or target in targets:
                raise ValueError("invalid/duplicate target path")
            targets.add(target)
            mode = int(item["mode"], 8)
            if mode & ~0o777:
                raise ValueError("unsupported file mode")
            file = source(path.parent, item["source"])
            if item.get("secret") and (file.stat().st_mode & 0o077 or mode & 0o077):
                raise ValueError("secret source/target permissions must exclude group/other")
            result["files"].append(
                {"target": target, "mode": item["mode"], "secret": bool(item.get("secret"))}
            )
        expected_mount = requested_shared_mount(path, data)
        if expected_mount is not None:
            result["shared_mount"] = expected_mount
        if data["role"].startswith("k8s"):
            expected = "/etc/kubernetes/" + (
                "init.yaml" if data["role"] == "k8s-controller" else "join.yaml"
            )
            if expected not in targets or "/etc/containerd/config.toml" not in targets:
                raise ValueError(
                    "Kubernetes bootstrap requires containerd config and a secret kubeadm init/join config"
                )
            for item in data["files"]:
                if item["target"] == expected and not item.get("secret"):
                    raise ValueError("kubeadm config must be marked secret")
        result["ready"] = True
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        result["errors"].append(str(error))
    return result


def execute(argv):
    # Never forward installer/join stdout/stderr: these may contain credentials.
    completed = subprocess.run(argv, text=True, capture_output=True, check=False, timeout=1800)
    if completed.returncode:
        raise RuntimeError(
            f"{Path(argv[0]).name} failed with exit {completed.returncode}; output withheld"
        )
    return completed.stdout.strip()


def installed_version(package, run):
    try:
        value = run(
            ["dpkg-query", "--show", "--showformat=${Version}\t${db:Status-Status}", package]
        )
        version, _, status = value.partition("\t")
        return version if status == "installed" else ""
    except RuntimeError:
        return ""


def write_file(path, content, mode):
    changed = (
        not path.exists() or path.read_bytes() != content or path.stat().st_mode & 0o777 != mode
    )
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent)
        try:
            os.fchmod(descriptor, mode)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return changed


def apply(path, root=Path("/"), execute=execute):
    preview = plan(path)
    if not preview["ready"]:
        raise ValueError("bootstrap inputs not ready: " + "; ".join(preview["errors"]))
    root = root.resolve()
    if root != Path("/") and execute is globals()["execute"]:
        raise ValueError("sandbox roots require an injected test executor")
    data = read_manifest(path)
    missing = []
    # Verify real Debian metadata before installing any package.
    for artifact in data["artifacts"]:
        package = source(path.parent, artifact["file"])
        metadata = execute(["dpkg-deb", "--field", str(package), "Package", "Version"]).splitlines()
        # dpkg-deb labels multiple fields; fixtures may return unlabelled values.
        metadata = [value.split(": ", 1)[-1] for value in metadata]
        if metadata != [artifact["package"], artifact["version"]]:
            raise ValueError(f"Debian metadata mismatch: {artifact['package']}")
        if installed_version(artifact["package"], execute) != artifact["version"]:
            missing.append(str(package))
    if missing:
        execute(["dpkg", "--install", *missing])
    for artifact in data["artifacts"]:
        if installed_version(artifact["package"], execute) != artifact["version"]:
            raise ValueError(f"installed package version mismatch: {artifact['package']}")
    changed = bool(missing)
    mount_activated = False

    def activate_mount():
        expected = preview.get("shared_mount")
        if expected is None:
            return
        observed = observed_shared_mount(root, execute)
        if observed is not None and observed != expected:
            raise SharedMountError(
                "shared mount SOURCE/FSTYPE mismatch; current mount left unchanged; no shared writes"
            )
        if observed is None:
            execute(["systemctl", "daemon-reload"])
            execute(["systemctl", "enable", "--now", "shared.mount"])
            observed = observed_shared_mount(root, execute)
            if observed != expected:
                raise SharedMountError(
                    "shared mount SOURCE/FSTYPE not verified after activation; no shared writes"
                )
        else:
            execute(["systemctl", "enable", "shared.mount"])

    for item in sorted(
        data.get("files", []), key=lambda item: item["target"].startswith("/shared/")
    ):
        if item["target"].startswith("/shared/"):
            # Recheck before every shared file, including subsequent payloads.
            activate_mount()
            mount_activated = True
        destination = root / item["target"].lstrip("/")
        if not destination.resolve().is_relative_to(root):
            raise ValueError("target symlink escapes root")
        changed |= write_file(
            destination, source(path.parent, item["source"]).read_bytes(), int(item["mode"], 8)
        )
        if item["target"] == "/etc/munge/munge.key":
            execute(["chown", "munge:munge", str(destination)])
    if not mount_activated:
        activate_mount()
    changed |= write_file(
        root / "etc/crucible/node-image-id", (data["node_image_id"] + "\n").encode(), 0o644
    )
    execute(["systemctl", "daemon-reload"])
    if data["role"].startswith("slurm"):
        for directory in ["/var/spool/slurmctld", "/var/spool/slurmd", "/var/log/slurm"]:
            execute(
                ["install", "-d", "-o", "slurm", "-g", "slurm", str(root / directory.lstrip("/"))]
            )
        services = ["munge", "slurmctld" if data["role"] == "slurm-controller" else "slurmd"]
    else:
        if changed:
            execute(["systemctl", "restart", "containerd"])
        controller = data["role"] == "k8s-controller"
        marker = root / (
            "etc/kubernetes/admin.conf" if controller else "etc/kubernetes/kubelet.conf"
        )
        if not marker.exists():
            execute(
                [
                    "kubeadm",
                    "init" if controller else "join",
                    "--config",
                    str(
                        root
                        / ("etc/kubernetes/init.yaml" if controller else "etc/kubernetes/join.yaml")
                    ),
                ]
            )
        services = ["containerd", "kubelet"]
    for service in services:
        execute(["systemctl", "enable", service])
        execute(["systemctl", "restart" if changed else "start", service])
    return {
        "applied": True,
        "changed": changed,
        "role": data["role"],
        "reboot_required": (root / "var/run/reboot-required").exists(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="explicit operator-only node mutation")
    args = parser.parse_args()
    preview = plan(args.manifest)
    if not args.apply:
        print(json.dumps(preview, indent=2))
        return 0 if preview["ready"] else 1
    if os.geteuid() != 0:
        parser.exit(1, "FAIL: --apply requires root on the target node\n")
    try:
        result = apply(args.manifest)
        print(json.dumps(result, indent=2))
        if result["reboot_required"]:
            return 2
    except SharedMountError as error:
        print(
            json.dumps(
                {"applied": False, "pending": "shared-mount-verification", "error": str(error)},
                indent=2,
            )
        )
        return 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError):
        # Error messages from child installers or secret paths are never exposed.
        parser.exit(1, "FAIL: node bootstrap failed; installer output withheld\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
