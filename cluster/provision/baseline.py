#!/usr/bin/env python3
"""Fail-closed cluster acceptance, with raw command evidence even on failures."""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLURM_LIST = ["srun", "-N4", "--ntasks-per-node=8", "--gpus-per-node=8", "nvidia-smi", "-L"]


class BaselineError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise BaselineError(message)


def gpu_ids(text):
    return set(re.findall(r"UUID: (GPU-[A-Za-z0-9-]+)\)", text))


def check_gpu_listing(text):
    cards = gpu_ids(text)
    require(len(cards) == 32, f"expected 32 unique physical GPU UUIDs; observed {len(cards)}")
    require("MIG " not in text, "MIG devices are not full H100 GPUs")
    return cards


def validate_lock(lock, scheduler):
    require(isinstance(lock, dict), "lock must be an object")
    image = lock.get("image", {})
    require(isinstance(image, dict), "image lock must be an object")
    require(
        bool(re.fullmatch(r"sha256:[a-f0-9]{64}", image.get("digest") or "")),
        "image digest is missing/unlocked (must be sha256:<64 hex>)",
    )
    node_image = lock.get("node_image", {})
    require(
        bool(node_image.get("id"))
        and bool(re.fullmatch(r"[a-f0-9]{64}", node_image.get("sha256") or "")),
        "node image id/digest is missing/unlocked",
    )
    if scheduler == "k8s":
        require(
            bool(lock.get("k8s_images"))
            and all(
                re.fullmatch(r"sha256:[a-f0-9]{64}", d or "") for d in lock["k8s_images"].values()
            ),
            "operator image digest inventory is missing/unlocked",
        )
    require(lock.get("status") == "locked", "versions.lock status must be locked")
    require(lock.get("schema") == 1, "unsupported versions.lock schema")
    require(
        lock.get("topology") == {"nodes": 4, "gpus_per_node": 8, "gpu_model_contains": "H100"},
        "topology must be 4 x 8 H100",
    )
    versions = lock.get("versions", {})
    required = ["driver", "cuda_toolkit", "cuda_runtime", "nccl", "ofed"]
    required += (
        ["slurm", "enroot", "pyxis"]
        if scheduler == "slurm"
        else ["kubernetes", "gpu_operator", "network_operator", "kueue"]
    )
    for key in required:
        value = versions.get(key)
        if key in ("cuda_runtime", "nccl"):
            require(type(value) is int and value > 0, f"unlocked version: {key}")
        else:
            require(
                isinstance(value, str)
                and bool(re.fullmatch(r"[A-Za-z0-9_.+-]*\d[A-Za-z0-9_.+-]*", value))
                and not any(word in value.lower() for word in ["latest", "todo", "unknown"]),
                f"unlocked version: {key}",
            )
    require(
        bool(re.fullmatch(r"[a-zA-Z0-9./_-]+", image.get("repository", ""))),
        "image repository is missing/invalid",
    )
    require(
        lock.get("shared_filesystems")
        and set(lock["shared_filesystems"]) <= {"nfs", "nfs4", "lustre", "beegfs", "ceph"},
        "shared filesystem allowlist is invalid",
    )
    if scheduler == "slurm":
        require(
            bool(re.fullmatch(r"[a-f0-9]{64}", image.get("slurm_sqsh_sha256") or "")),
            "Slurm squashfs digest is missing/unlocked",
        )
        require(
            image.get("slurm_sqsh_path", "").startswith("/shared/"), "squashfs must be on /shared"
        )


class Runner:
    def __init__(self, output, timeout):
        self.output, self.timeout, self.commands = output, timeout, []
        self.partial = {"hosts": [], "runtime": []}

    def run(self, argv, check=True):
        record = {"argv": argv, "started_utc": datetime.now(UTC).isoformat()}
        try:
            process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
                record.update(
                    returncode=124,
                    stdout=stdout,
                    stderr=stderr + f"\ncommand timeout after {self.timeout}s",
                )
            else:
                record.update(returncode=process.returncode, stdout=stdout, stderr=stderr)
        except OSError as error:
            record.update(returncode=127, stdout="", stderr=f"{type(error).__name__}: {error}")
        self.commands.append(record)
        (self.output / "commands.json").write_text(json.dumps(self.commands, indent=2) + "\n")
        require(
            not check or record["returncode"] == 0,
            f"command failed ({record['returncode']}): {' '.join(argv)}: {(record['stderr'] or record['stdout']).strip()}",
        )
        return record["stdout"]

    def collect(self, argv, kind):
        # Parse and persist partial success before interpreting the aggregate exit.
        text = self.run(argv, check=False)
        found, errors = [], []
        for line in text.splitlines():
            if line.startswith("CRUCIBLE_PROBE="):
                try:
                    report = json.loads(line.removeprefix("CRUCIBLE_PROBE="))
                    require(
                        isinstance(report, dict) and report.get("kind") == kind,
                        "invalid probe kind",
                    )
                    found.append(report)
                except (ValueError, BaselineError) as error:
                    errors.append(str(error))
        self.partial["hosts" if kind == "host" else "runtime"].extend(found)
        (self.output / "partial-reports.json").write_text(json.dumps(self.partial, indent=2) + "\n")
        archive_hosts(self.partial["hosts"], self.output)
        require(
            self.commands[-1]["returncode"] == 0,
            f"probe command failed ({self.commands[-1]['returncode']}): {' '.join(argv)}",
        )
        require(not errors and bool(found), "invalid/missing probe reports: " + "; ".join(errors))
        require(
            not any(report.get("errors") for report in found), "probe reported diagnostic errors"
        )
        return found


def reports(text):
    result = [
        json.loads(line.removeprefix("CRUCIBLE_PROBE="))
        for line in text.splitlines()
        if line.startswith("CRUCIBLE_PROBE=")
    ]
    require(bool(result), "probe emitted no reports")
    return result


def archive_hosts(hosts, output):
    # Archive every captured host before any version check can stop validation.
    for index, host in enumerate(hosts):
        (output / f"node-{index}-report.json").write_text(json.dumps(host, indent=2) + "\n")
        for field, suffix in [("topology", "topology"), ("ibstat", "ibstat")]:
            if isinstance(host.get(field), str):
                (output / f"node-{index}-{suffix}.txt").write_text(host[field] + "\n")


def validate_hosts(hosts, lock, scheduler, token, output):
    require(
        len(hosts) == 4 and len({host["node"] for host in hosts}) == 4,
        "expected four distinct host reports",
    )
    expected_probe = hashlib.sha256((HERE / "node_probe.py").read_bytes()).hexdigest()
    versions, all_cards = lock["versions"], set()
    for host in hosts:
        require(host["node_image_id"] == lock["node_image"]["id"], "node image identity mismatch")
        require(
            host["kind"] == "host" and host["probe_sha256"] == expected_probe, "stale host probe"
        )
        require(bool(host["topology"].strip()), f"empty GPU topology: {host['node']}")
        require(
            bool(re.search(r"State:\s*Active\b", host["ibstat"])),
            f"no Active IB/RoCE port: {host['node']}",
        )
        require(host["ofed"] == versions["ofed"], f"OFED version mismatch: {host['node']}")
        require(
            host["shared_fstype"] in lock["shared_filesystems"] and host["shared_token"] == token,
            f"shared storage unavailable or inconsistent: {host['node']}",
        )
        cards = host["gpus"]
        ids = {card["uuid"] for card in cards}
        require(
            len(cards) == len(ids) == 8 and not ids & all_cards,
            f"expected eight distinct GPUs per host: {host['node']}",
        )
        for card in cards:
            require(bool(re.fullmatch(r"GPU-[A-Za-z0-9-]+", card["uuid"])), "invalid GPU UUID")
            require("H100" in card["model"], f"expected H100: {host['node']}")
            require(
                card["driver"] == versions["driver"], f"driver version mismatch: {host['node']}"
            )
        all_cards.update(ids)
        if scheduler == "slurm":
            for key in ["slurm", "enroot", "pyxis"]:
                require(host[key] == versions[key], f"{key} version mismatch: {host['node']}")
            require(
                host["image_sha256"] == lock["image"]["slurm_sqsh_sha256"],
                f"squashfs digest mismatch: {host['node']}",
            )
    return all_cards


def validate_runtimes(runtime, hosts, lock, scheduler):
    require(len(runtime) == (4 if scheduler == "slurm" else 32), "wrong runtime report count")
    expected_probe = hashlib.sha256((HERE / "node_probe.py").read_bytes()).hexdigest()
    by_node = {host["node"]: {card["uuid"] for card in host["gpus"]} for host in hosts}
    seen = set()
    for report in runtime:
        require(
            report["kind"] == "runtime" and report["probe_sha256"] == expected_probe,
            "stale runtime probe",
        )
        require(report["node"] in by_node, "runtime is on an unexpected node")
        cards = set(report["uuids"])
        require(
            len(cards) == len(report["uuids"]) == (8 if scheduler == "slurm" else 1),
            "wrong GPU visibility in runtime",
        )
        require(
            cards <= by_node[report["node"]] and not cards & seen,
            "duplicated/misplaced runtime GPU",
        )
        seen.update(cards)
        for key in ["cuda_toolkit", "cuda_runtime", "nccl"]:
            require(
                report[key] == lock["versions"][key], f"{key} version mismatch: {report['node']}"
            )
    require(len(seen) == 32, "runtimes did not cover 32 distinct GPUs")


def slurm(runner, args, lock, token):
    allocation = ["srun", "-N4", "--ntasks-per-node=1", "--gpus-per-node=8"]
    host_command = [
        "python3",
        args.probe_path,
        "host",
        "--scheduler",
        "slurm",
        "--shared-root",
        args.shared_root,
        "--token",
        args.token_file,
        "--image",
        lock["image"]["slurm_sqsh_path"],
    ]
    hosts = runner.collect(allocation + host_command, "host")
    archive_hosts(hosts, args.output)
    runtime = runner.collect(
        allocation
        + [
            "--container-image=" + lock["image"]["slurm_sqsh_path"],
            f"--container-mounts={args.probe_path}:/probe.py:ro",
            "--no-container-entrypoint",
            "python3",
            "/probe.py",
            "runtime",
        ],
        "runtime",
    )
    return hosts, runtime


def checked_pods(data, count, digest):
    pods = data["items"]
    require(len(pods) == count, f"expected {count} GPU/host pods; observed {len(pods)}")
    for pod in pods:
        require(not pod["metadata"].get("deletionTimestamp"), "terminating baseline pod")
        status = pod["status"]
        require(
            status["phase"] == "Running"
            and any(
                c["type"] == "Ready" and c["status"] == "True" for c in status.get("conditions", [])
            ),
            "baseline pod is not Running/Ready",
        )
        containers = status.get("containerStatuses", [])
        require(
            len(containers) == 1
            and containers[0]["ready"]
            and containers[0]["imageID"].endswith("@" + digest),
            "pod image digest mismatch/unavailable",
        )
    return pods


def k8s(runner, args, lock, token):
    base = ["kubectl", "-n", args.namespace]
    version = json.loads(runner.run(["kubectl", "version", "-o", "json"]))
    require(
        version["serverVersion"]["gitVersion"] == lock["versions"]["kubernetes"],
        "Kubernetes version mismatch",
    )
    releases = json.loads(runner.run(["helm", "list", "-A", "-o", "json"]))
    for name, component, chart in [
        ("gpu-operator", "gpu_operator", "gpu-operator"),
        ("network-operator", "network_operator", "network-operator"),
        ("kueue", "kueue", "kueue"),
    ]:
        expected = f"{chart}-{lock['versions'][component]}"
        require(
            sum(
                r["name"] == name and r["chart"] == expected and r["status"] == "deployed"
                for r in releases
            )
            == 1,
            f"missing/mismatched Helm release: {name}",
        )
    for namespace in ["nvidia-gpu-operator", "nvidia-network-operator", "kueue-system"]:
        managed = json.loads(runner.run(["kubectl", "-n", namespace, "get", "pods", "-o", "json"]))[
            "items"
        ]
        require(bool(managed), f"operator namespace has no pods: {namespace}")
        for pod in managed:
            status = pod["status"]
            require(status["phase"] in ["Running", "Succeeded"], "operator pod is unhealthy")
            if status["phase"] == "Running":
                require(
                    all(c.get("ready", False) for c in status.get("containerStatuses", [])),
                    "operator pod is not ready",
                )
            statuses = pod["status"].get("containerStatuses", []) + pod["status"].get(
                "initContainerStatuses", []
            )
            require(bool(statuses), "operator container image identity unavailable")
            for container in statuses:
                digest = lock["k8s_images"].get(container["image"])
                require(
                    bool(digest) and container["imageID"].endswith("@" + digest),
                    f"operator container image digest missing/mismatched: {container['image']}",
                )
    gpu = checked_pods(
        json.loads(runner.run(base + ["get", "pods", "-l", "component=gpu", "-o", "json"])),
        32,
        lock["image"]["digest"],
    )
    require(
        sorted(Counter(p["spec"]["nodeName"] for p in gpu).values()) == [8] * 4,
        "GPU pods must be distributed 8 per node across four nodes",
    )
    for pod in gpu:
        require(
            len(pod["spec"]["containers"]) == 1
            and str(pod["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"]) == "1",
            "each GPU pod must request one GPU",
        )
    host_pods = checked_pods(
        json.loads(runner.run(base + ["get", "pods", "-l", "component=host", "-o", "json"])),
        4,
        lock["image"]["digest"],
    )
    require(
        {p["spec"]["nodeName"] for p in host_pods} == {p["spec"]["nodeName"] for p in gpu},
        "host/GPU nodes differ",
    )
    hosts, runtime = [], []
    for pods, kind, destination in [(host_pods, "host", hosts), (gpu, "runtime", runtime)]:
        for pod in pods:
            cmd = ["python3", "/probe/node_probe.py", kind, "--scheduler", "k8s"]
            if kind == "host":
                cmd += [
                    "--host-root",
                    "--shared-root",
                    args.shared_root,
                    "--token",
                    args.token_file,
                ]
            report = runner.collect(base + ["exec", pod["metadata"]["name"], "--"] + cmd, kind)
            require(
                len(report) == 1 and report[0]["node"] == pod["spec"]["nodeName"],
                "pod reported wrong node",
            )
            destination.extend(report)
        if kind == "host":
            archive_hosts(hosts, args.output)
    return hosts, runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler", choices=["slurm", "k8s"], default="slurm")
    parser.add_argument("--lock", type=Path, default=HERE / "versions.lock")
    parser.add_argument(
        "--output", type=Path, required=True, help="new evidence directory (never overwritten)"
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--namespace", default="crucible-baseline")
    parser.add_argument("--shared-root", default="/shared")
    parser.add_argument("--probe-path", default="/shared/crucible/cluster/provision/node_probe.py")
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 1800:
        parser.error("timeout must be >0 and <=1800 seconds")
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"FAIL: cannot create new evidence directory: {error}", file=sys.stderr)
        return 1
    runner = Runner(args.output, args.timeout)
    errors, cards = [], None
    # Always attempt the real scheduler first; unfinished locks cannot hide absent tools.
    try:
        if args.scheduler == "slurm":
            cards = check_gpu_listing(runner.run(SLURM_LIST))
        else:
            runner.run(
                [
                    "kubectl",
                    "-n",
                    args.namespace,
                    "get",
                    "pods",
                    "-l",
                    "component=gpu",
                    "-o",
                    "json",
                ]
            )
    except BaselineError as error:
        errors.append(str(error))
    lock = None
    try:
        lock = json.loads(args.lock.read_text())
        (args.output / "versions.lock").write_text(json.dumps(lock, indent=2) + "\n")
        validate_lock(lock, args.scheduler)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        IndexError,
        BaselineError,
    ) as error:
        errors.append(f"lock validation: {error}")
    args.token_file = f".crucible-baseline-{uuid.uuid4().hex}"
    token_path = Path(args.shared_root) / args.token_file
    if not errors:
        try:
            token = uuid.uuid4().hex
            token_path.write_text(token)
            hosts, runtime = (slurm if args.scheduler == "slurm" else k8s)(
                runner, args, lock, token
            )
            (args.output / "reports.json").write_text(
                json.dumps({"hosts": hosts, "runtime": runtime}, indent=2) + "\n"
            )
            actual = validate_hosts(hosts, lock, args.scheduler, token, args.output)
            if cards is not None:
                require(actual == cards, "listing and per-host GPU UUIDs differ")
            validate_runtimes(runtime, hosts, lock, args.scheduler)
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            IndexError,
            BaselineError,
        ) as error:
            errors.append(f"acceptance: {error}")
        finally:
            try:
                token_path.unlink(missing_ok=True)
            except OSError as error:
                errors.append(f"shared token cleanup: {error}")
    summary = {
        "passed": not errors,
        "scheduler": args.scheduler,
        "errors": errors,
        "captured_utc": datetime.now(UTC).isoformat(),
        "scope": "real-command execution; synthetic subprocesses only in explicitly labelled tests",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
