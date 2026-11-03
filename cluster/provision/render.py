#!/usr/bin/env python3
"""Render reviewed site configuration; never installs packages or contacts a cluster."""

import argparse
import json
import re
import shutil
from pathlib import Path

from baseline import HERE, BaselineError, require, validate_lock


def render(args):
    lock = json.loads(args.lock.read_text())
    validate_lock(lock, args.scheduler)
    require(
        bool(re.fullmatch(r"[A-Za-z0-9.-]+:/[A-Za-z0-9/_.-]+", args.nfs_export)),
        "provide NFS_SERVER:/export/path",
    )
    server, path = args.nfs_export.split(":", 1)
    replacements = {
        "NFS_EXPORT": args.nfs_export,
        "NFS_SERVER": server,
        "NFS_PATH": path,
        "IMAGE": lock["image"]["repository"] + "@" + lock["image"]["digest"],
        "LOCK_JSON": json.dumps(lock),
    }
    if args.scheduler == "slurm":
        require(
            bool(args.nodes)
            and len(args.nodes.split(",")) == 4
            and len(set(args.nodes.split(","))) == 4
            and all(re.fullmatch(r"[a-zA-Z0-9.-]+", n) for n in args.nodes.split(",")),
            "provide exactly four distinct comma-separated worker hostnames",
        )
        require(
            bool(re.fullmatch(r"[A-Za-z0-9.-]+", args.controller or "")),
            "provide controller hostname",
        )
        require(args.cpus > 0 and args.memory_mb > 0, "provide CPUs/RealMemory from slurmd -C")
        replacements.update(
            CONTROLLER=args.controller,
            NODES=args.nodes,
            CPUS=str(args.cpus),
            MEMORY_MB=str(args.memory_mb),
        )
    else:
        require(
            bool(re.fullmatch(r"[A-Za-z0-9_.-]+", args.rdma_interface or "")),
            "provide RDMA interface",
        )
        replacements["RDMA_INTERFACE"] = args.rdma_interface
    args.output.mkdir(parents=True, exist_ok=False)
    for source in [HERE / "cloud-init.yaml", *(HERE / args.scheduler).glob("*")]:
        if not source.is_file():
            continue
        text = source.read_text()
        for key, value in replacements.items():
            text = text.replace("@" + key + "@", value)
        require(not re.search(r"@[A-Z_]+@", text), f"unresolved placeholder: {source}")
        (args.output / source.name).write_text(text)
    shutil.copyfile(args.lock, args.output / "versions.lock")
    shutil.copyfile(HERE / "node_probe.py", args.output / "node_probe.py")
    mount = "\n".join(
        [
            "[Unit]",
            "Description=Crucible shared experiment storage",
            "Wants=network-online.target",
            "After=network-online.target",
            "[Mount]",
            "What=" + args.nfs_export,
            "Where=/shared",
            "Type=nfs4",
            "Options=rw,hard,_netdev,vers=4.1",
            "[Install]",
            "WantedBy=multi-user.target",
            "",
        ]
    )
    (args.output / "shared.mount").write_text(mount)
    if args.scheduler == "k8s":
        (args.output / "namespace.yaml").write_text(
            "apiVersion: v1\nkind: Namespace\nmetadata:\n  name: crucible-baseline\n"
        )
        (args.output / "runtimeclass.yaml").write_text(
            "apiVersion: node.k8s.io/v1\nkind: RuntimeClass\nmetadata:\n  name: nvidia\nhandler: nvidia\n"
        )
        # JSON is also valid YAML; ConfigMap data preserves the exact probe bytes.
        config = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": "node-probe", "namespace": "crucible-baseline"},
            "data": {"node_probe.py": (HERE / "node_probe.py").read_text()},
        }
        (args.output / "probe-configmap.json").write_text(json.dumps(config, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler", choices=["slurm", "k8s"], required=True)
    parser.add_argument("--lock", type=Path, default=HERE / "versions.lock")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nfs-export", required=True)
    parser.add_argument("--controller")
    parser.add_argument("--nodes")
    parser.add_argument("--cpus", type=int, default=0)
    parser.add_argument("--memory-mb", type=int, default=0)
    parser.add_argument("--rdma-interface")
    args = parser.parse_args()
    try:
        render(args)
    except (OSError, ValueError, TypeError, BaselineError) as error:
        parser.exit(1, f"FAIL: {error}\n")
    print(f"Rendered {args.scheduler} configuration: {args.output}")


if __name__ == "__main__":
    main()
