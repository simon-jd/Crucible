#!/usr/bin/env python3
"""Bind each OpenMPI local rank to its reviewed GPU/NUMA mapping before NCCL."""

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path


def rank_token(prefix, node, local_rank):
    return hashlib.sha256(f"{prefix}:{node}:{local_rank}".encode()).hexdigest()[:32]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--token-prefix", required=True)
    split = sys.argv.index("--")
    args = parser.parse_args(sys.argv[1:split])
    command = sys.argv[split + 1 :]
    inv = json.loads(Path(args.inventory).read_text())
    hostname = socket.gethostname().split(".")[0]
    nodes = [
        n for n in inv["nodes"] if hostname in [n["name"].split(".")[0], n["address"].split(".")[0]]
    ]
    if len(nodes) != 1 or not command:
        raise ValueError("rank host not uniquely present in inventory, or command missing")
    node = nodes[0]
    rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    gpu = next(g for g in node["gpus"] if g["index"] == rank)
    numa = int((Path("/sys/bus/pci/devices") / gpu["pci_bus_id"].lower() / "numa_node").read_text())
    if numa < 0 or numa != gpu["numa_node"]:
        raise ValueError("live GPU NUMA mismatch")
    hcas = [h for h in node["hcas"] if h["gpu_index"] == rank]
    if not hcas:
        raise ValueError("rank has no reviewed HCA mapping")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["NCCL_IB_HCA"] = "=" + ",".join(f"{h['name']}:{h['port']}" for h in hcas)
    if time.time() >= args.deadline_epoch:
        raise ValueError("absolute rank deadline reached")
    argv = [
        "python3",
        str(Path(__file__).with_name("remote_worker.py")),
        "run",
        "--token",
        rank_token(args.token_prefix, node["name"], rank),
        "--deadline-epoch",
        str(args.deadline_epoch),
        "--",
        "numactl",
        f"--cpunodebind={numa}",
        f"--membind={numa}",
    ] + command
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
