"""Offline 1/2/4-node all-subset matrix; future OpenMPI site execution commands."""

import re
import shlex
from itertools import combinations
from pathlib import PurePosixPath

COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "alltoall")


def validate_inventory(inv):
    nodes = inv.get("nodes", [])
    if inv.get("schema") != 1 or len(nodes) != 4:
        raise ValueError("schema 1 and exactly four site nodes required")
    names = set()
    for node in nodes:
        name = node.get("name", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in names:
            raise ValueError("unique safe node names required")
        names.add(name)
        for key in ["ssh", "address"]:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9@_.:-]*", node.get(key, "")):
                raise ValueError(f"invalid {key}")
        gpus = node.get("gpus", [])
        hcas = node.get("hcas", [])
        indices = {g.get("index") for g in gpus}
        if (
            not gpus
            or not hcas
            or indices != set(range(len(gpus)))
            or any(type(g.get("index")) is not int for g in gpus)
        ):
            raise ValueError("contiguous GPU indices and HCA inventory required")
        if len({g.get("pci_bus_id", "").lower() for g in gpus}) != len(gpus):
            raise ValueError("unique GPU PCI bus IDs required")
        if len({h["name"] for h in hcas}) != len(hcas):
            raise ValueError("unique HCA names required")
        if {h.get("gpu_index") for h in hcas} != indices:
            raise ValueError("every GPU requires an HCA mapping for rank affinity")
        for gpu in gpus:
            if not re.fullmatch(
                r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", gpu.get("pci_bus_id", "")
            ):
                raise ValueError("PCI bus id required")
            if type(gpu.get("numa_node")) is not int or gpu["numa_node"] < 0:
                raise ValueError("known GPU NUMA required")
        for hca in hcas:
            if (
                not re.fullmatch(r"[A-Za-z0-9_]+", hca.get("name", ""))
                or hca.get("gpu_index") not in indices
                or type(hca.get("port")) is not int
                or hca["port"] < 1
                or type(hca.get("numa_node")) is not int
                or hca["numa_node"] < 0
            ):
                raise ValueError("valid HCA port, GPU mapping and NUMA required")
            gpu = next(g for g in gpus if g["index"] == hca["gpu_index"])
            if gpu["numa_node"] != hca["numa_node"]:
                raise ValueError("HCA/GPU NUMA mismatch")
    if len({len(n["gpus"]) for n in nodes}) != 1:
        raise ValueError("uniform GPU count required by ppr mapping")
    if len({n["address"] for n in nodes}) != 4 or len({n["ssh"] for n in nodes}) != 4:
        raise ValueError("distinct site addresses and SSH targets required")
    if len({n["ssh"].split("@")[0] if "@" in n["ssh"] else None for n in nodes}) != 1:
        raise ValueError("OpenMPI backend requires the same SSH user on every node")
    for key in ["shared_root", "nccl_bin", "remote_inventory"]:
        path = inv.get(key, "")
        if (
            not isinstance(path, str)
            or not path.startswith("/")
            or ".." in PurePosixPath(path).parts
        ):
            raise ValueError(f"absolute site path required: {key}")


def ssh(node, command):
    return [
        "ssh",
        "-oBatchMode=yes",
        "-oStrictHostKeyChecking=yes",
        node["ssh"],
        shlex.join(command),
    ]


def build_matrix(inv):
    validate_inventory(inv)
    matrix = []
    gpus = len(inv["nodes"][0]["gpus"])
    mpi_ssh = ["ssh", "-oBatchMode=yes", "-oStrictHostKeyChecking=yes"]
    if "@" in inv["nodes"][0]["ssh"]:
        mpi_ssh += ["-l", inv["nodes"][0]["ssh"].split("@")[0]]
    for count in [1, 2, 4]:
        for subset in combinations(inv["nodes"], count):
            names = [n["name"] for n in subset]
            for collective in COLLECTIVES:
                key = collective + ":" + ",".join(names)
                argv = [
                    "mpirun",
                    "-np",
                    str(count * gpus),
                    "--host",
                    ",".join(f"{n['address']}:{gpus}" for n in subset),
                    "--map-by",
                    f"ppr:{gpus}:node",
                    "--bind-to",
                    "none",
                    "--mca",
                    "plm_rsh_agent",
                    shlex.join(mpi_ssh),
                    "python3",
                    inv["shared_root"] + "/cluster/health/rank_exec.py",
                    "--inventory",
                    inv["remote_inventory"],
                    "--",
                    inv["nccl_bin"] + "/" + collective + "_perf",
                    "-b",
                    "134217728",
                    "-e",
                    "1073741824",
                    "-f",
                    "2",
                    "-g",
                    "1",
                    "-w",
                    "5",
                    "-n",
                    "20",
                    "-c",
                    "1",
                ]
                matrix.append(
                    {
                        "id": "nccl:" + key,
                        "nodes": names,
                        "collective": collective,
                        "theory_key": key,
                        "argv": argv,
                        "expected_sizes": [134217728, 268435456, 536870912, 1073741824],
                    }
                )
    return matrix
