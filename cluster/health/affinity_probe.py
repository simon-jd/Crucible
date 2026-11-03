#!/usr/bin/env python3
"""Read-only site GPU/HCA identity, active port and NUMA verification."""

import argparse
import json
import subprocess
from pathlib import Path


def probe(node, sysroot=Path("/sys"), gpu_listing=None):
    if gpu_listing is None:
        gpu_listing = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"], text=True
        )
    actual = {}
    for line in gpu_listing.splitlines():
        index, bus = [s.strip() for s in line.split(",")]
        # nvidia-smi prints an eight-digit domain; Linux sysfs uses four.
        actual[int(index)] = bus.lower()[-12:]
    expected = {g["index"]: g["pci_bus_id"].lower() for g in node["gpus"]}
    if actual != expected:
        raise ValueError("actual GPU indices/PCI IDs differ from site inventory")
    observations = []
    if {p.name for p in (sysroot / "class/infiniband").iterdir()} != {
        h["name"] for h in node["hcas"]
    }:
        raise ValueError("actual HCA inventory differs from reviewed site inventory")
    gpu_numa = {}
    for gpu in node["gpus"]:
        path = sysroot / "bus/pci/devices" / gpu["pci_bus_id"].lower() / "numa_node"
        numa = int(path.read_text())
        if numa < 0 or numa != gpu["numa_node"]:
            raise ValueError("actual GPU NUMA differs or is unknown")
        gpu_numa[gpu["index"]] = numa
        observations.append({"gpu": gpu["index"], "numa_node": numa})
    for hca in node["hcas"]:
        device = sysroot / "class/infiniband" / hca["name"]
        numa = int((device / "device/numa_node").read_text())
        if numa < 0 or numa != hca["numa_node"] or numa != gpu_numa[hca["gpu_index"]]:
            raise ValueError("actual HCA/GPU NUMA mismatch or unknown")
        port = device / "ports" / str(hca["port"])
        if (port / "state").read_text().strip() != "4: ACTIVE":
            raise ValueError("HCA port inactive")
        # Measured active rate is independent from the supplied theoretical rate.
        rate = float((port / "rate").read_text().split()[0])
        if hca.get("link_gbps") is None or rate < hca["link_gbps"]:
            raise ValueError("active HCA link rate below site theory or theory missing")
        observations.append({"hca": hca["name"], "numa_node": numa, "active_rate_gbps": rate})
    return {"node": node["name"], "passed": True, "observations": observations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--node", required=True)
    args = parser.parse_args()
    inv = json.loads(Path(args.inventory).read_text())
    node = next(n for n in inv["nodes"] if n["name"] == args.node)
    try:
        result = probe(node)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result = {"node": args.node, "passed": False, "error": str(exc)}
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
