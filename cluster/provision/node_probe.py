#!/usr/bin/env python3
"""Runs on a real worker (host or workload image), never generates observations."""

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path


def run(argv, data=None):
    record = {"argv": argv}
    try:
        result = subprocess.run(argv, text=True, capture_output=True, timeout=30, check=False)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        record.update(
            returncode=124 if isinstance(error, subprocess.TimeoutExpired) else 127,
            stdout=str(getattr(error, "stdout", "") or ""),
            stderr=str(error),
        )
    if data is not None:
        data["commands"].append(record)
    if record["returncode"]:
        raise RuntimeError(json.dumps(record))
    return record["stdout"].strip()


def library_version(name, symbol):
    library = ctypes.CDLL(ctypes.util.find_library(name) or f"lib{name}.so")
    function = getattr(library, symbol)
    function.argtypes = [ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    value = ctypes.c_int()
    code = function(ctypes.byref(value))
    if code != 0:
        raise RuntimeError(f"{symbol} failed: {code}")
    return value.value


def probe(args):
    data = {
        "node": os.environ.get("NODE_NAME", socket.gethostname()),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kind": args.kind,
        "errors": [],
        "commands": [],
    }

    def capture(key, function):
        try:
            data[key] = function()
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            RuntimeError,
            subprocess.SubprocessError,
        ) as error:
            data["errors"].append({"field": key, "error": f"{type(error).__name__}: {error}"})

    def command(argv):
        return run(argv, data)

    if args.kind == "host":
        capture(
            "gpus",
            lambda: [
                dict(zip(("uuid", "model", "driver"), map(str.strip, row.split(","))))
                for row in command(
                    ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"]
                ).splitlines()
            ],
        )
        capture("topology", lambda: command(["nvidia-smi", "topo", "-m"]))
        capture("ibstat", lambda: command(["ibstat"]))
        host = ["chroot", "/host"] if args.host_root else []
        capture("ofed", lambda: command(host + ["ofed_info", "-s"]).rstrip(":"))
        capture("node_image_id", lambda: command(host + ["cat", "/etc/crucible/node-image-id"]))
        capture(
            "shared_fstype",
            lambda: command(["findmnt", "-n", "-o", "FSTYPE", "--target", args.shared_root]),
        )
        capture("shared_token", lambda: (Path(args.shared_root) / args.token).read_text())
        if args.scheduler == "slurm":
            capture("slurm", lambda: command(["srun", "--version"]).removeprefix("slurm "))
            capture("enroot", lambda: command(["enroot", "version"]))
            capture(
                "pyxis",
                lambda: command(
                    ["dpkg-query", "--show", "--showformat=${Version}", "nvslurm-plugin-pyxis"]
                ),
            )

            def image_digest():
                with open(args.image, "rb") as image:
                    return hashlib.file_digest(image, "sha256").hexdigest()

            capture("image_sha256", image_digest)
    else:
        capture(
            "uuids",
            lambda: command(
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]
            ).splitlines(),
        )
        capture(
            "cuda_toolkit",
            lambda: json.loads(Path("/usr/local/cuda/version.json").read_text())["cuda"]["version"],
        )
        capture("cuda_runtime", lambda: library_version("cudart", "cudaRuntimeGetVersion"))
        capture("nccl", lambda: library_version("nccl", "ncclGetVersion"))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["host", "runtime"])
    parser.add_argument("--scheduler", choices=["slurm", "k8s"], default="slurm")
    parser.add_argument("--host-root", action="store_true")
    parser.add_argument("--shared-root", default="/shared")
    parser.add_argument("--token", default="")
    parser.add_argument("--image", default="/shared/images/crucible.sqsh")
    args = parser.parse_args()
    data = probe(args)
    print("CRUCIBLE_PROBE=" + json.dumps(data, separators=(",", ":")), flush=True)
    return 1 if data["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
