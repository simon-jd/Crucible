"""Smoke payload: fetch one asset, report the image and GPUs. Usage: --args <asset name>."""
import importlib.metadata as md
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root inside the job
from scripts.snowflake.assets import fetch


def ver(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


try:
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                          check=False, capture_output=True, text=True).stdout.strip().splitlines()
except FileNotFoundError:
    gpus = []
got = fetch(sys.argv[1], "/tmp/asset") if len(sys.argv) > 1 else None
files = sorted(p.name for p in Path("/tmp/asset").rglob("*") if p.is_file())[:20] if got else []
print("RESULT " + json.dumps({"python": sys.version.split()[0], "gpus": gpus, "fetch": got, "files": files,
                              "packages": {p: ver(p) for p in ("torch", "vllm", "transformers", "ray")}}), flush=True)
