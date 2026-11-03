"""Explicit overlay-first entry point; native modes are used only under the GPU guard."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _name(name):
    return name.lower().replace("_", "-").replace(".", "-")


def inventory(lock, overlay, base):
    selected = {}
    for directory in (Path(overlay), Path(base)):
        for dist in importlib.metadata.distributions(path=[str(directory)]):
            name = dist.metadata.get("Name")
            if name:
                selected.setdefault(_name(name), (dist, directory))
    rows = []
    for expected in lock["packages"]:
        found = selected.get(expected["name"])
        if found is None:
            raise ValueError(f"inventory missing {expected['name']}")
        dist, directory = found
        origin = Path(overlay) if expected["origin"] == "overlay" else Path(base)
        digest = hashlib.sha256((dist._path / "METADATA").read_bytes()).hexdigest()
        if (
            dist.version != expected["version"]
            or directory != origin
            or digest != expected["metadata_sha256"]
        ):
            raise ValueError(f"inventory mismatch {expected['name']}")
        rows.append(
            {
                "name": expected["name"],
                "version": dist.version,
                "origin": str(directory),
                "metadata_sha256": digest,
            }
        )
    return rows


def runtime_environment(work):
    work = Path(work).resolve()
    hf = Path.home() / ".cache/huggingface"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "PIP_"))}
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        HF_HOME=str(hf),
        HF_HUB_CACHE=str(hf / "hub"),
        HUGGINGFACE_HUB_CACHE=str(hf / "hub"),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_DATASETS_CACHE=str(work / "datasets"),
        HF_MODULES_CACHE=str(work / "hf-modules"),
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_XET="1",
        HF_HUB_ENABLE_HF_TRANSFER="0",
        XDG_CACHE_HOME=str(work / "xdg"),
        TMPDIR=str(work / "tmp"),
        TRITON_CACHE_DIR=str(work / "triton"),
        CUDA_CACHE_PATH=str(work / "cuda"),
        CUDA_CACHE_MAXSIZE="134217728",
        TORCH_EXTENSIONS_DIR=str(work / "torch"),
        TORCHINDUCTOR_CACHE_DIR=str(work / "inductor"),
        TORCHINDUCTOR_COMPILE_THREADS="1",
        MAX_JOBS="1",
        UV_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        SGLANG_USE_MODELSCOPE="0",
    )
    env.pop("PYTHONPATH", None)
    env.pop("TRANSFORMERS_CACHE", None)
    return env


def validated_startup_path(overlay, audit):
    """Validate the two pinned .pth effects without evaluating any .pth code."""
    overlay = Path(overlay).resolve()
    hashes = {x["path"]: x["sha256"] for x in audit["source_files"]}
    for relative in (
        "nvidia_cutlass_dsl.pth",
        "_cuda_bindings_redirector.pth",
        "_cuda_bindings_redirector.py",
    ):
        path = overlay / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes.get(relative):
            raise ValueError("pinned startup source identity changed")
    entries = {}
    for name in ("nvidia_cutlass_dsl.pth", "_cuda_bindings_redirector.pth"):
        entries[name] = [
            line.strip()
            for line in (overlay / name).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    relative = "nvidia_cutlass_dsl/python_packages"
    if entries != {
        "nvidia_cutlass_dsl.pth": [relative],
        "_cuda_bindings_redirector.pth": ["import _cuda_bindings_redirector"],
    }:
        raise ValueError("unreviewed startup .pth content")
    directory = (overlay / relative).resolve()
    if not directory.is_relative_to(overlay) or not directory.is_dir():
        raise ValueError("pinned startup path missing or outside overlay")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--inventory-out", type=Path, required=True)
    parser.add_argument("--mode", choices=("abi", "server"), required=True)
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    base = ROOT / ".venv-ml/lib/python3.12/site-packages"
    lock = json.loads(args.lock.read_text())
    rows = inventory(lock, args.overlay.resolve(), base)
    audit = json.loads(args.cache_audit.read_text())
    # This artifact is created only after the installed pinned sources are audited.
    # It is evidence, not a substitute for the fixed guard or external authorization.
    if audit.get("lock_sha256") != hashlib.sha256(args.lock.read_bytes()).hexdigest():
        raise ValueError("cache source audit does not match runtime lock")
    if not audit.get("native_downloads_disabled") or not audit.get("source_files"):
        raise ValueError("native cache/download source audit incomplete")
    for item in audit["source_files"]:
        path = (args.overlay / item["path"]).resolve()
        if not path.is_relative_to(args.overlay.resolve()):
            raise ValueError("cache audit path escapes overlay")
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("audited cache source changed")
    cutlass_path = validated_startup_path(args.overlay, audit)
    env = runtime_environment(args.work)
    for key, relative in audit.get("environment", {}).items():
        if key in {"HOME", "PYTHONPATH", "PATH", "LD_LIBRARY_PATH"}:
            raise ValueError("cache audit cannot redirect runtime identity")
        path = (args.work / relative).resolve()
        if not path.is_relative_to(args.work.resolve()):
            raise ValueError("cache root escapes owned runtime")
        env[key] = str(path)
    for key, value in audit.get("disable_download_environment", {}).items():
        if key in {"HOME", "PATH", "PYTHONPATH", "LD_LIBRARY_PATH"}:
            raise ValueError("invalid download-disable variable")
        env[key] = value
    for value in env.values():
        if isinstance(value, str) and value.startswith(str(args.work.resolve()) + "/"):
            Path(value).mkdir(parents=True, exist_ok=True)
    os.environ.clear()
    os.environ.update(env)
    sys.path[:] = [str(args.overlay.resolve()), str(cutlass_path), str(base)] + [
        path for path in sys.path if "site-packages" not in path
    ]
    # Only the native entry invokes this audited, pure-Python namespace redirector.
    # Preparation validates its bytes above without executing it or arbitrary .pth.
    redirector = importlib.import_module("_cuda_bindings_redirector")
    if (
        Path(redirector.__file__).resolve()
        != args.overlay.resolve() / "_cuda_bindings_redirector.py"
    ):
        raise ValueError("startup module resolved outside overlay")
    result = {
        "inventory": rows,
        "python": sys.version,
        "executable": sys.executable,
        "lock_sha256": hashlib.sha256(args.lock.read_bytes()).hexdigest(),
        "cache_audit_sha256": hashlib.sha256(args.cache_audit.read_bytes()).hexdigest(),
        "python_sha256": _file_hash(sys.executable),
        "sys_path": sys.path,
    }
    if args.mode == "abi":
        for name in ("torch", "transformers", "sgl_kernel", "flashinfer", "sglang"):
            module = importlib.import_module(name)
            expected_origin = base if name == "torch" else args.overlay.resolve()
            if not Path(module.__file__).resolve().is_relative_to(expected_origin):
                raise ValueError(f"native import origin mismatch: {name}")
            result[name] = {
                "file": getattr(module, "__file__", None),
                "version": str(getattr(module, "__version__", "not exposed")),
            }
        torch = sys.modules["torch"]
        result["cuda"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        result["device"] = torch.cuda.get_device_name(0)
        result["nccl"] = torch.cuda.nccl.version()
        result["cudnn"] = torch.backends.cudnn.version()
        driver = Path("/proc/driver/nvidia/version")
        result["driver"] = driver.read_text() if driver.exists() else "not exposed"
        result["ofed"] = "not used by this single-GPU native ABI probe"
        libraries = set()
        for line in Path("/proc/self/maps").read_text().splitlines():
            parts = line.split(maxsplit=5)
            if len(parts) == 6 and parts[5].startswith("/") and ".so" in parts[5]:
                path = Path(parts[5])
                if path.is_file():
                    libraries.add(path.resolve())
        result["loaded_libraries"] = [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _file_hash(path)}
            for path in sorted(libraries)
        ]
    args.inventory_out.write_text(json.dumps(result, indent=2) + "\n")
    if args.mode == "server":
        server_args = args.server_args
        if server_args[:1] == ["--"]:
            server_args = server_args[1:]
        sys.argv = ["sglang.launch_server", *server_args]
        runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
