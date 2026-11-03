"""Explicit process-entry cache containment, called before Torch/model imports."""

import os
import tempfile
from pathlib import Path


def configure_runtime():
    root = Path(__file__).resolve().parents[2] / ".cache/task8/runtime"
    environment = {}
    for key, directory in {
        "CUDA_CACHE_PATH": "cuda",
        "XDG_CACHE_HOME": "xdg",
        "TORCH_HOME": "torch-home",
        "TORCHINDUCTOR_CACHE_DIR": "torch",
        "TRITON_CACHE_DIR": "triton",
        "CUPY_CACHE_DIR": "cupy",
        "NUMBA_CACHE_DIR": "numba",
        "VLLM_CACHE_ROOT": "vllm",
        "TMPDIR": "tmp",
    }.items():
        path = root / directory
        path.mkdir(parents=True, exist_ok=True)
        environment[key] = str(path)
    environment.update(
        {
            "HF_HOME": str(Path.home() / ".cache/huggingface"),
            "HF_HUB_CACHE": str(Path.home() / ".cache/huggingface/hub"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    for key in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        os.environ.pop(key, None)
    os.environ.update(environment)
    tempfile.tempdir = environment["TMPDIR"]
    return environment
