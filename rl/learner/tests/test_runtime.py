"""Dependency-light runtime checks run in the default CPU environment."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def test_package_import_is_inert():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, sys; before = dict(os.environ); import rl.learner; "
                "assert dict(os.environ) == before; assert 'torch' not in sys.modules"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert not result.stdout and not result.stderr


def test_explicit_runtime_contains_actual_model_caches(monkeypatch):
    from rl.learner.runtime import configure_runtime

    keys = (
        "CUDA_CACHE_PATH",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUPY_CACHE_DIR",
        "NUMBA_CACHE_DIR",
        "VLLM_CACHE_ROOT",
        "TMPDIR",
    )
    for key in keys + (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ):
        monkeypatch.setenv(key, "/unused")
    monkeypatch.setattr(tempfile, "tempdir", None)
    actual = configure_runtime()
    root = Path(__file__).resolve().parents[3] / ".cache/task8/runtime"
    for key in keys:
        assert Path(actual[key]).is_relative_to(root)
        assert os.environ[key] == actual[key]
        assert Path(actual[key]).is_dir()
    assert actual["HF_HUB_CACHE"] == str(Path.home() / ".cache/huggingface/hub")
    assert "HUGGINGFACE_HUB_CACHE" not in os.environ
    assert "TRANSFORMERS_CACHE" not in os.environ
    assert tempfile.gettempdir() == actual["TMPDIR"]
