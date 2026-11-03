"""Cache containment is a CPU-safe worker boundary, independent of test scripts."""

import os
import tempfile

from rl.rollout._worker import _configure_worker_environment


def test_worker_cache_paths_override_inherited_defaults(tmp_path, monkeypatch):
    keys = [
        "CUDA_CACHE_PATH",
        "XDG_CACHE_HOME",
        "VLLM_CACHE_ROOT",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_HOME",
        "CUPY_CACHE_DIR",
        "NUMBA_CACHE_DIR",
        "TMPDIR",
    ]
    for key in keys:
        monkeypatch.setenv(key, "/unapproved/inherited/cache")
    monkeypatch.setattr(tempfile, "tempdir", "/unapproved/already-cached-tempdir")
    environment = _configure_worker_environment(tmp_path)
    assert set(environment) == set(keys)
    for key, value in environment.items():
        assert value.startswith(str(tmp_path) + "/")
        assert os.environ[key] == value
        assert os.path.isdir(value)
    with tempfile.TemporaryDirectory() as generated:
        assert generated.startswith(environment["TMPDIR"] + "/")
