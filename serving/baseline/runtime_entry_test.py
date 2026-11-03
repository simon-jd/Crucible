import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from serving.baseline.runtime_entry import inventory, runtime_environment


@pytest.mark.parametrize("inherited", [None, "32", "invalid"])
def test_compile_threads_override_preserves_owned_runtime_controls(
    tmp_path, monkeypatch, inherited
):
    monkeypatch.delenv("TORCHINDUCTOR_COMPILE_THREADS", raising=False)
    before = runtime_environment(tmp_path)
    if inherited is not None:
        monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", inherited)
    env = runtime_environment(tmp_path)
    assert env["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    assert {k: v for k, v in env.items() if k != "TORCHINDUCTOR_COMPILE_THREADS"} == {
        k: v for k, v in before.items() if k != "TORCHINDUCTOR_COMPILE_THREADS"
    }
    assert env["MAX_JOBS"] == env["HF_HUB_OFFLINE"] == "1"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path / "inductor")


def test_compile_threads_applied_before_redirector_and_server(tmp_path, monkeypatch):
    from serving.baseline import runtime_entry as entry

    overlay = tmp_path / "overlay"
    overlay.mkdir()
    source = overlay / "_cuda_bindings_redirector.py"
    source.write_text("# CPU fixture; never executed\n")
    lock = tmp_path / "lock.json"
    lock.write_text('{"packages": []}')
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
                "native_downloads_disabled": True,
                "source_files": [
                    {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
                ],
                "disable_download_environment": {"FLASHINFER_DISABLE_VERSION_CHECK": "1"},
            }
        )
    )
    monkeypatch.setattr(entry.os, "environ", dict(os.environ, TORCHINDUCTOR_COMPILE_THREADS="32"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runtime_entry",
            "--lock",
            str(lock),
            "--overlay",
            str(overlay),
            "--cache-audit",
            str(audit),
            "--work",
            str(tmp_path / "work"),
            "--inventory-out",
            str(tmp_path / "inventory.json"),
            "--mode",
            "server",
        ],
    )
    monkeypatch.setattr(entry, "inventory", lambda *_: [])
    monkeypatch.setattr(entry, "validated_startup_path", lambda *_: overlay)
    events = []

    def check_environment(name):
        assert entry.os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
        assert entry.os.environ["HF_HUB_OFFLINE"] == "1"
        assert entry.os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] == "1"
        events.append(name)

    def redirector(name):
        assert name == "_cuda_bindings_redirector"
        check_environment(name)
        return SimpleNamespace(__file__=str(source))

    def server(name, **kwargs):
        assert name == "sglang.launch_server"
        check_environment(name)

    monkeypatch.setattr(entry.importlib, "import_module", redirector)
    monkeypatch.setattr(entry.runpy, "run_module", server)
    entry.main()
    assert events == ["_cuda_bindings_redirector", "sglang.launch_server"]


def dist(site, name, version):
    folder = site / f"{name}-{version}.dist-info"
    folder.mkdir(parents=True)
    data = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode()
    (folder / "METADATA").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_overlay_precedence_is_checked_without_native_imports(tmp_path):
    overlay, base = tmp_path / "overlay", tmp_path / "base"
    old = dist(base, "example", "1")
    new = dist(overlay, "example", "2")
    lock = {
        "packages": [
            {"name": "example", "version": "2", "origin": "overlay", "metadata_sha256": new}
        ]
    }
    assert inventory(lock, overlay, base)[0]["origin"] == str(overlay)
    (overlay / "example-2.dist-info/METADATA").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="inventory"):
        inventory(lock, overlay, base)
    assert old != new


def test_runtime_environment_keeps_all_writes_owned_and_shared_hf_explicit(tmp_path):
    env = runtime_environment(tmp_path)
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["HF_HUB_CACHE"].endswith("/.cache/huggingface/hub")
    for name in ("TMPDIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TORCH_EXTENSIONS_DIR"):
        assert Path(env[name]).is_relative_to(tmp_path)
    assert env["HF_HUB_OFFLINE"] == "1"


def test_native_environment_removes_inherited_deprecated_cache_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSFORMERS_CACHE", "/unreviewed/inherited")
    env = runtime_environment(tmp_path)
    assert "TRANSFORMERS_CACHE" not in env
    assert env["HF_HOME"] == str(Path.home() / ".cache/huggingface")
    assert env["HF_HUB_CACHE"] == env["HUGGINGFACE_HUB_CACHE"]
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_pinned_pth_layout_is_validated_without_executing_startup_code(tmp_path):
    from serving.baseline.runtime_entry import validated_startup_path

    relative = "nvidia_cutlass_dsl/python_packages"
    (tmp_path / relative).mkdir(parents=True)
    (tmp_path / "nvidia_cutlass_dsl.pth").write_text(relative)
    (tmp_path / "_cuda_bindings_redirector.pth").write_text(
        "# pinned comment\nimport _cuda_bindings_redirector\n"
    )
    (tmp_path / "_cuda_bindings_redirector.py").write_text(
        'raise AssertionError("must never execute during preparation")\n'
    )
    audit = {
        "source_files": [
            {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in tmp_path.iterdir()
            if p.is_file()
        ]
    }
    assert validated_startup_path(tmp_path, audit) == tmp_path / relative
    (tmp_path / "nvidia_cutlass_dsl.pth").write_text("../../escape")
    with pytest.raises(ValueError, match="startup"):
        validated_startup_path(tmp_path, audit)
