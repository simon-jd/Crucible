import os
import types

from serving.pd import vllm_entry


def test_cumem_override_replaces_the_connector_context_and_restores_the_environment(monkeypatch):
    mod = types.SimpleNamespace(set_p2p_nccl_context=None)
    monkeypatch.delenv(vllm_entry.CUMEM_ENV, raising=False)
    assert vllm_entry.override_cumem(mod) is False and mod.set_p2p_nccl_context is None
    monkeypatch.setenv(vllm_entry.CUMEM_ENV, "0")
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "7")
    monkeypatch.delenv("NCCL_MIN_NCHANNELS", raising=False)
    assert vllm_entry.override_cumem(mod) is True
    with mod.set_p2p_nccl_context("8"):
        assert os.environ["NCCL_CUMEM_ENABLE"] == "0"
        assert os.environ["NCCL_MIN_NCHANNELS"] == os.environ["NCCL_MAX_NCHANNELS"] == "8"
    assert os.environ["NCCL_CUMEM_ENABLE"] == "7" and "NCCL_MIN_NCHANNELS" not in os.environ
