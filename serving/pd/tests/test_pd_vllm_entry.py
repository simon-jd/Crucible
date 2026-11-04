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


def test_the_plugin_dist_info_registers_a_vllm_general_plugin(monkeypatch):
    import importlib.metadata as md
    import sys

    monkeypatch.syspath_prepend(vllm_entry.PLUGIN_DIR)
    eps = [e for e in md.entry_points(group="vllm.general_plugins") if e.name == "crucible_e4"]
    assert len(eps) == 1 and eps[0].value == "serving.pd.vllm_entry:plugin"
    assert eps[0].load() is vllm_entry.plugin
    assert "serving.pd.vllm_entry" in sys.modules
    monkeypatch.delenv("CRUCIBLE_KVPROBE_DIR", raising=False)
    monkeypatch.delenv(vllm_entry.CUMEM_ENV, raising=False)
    vllm_entry.plugin()  # no-op without its environment variables


def test_fix_passes_empty_block_lists_for_none_and_restores_the_scheduler_output(monkeypatch):
    seen = []

    class Connector:
        def build_connector_meta(self, scheduler_output):
            ids = scheduler_output.scheduled_cached_reqs.new_block_ids
            seen.append([b[0] for b in ids])  # what vLLM 0.10.2 does; raises on None
            return "meta"

    out = types.SimpleNamespace(scheduled_cached_reqs=types.SimpleNamespace(new_block_ids=[([1, 2],), None]))
    try:
        Connector().build_connector_meta(out)
        raise AssertionError("expected the unpatched TypeError")
    except TypeError:
        pass
    monkeypatch.delenv(vllm_entry.FIX_ENV, raising=False)
    mod = types.SimpleNamespace(P2pNcclConnector=Connector)
    assert vllm_entry.apply_fix(mod) is False
    monkeypatch.setenv(vllm_entry.FIX_ENV, "1")
    assert vllm_entry.apply_fix(mod) is True and vllm_entry.apply_fix(mod) is True  # idempotent
    assert Connector().build_connector_meta(out) == "meta"
    assert seen[-1] == [[1, 2], []]
    assert out.scheduled_cached_reqs.new_block_ids == [([1, 2],), None]


def test_fix_targets_the_installed_vllm_connector_when_available(monkeypatch):
    import pytest

    mod = pytest.importorskip(vllm_entry.CONNECTOR_MODULE)
    import inspect

    assert "new_block_ids[0]" in inspect.getsource(mod.P2pNcclConnector.build_connector_meta)
    monkeypatch.setenv(vllm_entry.FIX_ENV, "1")
    assert vllm_entry.apply_fix() is True and mod.P2pNcclConnector._crucible_fixed
