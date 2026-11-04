"""`python -m serving.pd.vllm_entry <vLLM OpenAI server args>`: vLLM's server with the KV probe.

With $CRUCIBLE_P2P_FIX=1, P2pNcclConnector.build_connector_meta gets a one-line bug fix: in vLLM
0.10.2 the producer indexes ``new_block_ids[0]`` of a continuing chunked-prefill request, which is
None when that step allocated no new block, and the engine core dies with "TypeError: 'NoneType'
object is not subscriptable" (run e4-full-pd-tcp, 4x level). The fix passes an empty block list for
those requests for the duration of the call and restores the scheduler output afterwards.

With $CRUCIBLE_P2P_NCCL_CUMEM set (0 or 1), the P2pNcclEngine communicators are created with that
NCCL_CUMEM_ENABLE instead of the 1 that vLLM 0.10.2's set_p2p_nccl_context forces.

The probe (serving.pd.kvprobe) is installed when $CRUCIBLE_KVPROBE_DIR is set. Installing it at
import time here is not enough: vLLM's engine-core process (where P2pNcclEngine lives) does not
re-import this module (smoke run e4-smoke4 recorded no events). So `plugin` is also registered as a
vLLM general plugin (serving/pd/plugin/*.dist-info, found through PYTHONPATH), which vLLM calls in
every process it starts, including the engine core.
"""

import os
import runpy
import sys
from contextlib import contextmanager

from serving.pd import kvprobe

CUMEM_ENV = "CRUCIBLE_P2P_NCCL_CUMEM"
FIX_ENV = "CRUCIBLE_P2P_FIX"
ENGINE_MODULE = kvprobe.MODULE
CONNECTOR_MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector"


def fix_build_connector_meta(connector_cls):
    """Wrap build_connector_meta so None entries of new_block_ids read as 'no new blocks'."""
    if getattr(connector_cls, "_crucible_fixed", False):
        return connector_cls
    original = connector_cls.build_connector_meta

    def build_connector_meta(self, scheduler_output):
        cached = scheduler_output.scheduled_cached_reqs
        saved = cached.new_block_ids
        if any(b is None for b in saved):
            cached.new_block_ids = [([],) if b is None else b for b in saved]
        try:
            return original(self, scheduler_output)
        finally:
            cached.new_block_ids = saved

    connector_cls.build_connector_meta = build_connector_meta
    connector_cls._crucible_fixed = True
    return connector_cls


def apply_fix(module=None):
    if os.environ.get(FIX_ENV) != "1":
        return False
    if module is None:
        import importlib

        module = importlib.import_module(CONNECTOR_MODULE)
    fix_build_connector_meta(module.P2pNcclConnector)
    return True


def nccl_context(cumem):
    """set_p2p_nccl_context with a chosen NCCL_CUMEM_ENABLE; restores the variables afterwards."""

    @contextmanager
    def ctx(num_channels):
        names = ("NCCL_MAX_NCHANNELS", "NCCL_MIN_NCHANNELS", "NCCL_CUMEM_ENABLE")
        saved = {k: os.environ.get(k) for k in names}
        try:
            os.environ["NCCL_MAX_NCHANNELS"] = os.environ["NCCL_MIN_NCHANNELS"] = num_channels
            os.environ["NCCL_CUMEM_ENABLE"] = cumem
            yield
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    return ctx


def override_cumem(module=None):
    cumem = os.environ.get(CUMEM_ENV)
    if cumem is None:
        return False
    if module is None:
        import importlib

        module = importlib.import_module(ENGINE_MODULE)
    module.set_p2p_nccl_context = nccl_context(cumem)
    return True


def plugin():
    """vLLM general-plugin entry point (vllm.general_plugins): idempotent, env-gated."""
    kvprobe.install_if_enabled()
    override_cumem()
    apply_fix()


PLUGIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugin")

plugin()

if __name__ == "__main__":
    sys.argv = ["vllm.entrypoints.openai.api_server", *sys.argv[1:]]
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__", alter_sys=True)
