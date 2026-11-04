"""`python -m serving.pd.vllm_entry <vLLM OpenAI server args>`: vLLM's server with the KV probe.

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
ENGINE_MODULE = kvprobe.MODULE


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


PLUGIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugin")

plugin()

if __name__ == "__main__":
    sys.argv = ["vllm.entrypoints.openai.api_server", *sys.argv[1:]]
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__", alter_sys=True)
