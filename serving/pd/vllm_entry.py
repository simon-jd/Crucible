"""`python -m serving.pd.vllm_entry <vLLM OpenAI server args>`: vLLM's server with the KV probe.

The probe (serving.pd.kvprobe) is installed at import time when $CRUCIBLE_KVPROBE_DIR is set, so
it is also in place in vLLM's engine-core process, whether that process is forked (inherits the
patched class) or spawned (re-imports this module as __mp_main__).
"""

import runpy
import sys

from serving.pd import kvprobe

kvprobe.install_if_enabled()

if __name__ == "__main__":
    sys.argv = ["vllm.entrypoints.openai.api_server", *sys.argv[1:]]
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__", alter_sys=True)
