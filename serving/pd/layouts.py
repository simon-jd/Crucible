"""Server and proxy plans for the four E4 layouts (pure functions, no process handling).

Layouts, all serving Qwen3-8B-AWQ with vLLM 0.10.2 behind serving.pd.proxy:

- ``colo``:   one colocated server on one A10 (proxy in "dp" mode with one backend);
- ``dp2``:    two colocated replicas on the two A10s of one node, least-outstanding routing;
- ``pd-shm``: 1 prefill + 1 decode server on the two A10s of one node, KV over NCCL's SHM transport
  (default host-copy mode);
- ``pd-shm-ce``: the same with NCCL_SHM_USE_CUDA_MEMCPY=1 (copy-engine SHM, E0's fastest on-node
  setting) and NCCL_CUMEM_ENABLE=0 for the connector's communicators: the connector forces cuMem on,
  and SHM/CE with cuMem fails on these vGPUs ("peer access is not supported", smoke run e4-smoke1b);
- ``pd-tcp``: prefill on node 0, decode on node 1 (two single-A10 nodes), KV over NCCL (NET/Socket).

Every server gets the Task 12 baseline arguments (seed 17, max-model-len 8192, max-num-seqs 64, no
prefix cache, CUDA graphs). The decode server of a P/D pair uses a lower --gpu-memory-utilization,
because P2pNcclConnector keeps a second copy of every received KV tensor (on the GPU up to
``kv_buffer_size`` bytes, then in a pinned host pool) until the request finishes.

Ports that cross nodes stay inside E4's band 12970-12999.
"""

import json
from dataclasses import dataclass, field

LAYOUTS = ("colo", "dp2", "pd-shm", "pd-shm-ce", "pd-tcp")
CUMEM_ENV = "CRUCIBLE_P2P_NCCL_CUMEM"  # read by serving.pd.vllm_entry
PROXY_PORT = 12970
HTTP_PORTS = (12971, 12972)
KV_PORTS = {"kv_producer": 12980, "kv_consumer": 12985}
BAND = range(12970, 13000)
BYTES_PER_TOKEN = 36 * 2 * 8 * 128 * 2  # Qwen3-8B: layers x (K, V) x KV heads x head dim x fp16
BLOCK_SIZE = 16

COLOCATED_UTIL = 0.8
DECODE_UTIL = 0.7


@dataclass
class Server:
    name: str
    node: int  # 0 = head instance (runs the proxy and the replayer), 1 = second instance
    gpu: str  # CUDA_VISIBLE_DEVICES inside that node
    port: int
    host: str  # bind address of the HTTP server
    gpu_util: float
    kv_role: str | None = None  # kv_producer / kv_consumer / None
    env: dict = field(default_factory=dict)


@dataclass
class Plan:
    layout: str
    nodes: int
    servers: list
    proxy_mode: str  # "dp" or "pd"
    notes: dict = field(default_factory=dict)


def kv_transfer_config(role, kv_port, http_port, pool_gb, buffer_bytes, send_type="PUT_ASYNC", channels=8):
    """--kv-transfer-config for P2pNcclConnector. No proxy_ip/proxy_port: the proxy is static, no ping."""
    if role not in KV_PORTS:
        raise ValueError(f"unknown kv role {role!r}")
    return {
        "kv_connector": "P2pNcclConnector",
        "kv_role": role,
        "kv_buffer_size": float(buffer_bytes),
        "kv_port": str(kv_port),
        "kv_connector_extra_config": {
            "http_port": str(http_port),
            "send_type": send_type,
            "nccl_num_channels": str(channels),
            "mem_pool_size_gb": str(pool_gb),
        },
    }


def plan(layout, decode_pool_gb=24, decode_buffer_bytes=2e9, socket_threads=8, socks_per_thread=8,
         nccl_buffsize=16 << 20, nccl_debug=True):
    """The servers of one layout. Cross-node NCCL settings are E0's best measured ones
    (docs/evidence/snowflake/e0-network.md on snowflake/e0); its NCCL_MIN_NCHANNELS=4 is moot because
    the connector pins NCCL_MIN/MAX_NCHANNELS to nccl_num_channels (8) while it creates a communicator."""
    nccl = {"NCCL_IB_DISABLE": "1"}
    if nccl_debug:
        nccl |= {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET"}
    if layout == "colo":
        return Plan(layout, 1, [Server("colo", 0, "0", HTTP_PORTS[0], "127.0.0.1", COLOCATED_UTIL)], "dp")
    if layout == "dp2":
        servers = [Server(f"replica{i}", 0, str(i), HTTP_PORTS[i], "127.0.0.1", COLOCATED_UTIL) for i in (0, 1)]
        return Plan(layout, 1, servers, "dp")
    if layout in ("pd-shm", "pd-shm-ce"):
        env = nccl | ({"NCCL_SHM_USE_CUDA_MEMCPY": "1", CUMEM_ENV: "0"} if layout == "pd-shm-ce" else {})
        servers = [
            Server("prefill", 0, "0", HTTP_PORTS[0], "127.0.0.1", COLOCATED_UTIL, "kv_producer", dict(env)),
            Server("decode", 0, "1", HTTP_PORTS[1], "127.0.0.1", DECODE_UTIL, "kv_consumer", dict(env)),
        ]
        p = Plan(layout, 1, servers, "pd")
        p.layout = layout
    elif layout == "pd-tcp":
        env = nccl | {"NCCL_SOCKET_NTHREADS": str(socket_threads), "NCCL_NSOCKS_PERTHREAD": str(socks_per_thread),
                      "NCCL_BUFFSIZE": str(nccl_buffsize)}
        servers = [
            Server("prefill", 0, "0", HTTP_PORTS[0], "127.0.0.1", COLOCATED_UTIL, "kv_producer", dict(env)),
            Server("decode", 1, "0", HTTP_PORTS[1], "0.0.0.0", DECODE_UTIL, "kv_consumer", dict(env)),
        ]
        p = Plan(layout, 2, servers, "pd")
    else:
        raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")
    p.notes = {"decode_pool_gb": decode_pool_gb, "decode_buffer_bytes": decode_buffer_bytes}
    return p


def server_args(server, snapshot, model_name, notes=None):
    """Arguments after `python -m serving.pd.vllm_entry` (the vLLM OpenAI server's own CLI)."""
    args = [
        "--model", snapshot, "--served-model-name", model_name,
        "--host", server.host, "--port", str(server.port),
        "--gpu-memory-utilization", str(server.gpu_util), "--seed", "17",
        "--max-model-len", "8192", "--max-num-seqs", "64",
        "--no-enable-prefix-caching", "--generation-config", "vllm",
    ]  # fmt: skip
    if server.kv_role:
        notes = notes or {}
        consumer = server.kv_role == "kv_consumer"
        cfg = kv_transfer_config(
            server.kv_role, KV_PORTS[server.kv_role], server.port,
            pool_gb=notes.get("decode_pool_gb", 24) if consumer else 1,
            buffer_bytes=notes.get("decode_buffer_bytes", 2e9) if consumer else 1e9,
        )  # fmt: skip
        args += ["--kv-transfer-config", json.dumps(cfg, separators=(",", ":"))]
    return args


def server_env(server, node_ip, cache_dir, probe_dir=None):
    """Environment additions for one server process (merged over the job's environment)."""
    env = {
        "CUDA_VISIBLE_DEVICES": server.gpu,
        "VLLM_USE_V1": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
        "VLLM_CACHE_ROOT": f"{cache_dir}/{server.name}",  # one compile cache per process, no write races
        # P2pNcclEngine binds its ZMQ socket on get_ip(); on one node loopback is enough.
        "VLLM_HOST_IP": node_ip,
        **server.env,
    }
    if probe_dir and server.kv_role:
        env["CRUCIBLE_KVPROBE_DIR"] = probe_dir
    return env


def check_ports(p):
    """Every port a two-node plan exposes across nodes must be in E4's band."""
    ports = [s.port for s in p.servers] + [PROXY_PORT] + [KV_PORTS[s.kv_role] for s in p.servers if s.kv_role]
    bad = [x for x in ports if x not in BAND]
    if bad:
        raise ValueError(f"ports outside 12970-12999: {bad}")
    return ports


def kv_bytes(prompt_tokens, bytes_per_token=BYTES_PER_TOKEN, block=BLOCK_SIZE):
    """Bytes P2pNcclConnector sends for one prompt: whole blocks of every layer's K and V."""
    blocks = -(-prompt_tokens // block)
    return blocks * block * bytes_per_token
