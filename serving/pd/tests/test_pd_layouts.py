import json

import pytest

from serving.pd import layouts


def test_every_layout_has_its_servers_on_the_right_nodes_and_gpus():
    colo = layouts.plan("colo")
    assert [(s.node, s.gpu, s.kv_role) for s in colo.servers] == [(0, "0", None)] and colo.proxy_mode == "dp"
    dp = layouts.plan("dp2")
    assert [(s.node, s.gpu) for s in dp.servers] == [(0, "0"), (0, "1")] and dp.nodes == 1
    shm = layouts.plan("pd-shm")
    assert [(s.node, s.gpu, s.kv_role) for s in shm.servers] == [(0, "0,1", "kv_producer"), (0, "1,0", "kv_consumer")]
    assert "NCCL_SHM_USE_CUDA_MEMCPY" not in shm.servers[0].env
    ce = layouts.plan("pd-shm-ce")
    assert ce.layout == "pd-shm-ce" and all(s.env["NCCL_SHM_USE_CUDA_MEMCPY"] == "1" for s in ce.servers)
    assert all(s.env[layouts.CUMEM_ENV] == "0" for s in ce.servers)
    tcp = layouts.plan("pd-tcp")
    assert tcp.nodes == 2 and [(s.node, s.gpu) for s in tcp.servers] == [(0, "0"), (1, "0")]
    assert tcp.servers[1].host == "0.0.0.0"  # the proxy on node 0 reaches it over the network
    env = tcp.servers[0].env
    assert (env["NCCL_SOCKET_NTHREADS"], env["NCCL_NSOCKS_PERTHREAD"], env["NCCL_BUFFSIZE"]) == ("8", "8", "16777216")
    with pytest.raises(ValueError):
        layouts.plan("pd-rdma")


def test_cross_node_ports_stay_in_the_e4_band():
    for name in layouts.LAYOUTS:
        ports = layouts.check_ports(layouts.plan(name))
        assert all(12970 <= x <= 12999 for x in ports)


def test_colocated_server_args_match_the_task12_baseline():
    s = layouts.plan("colo").servers[0]
    args = layouts.server_args(s, "/m", "Qwen/Qwen3-8B-AWQ")
    joined = " ".join(args)
    for flag in ("--gpu-memory-utilization 0.8", "--seed 17", "--max-model-len 8192", "--max-num-seqs 64",
                 "--no-enable-prefix-caching", "--generation-config vllm"):
        assert flag in joined
    assert "--kv-transfer-config" not in args and "--enforce-eager" not in args


def test_pd_servers_get_producer_and_consumer_kv_configs():
    p = layouts.plan("pd-tcp", decode_pool_gb=16, decode_buffer_bytes=3e9)
    pre, dec = (layouts.server_args(s, "/m", "m", p.notes) for s in p.servers)
    pc = json.loads(pre[pre.index("--kv-transfer-config") + 1])
    dc = json.loads(dec[dec.index("--kv-transfer-config") + 1])
    assert pc["kv_connector"] == dc["kv_connector"] == "P2pNcclConnector"
    assert (pc["kv_role"], dc["kv_role"]) == ("kv_producer", "kv_consumer")
    assert (pc["kv_port"], dc["kv_port"]) == ("12980", "12985")
    assert pc["kv_connector_extra_config"]["http_port"] == "12971"
    assert dc["kv_connector_extra_config"]["http_port"] == "12972"
    assert dc["kv_connector_extra_config"]["mem_pool_size_gb"] == "16"
    assert pc["kv_connector_extra_config"]["mem_pool_size_gb"] == "1"  # the producer never parks tensors
    assert dc["kv_buffer_size"] == 3e9
    assert "proxy_ip" not in dc["kv_connector_extra_config"]  # static proxy, no ping thread
    assert dec[dec.index("--gpu-memory-utilization") + 1] == "0.7"
    with pytest.raises(ValueError):
        layouts.kv_transfer_config("kv_both", 1, 2, 1, 1)


def test_decode_parks_received_kv_in_host_memory_by_default():
    p = layouts.plan("pd-shm")
    dec = layouts.server_args(p.servers[1], "/m", "m", p.notes)
    cfg = json.loads(dec[dec.index("--kv-transfer-config") + 1])
    assert cfg["kv_buffer_size"] == 1.0 and cfg["kv_connector_extra_config"]["mem_pool_size_gb"] == "64"


def test_server_env_pins_the_gpu_and_enables_the_probe_only_for_kv_servers():
    colo = layouts.plan("colo").servers[0]
    env = layouts.server_env(colo, "127.0.0.1", "/c", "/probe")
    assert env["CUDA_VISIBLE_DEVICES"] == "0" and env["VLLM_CACHE_ROOT"] == "/c/colo"
    assert "CRUCIBLE_KVPROBE_DIR" not in env and "CRUCIBLE_P2P_FIX" not in env
    dec = layouts.plan("pd-tcp").servers[1]
    env = layouts.server_env(dec, "10.0.0.2", "/c", "/probe")
    assert env["VLLM_HOST_IP"] == "10.0.0.2" and env["CRUCIBLE_KVPROBE_DIR"] == "/probe"
    assert env["CRUCIBLE_P2P_FIX"] == "1"
    assert env["NCCL_IB_DISABLE"] == "1"


def test_kv_bytes_counts_whole_blocks_of_all_layers():
    assert layouts.BYTES_PER_TOKEN == 147456
    assert layouts.kv_bytes(1) == 16 * 147456
    assert layouts.kv_bytes(16) == 16 * 147456
    assert layouts.kv_bytes(17) == 32 * 147456
