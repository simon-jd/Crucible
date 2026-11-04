"""CPU tests for the E0 network payload (scripts/snowflake/payloads/e0_network.py)."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "snowflake" / "payloads" / "e0_network.py"
spec = importlib.util.spec_from_file_location("e0_network", PATH)
e0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e0)


def test_split_bytes_covers_total():
    for total, streams in ((2 << 30, 1), (2 << 30, 8), (10, 3), (7, 8)):
        parts = e0.split_bytes(total, streams)
        assert len(parts) == streams and sum(parts) == total


@pytest.mark.parametrize("streams", [1, 4])
def test_loopback_moves_every_byte(streams):
    total = (8 << 20) + 3
    res = e0.loopback(streams, total)
    assert res["complete"] and res["bytes"] == total and res["streams"] == streams
    assert res["GBps"] > 0


def test_summarize_and_slower_rank_counts():
    assert e0.summarize([1.0, 2.0, 4.0]) == {"n": 3, "median": 2.0, "min": 1.0, "max": 4.0, "spread_pct": 150.0}
    assert e0.summarize([None]) is None
    row = lambda g: {"bw": [{"op": "all_reduce", "bytes": 1 << 20, "GBps": g}]}
    runs = [{"variant": "4x4", "pass": 0, "ranks": [row(0.5), row(0.4)]},
            {"variant": "4x4", "pass": 1, "ranks": [row(0.6), row(0.7)]},
            {"variant": "4x4", "pass": 2, "error": "boom"}]
    s = e0.summarize_runs(runs)["4x4"]["all_reduce"][str(1 << 20)]
    assert s["n"] == 2 and s["min"] == 0.4 and s["max"] == 0.6


def test_variants_and_ports_match_the_brief():
    assert all(12900 <= p <= 12919 for p in (e0.TCP_PORT, *e0.NCCL_PORTS))
    assert 1 in e0.TCP_STREAMS and 8 in e0.TCP_STREAMS
    grid = {n: (v["NCCL_SOCKET_NTHREADS"], v["NCCL_NSOCKS_PERTHREAD"]) for n, v in e0.SOCKET_VARIANTS}
    assert grid == {"1x1": ("1", "1"), "2x2": ("2", "2"), "4x4": ("4", "4"), "8x8": ("8", "8"),
                    "4x4_buff16m": ("4", "4")}
    assert dict(e0.SOCKET_VARIANTS)["4x4_buff16m"]["NCCL_BUFFSIZE"] == "16777216"
    shm = dict(e0.SHM_VARIANTS)
    assert shm["shm_default"] == {} and shm["shm_cudamemcpy"] == {"NCCL_SHM_USE_CUDA_MEMCPY": "1"}
    assert shm["shm_cudamemcpy_mode3"] == {"NCCL_SHM_USE_CUDA_MEMCPY": "1", "NCCL_SHM_MEMCPY_MODE": "3"}
    assert set(e0.ITERS_NET) == set(e0.ITERS_SHM) == set(e0.SIZES)


def test_run_collectives_on_gloo_cpu(tmp_path):
    torch = pytest.importorskip("torch")
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    try:
        bw, ok = e0.run_collectives(0, (4096,), {4096: (1, 2)}, device="cpu")
    finally:
        dist.destroy_process_group()
    assert ok and torch.distributed.is_available()
    assert [(b["op"], b["bytes"], b["iters"]) for b in bw] == [("all_reduce", 4096, 2), ("broadcast", 4096, 2)]
    assert all(b["GBps"] > 0 for b in bw)


def test_extra_variants_respect_nccl_socket_limits():
    for name, env in e0.EXTRA_VARIANTS + e0.SOCKET_VARIANTS:
        threads, socks = int(env["NCCL_SOCKET_NTHREADS"]), int(env["NCCL_NSOCKS_PERTHREAD"])
        assert threads <= 16 and threads * socks <= 64, name
    assert set(e0.ALL_VARIANTS) >= {"default", "8x8", "16x4", "8x8_ch4"}
