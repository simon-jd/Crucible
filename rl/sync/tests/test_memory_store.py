"""MemoryStore + pair group: WeightSync into memory between two processes (actual gloo, CPU)."""

import multiprocessing as mp
import socket
import time

import pytest

torch = pytest.importorskip("torch")

from rl.sync.memory_store import MemoryStore, tensor_checksum

SCHEMA = {"a.weight": ((3, 2), torch.bfloat16), "b.bias": ((5,), torch.float32)}


def shards(scale=1.0):
    return [
        (
            "a.weight",
            (torch.arange(6, dtype=torch.float32).reshape(3, 2) * scale).to(torch.bfloat16),
        ),
        ("b.bias", torch.linspace(-1, 1, 5) * scale),
    ]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_checksum_is_content_sensitive_and_order_independent():
    x = torch.randn(1000, dtype=torch.float32).to(torch.bfloat16)
    assert tensor_checksum(x) == tensor_checksum(x.clone())
    y = x.clone()
    y[17] = y[17] + 1
    assert tensor_checksum(x) != tensor_checksum(y)
    # swapping two elements keeps the plain sum but changes the position-weighted one
    z = x.clone()
    z[[3, 900]] = x[[900, 3]]
    if x[3] != x[900]:
        assert tensor_checksum(x) != tensor_checksum(z)
    assert tensor_checksum(x, "sha256") == tensor_checksum(x.clone(), "sha256")
    with pytest.raises(ValueError):
        tensor_checksum(x, "md5")


def test_stage_requires_complete_exact_schema_and_increasing_versions():
    store = MemoryStore(SCHEMA)
    stage = store.begin(1, time.monotonic() + 30)
    with pytest.raises(ValueError, match="shape/dtype"):
        stage.add("a.weight", torch.zeros(3, 2))  # FP32 where BF16 is declared: no implicit cast
    with pytest.raises(ValueError, match="unexpected"):
        stage.add("c", torch.zeros(1))
    stage.add(*shards()[0])
    with pytest.raises(ValueError, match="duplicate"):
        stage.add(*shards()[0])
    with pytest.raises(ValueError, match="incomplete"):
        stage.prepare()
    stage.add(*shards()[1])
    stage.prepare()
    stage.commit()
    taken = store.take(1)
    assert [n for n, _ in taken] == sorted(SCHEMA)
    with pytest.raises(ValueError):
        store.take(1)  # handed over once
    with pytest.raises(ValueError, match="increase"):
        store.begin(1, time.monotonic() + 30)
    bad = store.begin(2, time.monotonic() + 30)
    with pytest.raises(ValueError, match="finite"):
        bad.add("b.bias", torch.tensor([0.0, float("nan"), 0, 0, 0]))


def test_sender_store_keeps_nothing_but_same_digest():
    keep, send = MemoryStore(SCHEMA), MemoryStore(SCHEMA, keep=False)
    for store in (keep, send):
        stage = store.begin(3, time.monotonic() + 30)
        for name, tensor in shards():
            stage.add(name, tensor)
        stage.prepare()
        stage.commit()
    assert keep.digest == send.digest and send.tensors is None


def _pair_rank(rank, port, rendezvous, corrupt):
    from datetime import timedelta

    import torch.distributed as dist

    from rl.sync.memory_store import MemoryStage
    from rl.sync.nccl_broadcast import WeightSync
    from rl.sync.pair_group import destroy_pair_group, init_pair_group, pair_rank

    # Like the learner and the in-process vLLM: each process has its own world-1 default group.
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}-{rank}", rank=0, world_size=1,
                            timeout=timedelta(seconds=30))  # fmt: skip
    group, _ = init_pair_group("gloo", "127.0.0.1", port, rank, timeout=30)
    try:
        assert dist.get_rank() == 0 and pair_rank(group) == rank
        store = MemoryStore(SCHEMA, keep=rank == 1)
        sync = WeightSync(store, group=group, source=0, timeout=30)
        stats = sync.push(shards(2.0) if rank == 0 else None, 1)
        assert stats.bytes == 3 * 2 * 2 + 5 * 4
        if rank == 1:
            got = dict(store.take(1))
            for name, tensor in shards(2.0):
                assert torch.equal(got[name], tensor)
        if corrupt:
            original = MemoryStage.add

            def flip(self, name, tensor):
                if rank == 1 and name == "b.bias":
                    tensor = tensor.clone()
                    tensor[0] += 1
                return original(self, name, tensor)

            MemoryStage.add = flip
            try:
                with pytest.raises(RuntimeError):
                    sync.push(shards(3.0) if rank == 0 else None, 2)
            finally:
                MemoryStage.add = original
            assert store.version == 1  # the corrupted candidate never committed
    finally:
        destroy_pair_group(group)
        dist.destroy_process_group()


@pytest.mark.parametrize("corrupt", [False, True])
def test_weight_sync_over_pair_group_between_two_default_rank0_processes(tmp_path, corrupt):
    context = mp.get_context("spawn")
    port = free_port()
    children = [
        context.Process(target=_pair_rank, args=(rank, port, str(tmp_path / "rdv"), corrupt))
        for rank in range(2)
    ]
    for child in children:
        child.start()
    try:
        for child in children:
            child.join(90)
        assert [child.exitcode for child in children] == [0, 0]
    finally:
        for child in children:
            if child.is_alive():
                child.kill()
                child.join()
