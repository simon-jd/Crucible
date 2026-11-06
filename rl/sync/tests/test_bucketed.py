"""BucketedSync: layout, pack/unpack, manifest, and the two-process gloo protocol (CPU only)."""

import hashlib
import itertools
import math
import multiprocessing as mp
import socket
import time

import pytest

torch = pytest.importorskip("torch")

from rl.sync.bucketed import (
    BucketedSync,
    check_manifest,
    make_manifest,
    manifest_digest,
    pack,
    plan_buckets,
    unpack,
)
from rl.sync.memory_store import MemoryStore

SCHEMA = {
    "a.weight": ((3, 5), torch.bfloat16),
    "b.bias": ((7,), torch.float32),
    "c.steps": ((), torch.int64),
    "d.mask": ((4,), torch.bool),
    "e.empty": ((0, 3), torch.bfloat16),
    "f.big": ((300,), torch.bfloat16),
}


def shards(scale=1.0):
    return [
        (
            "a.weight",
            (torch.arange(15, dtype=torch.float32).reshape(3, 5) * scale).to(torch.bfloat16),
        ),
        ("b.bias", torch.linspace(-1, 1, 7) * scale),
        ("c.steps", torch.tensor(int(41 * scale))),
        ("d.mask", torch.tensor([True, False, True, scale > 1])),
        ("e.empty", torch.zeros(0, 3, dtype=torch.bfloat16)),
        ("f.big", (torch.arange(300, dtype=torch.float32) * scale / 7).to(torch.bfloat16)),
    ]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_plan_orders_aligns_and_covers_the_buffer_with_buckets():
    plan = plan_buckets(SCHEMA, 256, align=64)
    names = [slot.name for slot in plan.slots]
    assert names == sorted(SCHEMA)
    end = 0
    for slot in plan.slots:
        shape, dtype = SCHEMA[slot.name]
        assert slot.offset % 64 == 0 and slot.offset >= end
        assert slot.nbytes == math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        end = slot.offset + slot.nbytes
    assert plan.total_bytes % 64 == 0 and plan.total_bytes >= end
    assert plan.tensor_bytes == 30 + 28 + 8 + 4 + 0 + 600
    buckets = plan.buckets()
    assert buckets[0][0] == 0 and buckets[-1][1] == plan.total_bytes
    assert all(a[1] == b[0] for a, b in itertools.pairwise(buckets))
    assert len(buckets) == -(-plan.total_bytes // 256)
    # digest chunks never cross a bucket edge
    chunks = plan_buckets(SCHEMA, 256, align=64, digest_chunk=100).chunks()
    assert chunks[0][0] == 0 and chunks[-1][1] == plan.total_bytes
    for a, b in chunks:
        assert 0 < b - a <= 100 and a // 256 == (b - 1) // 256
    assert len(plan_buckets(SCHEMA, 1 << 30).buckets()) == 1


def test_layout_digest_depends_on_schema_and_bucket_size():
    base = plan_buckets(SCHEMA, 256)
    assert base.digest() == plan_buckets(dict(reversed(SCHEMA.items())), 256).digest()
    assert base.digest() != plan_buckets(SCHEMA, 512).digest()
    other = dict(SCHEMA, **{"b.bias": ((7,), torch.bfloat16)})
    assert base.digest() != plan_buckets(other, 256).digest()
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="bucket"):
            plan_buckets(SCHEMA, bad)
    with pytest.raises(ValueError, match="align"):
        plan_buckets(SCHEMA, 256, align=6)
    with pytest.raises(ValueError, match="schema"):
        plan_buckets({}, 256)


def test_pack_unpack_round_trip_is_exact_and_zero_copy():
    plan = plan_buckets(SCHEMA, 128)
    flat = torch.zeros(plan.total_bytes, dtype=torch.uint8)
    pack(plan, shards(2.0), flat)
    views = unpack(plan, flat)
    assert [name for name, _ in views] == sorted(SCHEMA)
    expected = dict(shards(2.0))
    for name, view in views:
        assert view.dtype == expected[name].dtype and view.shape == expected[name].shape
        assert torch.equal(view, expected[name])
        assert view.is_contiguous()
        if view.numel():
            assert view.untyped_storage().data_ptr() == flat.untyped_storage().data_ptr()
    # padding stays zero, so the packed bytes (and their digests) are deterministic
    used = torch.zeros(plan.total_bytes, dtype=torch.bool)
    for slot in plan.slots:
        used[slot.offset : slot.offset + slot.nbytes] = True
    assert int(flat[~used].count_nonzero()) == 0
    again = torch.zeros(plan.total_bytes, dtype=torch.uint8)
    pack(plan, reversed(shards(2.0)), again)  # arrival order does not matter
    assert torch.equal(flat, again)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda s: s + [("zzz", torch.zeros(1))], "unexpected"),
        (lambda s: s + [s[0]], "duplicate"),
        (lambda s: s[1:], "incomplete"),
        (lambda s: [("a.weight", s[0][1].float())] + s[1:], "shape/dtype"),
        (lambda s: [("a.weight", s[0][1].reshape(5, 3))] + s[1:], "shape/dtype"),
    ],
)
def test_pack_rejects_schema_violations_without_casting(mutate, match):
    plan = plan_buckets(SCHEMA, 128)
    flat = torch.zeros(plan.total_bytes, dtype=torch.uint8)
    with pytest.raises(ValueError, match=match):
        pack(plan, mutate(shards()), flat)


def test_manifest_names_the_first_differing_chunk():
    plan = plan_buckets(SCHEMA, 128, digest_chunk=64)
    flat = torch.zeros(plan.total_bytes, dtype=torch.uint8)
    pack(plan, shards(), flat)
    raw = flat.numpy()
    hashes = [hashlib.sha256(raw[a:b]).hexdigest() for a, b in plan.chunks()]
    source = make_manifest(7, "c" * 64, plan, hashes)
    assert source["chunk_sha256"] == hashes and source["layout_sha256"] == plan.digest()
    assert source["tensor_bytes"] == plan.tensor_bytes and source["total_bytes"] == plan.total_bytes
    check_manifest(source, make_manifest(7, "c" * 64, plan, list(hashes)))
    assert manifest_digest(source) == manifest_digest(
        make_manifest(7, "c" * 64, plan, list(hashes))
    )
    assert manifest_digest(source) != manifest_digest(make_manifest(8, "c" * 64, plan, hashes))
    corrupt = flat.clone()
    corrupt[plan.chunks()[2][0] + 3] ^= 1
    bad = [hashlib.sha256(corrupt.numpy()[a:b]).hexdigest() for a, b in plan.chunks()]
    with pytest.raises(ValueError, match=r"1 of \d+ chunks \(first: chunk 2\)"):
        check_manifest(source, make_manifest(7, "c" * 64, plan, bad))
    with pytest.raises(ValueError, match="version"):
        check_manifest(source, make_manifest(8, "c" * 64, plan, hashes))
    with pytest.raises(ValueError, match="contract"):
        check_manifest(source, make_manifest(7, "d" * 64, plan, hashes))


def test_stage_adopt_enforces_the_complete_schema_and_finite_values():
    store = MemoryStore(SCHEMA, digest="sha256")
    stage = store.begin(1, time.monotonic() + 30)
    with pytest.raises(ValueError, match="parameter name"):
        stage.adopt(shards()[1:], "x", 1)
    with pytest.raises(ValueError, match="shape/dtype"):
        stage.adopt([("a.weight", shards()[0][1].float())] + shards()[1:], "x", 1)
    nan = [(n, t.clone()) for n, t in shards()]
    nan[1][1][3] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        stage.adopt(nan, "x", 1)
    stage.adopt(shards(), "d" * 64, 670)
    stage.commit()
    assert store.version == 1 and store.digest == "d" * 64
    assert [n for n, _ in store.take(1)] == sorted(SCHEMA)


def test_bucketed_sync_requires_a_sha256_store_and_a_group():
    with pytest.raises(ValueError, match="sha256"):
        BucketedSync(MemoryStore(SCHEMA), group=object(), bucket_bytes=128)
    with pytest.raises(ValueError, match="group"):
        BucketedSync(MemoryStore(SCHEMA, digest="sha256"), group=None, bucket_bytes=128)


def _pair_rank(rank, port, rendezvous, case):
    from datetime import timedelta

    import torch.distributed as dist

    from rl.sync.pair_group import destroy_pair_group, init_pair_group

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}-{rank}", rank=0, world_size=1,
                            timeout=timedelta(seconds=30))  # fmt: skip
    group, _ = init_pair_group("gloo", "127.0.0.1", port, rank, timeout=30)
    sync = None
    try:
        store = MemoryStore(SCHEMA, keep=rank == 1, digest="sha256")
        # Receiver buckets differ from the source's in the layout case: rejected before data moves.
        bucket = 192 if case == "layout" and rank == 1 else 128
        sync = BucketedSync(
            store, group=group, source=0, bucket_bytes=bucket, timeout=30, threads=3
        )
        if case == "layout":
            with pytest.raises(RuntimeError, match="rejected"):
                sync.push(shards(2.0) if rank == 0 else None, 1)
            assert store.version is None and not sync.failed
            return
        stats = sync.push(shards(2.0) if rank == 0 else None, 1)
        assert stats.bytes == sync.plan.tensor_bytes
        assert sync.last["buckets"] == len(sync.plan.buckets())
        assert sync.last["collectives"] == len(sync.plan.buckets()) + 10
        first = store.digest
        if rank == 1:
            got = dict(store.take(1))
            for name, tensor in shards(2.0):
                assert got[name].dtype == tensor.dtype and torch.equal(got[name], tensor)
        if case == "corrupt":
            original = BucketedSync._landed

            def flip(self, hasher, flat, start, end, work, wait):
                if rank == 1 and start == 0:
                    wait(work)
                    flat[self.plan.slots[0].offset] ^= 1  # one bit of a.weight, after the wire
                original(self, hasher, flat, start, end, work, wait)

            BucketedSync._landed = flip
            try:
                with pytest.raises(RuntimeError, match="rejected") as info:
                    sync.push(shards(3.0) if rank == 0 else None, 2)
            finally:
                BucketedSync._landed = original
            if rank == 1:  # the receiver's own check names the corrupted chunk
                assert "content differs in 1 of" in str(info.value.__cause__)
            assert store.version == 1 and store.digest == first  # never committed
            assert not sync.failed
            # the owner stays usable: the next clean version goes through
            sync.push(shards(4.0) if rank == 0 else None, 3)
            assert store.version == 3
            if rank == 1:
                assert torch.equal(dict(store.take(3))["f.big"], dict(shards(4.0))["f.big"])
        if case == "nonfinite":
            bad = [(n, t.clone()) for n, t in shards(3.0)]
            bad[1][1][0] = float("inf")
            with pytest.raises(RuntimeError, match="rejected") as info:
                sync.push(bad if rank == 0 else None, 2)
            assert store.version == 1 and "finite" in str(info.value.__cause__)
    finally:
        if sync is not None:
            sync.close()
        destroy_pair_group(group)
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["clean", "corrupt", "layout", "nonfinite"])
def test_bucketed_sync_over_pair_group(tmp_path, case):
    context = mp.get_context("spawn")
    port = free_port()
    children = [
        context.Process(target=_pair_rank, args=(rank, port, str(tmp_path / "rdv"), case))
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
