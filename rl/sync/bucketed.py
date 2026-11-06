"""Bucketed weight sync: the whole BF16 state in a few large broadcasts.

`WeightSync` (rl/sync/nccl_broadcast.py) sends every tensor with its own control frame,
status all-reduces and checksum. One push of Qwen2.5-0.5B (290 tensors) is 1,478 sequential
collectives, and between two nodes over TCP that protocol, not the link, took most of the
~2 s per step (docs/evidence/snowflake/e2-async-grpo.md). `BucketedSync` has the same
interface (every rank calls `push(shards, version) -> SyncStats`, only the source supplies
shards, the receiver `take`s the committed tensors from its `MemoryStore`):

- Layout. Both ranks derive the same `BucketPlan` from the store's schema: tensors in sorted
  name order, each at a 256-byte aligned offset of one flat byte buffer, which is cut into
  fixed-size buckets (the last one shorter).
- Pack. The source copies every tensor into its slot: exact dtype and shape, no cast, every
  schema name exactly once. Padding is zero, so the packed bytes are deterministic.
- Transfer. One broadcast per bucket, straight into the receiver's flat buffer. Unpacking
  makes zero-copy views with the schema's shapes and dtypes; `model.load_weights` copies them
  into the engine.
- Integrity, equivalent to WeightSync with `MemoryStore(digest="sha256")`: the source sends
  {version, contract SHA-256, layout SHA-256} and the receiver rejects a mismatch before any
  data moves. Each rank SHA-256-hashes the bytes it holds in chunks of at most 16 MiB that
  never cross a bucket edge; on CUDA a side stream copies each bucket to pinned host memory as
  soon as it is final and a thread pool hashes it while later buckets are still on the wire.
  Floating tensors must be finite on both ranks (`MemoryStage.adopt`). The source then sends
  its manifest {version, contract, layout, chunk hashes, sizes}; the receiver commits only if
  it equals the manifest of the bytes it received, and both record SHA-256 of the canonical
  manifest as the version digest.

Collectives per push: n_buckets + 10 (6 four-byte status all-reduces, 2 control frames of two
broadcasts each). Memory: the source holds the packed copy (plan.total_bytes) next to its
shards; the receiver holds one flat buffer, as much as the per-tensor path's tensors; each
rank keeps one pinned host mirror of plan.total_bytes for hashing (CUDA only).

Failure semantics follow WeightSync: an ordinary rejection (layout, pack, finite, content)
aborts the candidate on every rank and the owner stays usable; a lost collective or a failure
after commit started marks the owner failed and the store in doubt. Single owner, not thread
safe beyond the store lock. Validated on CPU over gloo; the NCCL path was run on Snowflake
A10 nodes (docs/evidence/snowflake/e2b-bucketed-sync.md).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta

from rl.sync.nccl_broadcast import SyncStats
from rl.sync.version import canonical, check_time

ALIGN = 256
DIGEST_CHUNK = 16 << 20
CONTROL_LIMIT = 65536


@dataclass(frozen=True)
class Slot:
    name: str
    offset: int
    nbytes: int
    shape: tuple
    dtype: object  # torch.dtype


@dataclass(frozen=True)
class BucketPlan:
    slots: tuple
    total_bytes: int
    bucket_bytes: int
    align: int = ALIGN
    digest_chunk: int = DIGEST_CHUNK

    @property
    def tensor_bytes(self):
        return sum(slot.nbytes for slot in self.slots)

    def buckets(self):
        """[(start, end)) byte ranges, one broadcast each."""
        step = self.bucket_bytes
        return [(s, min(s + step, self.total_bytes)) for s in range(0, self.total_bytes, step)]

    def chunks(self):
        """[(start, end)) hash ranges: at most digest_chunk bytes, never across a bucket edge."""
        out = []
        for start, end in self.buckets():
            out += [
                (a, min(a + self.digest_chunk, end)) for a in range(start, end, self.digest_chunk)
            ]
        return out

    def describe(self):
        return {
            "slots": [
                [s.name, s.offset, s.nbytes, list(s.shape), str(s.dtype)] for s in self.slots
            ],
            "total_bytes": self.total_bytes,
            "bucket_bytes": self.bucket_bytes,
            "align": self.align,
            "digest_chunk": self.digest_chunk,
        }

    def digest(self):
        return hashlib.sha256(canonical(self.describe())).hexdigest()


def plan_buckets(schema, bucket_bytes, *, align=ALIGN, digest_chunk=DIGEST_CHUNK):
    """Deterministic layout of a complete schema {name: (shape, dtype)} in one flat byte buffer."""
    import torch

    for value, what in ((bucket_bytes, "bucket_bytes"), (digest_chunk, "digest_chunk")):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{what} must be a positive int")
    if type(align) is not int or align < 8 or align & (align - 1):
        raise ValueError("align must be a power of two >= 8")
    if not schema:
        raise ValueError("complete nonempty schema required")
    slots, offset = [], 0
    for name in sorted(schema):
        shape, dtype = schema[name]
        shape = tuple(int(d) for d in shape)
        nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        offset = -(-offset // align) * align
        slots.append(Slot(name, offset, nbytes, shape, dtype))
        offset += nbytes
    total = -(-offset // align) * align
    if total == 0:
        raise ValueError("schema holds no bytes")
    return BucketPlan(tuple(slots), total, bucket_bytes, align, digest_chunk)


def _slot_view(flat, slot):
    return flat[slot.offset : slot.offset + slot.nbytes].view(slot.dtype).view(slot.shape)


def pack(plan, shards, flat):
    """Copy (name, tensor) shards into their slots of `flat` (uint8, plan.total_bytes, zeroed)."""
    import torch

    slots = {slot.name: slot for slot in plan.slots}
    seen = set()
    for name, tensor in shards:
        slot = slots.get(name)
        if slot is None or name in seen:
            raise ValueError(f"duplicate or unexpected tensor {name!r}")
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError("dense Tensor required")
        if tuple(tensor.shape) != slot.shape or tensor.dtype != slot.dtype:
            raise ValueError(f"shape/dtype mismatch for {name!r}; no implicit cast")
        _slot_view(flat, slot).copy_(tensor.detach())
        seen.add(name)
    if seen != set(slots):
        raise ValueError(f"incomplete parameter set: {len(slots) - len(seen)} tensors missing")


def unpack(plan, flat):
    """[(name, view)] in schema-sorted order; the views share flat's memory."""
    return [(slot.name, _slot_view(flat, slot)) for slot in plan.slots]


def make_manifest(version, contract_sha256, plan, chunk_sha256):
    return {
        "version": version,
        "contract_sha256": contract_sha256,
        "layout_sha256": plan.digest(),
        "chunk_sha256": list(chunk_sha256),
        "tensor_bytes": plan.tensor_bytes,
        "total_bytes": plan.total_bytes,
    }


def manifest_digest(manifest):
    return hashlib.sha256(canonical(manifest)).hexdigest()


def check_manifest(source, local):
    """Raise ValueError naming the first difference between the source's and this rank's manifest."""
    if source == local:
        return
    for key in sorted(set(source) | set(local)):
        if key != "chunk_sha256" and source.get(key) != local.get(key):
            raise ValueError(f"manifest field {key} differs")
    theirs, mine = source.get("chunk_sha256") or [], local.get("chunk_sha256") or []
    if len(theirs) != len(mine):
        raise ValueError("manifest chunk count differs")
    bad = [i for i, (a, b) in enumerate(zip(theirs, mine, strict=True)) if a != b]
    raise ValueError(f"content differs in {len(bad)} of {len(mine)} chunks (first: chunk {bad[0]})")


def _sha256(buffer, start, end, event):
    if event is not None:
        event.synchronize()
    return hashlib.sha256(buffer[start:end]).hexdigest()


class _Hasher:
    """SHA-256 of plan.chunks() of one flat buffer, scheduled bucket by bucket as bytes become final."""

    def __init__(self, pool, plan, flat, host, stream):
        self.pool, self.plan, self.flat, self.host, self.stream = pool, plan, flat, host, stream
        self.chunks = plan.chunks()
        self.futures = [None] * len(self.chunks)
        self._next = 0

    def land(self, start, end):
        """Bytes [start, end) of flat are final (on CUDA: once the side stream reaches this point)."""
        import torch

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.host[start:end].copy_(self.flat[start:end], non_blocking=True)
                event = torch.cuda.Event()
                event.record(self.stream)
            buffer = self.host.numpy()
        else:
            event, buffer = None, self.flat.numpy()
        while self._next < len(self.chunks) and self.chunks[self._next][1] <= end:
            a, b = self.chunks[self._next]
            if a < start:
                raise RuntimeError("bucket landed out of order")
            self.futures[self._next] = self.pool.submit(_sha256, buffer, a, b, event)
            self._next += 1

    def result(self, deadline):
        if self._next != len(self.chunks):
            raise RuntimeError("not every bucket landed")
        return [f.result(timeout=max(0.001, deadline - time.monotonic())) for f in self.futures]

    def drain(self):
        """Cancel or finish outstanding hashes, so the host mirror can be reused."""
        for future in self.futures:
            if future is not None and not future.cancel():
                try:
                    future.result(timeout=60)
                except Exception:  # noqa: BLE001, S110 - the push already failed; nothing to keep
                    pass


class BucketedSync:
    def __init__(
        self, store, *, group, source=0, bucket_bytes, timeout=120, threads=None, host=None
    ):
        """host: optional pinned uint8 CPU tensor of >= plan.total_bytes, shared by owners that
        never push at the same time (the benchmark's several bucket sizes); default: own one."""
        if group is None:
            raise ValueError("BucketedSync needs a caller-owned process group")
        if not isinstance(timeout, (float, int)) or not 0 < timeout <= 4 * 3600:
            raise ValueError("positive whole-transaction timeout required")
        if type(source) is not int or source < 0:
            raise ValueError("source must be a global rank")
        if getattr(store, "digest_mode", None) != "sha256":
            raise ValueError("BucketedSync verifies SHA-256; use MemoryStore(..., digest='sha256')")
        self.store, self.group, self.source, self.timeout = store, group, source, timeout
        self.plan = plan_buckets(store.schema, bucket_bytes)
        # Refuse up front what the source could only discover mid-push (the receiver would hang).
        widest = make_manifest(2**62, "0" * 64, self.plan, ["0" * 64] * len(self.plan.chunks()))
        if len(canonical(widest)) > CONTROL_LIMIT:
            raise ValueError("manifest exceeds the 64KiB control frame; use larger buckets")
        self.threads = threads or max(1, min(8, os.cpu_count() or 1))
        self._pool = ThreadPoolExecutor(self.threads, thread_name_prefix="bucket-sha256")
        if host is not None:
            import torch

            if (
                host.dtype != torch.uint8
                or host.device.type != "cpu"
                or host.numel() < self.plan.total_bytes
            ):
                raise ValueError(
                    "shared host buffer must be a CPU uint8 tensor of >= plan.total_bytes"
                )
        self._host, self._stream = host, None
        self.failed = False
        self.last = None

    def close(self):
        self._pool.shutdown(wait=True)

    def _cuda_resources(self):
        """Pinned host mirror and copy stream, allocated once per owner (not inside a timed push)."""
        import torch

        if self._host is None:
            self._host = torch.empty(self.plan.total_bytes, dtype=torch.uint8, pin_memory=True)
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        return self._host, self._stream

    def warm(self):
        """Allocate the CUDA-side resources now (call after the group exists, before timing)."""
        import torch.distributed as dist

        if dist.get_backend(self.group) == "nccl":
            self._cuda_resources()

    def _landed(self, hasher, flat, start, end, work, wait):
        """Receiver: bucket [start, end) is on the wire as `work`; hash it once it has landed."""
        import torch

        if hasher.stream is not None:
            flat.record_stream(hasher.stream)
            with torch.cuda.stream(hasher.stream):
                wait(work)  # the copy stream (not the host) waits for the broadcast
        else:
            wait(work)
        hasher.land(start, end)

    def push(self, shards, version) -> SyncStats:
        """Prepare a candidate only; the receiver's engine activates it after store.take."""
        if self.failed:
            raise RuntimeError("transport owner failed; destroy group and recover phase")
        start = time.monotonic()
        deadline = start + self.timeout
        with self.store._lock:
            return self._distributed(shards, version, start, deadline)

    def _distributed(self, shards, version, start, deadline):
        import torch
        import torch.distributed as dist

        if self.source not in dist.get_process_group_ranks(self.group):
            raise ValueError("source rank is not in caller-owned group")
        backend = dist.get_backend(self.group)
        if backend not in ("gloo", "nccl"):
            raise ValueError("only gloo CPU or NCCL CUDA supported")
        cuda = backend == "nccl"
        device = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")
        source = dist.get_global_rank(self.group, self.group.rank()) == self.source
        plan = self.plan
        stage, hasher = None, None
        transport_failed = commit_started = False
        count = [0]
        marks = {}

        def wait(work):
            nonlocal transport_failed
            count[0] += 1
            try:
                check_time(deadline)
                work.wait(timeout=timedelta(seconds=max(0.001, deadline - time.monotonic())))
                check_time(deadline)
            except BaseException:
                transport_failed = True
                raise

        def status(error):
            flag = torch.tensor([int(error is not None)], dtype=torch.int32, device=device)
            wait(dist.all_reduce(flag, group=self.group, async_op=True))
            if flag.item():
                raise RuntimeError("receiver/source rejected synchronization") from error

        def control(value):
            raw = canonical(value) if source else b""
            if len(raw) > CONTROL_LIMIT:
                raise ValueError("control frame exceeds 64KiB")
            size = torch.tensor([len(raw)], dtype=torch.int64, device=device)
            wait(dist.broadcast(size, src=self.source, group=self.group, async_op=True))
            n = size.item()
            if not 0 < n <= CONTROL_LIMIT:
                raise ValueError("invalid control size")
            data = (
                torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
                if source
                else torch.empty(n, dtype=torch.uint8, device=device)
            )
            wait(dist.broadcast(data, src=self.source, group=self.group, async_op=True))
            return json.loads(bytes(data.cpu().numpy()))

        try:
            error, flat = None, None
            try:
                stage = self.store.begin(version, deadline)
                make = torch.zeros if source else torch.empty
                flat = make(plan.total_bytes, dtype=torch.uint8, device=device)
            except Exception as exc:  # noqa: BLE001 - propagate all ordinary rank failures
                error = exc
            status(error)
            contract = hashlib.sha256(canonical(self.store.contract())).hexdigest()
            header = {
                "version": version,
                "contract_sha256": contract,
                "layout_sha256": plan.digest(),
            }
            got = control(header)
            status(None if got == header else ValueError("schema/config/version/layout mismatch"))
            error = None
            if source:
                try:
                    pack(plan, shards, flat)
                except Exception as exc:  # noqa: BLE001 - communicate source rejection
                    error = exc
            status(error)  # on CUDA this all-reduce also waits for the pack kernels
            marks["packed"] = time.monotonic()
            host, stream = self._cuda_resources() if cuda else (None, None)
            hasher = _Hasher(self._pool, plan, flat, host, stream)
            if source:
                if stream is not None:
                    stream.wait_stream(torch.cuda.current_stream())
                    flat.record_stream(stream)
                for a, b in plan.buckets():
                    hasher.land(a, b)
            for a, b in plan.buckets():
                work = dist.broadcast(flat[a:b], src=self.source, group=self.group, async_op=True)
                if source:
                    wait(work)
                else:
                    self._landed(hasher, flat, a, b, work, wait)
            if cuda and not source:
                torch.cuda.current_stream().wait_stream(stream)  # views are read on this stream
            marks["sent"] = time.monotonic()
            error, mine = None, None
            try:
                digests = hasher.result(deadline)
                marks["hashed"] = time.monotonic()
                mine = make_manifest(version, contract, plan, digests)
                stage.adopt(unpack(plan, flat), manifest_digest(mine), plan.tensor_bytes)
            except Exception as exc:  # noqa: BLE001 - propagate all ordinary rank failures
                error = exc
            status(error)  # every rank holds a complete, finite candidate and its hashes
            theirs = control(mine)
            error = None
            try:
                check_manifest(theirs, mine)
            except ValueError as exc:
                error = exc
            status(error)
            error = None
            commit_started = True
            try:
                stage.commit()
            except Exception as exc:  # noqa: BLE001 - propagate all ordinary rank failures
                error = exc
            status(error)  # catalog ACK, never an engine-active ACK
            check_time(deadline)
            end = time.monotonic()
            self.last = {
                "seconds": end - start,
                "pack_s": marks["packed"] - start,
                "transfer_s": marks["sent"] - marks["packed"],
                "hash_wait_s": marks["hashed"] - marks["sent"],
                "verify_commit_s": end - marks["hashed"],
                "buckets": len(plan.buckets()),
                "bucket_bytes": plan.bucket_bytes,
                "chunks": len(hasher.chunks),
                "collectives": count[0],
                "tensor_bytes": plan.tensor_bytes,
                "total_bytes": plan.total_bytes,
                "digest": stage.digest,
            }
            return SyncStats(end - start, plan.tensor_bytes)
        except BaseException as failure:
            if hasher is not None:
                hasher.drain()
            if transport_failed or commit_started:
                self.failed = True
                self.store._in_doubt = True
            if stage is not None:
                try:
                    stage.abort()
                except BaseException as cleanup_error:  # noqa: BLE001 - retain both failure boundaries
                    self.failed = True
                    self.store._in_doubt = True
                    raise BaseExceptionGroup(
                        "synchronization and staging cleanup failed", [failure, cleanup_error]
                    ) from None
            raise


def make_weight_sync(schema, *, assets=None, keep, group, source=0, timeout=120, bucket_mb=0):
    """(store, sync): per-tensor WeightSync with sum64 (bucket_mb=0) or BucketedSync with SHA-256."""
    from rl.sync.memory_store import MemoryStore
    from rl.sync.nccl_broadcast import WeightSync

    if bucket_mb < 0:
        raise ValueError("bucket_mb must be >= 0")
    if not bucket_mb:
        store = MemoryStore(schema, assets=assets, keep=keep)
        return store, WeightSync(store, group=group, source=source, timeout=timeout)
    store = MemoryStore(schema, assets=assets, keep=keep, digest="sha256")
    sync = BucketedSync(
        store, group=group, source=source, bucket_bytes=int(bucket_mb * (1 << 20)), timeout=timeout
    )
    sync.warm()
    return store, sync
