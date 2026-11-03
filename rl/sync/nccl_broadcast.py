"""Complete tensor broadcast with receiver preparation acknowledgements.

Actual gloo CPU validation; NCCL CUDA path is implemented but unverified. The
caller owns/destroys the process group. Lost collectives invalidate this owner;
no distributed atomic-commit claim is made after lost ACK or process death.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from rl.sync.version import canonical, check_time

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class SyncStats:
    seconds: float
    bytes: int


class WeightSync:
    def __init__(self, store, *, group=None, source=0, timeout=120):
        if not isinstance(timeout, (float, int)) or not 0 < timeout <= 4 * 3600:
            raise ValueError("positive whole-transaction timeout required")
        if type(source) is not int or source < 0:
            raise ValueError("source must be a global rank")
        self.store, self.group, self.source, self.timeout = store, group, source, timeout
        self.failed = False

    def push(self, shards: Iterable[tuple[str, Tensor]], version: int) -> SyncStats:
        """Prepare a candidate only. engine.load_weights(version) activates it.

        Every rank calls push; only source supplies shards. Dense tensors must
        match the full schema exactly. No dtype conversion/omission is implicit.
        Blocking filesystem calls are checked before/after, not forcibly killed.
        """
        if self.failed:
            raise RuntimeError("transport owner failed; destroy group and recover phase")
        start = time.monotonic()
        deadline = start + self.timeout
        with self.store._lock:
            if self.group is None:
                stage = self.store.begin(version, deadline)
                try:
                    for name, tensor in shards:
                        stage.add(name, tensor)
                    stage.prepare()
                    stage.commit()
                    check_time(deadline)
                    return SyncStats(time.monotonic() - start, stage.bytes)
                except BaseException as failure:
                    if stage.committed:
                        self.failed = True
                        self.store._in_doubt = True
                    self._abort(stage, failure)
                    raise
            return self._distributed(shards, version, start, deadline)

    def _abort(self, stage, failure):
        try:
            stage.abort()
        except BaseException as cleanup_error:  # noqa: BLE001 - retain both failure boundaries
            self.failed = True
            self.store._in_doubt = True
            raise BaseExceptionGroup(
                "synchronization and staging cleanup failed", [failure, cleanup_error]
            )

    def _distributed(self, shards, version, start, deadline):
        import torch
        import torch.distributed as dist

        if self.source not in dist.get_process_group_ranks(self.group):
            raise ValueError("source rank is not in caller-owned group")
        backend = dist.get_backend(self.group)
        if backend not in ("gloo", "nccl"):
            raise ValueError("only gloo CPU or NCCL CUDA supported")
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if backend == "nccl"
            else torch.device("cpu")
        )
        source = dist.get_rank() == self.source
        stage = None
        transport_failed = False
        commit_started = False

        def wait(work):
            nonlocal transport_failed
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
            if len(raw) > 65536:
                raise ValueError("control frame exceeds 64KiB")
            size = torch.tensor([len(raw)], dtype=torch.int64, device=device)
            wait(dist.broadcast(size, src=self.source, group=self.group, async_op=True))
            n = size.item()
            if not 0 < n <= 65536:
                raise ValueError("invalid control size")
            data = (
                torch.tensor(list(raw), dtype=torch.uint8, device=device)
                if source
                else torch.empty(n, dtype=torch.uint8, device=device)
            )
            wait(dist.broadcast(data, src=self.source, group=self.group, async_op=True))
            return json.loads(bytes(data.cpu().tolist()))

        try:
            error = None
            try:
                if len(canonical({"version": version, "contract": self.store.contract()})) > 65536:
                    raise ValueError("complete control contract exceeds 64KiB")
                stage = self.store.begin(version, deadline)
            except Exception as exc:  # noqa: BLE001 - propagate all ordinary rank failures
                error = exc
            status(error)
            expected = control({"version": version, "contract": self.store.contract()})
            status(
                None
                if expected == {"version": version, "contract": self.store.contract()}
                else ValueError("schema/config/version mismatch")
            )
            iterator, error = None, None
            if source:
                try:
                    iterator = iter(shards)
                except Exception as exc:  # noqa: BLE001 - communicate source rejection
                    error = exc
            status(error)
            while True:
                tensor, error, message = None, None, None
                if source:
                    try:
                        name, tensor = next(iterator)
                        if name not in self.store.schema or name in stage.seen:
                            raise ValueError("duplicate or unexpected tensor")
                        shape, dtype = self.store.schema[name]
                        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
                            raise ValueError("shape/dtype mismatch")
                        tensor = tensor.detach().to(device=device).contiguous()
                        message = {"name": name}
                    except StopIteration:
                        message = {"end": True}
                    except Exception as exc:  # noqa: BLE001 - propagate all ordinary rank failures
                        error, message = exc, {"error": str(exc)[:1024]}
                message = control(message)
                if "error" in message:
                    raise RuntimeError("source failed: " + message["error"]) from error
                if "end" in message:
                    break
                name = message["name"]
                if not source:
                    try:
                        shape, dtype = self.store.schema[name]
                        tensor = torch.empty(shape, dtype=dtype, device=device)
                    except Exception as exc:  # noqa: BLE001 - propagate all ordinary rank failures
                        error = exc
                status(error)
                # Bound collective payloads to 16MiB; exact dtype retained.
                flat = tensor.reshape(-1)
                chunk = max(1, 16 * 1024 * 1024 // tensor.element_size())
                for offset in range(0, flat.numel(), chunk):
                    wait(
                        dist.broadcast(
                            flat[offset : offset + chunk],
                            src=self.source,
                            group=self.group,
                            async_op=True,
                        )
                    )
                error = None
                try:
                    stage.add(name, tensor)
                except Exception as exc:  # noqa: BLE001 - propagate all ordinary rank failures
                    error = exc
                status(error)
                del flat, tensor
            error = None
            try:
                stage.prepare()
            except Exception as exc:  # noqa: BLE001 - propagate all ordinary rank failures
                error = exc
            status(error)  # all receivers have durable complete candidates
            digest = control({"digest": stage.digest})
            status(None if digest["digest"] == stage.digest else ValueError("content differs"))
            error = None
            commit_started = True
            try:
                stage.commit()
            except Exception as exc:  # noqa: BLE001 - propagate all ordinary rank failures
                error = exc
            status(error)  # catalog ACK, never an engine-active ACK
            check_time(deadline)
            return SyncStats(time.monotonic() - start, stage.bytes)
        except BaseException as failure:
            if transport_failed or commit_started:
                self.failed = True
                self.store._in_doubt = True
            if stage is not None:
                self._abort(stage, failure)
            raise
