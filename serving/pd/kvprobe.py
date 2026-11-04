"""Per-tensor KV transfer timing for vLLM 0.10.2's P2pNcclEngine, written as JSON lines.

vLLM does not expose KV transfer bytes or times for P2pNcclConnector, so `install` wraps four
engine methods (it does not change what they do):

- ``send_sync`` (prefill, sender thread): one "send" event per (request, layer) tensor with its
  bytes and wall time, which covers the ZMQ PUT handshake, ncclSend and the stream sync;
- ``recv`` + ``have_received_tensor_id`` (decode, listener thread): one "recv" event per tensor
  with the ncclRecv wall time, and whether it was parked in the pinned host pool;
- ``recv_tensor`` (decode, model runner): one "load" event per tensor with the time the forward
  pass waited for it (plus the host-to-GPU copy when it came from the pool).

Events go to $CRUCIBLE_KVPROBE_DIR/kv-<pid>.jsonl. `summarize` turns them into per-request rows.
"""

import json
import os
import threading
import time

ENV = "CRUCIBLE_KVPROBE_DIR"
MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine"
_lock = threading.Lock()
_file = {}


def _emit(rec):
    d = os.environ.get(ENV)
    if not d:
        return
    with _lock:
        pid = os.getpid()
        f = _file.get(pid)
        if f is None:
            os.makedirs(d, exist_ok=True)
            f = _file[pid] = open(os.path.join(d, f"kv-{pid}.jsonl"), "a")  # noqa: SIM115 - process lifetime
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        f.flush()


def _nbytes(t):
    return int(t.element_size() * t.numel())


def install(engine_cls):
    """Wrap the engine's transfer methods; idempotent."""
    if getattr(engine_cls, "_crucible_probe", False):
        return engine_cls
    send_sync, recv, have_received, recv_tensor = (
        engine_cls.send_sync, engine_cls.recv, engine_cls.have_received_tensor_id, engine_cls.recv_tensor)  # fmt: skip

    def probed_send_sync(self, item):
        t0 = time.time()
        ok = send_sync(self, item)
        _emit({"ev": "send", "id": item.tensor_id, "bytes": _nbytes(item.tensor), "t0": t0, "t1": time.time(),
               "ok": bool(ok)})
        return ok

    def probed_recv(self, comm, tensor, src, stream=None):
        t0 = time.time()
        out = recv(self, comm, tensor, src, stream)
        self._crucible_last = (t0, time.time(), _nbytes(tensor))
        return out

    def probed_have_received(self, tensor_id):
        last, self._crucible_last = getattr(self, "_crucible_last", None), None
        if last is not None:
            pooled = isinstance(self.recv_store.get(tensor_id), tuple)
            _emit({"ev": "recv", "id": tensor_id, "bytes": last[2], "t0": last[0], "t1": last[1], "pool": pooled})
        return have_received(self, tensor_id)

    def probed_recv_tensor(self, tensor_id, remote_address=None):
        t0 = time.time()
        out = recv_tensor(self, tensor_id, remote_address)
        _emit({"ev": "load", "id": tensor_id, "t0": t0, "t1": time.time(), "hit": out is not None})
        return out

    engine_cls.send_sync = probed_send_sync
    engine_cls.recv = probed_recv
    engine_cls.have_received_tensor_id = probed_have_received
    engine_cls.recv_tensor = probed_recv_tensor
    engine_cls._crucible_probe = True
    return engine_cls


def install_if_enabled():
    if not os.environ.get(ENV):
        return False
    import importlib

    install(importlib.import_module(MODULE).P2pNcclEngine)
    return True


def read_events(paths):
    events = []
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    return events


def summarize(events):
    """Per request id: tensors, bytes, send/recv spans and summed times, decode-side wait."""
    out = {}
    for e in events:
        rid = e["id"].split("#", 1)[0]
        r = out.setdefault(rid, {"id": rid})
        kind = e["ev"]
        if kind == "send":
            r["send_tensors"] = r.get("send_tensors", 0) + 1
            r["send_bytes"] = r.get("send_bytes", 0) + e["bytes"]
            r["send_s"] = r.get("send_s", 0.0) + (e["t1"] - e["t0"])
            r["send_t0"] = min(r.get("send_t0", e["t0"]), e["t0"])
            r["send_t1"] = max(r.get("send_t1", e["t1"]), e["t1"])
            r["send_failed"] = r.get("send_failed", 0) + (not e["ok"])
        elif kind == "recv":
            r["recv_tensors"] = r.get("recv_tensors", 0) + 1
            r["recv_bytes"] = r.get("recv_bytes", 0) + e["bytes"]
            r["recv_s"] = r.get("recv_s", 0.0) + (e["t1"] - e["t0"])
            r["recv_t1"] = max(r.get("recv_t1", e["t1"]), e["t1"])
            r["pooled"] = r.get("pooled", 0) + bool(e["pool"])
        elif kind == "load":
            r["load_tensors"] = r.get("load_tensors", 0) + 1
            r["load_wait_s"] = r.get("load_wait_s", 0.0) + (e["t1"] - e["t0"])
            r["load_misses"] = r.get("load_misses", 0) + (not e["hit"])
    for r in out.values():
        if "send_t0" in r:
            r["send_span_s"] = r["send_t1"] - r["send_t0"]
    return out
