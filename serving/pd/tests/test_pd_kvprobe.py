import json
import threading
from dataclasses import dataclass

import pytest

from serving.pd import kvprobe


class FakeTensor:
    def __init__(self, n, elem=2):
        self.n, self.elem = n, elem

    def element_size(self):
        return self.elem

    def numel(self):
        return self.n


@dataclass
class Item:
    tensor_id: str
    remote_address: str
    tensor: FakeTensor


class FakeEngine:
    """The four methods kvprobe wraps, with P2pNcclEngine's call pattern (vLLM 0.10.2)."""

    def __init__(self):
        self.recv_store, self.sent, self.received = {}, [], []
        self.cv = threading.Condition()

    def send_sync(self, item):
        self.sent.append(item.tensor_id)
        return item.remote_address != "bad"

    def recv(self, comm, tensor, src, stream=None):
        return None

    def have_received_tensor_id(self, tensor_id):
        self.received.append(tensor_id)

    def recv_tensor(self, tensor_id, remote_address=None):
        return self.recv_store.get(tensor_id)

    # what listen_for_requests does for one PUT
    def on_put(self, tensor_id, tensor, pooled):
        self.recv(None, tensor, 0)
        self.recv_store[tensor_id] = ("addr", "dtype", "shape") if pooled else tensor
        self.have_received_tensor_id(tensor_id)


@pytest.fixture
def engine_cls():
    cls = type("Engine", (FakeEngine,), {})
    return kvprobe.install(cls)


def test_probe_records_send_recv_and_load_without_changing_results(engine_cls, tmp_path, monkeypatch):
    monkeypatch.setenv(kvprobe.ENV, str(tmp_path))
    kvprobe._file.clear()
    e = engine_cls()
    assert e.send_sync(Item("r1#layer.0", "10.0.0.2:12985", FakeTensor(100))) is True
    assert e.send_sync(Item("r1#layer.1", "bad", FakeTensor(100))) is False
    e.on_put("r1#layer.0", FakeTensor(50), pooled=False)
    e.on_put("r1#layer.1", FakeTensor(50), pooled=True)
    assert e.recv_tensor("r1#layer.0") is not None and e.recv_tensor("r2#layer.0") is None
    assert e.sent == ["r1#layer.0", "r1#layer.1"] and e.received == ["r1#layer.0", "r1#layer.1"]
    assert kvprobe.install(engine_cls) is engine_cls  # idempotent: no double wrapping
    files = sorted(tmp_path.glob("kv-*.jsonl"))
    assert len(files) == 1
    events = kvprobe.read_events(files)
    assert [ev["ev"] for ev in events] == ["send", "send", "recv", "recv", "load", "load"]
    rows = kvprobe.summarize(events)
    r1 = rows["r1"]
    assert r1["send_tensors"] == 2 and r1["send_bytes"] == 400 and r1["send_failed"] == 1
    assert r1["recv_tensors"] == 2 and r1["recv_bytes"] == 200 and r1["pooled"] == 1
    assert r1["load_tensors"] == 1 and r1["load_misses"] == 0 and r1["send_span_s"] >= 0
    assert rows["r2"]["load_misses"] == 1
    json.dumps(rows)


def test_probe_is_silent_without_the_directory(engine_cls, tmp_path, monkeypatch):
    monkeypatch.delenv(kvprobe.ENV, raising=False)
    kvprobe._file.clear()
    e = engine_cls()
    e.send_sync(Item("r#l", "x:1", FakeTensor(1)))
    assert not list(tmp_path.iterdir())
    assert kvprobe.install_if_enabled() is False


def test_probe_wraps_the_installed_vllm_engine_class_when_available(monkeypatch, tmp_path):
    mod = pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine")
    for name in ("send_sync", "recv", "have_received_tensor_id", "recv_tensor"):
        assert callable(getattr(mod.P2pNcclEngine, name))  # the methods kvprobe expects still exist
    monkeypatch.setenv(kvprobe.ENV, str(tmp_path))
    assert kvprobe.install_if_enabled() is True and mod.P2pNcclEngine._crucible_probe
