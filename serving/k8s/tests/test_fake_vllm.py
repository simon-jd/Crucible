import http.client
import json
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

from serving.k8s import fake_vllm
from serving.localscale import metrics


@pytest.fixture
def server():
    engine = fake_vllm.Engine(max_num_seqs=1, tpot_ms=1, prefill_ms_per_token=0, model="m")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), fake_vllm.make_handler(engine))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield engine, srv.server_address[1]
    srv.shutdown()


def complete(port, n, prompt="a b c"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    body = json.dumps({"model": "m", "prompt": prompt, "max_tokens": n, "stream": True, "return_token_ids": True})
    conn.request("POST", "/v1/completions", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    events = [line[6:] for line in resp.read().decode().split("\n") if line.startswith("data: ")]
    conn.close()
    return resp.status, events


def get(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", path)
    resp = conn.getresponse()
    out = resp.status, resp.read().decode()
    conn.close()
    return out


def test_stream_matches_what_the_replayer_parses(server):
    _, port = server
    status, events = complete(port, 3)
    assert status == 200 and events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    tokens = [c["choices"][0] for c in chunks if c["choices"]]
    assert [len(t["token_ids"]) for t in tokens] == [1, 1, 1]
    assert [t["finish_reason"] for t in tokens] == [None, None, "length"]
    assert chunks[-1] == {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 3}}


def test_metrics_parse_with_the_project_vllm_parser(server):
    _, port = server
    complete(port, 2)
    status, text = get(port, "/metrics")
    sample = metrics.parse(text)
    assert status == 200 and sample.success == 1 and sample.waiting == 0 and sample.running == 0
    assert sample.ttft_buckets[float("inf")] == 1


def test_requests_beyond_the_slots_wait(server):
    engine, port = server
    threads = [threading.Thread(target=complete, args=(port, 200)) for _ in range(3)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + 5
    seen = 0
    while time.monotonic() < deadline and seen < 2:
        seen = metrics.parse(get(port, "/metrics")[1]).waiting
        time.sleep(0.01)
    for t in threads:
        t.join()
    assert seen == 2 and engine.success == 3


def test_draining_replica_reports_unhealthy_but_still_serves(server):
    engine, port = server
    engine.draining = True
    assert get(port, "/health")[0] == 503 and get(port, "/metrics")[0] == 503
    assert complete(port, 1)[0] == 200


def test_rejects_non_streaming_requests(server):
    _, port = server
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/v1/completions", json.dumps({"prompt": "a", "max_tokens": 1}))
    resp = conn.getresponse()
    resp.read()
    assert resp.status == 400
    conn.close()
