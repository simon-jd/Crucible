"""Metrics adapter between local vLLM replicas and the Go router/autoscaler.

`GET /snapshot/<id>` returns the router snapshot (README "router adapter" schema) for a
replica that is serving: waiting queue, no prefix claims (prefix caching is off).
A replica that is starting, draining or stopped answers 503, so the router stops
selecting it once its last snapshot ages out (15 s). `GET /aggregate` returns the
autoscaler metrics: waiting requests per desired replica and p95 TTFT over the last
window, from vLLM's TTFT histogram deltas summed over replicas. With no TTFT samples
in the window p95 is null, which the autoscaler treats as missing, not as idle.
"""

import datetime
import json
import math
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from serving.localscale import metrics


def rfc3339(t):
    return datetime.datetime.fromtimestamp(t, datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class State:
    def __init__(self, model, replicas, desired, window_s=10.0):
        self.model, self.replicas, self.desired, self.window_s = model, dict(replicas), desired, window_s
        self.serving = dict.fromkeys(self.replicas, False)
        self.history = {name: deque(maxlen=600) for name in self.replicas}
        self.lock = threading.Lock()

    def record(self, name, sample, now):
        with self.lock:
            self.history[name].append((now, sample))

    def forget(self, name):
        with self.lock:
            self.history[name].clear()
            self.serving[name] = False

    def snapshot(self, name):
        with self.lock:
            if name not in self.replicas:
                return 404, None
            if not self.serving[name] or not self.history[name]:
                return 503, None
            t, s = self.history[name][-1]
        return 200, {"ObservedAt": rfc3339(t), "Queue": s.waiting, "Prefixes": []}

    def aggregate(self, now=None):
        with self.lock:
            latest = [h[-1] for h in self.history.values() if h]
            waiting = sum(s.waiting for _, s in latest)
            before, after = {}, {}
            for h in self.history.values():
                if not h:
                    continue
                edge = (now if now is not None else h[-1][0]) - self.window_s
                old = next((s for t, s in reversed(h) if t <= edge), h[0][1])
                for bound, count in h[-1][1].ttft_buckets.items():
                    after[bound] = after.get(bound, 0.0) + count
                    before[bound] = before.get(bound, 0.0) + old.ttft_buckets.get(bound, 0.0)
            observed = max((t for t, _ in latest), default=None)
            desired = self.desired
        p95 = metrics.quantile(0.95, before, after) if after else None
        return {
            "observed_at": None if observed is None else rfc3339(observed),
            "queue_per_replica": waiting / desired,
            "p95_ttft_seconds": None if p95 is None or math.isinf(p95) else p95,
            "replicas": desired,
        }


class Poller(threading.Thread):
    """Scrape every replica URL that has a live process, once per interval."""

    def __init__(self, state, live, interval=1.0):
        super().__init__(daemon=True)
        self.state, self.live, self.interval = state, live, interval
        self.halt = threading.Event()

    def run(self):
        while not self.halt.wait(self.interval):
            for name, url in self.state.replicas.items():
                if not self.live(name):
                    continue
                try:
                    with urllib.request.urlopen(url + "/metrics", timeout=2) as resp:
                        text = resp.read(4 << 20).decode()
                except OSError:
                    continue
                self.state.record(name, metrics.parse(text), time.time())


def serve(state, port, host="127.0.0.1"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/snapshot/"):
                code, body = state.snapshot(self.path.removeprefix("/snapshot/"))
            elif self.path == "/aggregate":
                code, body = 200, state.aggregate(time.time())
            else:
                code, body = 404, None
            data = json.dumps(body).encode() if body is not None else b""
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer((host, port), Handler)
