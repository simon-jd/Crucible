"""CPU stand-in for one vLLM replica, for the kind (no-GPU) Task 14 deployment.

GPUs do not pass into kind nodes, so pods run this instead of vLLM. It speaks the
subset the project's clients use: streaming /v1/completions with return_token_ids
and include_usage (what the Go replayer's `-api openai-text` parses), /health, and
the vLLM 0.10.2 Prometheus names that serving/localscale/metrics.py reads.

Latency model (synthetic, fixed): `--max-num-seqs` slots; a request waits for a
slot (counted as waiting), then prefill takes `--prefill-ms-per-token` per prompt
word, then each output token takes `--tpot-ms`. There is no batching slowdown and
no model. It measures the Kubernetes control path, not inference.

SIGTERM drains: /health and /metrics answer 503 (the adapter then reports the
replica as not serving), requests keep being served, and the process exits after
in-flight work is done and no request arrived for `--drain-idle-s`.
Stdlib only; runs on the project's standalone CPython (see manifests.py).
"""

import argparse
import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TTFT_BOUNDS = (0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0)


class Engine:
    def __init__(self, max_num_seqs, tpot_ms, prefill_ms_per_token, model):
        self.slots = threading.Semaphore(max_num_seqs)
        self.tpot, self.prefill, self.model = tpot_ms / 1000, prefill_ms_per_token / 1000, model
        self.lock = threading.Lock()
        self.waiting = self.running = self.success = 0
        self.ttft_counts = [0] * (len(TTFT_BOUNDS) + 1)
        self.draining = False
        self.last_arrival = time.monotonic()
        self.first_request_logged = False

    def observe_ttft(self, seconds):
        with self.lock:
            for i, bound in enumerate(TTFT_BOUNDS):
                if seconds <= bound:
                    self.ttft_counts[i] += 1
                    return
            self.ttft_counts[-1] += 1

    def metrics(self):
        labels = f'engine="0",model_name="{self.model}"'
        with self.lock:
            lines = [
                f"vllm:num_requests_waiting{{{labels}}} {self.waiting}",
                f"vllm:num_requests_running{{{labels}}} {self.running}",
                f'vllm:request_success_total{{finished_reason="length",{labels}}} {self.success}',
            ]
            cumulative = 0
            for bound, count in zip((*TTFT_BOUNDS, "+Inf"), self.ttft_counts, strict=True):
                cumulative += count
                lines.append(f'vllm:time_to_first_token_seconds_bucket{{{labels},le="{bound}"}} {cumulative}')
        return "\n".join(lines) + "\n"

    def generate(self, prompt_tokens, max_tokens, emit):
        """Run one request; emit(token_index, is_last) streams each token."""
        start = time.monotonic()
        with self.lock:
            self.waiting += 1
            self.last_arrival = start
            first = not self.first_request_logged
            self.first_request_logged = True
        if first:
            log("first_request")
        self.slots.acquire()
        try:
            with self.lock:
                self.waiting -= 1
                self.running += 1
            time.sleep(prompt_tokens * self.prefill)
            for i in range(max_tokens):
                if i:
                    time.sleep(self.tpot)
                emit(i, i == max_tokens - 1)
                if i == 0:
                    self.observe_ttft(time.monotonic() - start)
            with self.lock:
                self.success += 1
        finally:
            with self.lock:
                self.running -= 1
            self.slots.release()


LOG_FILES = []


def log(event, **fields):
    line = json.dumps({"t": time.time(), "event": event, **fields})
    print(line, flush=True)
    for f in LOG_FILES:
        f.write(line + "\n")
        f.flush()


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _plain(self, code, body, ctype="text/plain"):
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path not in ("/health", "/metrics"):
                return self._plain(404, "not found")
            if engine.draining:
                return self._plain(503, "draining")
            self._plain(200, engine.metrics() if self.path == "/metrics" else "ok")

        def do_POST(self):
            if self.path != "/v1/completions":
                return self._plain(404, "not found")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                n, prompt = int(req["max_tokens"]), req["prompt"]
                if n < 1 or not isinstance(prompt, str) or not req.get("stream"):
                    raise ValueError
            except (ValueError, KeyError, TypeError):
                return self._plain(400, "streaming string-prompt completion with max_tokens >= 1 required")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def send(obj):
                data = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n"
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()

            def emit(i, last):
                send({"choices": [{"index": 0, "text": " x", "token_ids": [1000 + i % 1000], "finish_reason": "length" if last else None}]})

            try:
                engine.generate(len(prompt.split()), n, emit)
                send({"choices": [], "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": n}})
                send(b"[DONE]")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--tpot-ms", type=float, default=5.0)
    p.add_argument("--prefill-ms-per-token", type=float, default=0.05)
    p.add_argument("--startup-s", type=float, default=0.0, help="sleep before listening (stand-in for model load)")
    p.add_argument("--drain-idle-s", type=float, default=3.0)
    p.add_argument("--log-file", help="also append JSON event lines here (hostPath in kind)")
    args = p.parse_args(argv)
    if args.log_file:
        LOG_FILES.append(open(args.log_file, "a"))  # noqa: SIM115 - process lifetime
    log("process_start")
    time.sleep(args.startup_s)
    engine = Engine(args.max_num_seqs, args.tpot_ms, args.prefill_ms_per_token, args.model)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    server.daemon_threads = True
    done = threading.Event()

    def drain(*_):
        engine.draining = True
        log("drain_start")

        def wait():
            while True:
                with engine.lock:
                    busy = engine.waiting + engine.running
                    idle = time.monotonic() - engine.last_arrival
                if busy == 0 and idle >= args.drain_idle_s:
                    break
                time.sleep(0.2)
            log("drained", success=engine.success)
            server.shutdown()
            done.set()

        threading.Thread(target=wait, daemon=True).start()

    signal.signal(signal.SIGTERM, drain)
    log("listening", port=args.port)
    server.serve_forever()
    done.wait(5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
