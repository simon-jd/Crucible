"""HTTP proxy in front of vLLM OpenAI servers for the E4 layouts (aiohttp, one process).

  python -m serving.pd.proxy --listen 127.0.0.1:12970 --mode dp --backend http://127.0.0.1:12971 [...]
  python -m serving.pd.proxy --listen 127.0.0.1:12970 --mode pd \\
      --prefill http://127.0.0.1:12971 --prefill-kv 127.0.0.1:12980 \\
      --decode http://127.0.0.1:12972 --decode-kv 127.0.0.1:12985 [--max-pd-inflight 96]

"dp" forwards each POST /v1/completions to the backend with the fewest outstanding requests (ties
round-robin) and streams the response back unchanged. With one backend it is a pass-through, so
the colocated layout sees the same client path as the others.

"pd" follows vLLM 0.10.2's P2pNcclConnector protocol (its toy proxy
examples/online_serving/disaggregated_serving_p2p_nccl_xpyd): the request id
``___prefill_addr_<P zmq>___decode_addr_<D zmq>_<uuid>`` goes to both servers in X-Request-Id;
the prefill server gets a copy of the request with max_tokens=1, no streaming, and pushes the KV
of every layer to the decode server's ZMQ address while it runs; when the prefill response is
back, the original request goes to the decode server, whose response streams to the client. The
decode server keeps every received KV tensor until the request finishes, so --max-pd-inflight
bounds how many requests sit between prefill start and decode finish; the rest wait here (the
wait is part of their TTFT).

Every request is logged as one JSON line (times are time.time() seconds) to --log.
GET /stats returns the in-flight count, used to wait for an idle system between trials.
"""

import argparse
import asyncio
import itertools
import json
import time
import uuid


class Balancer:
    """Least outstanding requests; ties go to the next backend after the last pick."""

    def __init__(self, n):
        if n < 1:
            raise ValueError("need at least one backend")
        self.outstanding = [0] * n
        self._next = 0

    def acquire(self):
        n = len(self.outstanding)
        best = None
        for k in range(n):
            i = (self._next + k) % n
            if best is None or self.outstanding[i] < self.outstanding[best]:
                best = i
        self._next = (best + 1) % n
        self.outstanding[best] += 1
        return best

    def release(self, i):
        if self.outstanding[i] <= 0:
            raise ValueError("release without acquire")
        self.outstanding[i] -= 1


def pd_request_id(prefill_kv, decode_kv, suffix=None):
    """X-Request-Id for P2pNcclConnector; vLLM turns it into cmpl-<id>-0 on both servers."""
    for addr in (prefill_kv, decode_kv):
        host, _, port = addr.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"expected host:port, got {addr!r}")
    return f"___prefill_addr_{prefill_kv}___decode_addr_{decode_kv}_{suffix or uuid.uuid4().hex}"


def engine_request_id(header_id):
    """The id vLLM's completions endpoint uses for the first (only) prompt of a request."""
    return f"cmpl-{header_id}-0"


def prefill_body(body):
    """The prefill server's copy of a completion request: one token, no stream."""
    out = dict(body)
    out["max_tokens"] = 1
    out.pop("min_tokens", None)
    out["stream"] = False
    out.pop("stream_options", None)
    return out


def prompt_tokens(body):
    p = body.get("prompt")
    return len(p) if isinstance(p, list) and all(isinstance(x, int) for x in p) else None


class Proxy:
    def __init__(self, mode, backends=(), prefill=None, decode=None, prefill_kv=None, decode_kv=None,
                 max_pd_inflight=96, log_path=None):
        if mode == "dp":
            if not backends:
                raise ValueError("dp mode needs --backend")
            self.balancer = Balancer(len(backends))
        elif mode == "pd":
            if not (prefill and decode and prefill_kv and decode_kv):
                raise ValueError("pd mode needs --prefill, --decode, --prefill-kv, --decode-kv")
            pd_request_id(prefill_kv, decode_kv, "check")
        else:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode, self.backends = mode, list(backends)
        self.prefill, self.decode, self.prefill_kv, self.decode_kv = prefill, decode, prefill_kv, decode_kv
        self.max_pd_inflight = max_pd_inflight
        self.inflight = 0
        self.counts = {"requests": 0, "errors": 0}
        self._seq = itertools.count()
        self._log = open(log_path, "a") if log_path else None  # noqa: SIM115 - closed in close()
        self.session = None
        self.sem = None

    async def start(self):
        import aiohttp

        self.sem = asyncio.Semaphore(self.max_pd_inflight)
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, force_close=False),
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
            auto_decompress=False,
        )

    async def close(self):
        if self.session:
            await self.session.close()
        if self._log:
            self._log.close()

    def _write(self, rec):
        if self._log:
            self._log.write(json.dumps(rec, separators=(",", ":")) + "\n")
            self._log.flush()

    async def _stream(self, request, url, data, headers, rec):
        from aiohttp import web

        async with self.session.post(url + "/v1/completions", data=data, headers=headers) as up:
            rec["status"] = up.status
            resp = web.StreamResponse(status=up.status)
            resp.content_type = up.headers.get("Content-Type", "application/json").split(";")[0]
            await resp.prepare(request)
            async for chunk in up.content.iter_any():
                if "t_first_byte" not in rec:
                    rec["t_first_byte"] = time.time()
                await resp.write(chunk)
            await resp.write_eof()
            return resp

    async def handle(self, request):
        from aiohttp import web

        raw = await request.read()
        rec = {"seq": next(self._seq), "mode": self.mode, "t_arrive": time.time()}
        self.inflight += 1
        self.counts["requests"] += 1
        headers = {"Content-Type": "application/json"}
        try:
            if self.mode == "dp":
                i = self.balancer.acquire()
                rec["backend"] = i
                try:
                    return await self._stream(request, self.backends[i], raw, headers, rec)
                finally:
                    self.balancer.release(i)
            body = json.loads(raw)
            rec["prompt_tokens"] = prompt_tokens(body)
            rid = pd_request_id(self.prefill_kv, self.decode_kv)
            rec["id"] = engine_request_id(rid)
            headers["X-Request-Id"] = rid
            async with self.sem:
                rec["t_admit"] = time.time()
                pre = json.dumps(prefill_body(body)).encode()
                async with self.session.post(self.prefill + "/v1/completions", data=pre, headers=headers) as r:
                    payload = await r.read()
                    rec["prefill_status"] = r.status
                rec["t_prefill_done"] = time.time()
                if rec["prefill_status"] != 200:
                    rec["status"] = rec["prefill_status"]
                    return web.Response(status=r.status, body=payload, content_type="application/json")
                return await self._stream(request, self.decode, raw, headers, rec)
        except asyncio.CancelledError:
            rec["error"] = "client cancelled"
            raise
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
            self.counts["errors"] += 1
            if "status" in rec:  # the stream had started; nothing sensible can be sent any more
                raise
            rec["status"] = 502
            return web.Response(status=502, text=rec["error"])
        finally:
            self.inflight -= 1
            rec["t_done"] = time.time()
            self._write(rec)

    async def stats(self, request):
        from aiohttp import web

        return web.json_response({"inflight": self.inflight, **self.counts})


async def serve(proxy, host, port):
    from aiohttp import web

    await proxy.start()
    app = web.Application(client_max_size=16 << 20)
    app.router.add_post("/v1/completions", proxy.handle)
    app.router.add_get("/stats", proxy.stats)
    runner = web.AppRunner(app, handler_cancellation=True, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port, backlog=1024)
    await site.start()
    return runner


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--listen", default="127.0.0.1:12970")
    p.add_argument("--mode", choices=("dp", "pd"), required=True)
    p.add_argument("--backend", action="append", default=[])
    p.add_argument("--prefill")
    p.add_argument("--decode")
    p.add_argument("--prefill-kv")
    p.add_argument("--decode-kv")
    p.add_argument("--max-pd-inflight", type=int, default=96)
    p.add_argument("--log")
    a = p.parse_args(argv)
    proxy = Proxy(a.mode, a.backend, a.prefill, a.decode, a.prefill_kv, a.decode_kv, a.max_pd_inflight, a.log)
    host, _, port = a.listen.rpartition(":")

    async def run():
        runner = await serve(proxy, host, int(port))
        print(f"proxy {a.mode} listening on {a.listen}", flush=True)
        try:
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()
            await proxy.close()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
