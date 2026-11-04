"""A stand-in for vLLM's OpenAI server for CPU dry runs of the E4 payload (aiohttp).

Accepts the vLLM CLI arguments the payload passes; serves /health, /metrics (running/waiting
gauges) and /v1/completions with vLLM's streamed shape (token_ids per chunk, finish_reason, a
usage chunk, [DONE]) or a one-shot JSON body when stream is false.
"""

import argparse
import asyncio
import json

from aiohttp import web


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    a, _ = p.parse_known_args()
    running = {"n": 0}

    async def health(request):
        return web.Response(text="")

    async def metrics(request):
        return web.Response(text=f'vllm:num_requests_running{{model_name="m"}} {running["n"]}.0\n'
                                 'vllm:num_requests_waiting{model_name="m"} 0.0\n')

    async def completions(request):
        body = await request.json()
        n = body["max_tokens"]
        if not body.get("stream"):
            return web.json_response({"choices": [{"text": "x", "token_ids": [7], "finish_reason": "length"}]})
        running["n"] += 1
        try:
            resp = web.StreamResponse()
            resp.content_type = "text/event-stream"
            await resp.prepare(request)
            for i in range(n):
                await asyncio.sleep(0.002)
                choice = {"index": 0, "text": "x", "token_ids": [7], "finish_reason": "length" if i == n - 1 else None}
                await resp.write(f"data: {json.dumps({'choices': [choice]})}\n\n".encode())
            await resp.write(f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': n}})}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        finally:
            running["n"] -= 1

    app = web.Application(client_max_size=16 << 20)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/v1/completions", completions)
    web.run_app(app, host=a.host, port=a.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
