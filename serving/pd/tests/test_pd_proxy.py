import asyncio
import json
import re

import pytest

from serving.pd import proxy as px


def test_balancer_picks_least_outstanding_and_round_robins_ties():
    b = px.Balancer(2)
    assert [b.acquire(), b.acquire()] == [0, 1]
    b.release(0)
    assert b.acquire() == 0  # 0 has fewer outstanding
    assert b.acquire() == 1  # tie 1:1, next after the last pick
    with pytest.raises(ValueError):
        px.Balancer(0)
    b2 = px.Balancer(1)
    with pytest.raises(ValueError):
        b2.release(0)


def test_request_id_round_trips_through_the_connector_regexes():
    rid = px.pd_request_id("10.0.0.1:12980", "10.0.0.2:12985", "abc123")
    engine_id = px.engine_request_id(rid)
    assert engine_id == f"cmpl-{rid}-0"
    # The same patterns P2pNcclConnector.parse_request_id uses (vLLM 0.10.2).
    m = re.search(r"___decode_addr_(.*):(\d+)", engine_id)
    assert (m.group(1), m.group(2)) == ("10.0.0.2", "12985")
    m = re.search(r"___prefill_addr_(.*):(\d+)___", engine_id)
    assert (m.group(1), m.group(2)) == ("10.0.0.1", "12980")
    with pytest.raises(ValueError):
        px.pd_request_id("nohost", "10.0.0.2:1")


def test_prefill_body_asks_for_one_unstreamed_token_and_keeps_the_prompt():
    body = {"model": "m", "prompt": [1, 2, 3], "max_tokens": 50, "min_tokens": 50, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}, "return_token_ids": True}
    pre = px.prefill_body(body)
    assert pre["max_tokens"] == 1 and pre["stream"] is False
    assert "min_tokens" not in pre and "stream_options" not in pre
    assert pre["prompt"] == [1, 2, 3] and body["max_tokens"] == 50  # original untouched
    assert px.prompt_tokens(body) == 3 and px.prompt_tokens({"prompt": "text"}) is None


def test_proxy_rejects_incomplete_configs():
    with pytest.raises(ValueError):
        px.Proxy("dp")
    with pytest.raises(ValueError):
        px.Proxy("pd", prefill="http://a")
    with pytest.raises(ValueError):
        px.Proxy("xx")


# ---- end to end over loopback with fake vLLM servers (needs aiohttp, e.g. .venv-ml) ----


def _fake_backend(web, name, calls):
    async def completions(request):
        body = await request.json()
        calls.append((name, request.headers.get("X-Request-Id"), body))
        if body.get("prompt") == [0]:
            return web.json_response({"error": "empty"}, status=400)
        if not body.get("stream"):
            return web.json_response({"choices": [{"text": "x"}], "from": name})
        resp = web.StreamResponse()
        resp.content_type = "text/event-stream"
        await resp.prepare(request)
        for i in range(body["max_tokens"]):
            await resp.write(f'data: {{"from": "{name}", "i": {i}}}\n\n'.encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def metrics(request):
        return web.Response(text="")

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/metrics", metrics)
    return app


async def _start(web, app):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def _run(coro):
    return asyncio.run(coro)


def test_dp_mode_spreads_requests_and_streams_them_back(tmp_path):
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web

    async def scenario():
        calls = []
        r0, u0 = await _start(web, _fake_backend(web, "a", calls))
        r1, u1 = await _start(web, _fake_backend(web, "b", calls))
        p = px.Proxy("dp", backends=[u0, u1], log_path=tmp_path / "proxy.jsonl")
        runner = await px.serve(p, "127.0.0.1", 0)
        port = next(iter(runner.sites))._server.sockets[0].getsockname()[1]
        body = {"model": "m", "prompt": [5, 6], "max_tokens": 3, "stream": True}
        async with aiohttp.ClientSession() as s:
            texts = []
            for _ in range(2):
                async with s.post(f"http://127.0.0.1:{port}/v1/completions", json=body) as r:
                    assert r.status == 200
                    texts.append(await r.text())
            async with s.get(f"http://127.0.0.1:{port}/stats") as r:
                stats = await r.json()
        await runner.cleanup()
        await p.close()
        for r in (r0, r1):
            await r.cleanup()
        return calls, texts, stats

    calls, texts, stats = _run(scenario())
    assert sorted(c[0] for c in calls) == ["a", "b"]
    assert all(t.count("data: ") == 4 and t.endswith("data: [DONE]\n\n") for t in texts)
    assert stats["inflight"] == 0 and stats["requests"] == 2
    recs = [json.loads(x) for x in (tmp_path / "proxy.jsonl").read_text().splitlines()]
    assert [r["status"] for r in recs] == [200, 200] and all(r["t_first_byte"] >= r["t_arrive"] for r in recs)


def test_pd_mode_prefills_one_token_then_streams_from_decode_with_one_request_id(tmp_path):
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web

    async def scenario():
        calls = []
        rp, up = await _start(web, _fake_backend(web, "prefill", calls))
        rd, ud = await _start(web, _fake_backend(web, "decode", calls))
        p = px.Proxy("pd", prefill=up, decode=ud, prefill_kv="127.0.0.1:12980", decode_kv="127.0.0.1:12985",
                     max_pd_inflight=4, log_path=tmp_path / "proxy.jsonl")
        runner = await px.serve(p, "127.0.0.1", 0)
        port = next(iter(runner.sites))._server.sockets[0].getsockname()[1]
        body = {"model": "m", "prompt": [5, 6, 7], "max_tokens": 4, "min_tokens": 4, "stream": True,
                "stream_options": {"include_usage": True}}
        async with aiohttp.ClientSession() as s:
            async with s.post(f"http://127.0.0.1:{port}/v1/completions", json=body) as r:
                status, text = r.status, await r.text()
            async with s.post(f"http://127.0.0.1:{port}/v1/completions", json={**body, "prompt": [0]}) as r:
                bad = r.status
        await runner.cleanup()
        await p.close()
        for r in (rp, rd):
            await r.cleanup()
        return calls, status, text, bad

    calls, status, text, bad = _run(scenario())
    assert status == 200 and text.count('"from": "decode"') == 4
    (n1, id1, b1), (n2, id2, b2), (n3, _, _) = calls
    assert (n1, n2, n3) == ("prefill", "decode", "prefill")  # the failed prefill never reaches decode
    assert id1 == id2 and "___decode_addr_127.0.0.1:12985_" in id1
    assert b1["max_tokens"] == 1 and b1["stream"] is False and "min_tokens" not in b1
    assert b2["max_tokens"] == 4 and b2["stream"] is True
    assert bad == 400
    recs = [json.loads(x) for x in (tmp_path / "proxy.jsonl").read_text().splitlines()]
    ok = recs[0]
    assert ok["id"] == f"cmpl-{id1}-0" and ok["prompt_tokens"] == 3
    assert ok["t_arrive"] <= ok["t_admit"] <= ok["t_prefill_done"] <= ok["t_first_byte"] <= ok["t_done"]
    assert recs[1]["status"] == 400 and "t_first_byte" not in recs[1]


def test_unreachable_backend_is_a_502_and_is_logged(tmp_path):
    aiohttp = pytest.importorskip("aiohttp")

    async def scenario():
        p = px.Proxy("dp", backends=["http://127.0.0.1:9"], log_path=tmp_path / "proxy.jsonl")
        runner = await px.serve(p, "127.0.0.1", 0)
        port = next(iter(runner.sites))._server.sockets[0].getsockname()[1]
        async with (aiohttp.ClientSession() as s,
                    s.post(f"http://127.0.0.1:{port}/v1/completions", json={"prompt": [1]}) as r):
            status = r.status
        await runner.cleanup()
        await p.close()
        return status, p.counts

    status, counts = _run(scenario())
    assert status == 502 and counts["errors"] == 1
    rec = json.loads((tmp_path / "proxy.jsonl").read_text())
    assert rec["status"] == 502 and "error" in rec
