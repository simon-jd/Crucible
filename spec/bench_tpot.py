"""Closed-loop batch-N TPOT and acceptance client for one SGLang server (T15).

For each batch size N, the same held-out prompts are sent with N requests in flight
(a new request starts when one finishes), greedy, streaming from /generate.
TPOT per request = (last chunk - first chunk) / (completion_tokens - 1); TTFT is
first chunk - send. Acceptance length = 1 + sum(spec_num_correct_drafts) /
sum(spec_verify_ct) over completed requests (SGLang 0.5.18 per-request counters;
the bonus token is the 1). Machine load (nproc, uptime) is recorded around each batch.
"""

import argparse
import http.client
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from spec.eval_acceptance import acceptance_length

SYSTEM_PROMPT = "You are a helpful assistant."  # SpecForge "qwen" template, used in training


def parse_sse_line(line):
    line = line.strip()
    if not line.startswith(b"data:"):
        return None
    body = line[5:].strip()
    if body == b"[DONE]":
        return "done"
    return json.loads(body)


def request_record(prompt_id, *, t_send, t_first, t_end, meta):
    tokens = meta["completion_tokens"]
    record = {
        "prompt_id": prompt_id,
        "status": "completed",
        "completion_tokens": tokens,
        "ttft_ms": (t_first - t_send) * 1000,
        "e2e_ms": (t_end - t_send) * 1000,
        "tpot_ms": (t_end - t_first) * 1000 / (tokens - 1) if tokens > 1 else None,
    }
    for key in ("spec_verify_ct", "spec_num_correct_drafts", "finish_reason"):
        if key in meta:
            record[key] = meta[key]
    return record


def _quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def summarize(records, wall_s):
    done = [r for r in records if r["status"] == "completed"]
    tpots = [r["tpot_ms"] for r in done if r["tpot_ms"] is not None]
    tokens = sum(r["completion_tokens"] for r in done)
    spec = [r for r in done if "spec_verify_ct" in r]
    accept = None
    if spec:
        accept = acceptance_length(
            [
                {
                    "request_id": r["prompt_id"],
                    "spec_accepted_tokens": r["spec_num_correct_drafts"],
                    "spec_verify_ct": r["spec_verify_ct"],
                }
                for r in spec
            ]
        )
    return {
        "requests": len(records),
        "completed": len(done),
        "failed": len(records) - len(done),
        "tpot_ms_median": statistics.median(tpots) if tpots else None,
        "tpot_ms_p90": _quantile(tpots, 0.9) if tpots else None,
        "ttft_ms_median": statistics.median(r["ttft_ms"] for r in done) if done else None,
        "output_tokens": tokens,
        "wall_s": wall_s,
        "output_tok_per_s": tokens / wall_s if wall_s > 0 else None,
        "acceptance_length": accept,
    }


def compare(base, spec):
    return {
        "tpot_speedup": base["tpot_ms_median"] / spec["tpot_ms_median"],
        "throughput_ratio": spec["output_tok_per_s"] / base["output_tok_per_s"],
    }


def aggregate(runs, mode="eagle3"):
    """Per batch size, pair round i of base with round i of `mode` (labels mode-rN).
    Batch sizes missing either side are skipped (the spec-config grid runs only 1 and 64)."""
    by = {}
    for run in sorted(runs, key=lambda r: r["label"]):
        label, rnd = run["label"].rsplit("-r", 1)
        for batch in run["batches"]:
            by.setdefault(batch["batch_size"], {}).setdefault(label, {})[int(rnd)] = batch
    rows = []
    for size in sorted(by):
        base, spec = by[size].get("base", {}), by[size].get(mode, {})
        rounds = sorted(set(base) & set(spec))
        if not rounds:
            continue
        b = [base[r]["summary"] for r in rounds]
        s = [spec[r]["summary"] for r in rounds]
        rows.append(
            {
                "batch_size": size,
                "rounds": rounds,
                "base_tpot_ms": [x["tpot_ms_median"] for x in b],
                f"{mode}_tpot_ms": [x["tpot_ms_median"] for x in s],
                "tpot_speedup": [compare(x, y)["tpot_speedup"] for x, y in zip(b, s)],
                "base_tok_per_s": [x["output_tok_per_s"] for x in b],
                f"{mode}_tok_per_s": [x["output_tok_per_s"] for x in s],
                "throughput_ratio": [compare(x, y)["throughput_ratio"] for x, y in zip(b, s)],
                "acceptance_length": [x["acceptance_length"] for x in s],
                "failed": [x["failed"] for x in b + s],
                "loadavg_1m": [m[r]["load_before"]["loadavg"][0] for m in (base, spec) for r in rounds],
            }
        )
    return rows


def machine_load():
    uptime = subprocess.run(["uptime"], capture_output=True, text=True, check=False).stdout.strip()
    return {"nproc": os.cpu_count(), "loadavg": os.getloadavg(), "uptime": uptime}


def stream_generate(url, prompt_id, text, max_tokens, timeout=600):
    parsed = urlparse(url)
    body = json.dumps(
        {
            "text": text,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": max_tokens},
            "stream": True,
        }
    )
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    t_send = time.perf_counter()
    try:
        conn.request("POST", "/generate", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            return {"prompt_id": prompt_id, "status": "failed", "error": f"HTTP {resp.status}"}
        t_first, t_end, meta = None, None, None
        for line in resp:
            event = parse_sse_line(line)
            if event is None:
                continue
            if event == "done":
                break
            now = time.perf_counter()
            meta = event["meta_info"]
            if t_first is None and meta.get("completion_tokens", 0) >= 1:
                t_first = now
            t_end = now
        if meta is None or t_first is None:
            return {"prompt_id": prompt_id, "status": "failed", "error": "no tokens streamed"}
        return request_record(prompt_id, t_send=t_send, t_first=t_first, t_end=t_end, meta=meta)
    except (OSError, ValueError, KeyError) as exc:
        return {"prompt_id": prompt_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        conn.close()


def run_batch(url, prompts, concurrency, max_tokens):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        records = list(
            pool.map(lambda p: stream_generate(url, p["prompt_id"], p["text"], max_tokens), prompts)
        )
    return records, time.perf_counter() - start


def render_prompts(tokenizer_path, rows):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    out = []
    for row in rows:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["prompt"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        out.append({"prompt_id": row["prompt_id"], "text": text})
    return out


def main(argv=None):
    if argv is None and sys.argv[1:2] == ["aggregate"]:
        # python -m spec.bench_tpot aggregate DIR [MODE]: write DIR/summary.json (MODE eagle3)
        # or DIR/summary-MODE.json from DIR/*-r*.json
        out_dir = Path(sys.argv[2])
        mode = sys.argv[3] if len(sys.argv) > 3 else "eagle3"
        runs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*-r*.json"))]
        rows = aggregate(runs, mode)
        name = "summary.json" if mode == "eagle3" else f"summary-{mode}.json"
        (out_dir / name).write_text(json.dumps(rows, indent=1) + "\n")
        print(json.dumps(rows))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--batches", default="1,4,16,64")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    prompts = render_prompts(args.tokenizer, rows)
    run_batch(args.url, prompts[: args.warmup], min(args.warmup, 8), args.max_tokens)
    result = {"label": args.label, "max_tokens": args.max_tokens, "prompts": len(prompts), "batches": []}
    for size in [int(b) for b in args.batches.split(",")]:
        before = machine_load()
        records, wall = run_batch(args.url, prompts, size, args.max_tokens)
        result["batches"].append(
            {
                "batch_size": size,
                "load_before": before,
                "load_after": machine_load(),
                "summary": summarize(records, wall),
                "records": records,
            }
        )
        s = result["batches"][-1]["summary"]
        print(json.dumps({"label": args.label, "batch_size": size, **s}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
