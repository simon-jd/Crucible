"""Task 13 single-GPU BF16 vs FP8 session: raw outputs only (one GPU-lock hold).

Same model snapshot (Qwen3-8B BF16); the candidate adds vLLM online FP8
(`--quantization fp8`, dynamic per-tensor weight/activation scales), or, with
--candidate-snapshot, serves a pre-quantized FP8 checkpoint (vLLM reads its
quantization_config; no extra flag). Everything else is identical. Sequence, with a server restart at each precision change:

  bf16: 3 calibration replays (fixed BF16 SLO), run 1-baseline, quality
  fp8:  1-candidate, 2-candidate, quality
  bf16: 2-baseline, 3-baseline
  fp8:  3-candidate

That is the AB, BA, AB run plan required by serving/eval/accuracy.py. The
replay load is the Task 12 Azure window at 0.25x source rate. Quality is
greedy, non-thinking chat on GSM8K and MMLU-Pro held-out subsets.
serving/eval/fp8_bundle.py turns the outputs into accuracy.py bundles.
"""

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from serving.baseline import vllm_baseline as vb

PLAN = [("1-baseline", 1, "baseline"), ("1-candidate", 1, "candidate"), ("2-candidate", 2, "candidate"),
        ("2-baseline", 2, "baseline"), ("3-baseline", 3, "baseline"), ("3-candidate", 3, "candidate")]  # fmt: skip
PRECISION = {"baseline": "bf16", "candidate": "fp8"}
# Attempt 1 used "...of the form '#### <number>'..."; some replies echoed '<number>' literally.
GSM8K_SUFFIX = "\nSolve step by step. End with a final line that is '####' followed by only the number, for example: #### 42"
LETTERS = "ABCDEFGHIJ"


def mmlu_prompt(row):
    options = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(row["options"]))
    return f"{row['question']}\n\n{options}\n\nAnswer with only the letter of the correct option."


def quality_requests(heldout):
    """[(set, id, label, prompt, max_tokens, guided_choice)] in declared order.

    MMLU-Pro output is constrained to the row's option letters (vLLM guided_choice):
    attempt 1, unconstrained, mostly answered "C. <option text>" and was truncated.
    """
    out = [("gsm8k", r["id"], r["answer"], r["question"] + GSM8K_SUFFIX, 768, None) for r in heldout["gsm8k"]]
    out += [("mmlu_pro", r["id"], r["answer"], mmlu_prompt(r), 8, list(LETTERS[: len(r["options"])])) for r in heldout["mmlu_pro"]]
    return out


def chat_body(model, prompt, max_tokens, choices):
    body = {
        "model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
        "temperature": 0, "seed": 17, "chat_template_kwargs": {"enable_thinking": False},
    }  # fmt: skip
    if choices:
        body["guided_choice"] = choices
    return body


def chat(port, model, prompt, max_tokens, choices=None):
    body = json.dumps(chat_body(model, prompt, max_tokens, choices)).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        choice = data["choices"][0]
        return {"status": "success" if choice["finish_reason"] == "stop" else "truncated", "answer": choice["message"]["content"],
                "finish_reason": choice["finish_reason"], "error": None}  # fmt: skip
    except Exception as exc:  # noqa: BLE001 - recorded per item
        return {"status": "error", "answer": None, "finish_reason": None, "error": f"{type(exc).__name__}: {exc}"}


def server_cmd(args, precision):
    checkpoint = precision == "fp8" and args.candidate_snapshot
    snapshot = args.candidate_snapshot if checkpoint else args.snapshot
    cmd = vb.server_command(snapshot, args.model_name, args.port, args.max_model_len, args.max_num_seqs)
    # Eager for both precisions: no torch.compile cache and more KV room for
    # BF16 weights on a 24 GB card.
    cmd += ["--enforce-eager"]
    if precision == "fp8" and not checkpoint:
        cmd += ["--quantization", "fp8"]
    return cmd


class Server:
    def __init__(self, args, precision, out, tag):
        cmd = server_cmd(args, precision)
        self.cmd = cmd[1:]
        self.log = open(out / f"server-{tag}.log", "wb")  # noqa: SIM115 - closed in stop()
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.monotonic()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{tag}: server exited {self.proc.returncode} before ready")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() - t0 > 900:
                raise RuntimeError(f"{tag}: not ready in 900 s")
            time.sleep(2)
        self.startup_s = time.monotonic() - t0

    def stop(self):
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()
        except ProcessLookupError:
            pass
        self.log.close()
        return self.proc.returncode


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--token-ids", required=True)
    p.add_argument("--replay-bin", required=True)
    p.add_argument("--heldout", type=Path, required=True)
    p.add_argument("--trace-dir", type=Path, default=Path("serving/baseline/traces/azure-60s"))
    p.add_argument("--mult", type=float, default=0.25)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--port", type=int, default=18081)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--quality-only", action="store_true", help="one bf16 and one fp8 server, quality sets only")
    p.add_argument("--candidate-snapshot", help="serve this pre-quantized FP8 checkpoint as the candidate instead of online FP8")
    args = p.parse_args(argv)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    replay_args = SimpleNamespace(replay_bin=args.replay_bin, token_ids=args.token_ids, model_name=args.model_name)
    heldout = json.loads(args.heldout.read_text())
    session = {"plan": PLAN, "candidate_snapshot": args.candidate_snapshot, "mult": args.mult, "trace_dir": str(args.trace_dir), "servers": [], "runs": [], "stop_reason": None}
    monitor = vb.Monitor(args.cache_root, 150 * 10**6)
    monitor.start()
    warm = args.cache_root / "warmup.csv"
    warm.write_text("Timestamp,Request tokens,Response tokens\n" + "".join(f"{i},{a},{b}\n" for i, (a, b) in enumerate(vb.WARMUP)))
    blocks = [("bf16", ["cal", "1-baseline", "quality"]), ("fp8", ["1-candidate", "2-candidate", "quality"]),
              ("bf16", ["2-baseline", "3-baseline"]), ("fp8", ["3-candidate"])]  # fmt: skip
    if args.quality_only:
        blocks = [("bf16", ["quality"]), ("fp8", ["quality"])]
    server = None
    try:
        for index, (precision, steps) in enumerate(blocks):
            server = Server(args, precision, out, f"{index}-{precision}")
            info = {"block": index, "precision": precision, "command": server.cmd, "startup_s": server.startup_s}
            code, _ = vb.run_replay(replay_args, warm, "burstgpt", 1.0, out / f"warmup-{index}.json", args.port)
            info["warmup_exit"] = code
            vb.wait_idle(args.port)
            for step in steps:
                if monitor.stop_reason:
                    raise RuntimeError(monitor.stop_reason)
                if step == "cal":
                    for r in range(3):
                        vb.run_replay(replay_args, args.trace_dir / "calibration.csv", "azure", args.mult, out / f"calibration-{r}.json", args.port)
                        vb.wait_idle(args.port)
                elif step == "quality":
                    requests = quality_requests(heldout)
                    t0 = time.monotonic()
                    with ThreadPoolExecutor(16) as pool:
                        results = list(pool.map(lambda q: chat(args.port, args.model_name, q[3], q[4], q[5]), requests))
                    with open(out / f"quality-{precision}.jsonl", "w") as f:
                        f.writelines(json.dumps({"set": q[0], "id": q[1], "label": q[2], "prompt": q[3], **r}) + "\n" for q, r in zip(requests, results, strict=True))
                    info["quality_s"] = time.monotonic() - t0
                    vb.wait_idle(args.port)
                else:
                    started = time.monotonic_ns()
                    code, _ = vb.run_replay(replay_args, args.trace_dir / "evaluation.csv", "azure", args.mult, out / f"run-{step}.json", args.port)
                    finished = time.monotonic_ns()
                    session["runs"].append({"id": step, "precision": precision, "exit": code, "started_at_ns": started, "finished_at_ns": finished})
                    print(json.dumps({"run": step, "exit": code, "s": (finished - started) / 1e9}), flush=True)
                    vb.wait_idle(args.port)
            info["exit"] = server.stop()
            server = None
            session["servers"].append(info)
    except Exception as exc:  # noqa: BLE001 - terminal cause is recorded
        session["stop_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if server is not None:
            server.stop()
        monitor.halt.set()
        monitor.join()
        session["peak_owned_rss_bytes"] = monitor.peak_rss
        session["peak_runtime_cache_bytes"] = monitor.peak_cache
        (out / "session.json").write_text(json.dumps(session, indent=1) + "\n")
    return 0 if session["stop_reason"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
