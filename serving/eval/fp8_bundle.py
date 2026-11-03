"""Turn a fp8_session output directory into serving/eval/accuracy.py schema-2 bundles.

One bundle per held-out set (GSM8K numeric-final-v1, MMLU-Pro choice-letter-v1);
both share the same six performance runs. Prompt hashes are of the actual
requests: the replayer's seeded synthetic token IDs (re-derived here with the
same LCG, pinned by tests on both sides) and the exact chat prompt text.
Writes bundle-*.json and report-*.json (the accuracy.py output) next to --output.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from serving.baseline import vllm_baseline as vb
from serving.eval import accuracy
from serving.eval.fp8_session import PLAN, PRECISION

NORMALIZATION = {"gsm8k": "numeric-final-v1", "mmlu_pro": "choice-letter-v1"}
MAX_NEW = {"gsm8k": 768, "mmlu_pro": 8}
HELDOUT = {
    "gsm8k": ("openai/gsm8k", "main/test held-out 200 (rl.local.prepare_data seed 17)", "740312add88f781978c0658806c59bc2815b9866"),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", "test seeded subset 300 (seed 17)", "b189ec765aa7ed75c8acfea42df31fdae71f97be"),
}


def prompt_ids(lengths, token_ids, seed):
    state, out = seed, []
    for n in lengths:
        row = []
        for _ in range(n):
            state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
            row.append(token_ids[(state >> 32) % len(token_ids)])
        out.append(row)
    return out


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _request(r, requested):
    n = r["OutputTokens"]
    ok = r["Status"] == "ok"
    return {
        "id": None,
        "status": "success" if ok else "error",
        "send_s": r["SentMs"] / 1000 if r["SentMs"] >= 0 else None,
        "end_s": r["CompletedMs"] / 1000,
        "ttft_ms": r["TTFTms"] if n >= 1 and r["TTFTms"] >= 0 else None,
        "tpot_ms": r["TPOTms"] if n >= 2 and r["TPOTms"] >= 0 else None,
        "output_tokens": min(n, requested),
        "kv_transfer_ms": None,
        "error": None if ok else f"{r['Status']}: {r['Error']}",
    }


def build_bundles(raw, trace, token_ids, *, model_sha256, runtime_sha256, source_sha256, model_name, revision,
                  quality_dir=None, candidate=None, tokenizer_sha256=None):
    """quality_dir: a --quality-only session; its servers must match the perf session's.

    candidate: {name, revision, sha256, tokenizer_sha256} of the pre-quantized FP8
    checkpoint the session served (fp8_session --candidate-snapshot); None for online FP8.
    """
    raw = Path(raw)
    quality_dir = Path(quality_dir) if quality_dir else raw
    session = json.loads((raw / "session.json").read_text())
    served = session.get("candidate_snapshot")
    if (candidate is None) != (served is None) or (candidate and Path(served).name != candidate["revision"]):
        raise ValueError(f"declared candidate checkpoint does not match the session's candidate_snapshot {served!r}")
    calibrations = [json.loads((raw / f"calibration-{i}.json").read_text()) for i in range(3)]
    slo = vb.calibrate(calibrations)
    first = json.loads((raw / f"run-{PLAN[0][0]}.json").read_text())
    seed = first["Config"]["Seed"]
    ids = prompt_ids([t[0] for t in trace], token_ids, seed)
    workload = [
        {"id": f"r{i:04d}", "prompt_sha256": _sha(json.dumps(p)), "scheduled_s": rec["ScheduledMs"] / 1000,
         "input_tokens": t[0], "requested_tokens": max(t[1], 1)}
        for i, (p, t, rec) in enumerate(zip(ids, trace, first["Run"]["Records"], strict=True))
    ]  # fmt: skip
    # Block 0 is the BF16 server; FP8 servers add exactly `--quantization fp8` (see backend).
    if session["servers"][0]["precision"] != "bf16":
        raise ValueError("first server block must be BF16")
    settings = {
        "layout": "colocated",
        "server_args_common": session["servers"][0]["command"],
        "warmup_requests": [list(w) for w in vb.WARMUP],
        "prefix_cache": False,
        "replay": {"trace_sha256": first["Trace"]["SHA256"], "mult": first["Config"]["Multiplier"], "timeout_s": vb.REQUEST_TIMEOUT_S, "max_inflight": vb.MAX_INFLIGHT},
    }
    configs = {
        "baseline": {"precision": "bf16", "backend": "vllm 0.10.2, BF16 weights", "settings": settings, "model_sha256": model_sha256, "runtime_sha256": runtime_sha256},
        "candidate": {"precision": "fp8", "backend": "vllm 0.10.2, online FP8 (--quantization fp8, dynamic scales)", "settings": settings,
                      "model_sha256": model_sha256, "runtime_sha256": runtime_sha256},
    }  # fmt: skip
    if candidate:
        configs["candidate"]["backend"] = f"vllm 0.10.2, pre-quantized FP8 checkpoint {candidate['name']}@{candidate['revision']}"
        configs["candidate"]["model_sha256"] = candidate["sha256"]
    calibration_sha = accuracy.digest([hashlib.sha256((raw / f"calibration-{i}.json").read_bytes()).hexdigest() for i in range(3)])
    timing = {r["id"]: r for r in session["runs"]}
    if quality_dir != raw:
        quality_servers = json.loads((quality_dir / "session.json").read_text())["servers"]
        perf = {s["precision"]: s["command"] for s in session["servers"]}
        if any(s["command"] != perf[s["precision"]] for s in quality_servers):
            raise ValueError("quality session server arguments differ from the performance session")
    quality_rows = {p: [json.loads(line) for line in (quality_dir / f"quality-{p}.jsonl").read_text().splitlines()] for p in ("bf16", "fp8")}
    bundles = {}
    for name in ("gsm8k", "mmlu_pro"):
        rows = [r for r in quality_rows["bf16"] if r["set"] == name]
        dataset, subset, data_rev = HELDOUT[name]
        conditions = {
            "model": {"name": model_name, "revision": revision, "sha256": model_sha256},
            "source_sha256": source_sha256, "runtime_sha256": runtime_sha256,
            "protocol_sha256": accuracy.digest({"settings": settings, "plan": PLAN, "quality": {"temperature": 0, "seed": 17, "thinking": False}}),
            "gpu_count": 1,
            "sampling": {"seed": 17, "temperature": 0, "max_new_tokens": MAX_NEW[name], "replay": "ignore_eos + min_tokens = trace output length", "chat_thinking": False},
            "slo": {"calibration_precision": "bf16", "calibration_sha256": calibration_sha, "ttft_ms": slo["ttft_ms"], "tpot_ms": slo["tpot_ms"]},
            "workload": workload,
            "heldout": {"dataset": dataset, "subset": subset, "revision": data_rev, "normalization": NORMALIZATION[name],
                        "examples": [{"id": r["id"], "prompt_sha256": _sha(r["prompt"]), "label": r["label"]} for r in rows]},
            "planned_rounds": 3,
            "run_plan": [{"id": i, "round": rnd, "variant": v} for i, rnd, v in PLAN],
        }  # fmt: skip
        if candidate:
            conditions["model"]["tokenizer_sha256"] = tokenizer_sha256
            conditions["candidate_model"] = {**candidate, "base_model": model_name}
        c_sha = accuracy.digest(conditions)
        runs = []
        for run_id, rnd, variant in PLAN:
            data = json.loads((raw / f"run-{run_id}.json").read_text())
            requests = []
            for w, r in zip(workload, data["Run"]["Records"], strict=True):
                req = _request(r, w["requested_tokens"])
                req["id"] = w["id"]
                requests.append(req)
            t = timing[run_id]
            if t["precision"] != PRECISION[variant]:
                raise ValueError(f"{run_id} ran as {t['precision']}")
            runs.append({
                "id": run_id, "round": rnd, "variant": variant, "aborted": bool(data["Run"]["Aborted"]),
                "conditions_sha256": c_sha, "config_sha256": accuracy.digest(configs[variant]),
                "started_at_ns": t["started_at_ns"], "finished_at_ns": t["finished_at_ns"],
                "observation_end_s": max(r["end_s"] for r in requests), "requests": requests,
            })  # fmt: skip
        quality = {}
        for variant in ("baseline", "candidate"):
            items = [
                {"id": r["id"], "prompt_sha256": _sha(r["prompt"]), "label": r["label"], "status": r["status"], "answer": r["answer"], "error": r["error"]}
                for r in quality_rows[PRECISION[variant]] if r["set"] == name
            ]  # fmt: skip
            quality[variant] = {"conditions_sha256": c_sha, "config_sha256": accuracy.digest(configs[variant]), "items": items}
        bundles[name] = {"schema": 2, "kind": "measured", "conditions": conditions, "configurations": configs, "runs": runs, "quality": quality}
    return bundles


def _model_manifest(snapshot):
    """HF snapshot files are symlinks to content-addressed blobs (LFS blobs are SHA256)."""
    snapshot = Path(snapshot)
    files = {f.name: Path(f.readlink()).name if f.is_symlink() else hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(snapshot.iterdir())}
    return accuracy.digest(files), files


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--trace", type=Path, default=Path("serving/baseline/traces/azure-60s/evaluation.csv"))
    p.add_argument("--token-ids", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--quality", type=Path, help="quality-only session directory (default: --raw)")
    p.add_argument("--candidate-snapshot", type=Path, help="pre-quantized FP8 checkpoint the session served")
    p.add_argument("--candidate-name", help="its model name, e.g. Qwen/Qwen3-8B-FP8")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    with args.trace.open(newline="") as f:
        trace = [(int(r["ContextTokens"]), int(r["GeneratedTokens"])) for r in csv.DictReader(f)]
    tokens = json.loads(args.token_ids.read_text())
    model_sha, files = _model_manifest(args.snapshot)
    candidate = None
    if args.candidate_snapshot:
        cand_sha, cand_files = _model_manifest(args.candidate_snapshot)
        candidate = {"name": args.candidate_name, "revision": args.candidate_snapshot.name, "sha256": cand_sha,
                     "tokenizer_sha256": cand_files["tokenizer.json"]}  # fmt: skip
    sources = sorted(Path("serving/eval").glob("fp8_*.py")) + [Path("serving/baseline/vllm_baseline.py")] + sorted(Path("serving/replay").glob("*.go"))
    source_sha = accuracy.digest({str(s): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources if not s.name.endswith("_test.go")})
    runtime_sha = accuracy.digest({"vllm": "0.10.2", "torch": "2.8.0+cu128", "python": "3.12.13"})
    bundles = build_bundles(args.raw, trace, tokens["token_ids"], model_sha256=model_sha, runtime_sha256=runtime_sha,
                            source_sha256=source_sha, model_name=args.model_name, revision=args.snapshot.name, quality_dir=args.quality,
                            candidate=candidate, tokenizer_sha256=files["tokenizer.json"])  # fmt: skip
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "model-files.json").write_text(json.dumps(files, indent=1) + "\n")
    if candidate:
        (args.output / "candidate-model-files.json").write_text(json.dumps(cand_files, indent=1) + "\n")
    for name, bundle in bundles.items():
        (args.output / f"bundle-{name}.json").write_text(json.dumps(bundle) + "\n")
        result = accuracy.report(bundle)
        (args.output / f"report-{name}.json").write_text(json.dumps(result) + "\n")
        print(json.dumps({"set": name, "performance_valid": result["performance_valid"], "throughput": result["throughput"],
                          "slo_goodput": result["slo_goodput"], "accuracy": {v: result["quality"][v]["accuracy"] for v in ("baseline", "candidate")},
                          "accuracy_delta": result["accuracy_delta"]}))  # fmt: skip


if __name__ == "__main__":
    main()
