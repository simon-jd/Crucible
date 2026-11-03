import json

import pytest

from serving.eval import accuracy
from serving.eval import fp8_bundle as fb
from serving.eval.fp8_session import PLAN


def test_prompt_ids_match_the_go_replayer():
    # Same values as serving/replay TestSyntheticPromptIDsArePinned.
    assert fb.prompt_ids([5, 3], [100, 101, 102, 103], seed=1) == [[100, 103, 101, 101, 100], [102, 100, 101]]


def rec(sched, sent, ttft, tpot, done, out, status="ok", error=""):
    return {"ScheduledMs": sched, "SentMs": sent, "TTFTms": ttft, "TPOTms": tpot, "CompletedMs": done,
            "OutputTokens": out, "Coalesced": False, "Status": status, "Error": error}  # fmt: skip


def replay(records):
    return {"Trace": {"SHA256": "ab" * 32}, "Config": {"Seed": 17, "Multiplier": 0.25},
            "Run": {"Records": records, "SenderValid": True, "Aborted": False, "DurationMs": max(r["CompletedMs"] for r in records)}}  # fmt: skip


@pytest.fixture
def raw(tmp_path):
    good = [rec(0, 0.5, 10, 2, 40, 4), rec(100, 100.4, 12, 3, 150, 4)]
    slow = [rec(0, 0.5, 30, 9, 80, 4), rec(100, 101.0, 12, 3, 150, 3, "stream_error", "stream read: context deadline exceeded")]
    for i in range(3):
        (tmp_path / f"calibration-{i}.json").write_text(json.dumps(replay(good)))
    runs, t = [], 1_000
    for run_id, _, variant in PLAN:
        (tmp_path / f"run-{run_id}.json").write_text(json.dumps(replay(good if variant == "baseline" else slow)))
        runs.append({"id": run_id, "precision": fb.PRECISION[variant], "exit": 0, "started_at_ns": t, "finished_at_ns": t + 10**9})
        t += 2 * 10**9
    (tmp_path / "session.json").write_text(json.dumps({"runs": runs, "servers": [{"precision": "bf16", "command": ["--model", "/m", "--seed", "17"]}], "stop_reason": None}))
    for precision, answer in (("bf16", "#### 7"), ("fp8", "#### 8")):
        lines = [
            {"set": "gsm8k", "id": "g1", "label": "7", "prompt": "Q", "status": "success", "answer": answer, "error": None},
            {"set": "mmlu_pro", "id": "m1", "label": "B", "prompt": "P", "status": "success", "answer": "B", "error": None},
        ]
        (tmp_path / f"quality-{precision}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return tmp_path


def test_bundles_pass_accuracy_report_and_keep_failures(raw):
    trace = [(8, 4), (9, 4)]  # (input, output) tokens per workload row
    bundles = fb.build_bundles(raw, trace, [1, 2, 3], model_sha256="1" * 64, runtime_sha256="2" * 64,
                               source_sha256="3" * 64, model_name="Qwen/Qwen3-8B", revision="r" * 40)  # fmt: skip
    gsm = accuracy.report(bundles["gsm8k"])
    assert gsm["performance_valid"] and gsm["rounds"] == 3
    assert gsm["accuracy_delta"] == -1.0 and gsm["quality"]["baseline"]["accuracy"] == 1.0
    fp8_run = next(r for r in gsm["runs"] if r["variant"] == "candidate")
    assert fp8_run["counts"]["error"] == 1 and fp8_run["counts"]["success"] == 1
    mmlu = accuracy.report(bundles["mmlu_pro"])
    assert mmlu["accuracy_delta"] == 0.0 and mmlu["conditions"]["heldout"]["normalization"] == "choice-letter-v1"
    # SLO comes from the BF16 calibration runs (pooled p99 of 6 samples -> max).
    assert bundles["gsm8k"]["conditions"]["slo"]["ttft_ms"] == 12 and bundles["gsm8k"]["conditions"]["slo"]["tpot_ms"] == 3


def test_quality_session_must_use_the_same_server_arguments(raw, tmp_path_factory):
    quality = tmp_path_factory.mktemp("quality")
    for name in ("quality-bf16.jsonl", "quality-fp8.jsonl"):
        (quality / name).write_text((raw / name).read_text())
    kwargs = {"model_sha256": "1" * 64, "runtime_sha256": "2" * 64, "source_sha256": "3" * 64, "model_name": "m", "revision": "r" * 40}
    same = [{"precision": "bf16", "command": ["--model", "/m", "--seed", "17"]}]
    (quality / "session.json").write_text(json.dumps({"servers": same}))
    assert fb.build_bundles(raw, [(8, 4), (9, 4)], [1, 2, 3], quality_dir=quality, **kwargs)
    (quality / "session.json").write_text(json.dumps({"servers": [{"precision": "bf16", "command": ["--seed", "18"]}]}))
    with pytest.raises(ValueError, match="differ"):
        fb.build_bundles(raw, [(8, 4), (9, 4)], [1, 2, 3], quality_dir=quality, **kwargs)


def test_prequantized_candidate_declares_its_checkpoint(raw):
    session = json.loads((raw / "session.json").read_text())
    session["candidate_snapshot"] = "/hub/models--Qwen--Qwen3-8B-FP8/snapshots/" + "c" * 40
    (raw / "session.json").write_text(json.dumps(session))
    kwargs = {"model_sha256": "1" * 64, "runtime_sha256": "2" * 64, "source_sha256": "3" * 64,
              "model_name": "Qwen/Qwen3-8B", "revision": "r" * 40, "tokenizer_sha256": "4" * 64}  # fmt: skip
    candidate = {"name": "Qwen/Qwen3-8B-FP8", "revision": "c" * 40, "sha256": "5" * 64, "tokenizer_sha256": "4" * 64}
    bundles = fb.build_bundles(raw, [(8, 4), (9, 4)], [1, 2, 3], candidate=candidate, **kwargs)
    result = accuracy.report(bundles["gsm8k"])
    assert result["performance_valid"] and result["candidate_checkpoint"] == "Qwen/Qwen3-8B-FP8"
    cand = bundles["gsm8k"]["conditions"]["candidate_model"]
    assert cand["base_model"] == "Qwen/Qwen3-8B" and bundles["gsm8k"]["configurations"]["candidate"]["model_sha256"] == "5" * 64
    assert "pre-quantized" in bundles["gsm8k"]["configurations"]["candidate"]["backend"]
    with pytest.raises(ValueError, match="candidate"):
        fb.build_bundles(raw, [(8, 4), (9, 4)], [1, 2, 3], candidate={**candidate, "revision": "d" * 40}, **kwargs)
    with pytest.raises(ValueError, match="candidate"):
        fb.build_bundles(raw, [(8, 4), (9, 4)], [1, 2, 3], **kwargs)
