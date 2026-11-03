import socket
import subprocess
import sys
from pathlib import Path

import pytest

from serving.baseline.experiment import calibrate, evaluate, stop_child, trial_order


def record(sent=0, status="ok", tokens=2, ttft=3, tpot=4):
    return {
        "ScheduledMs": 0,
        "SentMs": sent,
        "TTFTms": ttft,
        "TPOTms": tpot,
        "OutputTokens": tokens,
        "Status": status,
        "Coalesced": False,
    }


def run(records):
    return {"Records": records, "SenderValid": True, "Aborted": False, "DurationMs": 1000}


def test_slo_uses_frozen_calibration_and_all_planned_eval_denominators():
    runs = [
        run([record(ttft=x), record(status="http_error", tokens=0, ttft=-1)]) for x in (3, 5, 7)
    ]
    slo = calibrate(runs, ["a", "b", "c"])
    assert slo["TTFTms"] == 7 and slo["Samples"] == 3
    assert slo["failed"] == 3
    metric = evaluate(run([record(), record(status="http_error", tokens=0)]), slo, "eval")
    assert metric["goodput"] == 1 and metric["planned"] == 2
    assert metric["failed"] == 1
    assert metric["slo_sha256"] == slo["sha256"]


def test_invalid_sender_and_undefined_tokens_cannot_create_accepted_goodput():
    with pytest.raises(ValueError, match="sender"):
        calibrate([run([record(sent=5.001)])] * 3, ["a", "b", "c"])
    slo = calibrate([run([record()])] * 3, ["a", "b", "c"])
    result = evaluate(run([record(tokens=1, tpot=-1), record(sent=-1)]), slo, "eval")
    assert result["accepted"] is False and result["goodput"] is None
    assert result["planned"] == 2
    assert result["undefined"] == 1


def test_trial_order_is_fixed_counterbalanced_pairs():
    order = trial_order()
    assert [x["multiplier"] for x in order] == [1, 2, 2, 1, 1, 2, 4, 1, 1, 4, 4, 1]
    assert len({x["pair"] for x in order}) == 6


def test_owned_server_child_and_listener_exit_normally():
    child = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-c",
            (
                'import socket,time; s=socket.socket(); s.bind(("127.0.0.1",0)); '
                "s.listen(); print(s.getsockname()[1],flush=True); time.sleep(60)"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        port = int(child.stdout.readline())
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
        evidence = stop_child(child, port, grace=2)
        assert evidence["reaped"] and evidence["listener_closed"]
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=2)
        child.stdout.close()


def test_calibration_lineage_binds_model_runtime_source_and_server_config():
    lineage = {
        "model_manifest_sha256": "model-a",
        "runtime_sha256": "runtime-a",
        "sources_sha256": "source-a",
        "server_args": ["--dtype", "bfloat16"],
    }
    slo = calibrate([run([record()])] * 3, ["a", "b", "c"], lineage)
    assert slo["lineage"] == lineage
    with pytest.raises(ValueError, match="lineage"):
        evaluate(run([record()]), slo, "eval", {**lineage, "runtime_sha256": "changed"})


def test_matched_pairs_keep_invalid_and_missing_members_without_accepted_aggregate():
    from serving.baseline.experiment import matched_pairs

    order = trial_order()
    summaries = [
        {
            "trial": trial,
            "identity": str(i),
            "metrics": {"accepted": i != 1, "goodput": None if i == 1 else trial["multiplier"]},
        }
        for i, trial in enumerate(order)
    ]
    result = matched_pairs(summaries, order)
    assert len(result["pairs"]) == 6
    assert result["pairs"][0]["accepted"] is False
    assert result["comparisons"]["2x"]["goodput_difference"] is None
    assert result["comparisons"]["4x"]["goodput_difference"]["median"] == 3
    partial = matched_pairs(summaries[:1], order)
    assert len(partial["pairs"][0]["members"]) == 2
    assert partial["pairs"][0]["members"][1]["status"] == "not_run"


def test_fixed_warmups_execute_real_http_outside_source_windows(tmp_path):
    import http.server
    import json
    import threading

    from serving.baseline.experiment import fixed_warmups

    received = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(payload)
            count = payload["sampling_params"]["max_new_tokens"]
            data = json.dumps(
                {
                    "output_ids": [3] * count,
                    "meta_info": {
                        "prompt_tokens": len(payload["input_ids"]),
                        "completion_tokens": count,
                        "finish_reason": {"type": "length"},
                    },
                }
            )
            body = f"data: {data}\n\ndata: [DONE]\n\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        specs = [
            {"input_tokens": 128, "output_tokens": 16},
            {"input_tokens": 1024, "output_tokens": 32},
            {"input_tokens": 2048, "output_tokens": 32},
        ]
        fixed_warmups(f"http://127.0.0.1:{server.server_port}/generate", [4, 5], specs, tmp_path)
        assert [len(x["input_ids"]) for x in received] == [128, 1024, 2048]
        assert [x["sampling_params"]["max_new_tokens"] for x in received] == [16, 32, 32]
        assert len(list(tmp_path.glob("warmup-*.raw"))) == 3
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_live_observer_scans_only_mutable_roots_and_stops_before_reserve(tmp_path, monkeypatch):
    from serving.baseline import experiment

    work, output = tmp_path / "work", tmp_path / "output"
    visited = []

    def size(path, rescans=None):
        visited.append(path)
        assert path in (work, output)
        return 10

    monkeypatch.setattr(experiment, "canonical_bytes", size)
    monkeypatch.setattr(experiment, "_owned_rss", lambda: 1000)
    sample = experiment.check_resources(work, 13_000_000_000, output, canonical=False)
    assert visited == [work, output]
    assert sample["project_bytes"] == 13_050_000_020
    monkeypatch.setattr(experiment, "_owned_rss", lambda: 16_000_000_000)
    with pytest.raises(RuntimeError, match="high-water"):
        experiment.check_resources(work, 13_000_000_000, output, canonical=False)


def test_abi_only_controller_reaps_real_fixture_child_without_model_inputs(tmp_path, monkeypatch):
    import json

    from serving.baseline import experiment

    lock = tmp_path / "lock.json"
    lock.write_text("{}")
    out = tmp_path / "evidence"
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    # ABI-only mode never opens or needs a server port; None still denotes no listener.
    monkeypatch.setattr(experiment, "listener_closed", lambda port: port is None)
    original = subprocess.Popen

    def fixture(command, **kwargs):
        assert "--mode" in command and command[command.index("--mode") + 1] == "abi"
        target = command[command.index("--inventory-out") + 1]
        return original(
            [
                sys.executable,
                "-B",
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{}')",
                target,
            ],
            **kwargs,
        )

    monkeypatch.setattr(experiment.subprocess, "Popen", fixture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experiment",
            "--abi-only",
            "--lock",
            str(lock),
            "--overlay",
            str(tmp_path / "overlay"),
            "--cache-audit",
            str(tmp_path / "cache.json"),
            "--output",
            str(out),
        ],
    )
    experiment.main()
    cleanup = json.loads((out / "abi-cleanup.json").read_text())
    assert cleanup["reaped"] and not cleanup["forced_kill"]
    assert (out / "abi.json").read_text() == "{}"
    assert not (out / "server.log").exists()


def test_attempted_trial_retains_partial_stdout_and_reaps_child(tmp_path):
    import json

    from serving.baseline.experiment import execute_trial

    target = tmp_path / "attempt.json"
    summary = {"trial": {"pair": "2x-1", "multiplier": 1}, "planned_rows": 23}
    command = [sys.executable, "-B", "-c", "print('{partial',flush=True)"]
    with pytest.raises(json.JSONDecodeError):
        execute_trial(command, {}, target, summary, lambda: None)
    evidence = json.loads(target.with_suffix(".summary.json").read_text())
    assert evidence["status"] == "failed" and evidence["metrics"]["accepted"] is False
    assert evidence["planned_rows"] == 23 and evidence["returncode"] == 0
    assert evidence["raw_sha256"] and evidence["cleanup"]["reaped"]


def test_interrupted_trial_reaps_exact_owned_child_and_preserves_attempt(tmp_path):
    import json

    from serving.baseline.experiment import execute_trial

    def interrupted():
        raise InterruptedError("fixture interruption")

    target = tmp_path / "attempt.json"
    summary = {"trial": {"pair": "2x-1", "multiplier": 1}, "planned_rows": 23}
    with pytest.raises(InterruptedError, match="fixture"):
        execute_trial(
            [sys.executable, "-B", "-c", "import time;time.sleep(60)"],
            {},
            target,
            summary,
            interrupted,
        )
    evidence = json.loads(target.with_suffix(".summary.json").read_text())
    assert evidence["status"] == "failed" and evidence["cleanup"]["reaped"]
    assert evidence["metrics"]["goodput"] is None and evidence["planned_rows"] == 23


@pytest.mark.parametrize(
    "variant,expected_failure_exit",
    [
        ("known", True),
        ("unknown", False),
        ("code2", False),
        ("signal", False),
        ("partial", False),
        ("aborted", False),
        ("invalid_sender", False),
        ("no_failed_rows", False),
        ("oversized_stderr", False),
    ],
)
def test_trial_distinguishes_complete_request_failure_exit(
    tmp_path, variant, expected_failure_exit
):
    import json
    import signal

    from serving.baseline.experiment import execute_trial

    result = run([record(), {**record(status="http_error", tokens=0), "Error": "HTTP 400"}])
    stderr = "run contains failed requests\n"
    code = 1
    if variant == "unknown":
        stderr = "unexpected execution failure\n"
    elif variant == "code2":
        code = 2
    elif variant == "partial":
        result["Records"].pop()
    elif variant == "aborted":
        result["Aborted"] = True
    elif variant == "invalid_sender":
        result["Records"][0]["SentMs"] = -1
    elif variant == "no_failed_rows":
        result["Records"][1] = record()
    elif variant == "oversized_stderr":
        stderr += "x" * 4096
    source = (
        "import os,signal,sys;"
        f"print({json.dumps({'Run': result})!r},flush=True);"
        f"sys.stderr.write({stderr!r});sys.stderr.flush();"
        + ("os.kill(os.getpid(),signal.SIGTERM)" if variant == "signal" else f"sys.exit({code})")
    )
    target = tmp_path / "trial.json"
    summary = {"planned_rows": 2}
    actual = execute_trial([sys.executable, "-B", "-c", source], {}, target, summary, lambda: None)
    assert actual["Aborted"] is not expected_failure_exit
    assert summary["returncode"] == (-signal.SIGTERM if variant == "signal" else code)
    assert target.with_suffix(".stderr").read_text() == stderr
    assert json.loads(target.read_text())["Run"] == result
    assert summary["cleanup"]["reaped"]
    if expected_failure_exit:
        slo = calibrate([actual] * 3, ["a", "b", "c"])
        metrics = evaluate(actual, slo, "held-out")
        assert metrics["accepted"] and metrics["failed"] == 1 and metrics["planned"] == 2
        assert metrics["goodput"] == 1 and actual["Records"][1]["Error"] == "HTTP 400"


def test_verified_model_inputs_reject_mutated_blob_or_allowlist(tmp_path):
    import hashlib
    import json

    from serving.baseline.experiment import verify_model_inputs
    from serving.replay.tokenizer_allowlist import build

    revision = "a" * 40
    model = tmp_path / "models--Qwen--Qwen3-8B" / "snapshots" / revision
    model.mkdir(parents=True)
    blobs = model.parent.parent / "blobs"
    blobs.mkdir()
    docs = {
        "config.json": {"vocab_size": 8},
        "tokenizer.json": {"model": {"vocab": {"a": 1, "b": 2}}},
        "tokenizer_config.json": {},
    }
    plan = {"repo": "Qwen/Qwen3-8B", "revision": revision, "files": []}
    manifest = {"status": "verified", "repo": plan["repo"], "revision": revision, "files": []}
    for name, data in docs.items():
        raw = json.dumps(data).encode()
        digest = hashlib.sha256(raw).hexdigest()
        blob = blobs / digest
        blob.write_bytes(raw)
        (model / name).symlink_to(blob)
        item = {
            "name": name,
            "bytes": len(raw),
            "algorithm": "sha256",
            "upstream_hash": digest,
            "blob": digest,
        }
        plan["files"].append(item)
        manifest["files"].append(
            {**item, "sha256": digest, "blob_path": str(blob), "snapshot_path": str(model / name)}
        )
    tokens = build(model, plan["repo"], revision)
    verify_model_inputs(plan, manifest, model, tokens)
    with pytest.raises(ValueError, match="allowlist"):
        verify_model_inputs(plan, manifest, model, {**tokens, "token_ids": [7]})
    (blobs / plan["files"][0]["blob"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="size"):
        verify_model_inputs(plan, manifest, model, tokens)


def test_full_controller_reuses_prior_abi_without_spawning_another_probe(tmp_path, monkeypatch):
    import contextlib
    import json

    from serving.baseline import experiment

    root = experiment.ROOT
    out = tmp_path / "out"
    lock = tmp_path / "lock.json"
    revision = "b968826d9c46dd6066d109eabc6255188de91218"
    lock.write_text(
        json.dumps({"model": {"revision": revision, "repo": "Qwen/Qwen3-8B", "vocab_size": 151936}})
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "traces": [],
                "trial_order": experiment.trial_order(),
                "server_args": [],
                "warmup": [],
                "trial_boundary": experiment.TRIAL_BOUNDARY,
            }
        )
    )
    tokens = tmp_path / "tokens.json"
    tokens.write_text(
        json.dumps({"Revision": revision, "Model": "Qwen/Qwen3-8B", "token_ids": [1, 2]})
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "plan_sha256": experiment.sha256(root / "serving/baseline/model-files.json"),
                "runtime_lock_sha256": experiment.sha256(lock),
            }
        )
    )
    abi = tmp_path / "prior-abi.json"
    abi.write_text('{"inventory": []}')
    monkeypatch.setattr(experiment, "verify_model_inputs", lambda *args: None)
    monkeypatch.setattr(experiment, "inventory", lambda *args: [])
    monkeypatch.setattr(experiment, "listener_closed", lambda *args: True)
    monkeypatch.setattr(experiment, "check_resources", lambda *args: {"fixture": True})
    monkeypatch.setattr(experiment, "native_smoke", lambda *args: None)
    monkeypatch.setattr(experiment, "fixed_warmups", lambda *args: None)
    monkeypatch.setattr(experiment, "flush_barrier", lambda *args: {"accepted": True})
    monkeypatch.setattr(
        experiment.urllib.request,
        "urlopen",
        lambda *args, **kwargs: contextlib.nullcontext(type("Ready", (), {"status": 200})()),
    )
    original = subprocess.Popen
    modes = []

    def spawn(command, **kwargs):
        modes.append(command[command.index("--mode") + 1])
        (out / "server-inventory.json").write_text("{}")
        return original([sys.executable, "-B", "-c", "import time;time.sleep(60)"], **kwargs)

    monkeypatch.setattr(experiment.subprocess, "Popen", spawn)
    model = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots" / revision
    binary = tmp_path / "replay"
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    args = {
        "lock": lock,
        "overlay": tmp_path / "overlay",
        "cache-audit": root / "serving/baseline/cache-audit.json",
        "output": out,
        "model": model,
        "model-manifest": manifest,
        "protocol": protocol,
        "token-ids": tokens,
        "binary": binary,
        "abi-evidence": abi,
        "abi-evidence-sha256": experiment.sha256(abi),
    }
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "experiment",
            *[value for key, value in args.items() for value in ("--" + key, str(value))],
        ],
    )
    experiment.main()
    assert modes == ["server"] and not (out / "abi.log").exists()
    assert json.loads((out / "abi-reference.json").read_text())["rerun"] is False


def test_native_smoke_retains_bad_stream_and_failure_summary(tmp_path):
    import http.server
    import json
    import threading

    from serving.baseline.experiment import native_smoke

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = b"data: {malformed\n\n"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        with pytest.raises(ValueError):
            native_smoke(
                f"http://127.0.0.1:{server.server_port}/generate", [1, 2], 151936, tmp_path
            )
        result = json.loads((tmp_path / "native-smoke.json").read_text())
        assert len(result) == 1 and result[0]["accepted_contract"] is False
        assert result[0]["error"] and result[0]["raw_sha256"]
        assert (tmp_path / "native-smoke-zero.raw").read_bytes() == b"data: {malformed\n\n"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_native_absolute_deadline_interrupts_waiting_headers_and_joins_watchdog():
    import http.server
    import threading
    import time

    from serving.baseline.experiment import _native_response

    release = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            release.wait(timeout=2)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    before = set(threading.enumerate())
    began = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="absolute deadline"):
            _native_response(
                f"http://127.0.0.1:{server.server_port}/generate",
                b"{}",
                bytearray(),
                {},
                lambda: None,
                seconds=0.1,
            )
        assert time.monotonic() - began < 1
        assert not [
            t for t in threading.enumerate() if t not in before and isinstance(t, threading.Timer)
        ]
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


@pytest.fixture
def flush_server():
    import http.server
    import threading

    state = {"statuses": [400, 400, 200], "calls": []}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            state["calls"].append(self.path)
            status = state["statuses"][min(len(state["calls"]) - 1, len(state["statuses"]) - 1)]
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    yield state, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)


def test_flush_barrier_waits_for_supported_idle_success(tmp_path, flush_server):
    from serving.baseline.experiment import flush_barrier

    state, endpoint = flush_server
    result = flush_barrier(endpoint, tmp_path / "barrier.json", lambda: None, seconds=1)
    assert result["accepted"] and [a["status"] for a in result["attempts"]] == [400, 400, 200]
    assert state["calls"] == ["/flush_cache"] * 3


@pytest.mark.parametrize("status,exception", [(400, TimeoutError), (500, ValueError)])
def test_flush_barrier_stops_on_perpetual_busy_or_unexpected_status(
    tmp_path, flush_server, status, exception
):
    import json

    from serving.baseline.experiment import flush_barrier

    state, endpoint = flush_server
    state["statuses"] = [status]
    target = tmp_path / "barrier.json"
    with pytest.raises(exception):
        flush_barrier(endpoint, target, lambda: None, seconds=0.12)
    result = json.loads(target.read_text())
    assert not result["accepted"] and result["error"]
    assert result["attempts"][0]["status"] == status
    assert result["elapsed_seconds"] < 1


def test_trial_boundary_preserves_completed_current_outcome_before_stopping(tmp_path, monkeypatch):
    import json

    from serving.baseline import experiment

    target = tmp_path / "trial.json"
    summary = {"planned_rows": 1}
    payload = json.dumps({"Run": run([record()])})
    command = [sys.executable, "-B", "-c", "import sys;print(sys.argv[1])", payload]

    def failed_boundary(*args, **kwargs):
        saved = json.loads(target.with_suffix(".summary.json").read_text())
        assert saved["status"] == "completed" and saved["actual_records"] == 1
        assert saved["raw_sha256"] and saved["cleanup"]["reaped"]
        raise TimeoutError("server remained busy")

    monkeypatch.setattr(experiment, "flush_barrier", failed_boundary)
    later = []
    with pytest.raises(TimeoutError):
        experiment.execute_isolated_trial(
            command, {}, target, summary, lambda: None, "http://127.0.0.1:1"
        )
        later.append("must not start")
    saved = json.loads(target.with_suffix(".summary.json").read_text())
    assert saved["status"] == "completed" and saved["boundary_error"]
    assert saved["actual_output_tokens"] == 2 and not later
