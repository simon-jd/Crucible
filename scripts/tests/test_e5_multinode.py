"""CPU tests for the E5 multi-node payload (scripts/snowflake/payloads/e5_multinode.py)."""
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts" / "snowflake" / "payloads" / "e5_multinode.py"
spec = importlib.util.spec_from_file_location("e5_multinode", PATH)
e5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e5)


def test_bus_factor_follows_nccl_tests():
    assert e5.bus_factor("all_reduce", 2) == 1.0
    assert e5.bus_factor("all_reduce", 4) == 1.5
    assert e5.bus_factor("broadcast", 4) == 1.0
    rows = e5.with_busbw([{"op": "all_reduce", "bytes": 1, "GBps": 1.0}], 3)
    assert rows[0]["busGBps"] == pytest.approx(4 / 3, abs=1e-3)


def test_batch_plans_weak_and_strong():
    weak = e5.batch_plan("weak", 4)
    assert weak["accum"] == 8 and weak["global_seq"] == 64 and weak["tokens_per_step"] == 64 * 1024
    strong = e5.batch_plan("strong", 4)
    assert strong["accum"] == 2 and strong["global_seq"] == 16
    assert e5.batch_plan("strong", 1) == {**e5.batch_plan("weak", 1), "kind": "strong"}
    with pytest.raises(ValueError):
        e5.batch_plan("strong", 3)
    with pytest.raises(ValueError):
        e5.batch_plan("weak", 2, per_gpu_seq=3)


def test_strong_scaling_sees_the_same_global_batch_for_every_world_size():
    for step in (1, 2, 7):
        sets = []
        for world in (1, 2, 4):
            plan = e5.batch_plan("strong", world)
            ids = [i for r in range(world) for i in e5.sequence_ids(step, r, plan, n_seq=10_000)]
            assert len(ids) == len(set(ids)) == 16
            sets.append(ids)
        assert sets[0] == sets[1] == sets[2]
    plan = e5.batch_plan("weak", 2)
    assert e5.sequence_ids(1, 1, plan, n_seq=20)[0] == 16 and e5.sequence_ids(2, 0, plan, n_seq=20)[0] == 12


def test_pack_tokens_appends_eos_and_drops_the_tail():
    rows = e5.pack_tokens([[1, 2, 3], [4, 5]], 3, eos=0)
    assert rows == [[1, 2, 3], [0, 4, 5]]


def test_qwen_structure_matches_the_published_parameter_count():
    cfg = {"hidden_size": 896, "intermediate_size": 4864, "num_hidden_layers": 24, "num_attention_heads": 14,
           "num_key_value_heads": 2, "vocab_size": 151936}
    s = e5.qwen_structure(cfg)
    assert s["active_parameters"] + s["embedding_parameters"] == 494_032_768  # Qwen2.5-0.5B, tied embeddings
    assert (s["layers"], s["heads"], s["head_dim"]) == (24, 14, 64)


def test_prom_roundtrip_feeds_the_controller(tmp_path):
    from cluster.drain.controller import Controller
    from cluster.drain.metrics import StepTimeHook
    now = 1000.0
    metrics = {}
    for i, t in enumerate((1.0, 1.0, 1.1, 2.0)):
        hook = StepTimeHook(f"node-{i}", "e5-test", tmp_path / f"{i}.prom", every=5, clock=lambda: now - 1)
        assert hook.observe(5, t)
        parsed = e5.parse_prom((tmp_path / f"{i}.prom").read_text())
        assert parsed["crucible_step_index"][1] == 5 and parsed["crucible_step_time_seconds"][0]["node"] == f"node-{i}"
        metrics[f"node-{i}"] = e5.node_metrics(parsed, now=now, offset=0.0, link_up=True)
    ctrl = Controller(clock=lambda: now)
    assert ctrl.evaluate(metrics) == []  # one slow sample is not an alert
    assert ctrl.state["node-3"]["count"] == 1 and ctrl.state["node-2"]["count"] == 0


def _drive(times_per_sample):
    """Feed per-sample step times to a real Controller the way phase_drain does and build its log."""
    from cluster.drain.controller import Controller, NodeMetrics
    clock = {"t": 100.0}
    ctrl = Controller(clock=lambda: clock["t"])
    log = []
    for i, times in enumerate(times_per_sample, 1):
        clock["t"] += 10
        metrics = {n: NodeMetrics(clock["t"], 0, True, t, clock["t"] - 1, "run") for n, t in times.items()}
        log.append({"sample": i, "times": times, "alerts": e5.step_alerts(ctrl.evaluate(metrics))})
    return log


def test_verify_drain_flags_only_the_slow_node_after_three_samples():
    healthy = {"a": 1.0, "b": 1.02, "c": 0.99, "d": 1.01}
    slow = {**healthy, "d": 1.9}
    samples = [healthy] * 4 + [slow] * 5 + [healthy] * 3
    log = _drive(samples)
    v = e5.verify_drain(log, "d")
    assert v["match"] and v["healthy_flagged"] == []
    assert v["slow_node_first_slow_sample"] == 5 and v["slow_node_first_alert_sample"] == 7
    assert v["expected_alerts"] == [(7, "d")]


def test_synchronized_step_times_cannot_single_out_a_straggler():
    same = [{"a": 2.0, "b": 2.0, "c": 2.0, "d": 2.0}] * 4 + [{"a": 3.0, "b": 3.0, "c": 3.0, "d": 3.0}] * 5
    v = e5.verify_drain(_drive(same), "d")
    assert v["match"] and not v["slow_node_alerted"] and v["actual_alerts"] == []


def test_kernel_rows_feed_summarize_kernels():
    from scaling.analyze_nsys import summarize_kernels
    trace = {"traceEvents": [
        {"ph": "X", "cat": "kernel", "name": "ampere_bf16_gemm", "ts": 0.0, "dur": 10.0, "args": {"device": 0}},
        {"ph": "X", "cat": "kernel", "name": "ncclDevKernel_AllReduce_Sum_f32", "ts": 5.0, "dur": 15.0,
         "args": {"device": 0}},
        {"ph": "X", "cat": "kernel", "name": "zero", "ts": 30.0, "dur": 0.0, "args": {"device": 0}},
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 0.0, "dur": 1.0},
    ]}
    rows, skipped = e5.kernel_rows(trace)
    assert skipped == 1 and len(rows) == 2
    d = summarize_kernels(rows)["devices"]["0"]
    assert d["kernel_span_ns"] == 20_000 and d["nccl_without_compute_ns"] == 10_000
    # Without CUPTI the trace has no kernel records and the repo's analyzer refuses it.
    with pytest.raises(ValueError):
        summarize_kernels(e5.kernel_rows({"traceEvents": [trace["traceEvents"][3]]})[0])


def test_step_split_separates_exposed_communication():
    # accum 4: three no_sync micro-steps of 0.25 s, the last one 0.25 s compute + 2.0 s all-reduce tail, 0.1 s AdamW.
    steps = [{"t": 0.75 + 2.25 + 0.1, "local": 0.25, "last": 2.25, "opt": 0.1}] * 3
    s = e5.step_split(steps, accum=4, comm_alone=2.6)
    assert s["compute_s"] == pytest.approx(1.1) and s["exposed_comm_s"] == pytest.approx(2.0)
    assert s["exposed_comm_share"] == pytest.approx(2.0 / 3.1)
    assert s["comm_alone_share"] == pytest.approx(2.6 / 3.1) and s["hidden_comm_s"] == pytest.approx(0.6)
    one = e5.step_split([{"t": 1.0, "local": None, "last": 0.9, "opt": 0.1}], accum=1)
    assert "exposed_comm_s" not in one and one["last_micro_s"] == 0.9
    # A last micro-step faster than the no_sync median (noise) gives zero exposure, not a negative share.
    assert e5.step_split([{"t": 1.0, "local": 0.3, "last": 0.29, "opt": 0.1}], accum=3)["exposed_comm_s"] == 0.0


def test_redact_hides_the_account_in_every_case():
    assert e5.redact("https://AB-CD.x/ab-cd", ["Ab-Cd", ""]) == "https://AB-CD.x/ab-cd".replace("AB-CD", "<account>") \
        .replace("ab-cd", "<account>")
    line = ("Looking in indexes: https://1wtauthtoken:****@org1-acct2.registry-local.snowflakecomputing.com/x/simple "
            "jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIwIiwiZXhwIjoxfQ.abcdefghijk "
            "https://ORG1-ACCT2.west-us-2.azure.snowflakecomputing.com")
    out = e5.redact(line)
    assert "org1" not in out.lower() and "eyJ" not in out and "****" not in out
    assert out.count("<account>") == 2 and "<token>" in out and "//<credentials>@" in out


def test_repair_jsonl_drops_a_torn_tail(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n{"c": ')
    assert e5.repair_jsonl(p) == len('{"c": ')
    assert e5.read_jsonl(p) == [{"a": 1}, {"b": 2}] and p.read_text().endswith("\n")
    e5.jsonl_appender(p)({"d": 4})
    assert [next(iter(x)) for x in e5.read_jsonl(p)] == ["a", "b", "d"]
    assert e5.repair_jsonl(p) == 0 and e5.repair_jsonl(tmp_path / "missing.jsonl") == 0


def test_disk_rate_writes_and_cleans_up(tmp_path):
    r = e5.disk_rate(tmp_path / "d" / "w.bin", nbytes=4 << 20, block=1 << 20)
    assert r["bytes"] == 4 << 20 and r["GBps"] > 0 and not (tmp_path / "d" / "w.bin").exists()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def launch_world(tmp, name, cfg, world, per_rank=None):
    port = free_port()
    procs = []
    for r in range(world):
        d = tmp / f"{name}-r{r}"
        d.mkdir()
        c = {**cfg, "rank": r, "world": world, "master": "127.0.0.1", "port": port, "result": str(d / "result.json"),
             "out_dir": str(d), **((per_rank or {}).get(r, {}))}
        (d / "cfg.json").write_text(json.dumps(c))
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)}
        with open(d / "log.txt", "w") as log:
            procs.append(subprocess.Popen([sys.executable, str(PATH), "--mode", "rank", "--cfg", str(d / "cfg.json")],
                                          env=env, stdout=log, stderr=subprocess.STDOUT))
    return procs


def wait_all(procs, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline and any(p.poll() is None for p in procs):
        time.sleep(0.2)
    for p in procs:
        if p.poll() is None:
            p.kill()
            p.wait()
    return [p.returncode for p in procs]


def test_fault_restart_resumes_from_checkpoint_and_accounts_goodput(tmp_path):
    torch = pytest.importorskip("torch")
    from scaling.goodput import account_events
    tokens = tmp_path / "tokens.pt"
    torch.manual_seed(1)
    torch.save(torch.randint(0, 64, (40, 16), dtype=torch.int32), tokens)
    plan = e5.batch_plan("weak", 2, micro=2, per_gpu_seq=4, seq=16)
    events, prog = tmp_path / "events.jsonl", tmp_path / "progress.jsonl"
    cfg = {"task": "train", "kind": "fault", "plan": plan, "model": "tiny", "tokens": str(tokens), "device": "cpu",
           "backend": "gloo", "steps": 6, "ckpt_every": 2, "run_id": "e5-test", "clock_id": "monotonic:test",
           "events": str(events), "progress": str(prog), "ckpt_dir": str(tmp_path / "ckpt"), "lr": 1e-2,
           "pg_timeout": 60, "ckpt_threads": 2}
    # Attempt 0: rank 1 dies abruptly after step 3; rank 0 fails out of (or hangs in) the next all-reduce.
    procs = launch_world(tmp_path, "a0", {**cfg, "attempt": 0}, 2, per_rank={1: {"exit_after_step": 3}})
    assert wait_all(procs[1:], 120) == [137]
    wait_all(procs[:1], 20)
    assert e5.record_failure(events, "rank 1 exited") in ("driver", "rank0 operation_failed")
    procs = launch_world(tmp_path, "a1", {**cfg, "attempt": 1, "resume": True}, 2)
    rcs = wait_all(procs, 180)
    assert rcs == [0, 0], [(tmp_path / f"a1-r{r}" / "log.txt").read_text()[-3000:] for r in range(2)]
    log = e5.read_jsonl(events)
    acct = account_events(log)
    assert acct["retained_steps"] == 6 and acct["lost_steps"] == 1  # step 3 ran after the step-2 checkpoint
    assert 0 < acct["goodput"] < 1 and len(acct["recovery_seconds"]) == 1
    assert acct["final_checkpoint_id"].endswith("step-6@generation-0")
    kinds = [e["kind"] for e in log]
    assert kinds.count("failure") + kinds.count("operation_failed") == 1 and kinds[-1] == "run_end"
    # Restored state reproduces the lost step exactly (deterministic data order, CPU).
    losses = {}
    for p in e5.read_jsonl(prog):
        losses.setdefault(p["step"], {})[p["attempt"]] = p["loss"]
    assert losses[3][0] == pytest.approx(losses[3][1], abs=1e-6)
    steps = sorted(p.name for p in (tmp_path / "ckpt").iterdir() if p.name.startswith("step-"))
    assert steps == ["step-4", "step-6"]  # CheckpointRetention keeps two
    res = json.loads((tmp_path / "a1-r0" / "result.json").read_text())
    assert res["resumed_from"] == 2 and res["final_checkpoint"] == acct["final_checkpoint_id"]


def _tiny_tokens(tmp_path, torch):
    tokens = tmp_path / "tokens.pt"
    torch.manual_seed(1)
    torch.save(torch.randint(0, 64, (40, 16), dtype=torch.int32), tokens)
    return tokens


def test_scale_world_reports_split_and_comm_alone(tmp_path):
    torch = pytest.importorskip("torch")
    plan = e5.batch_plan("weak", 2, micro=2, per_gpu_seq=4, seq=16)
    cfg = {"task": "train", "kind": "scale", "plan": plan, "model": "tiny", "tokens": str(_tiny_tokens(tmp_path, torch)),
           "device": "cpu", "backend": "gloo", "warmup_steps": 1, "timed_steps": 3, "comm_alone_reps": 2,
           "pg_timeout": 60}
    procs = launch_world(tmp_path, "s", cfg, 2)
    assert wait_all(procs, 120) == [0, 0], (tmp_path / "s-r0" / "log.txt").read_text()[-3000:]
    res = [json.loads((tmp_path / f"s-r{r}" / "result.json").read_text()) for r in range(2)]
    for r in res:
        assert len(r["step_seconds"]) == len(r["last_micro_seconds"]) == len(r["opt_seconds"]) == 3
        assert r["comm_alone"]["bytes"] == 4 * r["parameters"] and len(r["comm_alone"]["seconds"]) == 2
        assert set(r["split"]) >= {"compute_s", "exposed_comm_s", "exposed_comm_share", "comm_alone_s"}
        assert r["global_tokens_per_s"] == pytest.approx(3 * plan["tokens_per_step"] / r["window_max_rank_seconds"], rel=0.01)
    assert res[0]["window_max_rank_seconds"] == res[1]["window_max_rank_seconds"]


def test_drain_world_publishes_both_step_time_feeds(tmp_path):
    torch = pytest.importorskip("torch")
    plan = e5.batch_plan("weak", 2, micro=2, per_gpu_seq=4, seq=16)
    cfg = {"task": "train", "kind": "drain", "plan": plan, "model": "tiny", "tokens": str(_tiny_tokens(tmp_path, torch)),
           "device": "cpu", "backend": "gloo", "steps": 6, "report_every": 3, "run_id": "e5-test", "pg_timeout": 60}
    per = {r: {"node": f"node-{r}", "prom_dir": str(tmp_path / f"prom{r}")} for r in range(2)}
    procs = launch_world(tmp_path, "d", cfg, 2, per_rank=per)
    assert wait_all(procs, 120) == [0, 0], (tmp_path / "d-r0" / "log.txt").read_text()[-3000:]
    for r in range(2):
        for feed in ("sync", "local"):
            p = e5.parse_prom((tmp_path / f"prom{r}" / f"{feed}.prom").read_text())
            assert p["crucible_step_index"][1] == 6
            assert p["crucible_step_time_seconds"][0] == {"node": f"node-{r}", "run_id": f"e5-test-{feed}"}
        steps = json.loads((tmp_path / f"d-r{r}" / "result.json").read_text())["steps"]
        assert [s["step"] for s in steps] == list(range(1, 7)) and all(s["local"] <= s["t"] for s in steps)


def test_fault_plan_interleaves_disk_rounds_and_one_shm_round():
    from types import SimpleNamespace
    a = SimpleNamespace(quick=False, fault_rounds=3, fault_shm_rounds=1)
    assert e5.fault_plan(a) == [(1, "disk"), (1, "shm"), (2, "disk"), (3, "disk")]
    assert e5.fault_plan(SimpleNamespace(quick=True, fault_rounds=3, fault_shm_rounds=0)) == [(1, "disk")]
    assert set(e5.FAULT_STORES) == {"disk", "shm"} and str(e5.FAULT_STORES["shm"]).startswith("/dev/shm/")
