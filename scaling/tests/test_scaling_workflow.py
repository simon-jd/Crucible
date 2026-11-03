import json
import sqlite3
import subprocess
import tomllib
from pathlib import Path

import pytest

from scaling.analyze_nsys import analyze_sqlite, summarize_kernels
from scaling.experiment import build_plan, summarize_results
from scaling.train import CompletedStepTimer, instrument_trainer

ROOT = Path(__file__).resolve().parents[2]


def test_trace_unions_do_not_sum_concurrent_durations(tmp_path):
    path = tmp_path / "fixture.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE StringIds(id INTEGER,value TEXT);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(deviceId INTEGER,start INTEGER,"
            "end INTEGER,demangledName INTEGER);"
        )
        db.executemany("INSERT INTO StringIds VALUES (?,?)", [(1, "gemm"), (2, "ncclAllToAll")])
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)",
            [(0, 0, 100, 1), (0, 50, 150, 2), (0, 75, 125, 2), (1, 0, 20, 2)],
        )
    result = analyze_sqlite(path)
    d = result["devices"]["0"]
    assert d["nccl_duration_sum_ns"] == 150
    assert d["nccl_union_ns"] == 100
    assert d["nccl_compute_overlap_ns"] == 50
    assert d["nccl_without_compute_ns"] == 50
    assert d["named_all_to_all_union_ns"] == 100
    assert result["devices"]["1"]["nccl_without_compute_ns"] == 20
    assert "not a critical path" in result["interpretation"]


def test_trace_rejects_missing_empty_unknown_and_bad_intervals(tmp_path):
    path = tmp_path / "missing.sqlite"
    with pytest.raises(sqlite3.Error):
        analyze_sqlite(path)
    assert not path.exists()
    with pytest.raises(ValueError):
        summarize_kernels([])
    for row in [(0, 10, 9, "gemm"), (0, 1, 2, None)]:
        with pytest.raises(ValueError):
            summarize_kernels([row])


def test_plan_covers_scales_precisions_ep_and_three_alternating_rounds():
    plan = build_plan()
    assert plan["status"] == "pending-authorization"
    assert plan["upstream"]["commit"] == "29624e37fa6dd319fd671b009338651b5c665da2"
    comparisons = plan["comparisons"]
    assert {(c["gpus"], c["precision"]) for c in comparisons if c["purpose"] == "tuning"} == {
        (g, p) for g in (8, 16, 32) for p in ("bf16", "fp8")
    }
    assert {c["candidate_ep"] for c in comparisons if c["purpose"] == "expert_parallel"} == {2, 4}
    for comparison in comparisons:
        runs = comparison["runs"]
        assert [(r["round"], r["variant"]) for r in runs] == [
            (n, v) for n in range(1, 4) for v in ("baseline", "candidate")
        ]
        for run in runs:
            assert len(run["nodes"]) == comparison["gpus"] // 8
            for rank, command in enumerate(run["nodes"]):
                assert command["env"]["NCCL_DEBUG"] == "INFO"
                assert "--node-rank" in command["argv"]
                assert command["argv"][command["argv"].index("--node-rank") + 1] == str(rank)
                assert "-m" in command["argv"]
                assert "scaling.train" in command["argv"]
            if run["fp8"]:
                assert "--model.converters" in run["nodes"][0]["argv"]


def test_offline_runner_does_not_launch_training():
    result = subprocess.run(
        ["bash", "scaling/run.sh", "plan"], check=False, cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "pending-authorization"
    refused = subprocess.run(
        ["bash", "scaling/run.sh", "launch"], check=False, cwd=ROOT, capture_output=True
    )
    assert refused.returncode != 0


def test_configs_use_v02_schema_and_real_small_moe():
    configs = [tomllib.loads(p.read_text()) for p in (ROOT / "scaling/configs").glob("*.toml")]
    llama = next(c for c in configs if c["model"]["name"] == "llama3")
    moe = next(c for c in configs if c["model"]["name"] == "deepseek_v3")
    assert llama["parallelism"]["tensor_parallel_degree"] == 2
    assert llama["training"]["global_batch_size"] == 256
    assert moe["model"]["flavor"] == "debugmodel"
    assert moe["parallelism"]["expert_parallel_degree"] == 2
    assert moe["parallelism"]["expert_tensor_parallel_degree"] == 1
    for config in configs:
        assert config["training"]["mixed_precision_param"] == "bfloat16"
        assert config["parallelism"]["data_parallel_shard_degree"] == -1
        assert config["parallelism"]["fsdp_reshard_after_forward"] == "always"
        assert config["checkpoint"]["enable"] is False
        assert config["validation"]["enable"] is False
        assert "float8" in config["quantize"]["linear"]


def test_completed_steps_emit_real_hook_at_50_and_exclude_warmup(tmp_path):
    from cluster.drain.metrics import StepTimeHook

    events, reports = [], []
    clock_values = iter([v for i in range(100) for v in (i * 3, i * 3 + 2)])
    hook = StepTimeHook("node-a", "comparable-round", tmp_path / "step.prom")
    timer = CompletedStepTimer(
        hook=hook,
        synchronize=lambda: events.append("sync"),
        clock=lambda: next(clock_values),
        tokens_per_step=1024,
        global_max=lambda seconds: seconds * 2,
        emit=reports.append,
    )
    for step in range(1, 101):
        timer.measure(step, lambda: events.append("train"))
    assert events[:3] == ["sync", "train", "sync"]
    assert len(reports) == 1
    assert reports[0]["global_tokens_per_s"] == 256
    assert reports[0]["completed_steps"] == 100
    assert (
        'crucible_step_index{node="node-a",run_id="comparable-round"} 100'
        in (tmp_path / "step.prom").read_text()
    )


def test_failed_or_async_submission_is_not_published():
    calls = []

    class Hook:
        def observe(self, *args):
            calls.append(args)

    timer = CompletedStepTimer(
        hook=Hook(),
        synchronize=lambda: None,
        tokens_per_step=1,
        global_max=lambda x: x,
        emit=lambda x: None,
    )

    def fail():
        raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError):
        timer.measure(1, fail)
    assert not calls
    with pytest.raises(ValueError):
        timer.measure(2, lambda: None)


def test_thin_trainer_subclass_calls_original_step():
    calls = []

    class Base:
        step = 1

        def train_step(self, data):
            calls.append(data)
            return 42

    class Timer:
        def measure(self, step, operation):
            assert step == 1
            return operation()

    trainer = instrument_trainer(Base)()
    trainer.crucible_timer = Timer()
    assert trainer.train_step("batch") == 42
    assert calls == ["batch"]


def test_fp8_results_require_heldout_delta_and_alternation():
    records = [
        {
            "round": n,
            "variant": v,
            "valid": True,
            "fp8": v == "candidate",
            "conditions": {"gpus": 8},
            "metrics": {"tokens_per_s_per_gpu": 100, "mfu": 0.2, "scaling_efficiency": 1},
        }
        for n in range(1, 4)
        for v in ("baseline", "candidate")
    ]
    with pytest.raises(ValueError, match="accuracy_delta"):
        summarize_results(records)
    for record in records:
        record["metrics"]["accuracy_delta"] = -0.01 if record["fp8"] else 0
    assert summarize_results(records)["rounds"] == 3
    records[1]["variant"] = "baseline"
    with pytest.raises(ValueError):
        summarize_results(records)


def test_scaling_metrics_use_global_tokens_and_same_precision_reference():
    from scaling.analyze_nsys import ModelStructure
    from scaling.experiment import scaling_metrics

    result = scaling_metrics(180, ModelStructure(1000, 2, 4, 8), 16, 16, 1e7, 100)
    assert result["tokens_per_s_per_gpu"] == 11.25
    assert result["scaling_efficiency"] == pytest.approx(0.9)
    with pytest.raises(ValueError):
        scaling_metrics(100, ModelStructure(1000, 2, 4, 8), 16, 16, 1e7, 0)


def test_capture_only_steady_steps_51_to_55():
    events = []
    clocks = iter([v for i in range(55) for v in (i * 3, i * 3 + 1)])
    timer = CompletedStepTimer(
        hook=None,
        synchronize=lambda: None,
        tokens_per_step=1,
        global_max=lambda x: x,
        emit=lambda x: None,
        clock=lambda: next(clocks),
        profile=lambda start: events.append((timer.last_step, start)),
    )
    for step in range(1, 56):
        timer.measure(step, lambda: None)
    assert events == [(50, True), (54, False)]


def test_training_guard_fails_before_importing_torch():
    import os

    env = dict(os.environ)
    env.pop("CRUCIBLE_MULTINODE_APPROVED", None)
    result = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-m", "scaling.train"],
        check=False,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "separate user budget authorization required" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


@pytest.mark.parametrize("gpus", [8, 16, 32])
@pytest.mark.parametrize("precision", ["bf16", "fp8"])
def test_tuning_preserves_fsdp_collective_algorithm_defaults(gpus, precision):
    plan = build_plan()
    comparison = next(
        c
        for c in plan["comparisons"]
        if c["purpose"] == "tuning" and c["gpus"] == gpus and c["precision"] == precision
    )
    for run in comparison["runs"]:
        for node in run["nodes"]:
            algorithm_overrides = {
                key: value
                for key, value in node["env"].items()
                if key in {"NCCL_ALGO", "NCCL_PROTO"}
            }
            assert algorithm_overrides == (
                {"NCCL_ALGO": "allreduce:Tree"} if run["variant"] == "candidate" else {}
            )
    assert (
        plan["required_before_execution"]["nccl_min_version_for_collective_algo_filter"] == "2.24"
    )


def test_precision_and_ep_comparisons_keep_default_collective_algorithms():
    for comparison in build_plan()["comparisons"]:
        if comparison["purpose"] == "tuning":
            continue
        for run in comparison["runs"]:
            for node in run["nodes"]:
                assert "NCCL_ALGO" not in node["env"]
                assert "NCCL_PROTO" not in node["env"]
