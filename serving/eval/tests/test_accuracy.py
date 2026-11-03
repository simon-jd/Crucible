"""Synthetic fixtures only; these are not model measurements."""

import copy
import hashlib
import json

import pytest


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fixture(rounds=3):
    conditions = {
        "planned_rounds": rounds,
        "run_plan": [
            {"id": f"{rnd}-{variant}", "round": rnd, "variant": variant}
            for rnd in range(1, rounds + 1)
            for variant in (("baseline", "candidate") if rnd % 2 else ("candidate", "baseline"))
        ],
        "model": {"name": "fixture/model", "revision": "a" * 40, "sha256": "a" * 64},
        "source_sha256": "b" * 64,
        "runtime_sha256": "c" * 64,
        "protocol_sha256": "d" * 64,
        "gpu_count": 1,
        "sampling": {"temperature": 0, "seed": 17, "max_new_tokens": 16},
        "slo": {
            "calibration_precision": "bf16",
            "calibration_sha256": "e" * 64,
            "ttft_ms": 100,
            "tpot_ms": 50,
        },
        "workload": [
            {
                "id": "p1",
                "prompt_sha256": "f" * 64,
                "scheduled_s": 0,
                "input_tokens": 10,
                "requested_tokens": 2,
            }
        ],
        "heldout": {
            "dataset": "fixture-gsm8k",
            "subset": "heldout-2",
            "revision": "1" * 40,
            "normalization": "numeric-final-v1",
            "examples": [
                {"id": "q1", "prompt_sha256": "2" * 64, "label": "42"},
                {"id": "q2", "prompt_sha256": "3" * 64, "label": "2"},
            ],
        },
    }
    configs = {
        v: {
            "precision": p,
            "backend": "fixture",
            "settings": {"layout": "colocated"},
            "model_sha256": conditions["model"]["sha256"],
            "runtime_sha256": conditions["runtime_sha256"],
        }
        for v, p in [("baseline", "bf16"), ("candidate", "fp8")]
    }
    runs = []
    for rnd in range(1, rounds + 1):
        for v in ["baseline", "candidate"] if rnd % 2 else ["candidate", "baseline"]:
            runs.append(
                {
                    "id": f"{rnd}-{v}",
                    "round": rnd,
                    "variant": v,
                    "started_at_ns": len(runs) * 1000000000,
                    "finished_at_ns": len(runs) * 1000000000 + 101000000,
                    "observation_end_s": 0.101,
                    "config_sha256": digest(configs[v]),
                    "conditions_sha256": digest(conditions),
                    "aborted": False,
                    "requests": [
                        {
                            "id": "p1",
                            "status": "success",
                            "send_s": 0.001,
                            "end_s": 0.101,
                            "ttft_ms": 20,
                            "tpot_ms": 40,
                            "output_tokens": 2,
                            "error": None,
                            "kv_transfer_ms": None,
                        }
                    ],
                }
            )
    quality = {}
    for v, config in configs.items():
        quality[v] = {
            "config_sha256": digest(config),
            "conditions_sha256": digest(conditions),
            "items": [
                {
                    "id": "q1",
                    "status": "success",
                    "answer": "#### 42",
                    "prompt_sha256": "2" * 64,
                    "label": "42",
                    "error": None,
                },
                {
                    "id": "q2",
                    "status": "success",
                    "prompt_sha256": "3" * 64,
                    "label": "2",
                    "answer": "2" if v == "baseline" else "3",
                    "error": None,
                },
            ],
        }
    return {
        "schema": 2,
        "kind": "fixture",
        "conditions": conditions,
        "configurations": configs,
        "runs": runs,
        "quality": quality,
    }


def test_accuracy_delta_reported():
    from serving.eval.accuracy import report

    result = report(fixture())
    assert result["throughput"]["candidate"]["median"] > 0
    assert result["accuracy_delta"] == -0.5
    assert result["absolute_accuracy_difference"] == 0.5
    assert result["benchmark_accepted"] is False


def test_failures_stay_in_heldout_denominator():
    from serving.eval.accuracy import report

    bundle = fixture()
    q = bundle["quality"]["candidate"]["items"][0]
    q.update(status="error", answer=None, error="fixture timeout")
    result = report(bundle)
    assert result["quality"]["candidate"]["accuracy"] == 0
    assert result["quality"]["candidate"]["counts"]["source"] == 2
    assert result["quality"]["candidate"]["counts"]["error"] == 1
    assert result["paired_outcomes"][0]["candidate"]["error"] == "fixture timeout"


def test_sender_failure_invalidates_whole_comparison_without_dropping_runs():
    from serving.eval.accuracy import report

    bundle = fixture()
    r = bundle["runs"][1]["requests"][0]
    r.update(
        status="missing",
        send_s=None,
        end_s=None,
        output_tokens=0,
        ttft_ms=None,
        tpot_ms=None,
        error="fixture unsent",
    )
    result = report(bundle)
    assert result["throughput"] is None
    assert result["performance_valid"] is False
    assert len(result["raw_runs"]) == 6
    assert result["runs"][1]["counts"]["source"] == 1
    assert result["runs"][1]["counts"]["sender_invalid"] == 1
    assert result["accuracy_delta"] == -0.5


def test_errors_truncations_and_undefined_timings_retained():
    from serving.eval.accuracy import report

    bundle = fixture()
    bundle["runs"][0]["requests"][0].update(
        status="error", output_tokens=0, ttft_ms=None, tpot_ms=None, error="fixture HTTP500"
    )
    bundle["quality"]["candidate"]["items"][0].update(status="truncated", answer="42")
    result = report(bundle)
    assert result["runs"][0]["counts"]["error"] == 1
    assert result["runs"][0]["counts"]["undefined_tpot"] == 1
    assert result["runs"][0]["throughput_tokens_s"] == 0
    assert result["quality"]["candidate"]["counts"]["truncated"] == 1
    assert result["quality"]["candidate"]["accuracy"] == 0


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("#### 1,234.00", "1234"),
        (" -0.50 ", "-.5"),
        ("work\n#### +42", "42"),
        ("42 apples", None),
        ("1,23", None),
        ("NaN", None),
        ("1e2", None),
    ],
)
def test_numeric_normalization(answer, expected):
    from fractions import Fraction

    from serving.eval.accuracy import normalize

    assert normalize(answer, "numeric-final-v1") == (Fraction(expected) if expected else None)


def test_choice_normalization():
    from serving.eval.accuracy import normalize

    assert normalize(" j ", "choice-letter-v1") == "J"
    assert normalize("Answer: A", "choice-letter-v1") is None
    assert normalize("K", "choice-letter-v1") is None


def set_at(bundle, path, value):
    target = bundle
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.parametrize(
    "path,value",
    [
        (["runs", 0, "requests", 0, "output_tokens"], -1),
        (["runs", 0, "requests", 0, "output_tokens"], 3),
        (["runs", 0, "requests", 0, "end_s"], -1),
        (["runs", 0, "requests", 0, "tpot_ms"], float("inf")),
        (["runs", 0, "requests", 0, "tpot_ms"], None),
        (["runs", 0, "requests", 0, "ttft_ms"], 999),
        (["runs", 0, "requests", 0, "kv_transfer_ms"], 1),
        (["runs", 0, "requests", 0, "send_s"], -1),
        (["runs", 0, "aborted"], 0),
        (["runs", 1, "started_at_ns"], 0),
        (["runs", 0, "finished_at_ns"], 1),
        (["runs", 0, "conditions_sha256"], "0" * 64),
        (["runs", 0, "config_sha256"], "0" * 64),
        (["quality", "candidate", "config_sha256"], "0" * 64),
        (["quality", "candidate", "conditions_sha256"], "0" * 64),
        (["quality", "candidate", "items", 0, "id"], "q3"),
        (["quality", "candidate", "items", 0, "label"], "99"),
        (["quality", "candidate", "items", 0, "prompt_sha256"], "0" * 64),
        (["quality", "candidate", "items", 0, "status"], "unknown"),
        (["conditions", "slo", "calibration_precision"], "fp8"),
        (["conditions", "model", "sha256"], "not-a-hash"),
        (["conditions", "heldout", "examples", 1, "id"], "q1"),
        (["conditions", "heldout", "examples", 0, "label"], "unknown"),
        (["configurations", "candidate", "model_sha256"], "0" * 64),
        (["configurations", "candidate", "settings", "layout"], "pd"),
        (["configurations", "candidate", "precision"], "bf16"),
    ],
)
def test_reject_impossible_or_mismatched_evidence(path, value):
    from serving.eval.accuracy import report

    bundle = fixture()
    set_at(bundle, path, value)
    with pytest.raises(ValueError):
        report(bundle)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_run",
        "duplicate_run",
        "wrong_order",
        "missing_quality",
        "duplicate_quality",
        "missing_metric",
    ],
)
def test_reject_missing_or_duplicate_sources(mutation):
    from serving.eval.accuracy import report

    b = fixture()
    if mutation == "missing_run":
        b["runs"].pop()
    elif mutation == "duplicate_run":
        b["runs"][1]["id"] = b["runs"][0]["id"]
    elif mutation == "wrong_order":
        b["runs"][2:4] = b["runs"][2:4][::-1]
    elif mutation == "missing_quality":
        b["quality"]["candidate"]["items"].pop()
    elif mutation == "duplicate_quality":
        b["quality"]["candidate"]["items"][1] = copy.deepcopy(b["quality"]["candidate"]["items"][0])
    else:
        del b["runs"][0]["requests"][0]["ttft_ms"]
    with pytest.raises((ValueError, KeyError)):
        report(b)


def rebind(b):
    for r in b["runs"]:
        r["conditions_sha256"] = digest(b["conditions"])
        r["config_sha256"] = digest(b["configurations"][r["variant"]])
    for v, q in b["quality"].items():
        q["conditions_sha256"] = digest(b["conditions"])
        q["config_sha256"] = digest(b["configurations"][v])


def test_pd_requires_per_request_measured_kv():
    from serving.eval.accuracy import report

    b = fixture()
    for c in b["configurations"].values():
        c["settings"]["layout"] = "pd"
    rebind(b)
    with pytest.raises(ValueError, match="KV transfer"):
        report(b)
    for r in b["runs"]:
        r["requests"][0]["kv_transfer_ms"] = 2
    assert report(b)["runs"][0]["kv_transfer_ms"]["median"] == 2


def test_actual_cli_roundtrip_and_fixture_rejection():
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(".cache/task13-cpu")
    root.mkdir(exist_ok=True)
    source = root / "cli-fixture.json"
    source.write_text(json.dumps(fixture()))
    command = [sys.executable, "-B", "-m", "serving.eval.accuracy", str(source)]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    rejected = subprocess.run(command, text=True, capture_output=True, env=env, check=False)
    assert rejected.returncode == 1 and "fixture requires" in rejected.stderr
    accepted = subprocess.run(
        command + ["--allow-fixture"], text=True, capture_output=True, env=env, check=False
    )
    assert accepted.returncode == 0, accepted.stderr
    result = json.loads(accepted.stdout)
    assert result["raw_evidence"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["benchmark_accepted"] is False
    assert result["throughput"] and result["absolute_accuracy_difference"] == 0.5
    source.write_text('{"schema":1,"schema":2}')
    rejected = subprocess.run(command, text=True, capture_output=True, env=env, check=False)
    assert rejected.returncode == 1 and "duplicate JSON key" in rejected.stderr
    source.unlink()


@pytest.mark.parametrize("missing,valid", [(1, True), (2, False)])
def test_sender_one_percent_boundary(missing, valid):
    from serving.eval.accuracy import report

    b = fixture()
    planned = b["conditions"]["workload"][0]
    b["conditions"]["workload"] = [{**planned, "id": f"p{i}"} for i in range(100)]
    for run in b["runs"]:
        original = run["requests"][0]
        run["requests"] = [{**original, "id": f"p{i}"} for i in range(100)]
    for req in b["runs"][0]["requests"][:missing]:
        req.update(
            status="missing",
            send_s=None,
            end_s=None,
            output_tokens=0,
            ttft_ms=None,
            tpot_ms=None,
            error="fixture unsent",
        )
    rebind(b)
    result = report(b)
    assert result["performance_valid"] is valid
    assert result["runs"][0]["counts"]["source"] == 100
    assert result["runs"][0]["counts"]["missing"] == missing


@pytest.mark.parametrize("send,valid", [(0.005, True), (0.00501, False)])
def test_sender_five_ms_boundary(send, valid):
    from serving.eval.accuracy import report

    b = fixture()
    b["runs"][0]["requests"][0]["send_s"] = send
    assert report(b)["performance_valid"] is valid


def test_aborted_round_retained():
    from serving.eval.accuracy import report

    b = fixture()
    b["runs"][4]["aborted"] = True
    result = report(b)
    assert not result["performance_valid"]
    assert result["throughput"] is None
    assert result["raw_runs"][4]["aborted"] is True


def test_medians_and_units_use_all_valid_counterbalanced_pairs():
    from serving.eval.accuracy import report

    result = report(fixture())
    assert result["rounds"] == 3
    assert [x["variant"] for x in result["raw_runs"]] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    ]
    assert result["throughput"]["baseline"] == dict.fromkeys(["median", "min", "max"], 2 / 0.101)
    assert result["slo_goodput"]["baseline"]["median"] == 1 / 0.101


def test_extra_raw_correct_field_cannot_shadow_computed_result():
    from serving.eval.accuracy import report

    b = fixture()
    b["quality"]["candidate"]["items"][1]["correct"] = True
    result = report(b)
    assert result["paired_outcomes"][1]["candidate"]["correct"] is False
    assert result["quality"]["candidate"]["counts"]["correct"] == 1


def test_config_cannot_hide_non_precision_change_outside_settings():
    from serving.eval.accuracy import report

    b = fixture()
    b["configurations"]["candidate"]["temperature"] = 2
    rebind(b)
    with pytest.raises(ValueError, match="all other options in settings"):
        report(b)


def test_failed_request_impossible_timings_rejected():
    from serving.eval.accuracy import report

    b = fixture()
    b["runs"][0]["requests"][0].update(status="error", error="fixture error", ttft_ms=1000)
    with pytest.raises(ValueError, match="exceeds request duration"):
        report(b)


def test_nonfinite_derived_throughput_rejected():
    from serving.eval.accuracy import report

    b = fixture()
    b["runs"][0]["requests"][0].update(send_s=0, end_s=1e-320, ttft_ms=0, tpot_ms=0)
    b["runs"][0]["observation_end_s"] = 1e-320
    with pytest.raises(ValueError, match="derived performance metric"):
        report(b)


def test_planned_four_rounds_missing_trailing_pair_rejected():
    from serving.eval.accuracy import report

    b = fixture(rounds=4)
    b["runs"] = b["runs"][:-2]
    with pytest.raises(ValueError, match="run plan"):
        report(b)


def trailing_missing_fixture(scheduled_s=100, observation_end_s=0.101, aborted=False):
    b = fixture()
    planned = b["conditions"]["workload"][0]
    b["conditions"]["workload"] = [
        {**planned, "id": f"p{i}", "scheduled_s": 0 if i < 99 else scheduled_s} for i in range(100)
    ]
    for index, run in enumerate(b["runs"]):
        original = run["requests"][0]
        run["requests"] = [{**original, "id": f"p{i}"} for i in range(100)]
        run["requests"][-1].update(
            status="missing",
            send_s=None,
            end_s=None,
            output_tokens=0,
            ttft_ms=None,
            tpot_ms=None,
            error="fixture unsent",
        )
        run["observation_end_s"] = observation_end_s
        run["started_at_ns"] = index * 200_000_000_000
        run["finished_at_ns"] = run["started_at_ns"] + round(observation_end_s * 1e9)
        run["aborted"] = aborted
    rebind(b)
    return b


def test_missing_trailing_schedule_outside_observation_rejected():
    from serving.eval.accuracy import report

    with pytest.raises(ValueError, match="observation.*schedule"):
        report(trailing_missing_fixture())


def test_complete_four_round_plan_retained_and_hashed():
    from serving.eval.accuracy import report

    b = fixture(rounds=4)
    result = report(b)
    assert result["rounds"] == 4
    assert len(result["raw_runs"]) == 8
    assert result["conditions"]["run_plan"] == b["conditions"]["run_plan"]
    assert result["identities"]["run_plan_sha256"] == digest(b["conditions"]["run_plan"])
    assert result["performance_valid"] is True


@pytest.mark.parametrize("change", ["missing_member", "extra_pair", "relabel", "reorder"])
def test_run_plan_requires_exact_member_coverage(change):
    from serving.eval.accuracy import report

    b = fixture(rounds=4)
    if change == "missing_member":
        b["runs"].pop()
    elif change == "extra_pair":
        b["runs"].extend(copy.deepcopy(b["runs"][-2:]))
    elif change == "relabel":
        b["runs"][-1]["id"] = "substitute-attempt"
    else:
        b["runs"][-2:] = b["runs"][-2:][::-1]
    with pytest.raises(ValueError, match="run plan"):
        report(b)


@pytest.mark.parametrize(
    "change",
    [
        "missing_plan",
        "missing_count",
        "conflicting_count",
        "duplicate_id",
        "wrong_variant",
        "boolean_round",
    ],
)
def test_run_plan_cannot_be_inferred_or_contradict_declared_count(change):
    from serving.eval.accuracy import report

    b = fixture(rounds=4)
    c = b["conditions"]
    if change == "missing_plan":
        del c["run_plan"]
    elif change == "missing_count":
        del c["planned_rounds"]
    elif change == "conflicting_count":
        c["planned_rounds"] = 5
    elif change == "duplicate_id":
        c["run_plan"][-1]["id"] = c["run_plan"][0]["id"]
    elif change == "wrong_variant":
        c["run_plan"][-1]["variant"] = "candidate"
    else:
        c["run_plan"][0]["round"] = True
    rebind(b)
    with pytest.raises((ValueError, KeyError)):
        report(b)


def test_changing_plan_requires_condition_identity_rebinding():
    from serving.eval.accuracy import report

    b = fixture(rounds=4)
    b["conditions"]["run_plan"][-1]["id"] = "replacement-baseline"
    b["runs"][-1]["id"] = "replacement-baseline"
    with pytest.raises(ValueError, match="conditions mismatch"):
        report(b)


def test_final_planned_pair_aborted_keeps_all_rounds_and_suppresses_metrics():
    from serving.eval.accuracy import report

    b = fixture(rounds=4)
    for run in b["runs"][-2:]:
        run["aborted"] = True
    result = report(b)
    assert result["rounds"] == 4
    assert len(result["raw_runs"]) == 8
    assert result["raw_runs"][-1]["aborted"] is True
    assert result["performance_valid"] is False
    assert result["throughput"] is None and result["slo_goodput"] is None


@pytest.mark.parametrize("schedule,bound", [(0.05, 0.2), (100, 100)])
def test_tolerated_missing_terminal_bound_extends_denominator(schedule, bound):
    from serving.eval.accuracy import report

    result = report(trailing_missing_fixture(schedule, bound))
    assert result["performance_valid"] is True
    assert result["runs"][0]["duration_s"] == bound
    assert result["runs"][0]["last_completion_s"] == 0.101
    assert result["runs"][0]["missing_terminal_observation_s"] == bound
    assert result["runs"][0]["counts"]["source"] == 100
    assert result["runs"][0]["counts"]["missing"] == 1
    assert result["throughput"]["baseline"]["median"] == 198 / bound
    assert result["slo_goodput"]["baseline"]["median"] == 99 / bound


def test_aborted_uncovered_schedule_retains_raw_and_suppresses_aggregates():
    from serving.eval.accuracy import report

    b = trailing_missing_fixture(aborted=True)
    result = report(b)
    assert result["throughput"] is None and result["slo_goodput"] is None
    assert result["performance_valid"] is False
    assert result["runs"][0]["schedule_covered"] is False
    assert result["runs"][0]["missing_terminal_observation_s"] == 0.101
    assert result["conditions"]["workload"][-1]["scheduled_s"] == 100
    assert result["raw_runs"] == b["runs"]


@pytest.mark.parametrize("bound", [None, -1, float("nan"), float("inf"), True])
def test_observation_bound_must_be_finite_nonnegative_number(bound):
    from serving.eval.accuracy import report

    b = fixture()
    b["runs"][0]["observation_end_s"] = bound
    with pytest.raises(ValueError):
        report(b)


@pytest.mark.parametrize("aborted", [False, True])
def test_observation_cannot_exceed_clock_even_for_aborted_runs(aborted):
    from serving.eval.accuracy import report

    b = trailing_missing_fixture(0.05, 0.2, aborted)
    b["runs"][0]["finished_at_ns"] = b["runs"][0]["started_at_ns"] + 199_999_999
    with pytest.raises(ValueError, match="clock bracket"):
        report(b)


def test_observation_cannot_precede_actual_completion():
    from serving.eval.accuracy import report

    b = trailing_missing_fixture(0.05, 0.1)
    with pytest.raises(ValueError, match="before an observed completion"):
        report(b)


def test_without_missing_rows_observation_is_final_completion_not_cleanup():
    from serving.eval.accuracy import report

    b = fixture()
    b["runs"][0]["observation_end_s"] = 0.2
    b["runs"][0]["finished_at_ns"] = 200_000_000
    with pytest.raises(ValueError, match="final completion"):
        report(b)


def test_schema_one_rejected_without_inferred_plan_or_bound():
    from pathlib import Path

    from serving.eval.accuracy import report

    original = json.loads(Path("docs/evidence/raw/task13-fixture-input.json").read_text())
    with pytest.raises(ValueError, match="schema 2 required"):
        report(original)


def prequantized(**changes):
    """A candidate served from a separately published quantized checkpoint."""
    b = fixture()
    b["conditions"]["model"]["tokenizer_sha256"] = "7" * 64
    b["conditions"]["candidate_model"] = {
        "name": "fixture/model-FP8", "revision": "8" * 40, "sha256": "9" * 64,
        "base_model": "fixture/model", "tokenizer_sha256": "7" * 64, **changes,
    }  # fmt: skip
    b["configurations"]["candidate"]["model_sha256"] = "9" * 64
    rebind(b)
    return b


def test_declared_prequantized_candidate_checkpoint_is_accepted():
    from serving.eval.accuracy import report

    result = report(prequantized())
    assert result["performance_valid"] and result["candidate_checkpoint"] == "fixture/model-FP8"
    assert report(fixture())["candidate_checkpoint"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"base_model": "other/model"},  # not derived from the baseline model
        {"tokenizer_sha256": "6" * 64},  # different tokenizer: different prompts
        {"sha256": "a" * 64},  # same artifact as the baseline: use the plain path
        {"sha256": "not-a-hash"},
        {"revision": ""},
    ],
)
def test_prequantized_candidate_lineage_is_checked(changes):
    from serving.eval.accuracy import report

    with pytest.raises(ValueError):
        report(prequantized(**changes))


def test_prequantized_candidate_config_must_name_the_declared_checkpoint():
    from serving.eval.accuracy import report

    b = prequantized()
    b["configurations"]["candidate"]["model_sha256"] = "5" * 64
    rebind(b)
    with pytest.raises(ValueError, match="lineage"):
        report(b)
    b = prequantized()
    b["configurations"]["baseline"]["model_sha256"] = "9" * 64
    rebind(b)
    with pytest.raises(ValueError, match="lineage"):
        report(b)
    b = prequantized()
    del b["conditions"]["model"]["tokenizer_sha256"]
    rebind(b)
    with pytest.raises((ValueError, KeyError)):
        report(b)
