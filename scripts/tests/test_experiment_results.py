import copy
import json
import math

import pytest

from scripts.experiment_results import summarize


def records():
    return [
        {
            "round": round_id,
            "variant": variant,
            "conditions": {"gpus": 8, "seed": 42, "dataset_sha256": "fixture-only"},
            "valid": True,
            "fp8": False,
            "metrics": {"tokens_s": value},
        }
        for round_id, baseline, candidate in [(1, 10, 14), (2, 12, 13), (3, 11, 15)]
        for variant, value in [("baseline", baseline), ("candidate", candidate)]
    ]


def test_three_alternating_rounds_report_median_and_range():
    result = summarize(records())
    assert result["baseline"]["tokens_s"] == {"median": 11, "min": 10, "max": 12}
    assert result["candidate"]["tokens_s"] == {"median": 14, "min": 13, "max": 15}
    assert result["rounds"] == 3


@pytest.mark.parametrize("mutation", ["few", "grouped", "different_conditions", "invalid"])
def test_rejects_noncomparable_or_invalid_measurements(mutation):
    data = records()
    if mutation == "few":
        data = data[:4]
    elif mutation == "grouped":
        data.sort(key=lambda r: r["variant"])
    elif mutation == "different_conditions":
        data[1]["conditions"]["gpus"] = 16
    else:
        data[2]["valid"] = False
    with pytest.raises(ValueError):
        summarize(data)


def test_fp8_cannot_hide_missing_accuracy_delta():
    data = records()
    for row in data:
        if row["variant"] == "candidate":
            row["fp8"] = True
    with pytest.raises(ValueError, match="accuracy_delta"):
        summarize(data)
    for row in data:
        row["metrics"]["accuracy_delta"] = -0.01 if row["fp8"] else 0
    assert summarize(data)["candidate"]["accuracy_delta"]["median"] == -0.01


@pytest.mark.parametrize("value", [math.nan, math.inf, True, "10"])
def test_rejects_nonfinite_or_nonnumeric_metrics(value):
    data = records()
    data[0]["metrics"]["tokens_s"] = value
    with pytest.raises(ValueError):
        summarize(data)


def test_no_mutation_and_all_round_metric_names_match():
    data = records()
    snapshot = copy.deepcopy(data)
    summarize(data)
    assert data == snapshot
    data[-1]["metrics"]["extra"] = 1
    with pytest.raises(ValueError, match="metric"):
        summarize(data)


def test_round_ids_cannot_be_booleans_and_variant_configuration_stays_fixed():
    data = records()
    data[0]["round"] = True
    with pytest.raises(ValueError, match="round"):
        summarize(data)
    data = records()
    for row in data:
        row["metrics"]["accuracy_delta"] = 0
    data[-1]["fp8"] = True
    with pytest.raises(ValueError, match="configuration"):
        summarize(data)


@pytest.mark.parametrize("data", [{}, [None] * 6, []])
def test_malformed_record_containers_fail_with_a_useful_error(data):
    with pytest.raises(ValueError):
        summarize(data)


def test_even_round_median_does_not_overflow_finite_inputs():
    data = records()
    fourth_round = copy.deepcopy(data[-2:])
    for row in fourth_round:
        row["round"] = 4
    data.extend(fourth_round)
    for row in data:
        row["metrics"]["tokens_s"] = 1e308
    result = summarize(data)
    assert result["baseline"]["tokens_s"]["median"] == 1e308
    json.dumps(result, allow_nan=False)
