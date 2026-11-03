from pathlib import Path

import pytest

from cluster.health.parse_nccl import evaluate, parse_nccl

SAMPLE = (Path(__file__).parent / "fixtures/allreduce_upstream.txt").read_text()
THRESHOLDS = {
    "bandwidth_unit": "GB/s",
    "time_unit": "us",
    "theoretical_busbw_gbs": 200,
    "min_busbw_fraction": 0.8,
    "min_size_bytes": 536870912,
}


def test_parse_allreduce_sample():
    rows = parse_nccl(SAMPLE)
    assert len(rows) == 12
    assert rows[0].size_bytes == 536870912
    assert rows[0].time_us == 5919.20
    assert rows[0].algbw_gbs == 90.70
    assert rows[0].busbw_gbs == 170.06
    assert rows[0].mode == "out-of-place"
    assert rows[1].mode == "in-place"
    assert not evaluate(rows, THRESHOLDS)


def test_threshold_flags_slow_pair():
    # Synthetic slowdown derived from real output, never an observed device result.
    rows = parse_nccl(SAMPLE.replace("170.06", "100.06"))
    violations = evaluate(rows, THRESHOLDS)
    assert any(v.code == "slow_bandwidth" for v in violations)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        SAMPLE.replace("(us)", "(ms)"),
        SAMPLE.replace("(GB/s)", "(GiB/s)"),
        SAMPLE.replace("90.70", "nan"),
        SAMPLE.replace("170.06", "-1"),
        SAMPLE.replace("# Out of bounds values : 0 OK", ""),
        SAMPLE.replace("# Out of bounds values : 0 OK", "# Out of bounds values : 1 FAILED"),
        SAMPLE.replace("5919.20   90.70", "5919.20"),
        SAMPLE.replace("172.526", "172.526 FAILED"),
    ],
)
def test_parser_fails_closed(bad):
    with pytest.raises(ValueError):
        parse_nccl(bad)


def test_correctness_checked_in_both_modes():
    bad = SAMPLE.replace("170.06       0", "170.06       1", 1)
    assert any(v.code == "correctness" for v in evaluate(parse_nccl(bad), THRESHOLDS))
    unchecked = SAMPLE.replace("170.06       0", "170.06     N/A", 1)
    assert any(v.code == "unchecked" for v in evaluate(parse_nccl(unchecked), THRESHOLDS))


@pytest.mark.parametrize(
    "patch",
    [
        {"theoretical_busbw_gbs": None},
        {"theoretical_busbw_gbs": 0},
        {"min_busbw_fraction": 80},
        {"min_busbw_fraction": float("nan")},
        {"min_size_bytes": 10**20},
        {"bandwidth_unit": "Gb/s"},
    ],
)
def test_threshold_inputs_fail_closed(patch):
    with pytest.raises(ValueError):
        evaluate(parse_nccl(SAMPLE), THRESHOLDS | patch)


def test_empty_measurements_fail_closed():
    with pytest.raises(ValueError):
        evaluate([], THRESHOLDS)
