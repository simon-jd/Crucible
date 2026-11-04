import json

import pytest

from rl.local.compare_layouts import compare


def _run(path, seed, before, after, pipeline=None):
    path.mkdir(parents=True)
    summary = {
        "args": {"seed": str(seed), "eval_n": "4"},
        "before": {"g": {"pass_at_1": 0.5, "per_problem_correct": before}},
        "after": {"g": {"pass_at_1": 0.6, "truncated": 1, "per_problem_correct": after}},
        "peak_cuda_allocated_bytes": 7,
    }
    if pipeline:
        summary["pipeline"] = pipeline
    (path / "summary.json").write_text(json.dumps(summary))
    steps = [{"wall_s": 10.0, "rollout_s": 4.0, "learn_s": 4.0, "sync_s": 0.0, "reward_s": 1.0, "trained_version": max(0, t - 1),
              "policy_version": t + 1, "accepted_samples": 64, "dropped_stale": 0} for t in range(3)]  # fmt: skip
    (path / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    return path


def test_pooled_and_between_layout_paired_differences(tmp_path):
    before = {"p1": 2, "p2": 0, "p3": 4}
    a = [_run(tmp_path / f"a{s}", s, before, {"p1": 3, "p2": 1, "p3": 4}) for s in (17, 18)]
    b = [_run(tmp_path / f"b{s}", s, before, {"p1": 4, "p2": 1, "p3": 4}) for s in (17, 18)]
    report = compare({"a": a, "b": b})
    ev = report["layouts"]["a"]["eval"]["g"]
    assert ev["delta"] == pytest.approx((1 / 4 + 1 / 4 + 0) / 3)
    assert report["pairs"]["b-a"]["g"]["delta"] == pytest.approx((1 / 4) / 3)
    costs = report["layouts"]["a"]["per_seed_costs"][0]
    assert costs["gpu_idle_share"] == pytest.approx(0.2) and costs[
        "trained_samples_per_s"
    ] == pytest.approx(6.4)
    assert costs["staleness_histogram"] == {"0": 64, "1": 128}


def test_seed_lists_must_match(tmp_path):
    a = [_run(tmp_path / "a", 17, {"p": 1}, {"p": 2})]
    b = [_run(tmp_path / "b", 18, {"p": 1}, {"p": 2})]
    with pytest.raises(ValueError, match="seed"):
        compare({"a": a, "b": b})
