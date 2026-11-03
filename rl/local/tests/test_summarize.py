import pytest

from rl.local import summarize as sm


def test_paired_delta_uses_per_problem_differences():
    before = {"a": 0, "b": 2, "c": 4}
    after = {"a": 2, "b": 2, "c": 4}
    d = sm.paired_delta(before, after, n=4)
    assert d["delta"] == pytest.approx(0.5 / 3)
    assert d["stderr"] == pytest.approx(((0.5 - 0.5 / 3) ** 2 + 2 * (0.5 / 3) ** 2) ** 0.5 / (2**0.5) / 3**0.5)
    assert d["improved"] == 1 and d["worse"] == 0


def test_step_stats_median_and_range():
    steps = [{"wall_s": w, "sync_s": 0.01, "gpu_idle_share": 0.1, "rollout_s": 1, "learn_s": 2, "dropped_stale": 0, "reward": r,
              "flags": {"truncated": t} if t else {}}
             for w, r, t in ((5, 0.2, 3), (7, 0.4, 0), (6, 0.6, 6))]  # fmt: skip
    s = sm.step_stats(steps)
    assert s["wall_s"] == {"median": 6, "min": 5, "max": 7}
    assert s["truncated_first10"] == pytest.approx(3)
    assert s["dropped_stale"] == 0 and s["reward_first10"] == pytest.approx(0.4)


def test_pooled_delta_averages_seeds_per_problem_before_the_stderr():
    runs = [({"a": 0, "b": 4}, {"a": 4, "b": 4}), ({"a": 0, "b": 4}, {"a": 0, "b": 0})]
    d = sm.pooled_delta(runs, n=4)
    # per problem: a -> mean(+1, 0) = 0.5, b -> mean(0, -1) = -0.5
    assert d["delta"] == pytest.approx(0.0)
    assert d["stderr"] == pytest.approx(0.5)
    assert d["seeds"] == 2 and d["per_seed"] == pytest.approx([0.5, -0.5])


def test_pooled_delta_rejects_mismatched_problem_sets():
    with pytest.raises(ValueError):
        sm.pooled_delta([({"a": 0}, {"a": 1}), ({"b": 0}, {"b": 1})], n=4)


def _write_run(path, seed, before, after, clip="0.2"):
    import json

    path.mkdir()
    def ev(pp):
        return {"pass_at_1": sum(pp.values()) / (4 * len(pp)), "truncated": 0, "per_problem_correct": pp}

    summary = {"args": {"staleness": "1", "kl": "0.001", "group": "8", "clip": clip, "reward": "eos", "max_new_tokens": "384",
                        "eval_n": "4", "seed": str(seed)},
               "before": {"gsm8k": ev(before)}, "after": {"gsm8k": ev(after)}}  # fmt: skip
    (path / "summary.json").write_text(json.dumps(summary))


def test_pool_runs_groups_by_config_ignoring_seed(tmp_path):
    _write_run(tmp_path / "s17", 17, {"a": 0, "b": 4}, {"a": 4, "b": 4})
    _write_run(tmp_path / "s18", 18, {"a": 0, "b": 4}, {"a": 0, "b": 0})
    _write_run(tmp_path / "c01", 17, {"a": 0, "b": 4}, {"a": 4, "b": 4}, clip="0.1")
    pooled = sm.pool_runs([tmp_path / "s17", tmp_path / "s18", tmp_path / "c01"])
    by_runs = {tuple(p["runs"]): p for p in pooled}
    assert set(by_runs) == {("s17", "s18"), ("c01",)}
    g = by_runs[("s17", "s18")]["eval"]["gsm8k"]
    assert g["seeds"] == 2 and g["delta"] == pytest.approx(0.0) and g["after_mean"] == pytest.approx(0.5)
