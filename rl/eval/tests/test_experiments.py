"""Exercise the concrete config family: one factor per ablation, equal budgets."""

from pathlib import Path

import pytest
import yaml

from rl.configs.experiments import load_family

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def copy_configs(tmp_path):
    target = tmp_path / "configs"
    (target / "ablations").mkdir(parents=True)
    for file in CONFIGS.rglob("*.yaml"):
        (target / file.relative_to(CONFIGS)).write_text(file.read_text())
    return target


def test_actual_config_family_complete_and_equal_budget():
    family = load_family(CONFIGS)
    assert len(family) == 11
    assert {item["schedule"]["max_staleness"] for item in family} == {0, 1, 2}
    assert {item["objective"]["group_size"] for item in family} == {4, 8, 16}
    assert {item["objective"]["kl"] for item in family} == {0.0, 0.001, 0.04}
    assert {item["objective"]["clip"] for item in family} == {0.1, 0.2, 0.3}
    assert all(item["train"]["planned_completions"] == 2048 for item in family)
    assert len({item["evaluation"]["planned_completions_per_phase"] for item in family}) == 1
    assert family[-1]["framework"]["runtime_verified"] is False


@pytest.mark.parametrize(
    "kind",
    [
        "unknown",
        "budget",
        "group",
        "stale",
        "runtime",
        "paid",
        "missing",
        "extends",
        "second_factor",
        "bool",
    ],
)
def test_actual_family_rejects_invalid_drift(tmp_path, kind):
    target = copy_configs(tmp_path)
    base_path = target / "qwen25_05b_math_code.yaml"
    base = yaml.safe_load(base_path.read_text())
    if kind == "unknown":
        base["unknown"] = 1
    elif kind == "group":
        base["objective"]["group_size"] = 7
    elif kind == "stale":
        base["schedule"]["max_staleness"] = 0
    elif kind == "runtime":
        base["resources"]["gpu_count"] = 2
    elif kind == "paid":
        base["resources"]["paid_budget_usd"] = 1
    elif kind == "bool":
        base["seed"] = True
    else:
        file = target / "ablations" / "sync_k0.yaml"
        if kind == "missing":
            file.unlink()
        else:
            variant = yaml.safe_load(file.read_text())
            if kind == "extends":
                variant["extends"] = "../other.yaml"
            elif kind == "budget":
                variant["overrides"]["train"] = {"planned_completions": 1024}
            else:
                variant["overrides"]["objective"] = {"clip": 0.1}
            file.write_text(yaml.safe_dump(variant))
    base_path.write_text(yaml.safe_dump(base))
    with pytest.raises(ValueError):
        load_family(target)


def test_base_config_is_editable_and_propagates(tmp_path):
    target = copy_configs(tmp_path)
    base_path = target / "qwen25_05b_math_code.yaml"
    base = yaml.safe_load(base_path.read_text())
    base["model"]["revision"] = "new-pinned-revision"
    base["train"]["planned_completions"] = 1024
    base_path.write_text(yaml.safe_dump(base))
    family = load_family(target)
    assert {item["model"]["revision"] for item in family} == {"new-pinned-revision"}
    assert {item["train"]["planned_completions"] for item in family} == {1024}
