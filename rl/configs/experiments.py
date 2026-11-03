"""Load the Task 11 config family: one base plus single-factor ablations.

The base YAML is the source of truth and may be edited (pins, budgets, hashes).
Validation enforces the plan's comparison rules and the resource budget limits only:
every ablation changes exactly its declared factor, all variants share one sample
budget, group sizes divide the batches, and resources stay on one local GPU.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

from scripts.resource_preflight import PROJECT_SHARE_BYTES

BASE_NAME = "qwen25_05b_math_code.yaml"
BASE_KEYS = {
    "id",
    "declared_factor",
    "seed",
    "model",
    "schedule",
    "objective",
    "optimizer",
    "sampler",
    "train",
    "evaluation",
    "datasets",
    "timing",
    "framework",
    "external_baseline_candidate",
    "resources",
    "readiness",
}
# Resource budget: $0, the single shared RTX 4090, <=4 h per GPU job, <=2
# checkpoints, <=16 GB resident services, the project disk share.
RESOURCE_LIMITS = {
    "gpu_count": 1,
    "paid_budget_usd": 0,
    "max_session_seconds": 14400,
    "checkpoint_count_ceiling": 2,
    "resident_services_cap_bytes": 16_000_000_000,
    "project_cap_bytes": PROJECT_SHARE_BYTES,
}
VARIANTS = {
    "sync_k0": ("schedule", {"schedule": {"mode": "sync", "max_staleness": 0}}),
    "async_k2": ("schedule", {"schedule": {"max_staleness": 2}}),
    "group4": ("group_size", {"objective": {"group_size": 4}}),
    "group16": ("group_size", {"objective": {"group_size": 16}}),
    "kl0": ("kl", {"objective": {"kl": 0.0}}),
    "kl004": ("kl", {"objective": {"kl": 0.04}}),
    "clip01": ("clip", {"objective": {"clip": 0.1}}),
    "clip03": ("clip", {"objective": {"clip": 0.3}}),
    "no_std": ("scale_by_std", {"objective": {"scale_by_std": False}}),
    "external_verl_sync": (
        "external_framework_and_sync_schedule",
        {
            "schedule": {"mode": "sync", "max_staleness": 0},
            "framework": {
                "name": "verl",
                "strategy": "fsdp",
                "backend": "hf",
                "revision": "8d9e350ea58c7ad4b50dd14d9dcb50577242c55f",
            },
        },
    ),
}


def _equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _equal(a, b) for a, b in zip(actual, expected, strict=True)
        )
    return actual == expected


def _merge(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read(path):
    if path.is_symlink():
        raise ValueError("symlink config not permitted")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid config file {path.name}: {exc}") from exc


def _integer(value):
    return type(value) is int


def _check(config, identity):
    objective, train = config["objective"], config["train"]
    group = objective["group_size"]
    if not (_integer(config["seed"]) and _integer(group) and group > 1):
        raise ValueError(f"{identity}: seed and group_size must be integers")
    if train["completions_per_prompt"] % group or config["optimizer"]["batch_completions"] % group:
        raise ValueError(f"{identity}: group does not divide fixed exposure and optimizer batch")
    if (config["schedule"]["mode"] == "sync") != (config["schedule"]["max_staleness"] == 0):
        raise ValueError(f"{identity}: incompatible schedule/staleness")
    for key, limit in RESOURCE_LIMITS.items():
        value = config["resources"][key]
        if not _integer(value) or value > limit:
            raise ValueError(f"{identity}: resources.{key}={value!r} exceeds {limit}")


def load_family(config_root=None):
    root = Path(config_root) if config_root is not None else Path(__file__).parent
    expected_files = {BASE_NAME} | {f"ablations/{name}.yaml" for name in VARIANTS}
    if {str(path.relative_to(root)) for path in root.rglob("*.yaml")} != expected_files:
        raise ValueError("incomplete or extra config family")
    base = _read(root / BASE_NAME)
    if not isinstance(base, dict) or set(base) != BASE_KEYS:
        raise ValueError("base config has unknown or missing top-level keys")
    _check(base, base["id"])
    result = [copy.deepcopy(base)]
    for identity, (factor, overrides) in VARIANTS.items():
        declaration = _read(root / "ablations" / f"{identity}.yaml")
        expected = {
            "extends": BASE_NAME,
            "id": identity,
            "declared_factor": factor,
            "overrides": overrides,
        }
        if not _equal(declaration, expected):
            raise ValueError(f"invalid inheritance/factor/overrides for {identity}")
        resolved = _merge(base, overrides)
        resolved.update(id=identity, declared_factor=factor)
        _check(resolved, identity)
        result.append(resolved)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path)
    args = parser.parse_args(argv)
    try:
        family = load_family(args.config_root)
        print(
            json.dumps(
                {
                    "family_count": len(family),
                    "ids": [item["id"] for item in family],
                    "planned_train_completions_per_variant": family[0]["train"][
                        "planned_completions"
                    ],
                    "planned_eval_completions_per_phase": family[0]["evaluation"][
                        "planned_completions_per_phase"
                    ],
                    "runtime_verified": False,
                },
                indent=2,
            )
        )
        return 0
    except ValueError as exc:
        print(f"experiments: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
