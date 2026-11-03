"""同条件交替实验的汇总门禁；校验输入格式，不证明测量数据的真实性。

JSON 输入保留按时间排列的 baseline/candidate 记录，至少三个完整轮次。
conditions 记录比较中保持相同的 GPU 数、数据、seed 等；被比较的唯一变化
写入独立实验配置。valid=False 的作废轮次拒绝汇总，原始日志须单独保留。
FP8 记录的 metrics 必须同时包含留出集 accuracy_delta 和性能指标。
"""

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path


def _median(series):
    ordered = sorted(series)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    # Exact addition avoids overflow while retaining the precision of small values.
    return float((Fraction(ordered[middle - 1]) + Fraction(ordered[middle])) / 2)


def summarize(records: list[dict]) -> dict:
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("records must be a list of objects")
    if len(records) < 6 or len(records) % 2:
        raise ValueError("at least three complete alternating rounds are required")
    first = records[0]
    conditions = first.get("conditions")
    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("nonempty conditions are required")
    if not isinstance(first.get("metrics"), dict):
        raise TypeError("metrics must be an object")
    metric_names = set(first["metrics"])
    if not metric_names or metric_names == {"accuracy_delta"}:
        raise ValueError("performance metrics are required")
    values = {variant: {name: [] for name in metric_names} for variant in ("baseline", "candidate")}
    fp8_configuration = {}
    for index, record in enumerate(records):
        expected = ("baseline", "candidate")[index % 2]
        round_id = index // 2 + 1
        if (
            record.get("variant") != expected
            or type(record.get("round")) is not int
            or record["round"] != round_id
        ):
            raise ValueError(f"expected round {round_id} {expected} in chronological order")
        if record.get("valid") is not True:
            raise ValueError(f"round {round_id} is invalid; keep its raw logs, do not summarize")
        if record.get("conditions") != conditions:
            raise ValueError("all baseline/candidate rounds must have identical conditions")
        metrics = record.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != metric_names:
            raise ValueError("all rounds must report the same metric names")
        if type(record.get("fp8")) is not bool:
            raise ValueError("fp8 must be explicitly true or false")
        if fp8_configuration.setdefault(expected, record["fp8"]) != record["fp8"]:
            raise ValueError("each variant must retain the same FP8 configuration in every round")
        if record["fp8"] and "accuracy_delta" not in metrics:
            raise ValueError("FP8 performance must include heldout accuracy_delta")
        for name, value in metrics.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"metric {name} must be a finite number")
            values[expected][name].append(value)
    result = {"rounds": len(records) // 2, "conditions": dict(conditions)}
    for variant, metrics in values.items():
        result[variant] = {
            name: {
                "median": _median(series),
                "min": min(series),
                "max": max(series),
            }
            for name, series in sorted(metrics.items())
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path, help="chronologically ordered JSON records")
    args = parser.parse_args()
    try:
        result = summarize(json.loads(args.records.read_text()))
    except (OSError, ValueError, TypeError, AttributeError) as error:
        parser.exit(1, f"invalid experiment: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
