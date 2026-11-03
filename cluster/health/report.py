"""Burn-in planner (default) and explicit future approved-site executor.

No subprocess is started without --execute and a complete reviewed site inventory.
"""

import argparse
import hashlib
import json
import re
import signal
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

from cluster.health.ib_perftest import build_pairs, evaluate_ib, run_pair
from cluster.health.nccl_matrix import build_matrix, ssh, validate_inventory
from cluster.health.parse_nccl import evaluate, fraction, parse_nccl, positive
from cluster.health.rank_exec import rank_token
from cluster.health.remote_control import remaining, run_remote, stop_remote
from cluster.health.runner import CommandRunner, Result
from scripts.experiment_results import summarize as summarize_experiment

NOTE = "未在真实多机上验证"
DEFAULT_THRESHOLDS = Path(__file__).with_name("thresholds.yaml")


def load_thresholds(path=DEFAULT_THRESHOLDS):
    config = yaml.safe_load(Path(path).read_text())
    if config.get("schema") != 1:
        raise ValueError("threshold schema 1 required")
    for key in ["nccl", "ib"]:
        if config[key].get("bandwidth_unit") != "GB/s" or config[key].get("time_unit") != "us":
            raise ValueError("threshold units must be GB/s and us")
    fraction(config["nccl"].get("min_busbw_fraction"), "min_busbw_fraction")
    for key in ["min_write_bw_fraction", "min_read_latency_efficiency_fraction"]:
        fraction(config["ib"].get(key), key)
    for key in ["write_size_bytes", "read_size_bytes"]:
        if type(config["ib"].get(key)) is not int or config["ib"][key] <= 0:
            raise ValueError("IB byte sizes must be positive integers")
    if config["nccl"].get("min_size_bytes") != 134217728:
        raise ValueError("NCCL matrix starts at 134217728 bytes; threshold size must match")
    return config


def create_plan(inv, thresholds=None):
    validate_inventory(inv)
    thresholds = thresholds or load_thresholds()
    root = inv["shared_root"] + "/cluster/health/"
    dcgm, affinity = [], []
    for node in inv["nodes"]:
        for group, command in [
            (dcgm, ["bash", root + "dcgm_diag.sh", "--execute"]),
            (
                affinity,
                [
                    "python3",
                    root + "affinity_probe.py",
                    "--inventory",
                    inv["remote_inventory"],
                    "--node",
                    node["name"],
                ],
            ),
        ]:
            prefix = "dcgm:" if group is dcgm else "affinity:"
            group.append(
                {
                    "id": prefix + node["name"],
                    "node": node["name"],
                    "remote_command": command,
                    "argv": ssh(
                        node,
                        [
                            "python3",
                            root + "remote_worker.py",
                            "run",
                            "--token",
                            uuid.uuid4().hex,
                            "--deadline-epoch",
                            "SITE_ABSOLUTE_DEADLINE_REQUIRED",
                            "--",
                        ]
                        + command,
                    ),
                }
            )
    return {
        "schema": 1,
        "status": "planned-unverified",
        "healthy": False,
        "note": NOTE,
        "inventory": inv,
        "thresholds": thresholds,
        "dcgm": dcgm,
        "affinity": affinity,
        "ib": build_pairs(
            inv, thresholds["ib"]["write_size_bytes"], thresholds["ib"]["read_size_bytes"]
        ),
        "nccl": build_matrix(inv),
        "timing": "3 identical-condition rounds, baseline then candidate; 相同配置健康重复，不代表优化对比",
        "budget": {
            "current_limit_usd": 0,
            "real_multi_node_execution": "paused",
            "stop_conditions": [
                "any preflight failure",
                "any command/threshold failure",
                "operator allocation deadline or cost ceiling",
            ],
            "performance_invocations": 6 * (len(build_matrix(inv)) + len(build_pairs(inv))),
            "paid_allocation_permitted": False,
        },
    }


def summarize(plan, results):
    expected = {task["id"] for key in ["dcgm", "affinity", "ib", "nccl"] for task in plan[key]}
    seen = {r["id"] for r in results}
    if len(seen) != len(results) or seen - expected:
        raise ValueError("duplicate or unexpected result identifiers")
    failures = [r["id"] for r in results if r.get("passed") is not True]
    missing = sorted(expected - seen)
    return {
        "healthy": not missing and not failures,
        "missing": missing,
        "failures": failures,
        "note": NOTE,
        "expected": len(expected),
        "reported": len(results),
    }


def validate_site(plan):
    inv = plan["inventory"]
    if inv.get("synthetic") is not False:
        raise ValueError("synthetic/missing site inventory cannot execute")
    execution = inv.get("execution", {})
    if not execution.get("owner") or not execution.get("approval_reference"):
        raise ValueError("future approved site execution owner/reference required")
    if execution.get("max_cost_usd") != 0:
        raise ValueError("current resource ceiling is exactly zero dollars")
    deadline = datetime.fromisoformat(execution.get("stop_at_utc", ""))
    if deadline.tzinfo is None or not 0 < (deadline - datetime.now(UTC)).total_seconds() <= 7200:
        raise ValueError("future allocation deadline must be timezone-aware and within two hours")
    runtime = inv.get("runtime", {})
    for key in [
        "driver",
        "cuda_toolkit",
        "nccl_library",
        "ofed",
        "dcgm",
        "perftest",
        "openmpi",
        "node_image_id",
        "nccl_tests_revision",
    ]:
        if not isinstance(runtime.get(key), str) or not runtime[key].strip():
            raise ValueError(f"missing runtime provenance: {key}")
    if not re.fullmatch(r"[0-9a-f]{40}", runtime["nccl_tests_revision"]):
        raise ValueError("nccl-tests full source commit required")
    for key in ["image_digest", "node_image_sha256"]:
        value = runtime.get(key, "")
        pattern = r"sha256:[0-9a-f]{64}" if key == "image_digest" else r"[0-9a-f]{64}"
        if not re.fullmatch(pattern, value):
            raise ValueError(f"real locked {key} required")
    # A site-owned Task 1 report is a prerequisite; candidate lock alone is insufficient.
    baseline_path = inv.get("baseline_summary")
    if not baseline_path or json.loads(Path(baseline_path).read_text()).get("passed") is not True:
        raise ValueError("successful real Task 1 baseline report required")
    expected_tests = inv.get("dcgm_required_tests")
    if not isinstance(expected_tests, list) or not expected_tests:
        raise ValueError("reviewed nonempty dcgm_required_tests required")
    for node in inv["nodes"]:
        for hca in node["hcas"]:
            positive(hca.get("link_gbps"), "site link_gbps")
            positive(hca.get("ideal_read_latency_us"), "site ideal_read_latency_us")
    for task in plan["nccl"]:
        theory = inv.get("nccl_theory", {}).get(task["theory_key"], {})
        positive(theory.get("theoretical_busbw_gbs"), task["theory_key"])
        if not isinstance(theory.get("basis"), str) or not theory["basis"].strip():
            raise ValueError("site theoretical bottleneck/collective basis required")


def parse_dcgm(text, node, required):
    data = json.loads(text)
    tests, statuses = [], []

    def visit(value):
        if isinstance(value, dict):
            if "status" in value:
                statuses.append(value["status"])
            if isinstance(value.get("tests"), list):
                tests.extend(value["tests"])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(data)
    if not statuses or any(s != "Pass" for s in statuses):
        raise ValueError("DCGM missing/pass-skipped/failed statuses")
    normalize = lambda s: s.lower().replace("_", " ").strip()
    for name in required:
        matches = [t for t in tests if normalize(t.get("name", "")) == normalize(name)]
        if len(matches) != 1:
            raise ValueError(f"DCGM missing/duplicate required test: {name}")
        ids = {
            r.get("entity_id")
            for r in matches[0].get("results", [])
            if r.get("entity_group") == "GPU" and r.get("status") == "Pass"
        }
        if ids != {g["index"] for g in node["gpus"]}:
            raise ValueError(f"DCGM incomplete GPU coverage: {name}")
    return data


def execute(plan, runner=None, timeout=180, checkpoint=None):
    validate_site(plan)
    runner = runner or CommandRunner()
    inv = plan["inventory"]
    nodes = {n["name"]: n for n in inv["nodes"]}
    results = []
    deadline = datetime.fromisoformat(inv["execution"]["stop_at_utc"]).timestamp()

    def task_deadline(seconds):
        remaining(deadline)
        return min(deadline, time.time() + seconds)

    def save(result):
        results.append(result)
        if checkpoint:
            checkpoint(results)

    for group in ["affinity", "dcgm"]:
        for task in plan[group]:
            result = run_remote(
                inv,
                nodes[task["node"]],
                task["remote_command"],
                runner,
                task_deadline(1800 if group == "dcgm" else 30),
            )
            result["id"] = task["id"]
            try:
                if not result["passed"]:
                    raise ValueError(result.get("error", "command failed"))
                if group == "dcgm":
                    result["report"] = parse_dcgm(
                        result["stdout"], nodes[task["node"]], inv["dcgm_required_tests"]
                    )
                else:
                    parsed = json.loads(result["stdout"])
                    if parsed.get("passed") is not True or parsed.get("node") != task["node"]:
                        raise ValueError("affinity probe failed/mismatched")
                    result["report"] = parsed
            except (ValueError, TypeError, KeyError) as exc:
                result.update(passed=False, error=str(exc))
            save(result)
            if not result["passed"]:
                return results
    for group in ["ib", "nccl"]:
        for task in plan[group]:
            measurements, records = [], []
            for round_id in range(1, 4):
                for variant in ["baseline", "candidate"]:
                    if group == "ib":
                        measurement = run_pair(task, inv, runner, timeout, deadline_epoch=deadline)
                        if measurement["passed"]:
                            config = plan["thresholds"]["ib"]
                            violations = evaluate_ib(
                                measurement["value"],
                                task["tool"],
                                task["link_gbps"],
                                task["ideal_read_latency_us"],
                                config["min_write_bw_fraction"],
                                config["min_read_latency_efficiency_fraction"],
                            )
                            measurement.update(passed=not violations, violations=violations)
                        metrics = {"value": measurement.get("value")}
                    else:
                        end = task_deadline(timeout)
                        prefix = uuid.uuid4().hex
                        argv = list(task["argv"])
                        inventory_index = argv.index("--inventory")
                        argv[inventory_index:inventory_index] = [
                            "--deadline-epoch",
                            str(end),
                            "--token-prefix",
                            prefix,
                        ]
                        measurement = {
                            "id": task["id"],
                            "passed": False,
                            "commands": [],
                            "cleanup": [],
                        }
                        try:
                            raw = runner.run(argv, remaining(end))
                        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - record transport failure and clean up
                            raw = Result(
                                argv,
                                130 if isinstance(exc, KeyboardInterrupt) else 127,
                                "",
                                str(exc) or type(exc).__name__,
                            )
                        finally:
                            for name in task["nodes"]:
                                for gpu in nodes[name]["gpus"]:
                                    cleanup = stop_remote(
                                        inv,
                                        nodes[name],
                                        rank_token(prefix, name, gpu["index"]),
                                        runner,
                                    )
                                    measurement["cleanup"].append(cleanup)
                                    measurement["commands"].append(cleanup["command"])
                        measurement["commands"].insert(0, raw.as_dict())
                        metrics = {}
                        try:
                            if raw.returncode or not all(
                                c["verified"] for c in measurement["cleanup"]
                            ):
                                raise ValueError("NCCL command failed or rank cleanup unverified")
                            rows = parse_nccl(raw.stdout)
                            coverage = {(r.size_bytes, r.mode) for r in rows}
                            expected = {
                                (s, mode)
                                for s in task["expected_sizes"]
                                for mode in ["in-place", "out-of-place"]
                            }
                            if coverage != expected or len(rows) != len(expected):
                                raise ValueError("incomplete/duplicate NCCL size/mode coverage")
                            thresholds = (
                                plan["thresholds"]["nccl"] | inv["nccl_theory"][task["theory_key"]]
                            )
                            # nccl-tests explicitly disables in-place alltoall validation.
                            # Preserve those raw columns and disclose the unsupported mode.
                            active = rows
                            if task["collective"] == "alltoall":
                                active = [
                                    r for r in rows if r.mode != "in-place" or r.wrong is not None
                                ]
                                measurement["unsupported_modes"] = {
                                    "in-place": "upstream alltoall does not support correctness validation"
                                }
                            violations = evaluate(active, thresholds)
                            measurement.update(
                                passed=not violations,
                                rows=[asdict(r) for r in rows],
                                violations=[asdict(v) for v in violations],
                            )
                            metrics = {
                                f"{r.size_bytes}:{r.mode}:{metric}": getattr(r, metric)
                                for r in active
                                for metric in ["time_us", "algbw_gbs", "busbw_gbs"]
                            }
                        except (ValueError, TypeError, KeyError) as exc:
                            measurement["error"] = str(exc)
                    measurement.update(round=round_id, variant=variant)
                    measurements.append(measurement)
                    records.append(
                        {
                            "round": round_id,
                            "variant": variant,
                            "valid": measurement["passed"],
                            "fp8": False,
                            "conditions": {"task": task, "runtime": inv["runtime"]},
                            "metrics": metrics,
                        }
                    )
                    if checkpoint:
                        checkpoint(
                            results
                            + [
                                {
                                    "id": task["id"],
                                    "passed": False,
                                    "status": "in-progress",
                                    "measurements": measurements,
                                    "experiment_records": records,
                                }
                            ]
                        )
                    if not measurement["passed"]:
                        break
                if measurements and not measurements[-1]["passed"]:
                    break
            result = {
                "id": task["id"],
                "passed": all(m["passed"] for m in measurements),
                "measurements": measurements,
                "experiment_records": records,
            }
            if result["passed"]:
                result["summary"] = summarize_experiment(records)
            save(result)
            if not result["passed"]:
                return results  # Stop stress on first failed task; remaining matrix stays missing.
    return results


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"operator cancellation: signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory", type=Path, default=Path(__file__).with_name("inventory.example.json")
    )
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    parser.add_argument("--execute", action="store_true", help="future approved site only")
    parser.add_argument(
        "--approval-reference", help="future explicit user approval record; no auto approval"
    )
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists; refusing to overwrite evidence")
    args.output.mkdir(parents=True)
    try:
        plan = create_plan(json.loads(args.inventory.read_text()), load_thresholds(args.thresholds))
        plan["input_sha256"] = {
            "inventory": hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
            "thresholds": hashlib.sha256(args.thresholds.read_bytes()).hexdigest(),
        }
        (args.output / "plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
        )
        if not args.execute:
            summary = summarize(plan, []) | {"status": "planned-unverified"}
        else:
            if not args.approval_reference or args.timeout <= 0:
                raise ValueError(
                    "explicit future user approval reference and positive timeout required"
                )
            plan["approval_reference"] = args.approval_reference
            if (
                plan["inventory"].get("execution", {}).get("approval_reference")
                != args.approval_reference
            ):
                raise ValueError("CLI approval reference must match reviewed site inventory")
            (args.output / "plan.json").write_text(
                json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
            )

            def checkpoint(results):
                (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")

            results = execute(plan, timeout=args.timeout, checkpoint=checkpoint)
            summary = summarize(plan, results) | {
                "status": "passed"
                if all(r["passed"] for r in results)
                and len(results) == sum(len(plan[k]) for k in ["affinity", "dcgm", "ib", "nccl"])
                else "failed"
            }
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(summary, ensure_ascii=False))
        return 0 if not args.execute or summary["healthy"] else 1
    except (ValueError, OSError, KeyError, TypeError, KeyboardInterrupt) as exc:
        summary = {"status": "blocked", "healthy": False, "error": str(exc), "note": NOTE}
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(summary, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
