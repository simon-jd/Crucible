"""Offline fault command plans and injected-runner rehearsals; no live subprocesses.

apply is deliberately hard-closed under the current zero-budget multi-node pause.
Rehearsals require an explicit command shim; they never authorize a host or target.
"""

import argparse
import copy
import json
import math
import re
from itertools import pairwise
from pathlib import Path

import yaml

PAUSED = "resource budget ($0): multi-node faults paused; separate user budget and dedicated lease authorization required"


def load_schedule(path=None):
    return yaml.safe_load(Path(path or Path(__file__).with_name("schedule.yaml")).read_text())


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("explicit safe resource identity required")
    return value


def _integer(value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("integer outside supported range")
    return value


def build_plan(spec=None):
    spec = copy.deepcopy(load_schedule() if spec is None else spec)
    if spec.get("schema") != 1 or spec.get("duration_seconds") != 7200:
        raise ValueError("schema 1 and fixed two-hour schedule required")
    faults, ids, methods = [], set(), set()
    for item in spec["faults"]:
        fault = dict(item)
        identity = _name(fault["id"])
        method = fault["method"]
        if identity in ids or method in methods:
            raise ValueError("unique fault identities and methods required")
        ids.add(identity)
        methods.add(method)
        _name(fault["node"])
        start, end = fault["at_seconds"], fault["restore_at_seconds"]
        if type(start) is not int or type(end) is not int or not 0 < start < end < 7200:
            raise ValueError("fault window must fit inside fixed 7200-second plan")
        if method == "gpu_power_cap":
            gpu = fault["gpu_uuid"]
            if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu):
                raise ValueError("exact single GPU UUID required")
            watts = _integer(fault["watts"])
            fault.update(
                baseline_queries=[
                    [
                        "nvidia-smi",
                        "-i",
                        gpu,
                        "--query-gpu=power.limit,power.min_limit,power.max_limit",
                        "--format=csv,noheader,nounits",
                    ]
                ],
                inject=[["nvidia-smi", "-i", gpu, "-pl", str(watts)]],
                restore=[["nvidia-smi", "-i", gpu, "-pl", "{saved_power_limit_watts}"]],
                verify_queries=[
                    [
                        "nvidia-smi",
                        "-i",
                        gpu,
                        "--query-gpu=power.limit",
                        "--format=csv,noheader,nounits",
                    ]
                ],
                preconditions=[
                    "dedicated GPU UUID; supported mutable power limit",
                    "persist queried original/min/max watts before mutation",
                ],
            )
        elif method == "ib_width":
            if fault.get("management") is not False or fault.get("transport") != "infiniband":
                raise ValueError("only verified non-management InfiniBand paths supported")
            ca = _name(fault["ca"])
            ca_port, port = _integer(fault["ca_port"]), _integer(fault["switch_port"])
            guid = fault["switch_guid"]
            if not re.fullmatch(r"0x[0-9a-fA-F]{16}", guid):
                raise ValueError("stable switch GUID required; do not rely on mutable LIDs")
            width = _integer(fault["width"])
            original, supported = (
                _integer(fault["original_width"]),
                _integer(fault["supported_width"]),
            )
            if (
                width != 1
                or original == width
                or supported & original != original
                or not original & 1
            ):
                raise ValueError("supported 1x target and original wider enabled mask required")
            base = ["ibportstate", "-C", ca, "-P", str(ca_port), "-G", guid, str(port)]
            fault.update(
                baseline_queries=[base + ["query"]],
                inject=[base + ["width", str(width)], base + ["reset"]],
                restore=[base + ["width", "{saved_width_enabled}"], base + ["reset"]],
                verify_queries=[base + ["query"]],
                preconditions=[
                    "provider permits switch-port width change and reset",
                    "capture PortInfo raw query: active/enabled/supported width and states",
                    "selected active NCCL IB data path; 1x is slower than current active width",
                    "out-of-band management and recovery route survive target reset",
                ],
            )
        elif method == "vm_power_loss":
            domain = fault["domain_uuid"]
            if not re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", domain):
                raise ValueError("exact persistent VM domain UUID required")
            _name(fault["victim_node"])
            fault.update(
                baseline_queries=[["virsh", "dominfo", domain], ["virsh", "domstate", domain]],
                inject=[["virsh", "destroy", domain]],
                restore=[["virsh", "start", domain]],
                verify_queries=[["virsh", "domstate", domain]],
                preconditions=[
                    "exclusive persistent GPU VM; owned hypervisor management context",
                    "baseline running; Persistent: yes; Managed save: no",
                    "durable checkpoint storage outside victim failure domain",
                ],
            )
        else:
            raise ValueError(
                "unsupported fault method; no rank-exit or scheduler-down substitution"
            )
        fault["restore_policy"] = (
            "restore device/VM only; retain scheduler quarantine pending health verification"
        )
        faults.append(fault)
    if methods != {"gpu_power_cap", "ib_width", "vm_power_loss"}:
        raise ValueError("all three distinct fault methods required")
    faults.sort(key=lambda f: f["at_seconds"])
    if any(a["restore_at_seconds"] >= b["at_seconds"] for a, b in pairwise(faults)):
        raise ValueError("fault windows must not overlap")
    return {
        "schema": 1,
        "status": "paused-budget",
        "duration_seconds": 7200,
        "training_stop_seconds": 7080,
        "final_checkpoint_reserve_seconds": 120,
        "recovery_budget_seconds": 300,
        "authorization": PAUSED,
        "example_targets": True,
        "faults": faults,
        "abort_conditions": [
            "any ownership/baseline/path mismatch",
            "checkpoint save/load failure",
            "management reachability loss",
            "restore verification failure",
            "previous incident not recovered before next scheduled fault",
        ],
        "required_run_manifest": [
            "image digest",
            "driver/CUDA/NCCL/OFED versions",
            "model revision",
            "lease ownership and provider permission",
            "checkpoint storage and at most two retained checkpoints",
        ],
    }


def _ib_field(output, name):
    match = re.search(rf"^{name}:[ \t]*[.]*[ \t]*(.+?)[ \t]*$", output, re.MULTILINE)
    if not match:
        raise ValueError(f"missing IB baseline field {name}")
    return match[1]


def _ib_width(output, name):
    value = _ib_field(output, name)
    widths = {"1X": 1, "2X": 16, "4X": 2, "8X": 4, "12X": 8}
    parts = re.split(r"\s+or\s+", value)
    if not parts or any(part not in widths for part in parts):
        raise ValueError(f"unsupported IB width encoding {value}")
    return sum(widths[part] for part in set(parts))


def _ib_active(output):
    return (
        _ib_field(output, "PortState") == "Active"
        and _ib_field(output, "PhysLinkState") == "LinkUp"
    )


class FaultExecutor:
    """Only shim rehearsal is supported today. apply cannot be enabled by a flag/env.

    Rehearsal reads and retains real-shaped baselines from a shim, resolves exact
    restore arguments, attempts restoration even after partial injection failure,
    and preserves all command outputs. No automatic scheduler resume is issued.
    """

    def apply(self, fault):
        raise PermissionError(PAUSED)

    def rehearse(self, fault, *, runner, clock=lambda: 0.0, hold=lambda: None, persist=None):
        evidence = []
        result = {"simulation": True, "status": "failed", "restored": False, "evidence": evidence}

        def call(argv, phase):
            entry = {"argv": argv, "phase": phase, "started_at": clock()}
            evidence.append(entry)
            if persist:
                persist({"kind": "command_started", "fault_id": fault["id"], **entry})
            try:
                output = runner(argv)
                entry.update(ok=True, stdout=output)
                return output
            except Exception as exc:
                entry.update(
                    ok=False,
                    error=str(exc),
                    stdout=getattr(exc, "stdout", None),
                    stderr=getattr(exc, "stderr", None),
                )
                raise
            finally:
                entry["finished_at"] = clock()
                if persist:
                    persist({"kind": "command_finished", "fault_id": fault["id"], **entry})

        try:
            baseline = [call(argv, "baseline") for argv in fault["baseline_queries"]]
            restore = copy.deepcopy(fault["restore"])
            method = fault["method"]
            if method == "gpu_power_cap":
                original, minimum, maximum = map(float, baseline[0].strip().split(","))
                if not all(math.isfinite(v) for v in (original, minimum, maximum)) or not (
                    0 < minimum <= fault["watts"] < original <= maximum
                ):
                    raise ValueError("invalid/unsupported power baseline or cap is not a slowdown")
                restore[0][-1] = f"{original:g}"
            elif method == "ib_width":
                enabled = _ib_width(baseline[0], "LinkWidthEnabled")
                active = _ib_width(baseline[0], "LinkWidthActive")
                supported = _ib_width(baseline[0], "LinkWidthSupported")
                if (
                    enabled != fault["original_width"]
                    or supported != fault["supported_width"]
                    or not _ib_active(baseline[0])
                    or active not in (2, 4, 8, 16)
                    or active & enabled != active
                ):
                    raise ValueError("IB baseline mismatch or link is not active and wider than 1x")
                restore[0][-1] = str(enabled)
            elif method == "vm_power_loss":
                if (
                    not re.search(r"^Persistent:\s+yes\s*$", baseline[0], re.MULTILINE)
                    or not re.search(r"^Managed save:\s+no\s*$", baseline[0], re.MULTILINE)
                    or baseline[1].strip() != "running"
                ):
                    raise ValueError("VM must be running, persistent and without managed save")
            else:
                raise ValueError("unsupported method")
            result["saved_baseline"] = baseline
            result["resolved_restore"] = restore
            if persist:
                persist(
                    {
                        "kind": "baseline_saved",
                        "fault_id": fault["id"],
                        "baseline": baseline,
                        "resolved_restore": restore,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - return captured evidence to caller
            result["error"] = str(exc)
            return result  # Never mutate without a captured, valid baseline.
        try:
            for argv in fault["inject"]:
                call(argv, "inject")
            injected = [call(argv, "verify_injection") for argv in fault["verify_queries"]]
            if method == "gpu_power_cap":
                verified = float(injected[0].strip()) == fault["watts"]
            elif method == "ib_width":
                verified = (
                    _ib_active(injected[0])
                    and _ib_width(injected[0], "LinkWidthEnabled") == 1
                    and _ib_width(injected[0], "LinkWidthActive") == 1
                )
            else:
                verified = injected[0].strip() == "shut off"
            if not verified:
                raise ValueError("fault effect not confirmed; do not label as injected")
            # Caller clock models fault duration; this rehearsal never sleeps.
            result["status"] = "injected"
            hold()
        except Exception as exc:  # noqa: BLE001 - return captured evidence to caller
            result.update(status="failed", error=str(exc))
        finally:
            try:
                for argv in restore:
                    call(argv, "restore")
                after = [call(argv, "verify_restore") for argv in fault["verify_queries"]]
                if method == "gpu_power_cap":
                    restored = float(after[0].strip()) == original
                elif method == "ib_width":
                    restored = (
                        _ib_active(after[0])
                        and _ib_width(after[0], "LinkWidthEnabled") == enabled
                        and _ib_width(after[0], "LinkWidthActive") == active
                    )
                else:
                    restored = after[0].strip() == "running"
                if not restored:
                    raise ValueError("restore state does not match baseline")
                result["restored"] = True
                if result["status"] == "injected":
                    result["status"] = "restored"
            except Exception as exc:  # noqa: BLE001 - preserve recovery failure evidence
                result.update(status="failed", restore_error=str(exc))
        return result

    def rehearse_schedule(self, plan, *, runner, clock, advance, persist=None):
        """CPU clock-shim dispatch of all windows, aborting before the next fault.

        advance(seconds) advances a simulated coordinator and may drive a fake
        training loop. There is no sleeping, scheduler launch or remote dispatch.
        """
        start = clock()
        results = []

        def until(offset):
            remaining = start + offset - clock()
            if remaining < 0:
                raise ValueError("schedule deadline overrun; stop fault campaign")
            advance(remaining)
            if clock() < start + offset:
                raise ValueError("simulation clock did not reach schedule deadline")

        for fault in plan["faults"]:
            until(fault["at_seconds"])
            result = self.rehearse(
                fault,
                runner=runner,
                clock=clock,
                persist=persist,
                hold=lambda f=fault: until(f["restore_at_seconds"]),
            )
            results.append(result)
            if result["status"] != "restored":
                return {"simulation": True, "status": "aborted", "results": results}
        until(plan["duration_seconds"])
        return {
            "simulation": True,
            "status": "complete",
            "results": results,
            "wall_time": clock() - start,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["gpu_power_cap", "ib_width", "vm_power_loss"])
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        parser.exit(2, PAUSED + "\n")
    plan = build_plan(load_schedule(args.schedule))
    if args.method:
        plan["faults"] = [f for f in plan["faults"] if f["method"] == args.method]
    print(json.dumps(plan, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
