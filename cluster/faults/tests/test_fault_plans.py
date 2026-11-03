import json
import os
import subprocess
from pathlib import Path

import pytest

from cluster.faults.plan import FaultExecutor, build_plan, load_schedule

ROOT = Path(__file__).resolve().parents[3]


def test_two_hour_schedule_has_three_distinct_faults_and_no_commands_run():
    plan = build_plan()
    assert plan["duration_seconds"] == 7200
    assert plan["status"] == "paused-budget"
    assert [f["method"] for f in plan["faults"]] == ["gpu_power_cap", "ib_width", "vm_power_loss"]
    assert [f["at_seconds"] for f in plan["faults"]] == [1800, 3600, 5400]
    for fault in plan["faults"]:
        assert fault["restore"]
        assert fault["baseline_queries"]
        assert fault["verify_queries"]
    power, link, vm = plan["faults"]
    assert power["inject"][0][-2:] == ["-pl", "200"]
    assert "-i" in power["inject"][0]
    assert link["inject"][0][-2:] == ["width", "1"]
    assert link["inject"][1][-1] == "reset"
    assert vm["inject"][0][1] == "destroy"
    assert vm["restore"][0][1] == "start"
    assert "uncordon" not in json.dumps(plan)


@pytest.mark.parametrize("name", ["power_cap", "link_degrade", "kill_node"])
def test_shell_entrypoints_only_plan_and_apply_is_unconditionally_paused(name):
    env = dict(os.environ, CRUCIBLE_MULTINODE_APPROVED="1", CRUCIBLE_FAULT_APPROVED="1")
    command = ["bash", f"cluster/faults/{name}.sh"]
    result = subprocess.run(command, cwd=ROOT, check=False, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "paused-budget"
    denied = subprocess.run(
        command + ["--apply"], cwd=ROOT, check=False, env=env, text=True, capture_output=True
    )
    assert denied.returncode != 0
    assert "resource budget" in denied.stderr


def test_invalid_or_management_link_and_unsafe_baselines_are_rejected():
    for key, value in [("management", True), ("transport", "roce"), ("original_width", 1)]:
        spec = load_schedule()
        spec["faults"][1][key] = value
        with pytest.raises(ValueError):
            build_plan(spec)
    spec = load_schedule()
    spec["faults"][0]["gpu_uuid"] = "0"
    with pytest.raises(ValueError):
        build_plan(spec)


def test_rehearsal_captures_baseline_restores_after_failure_and_keeps_evidence():
    fault = build_plan()["faults"][0]
    calls = []

    def shim(argv):
        calls.append(argv)
        if "-pl" in argv and argv[-1] == "200":
            raise RuntimeError("injected command failure")
        return (
            "450, 100, 450\n"
            if "--query-gpu=power.limit,power.min_limit,power.max_limit" in argv
            else "450\n"
        )

    executor = FaultExecutor()
    result = executor.rehearse(fault, runner=shim)
    assert result["status"] == "failed"
    assert result["restored"] is True
    assert result["simulation"] is True
    assert calls[-2][-2:] == ["-pl", "450"]
    assert any(not c["ok"] for c in result["evidence"])
    with pytest.raises(PermissionError, match="resource budget"):
        executor.apply(fault)


def test_ib_real_shaped_query_and_vm_rehearsals_verify_injection_and_restore():
    plan = build_plan()
    baseline = (
        "PortState:......................Active\nPhysLinkState:..................LinkUp\n"
        "LinkWidthSupported:.............1X or 4X\n"
        "LinkWidthEnabled:...............1X or 4X\nLinkWidthActive:................4X\n"
    )
    narrowed = baseline.replace(
        "LinkWidthEnabled:...............1X or 4X", "LinkWidthEnabled:...............1X"
    ).replace("LinkWidthActive:................4X", "LinkWidthActive:................1X")
    outputs = iter([baseline, "", "", narrowed, "", "", baseline])
    link = FaultExecutor().rehearse(plan["faults"][1], runner=lambda argv: next(outputs))
    assert link["status"] == "restored"
    assert link["resolved_restore"][0][-2:] == ["width", "3"]
    outputs = iter(
        ["Persistent: yes\nManaged save: no\n", "running", "", "shut off", "", "running"]
    )
    vm = FaultExecutor().rehearse(plan["faults"][2], runner=lambda argv: next(outputs))
    assert vm["status"] == "restored"
    assert [e["phase"] for e in vm["evidence"]][-1] == "verify_restore"


def test_invalid_baseline_causes_no_injection_and_failed_restore_retains_evidence():
    fault = build_plan()["faults"][0]
    calls = []

    def invalid(argv):
        calls.append(argv)
        return "N/A"

    result = FaultExecutor().rehearse(fault, runner=invalid)
    assert result["status"] == "failed"
    assert len(calls) == 1
    assert result["evidence"][0]["stdout"] == "N/A"
    outputs = iter(["450,100,450", "", "200", "", "200"])
    result = FaultExecutor().rehearse(fault, runner=lambda argv: next(outputs))
    assert result["status"] == "failed"
    assert result["restored"] is False
    assert "restore_error" in result


def test_schedule_shim_obeys_absolute_windows_and_finishes_at_two_hours():
    now, calls, journals = [0.0], [], []
    plan = build_plan()
    gpu_limit, vm_running, ib_narrow = [450], [True], [False]
    baseline = (
        "PortState: Active\nPhysLinkState: LinkUp\nLinkWidthSupported: 1X or 4X\n"
        "LinkWidthEnabled: 1X or 4X\nLinkWidthActive: 4X\n"
    )

    def shim(argv):
        calls.append((now[0], argv))
        if argv[0] == "nvidia-smi":
            if "-pl" in argv:
                gpu_limit[0] = int(argv[-1])
                return ""
            if "power.min_limit" in argv[-2]:
                return "450,100,450"
            return str(gpu_limit[0])
        if argv[0] == "ibportstate":
            if "width" in argv:
                ib_narrow[0] = argv[-1] == "1"
            if argv[-1] == "query":
                return (
                    baseline.replace("1X or 4X\nLinkWidthActive: 4X", "1X\nLinkWidthActive: 1X")
                    if ib_narrow[0]
                    else baseline
                )
            return ""
        if argv[1] == "dominfo":
            return "Persistent: yes\nManaged save: no"
        if argv[1] == "destroy":
            vm_running[0] = False
        elif argv[1] == "start":
            vm_running[0] = True
        return "running" if vm_running[0] else "shut off"

    result = FaultExecutor().rehearse_schedule(
        plan,
        runner=shim,
        clock=lambda: now[0],
        advance=lambda seconds: now.__setitem__(0, now[0] + seconds),
        persist=journals.append,
    )
    assert result["status"] == "complete"
    assert result["wall_time"] == 7200
    assert [(at, argv[-1]) for at, argv in calls if "-pl" in argv] == [(1800, "200"), (2100, "450")]
    assert [
        (at, argv[1])
        for at, argv in calls
        if argv[0] == "virsh" and argv[1] in ("destroy", "start")
    ] == [(5400, "destroy"), (5700, "start")]
    assert sum(e["kind"] == "baseline_saved" for e in journals) == 3


def test_baseline_journal_failure_stops_before_any_mutation():
    calls = []

    def shim(argv):
        calls.append(argv)
        return "450,100,450"

    def persist(event):
        if event["kind"] == "baseline_saved":
            raise OSError("journal not durable")

    result = FaultExecutor().rehearse(build_plan()["faults"][0], runner=shim, persist=persist)
    assert result["status"] == "failed"
    assert len(calls) == 1
    assert "journal not durable" in result["error"]


def test_hold_failure_survives_successful_restore_and_aborts_later_faults():
    plan = build_plan()
    now, calls, limit = [0.0], [], [450]
    missed = [False]

    def runner(argv):
        calls.append(argv)
        if "-pl" in argv:
            limit[0] = int(argv[-1])
            return ""
        if "power.min_limit" in argv[-2]:
            return "450,100,450"
        return str(limit[0])

    def advance(seconds):
        if now[0] == 1800 and not missed[0]:
            missed[0] = True
            raise TimeoutError("window missed")
        now[0] += seconds

    result = FaultExecutor().rehearse_schedule(
        plan, runner=runner, clock=lambda: now[0], advance=advance
    )
    assert result["status"] == "aborted"
    assert len(result["results"]) == 1
    assert result["results"][0]["status"] == "failed"
    assert result["results"][0]["restored"] is True
    assert result["results"][0]["error"] == "window missed"
