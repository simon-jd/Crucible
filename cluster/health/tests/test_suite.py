import json
import sys
import time
from pathlib import Path

import pytest

from cluster.health.ib_perftest import build_pairs, evaluate_ib, parse_perftest, run_pair
from cluster.health.nccl_matrix import build_matrix, validate_inventory
from cluster.health.report import create_plan, summarize
from cluster.health.runner import CommandRunner, Result


def inventory():
    nodes = []
    for i in range(4):
        nodes.append(
            {
                "name": f"n{i}",
                "ssh": f"operator@n{i}.example.invalid",
                "address": f"n{i}.example.invalid",
                "gpus": [
                    {"index": g, "pci_bus_id": f"0000:{g + 16:02x}:00.0", "numa_node": g}
                    for g in range(2)
                ],
                "hcas": [
                    {
                        "name": f"mlx5_{g}",
                        "port": 1,
                        "numa_node": g,
                        "gpu_index": g,
                        "link_gbps": 100,
                        "ideal_read_latency_us": 2,
                    }
                    for g in range(2)
                ],
            }
        )
    return {
        "schema": 1,
        "synthetic": True,
        "nodes": nodes,
        "shared_root": "/shared/crucible",
        "nccl_bin": "/opt/nccl-tests/build",
        "remote_inventory": "/shared/site/inventory.json",
        "runtime": {},
        "nccl_theory": {},
    }


def test_full_matrix_and_every_directed_hca_pair():
    inv = inventory()
    matrix = build_matrix(inv)
    assert len(matrix) == (4 + 6 + 1) * 4
    assert {len(x["nodes"]) for x in matrix} == {1, 2, 4}
    assert {x["collective"] for x in matrix} == {
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "alltoall",
    }
    assert len({x["id"] for x in matrix}) == len(matrix)
    assert all("-l operator" in x["argv"][x["argv"].index("plm_rsh_agent") + 1] for x in matrix)
    pairs = build_pairs(inv)
    assert len(pairs) == 4 * 3 * 2 * 2 * 2
    coverage = {
        (x["server"], x["client"], x["server_hca"], x["client_hca"], x["tool"]) for x in pairs
    }
    assert len(coverage) == len(pairs)
    assert all(
        "--use_cuda" in x["server_command"] and "--use_cuda" in x["client_command"] for x in pairs
    )
    assert all("-b" not in x["server_command"] for x in pairs)


def test_default_offline_plan_never_spawns(monkeypatch):
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: pytest.fail("offline spawned"))
    plan = create_plan(inventory())
    assert plan["status"] == "planned-unverified"
    assert len(plan["dcgm"]) == 4 and len(plan["affinity"]) == 4
    assert not summarize(plan, [])["healthy"]
    assert "未在真实多机上验证" in json.dumps(plan, ensure_ascii=False)


@pytest.mark.parametrize(
    "change",
    [
        lambda x: x["nodes"].pop(),
        lambda x: x["nodes"][0]["gpus"].clear(),
        lambda x: x["nodes"][0]["hcas"][0].update(gpu_index=999),
        lambda x: x["nodes"][0]["hcas"][0].update(numa_node=-1),
        lambda x: x["nodes"][0].update(ssh="bad;command"),
    ],
)
def test_inventory_invalid(change):
    inv = inventory()
    change(inv)
    with pytest.raises(ValueError):
        validate_inventory(inv)


def test_missing_or_failed_results_cannot_pass():
    plan = create_plan(inventory())
    assert not summarize(plan, [{"id": plan["dcgm"][0]["id"], "passed": True}])["healthy"]
    results = [
        {"id": x["id"], "passed": True}
        for key in ["dcgm", "affinity", "ib", "nccl"]
        for x in plan[key]
    ]
    assert summarize(plan, results)["healthy"]
    results[-1]["passed"] = False
    assert not summarize(plan, results)["healthy"]
    with pytest.raises(ValueError):
        summarize(plan, results + [results[0]])


BW = """#bytes #iterations BW peak[GB/sec] BW average[GB/sec] MsgRate[Mpps]
8388608 1000 11.3 10.0 0.001
"""
LAT = """#bytes #iterations t_min[usec] t_max[usec] t_typical[usec] t_avg[usec] t_stdev[usec] 99%[usec] 99.9%[usec]
8 1000 1.0 10.0 2.0 3.0 0.2 8.0 9.0
"""


def test_ib_units_and_fraction_checks():
    bw = parse_perftest(BW, "ib_write_bw", 8388608)
    lat = parse_perftest(LAT, "ib_read_lat", 8)
    assert bw == 10 and lat == 2
    assert not evaluate_ib(bw, "ib_write_bw", 100, 2, 0.7, 0.5)
    assert evaluate_ib(1, "ib_write_bw", 100, 2, 0.7, 0.5)
    assert evaluate_ib(9, "ib_read_lat", 100, 2, 0.7, 0.5)
    with pytest.raises(ValueError):
        parse_perftest(BW.replace("GB/sec", "Mb/sec"), "ib_write_bw", 8388608)
    with pytest.raises(ValueError):
        parse_perftest(BW.replace("10.0", "nan"), "ib_write_bw", 8388608)
    with pytest.raises(ValueError):
        parse_perftest(BW, "ib_write_bw", 8)


def test_timeout_cleans_local_process_group(tmp_path):
    pidfile = tmp_path / "child.pid"
    code = (
        "import subprocess,time,pathlib; "
        "p=subprocess.Popen(['sleep','60']); "
        f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    result = CommandRunner().run([sys.executable, "-c", code], timeout=0.3)
    assert result.returncode == 124 and result.timed_out
    pid = int(pidfile.read_text())
    # A reaped orphan may briefly remain a zombie, but cannot keep running.
    status = Path(f"/proc/{pid}/stat")
    assert not status.exists() or status.read_text().split()[2] == "Z"


def test_pair_always_requests_remote_server_cleanup():
    class Process:
        def wait_ready(self, timeout):
            return True

        def finish(self, timeout):
            return Result(["server"], 0, "", "")

        def terminate(self):
            pass

    class Runner:
        def __init__(self):
            self.calls = []

        def start(self, argv):
            return Process()

        def run(self, argv, timeout):
            self.calls.append(argv)
            return Result(argv, 7 if "stop" not in argv[-1] else 0, "", "synthetic failure")

    runner = Runner()
    pair = build_pairs(inventory())[0]
    result = run_pair(pair, inventory(), runner, timeout=1)
    assert not result["passed"]
    assert any("stop" in x[-1] for x in runner.calls)


def test_remote_worker_timeout_kills_child(tmp_path):
    # CPU shell command only; no SSH, GPU program or service is started.
    from cluster.health import remote_worker

    token = "a" * 32
    result = CommandRunner().run(
        [
            sys.executable,
            remote_worker.__file__,
            "run",
            "--token",
            token,
            "--deadline-epoch",
            str(time.time() + 0.2),
            "--state-dir",
            str(tmp_path),
            "--",
            "sleep",
            "60",
        ],
        timeout=2,
    )
    assert result.returncode == 124
    assert not (tmp_path / (token + ".json")).exists()
    assert json.loads((tmp_path / (token + ".done.json")).read_text())["verified"]
