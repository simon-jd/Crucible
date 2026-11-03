"""CPU command shims exercise orchestration, not GPU or multi-node acceptance."""

import json
import shlex
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cluster.health.affinity_probe import probe
from cluster.health.report import create_plan, execute, parse_dcgm, summarize, validate_site
from cluster.health.runner import CommandRunner
from cluster.health.tests.test_suite import BW, LAT, inventory


def dcgm(node, status="Pass"):
    return {
        "DCGM GPU Diagnostic": {
            "test_categories": [
                {
                    "category": "Hardware",
                    "tests": [
                        {
                            "name": "diagnostic",
                            "results": [
                                {"entity_group": "GPU", "entity_id": g["index"], "status": status}
                                for g in node["gpus"]
                            ],
                            "test_summary": {"status": status},
                        }
                    ],
                }
            ]
        }
    }


def site(tmp_path):
    inv = inventory()
    # Deliberately fake provenance strictly confined to this CPU test, not a site artifact.
    inv["synthetic"] = False
    inv["runtime"] = {
        k: "SYNTHETIC"
        for k in [
            "driver",
            "cuda_toolkit",
            "nccl_library",
            "ofed",
            "dcgm",
            "perftest",
            "openmpi",
            "node_image_id",
        ]
    }
    inv["runtime"].update(
        nccl_tests_revision="f" * 40, image_digest="sha256:" + "f" * 64, node_image_sha256="f" * 64
    )
    baseline = tmp_path / "SYNTHETIC-baseline.json"
    baseline.write_text('{"passed": true}')
    inv["baseline_summary"] = str(baseline)
    inv["dcgm_required_tests"] = ["diagnostic"]
    inv["execution"] = {
        "owner": "CPU synthetic test",
        "approval_reference": "NOT-A-REAL-APPROVAL",
        "max_cost_usd": 0,
        "stop_at_utc": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    }
    plan = create_plan(inv)
    inv["nccl_theory"] = {
        t["theory_key"]: {"theoretical_busbw_gbs": 12.5, "basis": "SYNTHETIC test theory"}
        for t in plan["nccl"]
    }
    return create_plan(inv)


def nccl_output(alltoall=False):
    sample = (Path(__file__).parent / "fixtures/allreduce_upstream.txt").read_text()
    headers = "\n".join(
        line
        for line in sample.splitlines()
        if line.startswith("#") and "Out of bounds" not in line and "Avg bus bandwidth" not in line
    )
    rows = []
    for size in [134217728, 268435456, 536870912, 1073741824]:
        wrong = "N/A" if alltoall else "0"
        rows.append(f"{size} {size // 4} float sum -1 1000 10 10 0 1000 10 10 {wrong}")
    return headers + "\n" + "\n".join(rows) + "\n# Out of bounds values : 0 OK\n"


class CpuShims:
    """Replace every planned executable with a local Python print command."""

    def __init__(self, inv, failure=None):
        self.inv = inv
        self.real = CommandRunner()
        self.failure = failure
        self.calls = []
        self.ib_tokens = set()

    def shim(self, text):
        return [sys.executable, "-c", "import sys;sys.stdout.write(sys.argv[1])", text]

    def start(self, argv):
        self.calls.append(argv)
        words = shlex.split(argv[-1])
        self.ib_tokens.add(words[words.index("--token") + 1])
        return self.real.start(self.shim("CRUCIBLE_READY\n" + BW))

    def run(self, argv, timeout):
        self.calls.append(argv)
        text = ""
        if " stop " in argv[-1]:
            text = json.dumps({"token": shlex.split(argv[-1])[-1], "verified": True})
        elif "affinity_probe.py" in " ".join(argv):
            words = shlex.split(argv[-1])
            text = json.dumps({"node": words[-1], "passed": True})
        elif "dcgm_diag.sh" in " ".join(argv):
            text = json.dumps(dcgm(self.inv["nodes"][0]))
        elif argv[0] == "mpirun":
            text = nccl_output("alltoall_perf" in " ".join(argv))
            if self.failure == "nccl":
                text = text.replace("10 10 0", "10 1 0", 1)
        elif "ib_write_bw" in argv[-1]:
            text = BW
        elif "ib_read_lat" in argv[-1]:
            text = LAT
        result = self.real.run(self.shim(text), timeout)
        if (
            self.failure == "cleanup"
            and " stop " in argv[-1]
            and shlex.split(argv[-1])[-1] in self.ib_tokens
        ):
            result.returncode = 9
        return result


def test_full_suite_through_cpu_command_shims(tmp_path):
    plan = site(tmp_path)
    runner = CpuShims(plan["inventory"])
    results = execute(plan, runner, timeout=2)
    assert summarize(plan, results)["healthy"]
    assert len(results) == 148
    timed = [r for r in results if "measurements" in r]
    assert len(timed) == 140
    assert all(len(r["measurements"]) == 6 for r in timed)
    assert all(r["summary"]["rounds"] == 3 for r in timed)
    assert all(r["summary"]["baseline"] == r["summary"]["candidate"] for r in timed)
    assert all(
        m["variant"] == ["baseline", "candidate"][i % 2]
        for r in timed
        for i, m in enumerate(r["measurements"])
    )
    assert all(
        c["argv"][0] == sys.executable
        for r in timed
        for m in r["measurements"]
        for c in m["commands"]
    )
    assert any("unsupported_modes" in m for r in timed for m in r["measurements"])
    assert any(key.endswith(":time_us") for r in timed for key in r["summary"]["baseline"])


@pytest.mark.parametrize("failure", ["nccl", "cleanup"])
def test_failed_measurement_stops_and_is_checkpointed(tmp_path, failure):
    plan = site(tmp_path)
    if failure == "nccl":
        plan["ib"] = []  # Full matrix coverage is checked by the successful shim test above.
    checkpoints = []
    results = execute(
        plan,
        CpuShims(plan["inventory"], failure),
        timeout=2,
        checkpoint=lambda x: checkpoints.append(json.loads(json.dumps(x))),
    )
    assert not summarize(plan, results)["healthy"]
    assert not results[-1]["passed"]
    assert len(results[-1]["measurements"]) == 1
    assert "summary" not in results[-1]
    assert checkpoints[-1][-1]["measurements"]


def test_site_missing_inputs_fail_before_commands(tmp_path):
    with pytest.raises(ValueError):
        execute(create_plan(inventory()), runner=object())
    plan = site(tmp_path)
    plan["inventory"]["nccl_theory"].clear()
    with pytest.raises(ValueError):
        validate_site(plan)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x.clear(),
        lambda x: x["DCGM GPU Diagnostic"]["test_categories"][0]["tests"][0]["results"].pop(),
        lambda x: x["DCGM GPU Diagnostic"]["test_categories"][0]["tests"][0].update(name="wrong"),
    ],
)
def test_dcgm_missing_coverage_rejected(mutate):
    node = inventory()["nodes"][0]
    data = dcgm(node)
    mutate(data)
    with pytest.raises(ValueError):
        parse_dcgm(json.dumps(data), node, ["diagnostic"])


@pytest.mark.parametrize("status", ["Fail", "Skip", "Warn", ""])
def test_dcgm_nonpassing_status_rejected(status):
    node = inventory()["nodes"][0]
    with pytest.raises(ValueError):
        parse_dcgm(json.dumps(dcgm(node, status)), node, ["diagnostic"])


def test_affinity_reads_actual_sysfs_and_rejects_downgrade(tmp_path):
    node = inventory()["nodes"][0]
    for gpu in node["gpus"]:
        p = tmp_path / "bus/pci/devices" / gpu["pci_bus_id"] / "numa_node"
        p.parent.mkdir(parents=True)
        p.write_text(str(gpu["numa_node"]))
    for hca in node["hcas"]:
        p = tmp_path / "class/infiniband" / hca["name"]
        (p / "device").mkdir(parents=True)
        (p / "device/numa_node").write_text(str(hca["numa_node"]))
        (p / "ports/1").mkdir(parents=True)
        (p / "ports/1/state").write_text("4: ACTIVE")
        (p / "ports/1/rate").write_text("100 Gb/sec (4X EDR)")
    listing = "0, 00000000:10:00.0\n1, 00000000:11:00.0\n"
    assert probe(node, tmp_path, listing)["passed"]
    (tmp_path / "class/infiniband/mlx5_0/ports/1/rate").write_text("50 Gb/sec")
    with pytest.raises(ValueError):
        probe(node, tmp_path, listing)
