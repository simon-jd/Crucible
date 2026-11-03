"""CPU-only synthetic fixtures; these do not represent cluster acceptance."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_missing_scheduler_is_archived(tmp_path):
    env = dict(os.environ, PATH=str(tmp_path), BASELINE_PYTHON=sys.executable)
    result = subprocess.run(
        check=False,
        args=["/bin/bash", str(ROOT / "check_baseline.sh"), "--output", str(tmp_path / "evidence")],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    summary = json.loads((tmp_path / "evidence" / "summary.json").read_text())
    assert summary["passed"] is False
    assert any("srun" in error for error in summary["errors"])
    commands = json.loads((tmp_path / "evidence" / "commands.json").read_text())
    assert commands[0]["argv"] == [
        "srun",
        "-N4",
        "--ntasks-per-node=8",
        "--gpus-per-node=8",
        "nvidia-smi",
        "-L",
    ]
    assert commands[0]["returncode"] == 127


def test_unique_gpu_count_rejects_repeated_or_missing_cards():
    baseline = load_module("baseline")
    assert len(baseline.gpu_ids("GPU 0: NVIDIA H100 (UUID: GPU-abc)\n" * 32)) == 1
    with pytest.raises(baseline.BaselineError, match="32"):
        baseline.check_gpu_listing("GPU 0: NVIDIA H100 (UUID: GPU-abc)\n" * 32)


def test_candidate_lock_is_not_accepted():
    baseline = load_module("baseline")
    lock = json.loads((ROOT / "versions.lock").read_text())
    with pytest.raises(baseline.BaselineError, match="digest"):
        baseline.validate_lock(lock, "slurm")


@pytest.fixture
def fixture_cluster(tmp_path):
    """Synthetic four-node command shim, executed as an actual local subprocess."""
    import hashlib

    lock = json.loads((ROOT / "versions.lock").read_text())
    lock["status"] = "locked"
    lock["image"]["digest"] = "sha256:" + "a" * 64
    lock["image"]["slurm_sqsh_sha256"] = "b" * 64
    lock["node_image"] = {"id": "synthetic-image", "sha256": "c" * 64}
    lock["k8s_images"] = {"operator:fixture": "sha256:" + "d" * 64}
    lock_path = tmp_path / "synthetic.lock"
    lock_path.write_text(json.dumps(lock))
    probe_hash = hashlib.sha256((ROOT / "node_probe.py").read_bytes()).hexdigest()
    hosts, runtime = [], []
    for node in range(4):
        ids = [f"GPU-{node}-{card}" for card in range(8)]
        hosts.append(
            {
                "kind": "host",
                "node": f"worker-{node}",
                "probe_sha256": probe_hash,
                "gpus": [
                    {
                        "uuid": value,
                        "model": "NVIDIA H100 SXM",
                        "driver": lock["versions"]["driver"],
                    }
                    for value in ids
                ],
                "topology": "GPU0 GPU1 NIC0\nGPU0 X NV18 PIX",
                "ibstat": "CA 'mlx5_0'\n  State: Active\n  Rate: 400",
                "shared_fstype": "nfs4",
                "ofed": lock["versions"]["ofed"],
                "node_image_id": "synthetic-image",
                "slurm": lock["versions"]["slurm"],
                "enroot": lock["versions"]["enroot"],
                "pyxis": lock["versions"]["pyxis"],
                "image_sha256": "b" * 64,
            }
        )
        runtime.append(
            {
                "kind": "runtime",
                "node": f"worker-{node}",
                "probe_sha256": probe_hash,
                "uuids": ids,
                **{key: lock["versions"][key] for key in ["cuda_toolkit", "cuda_runtime", "nccl"]},
            }
        )
    fixture = {"hosts": hosts, "runtime": runtime, "lock": lock}
    data = tmp_path / "synthetic-observations.json"
    data.write_text(json.dumps(fixture))
    shim = tmp_path / "srun"
    shim.write_text(f"""#!{sys.executable}
# SYNTHETIC COMMAND FIXTURE: never hardware evidence.
import json, sys
from pathlib import Path
x=json.loads(Path({str(data)!r}).read_text())
a=sys.argv[1:]
if a[-2:]==["nvidia-smi", "-L"]:
    for host in x["hosts"]:
        for i,gpu in enumerate(host["gpus"]):
            print(f"GPU {{i}}: {{gpu['model']}} (UUID: {{gpu['uuid']}})")
elif "host" in a:
    root=Path(a[a.index("--shared-root")+1])
    token=(root/a[a.index("--token")+1]).read_text()
    for host in x["hosts"]:
        host["shared_token"]=token
        print("CRUCIBLE_PROBE="+json.dumps(host))
else:
    for runtime in x["runtime"]:
        print("CRUCIBLE_PROBE="+json.dumps(runtime))
""")
    shim.chmod(0o755)
    return tmp_path, lock_path, data, fixture


def run_fixture(fixture_cluster, mutate=None):
    directory, lock_path, data, fixture = fixture_cluster
    if mutate:
        mutate(fixture)
        data.write_text(json.dumps(fixture))
    output = directory / "evidence"
    env = dict(os.environ, PATH=str(directory), BASELINE_PYTHON=sys.executable)
    result = subprocess.run(
        check=False,
        args=[
            "/bin/bash",
            str(ROOT / "check_baseline.sh"),
            "--output",
            str(output),
            "--lock",
            str(lock_path),
            "--shared-root",
            str(directory),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    return result, json.loads((output / "summary.json").read_text()), output


def test_synthetic_slurm_executable_path_passes_and_archives(fixture_cluster):
    result, summary, output = run_fixture(fixture_cluster)
    assert result.returncode == 0, result.stderr + result.stdout
    assert summary["passed"]
    commands = json.loads((output / "commands.json").read_text())
    assert len(commands) == 3
    assert commands[0]["argv"] == load_module("baseline").SLURM_LIST
    assert any(
        "--container-image=/shared/images/crucible.sqsh" == arg for arg in commands[2]["argv"]
    )
    assert len(list(output.glob("node-*-topology.txt"))) == 4
    assert len(list(output.glob("node-*-ibstat.txt"))) == 4
    assert not list(fixture_cluster[0].glob(".crucible-baseline-*"))


@pytest.mark.parametrize(
    "target,field,value,expected",
    [
        ("hosts", "ofed", "WRONG", "OFED version mismatch"),
        ("hosts", "slurm", "0.0", "slurm version mismatch"),
        ("hosts", "enroot", "0.0", "enroot version mismatch"),
        ("hosts", "pyxis", "0.0", "pyxis version mismatch"),
        ("hosts", "ibstat", "State: Down", "Active IB/RoCE"),
        ("hosts", "shared_fstype", "ext4", "shared storage"),
        ("hosts", "image_sha256", "0" * 64, "squashfs digest mismatch"),
        ("hosts", "probe_sha256", "0" * 64, "stale host probe"),
        ("hosts", "node_image_id", "other", "node image identity mismatch"),
        ("runtime", "cuda_toolkit", "0.0", "cuda_toolkit version mismatch"),
        ("runtime", "cuda_runtime", 0, "cuda_runtime version mismatch"),
        ("runtime", "nccl", 0, "nccl version mismatch"),
    ],
)
def test_synthetic_version_and_host_rejections(fixture_cluster, target, field, value, expected):
    result, summary, _ = run_fixture(fixture_cluster, lambda f: f[target][0].update({field: value}))
    assert result.returncode == 1
    assert expected in " ".join(summary["errors"])


def test_synthetic_driver_rejection(fixture_cluster):
    def mutate(f):
        f["hosts"][0]["gpus"][0]["driver"] = "0.0"

    result, summary, _ = run_fixture(fixture_cluster, mutate)
    assert result.returncode == 1
    assert "driver version mismatch" in " ".join(summary["errors"])


def test_synthetic_wrong_card_count_rejected(fixture_cluster):
    result, summary, output = run_fixture(fixture_cluster, lambda f: f["hosts"][0]["gpus"].pop())
    assert result.returncode == 1
    assert "observed 31" in " ".join(summary["errors"])
    assert len(json.loads((output / "commands.json").read_text())) == 1


def test_synthetic_duplicate_runtime_card_rejected(fixture_cluster):
    result, summary, _ = run_fixture(
        fixture_cluster, lambda f: f["runtime"][1].update(f["runtime"][0])
    )
    assert result.returncode == 1
    assert "duplicated/misplaced" in " ".join(summary["errors"])


def test_command_nonzero_and_timeout_archive_real_subprocesses(tmp_path):
    baseline = load_module("baseline")
    runner = baseline.Runner(tmp_path, 0.05)
    with pytest.raises(baseline.BaselineError, match="command failed \\(7\\)"):
        runner.run(
            [sys.executable, "-c", "import sys; print('diagnostic', file=sys.stderr); sys.exit(7)"]
        )
    with pytest.raises(baseline.BaselineError, match="timeout"):
        runner.run(
            [sys.executable, "-c", "import time; print('started',flush=True); time.sleep(5)"]
        )
    commands = json.loads((tmp_path / "commands.json").read_text())
    assert commands[0]["stderr"] == "diagnostic\n"
    assert commands[1]["returncode"] == 124
    assert commands[1]["stdout"] == "started\n"


def test_bad_lock_and_existing_evidence_fail_closed(tmp_path):
    bad = tmp_path / "bad.lock"
    bad.write_text("[]")
    output = tmp_path / "out"
    command = [
        sys.executable,
        str(ROOT / "baseline.py"),
        "--lock",
        str(bad),
        "--output",
        str(output),
    ]
    result = subprocess.run(check=False, args=command, text=True, capture_output=True)
    assert result.returncode == 1
    assert "lock must be an object" in result.stdout
    original = (output / "summary.json").read_bytes()
    assert subprocess.run(check=False, args=command, capture_output=True).returncode == 1
    assert (output / "summary.json").read_bytes() == original


def test_node_probe_local_missing_gpu_fails_without_fake_data(tmp_path):
    env = dict(os.environ, PATH=str(tmp_path))
    result = subprocess.run(
        check=False,
        args=[sys.executable, str(ROOT / "node_probe.py"), "host"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    report = load_module("baseline").reports(result.stdout)[0]
    assert any("nvidia-smi" in str(error) for error in report["errors"])
    assert "gpus" not in report


@pytest.mark.parametrize("scheduler", ["slurm", "k8s"])
def test_render_complete_synthetic_configs(fixture_cluster, scheduler):
    import yaml

    directory, lock_path, _, _ = fixture_cluster
    output = directory / "rendered"
    command = [
        sys.executable,
        str(ROOT / "render.py"),
        "--scheduler",
        scheduler,
        "--lock",
        str(lock_path),
        "--output",
        str(output),
        "--nfs-export",
        "storage.test:/crucible",
    ]
    command += (
        [
            "--controller",
            "head.test",
            "--nodes",
            "gpu1,gpu2,gpu3,gpu4",
            "--cpus",
            "192",
            "--memory-mb",
            "1000000",
        ]
        if scheduler == "slurm"
        else ["--rdma-interface", "ib0"]
    )
    result = subprocess.run(check=False, args=command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    cloud = yaml.safe_load((output / "cloud-init.yaml").read_text())
    assert cloud["mounts"][0][0] == "storage.test:/crucible"
    assert json.loads(cloud["write_files"][0]["content"])["status"] == "locked"
    if scheduler == "k8s":
        resources = list(yaml.safe_load_all((output / "resources.yaml").read_text()))
        job = next(r for r in resources if r["kind"] == "Job")
        assert job["spec"]["parallelism"] == 32
        assert job["spec"]["template"]["spec"]["containers"][0]["image"].endswith(
            "@sha256:" + "a" * 64
        )
        cm = json.loads((output / "probe-configmap.json").read_text())
        assert cm["data"]["node_probe.py"] == (ROOT / "node_probe.py").read_text()
    else:
        assert (
            "NodeName=gpu1,gpu2,gpu3,gpu4 CPUs=192 RealMemory=1000000"
            in (output / "slurm.conf").read_text()
        )


def test_unlocked_render_cannot_create_deployment(tmp_path):
    result = subprocess.run(
        check=False,
        args=[
            sys.executable,
            str(ROOT / "render.py"),
            "--scheduler",
            "k8s",
            "--output",
            str(tmp_path / "out"),
            "--nfs-export",
            "storage:/data",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "digest" in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.fixture
def fixture_k8s(fixture_cluster):
    directory, _lock_path, data, fixture = fixture_cluster
    fixture["runtime"] = [
        dict(report, uuids=[card]) for report in fixture["runtime"] for card in report["uuids"]
    ]
    digest = fixture["lock"]["image"]["digest"]

    def pod(name, node, gpu):
        container = (
            {"name": "probe", "resources": {"limits": {"nvidia.com/gpu": 1}}}
            if gpu
            else {"name": "probe"}
        )
        return {
            "metadata": {"name": name},
            "spec": {"nodeName": node, "containers": [container]},
            "status": {
                "phase": "Running",
                "conditions": [{"type": "Ready", "status": "True"}],
                "containerStatuses": [
                    {"ready": True, "imageID": "docker-pullable://fixture@" + digest}
                ],
            },
        }

    fixture["gpu_pods"] = [
        pod(f"gpu-{i}", report["node"], True) for i, report in enumerate(fixture["runtime"])
    ]
    fixture["host_pods"] = [
        pod(f"host-{i}", host["node"], False) for i, host in enumerate(fixture["hosts"])
    ]
    fixture["operator_pods"] = [
        {
            "status": {
                "phase": "Running",
                "containerStatuses": [
                    {
                        "image": "operator:fixture",
                        "imageID": "docker-pullable://operator@sha256:" + "d" * 64,
                        "ready": True,
                    }
                ],
            }
        }
    ]
    data.write_text(json.dumps(fixture))
    shim = directory / "kubectl"
    shim.write_text(f"""#!{sys.executable}
# SYNTHETIC Kubernetes API/exec fixture; no real cluster.
import json, sys
from pathlib import Path
x=json.loads(Path({str(data)!r}).read_text()); a=sys.argv[1:]
if a[0]=="version": print(json.dumps({{"serverVersion":{{"gitVersion":x["lock"]["versions"]["kubernetes"]}}}}))
elif "exec" in a:
    name=a[a.index("exec")+1]
    if name.startswith("host-"):
        report=x["hosts"][int(name.split("-")[1])]
        report["shared_token"]=(Path(a[a.index("--shared-root")+1])/a[a.index("--token")+1]).read_text()
    else: report=x["runtime"][int(name.split("-")[1])]
    print("CRUCIBLE_PROBE="+json.dumps(report))
else:
    key="host_pods" if "component=host" in a else "gpu_pods" if "component=gpu" in a else "operator_pods"
    print(json.dumps({{"items":x[key]}}))
""")
    shim.chmod(0o755)
    helm = directory / "helm"
    helm.write_text(f"""#!{sys.executable}
# SYNTHETIC Helm release inventory.
import json
print(json.dumps([{{"name":n,"chart":n+"-"+v,"status":"deployed"}} for n,v in
    [("gpu-operator","v25.3.2"),("network-operator","25.4.0"),("kueue","0.13.4")]]))
""")
    helm.chmod(0o755)
    return fixture_cluster


def run_k8s_fixture(fixture_k8s, mutate=None):
    directory, lock_path, data, fixture = fixture_k8s
    if mutate:
        mutate(fixture)
        data.write_text(json.dumps(fixture))
    output = directory / "evidence"
    env = dict(os.environ, PATH=str(directory), BASELINE_PYTHON=sys.executable)
    result = subprocess.run(
        check=False,
        args=[
            "/bin/bash",
            str(ROOT / "check_baseline.sh"),
            "--scheduler",
            "k8s",
            "--output",
            str(output),
            "--lock",
            str(lock_path),
            "--shared-root",
            str(directory),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    return result, json.loads((output / "summary.json").read_text()), output


def test_synthetic_k8s_full_executable_path(fixture_k8s):
    result, summary, output = run_k8s_fixture(fixture_k8s)
    assert result.returncode == 0, result.stdout + result.stderr
    assert summary["passed"]
    assert len(json.loads((output / "commands.json").read_text())) == 44
    assert len(json.loads((output / "reports.json").read_text())["runtime"]) == 32
    assert len(list(output.glob("node-*-ibstat.txt"))) == 4


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (lambda f: f["gpu_pods"].pop(), "expected 32"),
        (lambda f: f["gpu_pods"][0]["status"].update(phase="Pending"), "Running/Ready"),
        (
            lambda f: f["gpu_pods"][0]["status"]["containerStatuses"][0].update(
                imageID="unresolved"
            ),
            "image digest",
        ),
        (lambda f: f["gpu_pods"][0]["spec"].update(nodeName="worker-1"), "distributed 8 per node"),
        (
            lambda f: f["operator_pods"][0]["status"]["containerStatuses"][0].update(
                imageID="unresolved"
            ),
            "operator container image digest",
        ),
        (lambda f: f["runtime"][0].update(nccl=0), "nccl version mismatch"),
    ],
)
def test_synthetic_k8s_rejections(fixture_k8s, mutation, expected):
    result, summary, _ = run_k8s_fixture(fixture_k8s, mutation)
    assert result.returncode == 1
    assert expected in " ".join(summary["errors"])


def test_corrupt_probe_json_is_failure(fixture_cluster):
    directory = fixture_cluster[0]
    (directory / "srun").write_text(f"#!{sys.executable}\nprint('not JSON or GPU listing')\n")
    result, summary, _ = run_fixture(fixture_cluster)
    assert result.returncode == 1
    assert "observed 0" in " ".join(summary["errors"])


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("image", "digest", "latest"),
        ("image", "slurm_sqsh_sha256", None),
        ("node_image", "sha256", None),
        ("versions", "driver", "latest"),
        ("versions", "nccl", True),
        ("versions", "cuda_runtime", "12080"),
    ],
)
def test_unlocked_or_ambiguous_fields_rejected(fixture_cluster, section, key, value):
    baseline = load_module("baseline")
    fixture = fixture_cluster[3]
    fixture["lock"][section][key] = value
    with pytest.raises(baseline.BaselineError):
        baseline.validate_lock(fixture["lock"], "slurm")


def test_real_probe_subprocess_preserves_diagnostics_after_late_failure(tmp_path):
    """Actual node_probe subprocess; command fixtures are explicitly synthetic."""
    for name, source in {
        "nvidia-smi": "import sys; print('TOPOLOGY-SYNTHETIC' if 'topo' in sys.argv else 'GPU-fixture, NVIDIA H100, 570.133.20')",
        "ibstat": "print('FABRIC-SYNTHETIC State: Active')",
        "ofed_info": "import sys; print('OFED-LATE-FAILURE', file=sys.stderr); sys.exit(7)",
    }.items():
        path = tmp_path / name
        path.write_text(f"#!{sys.executable}\n{source}\n")
        path.chmod(0o755)
    result = subprocess.run(
        [sys.executable, str(ROOT / "node_probe.py"), "host"],
        env=dict(os.environ, PATH=str(tmp_path)),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    report = load_module("baseline").reports(result.stdout)[0]
    assert report["topology"] == "TOPOLOGY-SYNTHETIC"
    assert "FABRIC-SYNTHETIC" in report["ibstat"]
    assert any("OFED-LATE-FAILURE" in str(error) for error in report["errors"])
    assert any(command["returncode"] == 7 for command in report["commands"])


def test_slurm_nonzero_archives_healthy_and_partial_host_reports(fixture_cluster):
    directory = fixture_cluster[0]
    shim = directory / "srun"
    shim.write_text(
        shim.read_text().replace(
            "else:\n    for runtime", "    sys.exit(7)\nelse:\n    for runtime"
        )
    )

    def fail_one_host(fixture):
        fixture["hosts"][3].pop("ofed")
        fixture["hosts"][3]["errors"] = [{"field": "ofed", "error": "SYNTHETIC late failure"}]

    result, summary, output = run_fixture(fixture_cluster, fail_one_host)
    assert result.returncode == 1
    assert not summary["passed"]
    assert len(list(output.glob("node-*-topology.txt"))) == 4
    assert len(list(output.glob("node-*-ibstat.txt"))) == 4
    assert len(json.loads((output / "partial-reports.json").read_text())["hosts"]) == 4


def test_k8s_later_host_failure_keeps_prior_and_partial_diagnostics(fixture_k8s):
    directory = fixture_k8s[0]
    shim = directory / "kubectl"
    shim.write_text(
        shim.read_text().replace(
            'print("CRUCIBLE_PROBE="+json.dumps(report))',
            'print("CRUCIBLE_PROBE="+json.dumps(report),flush=True)\n    if name=="host-2": sys.exit(9)',
        )
    )

    def fail_one_host(fixture):
        fixture["hosts"][2].pop("ofed")
        fixture["hosts"][2]["errors"] = [{"field": "ofed", "error": "SYNTHETIC late failure"}]

    result, summary, output = run_k8s_fixture(fixture_k8s, fail_one_host)
    assert result.returncode == 1
    assert not summary["passed"]
    assert len(list(output.glob("node-*-topology.txt"))) == 3
    assert len(list(output.glob("node-*-ibstat.txt"))) == 3
    assert len(json.loads((output / "partial-reports.json").read_text())["hosts"]) == 3
