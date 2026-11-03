"""Offline structural and pinned-source contract tests; no native imports."""

import ast
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from serving.deploy.task13_validate import validate

LAYOUTS = ["pd-1p1d", "pd-2p1d", "pd-cross-node"]


def load(name):
    return yaml.safe_load(Path(f"serving/deploy/{name}.yaml").read_text())


@pytest.mark.parametrize("name", LAYOUTS)
def test_inert_source_checked_layout(name):
    result = validate(load(name))
    assert result["offline_valid"] is True
    assert result["execution_ready"] is False
    assert result["total_gpus"] == 3


@pytest.mark.parametrize("name", LAYOUTS)
def test_unresolved_execution_rejected(name):
    with pytest.raises(ValueError, match="execution rejected"):
        validate(load(name), execution=True)


@pytest.mark.parametrize(
    "mutation",
    [
        "replicas",
        "router_replicas",
        "image",
        "role",
        "endpoint",
        "bootstrap",
        "gpu",
        "node",
        "comparison",
        "handoff",
        "router_command",
        "missing_worker",
    ],
)
def test_invalid_layout_rejected(mutation):
    doc = load("pd-cross-node")
    deployment = doc["items"][1]
    c = deployment["spec"]["template"]["spec"]["containers"][0]
    contract = json.loads(doc["items"][0]["data"]["contract.json"])
    if mutation == "replicas":
        deployment["spec"]["replicas"] = 1
    elif mutation == "router_replicas":
        doc["items"][-2]["spec"]["replicas"] = 1
    elif mutation == "image":
        c["image"] = "sglang:latest"
    elif mutation == "role":
        c["args"][c["args"].index("--disaggregation-mode") + 1] = "decode"
    elif mutation == "endpoint":
        contract["connections"]["prefill"][0]["http"] = "http://wrong:30000"
    elif mutation == "bootstrap":
        doc["items"][2]["spec"]["ports"][1]["targetPort"] = 8999
    elif mutation == "gpu":
        c["resources"]["limits"]["nvidia.com/gpu"] = 1
    elif mutation == "node":
        deployment["spec"]["template"]["spec"]["nodeSelector"]["kubernetes.io/hostname"] = (
            "REPLACE_DECODE_NODE"
        )
    elif mutation == "comparison":
        contract["comparison"]["colocated_total_gpus"] = 2
    elif mutation == "handoff":
        contract["router"]["request_handoff"].remove("bootstrap_room")
    elif mutation == "router_command":
        doc["items"][-2]["spec"]["template"]["spec"]["containers"][0]["args"].append("--mini-lb")
    else:
        doc["items"].pop(1)
    doc["items"][0]["data"]["contract.json"] = json.dumps(contract)
    with pytest.raises(ValueError):
        validate(doc)


def test_pinned_installed_source_flags_and_tp_contract():
    contract = json.loads(Path("docs/evidence/raw/task13-source-contract.json").read_text())
    sources = {}
    for entry in contract["sources"]:
        path = Path(entry["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        sources[path.name.removeprefix("task13-sglang-").removesuffix(".txt")] = path.read_text()
    server = sources["server_args.py"]
    literals = {
        n.value
        for n in ast.walk(ast.parse(server))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    for name in LAYOUTS:
        for d in load(name)["items"]:
            if d["kind"] != "Deployment" or d["metadata"]["name"].endswith("-router"):
                continue
            args = d["spec"]["template"]["spec"]["containers"][0]["args"]
            assert set(args[::2]) <= literals
    assert "larger_tp % smaller_tp == 0" in server
    assert '"rdma"' in sources["transfer_engine.py"]
    assert "req.bootstrap_host}:{req.bootstrap_port}" in sources["decode.py"]
    for field in ("bootstrap_host", "bootstrap_port", "bootstrap_room"):
        assert field in sources["io_struct.py"]


def test_pinned_router_parser_contract_without_native_imports():
    records = json.loads(Path("docs/evidence/raw/task13-router-source-research.json").read_text())
    for record in records:
        assert hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() == record["sha256"]
    source = Path(records[0]["path"]).read_text()
    # Inspect AST instead of importing the router and its native dependencies.
    tree = ast.parse(source)
    flags = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.JoinedStr)
        ):
            suffix = node.args[0].values[-1]
            if isinstance(suffix, ast.Constant):
                flags[suffix.value] = {
                    k.arg: ast.literal_eval(k.value)
                    for k in node.keywords
                    if k.arg in ("nargs", "action")
                }
    assert flags["prefill"] == {"nargs": "+", "action": "append"}
    assert flags["decode"] == {"nargs": 1, "action": "append"}
    assert flags["pd-disaggregation"] == {"action": "store_true"}
