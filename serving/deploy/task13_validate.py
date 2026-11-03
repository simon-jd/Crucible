"""Validate only the three inert Task13 layouts; this is not a deployer."""

import argparse
import json
from pathlib import Path

import yaml

COMMIT = "0c006b8809cd99e1f95926401a2823dd952641c8"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(document, *, execution=False):
    require(document["apiVersion"] == "v1" and document["kind"] == "List", "expected v1 List")
    items = document["items"]
    names = [(x["kind"], x["metadata"]["name"]) for x in items]
    require(len(set(names)) == len(names), "duplicate resource names")
    require(
        all(x["kind"] in ("ConfigMap", "Deployment", "Service") for x in items),
        "unexpected resource kind",
    )
    contracts = [x for x in items if x["kind"] == "ConfigMap"]
    require(len(contracts) == 1, "one offline contract required")
    contract = json.loads(contracts[0]["data"]["contract.json"])
    name = contract["layout"]
    require(name in ("pd-1p1d", "pd-2p1d", "pd-cross-node"), "unknown layout")
    require(
        contract["phase"] == "OFFLINE_ONLY" and contract["source_commit"] == COMMIT,
        "incorrect source/offline contract",
    )
    require(
        contract["physical_validation"] == "未在真实多机上验证"
        and contract["kv_transfer_latency"] == "UNMEASURED",
        "unsupported physical claim",
    )
    comparison = contract["comparison"]
    require(
        comparison["total_gpus"] == comparison["colocated_total_gpus"] == 3,
        "same-card-count colocated comparison required",
    )
    require(
        comparison["model_revision"] == "REPLACE_MODEL_REVISION"
        and comparison["slo_sha256"] == "REPLACE_BF16_SLO_SHA256",
        "unresolved comparison gates",
    )
    require(
        all(
            isinstance(x, str) and x.startswith("REPLACE_")
            for x in contract["prerequisites"].values()
        )
        and set(contract["prerequisites"])
        == {
            "driver_cuda_nccl_ofed",
            "rdma_resource_mapping",
            "model_metadata",
            "router_revision",
            "network_checks",
        },
        "missing prerequisite",
    )
    router_args = [
        "--pd-disaggregation",
        "--host",
        "0.0.0.0",
        "--port",
        "30080",
        "--policy",
        "round_robin",
    ]
    for endpoint in contract["connections"]["prefill"]:
        router_args += ["--prefill", endpoint["http"], "8998"]
    for endpoint in contract["connections"]["decode"]:
        router_args += ["--decode", endpoint["http"]]
    require(
        contract["router"]["command"] == ["python3", "-m", "sglang_router.launch_router"]
        and contract["router"]["args"] == router_args
        and contract["router"]["source_commit"] == COMMIT
        and contract["router"]["image"] == "REPLACE_ROUTER_IMAGE@sha256:REPLACE_ROUTER_DIGEST",
        "incorrect router flags/source or unresolved runtime identity",
    )
    require(
        contract["router"]["request_handoff"]
        == ["bootstrap_host", "bootstrap_port", "bootstrap_room"],
        "wrong handoff contract",
    )
    router_name = name + "-router"
    routers = [
        x for x in items if x["kind"] == "Deployment" and x["metadata"]["name"] == router_name
    ]
    router_services = [
        x for x in items if x["kind"] == "Service" and x["metadata"]["name"] == router_name
    ]
    require(len(routers) == len(router_services) == 1, "one router and service required")
    router = routers[0]["spec"]
    require(
        type(router["replicas"]) is int and router["replicas"] == 0,
        "active router replicas forbidden",
    )
    labels = {"app": router_name, "task13-role": "router"}
    require(
        router["selector"]["matchLabels"] == router["template"]["metadata"]["labels"] == labels,
        "router selector mismatch",
    )
    require(
        router["template"]["spec"]
        == {
            "containers": [
                {
                    "name": "router",
                    "image": contract["router"]["image"],
                    "command": contract["router"]["command"],
                    "args": router_args,
                    "ports": [{"name": "http", "containerPort": 30080}],
                }
            ]
        },
        "router pod/endpoint mismatch",
    )
    require(
        router_services[0]["spec"]
        == {"selector": labels, "ports": [{"name": "http", "port": 30080, "targetPort": 30080}]},
        "router service mismatch",
    )
    deployments = [
        x for x in items if x["kind"] == "Deployment" and x["metadata"]["name"] != router_name
    ]
    services = {
        x["metadata"]["name"]: x
        for x in items
        if x["kind"] == "Service" and x["metadata"]["name"] != router_name
    }
    expected = {
        name + "-prefill-1": ("prefill", 1 if name == "pd-2p1d" else 2),
        name + "-decode-1": ("decode", 1),
    }
    if name == "pd-2p1d":
        expected[name + "-prefill-2"] = ("prefill", 1)
    require(
        {d["metadata"]["name"] for d in deployments} == set(services) == set(expected),
        "wrong role/worker/service count",
    )
    connections = {"prefill": [], "decode": []}
    total = 0
    for d in deployments:
        worker = d["metadata"]["name"]
        role, tp = expected[worker]
        spec = d["spec"]
        require(
            type(spec["replicas"]) is int and spec["replicas"] == 0, "active replicas forbidden"
        )
        labels = {"app": worker, "task13-role": role}
        require(
            spec["selector"]["matchLabels"] == spec["template"]["metadata"]["labels"] == labels,
            "selector/role mismatch",
        )
        pod = spec["template"]["spec"]
        require(
            not pod.get("hostNetwork", False) and not pod.get("initContainers"),
            "unexpected executable/network surface",
        )
        require(len(pod["containers"]) == 1, "one engine required")
        c = pod["containers"][0]
        require(
            c["image"] == "REPLACE_SGLANG_IMAGE@sha256:REPLACE_IMAGE_DIGEST",
            "unverified or unpinned image identity",
        )
        require(c["command"] == ["python3", "-m", "sglang.launch_server"], "wrong entrypoint")
        args = c["args"]
        require(
            len(args) % 2 == 0 and len(set(args[::2])) == len(args[::2]), "duplicate/broken flags"
        )
        actual = dict(zip(args[::2], args[1::2], strict=True))
        wanted = {
            "--model-path": "REPLACE_LOCAL_MODEL_PATH",
            "--revision": "REPLACE_MODEL_REVISION",
            "--host": "0.0.0.0",
            "--port": "30000",
            "--tensor-parallel-size": str(tp),
            "--dtype": "bfloat16",
            "--disaggregation-mode": role,
            "--disaggregation-transfer-backend": "mooncake",
            "--disaggregation-ib-device": "REPLACE_RDMA_DEVICE",
        }
        ports = [{"name": "http", "containerPort": 30000}]
        if role == "prefill":
            wanted.update(
                {"--disaggregation-decode-tp": "1", "--disaggregation-bootstrap-port": "8998"}
            )
            ports.append({"name": "bootstrap", "containerPort": 8998})
            connections[role].append(
                {"http": f"http://{worker}:30000", "bootstrap": f"{worker}:8998"}
            )
        else:
            wanted["--disaggregation-prefill-pp"] = "1"
            connections[role].append({"http": f"http://{worker}:30000"})
        require(actual == wanted, "incorrect server flags/role/TP/bootstrap contract")
        require(c["ports"] == ports, "container ports mismatch")
        require(
            c["resources"]
            == {"requests": {"nvidia.com/gpu": tp}, "limits": {"nvidia.com/gpu": tp}},
            "GPU allocation/TP mismatch",
        )
        total += tp
        node = "REPLACE_PREFILL_NODE" if role == "prefill" else "REPLACE_DECODE_NODE"
        if name != "pd-cross-node":
            node = "REPLACE_SHARED_NODE"
        require(pod["nodeSelector"] == {"kubernetes.io/hostname": node}, "node placement mismatch")
        require(
            c["env"] == [{"name": "HF_HUB_OFFLINE", "value": "1"}], "offline model access required"
        )
        require(
            c["volumeMounts"]
            == [{"name": "model", "mountPath": "REPLACE_LOCAL_MODEL_PATH", "readOnly": True}],
            "model mount mismatch",
        )
        require(
            pod["volumes"]
            == [
                {
                    "name": "model",
                    "hostPath": {"path": "REPLACE_LOCAL_MODEL_PATH", "type": "Directory"},
                }
            ],
            "model path mismatch",
        )
        service = services[worker]["spec"]
        require(
            service
            == {
                "clusterIP": "None",
                "selector": labels,
                "ports": [
                    {
                        "name": p["name"],
                        "port": p["containerPort"],
                        "targetPort": p["containerPort"],
                    }
                    for p in ports
                ],
            },
            "individual worker/bootstrap service mismatch",
        )
    require(total == comparison["total_gpus"], "comparison GPU count mismatch")
    require(connections == contract["connections"], "wrong prefill/decode/bootstrap endpoint")
    if execution:
        raise ValueError(
            "execution rejected: unresolved images/model/router/node/RDMA; native gates pending"
        )
    return {
        "layout": name,
        "offline_valid": True,
        "execution_ready": False,
        "total_gpus": total,
        "physical_validation": contract["physical_validation"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout", type=Path)
    parser.add_argument("--execution-ready", action="store_true")
    args = parser.parse_args()
    try:
        result = validate(yaml.safe_load(args.layout.read_text()), execution=args.execution_ready)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError) as error:
        parser.exit(1, f"invalid layout: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
