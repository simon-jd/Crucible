"""Kubernetes objects for the Task 14 kind deployment, rendered as JSON (valid YAML).

Topology (namespace `crucible`): StatefulSet `fake` (CPU vLLM stand-ins, stable DNS
fake-<i>.fake), the metrics adapter, the Go router (NodePort 30080) and the Go
autoscaler. The router lists every possible replica fake-0..max-1; a pod that does
not exist or is not serving answers no snapshot, so the router does not pick it.

No images are built, pulled or loaded: pods use the kube-proxy image that the kind
node already holds, only as a glibc root filesystem. The project's standalone
CPython (.tools/python), project code and the static Go binaries reach the node
through kind extraMounts (read-only), and pods write event logs to a mounted
project directory as the invoking uid, so the host can read and delete them.
"""

import json
from dataclasses import dataclass

NS = "crucible"
IMAGE = "registry.k8s.io/kube-proxy:v1.34.11"  # preloaded in kindest/node:v1.34.11
PYTHON = "/crucible-python/bin/python3"
SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"


@dataclass
class Settings:
    max_replicas: int = 3
    uid: int = 1000
    gid: int = 1000
    retry_dial: bool = False
    max_num_seqs: int = 4
    tpot_ms: float = 5.0
    prefill_ms_per_token: float = 0.05
    startup_s: float = 0.0
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    router_port: int = 30080
    adapter_port: int = 30090


def _meta(name, **labels):
    return {"name": name, "namespace": NS, "labels": {"app.kubernetes.io/part-of": "crucible", **labels}}


def _volumes():
    return [
        {"name": "src", "hostPath": {"path": "/crucible/serving", "type": "Directory"}},
        {"name": "bin", "hostPath": {"path": "/crucible/bin", "type": "Directory"}},
        {"name": "logs", "hostPath": {"path": "/crucible/logs", "type": "Directory"}},
        {"name": "python", "hostPath": {"path": "/crucible/python", "type": "Directory"}},
    ]


def _mounts():
    return [
        {"name": "src", "mountPath": "/src/serving", "readOnly": True},
        {"name": "bin", "mountPath": "/crucible-bin", "readOnly": True},
        {"name": "logs", "mountPath": "/logs"},
        {"name": "python", "mountPath": "/crucible-python", "readOnly": True},
    ]


def _pod(s, containers, account=None, grace=30, extra_volumes=()):
    spec = {
        "securityContext": {"runAsUser": s.uid, "runAsGroup": s.gid, "runAsNonRoot": True},
        "terminationGracePeriodSeconds": grace,
        "containers": containers,
        "volumes": _volumes() + list(extra_volumes),
        "automountServiceAccountToken": account is not None,
    }
    if account:
        spec["serviceAccountName"] = account
    return spec


def _container(name, command, port=None, env=(), mounts=(), probe=None):
    c = {
        "name": name, "image": IMAGE, "imagePullPolicy": "Never", "command": command, "workingDir": "/src",
        "env": [{"name": "PYTHONDONTWRITEBYTECODE", "value": "1"}, *env],
        "volumeMounts": _mounts() + list(mounts),
        "resources": {"requests": {"cpu": "100m", "memory": "64Mi"}, "limits": {"memory": "512Mi"}},
    }  # fmt: skip
    if port:
        c["ports"] = [{"containerPort": port}]
    if probe:
        c["readinessProbe"] = {"httpGet": {"path": probe, "port": port}, "periodSeconds": 1, "failureThreshold": 1}
    return c


def _deployment(name, pod):
    return {
        "apiVersion": "apps/v1", "kind": "Deployment", "metadata": _meta(name, app=name),
        "spec": {"replicas": 1, "selector": {"matchLabels": {"app": name}},
                 "template": {"metadata": {"labels": {"app": name}}, "spec": pod}},
    }  # fmt: skip


def _service(name, port, node_port=None, headless=False):
    spec = {"selector": {"app": name}, "ports": [{"name": "http", "port": port, "targetPort": port}]}
    if headless:
        spec["clusterIP"] = "None"
    if node_port:
        spec["type"] = "NodePort"
        spec["ports"][0]["nodePort"] = node_port
    return {"apiVersion": "v1", "kind": "Service", "metadata": _meta(name), "spec": spec}


def _role(name, rules, account):
    return [
        {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": _meta(account)},
        {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "Role", "metadata": _meta(name), "rules": rules},
        {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "RoleBinding", "metadata": _meta(name),
         "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": name},
         "subjects": [{"kind": "ServiceAccount", "name": account, "namespace": NS}]},
    ]  # fmt: skip


def backend_url(i):
    return f"http://fake-{i}.fake.{NS}.svc.cluster.local:8000"


def core(s):
    """Everything except the autoscaler, which each trial (re)creates."""
    pod_env = [{"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
               {"name": "POD_UID", "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}}}]  # fmt: skip
    fake_cmd = [
        PYTHON, "-u", "/src/serving/k8s/fake_vllm.py", "--port", "8000", "--model", s.model,
        "--max-num-seqs", str(s.max_num_seqs), "--tpot-ms", str(s.tpot_ms),
        "--prefill-ms-per-token", str(s.prefill_ms_per_token), "--startup-s", str(s.startup_s),
        "--log-file", "/logs/$(POD_NAME)-$(POD_UID).jsonl",
    ]  # fmt: skip
    fake = {
        "apiVersion": "apps/v1", "kind": "StatefulSet", "metadata": _meta("fake", app="fake"),
        "spec": {
            "serviceName": "fake", "replicas": 1, "podManagementPolicy": "Parallel",
            "selector": {"matchLabels": {"app": "fake"}},
            "template": {"metadata": {"labels": {"app": "fake"}},
                         "spec": _pod(s, [_container("fake", fake_cmd, 8000, pod_env, probe="/health")], grace=60)},
        },
    }  # fmt: skip
    adapter_cmd = [
        PYTHON, "-u", "-m", "serving.k8s.adapter_k8s", "--namespace", NS, "--statefulset", "fake",
        "--service", "fake", "--max-replicas", str(s.max_replicas), "--port", "9000", "--model", s.model,
        "--log-file", "/logs/adapter-$(POD_UID).jsonl",
    ]  # fmt: skip
    adapter = _pod(s, [_container("adapter", adapter_cmd, 9000, pod_env)], account="adapter")
    router_cfg = {
        "Scope": {"Model": s.model, "Revision": "kind-fake", "Adapter": "base"},
        "Backends": [{"ID": f"fake-{i}", "URL": backend_url(i),
                      "MetricsURL": f"http://adapter.{NS}.svc.cluster.local:9000/snapshot/fake-{i}"}
                     for i in range(s.max_replicas)],
    }  # fmt: skip
    router_cmd = ["/crucible-bin/router", "-config", "/cfg/router.json", "-listen", "0.0.0.0:8080"]
    if s.retry_dial:
        router_cmd.append("-retry-dial")
    router_pod = _pod(
        s, [_container("router", router_cmd, 8080, mounts=[{"name": "cfg", "mountPath": "/cfg", "readOnly": True}])],
        extra_volumes=[{"name": "cfg", "configMap": {"name": "router-config"}}],
    )  # fmt: skip
    return [
        {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": NS, "labels": {"app.kubernetes.io/part-of": "crucible"}}},
        *_role("adapter", [{"apiGroups": ["apps"], "resources": ["statefulsets/scale"], "resourceNames": ["fake"], "verbs": ["get"]}], "adapter"),
        _service("fake", 8000, headless=True),
        fake,
        _deployment("adapter", adapter),
        _service("adapter", 9000, node_port=s.adapter_port),
        {"apiVersion": "v1", "kind": "ConfigMap", "metadata": _meta("router-config"), "data": {"router.json": json.dumps(router_cfg)}},
        _deployment("router", router_pod),
        _service("router", 8080, node_port=s.router_port),
    ]  # fmt: skip


def autoscaler(s, apply, interval="2s"):
    cmd = [
        "/crucible-bin/autoscaler", "-metrics-url", f"http://adapter.{NS}.svc.cluster.local:9000/aggregate",
        "-interval", interval, "-min", "1", "-max", str(s.max_replicas), "-queue-high", "4", "-queue-low", "1",
        "-ttft-high", "0.8", "-ttft-low", "0.2",
    ]  # fmt: skip
    objs = []
    account = None
    if apply:
        cmd += ["-apply", "-api-server", "https://kubernetes.default.svc", "-namespace", NS, "-deployment", "fake",
                "-resource", "statefulsets", "-token-file", f"{SA_DIR}/token", "-ca-file", f"{SA_DIR}/ca.crt"]  # fmt: skip
        account = "autoscaler"
        objs += _role("autoscaler", [{"apiGroups": ["apps"], "resources": ["statefulsets/scale"], "resourceNames": ["fake"], "verbs": ["get", "update"]}], account)  # fmt: skip
    objs.append(_deployment("autoscaler", _pod(s, [_container("autoscaler", cmd)], account=account)))
    return objs


def kind_config(repo, s):
    return {
        "kind": "Cluster", "apiVersion": "kind.x-k8s.io/v1alpha4",
        "nodes": [{
            "role": "control-plane",
            "extraMounts": [
                {"hostPath": f"{repo}/serving", "containerPath": "/crucible/serving", "readOnly": True},
                {"hostPath": f"{repo}/.cache/k8s/bin", "containerPath": "/crucible/bin", "readOnly": True},
                {"hostPath": f"{repo}/.cache/k8s/logs", "containerPath": "/crucible/logs"},
                {"hostPath": f"{repo}/.tools/python/cpython-3.12.13-linux-x86_64-gnu", "containerPath": "/crucible/python", "readOnly": True},
            ],
            "extraPortMappings": [
                {"containerPort": s.router_port, "hostPort": 31380, "listenAddress": "127.0.0.1"},
                {"containerPort": s.adapter_port, "hostPort": 31390, "listenAddress": "127.0.0.1"},
            ],
        }],
    }  # fmt: skip


def as_list(objs):
    return json.dumps({"apiVersion": "v1", "kind": "List", "items": objs}, indent=1)
