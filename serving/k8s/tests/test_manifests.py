import json

from serving.k8s import manifests as mf


def by_kind(objs):
    out = {}
    for o in objs:
        out.setdefault(o["kind"], {})[o["metadata"]["name"]] = o
    return out


def test_core_objects_live_in_the_project_namespace():
    objs = mf.core(mf.Settings())
    kinds = by_kind(objs)
    assert set(kinds["Namespace"]) == {"crucible"}
    for o in objs:
        if o["kind"] != "Namespace":
            assert o["metadata"]["namespace"] == "crucible"
    sts = kinds["StatefulSet"]["fake"]
    assert sts["spec"]["serviceName"] == "fake" and kinds["Service"]["fake"]["spec"]["clusterIP"] == "None"
    pod = sts["spec"]["template"]["spec"]
    assert pod["securityContext"]["runAsUser"] == 1000
    container = pod["containers"][0]
    assert container["imagePullPolicy"] == "Never" and container["image"] == mf.IMAGE
    assert container["command"][0] == mf.PYTHON
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"


def test_router_config_lists_every_possible_replica_with_its_adapter_snapshot():
    s = mf.Settings(max_replicas=3)
    cm = by_kind(mf.core(s))["ConfigMap"]["router-config"]
    cfg = json.loads(cm["data"]["router.json"])
    ids = [b["ID"] for b in cfg["Backends"]]
    assert ids == ["fake-0", "fake-1", "fake-2"]
    assert cfg["Backends"][1]["URL"] == "http://fake-1.fake.crucible.svc.cluster.local:8000"
    assert cfg["Backends"][1]["MetricsURL"] == "http://adapter.crucible.svc.cluster.local:9000/snapshot/fake-1"


def test_router_retry_flag_and_nodeport():
    kinds = by_kind(mf.core(mf.Settings(retry_dial=True)))
    args = kinds["Deployment"]["router"]["spec"]["template"]["spec"]["containers"][0]["command"]
    assert "-retry-dial" in args
    assert kinds["Service"]["router"]["spec"]["ports"][0]["nodePort"] == 30080


def test_autoscaler_apply_targets_the_statefulset_scale_with_least_privilege():
    objs = mf.autoscaler(mf.Settings(), apply=True)
    kinds = by_kind(objs)
    cmd = kinds["Deployment"]["autoscaler"]["spec"]["template"]["spec"]["containers"][0]["command"]
    assert "-apply" in cmd and cmd[cmd.index("-resource") + 1] == "statefulsets"
    assert cmd[cmd.index("-deployment") + 1] == "fake"
    rules = kinds["Role"]["autoscaler"]["rules"]
    assert rules == [{"apiGroups": ["apps"], "resources": ["statefulsets/scale"], "resourceNames": ["fake"], "verbs": ["get", "update"]}]


def test_autoscaler_dry_run_has_no_apply_and_no_write_role():
    objs = mf.autoscaler(mf.Settings(), apply=False)
    cmd = by_kind(objs)["Deployment"]["autoscaler"]["spec"]["template"]["spec"]["containers"][0]["command"]
    assert "-apply" not in cmd
    assert "Role" not in by_kind(objs)


def test_kind_config_mounts_code_read_only_and_maps_ports_to_loopback():
    cfg = mf.kind_config("/repo", mf.Settings())
    node = cfg["nodes"][0]
    mounts = {m["containerPath"]: m for m in node["extraMounts"]}
    assert mounts["/crucible/serving"]["hostPath"] == "/repo/serving" and mounts["/crucible/serving"]["readOnly"]
    assert mounts["/crucible/logs"]["hostPath"] == "/repo/.cache/k8s/logs"
    assert mounts["/crucible/python"]["readOnly"]
    ports = {p["containerPort"]: p for p in node["extraPortMappings"]}
    assert ports[30080]["listenAddress"] == "127.0.0.1"
