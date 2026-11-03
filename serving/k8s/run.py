"""Task 14 on Kubernetes (kind, no GPU): static vs autoscaled trials of one step burst.

The cluster (`crucible-t14`) and the objects from serving/k8s/manifests.py must exist
(see docs/evidence/t14-kind.md for the commands). kubectl is the one inside the kind
node container, run with that cluster's /etc/kubernetes/admin.conf; nothing reads
~/.kube/config.

Per trial: scale StatefulSet `fake` to 1 and wait until only fake-0 serves, (re)create
the autoscaler (`auto`: -apply against statefulsets/fake/scale; `static`: dry-run,
proposals only), replay the Azure-length step burst from the host through the router
NodePort, then collect autoscaler/router logs and the pods' event logs.

Backends are CPU stand-ins (serving/k8s/fake_vllm.py), so latencies are synthetic.
What is real: the API-server scale write, StatefulSet controller, scheduler, kubelet
pod start, readiness, the adapter and router noticing the new pod, and the drain on
scale-down.
"""

import argparse
import datetime
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from serving.k8s import manifests as mf
from serving.localscale import analyze, trace

NODE = "crucible-t14-control-plane"
KUBECTL = ["docker", "exec", "-i", NODE, "kubectl", "--kubeconfig=/etc/kubernetes/admin.conf", "-n", mf.NS]
ROUTER_URL = "http://127.0.0.1:31380/v1/completions"
ADAPTER_URL = "http://127.0.0.1:31390"
TENANT = "bench"


def kubectl(*args, stdin=None, check=True):
    r = subprocess.run([*KUBECTL, *args], input=stdin, capture_output=True, text=True, timeout=120, check=False)
    if check and r.returncode:
        raise RuntimeError(f"kubectl {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def get_json(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status, json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, None
    except OSError:
        return None, None


def wait_for(check, timeout, what):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if check():
            return
        time.sleep(1)
    raise RuntimeError(f"timed out waiting for {what}")


def fake_pods():
    pods = json.loads(kubectl("get", "pods", "-l", "app=fake", "-o", "json"))["items"]
    out = {}
    for p in pods:
        ready = any(c["type"] == "Ready" and c["status"] == "True" for c in p["status"].get("conditions", []))
        out[p["metadata"]["name"]] = {"ready": ready, "deleting": "deletionTimestamp" in p["metadata"]}
    return out


def k8s_time(stamp):
    return datetime.datetime.fromisoformat(stamp).timestamp()


def autoscaler_decisions(log_text):
    """`kubectl logs --timestamps` lines: '<RFC3339Nano> {"current":..,"want":..}'."""
    out = []
    for line in log_text.splitlines():
        stamp, _, rest = line.partition(" ")
        if rest.startswith("{"):
            d = json.loads(rest)
            out.append({"t": k8s_time(stamp), **d})
    return out


def pod_events(log_dir, since):
    events = []
    for path in sorted(Path(log_dir).glob("*.jsonl")):
        pod = path.name.split("-")[0] + "-" + path.name.split("-")[1] if path.name.startswith("fake-") else "adapter"
        for line in path.read_text().splitlines():
            e = json.loads(line)
            if e["t"] >= since:
                events.append({**e, "pod": pod, "file": path.name})
    return sorted(events, key=lambda e: e["t"])


def reactions(events, decisions, start, burst_s, drop_s):
    """Seconds from the scheduled burst start (or load drop) to each scale milestone."""
    burst, drop = start + burst_s, start + drop_s

    def first(pred, after=0.0):
        return next((e["t"] for e in events if e["t"] >= after and pred(e)), None)

    def rel(t, origin):
        return None if t is None else round(t - origin, 3)

    up = next((d for d in decisions if d["t"] >= start and d["want"] > d["current"]), None)
    down = next((d for d in decisions if d["t"] >= drop and d["want"] < d["current"]), None)
    def new(kind):
        return lambda e: e.get("pod") == "fake-1" and e["event"] == kind

    return {
        "up_decision_s": rel(up and up["t"], burst),
        "up_applied": bool(up and up.get("applied")),
        "new_pod_process_start_s": rel(first(new("process_start"), start), burst),
        "new_pod_listening_s": rel(first(new("listening"), start), burst),
        "adapter_serving_new_pod_s": rel(first(lambda e: e.get("replica") == "fake-1" and e["event"] == "serving", start), burst),
        "new_pod_first_request_s": rel(first(new("first_request"), start), burst),
        "down_decision_after_drop_s": rel(down and down["t"], drop),
        "drain_start_after_drop_s": rel(first(new("drain_start"), drop), drop),
        "drained_after_drop_s": rel(first(new("drained"), drop), drop),
    }  # fmt: skip


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.halt, self.rows = threading.Event(), []

    def run(self):
        while not self.halt.wait(1.0):
            _, agg = get_json(ADAPTER_URL + "/aggregate")
            self.rows.append({"t": time.time(), "aggregate": agg})


def reset(s):
    kubectl("delete", "deployment", "autoscaler", "--ignore-not-found", "--wait=true")
    kubectl("scale", "statefulset", "fake", "--replicas=1")
    wait_for(lambda: list(fake_pods()) == ["fake-0"] and fake_pods()["fake-0"]["ready"], 300, "only fake-0 ready")
    wait_for(lambda: get_json(ADAPTER_URL + "/snapshot/fake-0")[0] == 200
             and all(get_json(f"{ADAPTER_URL}/snapshot/fake-{i}")[0] == 503 for i in range(1, s.max_replicas))
             and (get_json(ADAPTER_URL + "/aggregate")[1] or {}).get("replicas") == 1, 120, "adapter sees one replica")  # fmt: skip
    time.sleep(16)  # router snapshots of removed pods age out (FreshFor 15 s)


def run_trial(args, s, condition, out):
    out.mkdir(parents=True, exist_ok=True)
    reset(s)
    kubectl("apply", "-f", "-", stdin=mf.as_list(mf.autoscaler(s, apply=condition == "auto")))
    kubectl("rollout", "status", "deployment/autoscaler", "--timeout=120s")
    time.sleep(5)
    sampler = Sampler()
    sampler.start()
    load = {"nproc": os.cpu_count(), "loadavg_start": os.getloadavg()}
    start = time.time()
    cmd = [
        args.replay_bin, "-trace", str(args.trace_csv), "-format", "azure", "-url", ROUTER_URL, "-mult", "1",
        "-token-ids", args.token_ids, "-timeout", "60s", "-inflight", "256", "-seed", "17",
        "-api", "openai-text", "-model", s.model, "-tenant", TENANT, "-account",
    ]  # fmt: skip
    span = sum(sec for sec, _ in args.phases)
    with open(out / "replay.json", "wb") as stdout:
        proc = subprocess.run(cmd, stdout=stdout, stderr=subprocess.PIPE, timeout=span + 180, check=False)
    (out / "replay.stderr").write_bytes(proc.stderr[:65536])
    if condition == "auto":  # let a pending scale-down and drain finish
        deadline = time.time() + args.down_wait_s
        while time.time() < deadline and list(fake_pods()) != ["fake-0"]:
            time.sleep(2)
    sampler.halt.set()
    load["loadavg_end"] = os.getloadavg()
    auto_log = kubectl("logs", "deployment/autoscaler", "--timestamps")
    (out / "autoscaler.log").write_text(auto_log)
    (out / "router.log").write_text(kubectl("logs", "deployment/router", "--timestamps", "--since=20m"))
    events = pod_events(args.log_dir, start - 600)
    (out / "pod-events.json").write_text(json.dumps(events))
    (out / "timeline.json").write_text(json.dumps(sampler.rows))
    decisions = autoscaler_decisions(auto_log)
    (out / "decisions.json").write_text(json.dumps(decisions))
    result = json.loads((out / "replay.json").read_text())
    burst_s, drop_s = args.phases[0][0], args.phases[0][0] + args.phases[1][0]
    return {
        "condition": condition, "start": start, "replay_returncode": proc.returncode, **load,
        "sender_valid": bool(result["Run"]["SenderValid"]), "accounting": result.get("Accounting"),
        "records": result["Run"]["Records"],
        "reactions": reactions(events, decisions, start, burst_s, drop_s),
        "max_desired": max(((r["aggregate"] or {}).get("replicas", 0) for r in sampler.rows), default=None),
    }  # fmt: skip


def summarize(trials, phases):
    slo = analyze.slo_from_base_phase([t["records"] for t in trials if t["condition"] == "static"], phases)
    rows = []
    for t in trials:
        rows.append({k: v for k, v in t.items() if k != "records"} | {"phases": analyze.phase_summary(t["records"], phases, slo)})
    return {"slo": slo, "trials": rows}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--phases", type=lambda t: [(float(a), float(b)) for a, b in (x.split(":") for x in t.split(","))],
                   default=[(60.0, 1.0), (120.0, 4.0), (360.0, 1.0)])  # fmt: skip
    p.add_argument("--base-rps", type=float, default=1.0)
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--down-wait-s", type=float, default=180)
    p.add_argument("--replay-bin", default=".cache/t14/bin/replay")
    p.add_argument("--token-ids", default=".cache/t14/qwen25-05b-words.json")
    p.add_argument("--log-dir", default=".cache/k8s/logs")
    args = p.parse_args(argv)
    s = mf.Settings()
    args.output.mkdir(parents=True, exist_ok=True)
    lengths = trace.read_lengths("serving/baseline/traces/azure-60s/calibration.csv") + trace.read_lengths(
        "serving/baseline/traces/azure-60s/evaluation.csv"
    )
    args.trace_csv = args.output / "trace.csv"
    trace.write_azure_csv(trace.step_burst(lengths, args.phases, args.base_rps, seed=17), args.trace_csv)
    order = []
    for pair in range(args.pairs):
        order += ["static", "auto"] if pair % 2 == 0 else ["auto", "static"]
    trials = []
    for index, condition in enumerate(order):
        t = run_trial(args, s, condition, args.output / f"trial-{index}-{condition}")
        t["index"] = index
        trials.append(t)
        print(json.dumps({"trial": index, "condition": condition, "reactions": t["reactions"]}), flush=True)
    summary = summarize(trials, args.phases)
    summary.update(order=order, phases=args.phases, base_rps=args.base_rps, settings=vars(s))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
