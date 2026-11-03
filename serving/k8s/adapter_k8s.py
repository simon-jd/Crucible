"""In-cluster metrics adapter for the kind Task 14 deployment.

Same HTTP contract as serving/localscale/adapter.py (`/snapshot/<id>` for the Go
router, `/aggregate` for the Go autoscaler), reusing its State. The differences:
replicas are StatefulSet pods `<set>-<i>` with stable DNS, a replica counts as
serving while its /metrics scrape succeeds (a draining or deleted pod fails it),
and the desired count is the StatefulSet's spec.replicas, read from the API
server's scale subresource with the pod's service-account token every poll.
"""

import argparse
import json
import ssl
import threading
import time
import urllib.request

from serving.localscale import adapter, metrics

SA = "/var/run/secrets/kubernetes.io/serviceaccount"


def replica_urls(statefulset, service, namespace, max_replicas, port):
    return {
        f"{statefulset}-{i}": f"http://{statefulset}-{i}.{service}.{namespace}.svc.cluster.local:{port}"
        for i in range(max_replicas)
    }


def new_state(model, urls):
    return adapter.State(model, urls, desired=1)


def parse_scale(body):
    scale = json.loads(body)
    if scale.get("kind") != "Scale":
        raise ValueError("not a Scale object")
    return int(scale["spec"]["replicas"])


def poll_once(state, scrape, desired, now):
    for name, url in state.replicas.items():
        try:
            text = scrape(url + "/metrics")
        except OSError:
            if state.serving[name]:
                state.forget(name)
            continue
        state.record(name, metrics.parse(text), now)
        with state.lock:
            state.serving[name] = True
    try:
        want = desired()
    except (OSError, ValueError, KeyError):
        return
    with state.lock:
        state.desired = max(1, want)


def http_scrape(url):
    with urllib.request.urlopen(url, timeout=2) as resp:
        return resp.read(4 << 20).decode()


def scale_reader(namespace, statefulset):
    ctx = ssl.create_default_context(cafile=f"{SA}/ca.crt")
    url = f"https://kubernetes.default.svc/apis/apps/v1/namespaces/{namespace}/statefulsets/{statefulset}/scale"

    def read():
        with open(f"{SA}/token") as f:
            token = f.read().strip()
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=2, context=ctx) as resp:
            return parse_scale(resp.read(1 << 20))

    return read


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--namespace", default="crucible")
    p.add_argument("--statefulset", default="fake")
    p.add_argument("--service", default="fake")
    p.add_argument("--max-replicas", type=int, default=3)
    p.add_argument("--replica-port", type=int, default=8000)
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--log-file", help="also append JSON event lines here (hostPath in kind)")
    args = p.parse_args(argv)
    log_file = open(args.log_file, "a") if args.log_file else None  # noqa: SIM115 - process lifetime

    def log(**fields):
        line = json.dumps({"t": time.time(), **fields})
        print(line, flush=True)
        if log_file:
            log_file.write(line + "\n")
            log_file.flush()
    urls = replica_urls(args.statefulset, args.service, args.namespace, args.max_replicas, args.replica_port)
    state = new_state(args.model, urls)
    desired = scale_reader(args.namespace, args.statefulset)
    server = adapter.serve(state, args.port, host="0.0.0.0")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log(event="listening", port=args.port)
    serving = {}
    while True:
        poll_once(state, http_scrape, desired, time.time())
        with state.lock:
            now_serving = dict(state.serving)
        for name, up in now_serving.items():
            if serving.get(name) != up:
                log(event="serving" if up else "not_serving", replica=name, desired=state.desired)
        serving = now_serving
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
