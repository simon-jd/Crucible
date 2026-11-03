# Task 14 on Kubernetes (kind) — 2026-10-27, blocked by the host inotify limit

Result: not run. The kind node started, but its kube-proxy exits with
`failed complete: too many open files` and crash-loops
([log](t14-kind/kube-proxy.log), [pods](t14-kind/pods.txt)). Without kube-proxy there
is no ClusterIP routing, so CoreDNS cannot reach the API server at 10.96.0.1
([log](t14-kind/coredns.log)) and no pod resolves a Service or pod DNS name. The brief
names this failure: the host inotify instance limit (about 120 of 128 used by other
projects) is nearly exhausted, and raising it needs root. It is in the STATUS
blockers list. The cluster was deleted. The other project's kind cluster was not touched.

## What is ready (code + offline tests)

- `serving/k8s/manifests.py`: namespace `crucible`; StatefulSet `fake` (stable DNS
  fake-<i>.fake) of CPU vLLM stand-ins; metrics adapter; Go router (NodePort 30080,
  config lists fake-0..2); Go autoscaler as a Deployment. In `auto` it runs with `-apply`
  against `statefulsets/fake/scale`, and its Role allows only `get`/`update` on that one
  scale subresource. No image is built, pulled or loaded. Pods use the
  `kube-proxy:v1.34.11` image already in `kindest/node:v1.34.11` as a glibc root
  filesystem. The project's standalone CPython, code and static Go binaries come in
  read-only through kind extraMounts. (`kind load docker-image python:3.12.13-slim`
  failed with a containerd digest error on the multi-platform image, and `docker save`
  is outside the tool allowlist.)
- `serving/k8s/fake_vllm.py`: streams `/v1/completions` in the format the Go
  replayer's `-api openai-text` checks, and exposes vLLM 0.10.2 metric names.
  Synthetic latency: 4 slots, 5 ms per token, 0.05 ms per prompt word. It drains on SIGTERM.
- `serving/k8s/adapter_k8s.py`: same contract as the T14 local adapter. A replica counts
  as serving while its scrape succeeds. The desired count comes from the StatefulSet
  scale subresource.
- `serving/k8s/run.py`: static vs auto, 3 alternating pairs, the T14 Azure-length step
  burst (60 s 1×, 120 s 4×, 360 s 1×, 1 req/s base), replayed from the host. Reaction
  milestones: decision, new pod process start, adapter sees it, first request, drain.
- Autoscaler: new `-resource statefulsets` (`ScaleClient.Resource`).
- Tests: `serving/k8s/tests/` (20) and `serving/autoscaler/resource_test.go`.

## Commands (once the limit allows a working kind node)

```sh
make k8s-bins
.tools/kind/kind create cluster --name crucible-t14 --kubeconfig .cache/kind/kubeconfig \
  --image kindest/node:v1.34.11@sha256:44e222ee2132dab25ff87301682f89eb82c7880ea3a1bf543bfe9708fd08d67d \
  --config .cache/k8s/kind-config.json    # rendered by manifests.kind_config
docker exec -i crucible-t14-control-plane kubectl --kubeconfig=/etc/kubernetes/admin.conf apply -f - < .cache/k8s/core.json
.venv/bin/python -m serving.k8s.run --output docs/evidence/t14-kind/run
.tools/kind/kind delete cluster --name crucible-t14 --kubeconfig .cache/kind/kubeconfig
```

kubectl is the one inside the kind node, run against that node's own
`/etc/kubernetes/admin.conf`. There is no kubectl on the host, and its download host
(dl.k8s.io) is not on the allowed list. kind v0.33.0 came from its GitHub release, sha256
`aee61515…8e0d`, matching the published `.sha256sum`. The node image digest matches
the one kind v0.33.0 lists for v1.34.11.

Even when it runs, the backends are synthetic: GPUs do not pass into kind. The real part
is the Kubernetes control path: the scale write, StatefulSet controller, scheduling, pod
start, readiness, the router noticing the new pod, and the drain.
