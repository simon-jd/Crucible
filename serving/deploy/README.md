# Task 14 serving contracts and offline deployment

This is an independent stdlib Go implementation. The named `cache-aware-routing-lab` source was not available; no source port or upstream license provenance is claimed. No Kubernetes context, model server, object store, or production metrics adapter was accessed. `deployment.yaml` is deliberately incomplete and **has not been applied**. Replace all `REPLACE_*` values with verified image digests, content SHA256, model revision and real adapter URLs. Namespace, NVMe provisioning, scheduling, server readiness probes, services and trusted ingress must be supplied for the target environment. One available GPU cannot demonstrate multiple live GPU replicas; the candidate manifest caps the controller at one.

Build without network or home cache writes, from the repository root:

```bash
mkdir -p .cache/task14/{tmp,bin}
export GOCACHE="$PWD/.cache/task14/build" GOPATH="$PWD/.cache/task14/go"
export GOMODCACHE="$PWD/.cache/task14/mod" GOTMPDIR="$PWD/.cache/task14/tmp"
export TMPDIR="$GOTMPDIR" GOENV=off GOTOOLCHAIN=local GOPROXY=off GOMAXPROCS=2
.tools/go/bin/go build -p=1 -o .cache/task14/bin/router ./serving/router/cmd/router
.tools/go/bin/go build -p=1 -o .cache/task14/bin/autoscaler ./serving/autoscaler
.tools/go/bin/go build -p=1 -o .cache/task14/bin/experiments ./serving/experiments
```

The router starts with `router -config config.json`; its default listener is loopback. Example config (URLs are placeholders to provision, not native SGLang APIs):

```json
{"Scope":{"Model":"model","Revision":"immutable-revision","Adapter":"base"},"Backends":[{"ID":"one","URL":"http://127.0.0.1:8000","MetricsURL":"http://127.0.0.1:9000/router-snapshot"}]}
```

Each explicitly configured trusted adapter returns the following full snapshot every five seconds or faster. `ObservedAt` is RFC3339 UTC. The router rejects snapshots older than 15 seconds or in the future; an unsuccessful poll retains the prior snapshot only until its TTL. The adapter must invalidate its full prefix list when KV blocks are evicted or its process/model changes; refreshing a snapshot must not renew an unverified cache claim. Prefixes are hashed in router memory. They describe cached UTF-8 byte prefixes, not tokenizer block counts or measured KV hits. No prefix is learned from client claims or successful responses. Until a native backend adapter is built and measured these remain **adapter observations**, not confirmed GPU performance evidence.

```json
{"ObservedAt":"2026-10-24T12:00:00Z","Queue":2,"Prefixes":[{"Scope":{"Model":"model","Revision":"immutable-revision","Adapter":"base","Tenant":"tenant-a"},"Text":"shared prompt prefix"}]}
```

`Queue` is physical waiting depth. Selection pressure is `max(Queue, local outstanding reservations)` because requests may belong to both sets; this pressure floor is an estimate of load, not total queue length. Longest matching prefix wins, load breaks ties, and pressure strictly above four spills to the least-pressure fresh backend. Bounds: 256 backends; 1024 entries/backend; 4096 bytes/prefix; 1024 bytes for combined scope strings. A complete invalid snapshot is rejected atomically. Empty prefix lists clear affinity. Stale backends are ineligible.

The HTTP interface accepts `POST /generate` (`text` or `prompt` string) and `POST /v1/completions` (`prompt` string). A present `model` must match configuration. Chat messages and token arrays need a tokenizer-aware adapter and are rejected. Trusted ingress must authenticate callers and overwrite `X-Tenant`; do not expose the listener directly to untrusted clients. Tenant scope prevents cross-tenant affinity; it is not an authentication system. Client cache-hit/backend-URL control headers are stripped. Request bodies are limited to 1 MiB; admission to 64 simultaneous requests; the five-minute lifetime budget starts before body reads and covers upstream work and downstream socket writes. The handler requires Go net/http ResponseController read/write deadline support: middleware must preserve it directly or through `Unwrap() http.ResponseWriter`. Unsupported or failing deadline operations abort the exchange before admission/forwarding. Cancellation interrupts blocked reads/writes, then the callback is stopped or joined before handler return. Both deadlines remain through net/http post-handler body drain/final flush; net/http resets them for normal keepalive reuse. Timed-out/cancelled exchanges abort instead of reusing an incomplete connection. This contract is verified on the installed Go 1.27.3 HTTP/1 server, including a connected non-reading TCP client and reuse beyond the prior deadline. Streams flush immediately, cancellation reaches the backend, and a reservation is released once after completion/failure/cancellation. Failed POSTs are not replayed.

The autoscaler reads this explicit aggregate adapter contract:

```json
{"observed_at":"2026-10-24T12:00:00Z","queue_per_replica":5.0,"p95_ttft_seconds":0.9,"replicas":2}
```

The adapter must supply measured queue and p95 TTFT from the same fresh window, plus **observed desired replicas** (Scale.spec.replicas), not ready replicas. No samples must be represented as missing/invalid, not zero TTFT. Missing/null/nonfinite/negative metrics, future timestamps, and snapshots older than 15 seconds cannot trigger scaling. These default SLO thresholds are configurable examples; a real held-out workload must establish suitable values.

`autoscaler -metrics-url URL -once` prints a proposal; dry-run does not write Kubernetes or pretend to change replica count. Scale up by one when queue >4 OR p95 >0.8 seconds; scale down by one only when queue <1 AND p95 <0.2 seconds. The intervening band holds. First scale up is immediate; first scale down waits 300 seconds from policy construction. After an observed desired-replica change or checked scale ACK, up waits exactly >=60 seconds and down >=300 seconds. State is mutex-protected and process-local. Restart loses historical change timestamps, so first-up readiness and first-down grace restart; an HA controller needs shared durable ownership/state. Bounds apply to valid observations; invalid metrics hold current state even if outside configured bounds.

Future explicit application requires `-apply -api-server https://HOST -namespace NS -deployment NAME -token-file FILE [-ca-file FILE]`. There is no context discovery. Every tick GETs the explicit scale resource, then fetches metrics and requires their replica count to match. PUT carries the resourceVersion and UID; ACK must match kind, API version, target namespace/name, UID and desired count. Conflicts/transport errors/unknown ACKs do not consume a success cooldown. The next tick re-GETs current state, including after an uncertain write; there is no blind mutation retry. TLS verification is enabled, redirects disabled, and only loopback test endpoints may use HTTP for scale.

The NVMe candidate uses a node-local existing directory. Init fills it read/write using `experiments -mode stage -object-uri URL -destination FILE -sha256 HEX -max-bytes N`; the model mounts it read-only. The SHA256-named destination must not exist with different contents. Staging bounds bytes, verifies integrity, fsyncs and atomically links a temporary file; an existing matching artifact avoids download. It supports a single immutable artifact (a complete shard set needs a verified manifest adapter). Budget for existing cache plus the temporary download, including concurrent producers. Current execution uses only tiny project-owned fixtures; no model was downloaded, staged into the shared HF cache, or loaded.

Experiments:

```bash
.cache/task14/bin/autoscaler -simulate-burst -rounds 3
# Future, only after provisioning an approved adapter:
.cache/task14/bin/experiments -mode cold -adapter-url URL \
  -nvme-uri file:///mnt/nvme/crucible-models/IMMUTABLE_FILE \
  -object-uri https://APPROVED_OBJECT -sha256 CONTENT_SHA256 \
  -metadata 'image digest; model revision; hardware; identical reset/page-cache conditions' \
  -label real-backend -rounds 3
```

The burst runner injects a clock and alternates >=3 matched baseline/4x pairs. Logical reaction times are not physical scale readiness times. For physical burst validation, connect measured aggregate metrics, replay a held-out arrival trace at baseline and 4x intensity with >=3 alternating pairs, and record proposal, scale ACK and actual first-token readiness timestamps separately; no such replay/backend exists yet here.

The cold runner uses a **custom future protocol**, not a built-in SGLang API: `POST /start` with `{id,source,uri,sha256}`; `GET /ready?id=...`; `POST /stop` with the same trial. Every response must echo `{id,source,sha256}`; readiness additionally returns `ready:true`. The adapter must bind the hash to bytes actually loaded, stop the previous model, reset equivalent cache conditions, and make readiness mean a successful fresh inference. Timings include start-to-ready and use alternating object/NVMe order, >=3 pairs, median and range. The executable handles SIGINT/SIGTERM by cancelling its run context and unwinding into cleanup. Stop is attempted with a separate five-second context even after errors/timeouts/signals; cancellation exits with status 1 after the attempt. SIGKILL, host failure and externally killed cleanup cannot guarantee a stop attempt. A failed stop is reported as an error and must be treated as **unknown remote cleanup**, never as a stopped model. No default endpoint is provided. Local loopback tests measure only HTTP protocol overhead and cleanup.
