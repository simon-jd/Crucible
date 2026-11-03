# Native SGLang open-loop replay (CPU phase)

The Go package preserves `Replay([]Req, url, mult) []Record`; production callers
use `Run(ctx, trace, Config)` for explicit validation errors and run validity.
`Replay` uses fixture token IDs only; use the CLI/verified allowlist for real runs.
No GPU baseline or real SLO has been measured. CPU loopback results measure the
replay client's behavior, not SGLang performance.

## Contract and bounds

`ArrivalMs / mult` is an absolute offset from run start. Input arrivals must be
nonnegative, finite and sorted; equal arrivals remain simultaneous (no sorting,
smoothing or filtering). A multiplier changes arrival spacing, not token lengths.
Bounds: 10,000 requests, one-hour scaled horizon, 32,768 input+output tokens per
request, 16 MiB prepared payloads, 1–256 inflight requests, <=10-minute lifetime.
Trace input is <=4 MiB. Tighten lengths to the actual model context before running;
the importer never silently truncates source requests.

Payloads are built before the clock starts. Native `input_ids` length exactly
matches each trace input count. A fixed 64-bit LCG/seed selects IDs from the
allowlist; IDs are synthetic and do not recreate private prompts, semantic
quality, conversation reuse or production prefix locality. The deployment template
disables radix cache for the length-only baseline. Later cache experiments need
an explicit independently defined prefix policy and matched baseline. Zero-input
and zero-output trace rows remain present, are sent exactly, and may produce real
server errors; never substitute a prompt or count them as SLO successes.

A hard inflight bound rejects arrivals immediately with an explicit record.
It never blocks scheduling on completion. Each request's deadline is its schedule
plus timeout, including connection/write/read waits. Cancellation accounts for all
unsent rows and cancels active streams. The owned transport closes idle connections
and all response bodies close; there is no retry or redirect. No server process is
started by this package.

`SentMs` observes a successful complete request acceptance on the underlying
connection's `Write` return: a bounded <=64 KiB HTTP/1 header parser verifies
exact Content-Length and rejects Transfer-Encoding, then counts the prepared
JSON body. Failed/partial/unrecognized writes cannot establish emission. The
observer wraps plaintext above the standard verified TLS handshake; ALPN is
restricted to HTTP/1.1, HTTP/2 and proxy use are disabled. Ownership is attached
per request and released by identity, including keepalive reuse. It includes
DNS/TLS/pool/write waiting and is a socket-accepted complete-write upper bound,
not server receipt or a hardware packet timestamp. Go `WroteRequest` alone is
insufficient because it precedes the final buffered flush on this Go version.
Missing emission is `-1`
and counts as late. The run is sender-invalid iff strictly more than 1% of ALL
scheduled requests have missing emission or `SentMs-ScheduledMs > 5`. The CLI
writes raw invalid records and exits nonzero; `Goodput` and `Calibrate` reject
sender-invalid or aborted runs. Failures remain in records even for a valid sender.

## Verified source protocol; runtime still pending

Contract target: SGLang v0.5.5 source commit
`0c006b8809cd99e1f95926401a2823dd952641c8`, native POST `/generate`, normal tokenizer
initialization (not `--skip-tokenizer-init`). Its `BatchStrOutput` branch emits
**delta** `output_ids`, cumulative `meta_info.completion_tokens`, a final
`finish_reason`, then HTTP endpoint `[DONE]`. This is not OpenAI SSE and not the
Task 14 string-only router API. Connect directly to the explicitly chosen native
endpoint. Source inspection is not evidence of installed runtime compatibility.

Parser limits: 1 MiB line/event, 64 MiB stream, 65,536 events; memory is bounded
per active stream. Empty/metadata events do not count as tokens. Delta ID lengths
must agree with cumulative counts when present; finish requires count, success
reason and `[DONE]`, and final actual count must equal requested output. Malformed,
oversized, inconsistent, error, truncated, cancelled and non-200 responses fail.
TTFT is first actual output delivery minus complete write; TPOT is last-minus-first
output delivery divided by actual output count minus one. Zero-output TTFT and
zero/one-output TPOT are `-1` (undefined); partial error metrics are diagnostic only.
Multi-ID events set `Coalesced`; delivery TPOT can be zero when all output arrives
in one event. These are client-observed values, never internal per-token GPU times.

## Offline build and run

```bash
mkdir -p .cache/task12/{build,go,mod,tmp,bin}
export GOCACHE="$PWD/.cache/task12/build" GOPATH="$PWD/.cache/task12/go"
export GOMODCACHE="$PWD/.cache/task12/mod" GOTMPDIR="$PWD/.cache/task12/tmp"
export TMPDIR="$GOTMPDIR" GOENV=off GOTOOLCHAIN=local GOPROXY=off GOMAXPROCS=2
env -u HUGGINGFACE_HUB_CACHE -u TRANSFORMERS_CACHE \
 HF_HOME=/home/postedism/.cache/huggingface \
 HF_HUB_CACHE=/home/postedism/.cache/huggingface/hub \
 python3 scripts/resource_preflight.py --planned-project-bytes 1700000000 \
 --download-bytes 0 --checkpoints 0 --service-memory-bytes 1073741824
.tools/go/bin/go build -p=1 -o .cache/task12/bin/replay ./serving/replay/cmd/replay
# Later, after a pinned model snapshot is available and authorized:
python3 serving/replay/tokenizer_allowlist.py \
 --snapshot /home/postedism/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/REAL_COMMIT \
 --model Qwen/Qwen3-8B --revision REAL_COMMIT > .cache/task12/token-ids.json
.cache/task12/bin/replay -trace serving/replay/testdata/azure.csv -format azure \
 -token-ids .cache/task12/token-ids.json -url http://127.0.0.1:8000/generate \
 -mult 1 -seed 1 > .cache/task12/real-run.json
```

The stdlib helper reads/hashes bounded tokenizer.json, tokenizer_config.json and
config.json from the trusted local immutable snapshot. It validates dictionary
vocab IDs against model vocab_size, excludes added and explicit special IDs, and
chooses the first <=4096 sorted IDs. This is not a tokenizer implementation or
cryptographic authentication of a hostile host. It neither downloads nor mutates
shared cache. A future native smoke must check `meta_info.prompt_tokens` against
payload length and output-ID behavior with zero/one/multiple generated tokens
before accepting real trials. The helper's synthetic metadata tests do not satisfy
that gate.

## Trace provenance and SLO evaluation

`testdata/provenance.json` identifies official immutable sources. Each CSV is the
header and first 100 complete rows from a <=65,536-byte read; no failure filtering,
sorting, resampling or token truncation. These tiny prefixes are biased selections,
not representative full workloads. The BurstGPT prefix spans 28,566 seconds
(over the one-hour replay bound even at 4x); it is an importer fixture, not a
runnable evaluation selection. Choose and document shorter contiguous windows
before actual 1x/2x/4x experiments, without concealing failed/zero-output rows. Azure naive timestamps are parsed as a common
UTC reference only to calculate deltas; BurstGPT timestamps are seconds from its
publisher's origin. BurstGPT elapsed time is not used as TTFT. Import preserves
zero-output cases (five in this BurstGPT prefix) and hashes exact selected bytes.

Azure: CC-BY, Patel et al., *Splitwise: Efficient generative LLM inference using
phase splitting*, ISCA 2024. See the official [description](https://github.com/Azure/AzurePublicDataset/blob/215becdacba1ce682c7368642ce97d5a332de7a6/AzureLLMInferenceDataset2023.md).
BurstGPT: CC-BY-4.0, Wang et al., *BurstGPT: A Real-World Workload Dataset to
Optimize LLM Serving Systems*, KDD 2025, [DOI](https://doi.org/10.1145/3711896.3737413),
[official source](https://github.com/HPMLL/BurstGPT/tree/d895a53bb7b8ec137d0d2fe203b335835a78c10a).
Only selected-prefix hashes are claimed, not full-dataset hashes.

`Calibrate` uses nearest-rank p99 (`ceil(.99*N)-1`) for complete >=2-token
requests with defined metrics. Report total, successful, failed, zero/one-output,
coalesced and calibration-sample counts alongside thresholds. `Goodput` counts
successful requests meeting both inclusive thresholds, divided by run start to
last completion (including initial idle and drain). Failures and undefined TPOT
are not good requests. Calibration and evaluation require distinct nonempty IDs;
callers must bind IDs to immutable manifests and disjoint source row ranges.
The library detects identical IDs, not hidden overlap in mislabeled datasets.

Future real baseline: freeze calibration rows first, establish real baseline p99
thresholds, then freeze thresholds for disjoint evaluation at 1x/2x/4x. At least
three matched baseline/candidate pairs per comparison, counterbalanced AB/BA order,
identical seed/context/cache settings, report median and range across accepted
rounds and every rejected sender round separately. No real baseline thresholds or
accepted real goodput are currently available.
