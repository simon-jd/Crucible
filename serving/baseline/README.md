# Local single-GPU baseline

**Current baseline (2026-10-27):** `vllm_baseline.py` driven by
`scripts/run_serving_baseline_gpu.sh azure|burst OUT`, which serves Qwen3-8B-AWQ with vLLM 0.10.2
and replays `traces/azure-60s` and `traces/burst-300s`. The results are in
[serving-baseline.md](../../docs/evidence/serving-baseline.md). Everything below describes the
earlier SGLang-native BF16 harness. It is kept for reference and is no longer the baseline.

This directory prepares and runs the fixed Qwen3-8B BF16 experiment. The initial
CPU replay phase and the later native import-only ABI probe passed. Model serving
and real trace results are still pending; see the chronological
[evidence](../../docs/evidence/serving-baseline.md). The 235B and cross-node work is
paused: 未在真实多机上验证.

## Runtime identity and bootstrap

This budget-constrained run uses the approved native-runtime exception. **No
container was executed.** The inert deployment image from the earlier phase is
not runtime evidence. `runtime-lock.json` fixes Python 3.12.13, uv 0.12.23,
SGLang 0.5.5/source commit `0c006b8809cd99e1f95926401a2823dd952641c8`, every selected
wheel's URL/hash and installed package metadata/RECORD identities. Its 161-package
graph reuses 99 read-only base packages and installs 62 overlay wheels. Base Torch
2.8.0+cu128 and Triton 3.4.0 are not duplicated or upgraded; overlay Transformers
4.57.1 and FlashInfer 0.5.0 take precedence only in this runtime entry.

Prerequisites are the exact preprovisioned `.venv-ml` base described by the lock,
the project's stdlib/test Python environment, the fixed `scripts/with_gpu.sh`
guard, Go, and the host driver. The lock includes expected wheel provenance for
base dependencies. The separate installed-content audit uses `verify_record` to
check actual base RECORD content; the entry validates selected package metadata
and origins. `validate_lock` checks the overlay wheel plan. Package installation is
target-only, binary-only, hash-checked, serial and dependency-free through trusted
uv. It never resolves a replacement version at installation time.

On a fresh runtime preparation, build the already-tested Go CLI **before** delta
installation, with GOCACHE/GOTMPDIR/GOMODCACHE under the project. Retain its binary
and remove only that completed build's owned temporary/cache paths. The measured
binary was 10,138,001 bytes, SHA256
`68bb43b4aee08ca371d05a75b467a1f163c1fe8e959ccb534811740b63dffa93`.
The replay module is standard-library only:

```sh
mkdir -p .cache/task12-baseline/bin .cache/task12-baseline/go-build \
  .cache/task12-baseline/go-tmp .cache/task12-baseline/go-mod
GOCACHE="$PWD/.cache/task12-baseline/go-build" \
GOTMPDIR="$PWD/.cache/task12-baseline/go-tmp" \
GOMODCACHE="$PWD/.cache/task12-baseline/go-mod" \
.tools/go/bin/go build -o .cache/task12-baseline/bin/replay ./serving/replay/cmd/replay
```

Before using an already provisioned base on a clean reproduction, verify its
locked metadata/RECORD identities and every recorded file without importing CUDA:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B - <<'PY'
import json
from pathlib import Path
from serving.baseline.prepare_runtime import sha256, verify_record
root = Path.cwd()
lock = json.loads((root / 'serving/baseline/runtime-lock.json').read_text())
for package in lock['packages']:
    if package['origin'] != 'base-readonly':
        continue
    dist = root / package['dist_info']
    assert sha256(dist / 'METADATA') == package['metadata_sha256']
    assert sha256(dist / 'RECORD') == package['record_sha256']
    verify_record(dist.parent, dist / 'RECORD', root / '.venv-ml')
PY
```

Before installing, run the common resource guard with the explicitly reviewed
3.5 GB additional peak and the exact compressed download total 846,663,134 bytes.
All Git, caches, temp and compiler artifacts count under the 40 GB project share;
hardlinks count once by device/inode. Delta wheels expand to 2,868,380,729 bytes.
Use the generated small-wheel proof to establish actual same-filesystem uv
hardlink behavior, retaining inode evidence. Do not assume a copy fallback fits:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m serving.baseline.prepare_runtime \
  --prove-installer .cache/task12-baseline/fresh-proof \
  --output docs/evidence/raw/fresh-installer-proof.json

PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.cache/task12-baseline/tmp" \
  .venv/bin/python -B -m serving.baseline.prepare_runtime \
  --install-lock serving/baseline/runtime-lock.json \
  --output docs/evidence/raw/fresh-runtime-install.json
```

These commands require fresh owned destinations; an existing overlay is not a
fresh-install target. Serial compressed-size-descending installation keeps one
wheel/cache/stage at a time. Immutable large payloads must share cache/target
inodes. The one exact pinned py-spy script permits a separately budgeted possible
8,079,336-byte copy with full archive/cache/target hash checks. Actual py-spy used
a hardlink here; the generated relocation fixture observed a copy. Regenerated
RECORD is exempt only beside the single validated METADATA parent and still
counts in the overhead reserve. Stage cleanup occurs only after verification.
On failure, preserve the stage and evidence; `--continue-from` requires an audited
completed-prefix/pending-stage receipt. Never infer a safe continuation from a
partially populated target directory alone.

Each wheel transfer now uses the same Linux main-thread absolute deadline helper
as model preparation: 300 seconds across DNS/connect/TLS, headers, incremental
body reads, hashing and no-clobber publication. Redirect/error bodies are closed
without draining, and at most five redirects stay on the exact approved host.
The owned DNS child is killed/reaped on interruption; conflicting interval timers
are refused and the prior signal handler/mask are restored. Partial files remain
on failure. See [model preparation](model-README.md) for cancellation limits.

`runtime_entry.py` runs under `-B -S`, sets explicit overlay-first module paths,
validates the pinned CUTLASS path/pure-Python redirector and does not execute
arbitrary `.pth` lines. `cache-audit.json` binds installed cache/download-control
source hashes. Supported FlashInfer/TVM/SGLang/Triton/CUDA paths stay in the owned
runtime cache. The derived native environment explicitly removes inherited
`TRANSFORMERS_CACHE`; supported HF_HOME/HF_HUB_CACHE and offline settings remain.
Historical deprecation warning logs are retained. Optional builds are disabled; missing cubins fail against a local
repository instead of downloading. The combined runtime/JIT/compiler/temp budget
is 500 MB. Unknown acquisition/build requirements stop the experiment.

## Model and trace inputs

Follow [model-README.md](model-README.md) for the immutable 14-file BF16 plan,
shared-cache locks, bounded acquisition, actual manifest, deterministic tokenizer
allowlist and retained-failure policy. The current completed standalone ABI file
is `docs/evidence/raw/task12-baseline-abi-01/abi.json`, SHA256
`ba3df4a836f0cdb480399c8ca9c1044b122daff96de8a8e88e2e256193d6ef01`.
Full mode reuses this evidence and verifies current package metadata; it does not
launch another standalone ABI probe. The server entry records its own inventory.

`traces/manifest.json` contains immutable source URLs/commits, exact bounded
prefix SHA256, returned response metadata, attribution and row/window hashes.
Those are prefix hashes, not full upstream CSV hashes. `select_traces.py` checks
the prefix hash and selects windows using arrival times alone. On a clean
reproduction, read only the manifest's byte bounds from those pinned URLs, verify
the whole received prefix, then select Azure with duration30/min-rows1 and
BurstGPT with duration60/min-rows20. The selector CLI takes `--prefix`, `--sha256`,
`--format azure|burstgpt`, `--duration`, `--min-rows`, and a fresh `--output`.
Compare resulting CSV bytes to the committed files. Do not normalize Azure CRLF.

The committed Azure windows have 59 calibration and 132 evaluation requests;
BurstGPT has 23 each. All internal gaps and rows survive. Source-arrival selection
of dense BurstGPT windows is disclosed selection bias. Selected zero-output rows
are zero, so separate parser fixtures are not measured zero-output coverage.
Over-context rows remain real server failures rather than being truncated.
These traces provide arrival times and token lengths, not original prompt text.
Requests therefore use deterministic IDs from the verified tokenizer allowlist;
the measured workload is a real arrival/length trace with synthetic token content.

## Fixed experiment and lifecycle

`protocol.json` fixes both client replay seed17 and SGLang `--random-seed 17`,
BF16/context4096, static memory fraction.80,
eight running server requests, no CUDA graph/radix cache, Triton attention and
PyTorch sampling. The client admits at most192 in-flight requests and uses
absolute60-second deadlines. Arrivals are open-loop at 1x/2x/4x; no response pacing
or retry-to-pass. Excess admission is a recorded failure. Input/output lengths
come from the source rows, with requested and actual output totals reported
separately. The parser origin is the first selected request; throughput uses that
first scheduled launch through last completion, not the whole nominal window.
The frozen CLI exits1 when a complete run contains failed requests. The controller
preserves that exit code and stderr, and recognizes only the exact bounded stderr
`run contains failed requests\n` with every planned record, explicit non-aborted
state, verified valid sender and at least one failed row. These failures remain in
the denominator and calibration failure counts. Unknown exits, interruptions,
partial/malformed output and invalid senders cannot use this exception.
The replay retains at most16MiB of prepared request bodies for a whole trial.
Each active stream caps an event at1MiB and total received bytes at64MiB; it
processes events incrementally rather than retaining the entire response. The
event's scanner/string/JSON allocations and decoded integer slice can coexist.
Budgeting24MiB per active request at192 slots is4,831,838,208B, leaving about668MB
within the5.5GB client estimate for Go runtime, transport and retained records.
This is a conservative planning envelope, not a measured RSS guarantee. Server
9.5GB plus replay5.5GB plus controller0.5GB estimates total15.5GB; the observer
stops at16GB aggregate owned RSS. Model fit remains unproven until the guarded run.

Before model outputs, checkpoint the source, runtime lock/cache audit, trace
manifests, protocol, actual model manifest and allowlist hashes. Full mode rehashes
the actual snapshot and regenerates the allowlist to detect stale/mutated inputs.
The later launch must use the existing fixed GPU guard and a fresh evidence
directory. Its controller inputs are:

```text
--lock serving/baseline/runtime-lock.json
--overlay /home/postedism/Desktop/Crucible/.cache/task12-baseline/overlay
--cache-audit serving/baseline/cache-audit.json
--protocol serving/baseline/protocol.json
--model /home/postedism/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
--model-manifest <verified actual acquisition JSON>
--token-ids <generated actual allowlist JSON>
--binary /home/postedism/Desktop/Crucible/.cache/task12-baseline/bin/replay
--abi-evidence docs/evidence/raw/task12-baseline-abi-01/abi.json
--abi-evidence-sha256 ba3df4a836f0cdb480399c8ca9c1044b122daff96de8a8e88e2e256193d6ef01
--output <fresh local evidence directory>
```

Native smoke checks zero/one/multiple delta IDs, exact prompt/cumulative counts,
terminal/DONE ordering and parser rejection. It retains raw responses and failure
summaries; each owned loopback request has an absolute30-second socket watchdog,
which is cancelled/joined. Pinned SGLang also performs its own startup warmup
before readiness: with the tokenizer enabled, `http_server.py` sends the text
"The capital city of France is" with `max_new_tokens=8`. That runtime warmup is
outside source windows and distinct from our fixed preparation warmups
128/16,1024/32,2048/32, which also precede source windows. The explicit server seed
avoids the pinned server's random seed selection when that flag is omitted.
After the fixed warmups and every attempted calibration/evaluation trial, the
controller POSTs `/flush_cache` outside the measured interval. Only HTTP200
establishes the boundary; HTTP400 is retried within one absolute30-second
deadline, and any other status/deadline stops later trials. Attempts, elapsed
time and response hashes are retained. The pinned scheduler checks waiting,
running, last/current and overlap work before accepting this call and resetting
cache/memory pools. A finished client alone is insufficient because disconnect
abort is asynchronous. This same boundary applies to every condition, including
failed trials; their raw/current summaries and missing planned pair members stay
visible. Each response is capped at64KiB and cumulative responses at1MiB.
Three valid 1x calibration rounds pool defined successful >=2-token samples using
nearest-rank p99 TTFT/TPOT. Failure/undefined/coalesced counts remain visible.
Thresholds are saved and hashed before disjoint evaluation and bind actual model,
runtime, source, server config, tokenizer, protocol, trace and replay identities.
These short windows provide limited tail evidence. In particular, at most69
BurstGPT calibration requests means empirical nearest-rank p99 is the maximum
when all69 qualify. A frozen operational threshold is not a precise estimate of
production p99; failure/undefined counts can reduce the sample further.

Evaluation uses three matched pairs for each 1x↔2x and 1x↔4x comparison, with fixed
counterbalanced order. Every planned member remains in pair reports, including
failed/interrupted/invalid/not-run members. Accepted aggregate comparisons need
all three pairs; report median/min/max paired goodput difference. More than1%
missing sends or complete-write lag strictly >5ms rejects all accepted performance
for that run. The observable is socket-accepted complete-write, not server receipt.

The fixed guard owns the process group, GPU lock and final descendant drain;
the controller terminates/reaps exact Popen children before optional report I/O,
verifies listener closure, and never blindly signals a reaped PGID. Stop new
trials at3h15m; keep total GPU ownership including cleanup below4h. Resource
samples include aggregate owned RSS and bounded mutable caches/evidence; full
canonical scans occur between timed trials. A demonstrated disappearing-entry
race now restarts the whole canonical walk at most three times, never skipping
paths. Incomplete FD inspection of the preserved download partial is explicitly
recorded, not turned into a successful cleanup claim.

## Startup attribution diagnostic (not yet executed)

`startup_memory.py` is a separate bounded diagnostic following the failed full
startup at 16,277,946,368 bytes aggregate RSS. It keeps the same Qwen3-8B BF16
revision, context, static fraction, running limit, tokenizer mode and seed. Its
only server flag addition is `--skip-server-warmup`; the controller sends no HTTP
requests. It observes the exact pinned `The server is fired up and ready to roll!`
log marker, which follows setting ServerStatus.Up when startup warmup is skipped.
This includes markers split across pipe reads. The pinned `/health` endpoint
itself generates one token after startup, so it is unsuitable for this diagnostic.
An initial CPU draft used that endpoint; the pre-launch regression exposed and
removed it. No native diagnostic ran with that draft.

It sums the same `/proc/<pid>/stat` RSS for its controller and owned descendants,
records PID/starttime/PPID/group and status Anon/File/Shmem attribution, and retains
known identities across reparenting. It never deducts shared/file pages or uses
PSS. Reads are capped at 8 KiB, global enumeration at 8,192 processes, owned rows
at 64, and samples at 1,200. Ambiguous/reused/denied identities fail closed. The
100 ms target cadence is best effort: each sample includes actual monotonic time,
and status/RSS reads are non-atomic. The larger before/after stat RSS is retained.

The first of 10 GB aggregate RSS, 120 seconds, server exit, readiness or a resource
failure stops the probe. The unchanged service cap is 16 GB; the earlier threshold
cannot guarantee no transient overshoot. Samples and errors are written before
exact Popen cleanup. Logs use a bounded nonblocking pipe, copied in at most eight
64 KiB chunks per sample. All owned evidence is capped at 16 MiB with a 64 KiB
terminal reserve. Pipe backpressure may affect startup; unread shutdown-tail log
bytes can be lost when the pipe closes, and this is not performance evidence.
Initial/roughly one-second disk/resource checks reuse the existing canonical and
mutable-root accounting. Each complete threshold sample is retained, including
its per-PID rows. If the subsequent periodic resource check raises, its exact
repr payload is attached as `resource_error`; the complete bounded sample is
flushed before that same error is re-raised into the stop/cleanup path. Metadata
failure before sample completion or insufficient evidence capacity can still
prevent a complete current sample; those failures remain explicit.

The separate reviewed command uses the fixed GPU guard and a foreground
180-second timeout with 30-second kill grace. The controller signals only its
exact unreaped server child, reaps it and checks listener closure; descendant
drain belongs to the fixed guard. A ready diagnostic is not Task 12 acceptance.
No diagnostic GPU execution is claimed by the CPU regression results.

### Compile-worker resource candidate (CPU proof only)

The derived native environment now explicitly sets `TORCHINDUCTOR_COMPILE_THREADS=1`
for missing or conflicting inherited values, before the audited redirector and any
native imports. Existing `MAX_JOBS=1` controls a different path. All HF/offline and
owned cache/download controls remain unchanged. The next diagnostic command also
sets the compile-thread variable explicitly so spawned native children inherit it.

`compile-pool-source-contract.json` binds five installed Torch primary sources to
actual SHA256/size and their installed RECORD entries, plus the locked RECORD hash.
Pinned config gives this environment variable precedence. AsyncCompile returns
from warm_pool at <=1, executes submit synchronously, and avoids its process pool;
compile_fx imports async_compile, whose import can warm that pool. The default
subprocess manager otherwise creates an eager fork-worker pool. No installed
package, multiprocessing method, model, precision, workload or RSS accounting was
changed. The CPU tests intercept entry-point imports rather than importing Torch.

This source path is a hypothesis for diagnostic01's unknown parent and 23 similar
children. Their actual modules, historical effective compile-thread count and the
earliest actual importer were not captured. CPU evidence establishes neither
memory savings nor model fit; serial compilation may alter startup time. The new
`task12-startup-memory-02-proposal.json` and `-source-freeze.json` are prepared only
after the source checkpoint, for separate review and one-shot authorization. They
retain zero HTTP requests, exact readiness-log observation, skip-server-warmup,
100 ms target sampling, 10 GB early stop, 16 GB RSS cap and bounded exact-child
cleanup. Original diagnostic01 is terminal and will not be restarted.

Future baseline calibration/evaluation lineage must bind the new actual native
environment and source freeze; historical ABI binaries remain the same, but an
import-only ABI result is not model/performance acceptance. Native token smoke,
fixed warmups, six calibration and 24 matched evaluation members remain pending.

### Run lifecycle and invalidation rules

Launch the full run directly through `scripts/with_gpu.sh` (shared flock,
4-hour cap); there is no separate proposal/freeze launcher, native resource pool,
IPC admission scan or per-file evidence byte cap any more (retired 2026-10-26). The guards that remain are the plan's and
the resource budget's: the 40 GB project / 80 GB shared-cache / free-disk high-water
stops, the 16 GB owned-RSS stop, the 3 h 15 min stop-new-trials deadline, and
per-request/trial deadlines.

- Sender validity (plan Review Focus 4): a trial is invalid when more than 1% of
  requests are sent more than 5 ms late. Calibration needs three sender-valid runs.
- Native smoke/warmup responses are read up to 4 MiB with an absolute deadline.
  An empty `read1()` with Content-Length remaining raises `IncompleteRead`; such a
  body stays `http_body_complete=false` and its count is a lower bound.
- The terminal cause is written before cleanup; report failures cannot replace the
  primary failure or skip child cleanup, and they invalidate aggregate acceptance.
  All 30 planned trial statuses, including `not_run`, stay in the terminal record.
- A native exit before the controller requested TERM (including exit 0), a forced
  kill, or an unclosed listener invalidates both comparison summaries.

Tests: `serving/baseline/lifecycle_test.py`, `serving/baseline/experiment_test.py`.
