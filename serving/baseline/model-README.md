# Pinned local model preparation

The baseline uses **Qwen/Qwen3-8B BF16**, revision
`b968826d9c46dd6066d109eabc6255188de91218`. `model-files.json` fixes all 14
files, official revision URLs, exact sizes and upstream identities. The total is
16,397,459,696 bytes, including 16,381,516,776 bytes in five weight shards.
The cached AWQ model is a different artifact and is not a BF16 substitute.

`model_snapshot.py` uses only the Python standard library and the project's
canonical resource counter. It never imports model code. Files use the standard
shared HF layout under `~/.cache/huggingface/hub/models--Qwen--Qwen3-8B/`:
content-addressed `blobs/`, relative `snapshots/<revision>/` symlinks, and
`hub/.locks/models--Qwen--Qwen3-8B/<etag>.lock`. Two-way subprocess tests prove
the Linux `flock` seam excludes the installed trusted `filelock.FileLock`.
Shared lock files are never unlinked. No global cache is cleared.

From the repository root, use fresh evidence filenames and command-scoped paths:

```sh
HF_HOME=/home/postedism/.cache/huggingface \
HF_HUB_CACHE=/home/postedism/.cache/huggingface/hub \
HUGGINGFACE_HUB_CACHE=/home/postedism/.cache/huggingface/hub \
TRANSFORMERS_CACHE=/home/postedism/.cache/huggingface/hub \
PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.cache/task12-baseline/tmp" \
.venv/bin/python -B scripts/resource_preflight.py \
  --planned-project-bytes 650000000 --download-bytes 16397459696 \
  --checkpoints 0 --service-memory-bytes 500000000

PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.cache/task12-baseline/tmp" \
timeout --foreground --signal=TERM --kill-after=30s 3660s \
  .venv/bin/python -B -m serving.baseline.model_snapshot --download \
  --output docs/evidence/raw/task12-model-acquisition-01.json
```

The helper enforces a 3,600-second total deadline, 900 seconds per file including
hashing/publication, 300 seconds per blob lock, and at most 30 seconds per socket
operation. It checks time after transfer/hash and publication too; a late result
cannot become a verified success. SIGTERM records failure when Python can handle
it; the outer 3,660 + 30-second envelope bounds blocked cleanup. SIGKILL or host
failure may leave the last progress record incomplete; never infer success from
missing logs. The maintained downloader now owns a short-lived, stdlib-only DNS
resolver child for each nonliteral host. It uses an absolute subprocess timeout
and kills/reaps that exact child on timeout or interruption; it owns no persistent
service. Historical acquisition 17692 used the earlier helper and had no child;
its immutable lifecycle receipt remains unchanged.

The shared `deadline_http.py` helper requires the Linux main thread and refuses
an active, blocked or pending SIGALRM/ITIMER_REAL timer. It guards connect/TLS,
headers, incremental body reads, hashing and publication with one absolute POSIX
timer; a DNS subprocess avoids an uninterruptible main-thread libc resolver.
Teardown blocks/disarms the owned timer, consumes its pending signal, and restores
the caller's handler and signal mask. Deadline expiry cannot be reported as a
verified success. A syscall that completes concurrently with expiry can leave a
verified blob/partial for inspection; no rollback deletes shared cache content.
Final failure reporting and exact child cleanup are inside the external timeout
and kill-grace envelope. This is a Linux/CPython bounded cancellation mechanism,
not a hard real-time scheduling or atomic disk quota guarantee.

HTTPS certificate verification is standard-library default. GET redirects are
limited to five approved official hosts; response bodies for redirects/errors
are closed without draining. No proxy, credentials, unbounded body, source
build or fallback revision is used. Each successful response must have its exact
planned Content-Length. Bodies stream serially in at most 1 MiB chunks. Every
blob is checked against the LFS SHA256 or Git blob SHA1, with actual SHA256 also
recorded. These checks bind trusted upstream metadata to local bytes; they do not
authenticate a malicious host or certify model/runtime compatibility.

The resource observer counts all project files including Git/caches/temp by
unique device/inode, skips symlinks, and counts the entire shared HF cache.
It preserves 600 MB of future project growth plus 50 MB cleanup allowance under
the 40 GB share. Shared cache plus remaining model bytes must stay below 80 GB;
free space must remain at least 50 GB plus the full 16,397,459,696-byte download.
Before/after-file and roughly one-second transfer samples detect concurrent
growth; this is a sampled stop, not an atomic filesystem quota. No model bytes
are placed inside the project. A shard partial and its published blob share an
inode during the atomic no-clobber link/unlink step, so no second shard copy is
needed. A failure stops the sequence with no automatic retry.

A Task 12 wrapper restarts the **whole** canonical resource walk only when an
entry disappears with FileNotFoundError, at most three total attempts. It never
skips a directory or changes the common resource guard. Exhaustion and all other
errors propagate. Future rescan events retain repr, readable message, errno and
both filename fields; the three historical acquisition-02 paths remain unknown.
Successful resource samples retain the rescan events. This
addresses the observed transient `.agents` directory race in acquisition 53114.

Existing blobs are verified and reused. Existing incorrect blobs, partials or
snapshot entries cause failure; none are overwritten. New partials have unique
`.task12-<uuid>.incomplete` names, and progress records their exact path, device,
inode and written-byte count. Failure partials remain for inspection. A later
cleanup must first establish terminal process state, acquire the same blob lock,
and verify recorded path/device/inode/type before removing only that owned
partial. Shared HF `.incomplete` files, other tasks' partials, blobs, pointers and
locks are never cleanup candidates. Preserve the failure manifest and record
actual deleted size/identity; a rerun requires a fresh evidence file.

For the first failed acquisition, full host FD exclusion was unavailable because
the systemd user manager protects its FDs. Cleanup stopped without deleting data.
A reviewed `renameat2(RENAME_NOREPLACE)` under the same canonical blob lock moved
only that exact owned partial to `.task12-retained-failures/` within this model's
cache storage, preserving inode and all 303,796,954 bytes. Its receipt and prefix
hash remain in `raw/task12-model-partial-preservation.json`. This retained partial
counts against shared-cache/free-space limits throughout subsequent acquisition;
it is not a model blob or a source for silent range resume.

`model-manifest.schema.json` documents the evidence format. `status=verified`
requires all 14 rows; the runner must also check their exact name set, planned
sizes/upstream hashes and snapshot paths against the pinned plan, and rehash
actual snapshot files before inference. Omitting `--download` verifies an
existing complete snapshot without fetching files. Hash the actual manifest.

After acquisition, generate the deterministic input-ID allowlist without native
imports, using the existing tested helper and a fresh output file:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m serving.replay.tokenizer_allowlist \
  --snapshot /home/postedism/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 \
  > docs/evidence/raw/task12-model-token-allowlist.json
```

The allowlist binds config/tokenizer hashes, enforces unique integer IDs within
`vocab_size`, and excludes added/special IDs. Commit the source, exact runtime
lock/cache audit, trace selection manifests and fixed protocol before inference;
then bind actual acquisition/allowlist hashes in the pre-output checkpoint.
Later fixed-guard native smoke must verify prompt counts and zero/one/multiple
delta IDs, terminal/error behavior and cleanup before trace metrics. CPU parsing
and the successful import-only ABI probe do not establish serving compatibility,
memory fit, SLOs or goodput. No container has been executed.
