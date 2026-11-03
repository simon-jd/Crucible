# Verifier boundary

Public interfaces:

```python
from rl.verifiers.math import verify_math
from rl.verifiers.code import Case, TestBundle, Sandbox, verify_code
from rl.rewards.hacks import detect_hacks

verify_math(r"\boxed{\frac{1}{2}}", "0.5")  # Verdict(correct=True, ...)
tests = TestBundle(public=(Case(1, 2),), hidden=(Case(9, 10),))
# Provide a provisioned LocalRunsc or WardenSandbox; no host fallback exists.
# verify_code("def solve(x): return x + 1", tests, sandbox, timeout_s=10)
```

`Verdict(correct, reason)` has a derived binary `reward` property. Every unsupported input, uncertain execution, timeout, incomplete cleanup, or failed case gives zero reward.

## Supported code tasks and trust

A candidate defines synchronous `solve(value)` accepting one JSON value and returning one JSON value. The bootstrap serializes its return value. Extra stdout makes the answer invalid. Comparison is exact canonical JSON, including numeric representation (`2`, `2.0`, and `true` differ); keys are ordered canonically. NaN, infinity, duplicate keys, excessive nesting, and non-JSON values are rejected. No arbitrary pytest/unittest bundles, multiple function names, interactive tasks, stdin programs, third-party packages, or tolerant floating comparisons are promised.

The host grader never imports, compiles, or executes candidate Python. The guest bootstrap is **not a trusted grader**: a candidate can monkeypatch it, bypass `solve`, or emit arbitrary answer JSON. That grants no ability to assert success. Host code compares every response with its own expectation, requires authenticated/runtime-derived exit and cleanup evidence, and checks every public and hidden case. A constant answer matching a public case fails a different hidden case. The design does not prove that a returned value actually came from calling `solve`.

Each case gets a fresh isolated process/sandbox. Public cases are canonical JSON at `/tests/public.json`, mounted readonly for local runsc or baked into a verified readonly Warden image. Hidden expected values stay in the host `TestBundle`; neither bootstrap command nor runtime mount receives them. The one current hidden **input** necessarily reaches the candidate and may be read from its process arguments. Do not include `TestBundle.hidden`, retained transport payloads, or candidate logs in model prompts. Readonly files provide integrity, not confidentiality.

`detect_hacks` flags obvious early exit (including common import aliases), unittest/builtins monkeypatches, test-file writes, and literal prints. This is conservative static/log screening, with false positives and obfuscation gaps. It does not establish isolation or absence of cheating. The safety boundary remains gVisor and the independent grader.

Limits: 64 total cases with both public and hidden groups required; source at most 65,536 UTF-8 bytes; each canonical JSON value/public bundle/combined bootstrap request at most 65,536 bytes; JSON depth 16, collection size 1,024, integer magnitude 256 bits. The default `timeout_s=10` is one wall budget for the **whole bundle**, not 10 seconds per case. An adapter receives remaining time. Setup, queueing, execution and cleanup all consume that budget; no reward is issued after its expiry.

## Math grammar

Inputs are at most 4,096 characters. Extract the last balanced `\boxed{...}` answer (nested fractions/boxes supported); without a box, require the entire expression. No partial-prefix parsing. Supported atoms: decimal integers and decimals with leading digits, symbols `x`/`y`, `pi`/`\pi`, parenthesized/braced groups, `\frac{a}{b}`, `sqrt(a)`/`\sqrt{a}`. Operators are explicit `+ - * / ^`, with ordinary unary-minus precedence; powers must be literal integers from -8 through 8. Multiplication must be explicit. Other functions, implicit multiplication, scientific notation, units, text, sets, tuples, matrices, equations, and general LaTeX are unsupported. Literal negative radicands and undefined/nonfinite arithmetic are rejected.

A whitelist parser constructs SymPy objects directly; `eval`, `sympify(string)`, `parse_expr`, and general LaTeX parsers are never used. Limits include 96 tokens, 32 arithmetic operators, nesting 12, 32-character numeric tokens and 128 intermediate SymPy operations. SymPy cancellation/simplification runs in a separate trusted worker with 2 CPU seconds, 512 MiB address space, and 3 seconds wall time. Over-complex expressions fail closed even if mathematically valid. Symbolic equivalence uses algebraic expressions/rational functions, ignoring removable-pole domain differences; it does not prove equality of functions on identical domains. Nonzero/undecidable differences return `not_equivalent_or_unknown`, grammar failures return explicit `invalid_math:*` reasons. This is a useful bounded math subset, not complete MATH-500 coverage.

## Local gVisor prerequisites

`LocalRunsc(RunscConfig(...))` takes the verified project-owned runsc bundle, dedicated rootfs + SHA256-pinned manifest, project/private work directory, delegated cgroup parent, and optional explicitly provisioned trusted entrypoint argv. It never uses `runsc do`, mounts `/`, copies HOME, modifies the daemon, or falls back to host Python/runc. The installed bundle is `.tools/gvisor/20261019.0`; retain its official SHA512 and sidecar files, as recorded in `docs/evidence/sandbox-runtime.md`.

The execution entrypoint must supply working namespace privileges (including uid/gid 65534), namespace setup, and **delegated cgroup v2 cpu/memory/pids controllers**. A configured entrypoint is trusted operator code, not a proof of these capabilities. The adapter checks readable/enabled controllers and write access before launch and requires runtime cleanup evidence after execution. No such working entrypoint is currently provisioned here. The known rootless probe failed at `/proc/self/exe`, and the userns probe failed at uid_map; neither was rerun for this task. The adapter consequently fails closed under current prerequisites. Any new operational permission profile requires its own concrete review; this task grants none.

Generated OCI: readonly dedicated rootfs, uid/gid 65534, no new privileges/capabilities, separate PID/mount/network/IPC/UTS namespaces, `--network=none --platform=systrap`, 1 CPU quota, 512 MiB memory, 128 pids, bounded rlimits, 16 MiB private tmpfs, and only the readonly public-case bind mount. The candidate receives no host-root/HOME/repository/credential mount. Host capture is capped at 64 KiB per stdout/stderr stream and rejects invalid UTF-8. The adapter reserves up to one second (20% of remaining budget) for `kill --all` and `delete --force`, confirms deletion success and empty/absent dedicated cgroup, and reaps the launcher. Any uncertain cleanup gives zero reward and retains the private bundle with its run ID for operator recovery; do not blindly delete it or start an unlimited stream of retries. Cancellation by process crash still requires an external supervisor/reaper; no crash-proof orchestration is claimed.

Rootfs preparation is file copying only. With the trusted pinned local CPython already installed, reproduce the recorded file set with:

```bash
.venv/bin/python scripts/prepare_verifier_rootfs.py \
  --destination .tools/verifier-rootfs/reproduced \
  --expected-python-sha256 021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7 \
  --expected-manifest docs/evidence/verifiers/rootfs-manifest.json
```

Use `--dry-run` to estimate before preparing. Source dependencies must match the checked-in hashes; the command refuses a changed file set before publishing. An empty node must first provide this CPython build and matching system ELF libraries (source locations and all hashes are in the manifest); this tool does not install OS packages or fetch an unpinned image. It copies CPython, explicit ELF dependencies, and standard-library files, excluding site packages, caches, tests, pip/venv and GUI modules including `_tkinter`. It records every file hash and mode, rejects escaping symlinks, bounds rootfs size to 200 MiB and construction estimate to twice source bytes, and preserves 50 GiB disk headroom. The current prepared rootfs has not been booted in gVisor.

## Warden prerequisites and recovery

`WardenSandbox` requires an explicit HTTPS origin, bearer token, and `RunnerImage` deployment metadata. No service URL is guessed. It uses only POST `/v1/runs`, GET `/v1/runs/{id}`, and GET `/v1/runs/{id}/logs`; redirects are rejected. The imported immutable `sha256:<64 lowercase hex>` image must contain `/usr/bin/python3`, this exact `/verifier/_candidate.py`, and the matching canonical public JSON at `/tests/public.json`. Image metadata records bootstrap/public hashes and protocol `crucible-solve-json-v1`; the operator must verify/import the image and keep the underlying deployment trusted. Metadata alone is not image-content or isolation attestation. Because the API has no uploads/mounts/stdin, changing public cases requires another verified imported image. No such image or authentication setup exists for Crucible today.

Requests explicitly select `runtime=gvisor`, `network=none`, cpu=1/memory_mib=512/pids=128 and a bounded execution timeout. Client checks the returned immutable spec and fixed readonly/uid/no-new-privilege fields, terminal state, exact-case runtime result fields (`Code`, `OOMKilled`, `TimedOut`, `CgroupEmpty`, `LogsTruncated`, `DroppedLogBytes`, `PidsMaxEvents`, `PeakPids`), `cleanup_pending=false`, and complete undropped logs. Base64 chunks are joined before decoding; complete empty logs are transport-valid but not a valid JSON answer. HTTP 200, exit 0, or printed success alone never awards reward.

Execution timeout is rounded up to whole seconds for the API. Queue delay is charged against the local wall budget, so remote work may outlive a client timeout. There is **no public cancel endpoint**. `RunResult`/failed verdict retain the known run ID and idempotency key, and `WardenSandbox.pending` retains the exact payload for `retry(key, timeout_s)`. A lost POST response reuses the same key/payload; it does not submit a new run under a new identity. Callers needing recovery across process restarts must durably persist this mapping before submitting. A timeout before authoritative cleanup never claims the remote cgroup is empty. Current Warden has public result/log schemas but production auth and gVisor registration are unavailable; HTTP fixtures are not service acceptance.

HTTP I/O runs in a separate fixed trusted `_warden_http.py` process under the current interpreter with `-I -S` and a minimal environment. The bearer and request body travel only over stdin; the worker never executes candidate code. Its request/body/response envelope sizes are bounded and diagnostics do not echo credentials. The parent applies one absolute monotonic deadline across process startup, DNS, connect, TLS, headers, body and process reaping; it reserves up to 0.2 seconds (25% of the request budget) for local cleanup. Socket inactivity timeouts are only supplemental. Slow trickling cannot reset the parent deadline. A normal timeout kills and reaps the owned worker, closing its sockets; this does **not** cancel or establish cleanup of a remote Warden run.

An exceptional inability to signal/reap the local worker returns `local_http_cleanup_unknown:pid=...`, retains process ownership, and blocks further HTTP launches until exit/reaping is confirmed. The client does not silently abandon a network thread or claim cleanup. No hard cleanup guarantee is made for external SIGKILL of the controller, kernel failure, or a host that refuses process control; external supervision is still required. Use one in-flight call per adapter.

Once the authenticated terminal state, typed/range-checked result, matching safe spec and remote cleanup are all accepted, `pending` evicts the full submission on **every** outcome: success, execution failure, timeout and subsequent log failure. Incomplete logs still give zero reward, while preserving already-known remote cleanup. Lost POST/status, unconfirmed cleanup, malformed result and mismatched spec retain the exact retry record. This prevents routine failed candidates from accumulating full source/input payloads; genuinely uncertain work remains available for recovery.

## Verification status

`rl/verifiers/tests` uses real bounded math workers, host grading, generated OCI inspection, synthetic rootfs fixtures, and trusted fake runtime/HTTP fixtures. Fake drivers never execute candidate Python. These prove code behavior, not readonly enforcement, network isolation, fork/memory containment, hidden-test isolation or real cleanup in gVisor. Those require a future approved real runtime acceptance run; see `docs/evidence/verifiers.md`.
