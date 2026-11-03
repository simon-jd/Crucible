# Task 6 verifier evidence — 2026-10-23

## Scope and result

Implemented the bounded math verifier, binary verdict/reward, pure `solve(JSON)->JSON` code grader, hack heuristics, direct OCI runsc adapter, documented Warden API adapter and dedicated CPython rootfs preparation. Public interfaces match the task brief. Protocol and supported syntax are in [the verifier README](../../rl/verifiers/README.md).

**Code/fixture evidence is not real sandbox acceptance.** No candidate Python ran on the host, no gVisor workload ran, no Warden service was contacted, and no GPU/cloud/paid resource was used. The earlier native `/proc/self/exe` and uid_map failures remain the actual runtime evidence; see [sandbox runtime evidence](sandbox-runtime.md). This task only inspected `runsc run --help` (exit 128 is its help-path behavior) and upstream source to check lifecycle flags/forced deletion; it did not repeat the failed workload.

## CPU behavior

Required RED command:

```text
.venv/bin/python -m pytest -q rl/verifiers/tests/test_verifiers.py
ModuleNotFoundError: No module named 'rl.verifiers.math'
ERROR rl/verifiers/tests/test_verifiers.py
1 error in 0.04s
```

Expected failure: verifier modules did not yet exist. The original final focused GREEN command was:

```text
.venv/bin/python -m pytest -q rl/verifiers/tests
........................................................................ [ 77%]
.....................                                                    [100%]
93 passed in 6.03s
```

Coverage includes real bounded SymPy worker equivalence for fractions/decimals/nested boxes/radicals/pi; wrong/unsupported/malicious math, grammar/depth limits and worker timeout; all-case host grading, hidden expectation exclusion, fixed-public-answer defeat, strict JSON/Unicode/source limits, malformed sandbox results and retry identity; all requested hack signals; generated OCI mounts/namespaces/resources; manifest tampering, symlink escape, pin/dependency-lock failure; fixture launch/output/timeout/delete failure; and Warden's exact runtime fields, spec matching, incomplete/empty/base64 logs, queued/lost-POST deadlines and identical-key retry without a fictitious cancel.

One intermediate fixture failed because the test tried to mutate a chmod-0444 fixture without first changing its host test mode (84 passed/1 failed). The fixture now explicitly chmods before tampering; this was a test setup correction, not evidence of guest readonly enforcement. Earlier focused stages passed 35, 74 and 85 tests. Ruff passed after self-review (`.venv/bin/ruff check rl/verifiers rl/rewards scripts/prepare_verifier_rootfs.py`: `All checks passed!`).

The single full CPU gate is recorded verbatim in [full-cpu.txt](verifiers/full-cpu.txt). It runs `.venv/bin/python -m pytest -q` on the original finalized source snapshot (`711b5cb`), after self-review and before the fix round below. No additional GPU or real-runtime gate is inferred from it.

## Dedicated rootfs preparation

The trusted local CPython version is `3.12.13 (main, Aug 7 2026, 02:25:39) [Clang 22.1.3]`; executable SHA256 is `021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7`. Source inventory: 621 files including the public-test placeholder; copied source bytes 47,664,649 plus the 2-byte generated placeholder. Estimated construction peak 95,329,298 bytes (twice source bytes), no downloads. The first dry run identified missing Tk ELF dependencies; GUI/Tk was excluded consistently with the intended computation-only stdlib.

Preflight initially rejected inherited `HF_HOME`/`HF_HUB_CACHE` pointing outside the mandated shared cache. Setting those two variables for the **preflight command only** to `/home/postedism/.cache/huggingface` and its `hub` passed. No cache, global environment or resource-guard source changed.

```bash
HF_HOME=/home/postedism/.cache/huggingface \
HF_HUB_CACHE=/home/postedism/.cache/huggingface/hub \
.venv/bin/python scripts/resource_preflight.py \
  --planned-project-bytes 95329298 --download-bytes 0 \
  --checkpoints 0 --service-memory-bytes 0
```

Preflight: `ok=true`, project bytes `7297012932`, free bytes `377140715520`, errors `[]`. Then preparation command:

```bash
.venv/bin/python scripts/prepare_verifier_rootfs.py \
  --destination .tools/verifier-rootfs/cpython3.12.13 \
  --expected-python-sha256 021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7
```

[Preparation metadata](verifiers/rootfs-preparation.json) and the complete [dependency/source/file manifest](verifiers/rootfs-manifest.json) are committed. The generated manifest was renamed to append `.manifest.json` to the full versioned rootfs name (instead of replacing the `.13` suffix); its content/hash is unchanged. SHA256: `38e6fd2921471a327187b792db7431576e653d733d789fdcd5b410f8d993c935`. `verify_rootfs` verified all 621 entries, exact hashes/modes and guest bootstrap identity after preparation. `--expected-manifest` supports exact-file-set reconstruction from pinned dependencies; GUI/site-packages/tests/caches are absent. The generated runtime files remain in ignored project `.tools`, not Git.

[Postflight](verifiers/resource-post.json): `ok=true`, project bytes `7345100120`, free bytes `376807292928`, errors `[]`; command is the same preflight with `--planned-project-bytes 0`. The disk numbers measure the host at those moments and are not attributed entirely to this task on a shared machine.

## Remaining real acceptance prerequisites

1. A specifically reviewed working execution entrypoint with namespaces, uid/gid 65534 and delegated cgroup-v2 CPU/memory/pids controllers. Do not rerun the known failing native entrypoint, modify host policy, register Docker runtimes or borrow Warden labs/images. Configuration/fixtures are not proof this prerequisite is met.
2. Boot the pinned dedicated rootfs inside actual gVisor, then demonstrate correct/wrong solve outputs, quiet-output protocol, readonly public-file write rejection, hidden expectation absence, host-secret/network denial, bounded loop/fork/output/memory cases and confirmed process/cgroup cleanup. No such claims are currently verified. Runtime cancellation by controller crash requires external lifecycle supervision.
3. For Warden, real trusted production authentication and registered gVisor driver; independently imported immutable runner image containing exact bootstrap and matching public JSON; authoritative completion/spec/cgroup/logs checked over its documented API. The current public service progress is result/log schemas and runc lab evidence, not available gVisor/authenticated end-to-end service.
4. Persist Warden pending payload/key/run ID before submission when recovery must survive a client restart. A queue/client timeout can leave remote work alive; no cancel endpoint exists. Never equate timeout/HTTP200/stdout/exit0 with completion or cleanup.

Math intentionally covers a bounded expression subset, not every MATH-500 answer format. Hack screening is heuristic. Hidden-case quality still matters: output-based judging cannot distinguish a genuine computation from an answer that happens to match all held-out expectations.

## Sources consulted

- Task 6 brief/constraints/context/dispatch supplied by the controller, including its 2026-10-23 read-only snapshot of Warden's public README. No private adjacent implementation or credentials were read and no undocumented endpoint was assumed.
- [SymPy parsing documentation](https://docs.sympy.org/latest/modules/parsing.html): unsafe general-string parsing and strictness concerns motivated explicit syntax construction.
- [gVisor rootless guide](https://gvisor.dev/docs/user_guide/rootless/): namespace and cgroup limitations require an explicitly prepared OCI entrypoint.
- Pinned gVisor `release-20261019.0` [run implementation](https://raw.githubusercontent.com/google/gvisor/release-20261019.0/runsc/cmd/run.go) and [delete implementation](https://raw.githubusercontent.com/google/gvisor/release-20261019.0/runsc/cmd/delete.go): normal run lifecycle and forced deletion accepting already-absent state. No unsupported `--keep` flag was added.
- Existing project [sandbox runtime evidence](sandbox-runtime.md) supplies verified bundle provenance and actual failed native probe results; this task does not replace them with fixtures.

## Fix round 1 — absolute HTTP deadline and terminal payload eviction

Independent review found that urllib's socket inactivity timeout did not bound a slowly trickling body, and that confirmed failed/timed-out runs retained full submissions in `pending`. Both were reproduced before the fix: the slow-body regression did not raise `TimeoutError`, and clean failed/timeout records remained. RED command:

```text
.venv/bin/python -m pytest -q rl/verifiers/tests/test_warden_http_deadline.py rl/verifiers/tests/test_warden_adapter.py -k 'absolute_deadline or terminal_evicts'
3 failed, 30 deselected in 0.26s
```

The HTTP exchange now runs in one owned trusted `-I -S` worker, with bearer/request only over stdin and bounded output. The parent applies an absolute monotonic deadline, reserves cleanup time, and kills/reaps the worker on timeout. A failure to confirm local reaping retains ownership/PID and blocks new HTTP launches. This is local request cleanup, **not remote cancellation**. After an accepted authoritative terminal state, matching safe spec, typed result and remote cleanup, every outcome evicts the full payload; log failures preserve the known cleanup status. Malformed/absent authority or uncertain remote work retains retry identity.

After self-review, focused covering tests and lint passed:

```text
.venv/bin/ruff check rl/verifiers/warden.py rl/verifiers/_warden_http.py rl/verifiers/tests/test_warden_adapter.py rl/verifiers/tests/test_warden_http_deadline.py
All checks passed!
.venv/bin/python -m pytest -q rl/verifiers/tests/test_warden_http_deadline.py rl/verifiers/tests/test_warden_adapter.py rl/verifiers/tests/test_verifiers.py
........................................................................ [ 67%]
..................................                                       [100%]
106 passed in 5.65s
```

Raw final focused output: [fix1-focused.txt](verifiers/fix1-focused.txt). Tests use actual `HTTPResponse` socketpair I/O in the owned worker for slow connect/header/body, prompt/invalid/oversized responses, wrapped inactivity timeout, and uncertain POST identity. The opener is a trusted socketpair seam; no external service, DNS request or live TLS connection is made. Every real fixture worker is reaped and its pipes closed. Separate synthetic process fixtures exercise denied signalling/unconfirmed reaping, retained PID ownership and blocked subsequent I/O without leaving any actual process alive. Protocol fixtures cover success/failed/interrupted/timed-out eviction, log errors after confirmed cleanup, malformed/uncertain retention, and 200 consecutive clean failures with an empty pending map.

The earlier **489-test full CPU result is pre-fix evidence**. Only the 106 focused covering tests were run for this fix; no full suite, GPU, native runtime, service or cloud action was repeated. Real gVisor/Warden prerequisites remain unchanged and unverified.
