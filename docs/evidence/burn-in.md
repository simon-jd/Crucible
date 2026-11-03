# Task 2 burn-in evidence

**未在真实多机上验证。** This is local CPU implementation evidence, not cluster
acceptance. Budget: **$0**. No GPU workload, remote GPU connection, rental,
installation or host/network service change was run.

The runnable default is:

```bash
.venv/bin/python -m cluster.health.report --output /tmp/crucible-burnin-UNIQUE
```

The checked-in [offline plan](burn-in-local/offline-plan-final/plan.json) contains four
DCGM level-3 commands, four actual-topology probes, 96 directed cross-node HCA
perftest configurations and 44 NCCL collective/subset configurations. These are
planned commands only. Its [summary](burn-in-local/offline-plan-final/summary.json)
has `status=planned-unverified`, `healthy=false`, 148 missing execution results.
The synthetic example has two GPUs/two HCA devices per node; it is not an H100,
NIC or RTX 4090 hardware observation.

Every performance configuration is scheduled for three alternating
baseline/candidate rounds (840 invocations in this example). **相同配置健康重复，不代表优化对比。**
The public experiment-results module enforces same conditions and reports
median/min/max. Individual slow/error/unchecked/failed/timeout measurements
cannot be hidden by aggregation. Remaining tasks stay missing after a failure.

The [real upstream fixture](../../cluster/health/tests/fixtures/allreduce_upstream.txt)
comes from [NVIDIA/nccl-tests issue 389](https://github.com/NVIDIA/nccl-tests/issues/389),
reporting B200 hardware, nccl-tests 2.19.6 and NCCL 23007. These are upstream
user-reported values, not Crucible observations. Its actual binary build commit
was not supplied; [provenance](../../cluster/health/tests/fixtures/provenance.json)
keeps that unknown separate from the pinned v2.19.6 source-format commit
`a0b82b2260cf5152b9f8c061bbf7eaf0ba096432` and records hashes/configuration.
Synthetic slowdowns and local Python command shims are explicitly labelled in
tests. No command-shim result is asserted to be a real multi-node result.

The initial [RED output](burn-in-local/red.txt) shows collection failing because
the parser did not yet exist. [Focused GREEN output](burn-in-local/green-focused.txt)
records parser, coverage, threshold and process-cleanup tests. The final
[full-suite output](burn-in-local/green-full.txt) and
[exit record](burn-in-local/green-full-exit.json) record the one pre-commit full run.
The complete CPU shim test exercises every matrix task, six measurements per
performance task, units/correctness/failures, checkpointing and statistical
summaries; it substitutes local Python processes for all SSH/MPI/GPU programs.

Threshold fractions and units are explained in
[health README](../../cluster/health/README.md). No theoretical H100/NIC values
are invented: actual inventory, reviewed peak/latency references, source/build
revisions, runtime versions/digests and a successful real Task 1 baseline remain
required site inputs. The current Task 1 lock is still candidate-unverified.
NCCL alltoall's upstream unsupported in-place validation is disclosed and excluded
from supported-mode summaries; N/A elsewhere fails closed. Perftest measures
unidirectional average write bandwidth and median read latency, not a combined
bidirectional bandwidth or peak-sample metric.

Future real acceptance remains paused pending new user authorization and the
resource rules. It must verify actual four-node DCGM/fabric/NUMA behavior, full
matrix coverage and measured fractions against reviewed theory, collect the raw
reports, and verify process cleanup on the real transport. Local tests establish
none of those hardware observations.

## Review fix round 1

The first independent review requested two corrections: stop immediately after
an initial affinity/DCGM failure, and bound remote lifetimes independently of
local transport lifetime. Both are covered by new CPU regressions.

Every initial result now saves its token cleanup proof and returns on failure;
all later tasks remain missing. DCGM/affinity, perftest and MPI ranks now receive
absolute epoch deadlines no later than the operator cutoff. Readiness/setup
consume the existing pair budget, and each launch recomputes its remaining time.
The controller explicitly stops and verifies tokens after success, timeout,
exception and cancellation; false/missing/mismatched proof is failure and remains
in checkpointed results. A lock and stop tombstone block delayed launches after
cleanup; PID/start-time checks avoid signalling a reused process identity.

[RED](burn-in-local/fix-round-1/red.txt) records four reproduced failures.
[Focused GREEN](burn-in-local/fix-round-1/green-focused.txt) includes CPU proxy
cleanup records for disconnected, timed-out, exceptional, cancelled and
unreachable-cleanup cases. In these tests an exited local proxy leaves an
independently detached CPU worker and its sleep descendant; the worker's absolute
deadline and explicit token stop are tested without SSH, MPI or GPU execution.
[Final full output](burn-in-local/fix-round-1/green-full.txt) and
[exit record](burn-in-local/fix-round-1/green-full-exit.json) record verification.

Absolute cutoffs assume synchronized, stable controller/node UTC clocks; skew
is not measured here and a slow node clock can delay its cutoff by that skew.
Cleanup verification has a bounded grace period after workload cutoff. Private
completion receipts/stop tombstones/locks remain through the run to prove exit
and prevent late starts; active process state is removed. This is not a real
network-partition test or proof against machine failure, an uninterruptible task
or SIGKILL of the deadline worker. **未在真实多机上验证。** Budget remains $0.

Final fix-round-1 repository run: **221 passed in 56.78s**, exit **0**.
Only one repository-wide test execution was run for this fix. Ruff/shell/diff
checks passed; offline CLI still reports 148 missing results and healthy=false.
