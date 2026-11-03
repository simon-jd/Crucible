# Burn-in suite

**未在真实多机上验证。** Current budget is **$0**, multi-node execution is paused,
and only the local RTX 4090 is available. This task runs no GPU workload, SSH,
remote GPU connection, paid allocation, installation or host/network service change.
The live commands below are future operator interfaces, not authorization to use them.

The suite is implemented for an already provisioned, approved four-node site.
Task 1's checked-in `versions.lock` is still `candidate-unverified`, with null
digests; it cannot supply the real runtime provenance or baseline required here.
This suite does not provision a cluster, build images, install tools, start DCGM
daemons or change NIC/GPU settings.

From the repository root, the default command produces a complete offline plan
without starting any subprocess or making any network connection:

```bash
.venv/bin/python -m cluster.health.report --output /tmp/crucible-burnin-UNIQUE
bash cluster/health/dcgm_diag.sh  # prints the DCGM plan only
```

Use a fresh output directory. The example inventory is explicitly synthetic,
including its PCI/NUMA layout, and contains no observed H100, 4090 or NIC values.
It has no usable runtime digests, link theory, latency references, deadline or
authorization. Its offline summary is `planned-unverified`, `healthy=false`;
zero executed measurements never means healthy.

For four nodes with two HCA devices each, the plan contains:

- 4 DCGM level-3 commands (`dcgmi diag -r 3 -j`), with bounded remote helpers;
- 4 read-only affinity probes, checking actual GPU index/PCI identity, every HCA,
  active link rate and NIC/GPU sysfs NUMA association against inventory;
- 96 perftest configurations: every ordered pair of distinct nodes × every
  server/client HCA pair × `ib_write_bw` and `ib_read_lat`. Both ends use
  `--use_cuda <mapped GPU index>`, the chosen HCA port and `numactl` bindings;
- 44 NCCL configurations: every one-node subset (4), two-node subset (6) and
  four-node subset (1) × all_reduce, all_gather, reduce_scatter and alltoall.
  Every configuration scans 128 MiB, 256 MiB, 512 MiB and 1 GiB, with correctness
  enabled (`-c 1`). One OpenMPI process per GPU uses an explicit host/slot map,
  GPU selection, HCA selection and CPU/memory NUMA binding.

Each of the 140 performance configurations runs baseline/candidate in three
alternating rounds, giving 840 performance invocations, plus the eight initial
checks. **相同配置健康重复，不代表优化对比。** Both labels execute identical settings;
there is no optimization gain claim. Raw values are checked individually, then
`scripts/experiment_results.py` validates same-condition alternation and reports
medians/min/max. Any failed command, threshold, correctness check or cleanup stops
the remaining stress matrix. Missing configurations keep the final result false.

## Units and interpretation

`thresholds.yaml` stores fractions, for example `0.70 = 70%`, never `70`.
The policy values are candidates, not measured hardware capability.

NCCL reports bytes, microseconds and decimal GB/s (10^9 bytes/sec). Both
out-of-place and in-place columns, including both `#wrong` columns, are parsed.
N/A correctness is rejected by default. Upstream alltoall explicitly does not
support in-place validation; that mode is retained and labelled unsupported,
excluded from performance summaries, and never presented as validated. Its
out-of-place correctness is mandatory. Any actual nonzero error count fails.
The parser deliberately rejects alternate column formats such as optional
per-iteration timings or timestamps, rather than guessing their positions.

NCCL bus bandwidth is algorithm bandwidth times 2(N−1)/N for all_reduce and
(N−1)/N for all_gather, reduce_scatter and alltoall. Supply a theoretical
**per-rank bottleneck bus bandwidth** for every collective/node-subset key in
`inventory.nccl_theory`, plus a written basis. Account for NIC sharing, rails,
intra-node fabric, PCIe/NUMA paths and rank topology; do not divide a marketing
aggregate bidirectional number by rank count without justification. No H100
NVLink or NIC peak is assumed here.

Perftest is **unidirectional**: no `-b` flag. It requests `--report_gbits` and
converts `Gb/sec` to GB/s by dividing by eight; default `GB/sec` is also accepted.
The selected metric is **average** write bandwidth, not peak bandwidth.
Theoretical write ceiling = min(server link_gbps, client link_gbps) / 8, explicitly
not the sum of send and receive rates. This line-rate ceiling excludes protocol
overhead; the site must review the selected fraction for its fabric.
For reads, the selected `t_typical[usec]` is the median microsecond value.
Latency efficiency = site-supplied ideal_read_latency_us / observed median_us;
the 0.50 policy therefore permits at most twice that ideal. This ideal is a
reviewed reference, not a latency derived from link bandwidth. Perftest's latency
benchmarks use their own one-way/RTT semantics; asymmetry and read completion
cost must be considered when supplying the reference. Both endpoints need a
positive reference; the pair uses the larger reference. Server exit status and
raw output are required even though upstream `ib_read_lat` only measures on the
client. No fabricated server measurement is required.

## Future site inputs and execution

After explicit new user authorization, prepare a separate site inventory:

1. Replace synthetic node names/SSH targets/addresses, contiguous GPU indices,
   real PCI bus IDs/NUMA IDs, **all** HCA devices/ports, GPU mappings, theoretical
   link rates and ideal read latency. Every GPU needs an HCA mapping.
2. Set `synthetic=false`; supply separate runtime fields for driver, CUDA toolkit,
   NCCL library version, OFED, DCGM, perftest, OpenMPI, node-image ID/SHA256, OCI
   image digest and full **nccl-tests source commit**. A library version is not a
   tool commit. Record these actual versions from the approved site; local
   synthetic tests do not establish them.
3. Point `baseline_summary` at a real successful Task 1 `summary.json`. Review it
   and its node/runtime reports for this exact site; the suite checks `passed`
   and records the supplied provenance, but does not independently reproduce
   Task 1's entire baseline. Keep that baseline evidence alongside this run.
4. Fill all 44 `nccl_theory` keys (shown by the plan), each with
   `theoretical_busbw_gbs` and `basis`. Review `dcgm_required_tests` against the
   site's DCGM version; every required test must cover every GPU with `Pass`.
   Any reported Warn/Skip/Fail or missing result fails closed.
5. Place this reviewed checkout, inventory and pinned MPI-enabled nccl-tests
   binaries at the declared existing shared paths on all nodes. Preapprove SSH
   host keys for both SSH targets and MPI host addresses. The OpenMPI backend
   requires the same explicit SSH username on all nodes (or the same default
   user) and carries that user into its launch agent. This executor neither
   transfers files nor installs software.
6. Supply `execution.owner`, `execution.approval_reference`,
   `execution.max_cost_usd=0`, and timezone-aware `execution.stop_at_utc` within
   the next two hours. The current zero-dollar ceiling remains enforced;
   rentals require a separately authorized budget change. The suite records
   stop conditions and stops at the first failure or operator deadline.

Future invocation, **not run in this task**:

```bash
.venv/bin/python -m cluster.health.report --inventory /path/approved-site.json \
  --output /path/new-evidence --execute --approval-reference USER-APPROVAL-RECORD
```

The CLI approval reference must match the reviewed inventory. Nothing in this
example constitutes approval. GPU work on the shared local 4090, if separately
requested later, must use the project's existing approved GPU guard and shared
`flock /var/tmp/rtx4090.lock`; this four-node SSH executor is not that workflow.

`plan.json` preserves inputs, hashes, matrix and raw argv. `results.json` is
checkpointed after each measurement and preserves argv, stdout, stderr, exit
status, timeout, parsed rows, violations, rounds and statistical summaries.
`summary.json` lists failures and missing tasks. Evidence directories are never
overwritten. A failed affinity/DCGM result is checkpointed and stops all later
workloads immediately. Both checks use the token/deadline helper; a local
transport exiting never proves remote exit.

Every launch receives an absolute Unix epoch deadline, at most the operator's
`stop_at_utc` and the earlier task time cap. Server readiness and setup consume
the same perftest deadline: the client never receives a restarted time budget.
Every MPI rank receives the same deadline for that measurement. Helpers recheck
remaining time before launch and while waiting, independently of their transport.
The controller explicitly sends token `stop` for affinity/DCGM, both perftest
endpoints and every MPI rank after success, timeout, exception or cancellation.
Results retain cleanup argv/status/output and token-matched `verified=true`
proof; missing, malformed or failed proof makes the task fail. Verification has
its own bounded grace period after workload cutoff.

The private 0700 state directory and PID/start-time checks protect other tasks.
Launch and stop use a token lock; a stop tombstone prevents a delayed connection
from launching that token after cleanup returned. Workers kill/reap their owned
process groups and remove active `.json` state. Completion receipts, stop
tombstones and lock files remain in the private temporary directory to support
verification and prevent late starts; they are small audit/control artifacts,
not active processes. Retain them through the approved run's deadline and review
before later operator cleanup. Readiness checks the server's owned listening
socket without consuming the connection.

Absolute epochs require synchronized, stable UTC clocks on controller and nodes;
the suite does not measure or certify clock skew. Review clock offsets before
future approved execution. A node clock behind the controller can delay its
local cutoff by that offset, plus scheduling latency. These assumptions and
transport behavior remain unverified here. CPU detached-worker tests simulate
losing a local connection without signalling the worker; they are not real
network-partition acceptance. Abrupt machine failure, uninterruptible kernel
work or SIGKILL of the deadline worker cannot be certified locally. Missing
cleanup proof stays unverified; operators must inspect the real site.

## Fixtures and local validation

`tests/fixtures/allreduce_upstream.txt` is a verbatim real user-reported output
from [NVIDIA/nccl-tests issue 389](https://github.com/NVIDIA/nccl-tests/issues/389).
It reports two B200 nodes and version 2.19.6; it is **not** a Crucible observation.
The actual reported binary build SHA was not supplied upstream and remains null.
`provenance.json` records source/update time, exact SHA256, configuration, NCCL
library/header versions, and the independently resolved v2.19.6 format-reference
commit `a0b82b2260cf5152b9f8c061bbf7eaf0ba096432` with source hashes. That reference
commit is not falsely attributed to the reported binary.

Synthetic slowdowns are derived in tests and labelled there. CPU command shims
replace every SSH/MPI/DCGM/perftest executable with a local Python print process;
their result argv identify those actual shims. They test the entire orchestration,
failure/checkpoint/cleanup logic and statistical contract, not the fabric.

```bash
.venv/bin/python -m pytest cluster/health/tests -q
.venv/bin/ruff check cluster/health
bash -n cluster/health/dcgm_diag.sh
```

Primary format references: [NCCL performance definitions](https://github.com/NVIDIA/nccl-tests/blob/a0b82b2260cf5152b9f8c061bbf7eaf0ba096432/doc/PERFORMANCE.md),
[upstream perftest](https://github.com/linux-rdma/perftest),
[DCGM diagnostic documentation](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html).
