# Task 5 — fault injection and goodput

**未在真实多机上验证。** Budget is $0; multi-node fault experiments are paused.
No GPU power/network/process/node operation, SSH, scheduler command or cloud
mutation was executed. CPU command shims and injected clocks are the only execution
behind the evidence below. No actual two-hour training or fault measurement exists.

Implemented a fixed 7,200-second offline schedule for a single-GPU UUID power cap,
a specific InfiniBand switch-port width change plus reset, and abrupt loss of one
persistent VM node. The VM method is not physical chassis power loss. Rank exit,
Slurm DOWN and TCP netem are not substituted for these effects. Generated commands
include exact scope, raw-baseline queries, original-state restoration and verification.
All apply/launch CLI paths remain hard-closed under the $0 resource budget even if ordinary
approval environment variables are set. See [fault README](../../cluster/faults/README.md).

The event protocol records Task 3 detection and actual isolation command times,
retains batch failure evidence, and computes checkpoint lineage across successful
saves, failed attempts and restarts. The separate checkpoint-enabled thin TorchTitan
adapter preserves the upstream training loop and connects synchronized successful
steps, native full-state saves, successful loads and final retained state. It adds
rank-keyed RNG and checkpoint-generation verification; no Task 4 fresh-run metrics
are reused as fault goodput.

| Measurement | Real multi-node result |
| --- | --- |
| 2-hour training completion | Not run |
| Detection / drain / checkpoint recovery latency | Not measured |
| Overall goodput / lost steps | Not measured |
| Repeated baseline/candidate median and range | Not measured |

CPU-only regression example (synthetic clock, **not a performance measurement**):
three original 2-second steps, checkpoint after step 2, rollback losing step 3,
3 seconds idle, 4-second checkpoint restore, a 3-second redo of step 3, and a final
checkpoint. Including two 1-second saves gives 18 seconds synthetic wall time,
7 seconds useful retained work, 2 seconds discarded completed work, 4 completed
attempts, and goodput `7 / 18`. The redo is counted once; failed partial attempts
and unsaved tails remain denominator overhead. The required
`test_goodput_accounts_restart_and_lost_steps` covers these values.

- [Generated offline schedule](raw/task5-fault-plan.json) — example identities only.
- [CPU event replay fixture](raw/task5-goodput-fixture.json) — simulated events/results.
- [Focused CPU tests](raw/task5-cpu-focused.txt).
- [Final full CPU test suite](raw/task5-cpu-suite.txt).
- [Earlier pre-window full CPU suite](raw/task5-cpu-suite-pre-window.txt), retained history (353 passed).
- [Pre-supervisor-fix full suite](raw/task5-cpu-suite-pre-supervisor-fix.txt), retained history (358 passed).

Timing uses one surviving coordinator clock domain. Prometheus/source times remain
raw evidence, not subtractable local monotonic values. Detection includes collection,
polling and the 50-step reporting interval. Drain time excludes requeue; missing or
failed isolation remains unmeasured. Device/VM restoration is separate from training
checkpoint recovery. Actual onset is bounded by injection start and verification;
confirmation-to-detection is not a precise physical-onset latency.

Real acceptance still needs dedicated authorized leases, provider approval and
identity/path checks; pinned runtime/DCP storage and per-rank recovery validation;
checkpoint retention/storage checks; and three alternating matching-condition
baseline/candidate rounds with median and range. No existing pause is lifted by
these artifacts. Primary command and pinned API references are in the README.

The reviewed time-based adapter stops new steps at 7,080 seconds using one rank-zero
broadcast decision. Its 120-second reserve covers final full-state save and worker
exit; workers release before 7,200. The surviving coordinator observes closure later,
recording actual wake time and observation lateness without falsifying an exact
7,200 timestamp. Actual worker/save/whole-scope exit overruns remain invalid. Goodput
uses actual observed wall time. Recovery has a 300-second configured budget.

The owned launcher uses a pre-exec permission gate and an exclusive existing cgroup-v2
scope. The scope must be attached and verified before training can fork. Recursive
populated=0, followed by launcher reap, is the completion condition. TERM uses verified
member pidfds; KILL uses the pinned cgroup subtree. Lease/owner/path replacement or
missing delegation fails closed. No real cgroup, PID signal, process launch or hardware
operation was used to claim production acceptance.

Strict retention now disables the upstream asynchronous purger in a config copy and
uses rank-zero synchronous pre-save slot reservation/post-save confirmation, keeping
the active confirmed recovery anchor (including an explicitly restored older checkpoint) and never exceeding two. Unknown/incomplete/foreign
paths are retained and block saving. All ranks receive the same outcome.

Initial validation history remains above (43 focused/361 full before review). The
[review-fix focused evidence](raw/task5-fix1-focused.txt) covers all five findings and
[review-fix Ruff evidence](raw/task5-fix1-lint.txt) records static checks. No new full
suite was required for these scoped changes. 未在真实多机上验证。

Round 2 review fix: explicit step-2 recovery with retained step-2/step-4 now protects
step-2 while replay replaces superseded step-4 with generation 1. Failed new saves,
including partial writes, retain the anchor and at most two directories; unknown
partial evidence remains for reconciliation. Existing same-step periodic final
state is reused without overwrite. [Scoped CPU output](raw/task5-fix2-focused.txt)
and [Ruff output](raw/task5-fix2-lint.txt) cover this fix; no new full or real GPU run.
