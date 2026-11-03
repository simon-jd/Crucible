# Sender diagnosis after actual full02v3 failure

The complete native baseline failed. Its original three Burst calibrations each
had one of 23 requests sent more than 5 ms late, exceeding the unchanged 1% rule.
All three delayed the same index 2, scheduled at 31,000 ms, by
16.378060 / 16.589153 / 17.147177 ms. Azure evaluation 4 also violated the rule
with two late sends among 132 requests. Original failures and all not-run members
remain in [full02 terminal](raw/task12-baseline-full-02/terminal.json) and the
[independent raw analysis](raw/task12-full02-root-analysis-resume-24.json).

All payloads are prepared before the Go run clock starts. `SentMs` records the
complete request-body socket write, including preceding scheduling and connection
work; it does not identify timer wakeup, connection acquisition or individual
write delays. The delayed Burst body is smaller than the timely first body.
These facts exclude neither runtime/host interference nor network-write effects;
they do not prove a bottom-level cause.

## One CPU diagnostic, separate from native performance

Commit `a5b43bf` adds a bounded, short-lived loopback fixture and lifecycle proof.
It uses the original frozen replay binary, original 23-row Burst calibration,
seed 17, multiplier 1, admission limit 192 and 60-second request deadline. The
fixture supplies immediate native-shaped SSE with all requested output IDs in one
event. These coalesced responses change streaming allocations, pacing and idle
durations; this is a client screen, not model compatibility, SLO or goodput proof.

Sixteen new focused cases passed once in 0.33 seconds. They cover exact request
identities, malformed HTTP, bounded append refusal, evidence EIO, keepalive,
normal/timeout cleanup, actual child hard limits, initial/later process-sampling
failure cleanup, PID identity mismatch and duplicate/unknown requests. Test names
and output are in [focused log](raw/task12-sender-cpu-01-focused.log). Temporary
files were inventoried and removed only from the exact owned pytest tree.

The single diagnostic completed in 59.036011 seconds. Its outer exit 0 means the
diagnostic completed; the original replay exited **1, sender-invalid**. All 23
records were `ok`, no raw abort occurred and all 1,556 requested output tokens
were counted. Index 1 was 6.790109 ms late and index 2 was 17.557943 ms late.
Both remain in the denominator; no performance result is accepted.

The original index-2 pattern therefore recurs with no native model or GPU task.
The investigation is now focused on client/runtime/host/loopback behavior. This
does not establish that the two runs share the same internal cause. No timing
threshold, trace row, CPU priority, affinity, runtime setting or client source was
changed to obtain a pass; no repeated trial followed.

See the [raw terminal](raw/task12-sender-cpu-01/terminal.json),
[result](raw/task12-sender-cpu-01-result.json),
[input/protocol identities](raw/task12-sender-cpu-01/provenance.json), and
[root verification](raw/task12-sender-cpu-01-root-check-resume-24.json).

## Resources and cleanup

The envelope was 90 seconds including cleanup, 512 MB aggregate sampled RSS early
stop and 2 MB total diagnostic scratch/evidence, inside the existing 10 MB CPU
slot and unchanged 690 MB future reserve. Child stdout/stderr each had a 64 KiB
hard file limit; bounded parent writes retained invalidity on errors or caps.
The 582 RSS samples have a verified sum and sampled maximum of 39,669,760 bytes;
this is not a continuous peak guarantee.

The replay child exited 1 and was reaped without TERM. The controller exited;
all fixture threads and sockets closed. A separate read-only host check found
both recorded owned PIDs absent and no listener on port 31013. Original 234
frozen source/artifact entries, executable identities and the 100 coordinator
files remained unchanged. Shared model cache was 23,815,435,170 bytes. The
postcommit project total was 13,297,174,043 bytes; adding the unchanged reserve
remained below 14 GB. No build, download, native import, GPU run, cloud resource,
model API or public publication occurred in this diagnostic.

The independent phase review approves specification and code quality with no findings. Precise
client-phase observation remains the next step; no additional experiment is
authorized by this document. Task 12 and the full project remain incomplete.


## Timing-slack hypothesis, with a non-reproduction

The installed Go runtime's Linux netpoll path uses a millisecond epoll timeout.
The host reports Ubuntu7.0.0-38.38 based on7.0.14. Upstream stable
[select accuracy source](https://raw.githubusercontent.com/gregkh/linux/v7.0.14/fs/select.c)
computes a timeout-dependent slack, normally0.1% for non-positive nice, bounded
by100ms and compared with per-task slack. Its
[epoll implementation](https://raw.githubusercontent.com/gregkh/linux/v7.0.14/fs/eventpoll.c)
uses that estimate for timed waiting. This supports a hypothesis for long-idle
waits; upstream source alone does not attest the exact Ubuntu binary path or
attribute these Go delays.

A separate one-shot CPU probe performed one8-second epoll wait with no registered
event descriptors. At nice0 and observed timer_slack_ns50000, it elapsed
8.000870494 seconds:0.870494ms overshoot, not a >5ms reproduction. The
[raw probe](raw/task12-sender-epoll-cpu-01.json) retains that negative result.
No repeated probe or timer/priority/affinity setting change followed. Direct
client syscall/phase observations are still needed.
