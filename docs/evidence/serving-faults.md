# Task 15: kill-decode fault with request accounting, one RTX 4090 — 2026-10-27

## Update (phase 5, 2026-10-28): client resend of cut streams — 0 failures per kill, 1 retried

The replayer can now resend a request (`-client-retries N`, 0–3; off by default). It resends
only on a fault on the path: a stream cut mid-response (`stream read: …`), HTTP 502/503, or a
transport error. It never resends a malformed stream, another HTTP status, or after the
request's deadline. The resend stays inside the original deadline and inflight slot. The record
keeps every attempt (`Record.Attempts`). `SentMs` stays the first attempt's emission, so
sender validity is unchanged, and TTFT is measured from it, so the lost time stays in the latency.
`AccountRecords` turns each attempt chain into the `retried` category
(`Completed + Failed + Retried == Planned`) and adds `RetriedCompleted`. The resend is new work
on the server: the plan's accounting counts it as retried, not as completed
(`serving/replay/retry_test.go`, `cmd/replay/retry_test.go`).

Same setup as phase 4 (2 replicas, 3 req/s, 90 s, 260 requests, SIGKILL of `b` at 30 s), router
with ejection and `-retry-dial` in both kill conditions. Three conditions alternated over 3 pairs
(order kill-client-retry, kill-retry, none, none, kill-retry, kill-client-retry, kill-client-retry,
kill-retry, none). All 9 trials were sender-valid and every one had `Completed + Failed + Retried
== Planned`. Machine: 32 cores, load average 3.0 / 3.1 / 4.0 at the end, 5.8 when queued (`machine-load-queued.txt`), with
other projects running (Keel's GPU matrix). Peak owned RSS 5.72 GB.

| Condition | Completed / Failed / Retried (of 260) | Kill → ejection (s) |
|---|---|---|
| kill, eject + `-retry-dial` + client resend once | 259 / 0 / 1 (retried request completed), in all 3 trials | 0.035 / 0.025 / 0.034 |
| kill, eject + `-retry-dial` (phase-4 condition, same session) | 259 / 1 / 0, in all 3 trials (`stream read: unexpected EOF`) | 0.52 / 0.47 / 0.61 |
| none | 260 / 0 / 0, in all 3 trials | no ejection |

- In each kill-client-retry trial, the one request streaming from `b` at the kill was cut
  1.02–1.04 s after it was sent. It was resent 0.1 ms later, served by `a`, and completed
  (383–424 tokens). Its TTFT, from the first send, was 1,036–1,064 ms. In the control trial
  `trial-2-none`, TTFT is p50 20.2 ms, p99 69.2 ms, max 74.5 ms. So the cost of the fault stays
  visible in the latency numbers.
- Ejection is faster with the client resend (about 0.03 s against about 0.5 s) because the
  resend itself is the first request the router sends to the dead `b`. It gets connection
  refused, is ejected and dial-retried to `a`. Without the resend, ejection waits for the next
  scheduled arrival.
- Controls had 0 ejections and 0 retried, so the resend never fired without a fault.
- Raw outputs: `t15-faults/kill-decode-client-retry/` (per-trial `replay.json` with
  `Accounting` and `Attempts`, `router.log`, `events.json`; `fault-summary.json`). Command:
  `bash scripts/run_localscale_gpu.sh fault <out> --phases 90:1 --base-rps 3 --kill-at 30 --pairs 3
  --fault-conditions kill-client-retry,kill-retry,none` (after `make t14-bins`).

## Update (phase 4): router ejects on transport errors — loss 32–34 → 0–3 per kill

The router (`serving/router`) now ejects a backend on its first transport error, unless
the client itself cancelled. The ejection holds until the adapter serves a snapshot
observed after it. Re-serving the dead replica's last scrape does not readmit it. With
`-retry-dial`, a request whose connect failed is resent once to another backend. The
POST never reached the dead backend, so it cannot run twice. Errors after a backend
accepted the request are never resent (`serving/router/eject_test.go`).

Same setup as below: 3 req/s, 90 s, 260 requests, SIGKILL of replica `b` at 30 s. Three
conditions alternated over 3 rounds (order kill, kill-retry, none, none, kill-retry,
kill, kill, kill-retry, none). All 9 trials were sender-valid, and every trial had
`Completed + Failed == Planned` (0 retried at the replayer, 0 lost). Machine: 32 cores,
load average 4.7 / 5.6 / 6.0 (1/5/15 min) at the end, with other projects running (Keel's GPU matrix, PathWatch).

| Condition | Failed per trial (of 260) | Failure kinds | Kill → ejection (s) |
|---|---|---|---|
| kill, old router (phase 3, below) | 34 / 32 / 32 | HTTP 502 until the snapshot aged out (+14.06 s), plus 0–1 cut streams | — (snapshot expiry ≈ 14 s) |
| kill, eject | 2 / 2 / 3 | 1–2 cut streams (in flight at the kill) + 1 × HTTP 502 | 0.44 / 0.46 / 1.25 |
| kill, eject + `-retry-dial` | 1 / 0 / 1 | 0–1 cut streams only | 0.44 / 0.44 / 0.44 |
| none | 0 / 0 / 0 | — | no ejection |

- Every ejection was `dial tcp 127.0.0.1:31102: connect: connection refused`. It was
  logged by the first request routed to `b` after the kill
  (`trial-*/router.log`, `router stats` line at shutdown).
- What is left is requests already streaming from `b` when it died
  (`stream read: unexpected EOF`, scheduled 1.07–1.22 s before the kill). They cannot be
  resent without risking duplicate work. Recovering them needs a client-side retry,
  which the accounting would then count as `retried`.
- Without `-retry-dial`, the request that discovers the dead backend still fails
  (HTTP 502). With it, that request is served by `a`.
- Controls had 0 ejections, so the router did not eject healthy backends.
- Peak owned RSS 5.89 GB. Raw outputs: `t15-faults/kill-decode-eject/` (per-trial
  `replay.json` with `Accounting`, `router.log`, `events.json`; `fault-summary.json`).
  Command: `bash scripts/run_localscale_gpu.sh fault <out> --phases 90:1 --base-rps 3
  --kill-at 30 --pairs 3 --fault-conditions kill,kill-retry,none`.

## Phase 3 run (router without ejection)

Result: killing a serving replica mid-run loses no request silently. In 3 kill trials, all 260
scheduled requests ended as completed or failed (`Completed + Failed == Planned`, 0 retried),
with 32–34 failures each. In the 3 no-kill controls, 260/260 completed. Failures are the requests
in flight on the killed replica plus everything the router sent it until its last metrics snapshot
aged out (≤15 s). Other projects were running (32 cores, load average 5.6 at the end).

## Setup

- Two vLLM replicas of `Qwen/Qwen2.5-0.5B-Instruct` (same settings as
  [autoscaling.md](autoscaling.md#setup)) behind the Go router, with the metrics adapter.
  No autoscaler.
- Load: 90 s of seeded Poisson arrivals at 3 req/s with Azure conv lengths (260 requests). Sent by
  the Go replayer (`-api openai-text -account`, 60 s deadline). The replayer makes one attempt per
  request and never retries.
- Fault: 30 s after replay start, `serving/faults/kill_decode.sh <pgid>` SIGKILLs replica `b`'s
  process group. The script refuses any process that is not a vLLM OpenAI server group leader.
  Nothing tells the router; it learns only when `b`'s snapshot goes stale.
- Plan: kill / none alternating over 3 pairs (order kill, none, none, kill, kill, none). Replica `b` is
  freshly started before every trial.
- Accounting: `replay.AccountRecords` (the `serving/replay/accounting.go` chain check) on every
  record. Its counts are in each `replay.json` under `Accounting`.

## Results

| Trial | Condition | Planned | Completed | Failed | Failure kinds | Failures scheduled (s from kill) |
|---|---|---|---|---|---|---|
| 0 | kill | 260 | 226 | 34 | 33 × HTTP 502, 1 × stream EOF | −1.07 … +14.06 |
| 1 | none | 260 | 260 | 0 | — | — |
| 2 | none | 260 | 260 | 0 | — | — |
| 3 | kill | 260 | 228 | 32 | 31 × HTTP 502, 1 × stream EOF | −1.22 … +14.07 |
| 4 | kill | 260 | 228 | 32 | 32 × HTTP 502 | +0.37 … +14.06 |
| 5 | none | 260 | 260 | 0 | — | — |

- Every kill script call exited 0, and the kill landed 30.03 s after replay start.
- The stream-EOF failures are requests cut mid-response. The failures scheduled before the kill
  (down to −1.2 s) were already running on `b`. The HTTP 502s are new requests the router
  still sent to the dead replica. They stop 14.06 s after the kill, which is the router's 15 s
  snapshot freshness bound (the adapter keeps the last successful scrape).
- So detection time is set by snapshot freshness, not by the fault. A health check on scrape
  failure, or a router retry on connect errors (POSTs are deliberately not replayed today), would
  shorten the loss window. Neither is implemented.
- Peak owned RSS was 5.71 GB.

## Not done

- `serving/faults/nic_down.sh` is written for the multi-node cluster and exits 3 ("paused") unless
  `CRUCIBLE_MULTINODE_APPROVED=1`. It was not run: 未在真实多机上验证.
- Killing a decode Pod in a P/D deployment (Kubernetes, separate prefill/decode) is paused with
  Task 13 P/D. Here a whole colocated replica is killed.
- Raw outputs: `t15-faults/kill-decode/` (per trial `replay.json` with `Accounting`, `events.json`,
  `timeline.json`, router and server logs; `fault-summary.json`). Command:
  `bash scripts/run_localscale_gpu.sh fault <out> --phases 90:1 --base-rps 3 --kill-at 30 --pairs 3`.

# Earlier CPU phase: exact request accounting

This phase implements a terminal-chain accounting utility independent of frozen
replay transport/Record/binary. All examples are synthetic observations. No Pod,
NIC, network or service fault was injected. 未在真实多机上验证. Whole Task15 and
the project remain open.

Every predeclared logical ID needs contiguous terminal attempts from zero. A
nonfinal failed attempt explicitly hands off to a required successor; one final
marker closes the last attempt. Stopping records is never interpreted as finality.
An omitted whole request, missing retry successor, sparse sequence, duplicate,
unknown ID, success before retry, or conflicting/open closure is an error with
zero valid-looking partial counts.

Single-attempt chains classify as completed/failed; multi-attempt chains classify
as retried. These exclusive categories sum exactly to the number planned.
A retried chain separately preserves its final completed/failed status, every raw
error and attempt, and its attempt count. A failed retry cannot disappear into a
success-looking retried count. Results are deterministic copies and do not mutate
caller slices. The caller must truthfully record closure/handoff information;
the utility cannot detect a retry omitted while falsely declaring final closure.
See [accounting API](../../serving/replay/accounting.md).

The named loss test is `TestAccountingNoSilentLoss`. Targeted Go RED/GREEN, vet,
formatting, monitored build resources and exact owned-cache cleanup are documented
in the Task15 implementation report and compact `raw/task15-cpu-*` evidence.
The tests cover the utility; they do not validate fault actuators or real service
outcomes. Existing Go/native/ML/replay suites were not rerun.

Still required: implement safe `serving/faults/kill_decode.sh` and `nic_down.sh`,
integrate truthful terminal attempt/finality observations, inject actual faults,
and verify request conservation under those faults. Deferred SpecForge training,
heldout/model inference and batch1–64 TPOT are also required. No placeholder
trainer or actuator was added. Task12's rejected v3 native launch remains
unapproved and was neither retried nor bypassed.
