# Exact logical-request accounting

`AccountRequests(planned []string, attempts []Attempt) (Accounting, error)` is a
sequential CPU utility independent of the existing replay transport and Record.
It neither schedules retries nor performs faults.

`Attempt` contains `RequestID string`, `Number int`, `Status string`, `Error
string`, and `Final bool`. Attempt numbers start at zero and are contiguous.
Every attempt is terminal `completed` or `failed`. `Final: false` is an explicit
retry handoff: the attempt must have failed and its contiguous terminal successor
must be present. `Final: true` closes the chain and permits no later attempt.
The caller must supply the actual handoff/finality information; this utility
cannot discover a retry the caller omitted or falsely marked closed.

Every planned logical ID needs a complete terminal chain. Blank/duplicate planned
IDs, missing/unknown requests, invalid/sparse/duplicate attempt identities,
open retry tails, success before retry, and conflicting closures return an error
and zero Accounting, never valid-looking partial counts. An empty plan is invalid.

A single attempt is classified `completed` or `failed`; any chain with multiple
attempts is classified `retried`. Therefore `Completed + Failed + Retried ==
Planned` exactly. `RequestAccounting.FinalStatus` separately preserves its final
completed/failed outcome, including failed retries, with `AttemptCount` and every
copied attempt/error. Retried does not imply eventual success.

Results sort requests by ID and attempts by number without modifying or aliasing
input slices. There is no shared mutable state or concurrency framework. The
Task15 fixtures are synthetic transport-independent observations, with no claim
of actual injected faults or multi-machine validation.
