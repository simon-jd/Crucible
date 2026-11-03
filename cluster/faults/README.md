# Fault plans and checkpoint goodput

**未在真实多机上验证。** All multi-node faults are paused at a $0
budget. No GPU power, networking, process/node termination, SSH, scheduler, cloud,
or paid-resource action was run. The shared RTX 4090 is never an injection target.

From the repository root, these commands only print JSON:

```bash
bash cluster/faults/power_cap.sh
bash cluster/faults/link_degrade.sh
bash cluster/faults/kill_node.sh
.venv/bin/python -m cluster.faults.plan
```

All `--apply` requests and `FaultExecutor.apply()` fail unconditionally. Environment
approval flags do not override the pause. `scaling.fault_train` also refuses CLI
execution before importing Torch. Re-enabling real operation requires a separately
reviewed change after explicit user budget and dedicated lease authorization; no
live command runner is supplied here. `rehearse` and `rehearse_schedule` are CPU
injected-command/clock-shim APIs, and always mark results `simulation: true`.
Never pass a real subprocess/SSH/remote runner to a rehearsal.

## The fixed two-hour campaign

`schedule.yaml` contains **example identities, not an authorized inventory**. Each
planned campaign lasts 7,200 seconds. The training soft stop is 7,080 seconds,
reserving 120 seconds for a final synchronous checkpoint. Confirmed failure to
completed checkpoint restore has a 300-second budget. These are configured limits,
not measured checkpoint/recovery durations. It applies one fault at a time, with planned device/VM
restoration 300 seconds later:

| Offset | Method | Exact effect | Restoration |
| --- | --- | --- | --- |
| 1,800 s | `gpu_power_cap` | One explicit GPU UUID, `nvidia-smi -i UUID -pl 200` | Queried original limit, never a factory default |
| 3,600 s | `ib_width` | One verified switch data port, enabled width mask 1 (1x), then link reset | Queried original enabled-width mask, reset, verify original active width |
| 5,400 s | `vm_power_loss` | `virsh destroy` one persistent GPU VM UUID on its owned hypervisor | `virsh start`, verify running, then separate checkpoint recovery |

The IB experiment changes an InfiniBand path, not a TCP qdisc. Width renegotiation
requires a reset: this includes a transient disruption and must not be described as
purely silent throttling. Actual selection of that port by NCCL must be established
from Task 2 topology and NCCL logs. RoCE is deliberately unsupported; a different
provider-approved implementation is needed. Management links cannot be targets,
and the recovery management route must survive the selected data-port reset.

The VM experiment represents abrupt loss of **one whole VM node**. It does not
claim physical chassis power loss. A rank process exit affects a training process;
Slurm DOWN affects scheduling state. Neither is implemented or substituted for
VM power loss. `virsh destroy` needs an exclusive, persistent domain with no managed
save image, and a checkpoint failure domain outside that VM. Bare-metal leases
without an authorized VM/hypervisor interface need their own reviewed method.

Before any future campaign: record an exclusive lease, owner/provider permission,
GPU UUID and bounds, switch GUID/port and active/supported/enabled widths, persistent
VM UUID and state, image digest, driver/CUDA/NCCL/OFED versions, model revision,
checkpoint storage, recovery procedure, and stop conditions. Rehearsals capture raw
baselines and resolve restore argv; an optional `persist=JsonlSink(path)` fsyncs the
baseline and exact restoration commands before any injected mutation. An invalid
baseline or baseline-journal failure prevents injection. Partial injection failure
still attempts restoration, with every command and recovery error retained.
Injection and restoration each have explicit state queries; success is never inferred
from an argv exit alone. Health revalidation and scheduler re-admission are separate:
no `uncordon`, RESUME, state deletion or incident reset is issued automatically.

`rehearse_schedule` dispatches against absolute coordinator-clock offsets using an
injected `advance(seconds)` simulation. It restores before moving on, aborts after
any failed injection/restoration or deadline overrun, and advances successful
simulations to 7,200 seconds. This is no evidence of a two-hour real training run.
Future comparisons must alternate baseline/candidate under matching conditions at
least three times and report median and range; no comparative result exists yet.

## Task 3 observations and clock domains

`FaultTimeline.observe_controller` calls the public `Controller.evaluate()` and
records first genuine Action receipt for an incident. It retains reason, health
sample time and step sample time. `Action` has no timestamp field. Unknown/stale
sample alerts are not counted as fault detections. A step-time slow-card alert does
not become an invented drain: Task 3 only alerts on slow steps under its current
policy, while hard XID/link faults can request drain and requeue.

`ObservedSchedulerExecutor` subclasses Task 3 only to timestamp `_call` boundaries.
It preserves Task 3's entire-batch isolation-before-requeue ordering and all failure
evidence. It supports dry runs or injected CPU command shims, with no production
command runner. Capture the batch start immediately before `execute()`, then pass
its results to `FaultTimeline.scheduler_results()`. For drained results the timeline
uses the last **drain command's** timestamp, excluding subsequent requeue work.
Any failed/planned/missing drain prevents a measured isolation success. Untimed
external results retain their outcome but leave drain latency `None`.

All timeline durations require one monotonic **coordinator** clock ID that survives
training worker/reporter replacement. Remote source timestamps remain evidence;
they are never subtracted from local monotonic timestamps. Cross-host `time.monotonic`
values are incomparable. Default local clocks are suitable only for same-coordinator
runs and CPU fixtures. A real deployment must provide a surviving coordinator clock
(or a measured clock conversion with uncertainty), not reuse a clock ID on a new
host. Sampling every 50 completed steps, collection/polling delay, transport and
controller evaluation contribute to detection latency. Injection confirmation is a
post-command query; the physical effect occurred between command start and query
completion. The reported confirmation-to-detection duration omits that uncertainty;
keep both raw boundaries and report the interval rather than claiming precise onset.

`FaultTimeline.metrics()` leaves missing detection, isolation or recovery phases
`None`, and rejects reversed phases. `recovery_start/complete` mean the checkpoint
restore boundary; device/VM restoration alone does not mean training recovery.
`JsonlSink` is a single-writer append journal and fsyncs each record. A missing or
truncated journal must be reconciled from retained evidence, never reconstructed
with guessed timestamps. Command raw output can be deployment-sensitive: review
it before publishing.

## Training and goodput protocol

`scaling.goodput.goodput(useful_steps_time, wall_time)` preserves the brief's public
signature and returns a finite productive-time fraction in `[0, 1]`. It is not
baseline-speed-normalized throughput or scaling MFU.

`FaultTrainingHook` emits schema 1 coordinator events with run ID, clock ID and
reporting rank zero only:

- `run_start` includes fixed world size; global logical steps start at 1.
- `step_complete` is emitted after the optimizer operation and synchronization succeed,
  with its actual `started_at`. Failed/partial steps publish no completion.
- `checkpoint_saved` identifies a successful synchronous full-state save and its step.
- `failure`, `restore_start`, `restore_complete` identify an attempt rollback; restore
  succeeds only after the named checkpoint actually loads the expected trainer state.
- `operation_failed` preserves failed step/save/load attempts without a success claim.
- `worker_complete` marks a successful final full-state save before workers exit;
  fault-mode workers do not sleep until the observation deadline or publish run_end.
- `run_end` names the actual final retained checkpoint. For the fixed campaign only
  the surviving coordinator publishes it after confirming every worker has exited
  within budget and observing the end of the window. An interrupted log has no goodput.

`account_events()` replays checkpoint lineage. It discards all completed attempts
lost on rollback, charges only the surviving attempt of each logical step, excludes
unsaved final tails, and includes initialization, checkpoint I/O, partial attempts,
detection, isolation, scheduling, restart and idle periods in total start-to-end wall
time. Rank duplicates, clock changes, unknown checkpoints, missing terminals,
noncontiguous steps and inconsistent restoration are rejected. No guessed steps are
credited. `lost_steps` counts discarded **completed** attempts; a partial failed
step remains overhead in the denominator, not a fabricated completed-step duration.
`FaultTrainingHook.resume()` replays the same retained journal without a second
run start; an incomplete restore needs explicit evidence reconciliation first.

`scaling.fault_train.instrument_fault_trainer(Trainer)` is the checkpoint-enabled
adapter for pinned TorchTitan v0.2.0 commit
`29624e37fa6dd319fd671b009338651b5c665da2`. It uses upstream `train()` unchanged,
wrapping public `train_step`, `CheckpointManager.save(curr_step, last_step)` and
`load(step) -> bool`. The default Task 4 `scaling/train.py` remains the separate
checkpoint-disabled fresh-run scaling adapter.

The fault adapter requires checkpoint enable, synchronous `async_mode="disabled"`,
full native save/load, explicit positive recovery step, `keep_latest_k=2`, and no
TorchFT, external/partial load, seed checkpoint, or model-only/HF final save.
Upstream step 0 loads model-only, so it is rejected for recovery. A skipped interval
save publishes nothing; an exception or `load() is False` never publishes success.
The final saved step must match successfully completed optimizer work.

The adapter's public `state_dict/load_state_dict` extends upstream model, optimizer,
lr-scheduler, dataloader and trainer state with checkpoint generation identity and
rank-specific Python/NumPy/Torch CPU/CUDA RNG state. Distinct `crucible_rng_rank_N`
keys avoid DCP de-duplicating different ranks' RNG as one replicated value. Recovery
requires unchanged world size and logical parallel-rank mapping; elastic resizing
is unsupported. Generation identity is verified on load, so an unacknowledged
partial overwrite of `step-N` cannot silently stand in for an earlier save event.
The caller's configuration still requires `keep_latest_k=2`. A configuration copy
passed to upstream sets its supported `keep_latest_k=0`, disabling its asynchronous
purger (including the final-save bypass). `scaling/checkpoint_retention.py` enforces
the actual two-slot limit synchronously: rank zero alone locks an exclusive owned
run directory, validates confirmed native metadata/run ownership, protects the active
confirmed save or explicitly restored generation, and frees the other slot before
any new save. After restoring an older retained checkpoint, that restored state is
the recovery anchor; a superseded higher-step target can be retired and re-saved
under a new generation. The active anchor itself cannot be overwritten; a missing
or unknown anchor fails closed. With no active anchor, the latest confirmed state
is protected. After native
save succeeds it verifies metadata/count and writes a confirmation record before
publishing checkpoint success. All outcomes are broadcast to every rank. Followers
never acquire the filesystem lock; unknown/incomplete/foreign/symlink entries are
preserved and stop the save. A failed save retains the active confirmed recovery anchor.
It never permanently or temporarily creates a third checkpoint through this adapter.

The runtime wiring is implemented as library APIs, without a live launch path:
`RNGState(torch=..., numpy=..., device=...)`,
`make_torch_hook(torch=..., device=..., run_id=..., clock_id=..., clock=..., emit=...)`,
and `instrument_fault_trainer(Trainer)(config, hook=hook, rng=rng)` for prebuilt
CPU/in-process hooks. For the normal upstream initialization order, use
`create_fault_trainer(Trainer, config, torch=..., numpy=..., run_id=..., clock_id=...,
clock=..., emit=..., wait_until=..., prior_events=...)`: it records the campaign
origin before upstream initializes its process group/device, then constructs the
runtime hook and RNG binding. Under torchrun, RANK/WORLD_SIZE identify the reporter
before process-group creation; actual world size is checked afterward. A launcher's
`CRUCIBLE_CAMPAIGN_ORIGIN_JSON` or retained `prior_events` preserves an earlier
origin, so launch, initialization and all restart downtime remain in the denominator. The hook factory
waits for CUDA, barriers all ranks, and waits for CUDA before recording completion.
After a failure, reconstruct the same coordinator journal via `prior_events`,
reconstruct the trainer with an explicit `checkpoint.load_step`, and retain the same
parallel layout. If the old worker died without publishing failure, the surviving
observer must confirm and record failure before restore. New CUDA/Trainer startup
and placement time stay inside the overall wall-time denominator; the separate
recovery metric covers the public checkpoint `load` call and synchronization.
A fresh hook refuses unexpected existing checkpoints.

## Stop boundaries, coordinator closure and owned-subtree supervision

Fault mode overrides public `should_continue_training` for time-based termination;
Task 4's fresh 200-step target is not substituted for two-hour fault training.
Every rank participates in `torch.distributed.broadcast_object_list`. Only rank zero
evaluates the entire stop predicate (not merely a timestamp); its boolean or error
is broadcast before any rank branches. The same coordination covers budget checks
and retention results. CPU tests use staggered rank observations and different local
origins to verify identical decisions. Training/LR/model conditions still need the
separate authorized fault-run configuration.

The step soft stop is 7,080 seconds. The reserved 120 seconds covers final synchronous
checkpoint, worker completion and teardown. `CampaignWindow.finish()` checks budget;
it does not wait until 7,200. Workers publish `worker_complete` with the actual saved
state and release. The outer supervisor must confirm the entire owned subtree empty
and reap the launcher by 7,200. Failure, incomplete final state, nonzero exit or work
exit observed after that budget yields no successful terminal.

After workers safely exit, the surviving coordinator waits for the observation
window. Late wakeups are allowed: a wake at 7,200.000001 records that actual `at`,
the planned boundary 7,200, the earlier verified worker-exit time and the observation
lateness separately. Goodput uses actual observed wall time, including this delay;
no exact timestamp is fabricated. Late observation cannot excuse late worker exit.
`CampaignWindow` is cooperative at step boundaries; the outer supervisor handles
blocked steps or saves without waiting for a training hook.

Process completion is based on **one exclusive, already delegated cgroup-v2 tree**,
not a launcher PID/PGID. `CgroupScope` in `scope.py` requires an operator-provisioned
0600-style lease file with `owner`, `run_id`, `exclusive: true`, `uid`, `device`,
`inode`, absolute `path`, `delegated_root`, and kernel-visible absolute `cgroup_path`.
The scope must be a private owned directory below that delegation (not the root or
delegation itself), initially empty, domain typed, and expose cgroup.kill. The code
checks filesystem magic, ownership/modes, symlinks, lease contents and path inode,
and pins directory/lease FDs with exclusive locks on both inodes. Duplicate lease
files cannot independently adopt the same empty scope. A replaced path, changed
lease, or shared/populated scope is rejected. It never creates a cgroup or grants
host permissions. Keep the lease/handle for reconciliation if cleanup is uncertain.

The concrete launcher starts an isolated `Python -I -S` startup gate, whose only
pre-exec task is reading a permit pipe. The parent moves the blocked unreaped child
into the scope and verifies membership, then releases it to exec training. Thus
training cannot fork children before scope attachment; descendants stay in the tree
even if a launcher creates new sessions/process groups. EOF/failure never releases
training. Default SIGCHLD and a single owned-child reaper preserve child PID identity
through attachment. Actual production activation remains disabled.

`cgroup.events` populated includes descendants. `wait_empty` must observe zero before
the leader can be reaped and completion considered; an early launcher exit does not
hide live workers. Timeout issues best-effort TERM to verified current members via
pidfds, waits 5 seconds, then uses pinned-scope cgroup.kill and a final bounded wait.
Pidfds avoid reused-PID targeting; cgroup.kill covers the subtree and concurrent
forks. No numeric PGID signal or shared-scope fallback exists. Unknown ownership
blocks cleanup and retains failure evidence. Unconfirmed subtree emptiness is an
explicit cleanup error, never successful goodput. Cleanup grace after the budget
is not successful training time.

All cgroup/Popen/pidfd operations were tested using temporary-file/syscall shims.
**No real process signal, cgroup change, GPU interruption or production hard-stop
acceptance was tested.** `apply` and the fault-training CLI remain hard-closed. An
authorized entry point must call the adapter with the same surviving coordinator
clock and launcher origin, keep journals outside the victim failure domain and use
a separately provisioned exclusive delegation; absence of delegation fails closed.
There is no process-group fallback. The kernel source for recursive population and
kill semantics is [the cgroup-v2 documentation](https://cdn.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html).

The supervisor, adapter, faults and journal must share a single surviving coordinator
time source. A numeric origin in the environment is insufficient if a child reads an
unrelated local clock. Cross-host clock transport/delegation and real pinned DCP
remain deployment acceptance requirements; CPU fixtures use one injected source.
Workers accept `wait_until` for prior API compatibility but never call it; only the
surviving supervisor receives the observation-window waiter.

## CPU validation and sources

```bash
.venv/bin/python -m pytest -q scaling/tests/test_goodput.py \
  scaling/tests/test_fault_training_adapter.py scaling/tests/test_fault_checkpoint_retention.py \
  cluster/faults/tests
```

Tests use only fake Trainer/CheckpointManager, command shims and clocks. Actual
TorchTitan DCP storage, per-rank distributed RNG, GPUs, IB width renegotiation,
hypervisor loss/restart, two-hour training and fault latencies remain unverified.
See [the evidence record](../../docs/evidence/fault-goodput.md).

Command/API semantics were checked against primary sources:
[NVIDIA nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html),
[rdma-core ibportstate](https://github.com/linux-rdma/rdma-core/blob/master/infiniband-diags/man/ibportstate.8.in.rst),
[libvirt virsh](https://www.libvirt.org/manpages/virsh.html),
[pinned Trainer](https://github.com/pytorch/torchtitan/blob/29624e37fa6dd319fd671b009338651b5c665da2/torchtitan/train.py),
and [pinned CheckpointManager](https://github.com/pytorch/torchtitan/blob/29624e37fa6dd319fd671b009338651b5c665da2/torchtitan/components/checkpoint.py).
