# E5: first multi-node runs of Crucible's cluster code on Snowflake GPU nodes

Date: 2026-11-05. Branch `snowflake/e5`. Payload: `scripts/snowflake/payloads/e5_multinode.py`; tables:
`scripts/snowflake/e5_results.py` (CPU tests: `scripts/tests/test_e5_multinode.py`, `scripts/tests/test_e5_results.py`).
Raw outputs: `docs/evidence/snowflake/e5/<run>/` (`report.json` with every RESULT line, `instance<i>.log`,
`driver.log`; for fullB also `phases/<phase>.json`, the full per-phase results the payload put on the stage);
`summary.md`/`summary.json` are the tables below as `e5_results.py summary` derives them. The PyPI proxy token and
the account identifier are removed from every stored file (`e5_results.py export`).

Hardware: **4 Snowflake GPU_NV_SM nodes (1× A10 24 GB each, vGPU "A10-24Q") over TCP**, pool `CRUCIBLE_A10X4_POOL`,
one ML Job with `--instances 4` per run. No Slurm/K8s, no H100, no NVLink, no InfiniBand or RDMA: NCCL uses
NET/Socket over the pods' `eth0`.

Purpose: run, for the first time on more than two machines, the parts of `cluster/` and `scaling/` that were written
for a four-node H100 cluster and so far had only CPU tests (Tasks 2–5: burn-in, scaling, drain, fault goodput).
Their apply/launch paths and $0 gates are unchanged; E5 imports only their pure logic.

## Summary

- **Collectives** (fullA, 3 passes): with E0's tuned socket settings, all-reduce bus bandwidth at 1 GiB is
  1.29 GB/s on 2 nodes, 1.03 on 3 and 0.83 on 4 (`tuned_buff16m`); defaults stay at 0.18–0.21 GB/s on every node
  count. Broadcast reaches 2.5–2.6 GB/s on 2 nodes but only 0.8–1.2 GB/s on 3–4, with wide spread.
  `NCCL_IB_DISABLE=0` still selects `NET/Socket`: the nodes show an `mlx5_0` device with an Ethernet link layer and
  no `/dev/infiniband`.
- **Burn-in tools**: DCGM, nccl-tests, perftest and nsys are absent; CUPTI does not initialize on the vGPU (no kernel
  timelines); nvidia-smi reports ECC on with 0 uncorrected errors, most other health fields `[N/A]`. The repo's
  bandwidth gate (0.70 × theory at ≥ 128 MiB, with the interface's reported 1.25 GB/s as "theory") passes the tuned
  2-node runs and 3-node all-reduce with the 16 MiB buffer, and fails every 4-node all-reduce and every default run.
- **Scaling** (fullB, 3 rounds, DDP of Qwen2.5-0.5B, fp32 gradients): weak-scaling efficiency **0.54** on 2 nodes and
  **0.36** on 4 (4,366 and 2,849 tokens/s/GPU vs 8,019 on one); strong scaling (16 sequences/step) **0.38** and
  **0.12**. Exposed communication is 46% / 65% (weak) and 62% / 87% (strong) of the step; the 1.98 GB gradient
  all-reduce alone takes 2.0 s on 2 nodes and 4.2 s on 4, against 2.0 s of compute per weak step. MFU on one A10
  0.155 (vs the 125 TFLOPS datasheet peak). Throughput spread across rounds ≤ 3%.
- **Drain** (fullB): with a competing CUDA process on one node, the node-local step-time feed flags that node and only
  that node, at the third consecutive slow sample (slow from sample 5, alert at sample 7, once); the synchronized
  step time rises equally on all four nodes (4.7 → 6.1 s) and is never flagged.
- **Fault** (fullB, 3 rounds + 1): SIGKILL of one rank process after step 25, restart of all ranks from the step-20
  checkpoint: goodput **0.430 [0.422–0.445]** with checkpoints on the head node's disk (6.48 GB saves take 39–49 s at
  ~0.14 GB/s) and **0.673** with `/dev/shm` (3–4 s saves); every round loses steps 21–25 (23.1–23.9 s), restarts in
  11.1–12.0 s and restores in 9.2–11.4 s.
- TorchTitan's pinned nightly stack cannot be installed (no nightly torch/torchao on Snowflake's PyPI proxy, no
  egress), so scaling used plain PyTorch DDP. Credits used: **17.14** (ledger, upper bound; cap 75).

## Setup

| | |
|---|---|
| Hardware | 4 Snowflake GPU_NV_SM nodes (1× A10 24 GB each) over TCP; per node 36 CPUs, driver 570.211.01, `eth0` MTU 1500, reported speed 10000 Mb/s; fullA and fullB ran on different node sets |
| Software | `--pip torch==2.8.0 torchvision==0.23.0 transformers==4.55.4 numpy==2.2.6` (Crucible's pins; torchvision from `requirements/rollout.lock`) → torch 2.8.0+cu12.8, NCCL 2.27.3, Ray 2.55.1 (image) |
| Model / data | `models/Qwen2.5-0.5B-Instruct` (stage asset, 494,032,768 parameters, tied embeddings; N = 357,898,112 without the embedding, L 24, H 14, D 64); GSM8K train (`data/gsm8k`, 7,473 examples) as "Question … Answer …" text, packed into 1,358 sequences of 1,024 tokens |
| Ports | NCCL rendezvous cycles through 12950–12999 (E5's band); NCCL data sockets are ephemeral (open range) |
| Process model | one Ray actor per node (STRICT_SPREAD placement group, head node first); every NCCL world is a set of fresh rank processes (`--mode rank`), one per node, started and killed by the actors; rank 0 runs on the head node next to the driver |
| Repo code reused | `cluster/provision/node_probe.probe`; `cluster/health/parse_nccl.evaluate` + `thresholds.yaml`; `cluster/drain/metrics.StepTimeHook`, `controller.Controller`, `SchedulerExecutor(dry_run=True)`; `scaling/analyze_nsys.ModelStructure`/`mfu`, `scaling/experiment.scaling_metrics`; `scaling/goodput.FaultTrainingHook`/`account_events`, `scaling/fault_train.make_torch_hook`/`RNGState`, `scaling/checkpoint_retention.CheckpointRetention`; E0's `run_collectives` |

Runs (credits are the driver's upper-bound estimates: submit-to-stop wall time × 4 nodes × 1.70/h):

| Run | Phases | What happened | Wall s | Credits |
|---|---|---|---|---|
| `smoke4-fail1` | all, `--quick` | failed in setup: an actor-side asset fetch has no Snowpark session (only the driver process has one); inputs are now fetched once on the head and shipped through Ray | 518 | 0.978 |
| `smoke4-fail2` | all, `--quick` | health and collectives fine; every training world failed: the image's torchvision does not match torch 2.8 (`operator torchvision::nms does not exist` when transformers imports it); fixed by pinning `torchvision==0.23.0` | 730 | 1.379 |
| `smoke4` | all, `--quick` | drain and fault fine; scaling failed: torch.profiler recorded no CUDA kernels (CUPTI unavailable), so `summarize_kernels` refused the trace; replaced by the wall-time split below | 1319 | 2.491 |
| `smoke5` | health, scaling, fault, `--quick` | all fine | 964 | 1.820 |
| `fullA` | health, collectives, scaling | health and collectives fine; the job-log fetch cut the 139 KB collectives RESULT line and returned nothing after it, so the scaling phase (it ran) left no data. The 23 collective runs were recovered from the cut line (`fullA/collectives-salvaged.json`, only the trailing summary was lost and is recomputed) | 2365 | 4.467 |
| `fullB` | health, scaling, drain, fault | all fine; phases now print slim RESULT lines and put full results on the stage | 3177 | 6.004 |
| Total | | | | **17.139** |

The ledger lines of the three `smoke4*` runs all name `.cache/snowflake/e5/smoke4`: the first two directories were
renamed afterwards. The earlier quick runs (smoke4, smoke5) give the same picture as fullB at smaller sizes; their
numbers are not used below.

## 1. Burn-in checks and collectives

### What runs on these vGPU nodes (fullA and fullB, all 8 nodes alike)

| Check (original Task 2 tool) | Result on GPU_NV_SM |
|---|---|
| DCGM diag level 3 (`dcgmi`, `nv-hostengine`, `libdcgm.so`) | **absent**: no binary, no library, nothing to install from (no egress; DCGM also needs host-level access a vGPU guest lacks). Not run |
| nccl-tests (`all_reduce_perf`) | **absent** (not on `PATH`, not under `/opt` or `/usr/local`); no egress to build it. Replaced by torch.distributed all-reduce/broadcast with nccl-tests' bus-bandwidth formula |
| HCA perftest (`ib_write_bw`, `ib_read_lat`) | **absent**. `/sys/class/infiniband/mlx5_0` exists (port 1 `ACTIVE`, `LinkUp`, `100 Gb/sec (4X EDR)`, link layer **Ethernet**), but `/dev/infiniband` does not, `ibv_devinfo` says "No IB devices found" and `ibstat` "stat of IB device 'mlx5_0' failed". With `NCCL_IB_DISABLE=0` NCCL still uses `NET/Socket` (2 and 4 nodes) |
| Topology / affinity (`node_probe`, `affinity_probe`) | `node_probe.probe` (host and workload kinds) ran on every node: one `NVIDIA A10-24Q`, CUDA runtime 12080, NCCL 22703; `ofed_info` and the Crucible image ID absent. `nvidia-smi topo -m`: GPU0–NIC0 `NODE`, CPU affinity 0–17, NUMA 0. `affinity_probe` needs a reviewed site inventory; not run |
| nvidia-smi health fields | ECC enabled, 0 uncorrected errors (volatile and aggregate); temperature, power limit, clock-event reasons, retired pages and PCIe link are `[N/A]` on the vGPU |
| Kernel profiling (nsys / CUPTI) | `nsys` absent; `ncu` 2025.1.0 present, but CUPTI fails (`CUPTI_ERROR_NOT_INITIALIZED`) and torch.profiler records 0 CUDA kernels, so no kernel timeline can be taken |
| bf16 matmul, 3 × 20 calls (added) | 8192³: 75.5–79.4 TFLOPS per node (median of 3); 4096³: first set 83–90, then 59–71 (per-node medians 59.8–70.8). Datasheet: 125 TFLOPS dense BF16 |
| Disk write with fsync, 2 GiB (added) | `/tmp` (overlay): 0.111–0.159 GB/s; `/dev/shm` (tmpfs): 3.37–3.48 GB/s |
| TorchTitan stack (head node, `pip download`) | `torchtitan==0.2.0` wheel: available on Snowflake's PyPI proxy. `torch==2.10.0.dev20251019` (the pinned nightly): not on the proxy (stable 2.2.0–2.15.0 only); the `download.pytorch.org` nightly index: unreachable; `torchao==0.15.0.dev20251015`: not found |

### All-reduce and broadcast on 2, 3 and 4 nodes (fullA)

fp32 tensors of 1 MiB–1 GiB, (warm-up, timed) calls (5, 50), (3, 20), (1, 5), (1, 3); the slowest rank's time counts;
`NCCL_IB_DISABLE=1` except in the IB probe. **Bus bandwidth** in GB/s (nccl-tests convention: all-reduce
algbw × 2(n−1)/n, broadcast = algbw), median [min–max] of 3 interleaved passes; "default" and the IB probe ran once,
the probe with ≤ 16 MiB. `tuned` = E0's `NCCL_SOCKET_NTHREADS=8 NCCL_NSOCKS_PERTHREAD=8 NCCL_MIN_NCHANNELS=4`;
`tuned_buff16m` adds `NCCL_BUFFSIZE=16777216`. Every run passed the small all-reduce/broadcast result check; NCCL logs
show `NET/Socket`, and 8 threads × 8 sockets in the tuned runs.

All-reduce:

| Nodes | Variant | 1 MiB | 16 MiB | 256 MiB | 1 GiB |
|---|---|---|---|---|---|
| 2 | default | 0.173 | 0.206 | 0.189 | 0.179 |
| 2 | `NCCL_IB_DISABLE=0` | 0.180 | 0.170 | – | – |
| 2 | tuned | 0.252 [0.248–0.262] | 0.724 [0.630–0.734] | 1.043 [1.033–1.092] | 1.072 [1.037–1.078] |
| 2 | tuned_buff16m | 0.237 [0.222–0.244] | 0.745 [0.711–0.776] | **1.322** [1.258–1.357] | **1.287** [1.242–1.367] |
| 3 | default | 0.204 | 0.221 | 0.213 | 0.207 |
| 3 | tuned | 0.480 [0.457–0.492] | 0.651 [0.625–0.697] | 0.757 [0.389–0.761] | 0.832 [0.831–0.872] |
| 3 | tuned_buff16m | 0.197 [0.193–0.203] | 0.676 [0.673–0.683] | 0.944 [0.924–1.084] | **1.029** [1.029–1.075] |
| 4 | default | 0.164 | 0.234 | 0.214 | 0.211 |
| 4 | `NCCL_IB_DISABLE=0` | 0.161 | 0.184 | – | – |
| 4 | tuned | 0.320 [0.300–0.336] | 0.581 [0.426–0.596] | 0.669 [0.332–0.784] | 0.792 [0.652–0.847] |
| 4 | tuned_buff16m | 0.138 [0.126–0.147] | 0.578 [0.323–0.581] | 0.708 [0.561–0.822] | **0.827** [0.696–0.830] |

Broadcast:

| Nodes | Variant | 1 MiB | 16 MiB | 256 MiB | 1 GiB |
|---|---|---|---|---|---|
| 2 | default | 0.238 | 0.250 | 0.254 | 0.247 |
| 2 | `NCCL_IB_DISABLE=0` | 0.200 | 0.222 | – | – |
| 2 | tuned | 1.280 [1.134–1.471] | 2.176 [1.038–2.382] | 2.316 [0.536–2.322] | **2.605** [2.557–2.664] |
| 2 | tuned_buff16m | 1.144 [1.072–1.149] | 2.962 [2.606–3.063] | 2.739 [1.112–2.903] | 2.470 [2.289–2.917] |
| 3 | default | 0.189 | 0.211 | 0.218 | 0.219 |
| 3 | tuned | 0.225 [0.214–0.579] | 0.273 [0.164–0.363] | 0.581 [0.391–1.015] | **1.154** [0.652–1.155] |
| 3 | tuned_buff16m | 0.569 [0.525–0.619] | 0.802 [0.606–1.144] | 0.451 [0.392–1.164] | 0.904 [0.645–1.513] |
| 4 | default | 0.180 | 0.221 | 0.230 | 0.233 |
| 4 | `NCCL_IB_DISABLE=0` | 0.180 | 0.198 | – | – |
| 4 | tuned | 0.568 [0.442–0.607] | 0.792 [0.283–0.845] | 0.869 [0.844–1.093] | **0.908** [0.747–1.233] |
| 4 | tuned_buff16m | 0.444 [0.395–0.641] | 0.749 [0.215–0.787] | 0.998 [0.240–1.185] | 0.831 [0.581–1.005] |

- The 2-node numbers repeat E0 on another node pair (E0: all-reduce 1.26 GB/s, broadcast 2.6 GB/s at 1 GiB).
- Defaults sit at the single-TCP-flow rate (~0.2 GB/s) on every node count; at 1 GiB the tuned settings give 4–7×
  that for all-reduce and 4–11× for broadcast.
- All-reduce bus bandwidth falls with node count (1.29 → 1.03 → 0.83 GB/s at 1 GiB), so per-node link rate is not
  what limits it; in algorithm bandwidth that is 1.29, 0.77 and 0.55 GB/s, the rate a gradient all-reduce sees.
- Broadcast drops from 2.6 to ~1 GB/s beyond two nodes and varies widely (some passes 0.2–0.4 GB/s at
  16–256 MiB). Neither 3 passes nor these logs say which link or hop was slow; the pods' network beneath `eth0` is
  not visible.

Burn-in gate: `cluster/health/parse_nccl.evaluate` with `thresholds.yaml` (bus bandwidth ≥ 0.70 × site theory for
every size ≥ 128 MiB, results checked). The site has no theoretical bus bandwidth, so the gate used the pod
interface's reported 10000 Mb/s = 1.25 GB/s (not a measured or published limit; NCCL broadcast exceeds it). Verdicts
per pass: all-reduce `tuned_buff16m` passes on 2 and 3 nodes (3/3 each) and fails on 4 (0/3); `tuned` passes on 2
nodes only (3/3); broadcast passes on 2 nodes (5 of 6 passes), 1/3 or 0/3 per variant on 3 and 4 nodes; every
default run fails; the IB probe is not evaluated (no size ≥ 128 MiB).

## 2. Data-parallel scaling (fullB)

Training: PyTorch DDP (torch 2.8.0, `bucket_cap_mb=25`, `gradient_as_bucket_view`), bf16 autocast with fp32 master
weights, fp32 gradients and fp32 all-reduce, fused AdamW (lr 1e-5, no weight decay), micro-batch 2 × 1,024 tokens,
gradient accumulation with `no_sync` for all but the last micro-step; NCCL env `tuned_buff16m`. Weak scaling:
16 sequences per GPU per step; strong scaling: 16 sequences per step in total (8 per GPU on 2 nodes, 4 on 4). The
1-node run is both the weak and the strong reference (same plan). Data order is deterministic (step s uses
sequences [(s−1)·G, s·G) of the packed set, each rank its slice), so a strong-scaling step sees the same global batch
at every world size. Per run: 3 warm-up and 10 timed steps, then the gradient volume (1,976,131,072 bytes) all-reduced
alone 3 times in 25 MiB buckets. 3 rounds, the five configs interleaved within each round. Timing is synchronized
(`cuda.synchronize` at step start and end); tokens/s uses the slowest rank's 10-step window (as
`scaling.train.CompletedStepTimer` does).

Definitions (the repo's where they apply): tokens/s/GPU = global tokens/s ÷ GPUs; MFU = `scaling.analyze_nsys.mfu`
(6N + 12LHDS per token, N without the embedding, peak 125 TFLOPS dense BF16 per A10); scaling efficiency =
`scaling.experiment.scaling_metrics` (MFU ÷ reference MFU) with the 1-GPU run as reference, i.e. tokens/s/GPU
relative to one GPU, for weak and strong scaling alike. Communication share: CUPTI is unavailable, so
`summarize_kernels` cannot be fed; instead each step is split by wall clock into the communication-free micro-steps,
the last micro-step (its backward launches the bucketed all-reduce; timed until the all-reduce has finished) and the
optimizer. Compute = accum × one communication-free micro-step + optimizer; exposed communication = last micro-step
− one communication-free micro-step; share = exposed ÷ step time. "All-reduce alone" is the same gradient volume with
nothing else running.

Median [min–max] of 3 rounds:

| Scaling | Nodes | Global seq/step | Accum | Step s | Tokens/s/GPU | MFU | Scaling eff. | Compute s | Exposed comm s | Exposed comm share | All-reduce alone s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | 1 | 16 | 8 | 2.04 [2.03–2.05] | 8,019 [8,012–8,062] | 0.155 [0.155–0.156] | 1.00 | 2.03 [2.02–2.03] | 0.01 | 0.01 | – |
| weak | 2 | 32 | 8 | 3.72 [3.70–3.78] | 4,366 [4,322–4,424] | 0.084 [0.083–0.085] | **0.54** [0.54–0.55] | 2.02 [2.02–2.03] | 1.70 [1.67–1.76] | **0.46** [0.45–0.47] | 2.06 [2.01–2.07] |
| weak | 4 | 64 | 8 | 5.75 [5.64–5.82] | 2,849 [2,805–2,885] | 0.055 [0.054–0.056] | **0.36** [0.35–0.36] | 2.03 [2.03–2.03] | 3.72 [3.61–3.79] | **0.65** [0.64–0.65] | 4.15 [4.11–4.25] |
| strong | 2 | 16 | 4 | 2.68 [2.66–2.69] | 3,029 [3,007–3,052] | 0.058 [0.058–0.059] | **0.38** [0.37–0.38] | 1.03 [1.03–1.03] | 1.65 [1.63–1.66] | **0.62** [0.61–0.62] | 2.02 [2.00–2.05] |
| strong | 4 | 16 | 2 | 4.18 [4.14–4.20] | 982 [961–986] | 0.019 [0.019–0.019] | **0.12** [0.12–0.12] | 0.53 [0.52–0.53] | 3.65 [3.62–3.68] | **0.87** [0.87–0.87] | 4.22 [4.21–4.30] |

- Global throughput: 8,019 tokens/s on one node; weak scaling 8,731 (2 nodes, 1.09×) and 11,396 (4 nodes, 1.42×);
  strong scaling 6,058 and 3,929, i.e. slower than one node.
- Compute per step stays at 0.25 s per micro-step whatever the node count (2.03 s for 8 micro-steps): the nodes are
  equally fast and the work splits evenly. The extra time is the gradient all-reduce: fp32 gradients are 1.98 GB per
  step independent of batch size, which at the measured 0.96 GB/s (2 nodes) and 0.48 GB/s (4 nodes) algorithm
  bandwidth in 25 MiB buckets takes 2.0 and 4.2 s. DDP hides only 0.3–0.6 s of it behind the last backward
  (all-reduce alone − exposed), because the other micro-steps run under `no_sync` and the bucket all-reduces queue on
  the slow link.
- The repeat spread is small (≤ 3% for throughput, ≤ 5% for the communication split), so these are properties of
  the setup, not noise. MFU on one
  A10 is 0.155 of the 125 TFLOPS datasheet peak (≈ 0.25 of the ~78 TFLOPS the 8192³ matmul reaches).
- Peak allocated memory 17.0–18.8 GiB of 24 GB. Rank 0's training loss falls from 0.69–0.72 to 0.46–0.54 in 13
  steps and repeats within 0.001 across rounds.

## 3. Drain controller on step times (fullB)

4-node DDP run, 8 sequences per GPU per step (accum 4), 56 steps. Every rank runs two
`cluster.drain.metrics.StepTimeHook`s with `every=4`: the synchronized step time (`run_id e5-drain-sync`, what a hook
in the training loop sees) and the node-local time of one communication-free micro-step (`e5-drain-local`). After each
window the driver reads the four textfiles through the actors, turns them into `NodeMetrics` (sample timestamp
shifted by a Ray-measured clock offset, |offset| ≤ 1 ms; `link_up` from `/sys/class/net/eth0/carrier`;
`xid_errors=0` because there is no DCGM) and calls `Controller.evaluate` (`rules.yaml` unchanged: > 1.2 × median of
the fresh comparable nodes including itself, 3 consecutive checks, ≥ 3 comparison nodes, 120 s freshness) and
`SchedulerExecutor("slurm", ..., dry_run=True)`. **Slow-down method**: after sample 4 is evaluated the driver starts a
second CUDA process on the last node's (job-3) GPU — back-to-back bf16 4096 × 4096 matmuls — and kills it after
sample 10; the two CUDA contexts time-slice the vGPU.

Each cell: local micro-step s / synchronized step s; "Load on" = the competing process ran during (part of) that
window.

| Sample | Step | Load on | job-0 | job-2 | job-1 | job-3 | Alerts (local) | Alerts (sync) |
|---|---|---|---|---|---|---|---|---|
| 1 | 4 |  | 0.263 / 5.03 | 0.270 / 4.94 | 0.270 / 4.88 | 0.267 / 4.88 | – | – |
| 2 | 8 |  | 0.243 / 4.68 | 0.252 / 4.68 | 0.251 / 4.68 | 0.248 / 4.68 | – | – |
| 3 | 12 |  | 0.244 / 4.72 | 0.252 / 4.72 | 0.251 / 4.72 | 0.248 / 4.72 | – | – |
| 4 | 16 |  | 0.244 / 4.61 | 0.251 / 4.61 | 0.249 / 4.61 | 0.248 / 4.61 | – | – |
| 5 | 20 | yes | 0.244 / 5.80 | 0.251 / 5.80 | 0.249 / 5.80 | **0.532** / 5.82 | – | – |
| 6 | 24 | yes | 0.243 / 6.09 | 0.251 / 6.09 | 0.249 / 6.09 | **0.630** / 6.09 | – | – |
| 7 | 28 | yes | 0.243 / 6.11 | 0.251 / 6.11 | 0.249 / 6.11 | **0.628** / 6.11 | **job-3** | – |
| 8 | 32 | yes | 0.244 / 6.06 | 0.251 / 6.06 | 0.249 / 6.06 | **0.631** / 6.06 | – | – |
| 9 | 36 | yes | 0.244 / 6.19 | 0.252 / 6.19 | 0.249 / 6.19 | **0.631** / 6.19 | – | – |
| 10 | 40 | yes | 0.244 / 6.12 | 0.252 / 6.12 | 0.249 / 6.12 | **0.632** / 6.12 | – | – |
| 11 | 44 |  | 0.244 / 4.76 | 0.251 / 4.76 | 0.249 / 4.76 | 0.274 / 4.74 | – | – |
| 12 | 48 |  | 0.244 / 4.74 | 0.251 / 4.74 | 0.249 / 4.74 | 0.250 / 4.74 | – | – |
| 13 | 52 |  | 0.244 / 4.70 | 0.252 / 4.70 | 0.250 / 4.70 | 0.249 / 4.70 | – | – |
| 14 | 56 |  | 0.244 / 4.72 | 0.251 / 4.72 | 0.250 / 4.72 | 0.249 / 4.72 | – | – |

(bold = above 1.2 × that sample's median)

- Local feed: job-3 is above 1.2 × median in samples 5–10 (2.1–2.5× the others) and the controller raised exactly one
  action, `alert job-3 "step time > median × 1.2 for 3 checks"`, at sample 7, the third consecutive slow sample; no
  other node was ever flagged, the latch held (no repeat in samples 8–10), and the count reset after the load stopped.
  An independent recomputation from the logged samples (`verify_drain`) expects the same single alert. The dry-run
  executor returned status `alert` (alerts need no scheduler command); no drain/requeue was planned because XID and
  link were healthy.
- Synchronized feed: all four nodes' step times rose together (4.7 → 5.8–6.2 s: the straggler slows everyone in
  synchronous DP), so no node exceeded 1.2 × the median and nothing was flagged. A step-time hook placed on the
  synchronized training step, as Task 3 wires it, cannot find a straggler inside one synchronous job; it needs a
  node-local timing (here the `no_sync` micro-steps) or comparison across jobs.
- No "indeterminate" alerts: every sample was fresh, comparable and from four nodes.

## 4. Rank kill, restart from checkpoint, goodput (fullB)

4-node DDP run (the drain run's plan), 30 steps, a full-state checkpoint every 10 steps and at the end. Rank 0 (head
node) writes it with `torch.distributed.checkpoint` (`FileSystemWriter`, 4 threads, `sync_files=True`, no process
group): model and AdamW state (the DDP replicas are identical), param groups, every rank's RNG state (`RNGState`,
gathered to rank 0) and the checkpoint identity; `CheckpointRetention` keeps at most two; 6.48 GB per checkpoint.
Every rank wraps its steps, saves and the restore in `FaultTrainingHook` via `make_torch_hook` (rank 0 emits; the
coordinator clock is the head node's `CLOCK_MONOTONIC`, shared by rank 0 and the driver). **Fault**: once step 25 is
logged, the driver SIGKILLs the process group of the last rank (job-3, another node than the checkpoint), kills the
three survivors (they block in the next all-reduce), records `failure` through the hook API, and starts four new rank
processes that load the last checkpoint on rank 0, broadcast model and optimizer tensors, scatter the RNG states,
replay from step 21 and finish. Goodput = `account_events` on the logged events (useful = time of the retained
steps; wall = run start to run end, including model loading and every save). This is a **rank-process kill, not the
node, power-cap or InfiniBand faults** of the original plan. Rounds 1–3 checkpoint to the head node's overlay disk
(`/tmp`); one extra round, right after round 1, writes to `/dev/shm`.

| Round | Checkpoint store | Goodput | Useful s | Wall s | Lost steps (s) | Save s (median of 3) | Restore s | Failure → restore start s |
|---|---|---|---|---|---|---|---|---|
| 1 | disk | 0.430 | 141.0 | 328.0 | 5 (23.1) | 43.6 | 9.2 | 11.4 |
| 2 | disk | 0.445 | 144.9 | 325.3 | 5 (23.9) | 41.0 | 10.1 | 12.0 |
| 3 | disk | 0.422 | 142.8 | 338.0 | 5 (23.5) | 47.6 | 10.9 | 11.1 |
| 1 | `/dev/shm` | 0.673 | 141.5 | 210.2 | 5 (23.4) | 3.3 | 11.4 | 11.4 |

- Disk rounds: goodput **0.430 [0.422–0.445]**. Round 1's 328 s: 141 s retained steps, 131 s for three saves
  (6.48 GB each at ~0.15 GB/s, the disk's fsync write rate), 23 s of lost steps 21–25, 11 s from the failure record
  to the restore start (process start, model load, NCCL init), 9 s restore, 11 s before the first step. The kill-to-
  failure record took 1.3 s (survivor teardown). With `/dev/shm` the saves drop to 3–4 s and goodput rises to 0.673;
  lost work and restart cost are unchanged.
- Every round: kill after step 25, last checkpoint step 20, 5 lost steps, 35 completed step attempts for 30 retained,
  final checkpoint `step-30@generation-0`, and the replayed log validates (`account_events` accepts it and equals the
  payload's own accounting).
- Restore check: the first replayed step (21) gives rank 0 the same loss as before the kill to 5 decimals in all four
  rounds (e.g. 0.45576 / 0.45576), so weights and data position were restored exactly; steps 22–25 differ by up to
  0.0010, consistent with non-deterministic GPU reductions after the first update. This does not prove the optimizer
  and RNG states bit for bit.
- The survivors were still running when the driver tore them down 0.4–0.5 s after the kill (rc `None`, then −9), so
  this run does not show how NCCL over sockets reacts to a dead peer on its own.

## Deviations from the original Task 2–5 definitions

Task 2 (burn-in, `cluster/health`):
- No DCGM level-3 diagnostics (DCGM absent); nvidia-smi fields instead, most `[N/A]` on the vGPU.
- No nccl-tests binaries and no mpirun/ssh launch: torch.distributed all-reduce and broadcast (E0's harness) in rank
  processes started by Ray actors; bus bandwidth by the nccl-tests formula. Only all-reduce and broadcast, fp32,
  4 sizes, on the first 2, 3 and 4 nodes (not all subsets, no all-gather/reduce-scatter/all-to-all), 3 passes, no
  baseline/candidate pairs.
- No HCA perftest (no verbs device); the IB question was answered with an NCCL `NCCL_IB_DISABLE=0` probe.
- No site inventory: `affinity_probe` not run; the bandwidth gate's "theory" is the reported interface speed.
- Added checks: bf16 matmul rate, fsync disk write rate, CUPTI probe, pip probe of the TorchTitan stack.

Task 4 (scaling):
- TorchTitan v0.2.0 with torch 2.10.0.dev20251019+cu126 / torchao 0.15.0.dev20251015 cannot be installed (nightlies
  not on the proxy, no egress); plain PyTorch 2.8.0 DDP instead of TorchTitan FSDP2 (+TP2).
- Model Qwen2.5-0.5B-Instruct, not Llama3 8B or the small MoE; BF16 autocast only (no FP8, no EP, no held-out
  accuracy delta); seq 1024, not 8192; global batch 16–64 sequences, not 256; 1/2/4 GPUs, not 8/16/32 H100.
- 3 warm-up + 10 timed steps, not 50 + 150 of 200; `CompletedStepTimer` itself not used (it requires warm-up in
  multiples of 50 and publishes every 50 steps); its scope (synchronized step, slowest rank) is kept.
- No seed 42: weights come from the pretrained checkpoint and the data order is fixed (losses repeat within 0.001).
- No nsys and no kernel-interval analysis (CUPTI unavailable): the communication share is a wall-clock split plus an
  all-reduce-alone timing, not `summarize_kernels`' NCCL-without-compute union.
- MFU peak is the A10 datasheet (125 TFLOPS BF16 dense), not H100; scaling efficiency is against 1 GPU (the repo's
  `scaling_metrics` with `reference_gpus=1`), not 8, and is also applied to weak scaling.
- No `NCCL_ALGO=allreduce:Tree` candidate; NCCL env is E0's tuned socket setting.

Task 3 (drain):
- StepTimeHook every 4 steps, not 50; textfiles read through Ray, no Prometheus/node_exporter scrape.
- XID supplied as 0 (no DCGM); link state from the pod's `eth0` carrier, not a managed NIC exporter.
- Executor in dry-run with no scheduler and no jobs (alerts only); no Slurm/K8s.
- A second, node-local step-time feed was added next to the synchronized one; the slow-down is a competing CUDA
  process, not a degraded GPU or link.

Task 5 (fault and goodput):
- A SIGKILL of one rank process, not a GPU power cap, an InfiniBand switch-port width change or the loss of a VM
  node; no detection latency (the driver kills the rank itself and acts at once; Task 3 detection not involved).
- About 5.5 min of training per round, not a 7,200 s campaign; no `CampaignWindow`/time-based adapter, no cgroup
  launcher; the supervisor role is the E5 driver (like torchrun's agent), restarting all ranks in the same ML Job on
  the same nodes, not a scheduler requeue or a new job.
- Checkpoints are written by rank 0 alone to node-local storage on the head node (survives a rank or other-node
  fault, not the head node's loss), not shared storage; DCP without a process group.
- 3 disk rounds + 1 `/dev/shm` round with median and range, but no baseline/candidate pairs.

## Caveats

- The GPUs are virtual ("A10-24Q", one per node); clocks, power and thermals are hidden. The first of three 4096³
  matmul sets runs at 83–90 TFLOPS and later ones at 59–71, so sustained rates are below a burst; the 8192³ rate is
  steady at ~72–81.
- Collectives (fullA) and training (fullB) ran on different node sets, and the network beneath `eth0` is shared and
  not visible; the broadcast spread at 3–4 nodes may come from other tenants.
- Exposed communication assumes the last micro-step's compute equals a `no_sync` micro-step; with 1 node the residual
  is 0.01 s, so the error of that assumption is small.
- fullA's scaling phase ran but its output was lost to the log truncation; fullB repeats it with the same code.
- Credits are the driver's upper-bound estimates (include node start, the ~3 min pip install and queueing behind
  other agents' jobs), not billed amounts.
- After fullB the payload is unchanged; `e5_results.py` only gained the export of per-phase files.

## Tests

`CUDA_VISIBLE_DEVICES= .venv-ml/bin/python -m pytest -q scripts/tests/test_e5_multinode.py scripts/tests/test_e5_results.py`
(CPU, gloo): **22 passed**. They cover the batch plans and data order, bus-bandwidth factor, the Qwen parameter count, the
StepTimeHook → `parse_prom` → Controller path, `verify_drain` (alert only at the third slow sample; synchronized
times never single out a node), the wall-time split, redaction and export, the compact RESULT lines, and 2-rank CPU
worlds for the scaling, drain and fault paths (kill after step 3, restart from the step-2 checkpoint, goodput with one
lost step, same loss on the replayed step, two retained checkpoints). Existing tests: `pytest -q scripts/tests` **216 passed** (includes the 22);
`pytest -q cluster/drain/tests cluster/health/tests cluster/provision/tests scaling/tests` (the packages the payload
imports; none of their files changed) **269 passed**; `ruff check scripts/snowflake` and the two test files clean.
