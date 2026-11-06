# Crucible

Crucible is a GPU and ML infrastructure project in five parts: bring-up tooling for a multi-node
GPU cluster, asynchronous RL post-training with GRPO (rollout engine, learner and weight sync),
LLM serving with a trace replayer, router and autoscaler (including prefill/decode disaggregated
layouts), EAGLE-3 speculative decoding, and a PyTorch custom op for fused token log-probabilities.
Most measurements ran on a single RTX 4090 (24 GB) in a shared workstation, with models scaled down
to fit it. A second set ran on Snowflake trial GPU nodes: two A10 24 GB GPUs in one node (no
NVLink/P2P), or two to four single-A10 nodes over TCP. The multi-node parts were written for a
four-node H100 cluster with InfiniBand (cluster checks, burn-in, health-based drain, fault injection,
scaling runs, cross-node P/D serving with KV transfer over RDMA) and have not run on one. Their
logic ran once on four A10 nodes over TCP; the RDMA, DCGM, Slurm/Kubernetes and TorchTitan parts
have never run on real hardware.

## Components

**Cluster bring-up** (`cluster/`, `scaling/`; offline only)

- [cluster/provision](cluster/provision/README.md): Slurm and Kubernetes checks for a four-node
  H100 cluster with eight GPUs per node. The Slurm profile requires 32 unique GPU UUIDs from
  `srun -N4 --ntasks-per-node=8 --gpus-per-node=8 nvidia-smi -L`; the Kubernetes profile requires
  32 one-GPU pods, eight per node, admitted by Kueue, and a shared NFS volume seen by every worker.
  Driver, CUDA, NCCL, OFED, Slurm/Enroot/Pyxis and GPU/Network Operator versions are pinned as
  candidates (not tested together on a real site).
- [cluster/health](cluster/health/README.md): burn-in plan with DCGM level-3 diagnostics, topology
  probes, cross-node HCA bandwidth tests (perftest) and NCCL collective tests.
- [cluster/drain](cluster/drain/README.md): a controller that drains or requeues a node on an XID
  error or NIC link-down and flags a straggler whose step time stays above 1.2× the median of
  healthy nodes for 3 consecutive samples; Prometheus scrape and alert templates and a step-time
  hook for training.
- [cluster/faults](cluster/faults/README.md): fault plans (GPU power cap, InfiniBand port width
  change, loss of one node) on a fixed 7,200 s schedule, with goodput computed from checkpoint
  lineage.
- [scaling/](scaling/README.md): TorchTitan v0.2.0 configurations and launch plans for Llama 3 8B
  (FSDP2 + tensor parallel 2) and a small MoE with expert parallelism, MFU accounting and Nsight
  Systems trace analysis.

**RL post-training** (`rl/`)

- [rl/rollout](rl/rollout/README.md): vLLM 0.10.2 rollout engine; every sample carries the policy
  version that produced it.
- [rl/learner](rl/learner/README.md): FSDP2 GRPO learner with FP32 master weights, BF16 compute and
  a frozen BF16 reference model; group-normalized advantages, clipping, KL penalty and a staleness
  check on every sample.
- [rl/sync](rl/sync/README.md): versioned learner → rollout weight sync with activation and
  rollback. In the training loop every policy weight is copied as BF16 into vLLM in GPU memory
  each step, without touching disk. Across GPUs, `--bucket-mb` sends the packed weights as a few
  large NCCL broadcasts checked by a SHA-256 manifest instead of one broadcast per tensor.
- [rl/verifiers](rl/verifiers/README.md): bounded math verifier (last `\boxed{}` answer) and a code
  grader with a gVisor adapter (not usable on this host, see the limitations).
- `rl/local`: the single-GPU loop (`grpo_run.py`) that time-shares the card between rollout and
  learner. `rl/verl`: the harness for the verl comparison.

**Serving** (`serving/`)

- [serving/replay](serving/replay/README.md): Go open-loop replayer. It sends requests at the
  trace's arrival times, records planned and actual send times, voids a trial when more than 1% of
  requests go out more than 5 ms late, and accounts every request as completed, failed or retried.
- [serving/baseline](serving/baseline/README.md), `serving/eval`: vLLM drivers, SLO and goodput
  calculation, BF16 vs FP8 sessions and accuracy reports.
- `serving/router`, `serving/autoscaler`, `serving/localscale`: Go router (least-queue routing over
  fresh backend snapshots; ejects a backend on its first transport error and can retry a failed
  connect on another backend), Go autoscaler (queue length and p95 TTFT thresholds with
  cooldowns), and a local actuator that starts and stops vLLM replicas.
- [serving/deploy](serving/deploy/README.md), `serving/k8s`: Kubernetes manifests, including the
  prefill/decode disaggregated layouts (1P:1D, 2P:1D, cross-node) and a kind deployment of the
  router and autoscaler.

**Speculative decoding** ([spec/](spec/README.md)): an EAGLE-3 pipeline on SpecForge and SGLang.
The target model regenerates its own answers to the training prompts; hidden-state features are
captured and trained on offline in batches and deleted after each batch; the draft is exported to
SGLang; `bench_tpot.py` compares servers with and without speculation. A public draft can be
served from the same harness for comparison.

**Custom op** (`kernels/`): `torch.ops.crucible.token_logprobs(logits, targets)`, registered through
`torch.library` (custom op, fake tensor and autograd registrations) and working under
`torch.compile(fullgraph=True)`. The CUDA path is a Triton kernel over 1,024-wide vocabulary tiles
that never materializes the softmax; backward writes only the required dense logits gradient. The
learner uses it to score both the policy and the reference model.

## Results

Hardware for every row outside the Snowflake section: one NVIDIA RTX 4090 (24 GB) in a 32-core Linux
workstation that ran other workloads at the same time. GPU jobs ran one at a time under a lock; the CPU was shared. Each row
links the write-up with the full setup, raw outputs and caveats. Longer technical reports (in
Chinese) are in [reports/](reports/) with PDFs in [reports/pdf/](reports/pdf/); commits, model
revisions and environment hashes for the report runs are in
[reports/repro/README.md](reports/repro/README.md).

### RL post-training (Qwen2.5-0.5B-Instruct, math)

Each run trains on 6,400 completions (100 steps × 8 GSM8K training prompts × 8 samples) with a 1/0
reward on the last `\boxed{}` answer. "Async (k=1)" means each batch was generated by the previous
policy version; "sync (k=0)" by the current one. Evaluation is pass@1 with 4 samples per problem and
fixed per-problem seeds, identical before and after training; SE is over per-problem paired
differences.

| What | Setup | Result | Evidence |
|---|---|---|---|
| Held-out accuracy | Async (k=1); a completion that hits the token limit without EOS scores 0; seeds 17/18/19; all 1,319 GSM8K test problems and the 377 of 500 MATH-500 problems the verifier can grade | GSM8K 0.4503 → 0.4874, +3.71 ± 0.67 pp (5.51 SE); MATH-500 0.2467 → 0.2832, +3.65 ± 0.98 pp (3.72 SE). Both clear the 2 SE bar fixed before the runs, and every seed improves. Group 16 (4 prompts per step, same 6,400 samples): +4.45 ± 0.71 pp and +2.61 ± 0.92 pp | [rl-grpo.md](docs/evidence/rl-grpo.md), [pooled.json](docs/evidence/rl-grpo-p5/pooled.json) |
| Cost per step | Same three group-8 runs | Median step 5.47–5.59 s: rollout 1.76–1.82 s, learner 2.62–2.66 s, weight sync 0.018 s. GPU-idle share 0.19 (time outside those three, mostly CPU math grading); 0 stale samples dropped; peak CUDA allocation 14.8 GB | [rl-grpo.md](docs/evidence/rl-grpo.md) |
| Reward hack found | Earlier runs without the EOS rule, one seed each | The sync run learned to write the boxed answer and keep generating: after training, 703 of 800 GSM8K eval completions hit the 512-token limit, and 685 of those already contain a boxed answer. With the EOS rule, sync runs end with 1–9 truncated | [rl-grpo.md](docs/evidence/rl-grpo.md) |
| Against verl 0.9.1 | Sync, EOS rule, seed 17. Same model, the same 800 prompts in the same order, same reward and hyperparameters; both trained models scored by the same eval script on 200 GSM8K and 100 MATH-500 problems | GSM8K 0.4475 (this repo) vs 0.44 (verl), difference −0.75 ± 2.41 pp. MATH-500 0.295 vs 0.34, verl +4.5 ± 2.47 pp (1.8 SE). 11.07 vs 4.51 samples/s: verl was 2.45× slower per sample, mostly from its fp32 reference forward, padded attention without remove-padding and a second process | [t11-verl/README.md](docs/evidence/t11-verl/README.md), [comparison.json](docs/evidence/t11-verl/comparison.json) |
| Weight sync correctness | vLLM logits after a sync vs a Hugging Face BF16 reference: 6 prompts × 2 weight versions, all 151,936 vocabulary entries | All 1,823,232 coordinates within an error bound fixed before the run; 0 failures | [weight-sync.md](docs/evidence/weight-sync.md) |
| Fused log-prob op, isolated | FP32 logits [2, 32, 152000], forward + backward | Incremental peak memory 119,538,688 → 39,847,936 bytes (−66.67%), including the required dense gradient | [fused-logprob.md](docs/evidence/fused-logprob.md) |
| Fused log-prob op in the learner | One full FSDP2 learner step on Qwen2.5-0.5B, 3 alternating baseline/candidate pairs | Peak allocation unchanged (median 10,020,235,776 bytes for both). Median step 234.800 vs 228.713 ms with overlapping ranges, so no speedup is shown. Correctness: 51 CPU tests (FP64 gradcheck, `torch.compile` fullgraph, opcheck) and 16 CUDA tests pass | [fused-logprob.md](docs/evidence/fused-logprob.md) |

### Serving

| What | Setup | Result | Evidence |
|---|---|---|---|
| Trace-replay baseline | vLLM 0.10.2 with CUDA graphs, Qwen3-8B-AWQ (4-bit). Open-loop replay of Azure LLM inference 2023 conv (265 requests) and BurstGPT (123 requests); SLO = p99 TTFT and TPOT from three calibration runs; 3 rounds per level. Azure "1x" is 0.25× the trace's rate, so 4x is its real rate | Azure goodput 1.089 / 2.041 / 0.104 req/s at 1x / 2x / 4x: at 4x every request finishes, but p50 TPOT reaches 33.2 ms against a 21.51 ms SLO. BurstGPT stays inside its SLO up to 4x (1.502 req/s). No timeouts and no voided trial | [serving-baseline.md](docs/evidence/serving-baseline.md) |
| BF16 vs online FP8 | Same Qwen3-8B checkpoint, vLLM BF16 vs `--quantization fp8`, both in eager mode; Azure window at 0.25× the trace's rate; order AB, BA, AB | p50 TPOT 24.6 → 15.5 ms (−37%), p50 TTFT 128 → 89 ms, KV cache 22,192 → 68,016 tokens. Throughput +1.4% and goodput +2.2% only, because BF16 already met the SLO on almost every request. Accuracy: GSM8K 0.920 → 0.910, MMLU-Pro 0.377 → 0.380, both within one SE (about ±1.9 and ±2.8 pp) | [pd-disaggregation.md](docs/evidence/pd-disaggregation.md) |
| Autoscaling on one GPU | Router + autoscaler, up to two vLLM replicas of Qwen2.5-0.5B on the same GPU; a 120 s burst at 4x load; 3 trials each | The new replica served its first request a median 35.5 s after the burst began, but burst goodput fell from 4.00 req/s (one fixed replica) to 0.675 req/s: two processes on one GPU slow every decode step (p50 TPOT 3.7 → 5.6 ms) | [autoscaling.md](docs/evidence/autoscaling.md) |
| Replica kill | 2 replicas, 260 requests at 3 req/s, SIGKILL of one replica at 30 s; 3 trials per condition | Failed requests per kill: 32–34 with the original router, which kept routing to the dead replica until its snapshot expired; 0–1 after ejecting on transport errors and retrying failed connects; 0 failed and 1 retried (then completed) with a client resend of cut streams. Every request is accounted for in every trial | [serving-faults.md](docs/evidence/serving-faults.md) |

### Two A10 GPUs on Snowflake

Hardware: a Snowflake trial account, November 2026. "One node" is Snowflake GPU_NV_2M: 2× A10 24 GB
(vGPU) in one node, no NVLink/P2P. "Two nodes" is 2 Snowflake GPU_NV_SM nodes (1× A10 24 GB each)
over TCP, "four nodes" 4 of them. Each run was a Snowflake ML Job with vLLM 0.10.2, torch 2.8.0 (NCCL 2.27.3) and
transformers 4.55.4 ([scripts/snowflake/](scripts/snowflake/)). Compare the A10 numbers with each
other, not with the 4090 rows.

| What | Setup | Result | Evidence |
|---|---|---|---|
| Network and NCCL baseline | 1 GiB all-reduce and broadcast between the two GPUs; default NCCL settings vs tuned; raw TCP streams; 3 passes each | Two nodes: all-reduce 0.19 → 1.26 GB/s and broadcast 0.27 → 2.6 GB/s with 8 threads × 8 sockets and at least 4 channels. One TCP stream carries about 0.19 GB/s, so the defaults were the limit, not the link. One node: all-reduce 2.5 → 9.9 GB/s and broadcast 2.0 → 18.6 GB/s with `NCCL_SHM_USE_CUDA_MEMCPY=1` | [e0-network.md](docs/evidence/snowflake/e0-network.md) |
| Weight sync over NCCL | The weight-sync correctness check above, with the learner broadcasting BF16 weights over NCCL to vLLM on the other A10; same prompts, versions and error bound | All 1,823,232 coordinates within the bound in both layouts; 0 failures. One 988 MB version took 2.21 s in one node and 5.9 s across two nodes (default socket settings), including a per-tensor SHA-256 | [e1-weight-sync.md](docs/evidence/snowflake/e1-weight-sync.md) |
| Async GRPO on separate GPUs | The RL setup above, seeds 17/18/19. (a) the unchanged time-shared loop on one A10; (b) learner on GPU0 and vLLM rollout on GPU1 of one node; (c) the same across two nodes. Weights synced over NCCL every step | Training time 1,207 s (a), 741 s (b, 1.63× faster) and 933 s (c, 1.29×); 860 s (1.40×) when (c) was rerun with the tuned socket settings from the network baseline. In (b) and (c) the learner GPU is idle 1% of the time; sync takes 1.06 s (b), 2.9 s (c) and 2.0 s (c rerun) per step; every batch is one version old, 0 dropped. GSM8K +3.80 / +3.78 / +2.99 pp (all above 4 SE). MATH-500 +1.79 pp in (a), 1.76 SE, below the bar; +2.21 pp (2.22 SE) in (b) and +3.12 pp (3.08 SE) in (c). No difference between layouts reaches 2 SE | [e2-async-grpo.md](docs/evidence/snowflake/e2-async-grpo.md) |
| Bucketed weight sync | Two nodes. The 988 MB BF16 state packed into 64 MiB buckets, one broadcast each, checked with a SHA-256 manifest over 16 MiB chunks, against the per-tensor protocol (1,478 collectives per push); 3 timed passes. Then the weight-sync check and one layout (c) run, seed 17, with the bucketed path | Push 2.02 s → 0.45 s with 25 collectives; the raw link takes 0.38 s. The weight-sync check is unchanged: all 1,823,232 coordinates within the bound, logits bit-identical to the per-tensor run. Layout (c), one run: sync 2.22 → 0.48 s per step, training 889 → 689 s, 1.75× faster than (a). GSM8K +3.13 pp (3.57 SE); MATH-500 +2.32 pp, 1.95 SE, below the bar. No difference between layouts reaches 2 SE | [e2b-bucketed-sync.md](docs/evidence/snowflake/e2b-bucketed-sync.md) |
| Autoscaling onto a second GPU | The autoscaling setup above in one node, with the new replica on GPU1; fixed 1 replica / autoscaling 1 → 2 / fixed 2 replicas; 3 trials each; SLO re-derived on the A10 (TTFT ≤ 85.7 ms, TPOT ≤ 9.37 ms) | Burst goodput 0.042 / 0.308 (0.025–0.433 across trials) / 0.525 req/s; failed requests 210–213 / 29–40 / 0 of 1,123. The second replica no longer slows the first (burst p50 TPOT 9.45 / 9.45 / 9.50 ms) and served its first request a median 38.5 s after the burst began. Goodput stays low in every condition because one A10 replica's TPOT under a full batch is above the SLO | [e3-autoscaling.md](docs/evidence/snowflake/e3-autoscaling.md) |
| Prefill/decode disaggregation 1P:1D | Qwen3-8B-AWQ with the trace-replay setup above (Azure conv, 3 rounds per level) and vLLM 0.10.2's NCCL KV connector, not the SGLang manifests. Four layouts: one A10; two data-parallel replicas in one node; 1P:1D in one node (KV over NCCL SHM); 1P:1D across two nodes (KV over TCP). SLO re-derived on one A10: TTFT ≤ 2,110.7 ms, TPOT ≤ 90.27 ms | Goodput at 1x / 2x / 4x: one A10 1.06 / 0.22 req/s (4x trials voided, 0 of 265 good); two replicas 1.09 / 2.09 / 2.52; 1P:1D in one node 1.07 / 2.04 / 0.20; 1P:1D across two nodes 1.04 / 1.50 / 0.06. In one node P/D has a lower decode tail at 2x (p99 TPOT 22.8 vs 40.5 ms) but collapses at 4x: the one prefill GPU waits for every KV send (median prefill 0.29 s at 1x, 9.1 s at 4x). A median prompt carries 151 MB of KV, sent at 2.8 GB/s over SHM and 0.36–0.48 GB/s over TCP. Two bugs in vLLM 0.10.2's connector needed fixes | [e4-pd.md](docs/evidence/snowflake/e4-pd.md) |
| Multi-node collectives and scaling | Four nodes. 1 GiB all-reduce and broadcast on 2 / 3 / 4 nodes, 3 passes. Data-parallel (DDP) training of Qwen2.5-0.5B on 1 / 2 / 4 nodes with fp32 gradients: fixed per-GPU batch (weak scaling) and a fixed global batch of 16 sequences (strong scaling), 3 rounds. TorchTitan's pinned nightly stack could not be installed | All-reduce 1.29 / 1.03 / 0.83 GB/s on 2 / 3 / 4 nodes with tuned settings, about 0.2 GB/s by default; NCCL used TCP sockets, as the nodes have no InfiniBand device. Scaling efficiency: weak 0.54 on 2 nodes and 0.36 on 4, strong 0.38 and 0.12. The 1.98 GB gradient all-reduce takes 2.0 s on 2 nodes and 4.2 s on 4, against 2.0 s of compute per step. DCGM, nccl-tests and perftest are not on these nodes, and CUPTI does not start on the vGPU | [e5-multinode.md](docs/evidence/snowflake/e5-multinode.md) |
| Slow-node drain and rank-kill restart | Four nodes. Drain: one node slowed by a competing CUDA process, step times fed to the drain controller in dry-run mode. Fault: SIGKILL of one rank's process after step 25, all ranks restarted from the step-20 checkpoint, 3 rounds | The node-local step time flagged only the slowed node, once, at its third slow sample; the synchronized step time slows every node equally and flags none. Goodput 0.430 (0.422–0.445) with checkpoints on disk, where each 6.48 GB save takes 39–49 s, and 0.673 with `/dev/shm`. Every round lost 5 steps and the first replayed step reproduced the loss. This is a process kill, not a node, power or network fault | [e5-multinode.md](docs/evidence/snowflake/e5-multinode.md) |

### Speculative decoding (EAGLE-3, Qwen3-4B target)

The draft uses SpecForge's standard Qwen3-4B EAGLE-3 config and was trained for 60,480 steps on
60,697 UltraChat conversations that Qwen3-4B regenerated itself (greedy, thinking off); feature
capture and training took 5.50 GPU-hours. The benchmark serves the target with SGLang 0.5.18 and
sends 128 held-out UltraChat prompts (disjoint from the training data), greedy, at most 256 new
tokens, closed loop at 1/4/16/64 requests in flight, with 3 draft steps, top-k 1 and 4 draft
tokens. Two rounds, a server restart before every run, all 3,072 requests completed. TPOT is the
median over 128 requests; values are round 1 / round 2. The public draft is
[AngelSlim/Qwen3-4B_eagle3](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3) on the same harness.
Source: [EAGLE-3 write-up](docs/evidence/spec-decoding/phase7/README.md), summaries
[ours](docs/evidence/spec-decoding/phase7/main/summary.json) and
[public](docs/evidence/spec-decoding/phase7/main/summary-public.json).

| Batch | Base TPOT (ms) | EAGLE-3 TPOT (ms) | TPOT speedup | Acceptance length | Public draft: TPOT speedup; acceptance |
|---|---|---|---|---|---|
| 1 | 9.31 / 9.32 | 5.66 / 5.65 | 1.646 / 1.649 | 2.159 / 2.159 | 1.629 / 1.633; 2.107 / 2.110 |
| 4 | 10.15 / 10.16 | 6.07 / 6.07 | 1.670 / 1.673 | 2.152 / 2.152 | 1.675 / 1.669; 2.108 / 2.114 |
| 16 | 11.02 / 11.03 | 7.26 / 7.22 | 1.518 / 1.528 | 2.156 / 2.160 | 1.507 / 1.509; 2.104 / 2.113 |
| 64 | 13.62 / 13.45 | 11.78 / 11.87 | 1.157 / 1.133 | 2.165 / 2.164 | 1.141 / 1.118; 2.111 / 2.106 |

- An earlier draft with a reduced vocabulary and MLP, trained on only 240 conversations, reached an
  acceptance length of 1.42 and made batch 64 slower than no speculation (TPOT speedup 0.772 /
  0.803) ([spec-decoding.md](docs/evidence/spec-decoding.md)).
- The public draft is within 1–3% of ours. The held-out prompts come from UltraChat, the same
  family as our training data, which may favour our draft.
- Three other configurations were [declared before running](docs/evidence/spec-decoding/phase7/grid-declaration.md)
  and measured at batch 1 and 64 only. Tree drafting (top-k 4, 8 draft tokens) is fastest at batch
  1 (TPOT speedup 1.882 / 1.900, acceptance 2.595 / 2.588); a 2-step chain is fastest at batch 64
  (1.286 / 1.295); the 5-step chain and the tree are slower than no speculation at batch 64. No
  single configuration wins at both.

## Not done and limitations

Not run on hardware:

- **Multi-node cluster work on the planned hardware.** The four-node H100 checks, burn-in suite,
  drain controller, fault plans and scaling runs exist as code, configurations and launch plans,
  tested with CPU unit tests, command shims and synthetic fixtures
  ([provisioning](docs/evidence/cluster-baseline.md), [burn-in](docs/evidence/burn-in.md),
  [drain](docs/evidence/drain.md), [faults](docs/evidence/fault-goodput.md),
  [scaling](docs/evidence/scaling.md)). Only their logic ran on hardware, once, on four Snowflake
  A10 nodes over TCP. There are no InfiniBand/HCA tests, DCGM burn-in, Slurm or Kubernetes drain
  actions, node or power faults, TorchTitan runs, or H100 scaling and MFU numbers.
- **Prefill/decode disaggregation as designed.** The SGLang manifests in `serving/deploy/pd-*.yaml`
  (1P:1D, 2P:1D and cross-node, KV over RDMA) were never run. 1P:1D was measured only on a different
  stack: vLLM's NCCL KV connector on two A10s, KV over SHM or TCP. 2P:1D was not measured.
- **Kubernetes deployment of the router and autoscaler.** The kind node started, but kube-proxy
  crash-looped with "too many open files" (the host's inotify instance limit), so the deployment was
  not exercised ([record](docs/evidence/t14-kind.md)).
- **Code rewards.** Rootless gVisor could not start on this host
  ([record](docs/evidence/sandbox-runtime.md)), so RL used math rewards only; the code grader and
  sandbox adapter are tested only against fixtures.

Scaled down to fit one GPU: a 0.5B model for RL; Qwen3-8B (AWQ for the baseline) instead of the
planned Qwen3-30B-A3B MoE for serving, with vLLM instead of SGLang for the serving baseline; and
Qwen3-4B as the speculative decoding target.

Limits of the results above:

- On the 4090, rollout and learning take turns, so "async" changes only how stale the training data
  is; step times match the sync runs. Only the A10 runs on Snowflake overlap them.
- MATH-500 is restricted to the 377 problems the verifier can grade, a disclosed selection. The group
  size and clip ablations have one seed each, on smaller held-out sets, and fall within the
  seed-to-seed spread of the reference configuration.
- The verl comparison has one seed per framework on 200 / 100 problems; its SE covers eval sampling,
  not training seeds. Twelve implementation differences remain (seeding, Bessel's correction in the
  GRPO std, process layout, library versions and others), listed in
  [scripts/run_t11_verl.sh](scripts/run_t11_verl.sh). Its peak-memory figures are not comparable
  (one process's torch allocator vs whole-GPU nvidia-smi).
- Serving workloads are deterministic (same arrivals, seeded prompts, forced output lengths), so the
  run-to-run ranges show system noise, not variation between trace samples. Each trace has one
  evaluation window, and the p99 SLOs come from 191 and 105 calibration requests.
- FP8 is vLLM's online dynamic per-tensor quantization, not a calibrated FP8 checkpoint, measured
  at one load level.
- On the 4090 the second replica only slows the first. On two A10s it does not, but there the TPOT
  SLO caps goodput in every condition, and autoscaling goodput varies a lot between trials. In every
  autoscaling trial a short cluster of arrivals caused a second scale-up about 300 s after the load
  dropped.
- Snowflake runs: at most four GPUs, at most two per node; the A10s are virtual GPUs ("A10-24Q"); the network is shared TCP
  with no RDMA, and results vary between runs. With the per-tensor sync protocol (about 1,500
  collectives per push), two-node weight sync in GRPO stays at about 2 s per step even with tuned
  sockets; bucketed sync brings it to 0.48 s, about 0.1 s above the link, in one run with one seed.
  Link speed differs between node pairs and was not controlled. The A10 GRPO runs reuse the base-model
  eval of the time-shared run for each seed (checked identical for seed 17). In the P/D runs the
  one-A10 and two-node layouts ran on GPU_NV_SM nodes and the others on GPU_NV_2M, and the final
  P/D configuration had no separate smoke run.
- Speculative decoding: one target model, one dataset family, 256 output tokens. The tree
  configuration's two rounds differ a lot at batch 64 (22.37 vs 16.14 ms TPOT).
- The fused log-prob op reduces memory only in isolation; a whole learner step shows no memory or
  speed gain. It supports first-order autograd and FP32/BF16 (plus FP64 on CPU), not FP16.
- Timings come from a shared machine; compare numbers within one run.

## Repository layout

| Path | Contents |
|---|---|
| `cluster/` | provisioning checks (Slurm, Kubernetes), burn-in, health and drain, fault plans |
| `scaling/` | TorchTitan scaling configurations, MFU and Nsight Systems analysis |
| `rl/` | rollout, learner, weight sync, verifiers, evaluation, single-GPU GRPO loop, verl harness, configs |
| `serving/` | replayer, baseline drivers, router, autoscaler, local replica actuator, fault tests, evaluation, Kubernetes manifests |
| `spec/` | EAGLE-3 data, training-cycle and benchmark scripts, SpecForge configs |
| `kernels/` | fused log-prob custom op (Triton) and its learner benchmark |
| `scripts/` | bootstrap, GPU wrapper, resource preflight, GPU run scripts, report tools |
| `docs/evidence/` | per-experiment write-ups and raw outputs behind the results |
| `reports/` | technical reports (Chinese), PDFs and the reproduction appendix |

## Build and test

Linux amd64 with [uv](https://docs.astral.sh/uv/) and make. Python 3.12, the virtual environments
and the Go toolchain (1.27.3) are installed inside the checkout (`.tools/`, `.venv*`, `.cache/`),
nothing system-wide.

```sh
make bootstrap      # .venv with the Python dev dependencies, Go toolchain in .tools/
make bootstrap-ml   # .venv-ml: PyTorch 2.8.0, Triton 3.4.0, CUDA 12.8 wheels; no model downloads
make test           # offline suites: Python, torch tests in .venv-ml, Go (serving/); needs both envs
make check          # ruff, gofmt, go vet
```

`make test` hides the GPU; `CRUCIBLE_GPU_TESTS=1` opts in to GPU tests (run them through the GPU
wrapper below) and `CRUCIBLE_LIVE_TESTS=1` to live integration tests. Other environments:
`scripts/bootstrap_rollout.sh` adds vLLM 0.10.2 to `.venv-ml`, `scripts/bootstrap_specforge.sh`
builds `.venv-specforge` (SpecForge, SGLang 0.5.18) and `scripts/bootstrap_verl.sh` builds
`.venv-verl` (verl 0.9.1 with its own torch and vLLM).

GPU runs (scripts without `with_gpu.sh` in front take the GPU lock themselves):

```sh
bash scripts/run_grpo_gpu.sh <out> --staleness 1 --reward eos --seed 17   # one GRPO run
bash scripts/with_gpu.sh -- bash scripts/run_t11_verl.sh <out>            # verl comparison
bash scripts/run_serving_baseline_gpu.sh azure <out>                      # or burst
bash scripts/run_fp8_compare_gpu.sh <out>                                 # BF16 vs FP8
make t14-bins && bash scripts/run_localscale_gpu.sh scale <out>           # also cold, fault
bash scripts/with_gpu.sh -- bash spec/t15_gpu.sh bench <out>              # also regen, features, cycle
bash scripts/test_logprob_gpu.sh                                          # custom op CUDA tests + learner benchmark
```

The run scripts point at model snapshots by path in the development machine's Hugging Face cache
and expect prepared data under `.cache/`; [reports/repro/README.md](reports/repro/README.md) lists
data preparation, commits and model revisions.

Operational notes:

- **GPU wrapper.** `scripts/with_gpu.sh -- <command>` runs a foreground command under an exclusive
  `flock` on `/var/tmp/rtx4090.lock`, so one GPU job runs at a time on the machine. It refuses
  background or daemon flags (`nohup`, `setsid`, `--daemon`, ...), stops the job after at most 4
  hours, and releases the lock only once every descendant process has exited. Do not nest it: the
  `run_*_gpu.sh` scripts already call it.
- **Disk and caches.** Models are read from the shared Hugging Face cache (`HF_HOME`, default
  `~/.cache/huggingface`), not downloaded into the project. `make resource-preflight
  PREFLIGHT_ARGS='--planned-project-bytes N --download-bytes N --checkpoints N --service-memory-bytes N'`
  checks a planned download or run against free disk before it starts. Environment sizes:
  `.venv-ml` about 6.4 GiB, `.venv-specforge` 9.08 GB, `.venv-verl` 10.15 GB. EAGLE-3 features
  take about 17.6 MB per conversation (about 1.08 TB for the full data set), which is why training
  captures, trains on and deletes them in batches of 576 conversations; peak project disk in that
  run was 39.2 GB.
