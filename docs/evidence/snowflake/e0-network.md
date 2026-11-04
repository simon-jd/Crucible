# E0: network and communication baseline on Snowflake GPU nodes

Date: 2026-11-04. Branch `snowflake/e0`. Payload: `scripts/snowflake/payloads/e0_network.py`
(CPU tests: `scripts/tests/test_e0_network.py`). Raw outputs: `docs/evidence/snowflake/e0/<run>/`
(`report.json` with every RESULT, `instance<i>.log`, `driver.log`).

Purpose: give the later experiments (weight sync, async GRPO, P/D KV transfer) measured link rates and the
NCCL settings to use, and a way to explain their own numbers.

## Setup

| | |
|---|---|
| Hardware A | 2 Snowflake GPU_NV_SM nodes (1× A10 24 GB each) over TCP, pool `SYSTEM_COMPUTE_POOL_GPU`, `--instances 2` |
| Hardware B | Snowflake GPU_NV_2M: 2× A10 24 GB (vGPU "A10-24Q") in one node, no NVLink/P2P, pool `CRUCIBLE_2XA10_POOL` |
| Software | `--pip torch==2.8.0 numpy==2.2.6` → torch 2.8.0+cu12.8, **NCCL 2.27.3** (`nvidia-nccl-cu12 2.27.3`, log: `NCCL version 2.27.3+cuda12.9`), Ray 2.55.1, driver 570.211.01 |
| Node facts | GPU_NV_SM: 36 CPUs; GPU_NV_2M: 72 CPUs. Pod interface `eth0`, MTU 1500, reported speed 10000 (virtual; NCCL exceeded it), TCP cubic, `tcp_wmem` max 4 MiB, ephemeral ports 32768–59999 |
| GPU_NV_2M topology | `nvidia-smi topo -m`: GPU0–GPU1 `SYS` (GPU0 on NUMA 2, GPU1 on NUMA 0), `can_device_access_peer` false both ways |
| Ports | raw TCP on 12900, NCCL rendezvous 12901–12919 (E0's band); NCCL's own data sockets are ephemeral (open range) |

Method:
- Two nodes: one Ray actor per node (STRICT_SPREAD). Raw TCP is a Python socket sender/receiver (4 MiB
  `sendall`/`recv_into`, one thread per stream), 2 GiB per test, timed first to last byte at the receiver.
  "bidir" = both nodes send 2 GiB to each other at the same time (the pattern of a 2-rank ring all-reduce).
- NCCL: `torch.distributed` all-reduce and broadcast (src 0) of fp32 tensors of 1 MiB, 16 MiB, 256 MiB, 1 GiB,
  after warm-up and a barrier; (warm-up, timed) calls = (5, 50), (3, 20), (1, 5), (1, 3) across nodes and
  (5, 50), (3, 20), (2, 10), (2, 5) on one node. GB/s = bytes / s / 1e9; for 2 ranks bus bandwidth equals
  algorithm bandwidth. The slower rank's time counts. Every variant runs in fresh processes (NCCL reads its
  env once), with `NCCL_IB_DISABLE=1`; NCCL logs confirm `NET/Socket` and the requested thread/socket counts.
- Repeats: every config ran in 3 interleaved passes (pass 1 = all variants, then pass 2, then pass 3), so the
  spread includes time drift; "default" (no socket env) ran once. Each run also checks a small all-reduce and
  broadcast result (all `check_ok` true).
- One node: `torchrun --standalone --nproc_per_node=2` per variant, 3 interleaved passes; copies are
  `tensor.copy_` between `cuda:0`/`cuda:1`/pinned host, 5 timed copies × 3 repeats.

Runs (ledger estimates are upper bounds: submit-to-stop wall time × nodes × rate):

| Run | Pool | What | Wall s | Credits |
|---|---|---|---|---|
| `smoke-2node` | GPU_NV_SM ×2 | `--quick` (small sizes, 1x1/4x4, 256 MiB TCP) | 474 | 0.448 |
| `smoke-1node` | GPU_NV_2M | `--quick` | 352 | 0.343 |
| `full-2node` | GPU_NV_SM ×2 | TCP 1/8/32 streams, bidir 8/32; NCCL default + 5-variant grid × 3 | 1174 | 1.109 |
| `follow-2node` | GPU_NV_SM ×2 | TCP 32/64/128, bidir 64/128; NCCL 8x8, 16x4, 8x8+16 MiB buffer, 8x8+4 channels (± 16 MiB) × 3 | 810 | 0.765 |
| `full-1node` | GPU_NV_2M | copies; SHM default / CUDA memcpy / + mode 3 × 3 | 384 | 0.373 |
| Total | | | | **3.038** |

Payload time on the nodes was 699 s (full-2node), 307 s (follow-2node) and 88 s (full-1node); the rest is
node start, the pip install (~3 min) and queueing behind other agents' jobs.

## 1. Two GPU_NV_SM nodes over TCP

### Raw TCP (GB/s, median [min–max] of 3; 2 GiB each)

| Streams | a→b | b→a | bidir, into a / into b (each at once) |
|---|---|---|---|
| 1 | 0.195 [0.186–0.198] | 0.191 [0.190–0.195] | – |
| 8 | 0.595 [0.556–0.608] | 0.621 [0.439–0.629] | 0.418 / 0.411 |
| 32 (full) | 0.991 [0.766–1.020] | 1.054 [0.960–1.129] | 0.659 / 0.678 |
| 32 (follow) | 1.046 [0.958–1.053] | 1.107 [1.102–1.161] | – |
| 64 (follow) | 1.129 [1.124–1.366] | 1.163 [1.111–1.386] | 0.742 / 0.749 |
| 128 (follow) | 1.098 [0.958–1.105] | 1.103 [1.093–1.117] | 0.646 / 0.647 |

full-2node and follow-2node ran on different node pairs; 32 streams agree within 6–10%. Loopback on one node
with the same tool: 10.3 / 8.0 GB/s (1 stream), 26.5 / 27.0 GB/s (8), 9.4 GB/s (128), so the tool itself is
not the limit at these rates. A single TCP flow stays at ~0.19 GB/s in both directions; aggregate rate grows
with flow count up to ~1.1 GB/s one way (32–128 streams) and ~1.4–1.5 GB/s summed when both ways run at once.

### NCCL NET/Socket (GB/s, median of 3 passes [min–max]; default n=1)

All-reduce:

| Variant | 1 MiB | 16 MiB | 256 MiB | 1 GiB |
|---|---|---|---|---|
| default (unset) | 0.169 | 0.197 | 0.193 | 0.189 |
| 1x1 | 0.141 [0.115–0.185] | 0.176 [0.166–0.222] | 0.173 [0.169–0.203] | 0.225 [0.198–0.226] |
| 2x2 | 0.162 [0.148–0.175] | 0.333 [0.320–0.334] | 0.274 [0.274–0.284] | 0.261 [0.249–0.281] |
| 4x4 | 0.170 [0.167–0.174] | 0.459 [0.452–0.482] | 0.428 [0.427–0.441] | 0.394 [0.388–0.404] |
| 8x8 | 0.190 [0.189–0.192] | 0.677 [0.652–0.679] | 0.764 [0.746–0.765] | 0.758 [0.758–0.781] |
| 4x4 + BUFFSIZE 16 MiB | 0.171 [0.163–0.172] | 0.626 [0.622–0.633] | 0.666 [0.597–0.667] | 0.600 [0.591–0.603] |
| *follow-up:* 8x8 | 0.186 [0.183–0.195] | 0.672 [0.672–0.682] | 0.786 [0.744–0.794] | 0.775 [0.754–0.777] |
| 16x4 | 0.192 [0.190–0.198] | 0.688 [0.683–0.700] | 0.772 [0.766–0.780] | 0.786 [0.773–0.790] |
| 8x8 + BUFFSIZE 16 MiB | 0.191 [0.187–0.193] | 0.710 [0.692–0.751] | 0.960 [0.913–0.961] | 0.910 [0.884–0.918] |
| 8x8 + MIN_NCHANNELS 4 | 0.243 [0.242–0.262] | 0.726 [0.718–0.737] | 1.046 [1.016–1.070] | 1.051 [1.032–1.058] |
| **8x8 + MIN_NCHANNELS 4 + BUFFSIZE 16 MiB** | 0.239 [0.228–0.251] | 0.748 [0.727–0.754] | **1.246** [1.245–1.256] | **1.262** [1.220–1.317] |

Broadcast:

| Variant | 1 MiB | 16 MiB | 256 MiB | 1 GiB |
|---|---|---|---|---|
| default (unset) | 0.237 | 0.279 | 0.282 | 0.265 |
| 1x1 | 0.303 [0.297–0.311] | 0.309 [0.307–0.311] | 0.310 [0.307–0.316] | 0.312 [0.311–0.317] |
| 2x2 | 0.531 [0.507–0.552] | 0.596 [0.529–0.632] | 0.607 [0.601–0.627] | 0.570 [0.569–0.604] |
| 4x4 | 0.655 [0.615–0.701] | 0.843 [0.833–0.958] | 0.864 [0.860–0.968] | 0.826 [0.808–0.890] |
| 8x8 | 1.072 [0.763–1.278] | 1.434 [0.245–1.526] | 1.602 [1.528–1.606] | 1.484 [1.484–1.558] |
| 4x4 + BUFFSIZE 16 MiB | 0.549 [0.523–0.611] | 0.859 [0.801–0.932] | 0.913 [0.881–1.034] | 0.935 [0.893–1.025] |
| *follow-up:* 8x8 | 1.206 [1.193–1.312] | 1.490 [1.409–1.569] | 1.537 [1.498–1.561] | 1.527 [1.471–1.535] |
| 16x4 | 1.255 [1.215–1.483] | 1.466 [1.427–1.585] | 1.505 [1.480–1.553] | 1.551 [1.350–1.552] |
| 8x8 + BUFFSIZE 16 MiB | 0.985 [0.960–1.033] | 1.742 [1.709–1.877] | 1.742 [1.730–1.751] | 1.769 [1.577–1.883] |
| **8x8 + MIN_NCHANNELS 4** | 1.254 [1.224–1.401] | 2.215 [2.026–2.307] | **2.389** [2.272–2.527] | **2.593** [2.464–2.692] |
| 8x8 + MIN_NCHANNELS 4 + BUFFSIZE 16 MiB | 1.102 [1.004–1.149] | 2.492 [2.194–2.652] | 2.023 [1.035–2.752] | 2.101 [1.545–2.883] |

### Spread

- At ≥ 16 MiB most configs vary by under 10% across the 3 passes; the 8x8 anchor repeated on a second node pair
  (follow-up) within 2–4% of the first.
- Outliers: one 8x8 16 MiB broadcast pass at 0.245 GB/s (others ~1.5); 8x8 + 4 channels + 16 MiB buffer
  broadcast swings 1.0–2.9 GB/s at 256 MiB/1 GiB, so it is not recommended for broadcast; raw TCP 8 streams b→a
  had one 0.44 run and 64 streams one 1.37–1.39 run.
- 1 MiB is latency-bound everywhere: ~4–7 ms per all-reduce (0.14–0.26 GB/s), more threads do not help it;
  4 channels help a little (0.24 GB/s).

### Does tuned NCCL reach the raw TCP ceiling?

Yes, and it passes what the raw test measured. The link behaves like a per-flow cap (~0.19 GB/s, a constant
across directions and node pairs) times the number of flows, flattening at ~1.1 GB/s one way for 32–128 Python
streams. NCCL's default/1x1 (2 channels, 1 socket each) sits at the single-flow rate (0.19 all-reduce). 8x8
matches the raw 32–128-stream bidirectional rate for all-reduce (0.76–0.79 vs 0.65–0.75 GB/s per direction)
and beats the raw one-way rate for broadcast (1.5 vs 1.1 GB/s). With 4 channels × 64 sockets (256 flows per
direction) all-reduce reaches 1.26 GB/s (6.7× default) and broadcast 2.6 GB/s (9.8× default), above every raw
TCP number here. So the raw Python test is a floor, not a hard ceiling: the true link limit was not reached
and is at least 2.6 GB/s one way. NCCL caps threads × sockets at 64 per connection, so more flows come from
more channels; 8 channels were not tried.

## 2. Snowflake GPU_NV_2M: 2× A10 24 GB (vGPU) in one node, no NVLink/P2P

Copies (GB/s, median of 3, spread ≤ 1.2%):

| Copy | 16 MiB | 256 MiB | 1 GiB |
|---|---|---|---|
| GPU0 → GPU1 | 20.18 | 21.57 | 21.77 |
| GPU1 → GPU0 | 15.58 | 16.36 | 16.48 |
| pinned host → GPU0 | 26.54 | 26.65 | 26.72 |
| GPU0 → pinned host | 26.17 | 26.34 | 26.35 |

The two directions differ by ~30%; the smoke run (another node) had the reverse (GPU0→GPU1 16.9,
GPU1→GPU0 21.7 at 256 MiB), so it depends on the node's NUMA placement, not on the GPU index.

NCCL (SHM transport; GB/s, median [min–max] of 3):

| Variant (NCCL log transport) | op | 1 MiB | 16 MiB | 256 MiB | 1 GiB |
|---|---|---|---|---|---|
| default (`SHM/direct/direct`) | all-reduce | 1.979 [1.95–2.00] | 2.311 [2.24–2.36] | 2.390 [2.32–2.41] | 2.480 [2.38–2.49] |
| | broadcast | 2.110 [2.10–2.15] | 2.012 [2.01–2.12] | 1.956 [1.94–2.08] | 1.973 [1.92–2.08] |
| `NCCL_SHM_USE_CUDA_MEMCPY=1` (`SHM/CE/direct`) | all-reduce | 4.627 [4.45–4.72] | 9.731 [9.73–9.76] | 9.887 [9.88–9.93] | **9.947** [9.88–9.95] |
| | broadcast | 14.87 [14.86–16.77] | 18.26 [17.98–19.59] | 18.61 [18.57–19.80] | **18.60** [18.60–19.83] |
| + `NCCL_SHM_MEMCPY_MODE=3` (`SHM/CE/CE`) | all-reduce | 3.279 [3.15–4.06] | 9.535 [9.43–9.60] | 10.068 [9.97–10.11] | 10.084 [9.98–10.12] |
| | broadcast | 9.959 [8.70–10.47] | 16.06 [15.35–17.54] | 16.21 [15.72–18.61] | 16.28 [15.73–18.69] |

`NCCL_SHM_USE_CUDA_MEMCPY=1` gives 4.0× the default all-reduce and 9.4× the default broadcast at 1 GiB;
broadcast then reaches ~85–113% of the plain GPU↔GPU copy (it moves data through host memory with copy
engines). Mode 3 (copy engine on both sides) adds ~1% to large all-reduce and costs 13% on broadcast and ~30%
at 1 MiB. These match the earlier probe (2.3 / 8.4 GB/s all-reduce) and improve on its 8.4 with torch 2.8 /
NCCL 2.27.3.

## Settings to use

Two GPU_NV_SM nodes (NCCL over TCP), for weight sync / broadcast and cross-node all-reduce:

```
NCCL_IB_DISABLE=1
NCCL_SOCKET_NTHREADS=8
NCCL_NSOCKS_PERTHREAD=8
NCCL_MIN_NCHANNELS=4
```

Expect ~2.4–2.6 GB/s broadcast and ~1.05 GB/s all-reduce at ≥ 256 MiB; a 1 GiB bf16 weight broadcast is
~0.4 s. For all-reduce-heavy work add `NCCL_BUFFSIZE=16777216` (1.25 GB/s), but not for broadcast (unstable,
1.0–2.9 GB/s). If more channels are not wanted, `8/8` alone gives 0.76 all-reduce / 1.5 broadcast. Small
messages (≤ 1 MiB) cost ~4–7 ms each whatever the setting, so batch them. For a raw-socket path (KV transfer,
HTTP), one TCP connection is capped at ~0.19 GB/s: use ≥ 32 parallel connections (~1.0–1.1 GB/s one way).
Keep every cross-node listening port in the experiment's band (12900–12999).

Two GPUs on one GPU_NV_2M node (NCCL SHM, no P2P):

```
NCCL_SHM_USE_CUDA_MEMCPY=1
```

(Leave `NCCL_SHM_MEMCPY_MODE` unset.) Expect ~9.9 GB/s all-reduce and ~18.6 GB/s broadcast at ≥ 16 MiB. A
same-process `copy_` GPU→GPU is 16–22 GB/s depending on direction; pinned host↔GPU is ~26.5 GB/s.

## Caveats and deviations

- Deviation: the brief asked for raw TCP at 1 and 8 streams. Those did not bound NCCL, so the full run added 32
  streams (and bidir 8/32), and one extra follow-up run (0.765 credits) added 64/128 streams and the 16x4,
  8x8+buffer and 4-channel variants. The brief's grid is complete as asked.
- The raw TCP test is Python sockets, not iperf (no internet egress for packages); the loopback check shows the
  tool handles ≥ 8 GB/s, but a C tool with more flows might go higher than 1.1 GB/s.
- Two-node runs used two different node pairs; the pods' network beneath `eth0` (overlay, VM NIC) is not
  visible from the container, and the reported 10000 Mb/s speed is not a real cap (NCCL moved 2.6 GB/s).
- Only 3 passes per config, 3–5 timed calls at 256 MiB/1 GiB; medians, not long-run averages. Other agents
  shared the pools; the driver's pool lock serializes driver jobs, so no other driver job ran on the nodes
  during these runs.
- Credits are the driver's upper-bound estimates (include node start and queueing), not billed amounts.
- The smoke and full runs used the payload before `--tcp-streams`/`--bidir-streams`/`--variants` were added
  (same defaults); the follow-up used those flags. After the runs the payload got lint-only edits (ruff: loop-variable binding via default args, `Path`
  reads, `check=False`, `noqa` on best-effort `except`), no change in what it measures.
