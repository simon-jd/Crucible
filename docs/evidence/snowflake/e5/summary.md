## health

| Node | Present | Absent | IB sysfs (link layer, rate) | /dev/infiniband | bf16 matmul TFLOPS 4096 / 8192 | CUPTI kernel records | Disk write GB/s (/tmp, /dev/shm) |
|---|---|---|---|---|---|---|---|
| job-0 | ibstat, ncu, nvcc | all_reduce_perf, dcgmi, ib_write_bw, mpirun, nsys, nv-hostengine | mlx5_0/1: Ethernet, 100 Gb/sec (4X EDR) | absent | 70.8 / 79.1 | 0 | 0.111 / 3.372 |
| job-1 | ibstat, ncu, nvcc | all_reduce_perf, dcgmi, ib_write_bw, mpirun, nsys, nv-hostengine | mlx5_0/1: Ethernet, 100 Gb/sec (4X EDR) | absent | 63.3 / 76.9 | 0 | 0.115 / 3.433 |
| job-3 | ibstat, ncu, nvcc | all_reduce_perf, dcgmi, ib_write_bw, mpirun, nsys, nv-hostengine | mlx5_0/1: Ethernet, 100 Gb/sec (4X EDR) | absent | 61.2 / 75.9 | 0 | 0.135 / 3.473 |
| job-2 | ibstat, ncu, nvcc | all_reduce_perf, dcgmi, ib_write_bw, mpirun, nsys, nv-hostengine | mlx5_0/1: Ethernet, 100 Gb/sec (4X EDR) | absent | 63.8 / 79.4 | 0 | 0.159 / 3.46 |

## health

| Node | Present | Absent | IB sysfs (link layer, rate) | /dev/infiniband | bf16 matmul TFLOPS 4096 / 8192 | CUPTI kernel records | Disk write GB/s (/tmp, /dev/shm) |
|---|---|---|---|---|---|---|---|
| job-0 | ibstat, ncu, nvcc | all_reduce_perf, dcgmi, ib_write_bw, mpirun, nsys, nv-hostengine | mlx5_0/1: Ethernet, 100 Gb/sec (4X EDR) | absent | 65.6 / 78.5 | 0 | 0.137 / 3.463 |
| job-2 | ibstat, ncu, nvcc | all_reduce_perf, dcgmi, ib_write_bw, mpirun, nsys, nv-hostengine | mlx5_0/1: Ethernet, 100 Gb/sec (4X EDR) | absent | 62.8 / 75.9 | 0 | 0.136 / 3.446 |
| job-1 | ibstat, ncu, nvcc | all_reduce_perf, dcgmi, ib_write_bw, mpirun, nsys, nv-hostengine | mlx5_0/1: Ethernet, 100 Gb/sec (4X EDR) | absent | 59.8 / 75.5 | 0 | 0.128 / 3.479 |
| job-3 | ibstat, ncu, nvcc | all_reduce_perf, dcgmi, ib_write_bw, mpirun, nsys, nv-hostengine | mlx5_0/1: Ethernet, 100 Gb/sec (4X EDR) | absent | 62.8 / 78.8 | 0 | 0.158 / 3.428 |

## collectives

| Nodes | Variant | Op | 1 MiB | 16 MiB | 256 MiB | 1024 MiB |
|---|---|---|---|---|---|---|
| 2 | default | all_reduce | 0.173 | 0.206 | 0.189 | 0.179 |
| 2 | default | broadcast | 0.238 | 0.250 | 0.254 | 0.247 |
| 2 | ib_enabled | all_reduce | 0.180 | 0.170 | – | – |
| 2 | ib_enabled | broadcast | 0.200 | 0.222 | – | – |
| 2 | tuned | all_reduce | 0.252 [0.248–0.262] | 0.724 [0.630–0.734] | 1.043 [1.033–1.092] | 1.072 [1.037–1.078] |
| 2 | tuned | broadcast | 1.280 [1.134–1.471] | 2.176 [1.038–2.382] | 2.316 [0.536–2.322] | 2.605 [2.557–2.664] |
| 2 | tuned_buff16m | all_reduce | 0.237 [0.222–0.244] | 0.745 [0.711–0.776] | 1.322 [1.258–1.357] | 1.287 [1.242–1.367] |
| 2 | tuned_buff16m | broadcast | 1.144 [1.072–1.149] | 2.962 [2.606–3.063] | 2.739 [1.112–2.903] | 2.470 [2.289–2.917] |
| 3 | default | all_reduce | 0.204 | 0.221 | 0.213 | 0.207 |
| 3 | default | broadcast | 0.189 | 0.211 | 0.218 | 0.219 |
| 3 | tuned | all_reduce | 0.480 [0.457–0.492] | 0.651 [0.625–0.697] | 0.757 [0.389–0.761] | 0.832 [0.831–0.872] |
| 3 | tuned | broadcast | 0.225 [0.214–0.579] | 0.273 [0.164–0.363] | 0.581 [0.391–1.015] | 1.154 [0.652–1.155] |
| 3 | tuned_buff16m | all_reduce | 0.197 [0.193–0.203] | 0.676 [0.673–0.683] | 0.944 [0.924–1.084] | 1.029 [1.029–1.075] |
| 3 | tuned_buff16m | broadcast | 0.569 [0.525–0.619] | 0.802 [0.606–1.144] | 0.451 [0.392–1.164] | 0.904 [0.645–1.513] |
| 4 | default | all_reduce | 0.164 | 0.234 | 0.214 | 0.211 |
| 4 | default | broadcast | 0.180 | 0.221 | 0.230 | 0.233 |
| 4 | ib_enabled | all_reduce | 0.161 | 0.184 | – | – |
| 4 | ib_enabled | broadcast | 0.180 | 0.198 | – | – |
| 4 | tuned | all_reduce | 0.320 [0.300–0.336] | 0.581 [0.426–0.596] | 0.669 [0.332–0.784] | 0.792 [0.652–0.847] |
| 4 | tuned | broadcast | 0.568 [0.442–0.607] | 0.792 [0.283–0.845] | 0.869 [0.844–1.093] | 0.908 [0.747–1.233] |
| 4 | tuned_buff16m | all_reduce | 0.138 [0.126–0.147] | 0.578 [0.323–0.581] | 0.708 [0.561–0.822] | 0.827 [0.696–0.830] |
| 4 | tuned_buff16m | broadcast | 0.444 [0.395–0.641] | 0.749 [0.215–0.787] | 0.998 [0.240–1.185] | 0.831 [0.581–1.005] |

## scaling

| Scaling | Nodes | Global seq/step | Accum | Step s | Tokens/s/GPU | MFU | Scaling eff. | Compute s | Exposed comm s | Exposed comm share | All-reduce alone s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| weak | 1 | 16 | 8 | 2.04 [2.03–2.05] | 8019 [8012–8062] | 0.155 [0.155–0.156] | 1.00 [1.00–1.01] | 2.03 [2.02–2.03] | 0.01 [0.01–0.01] | 0.01 [0.01–0.01] | – |
| strong | 2 | 16 | 4 | 2.68 [2.66–2.69] | 3029 [3007–3052] | 0.058 [0.058–0.059] | 0.38 [0.37–0.38] | 1.03 [1.03–1.03] | 1.65 [1.63–1.66] | 0.62 [0.61–0.62] | 2.02 [2.00–2.05] |
| weak | 2 | 32 | 8 | 3.72 [3.70–3.78] | 4366 [4322–4424] | 0.084 [0.083–0.085] | 0.54 [0.54–0.55] | 2.02 [2.02–2.03] | 1.70 [1.67–1.76] | 0.46 [0.45–0.47] | 2.06 [2.01–2.07] |
| strong | 4 | 16 | 2 | 4.18 [4.14–4.20] | 982 [961–986] | 0.019 [0.019–0.019] | 0.12 [0.12–0.12] | 0.53 [0.52–0.53] | 3.65 [3.62–3.68] | 0.87 [0.87–0.87] | 4.22 [4.21–4.30] |
| weak | 4 | 64 | 8 | 5.75 [5.64–5.82] | 2849 [2805–2885] | 0.055 [0.054–0.056] | 0.36 [0.35–0.36] | 2.03 [2.03–2.03] | 3.72 [3.61–3.79] | 0.65 [0.64–0.65] | 4.15 [4.11–4.25] |

## drain

| Sample | Step | Load on | job-0 | job-2 | job-1 | job-3 | Alerts (local) | Alerts (sync) |
|---|---|---|---|---|---|---|---|---|
| 1 | 4 |  | 0.263 / 5.03 | 0.270 / 4.94 | 0.270 / 4.88 | 0.267 / 4.88 | – | – |
| 2 | 8 |  | 0.243 / 4.68 | 0.252 / 4.68 | 0.251 / 4.68 | 0.248 / 4.68 | – | – |
| 3 | 12 |  | 0.244 / 4.72 | 0.252 / 4.72 | 0.251 / 4.72 | 0.248 / 4.72 | – | – |
| 4 | 16 |  | 0.244 / 4.61 | 0.251 / 4.61 | 0.249 / 4.61 | 0.248 / 4.61 | – | – |
| 5 | 20 | yes | 0.244 / 5.80 | 0.251 / 5.80 | 0.249 / 5.80 | 0.532 / 5.82 **>1.2×** | – | – |
| 6 | 24 | yes | 0.243 / 6.09 | 0.251 / 6.09 | 0.249 / 6.09 | 0.630 / 6.09 **>1.2×** | – | – |
| 7 | 28 | yes | 0.243 / 6.11 | 0.251 / 6.11 | 0.249 / 6.11 | 0.628 / 6.11 **>1.2×** | job-3 | – |
| 8 | 32 | yes | 0.244 / 6.06 | 0.251 / 6.06 | 0.249 / 6.06 | 0.631 / 6.06 **>1.2×** | – | – |
| 9 | 36 | yes | 0.244 / 6.19 | 0.252 / 6.19 | 0.249 / 6.19 | 0.631 / 6.19 **>1.2×** | – | – |
| 10 | 40 | yes | 0.244 / 6.12 | 0.252 / 6.12 | 0.249 / 6.12 | 0.632 / 6.12 **>1.2×** | – | – |
| 11 | 44 |  | 0.244 / 4.76 | 0.251 / 4.76 | 0.249 / 4.76 | 0.274 / 4.74 | – | – |
| 12 | 48 |  | 0.244 / 4.74 | 0.251 / 4.74 | 0.249 / 4.74 | 0.250 / 4.74 | – | – |
| 13 | 52 |  | 0.244 / 4.70 | 0.252 / 4.70 | 0.250 / 4.70 | 0.249 / 4.70 | – | – |
| 14 | 56 |  | 0.244 / 4.72 | 0.251 / 4.72 | 0.250 / 4.72 | 0.249 / 4.72 | – | – |

## fault

| Round | Checkpoint store | Goodput | Useful s | Wall s | Lost steps (s) | Save s | Restore s | Failure → restore start s |
|---|---|---|---|---|---|---|---|---|
| 1 | disk | 0.430 | 141.0 | 328.0 | 5 (23.1) | 43.6 | 9.2 | 11.4 |
| 1 | shm | 0.673 | 141.5 | 210.2 | 5 (23.4) | 3.3 | 11.4 | 11.4 |
| 2 | disk | 0.445 | 144.9 | 325.3 | 5 (23.9) | 41.0 | 10.1 | 12.0 |
| 3 | disk | 0.422 | 142.8 | 338.0 | 5 (23.5) | 47.6 | 10.9 | 11.1 |

failed: []

