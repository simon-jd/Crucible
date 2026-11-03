# 扩展实验：离线准备与授权后的训练接线

**未在真实多机上验证**。0 美元资源预算暂停 Task4 的真实多机训练、GPU、nsys 抓取和调优。`run.sh` 只生成计划、汇总已有结果和读取已有 SQLite，不执行计划中的命令。没有创建云资源、模型下载或环境安装。本机 stable Torch2.8.0+cu128 不代表下述候选栈兼容。

```bash
bash scaling/run.sh plan > /tmp/crucible-scaling-plan.json
.venv/bin/python -m pytest scaling/tests -q
bash scaling/run.sh analyze /path/to/real-node.sqlite
bash scaling/run.sh summarize /path/to/measured-records.json
```

最后两个命令需要真实输入；项目没有附带真实性能记录、trace 或截图。单元测试内的 SQLite 和吞吐数字仅验证算法。

## 固定接口与候选环境

两份配置统一对接官方 [TorchTitan v0.2.0](https://github.com/pytorch/torchtitan/releases/tag/v0.2.0)，提交 `29624e37fa6dd319fd671b009338651b5c665da2`。发布依赖是 Torch `2.10.0.dev20251019+cu126`、torchao `0.15.0.dev20251015+cu126`，均未在本项目安装或认证。小份官方源码仅下载到忽略的 `.cache/torchtitan-v02-task4/`，用于 CPU schema 核验。

- [JobConfig](https://github.com/pytorch/torchtitan/blob/29624e37fa6dd319fd671b009338651b5c665da2/torchtitan/config/job_config.py) 和 [CLI parser](https://github.com/pytorch/torchtitan/blob/29624e37fa6dd319fd671b009338651b5c665da2/torchtitan/config/manager.py)：`checkpoint.enable`、`compile.enable`、`quantize.linear.float8`、`model.converters=["float8"]` 是这一版本的接口。
- [Llama recipe](https://github.com/pytorch/torchtitan/blob/29624e37fa6dd319fd671b009338651b5c665da2/torchtitan/models/llama3/__init__.py)：`llama3/8B`、FSDP2、TP=2。8卡/节点，常规连续 rank 映射才能让 TP 组留在节点内，现场须验证 mesh/rank/GPU 拓扑。
- [DeepSeek recipe](https://github.com/pytorch/torchtitan/blob/29624e37fa6dd319fd671b009338651b5c665da2/torchtitan/models/deepseek_v3/__init__.py)：正式 `deepseek_v3/debugmodel`，6层、dim256、16头、8个 routed experts、top-k=3、2个 shared experts。TP=1、EP=1/2/4，在固定8卡的 FSDP shard 维度内分组；EP不额外乘进总卡数或全局 token 数。
- [EP dispatch](https://github.com/pytorch/torchtitan/blob/29624e37fa6dd319fd671b009338651b5c665da2/torchtitan/distributed/expert_parallel.py)：`all_to_all_single` / `all_to_all_single_autograd` 实现路由与回传。配置使用正式 `parallelism.expert_parallel_degree`，不沿用 v0.1.0 实验 custom_args。

Task1 的 H100 镜像/驱动/CUDA/NCCL/OFED 锁也是未认证候选，其 CUDA12.8 镜像不等于此 cu126 nightly 环境。尚缺实际镜像 digest、node image、硬件观察、tokenizer/数据/留出集 revision 与 SHA256、精度峰值和质量阈值；计划将这些字段保留 null。随机初始化模型由固定源码 recipe/配置/seed定义，资产 revision 仍必须固定。不能将该计划标为可直接现场验收。

## 计划与训练边界

计划包含11组比较，每组严格按 `baseline,candidate` 交替3轮，共66次运行：

| 比较 | 固定条件 | 唯一候选变化 |
| --- | --- | --- |
| 8/16/32卡的精度比较 | 模型、数据、全局batch256、seq8192、seed42 | BF16 → torchao FP8 |
| 每规模、每精度的调优前后 | 同上及同精度 | AllReduce 默认自动算法 → `NCCL_ALGO=allreduce:Tree`，其它collective保持默认 |
| 小MoE，固定8卡 | 全局batch256、seq1024、seed42、BF16 | EP1 → EP2 或 EP4 |

调优候选仅把 AllReduce 限定为 Tree；FSDP2 所需 AllGather/ReduceScatter 保留默认算法，不能用全局 `NCCL_ALGO=Tree`。按 [NCCL 官方环境变量文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-algo)，collective前缀语法从2.24开始支持，现场必须核验实际 NCCL≥2.24；该版本起未保留可用算法会失败，不能依赖隐式 Ring fallback。[NCCL 2.27.7支持矩阵](https://github.com/NVIDIA/nccl/blob/v2.27.7-1/src/graph/tuning.cc#L231-L244)也表明Tree不用于AllGather/ReduceScatter。这个限定AllReduce的性能假设可能变慢或没有影响，不声称最优。所有 baseline 必须使用干净的 NCCL 环境，不能继承隐藏的 ALGO/PROTO 变量。跨规模采用固定全局batch的强扩展口径，local batch=1，梯度累积随数据并行度调整。200步中前50步暖机，后三个50步窗口形成每次运行的中位数；再把每次运行值送三轮门禁。保留窗口原始记录，不把窗口当作独立实验轮次。

`train.py` 只覆盖 [上游 Trainer.train_step](https://github.com/pytorch/torchtitan/blob/29624e37fa6dd319fd671b009338651b5c665da2/torchtitan/train.py)，其余训练循环由上游运行。每步前后同步CUDA，按已完成 wall time 累计50步；包含 train_step 内的数据读取、梯度累积、优化器与日志，不含外层GC/验证/checkpoint。该同步会增加开销，所有比较必须使用同一接线，结果标记 `synchronized_train_step`，不冒称端到端生产吞吐。

每节点 local rank0 用 Task3 `StepTimeHook` 写自己的 `.prom`，默认50步；同一 run_id 只在同一可比运行的节点间共享。每个50步窗口全部 rank 对累计 wall time 做 MAX，然后 rank0 写 `throughput.jsonl`，全局 token 数为 `global_batch_size * seq_len * 50`，不重复计入 TP/EP rank。少于3个 fresh 健康节点不能判定多机 step-time outlier；8卡单节点的 hook 测试不构成多机异常验证。

授权后的操作者必须先完成计划的所有 null 前置项、健康门禁、实验清单和停机条件，再在精确源码 checkout 的候选环境运行。`TORCHTITAN_SOURCE` 指向干净的 pinned checkout；安装路径必须落在该 checkout，入口检查源码 SHA 与 Torch/torchao 版本。Crucible 仓库须在 PYTHONPATH，TorchTitan 相对 tokenizer 路径须映射到已核验资产。计划中 `REQUIRED_HOST:PORT` 必须由所有节点一致替换。每轮先创建输出目录，按 node_rank 将 env+argv交给已有且已授权的集群启动机制；本项目不提供或执行 SSH/Slurm/K8s 启动器。只有单独用户授权后才能设置 `CRUCIBLE_MULTINODE_APPROVED=1`，这个环境变量是防误启动开关，不替代预算授权。

停机：每轮≤4小时、授权/预算到期即停；XID/NIC/健康失败、非有限loss、磁盘剩余<50GB或artifact配额超限即停止；同负载3次 fresh step time >中位数1.2倍须调查。现场必须有操作者限时器/存储告警，JSON计划不实施这些控制。本机GPU若未来获准单独 smoke，仍须工具批准并包 `flock /var/tmp/rtx4090.lock`，本任务没有运行GPU。

## Nsight Systems 与诊断

命令按 [NVIDIA CLI](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) 输出每节点独立 nsys-rep，`NCCL_DEBUG=INFO` 与 INIT/COLL 原始日志使用 `%h.%p` 分文件。`CRUCIBLE_PROFILE=1` 在已完成暖机后第51步启动 CUDA profiler API、第55步同步完成后停止；nsys 用 `--capture-range=cudaProfilerApi --capture-range-end=stop`，不终止后续训练。现场需先核对 nsys 版本的多进程 capture 行为和 trace 完整性，再使用 export_argv 导出 SQLite。

分析器只支持 `CUPTI_ACTIVITY_KIND_KERNEL(deviceId,start,end,demangledName)` 与 `StringIds(id,value)` 的导出 schema；其它版本明确失败，不猜测。按同节点设备计算 NCCL duration总和、区间并集、与非NCCL kernel并集的交集及无kernel区间；跨设备/节点不会相加为时间。NCCL kernel 名称识别是启发式，EP all-to-all 可能表现为 SendRecv，不显示 alltoall 名称不代表没有EP通信。

`nccl_without_compute_ns` 仅为可能暴露的通信区间，不能证明关键路径或因果瓶颈。需要把真实时间线、NVTX/API关联和NCCL日志合读：定位collective等待与依赖、确认bucket影响、比较计算通信重叠、核对NCCL算法/协议和NUMA/CPU亲和性。一次只调一个因素，用同条件三轮交替检验；CPU采样关闭时，不能只凭GPU trace推断CPU亲和性原因。调优前后截图应展示同样的稳态窗口、rank/设备/时间刻度，并链接原始 trace SHA256；当前均待采集。

## 指标与精度门禁

`analyze_nsys.mfu(tokens_per_s, params, seq_len, n_gpus, peak_flops)` 保留五参数。`params=ModelStructure(N,L,H,D)` 强制显式结构：每token激活的非embedding参数量N、层数L、query头数H、头维度D；不支持用一个参数总量静默猜模型。公式为 `tokens_per_s*(6*N+12*L*H*D*seq_len)/(n_gpus*peak_flops)`，不按causal三角折半，不钳制>1的结果。tokens_per_s是全局有效token/s，peak_flops为实测精度对应的每卡理论FLOP/s（明确是否稀疏峰值，不能混用）；返回比例。省略通信、内存和重计算成本。

与 [上游 FLOPs helper](https://github.com/pytorch/torchtitan/blob/29624e37fa6dd319fd671b009338651b5c665da2/torchtitan/models/utils.py) 一致，Llama8B用L=32、H=32、D=128，N必须按实际词表和实例化参数确认。MoE的N应为 `dense_non_embedding + routers + shared_experts + routed_expert_total * top_k/num_experts`，不能传总参数量。上游DeepSeek helper用 `D=dim/n_heads=16` 作为同公式的代理头维度；它不等于MLA的q/k、v真实维度，不是精确MLA FLOPs计数。MoE结果应明确标为此近似并保存激活参数分解，不能把未实例化模型的N虚填为测量值。

`scaling_metrics` 计算 tokens/s/GPU、MFU、同精度8卡基准的强扩展效率 `T_n/T_8 * 8/n`；调用者负责硬件/负载/精度相同。每种精度的理论峰值分别填入，不用BF16分母为FP8制造收益。对每次运行的窗口中位数，再用 `run.sh summarize` 复用 `scripts.experiment_results.summarize` 检查至少三轮严格交替、有效性、相同conditions与指标集合，输出每变体中位数和min/max。

结果record须有 round、variant、valid、fp8、conditions、metrics；metrics至少同时含 `tokens_per_s_per_gpu`、`mfu`、`scaling_efficiency`。有FP8比较时所有record还需同名 `accuracy_delta`，规定为同留出集、同训练步数的 candidate减baseline，指标名称/方向/样本数/seed和预先批准的最大容差放conditions并提供原始评估日志。训练配置关闭验证以隔离计时；留出集评估须在计时外单独完成，缺评估的FP8记录不得变为valid。当前没有已实现/已运行的任务特定精度评估器或质量通过结论。共享门禁检验格式与精度差存在，不证明数据来源真实，也不替操作者选择容差。
