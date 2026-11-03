# Task4 扩展实验与诊断证据

**未在真实多机上验证**。2026-10-23，本地完成配置、离线命令生成、训练同步计时/Task3 hook接线、MFU、SQLite分析和CPU回归；原始Task4的现场验收仍暂停。本项目预算0美元，未执行SSH、Slurm、K8s、云、GPU、多机训练或故障操作，没有安装TorchTitan/nightly环境。

| 验收项 | 当前证据 | 状态 |
| --- | --- | --- |
| Llama3 8B FSDP2 + 节点内TP2 | 固定v0.2.0配置与schema核验 | 本地准备通过，硬件待验证 |
| BF16/FP8，8/16/32卡 | 66次运行中的三规模精度/调优矩阵 | 待授权执行 |
| 小MoE EP1 vs EP2/4 all-to-all | 正式debugmodel配置、上游EP dispatch接口核验 | 接线完成，通信性能待验证 |
| tokens/s/GPU、MFU、扩展效率 | 手算MFU与纯函数CPU测试；真实数据为空 | 待测 |
| FP8留出集精度差 | 缺accuracy_delta拒绝汇总的门禁测试 | 真实评估/容差待落实 |
| nsys调优前后截图 | 无真实trace，不提供模拟截图 | 待采集 |
| 扩展损失与调优收益 | 区间并集/交集分析测试；AllReduce-only Tree为候选变量 | 无真实归因/收益结论 |

软件基线是 [TorchTitan v0.2.0](https://github.com/pytorch/torchtitan/releases/tag/v0.2.0)，官方tag解析提交为 `29624e37fa6dd319fd671b009338651b5c665da2`，发布依赖Torch `2.10.0.dev20251019+cu126`、torchao `0.15.0.dev20251015+cu126`。相较最初v0.1.0线索，该版本提供正式DeepSeek debugmodel/EP接口。[源码schema证据](scaling-local/source-schema.json)记录实际下载的官方文件SHA256和对真实JobConfig dataclass逐项核验的TOML键；这不是tyro CLI、训练或运行时兼容认证。本机stableTorch2.8.0+cu128与Task1候选H100镜像均未认证兼容此nightly栈。

[复现说明](../../scaling/README.md)列出固定官方配置/CLI、Trainer、DeepSeek recipe、EP dispatch、FLOPs helper及NVIDIA nsys来源。两配置固定seed42、global batch256；Llama seq8192/TP2，MoE seq1024/TP1/EP2；FSDP2与完整reshard，训练200步暖机50步。计划每组三轮baseline/candidate交替，按每次运行窗口中位数，再报三轮中位数和范围。真实镜像digest、驱动/CUDA/NCCL/OFED观察、资产/数据/留出集revision、精度峰值与质量容差均保留pending，不能从现有Task1候选锁补成“已观察”。

MFU使用全局有效tokens/s，TP/EP重复rank不计数，`6N+12LHDS`不按causal三角折半，分母为n_gpus乘该精度每卡理论FLOP/s；N为激活的非embedding参数，MoE包括dense/router/shared和top-k激活expert份额。MoE MLA使用上游helper的代理头维度时必须标明近似。强扩展效率对同精度8卡参考计算，不能混用BF16/FP8分母或参考。没有实测值填入本文。

SQLite分析只读：按设备区间并集区分NCCL总duration、并发重叠和无计算覆盖时段。累计duration不等于关键路径；通信暴露、bucket、重叠、算法/协议和CPU亲和性都需真实时间线、原始NCCL INFO日志与单变量交替重跑佐证。MoE SendRecv核名不能可靠恢复完整all-to-all。当前无真实trace，也没有用测试SQLite生成“调优证据”。

初版本地门禁：`.venv/bin/python -m pytest scaling/tests -q` 为 **23 passed**；MFU手算验证18288 FLOP/token与0.09144；SQLite并发区间、异常输入、强扩展计算、计划覆盖、拒绝launch、薄Trainer透传、真实StepTimeHook第50/100步写入、暖机排除、失败步骤不发布、51–55步捕获窗口和FP8精度差门禁均通过。CPU函数测试中使用的小数字与fixture仅为测试输入，不能解释成真实GPU吞吐或nsys观测。最终提交前全量CPU测试 **311 passed in 55.15s**（只跑一次）；结果见 [本地测试日志](scaling-local/cpu-tests.txt)。

Fixround1：调优计划改用 `NCCL_ALGO=allreduce:Tree`，只改变AllReduce算法选择，FSDP2 AllGather/ReduceScatter及其它collective保留默认选择，precision/EP比较不覆盖算法或协议。执行前须核验实际NCCL≥2.24，计划显式记录这个语法最低版本。[NCCL官方文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-algo)说明该版本起支持collective前缀且无可用算法会报错；[固定2.27.7源码支持矩阵](https://github.com/NVIDIA/nccl/blob/v2.27.7-1/src/graph/tuning.cc#L231-L244)确认Tree不适用于AllGather/ReduceScatter。此前全局Tree是生成配置错误，已修复，不能归为待硬件验证的性能假设。增加离线回归覆盖全部规模/精度/轮次/节点的限定语义及未调优实验保持默认；本轮聚焦验证见 [fixround1日志](scaling-local/fixround1-tests.txt)。原311项全量CPU记录保留，本轮按范围仅跑scaling聚焦检查。**未在真实多机上验证**。
