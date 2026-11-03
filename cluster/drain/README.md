# 健康自动化与 drain（本地实现）

**未在真实多机上验证**。本任务未启动 Prometheus/DCGM/node_exporter、训练、Slurm、K8s、SSH、云或 GPU；下述部署配置是待授权现场接线模板，测试中的 scheduler runner 与 Prometheus 数据均为模拟。Task4 尚无训练脚本接入，本目录提供可直接导入的 hook。

## 训练与 Prometheus 接线

训练方在每节点一个 reporting rank 上调用以下 hook。`run_id` 必须唯一标识一次可比实验：模型 revision、精度、batch/sequence、并行配置、硬件类别和训练阶段相同，不能把不同负载混算；恢复训练应使用新的 run_id 或仅上报尚未发布的 step。hook 不启动服务、不导入 Torch、不占用 GPU。

```python
from cluster.drain.metrics import StepTimeHook

hook = StepTimeHook(
    node="lab-a", run_id="experiment-config-round-1", path="/tmp/crucible-training/step.prom"
)
# 每完成一个训练 step：累计当前 50-step window 的真实 wall time。
# 含异步 CUDA 的训练需由训练方使用已完成/同步后的 wall time。
# completed_steps 是全局单调的已完成步数，mean_seconds 是本 window 的均值。
hook.observe(completed_steps, mean_seconds)
```

默认每 **50** 步原子更新一次 Prometheus textfile；输出 `crucible_step_time_seconds`、`crucible_step_sample_timestamp_seconds` 和 `crucible_step_index`，标签为 `node`、`run_id`。重复/倒序 step 不写，写失败不推进 hook 状态。每个 hook 独占一个 `.prom` 文件；一次实验结束应由训练方移除自己的文件，避免残留。可用 `every=` 调整，计划默认保持50。

标签使用UTF-8文本，中文等Unicode字符保留原文；仅按Prometheus协议将反斜杠、双引号、LF转义为 `\\`、`\"`、`\n`。空值、非字符串、其它C0/C1控制字符（含TAB/CR/NUL/DEL）及surrogate在构造hook时拒绝，不写文件；不会输出JSON的 `\uXXXX` 转义。

`prometheus.yaml` 通过三个 file_sd JSON 接入 DCGM exporter、node_exporter 和训练 textfile collector，默认 scrape/evaluation 为15秒。现场应先创建文件（示例地址仅为占位，不在本任务调用）：

```json
[{"targets": ["lab-a:9400"], "labels": {"node": "lab-a"}}]
```

上例保存为 `dcgm-targets.json`；`node-targets.json` 对应 node_exporter 的 `lab-a:9100`，`training-targets.json` 对应暴露专用 textfile directory 的 node_exporter endpoint（可为同一 endpoint，训练 scrape 只保留训练指标，node scrape 只保留 carrier，避免重复）。node_exporter 需由操作者配置 `--collector.textfile.directory=/tmp/crucible-training`；此处不安装/启动服务。所有节点须有明确的 `node` 标签，GPU 用 DCGM 的 `UUID`，网卡用 `device`。DCGM `DCGM_FI_DEV_XID_ERRORS` 是最后 XID **gauge**，不能当作 counter 用 `increase()`；持续非零的旧 XID 不会自动恢复。

```python
from cluster.drain.controller import Controller, SchedulerExecutor, atomic_write
from cluster.drain.metrics import PrometheusSource
import json

source = PrometheusSource({"lab-a": {"gpus": ["GPU-site-uuid"], "links": ["ib0"]}})
controller = Controller(state_path="/tmp/crucible-drain/controller-state.json")
executor = SchedulerExecutor("slurm", {"lab-a"}, owner="lab-owner")  # 默认 dry-run
# 仅在已授权的现场 poll loop 中调用（本任务不运行 collect 的真实 HTTP）：
actions = controller.evaluate(source.collect())
results = executor.execute(actions, jobs=[{"id": "42"}])
atomic_write("/tmp/crucible-drain/last-execution.json", json.dumps(results))
```

`collect()` 的 HTTP 超时为每次10秒，所有查询使用同一 evaluation time；从 `timestamp(metric)` 读取真实 scrape 时间，不能把 instant API 的 evaluation timestamp 当作新样本。GPU UUID/NIC 必须齐全且唯一，否则该节点健康未知。重复训练 scrape 使用 training sample timestamp 去重；timestamp 超前或缺失拒绝。Prometheus HTTP/格式失败抛异常，不产生 action，poll loop 必须保存失败并通知操作者。`prometheus-rules.yaml` 提供硬件/训练陈旧告警，3个训练采样判断由 Python 控制器负责，不能用时间长度 `for:` 代替。

## 判定与状态

`rules.yaml` 可配置默认值：freshness **120秒**、step ratio **1.2**、连续 **3** 次、drain cooldown **600秒**、最少比较集合大小 **3**（含自身），以及 hook 默认报告间隔 **50**。运行期比例与 `cluster/health/thresholds.yaml` 的静态带宽效率门槛完全独立。训练报告间隔耗时超过120秒时必须结合实际实验调整 freshness，并同步陈旧告警表达式；不应把陈旧指标当正常值。

比较集合为同一个 run_id、fresh、硬件健康且 step time 有限正数的节点，含目标节点自身。集合至少3个；单节点/双节点/缺失指标是 **indeterminate** 告警，既不推进连续次数，也不清除已有异常。默认三次均严格 `> median × 1.2` 才告警；相等不超阈值。相同或更早 sample timestamp 不重复计数；告警持续时只发一次，新的可比较正常样本恢复后重新计数。不同 run_id 有独立实验语义并重置该节点 step 连续状态。硬件-only 直接输入可以省略 step 字段；Prometheus source 则要求训练报告可判定，缺失时发 indeterminate。

XID非零或任一受控 NIC carrier=0，产生按序 `drain`、`requeue` action。任一硬件输入缺失/陈旧/非法则只告警。持续事件不重复 drain；只在新的硬件健康样本后解除 incident latch，恢复后复发仍要距离上次 drain 至少600秒。未知/旧样本不算恢复，不自动 resume、uncordon 或移除 taint。

使用 `state_path` 时原子写（文件及父目录 fsync），重启保留采样水位、连续计数、告警/事件 latch 和 epoch cooldown。坏 schema/状态启动失败；写失败抛异常，**不返回 action、不更新内存**。不配置 state_path 为进程内模式，重启会丢失冷却状态，不用于自动现场操作。一个 state file 只能由一个 controller process 使用，需要时钟同步；时钟回退会保守延长冷却。

状态记录的是 **action intent**，不提供跨 scheduler 的事务或 exactly-once：在持久化后、命令执行/审计落盘前崩溃可能留下未执行或部分完成的 intent。必须由操作者根据 action、scheduler 及 command evidence 进行 reconcile；不能删除 state 重放，也不自动重试失败命令。真实自动化交付还需现场 operator reconciliation/告警与部署权限配置。

## 执行器的归属边界

默认 `dry_run=True` 完全不调用 runner，结果为 `planned`（不是 drained/requeued）；`managed_nodes` 只允许显式受控节点。未做自动作业发现，不会全节点驱逐，不会影响未列入 `jobs` 的工作负载。现场 adapter 需另行明确授权；本任务测试只注入 runner，不执行默认 subprocess。默认单条命令超时30秒，argv调用无 shell。

Slurm 使用 `scontrol update nodename=<n> state=drain reason=<r>`。仅允许明确 scalar job ID，`scontrol show job <id> --json` 核对 ID、`comment == crucible:<owner>`、RUNNING 状态，并通过 `scontrol show hostnames <nodes>` 核对节点；然后 `scontrol requeue <id>`，再次读取为 PENDING 才记 requeued。提交作业时操作者应设 `--comment=crucible:<owner>` 并自行校验 UID/权限。当前 JSON adapter 采用 `jobs[].job_id/comment/nodes/job_state`（state为字符串列表）的 Slurm data_parser 合约，部署前需核对具体版本；数组/异构作业、其它JSON schema不支持，保持失败。

K8s 使用 cordon + `crucible/health=failed:NoSchedule` taint（非 NoExecute，不驱逐其它Pod）。`jobs` 逐项为 `{"namespace": "lab", "job": "train", "pod": "train-1"}`；读取 Job 要求 `crucible/owner` label匹配、未完成/失败/暂停，Pod 必须由相同 Job UID 控制且在故障节点。只向该 Pod eviction endpoint 发送 `policy/v1` Eviction，带 `deleteOptions.preconditions.uid`，保留 PDB 的拒绝语义。取驱逐前 UID 集合后，只有观察到同 Job UID 控制的新 Pending/Running Pod 且不在故障节点才记 requeued；Pending代表已重新入队，**不代表已运行**。只做一次有界检查，异步替代未及时出现即 failed，需现场 reconcile，不宣称成功。该适配器仅支持普通 Job 控制Pod，不支持 Kueue/MPIJob/分布式gang重启，未声称已集成。

命令失败、ownership不符、UID不符、部分cordon/taint成功、PDB拒绝或replacement未确认均保留 `failed` 及原始命令证据。必须有同一次执行内成功 drain 才尝试 requeue；每个job有单独结果，部分失败保持总体failed，不自动恢复节点。

执行器按**整批隔离 → 重排**两阶段执行，返回结果仍按输入action顺序排列。所有 `drain` 和 `requeue` action涉及的故障节点都必须在本次执行中隔离成功；任何节点drain/cordon/taint失败或缺少对应drain，都令本批所有requeue action标记 `failed`、作业标记 `not_executed`，不发出任何requeue/eviction，也不撤销已经成功的隔离。隔离失败后需操作者根据原始命令证据reconcile。

隔离全部成功后，逐个核对显式作业的真实归属，结果通过 `jobs[].nodes` 关联实际受影响节点。Slurm按scalar job ID去重，跨多个故障节点的同一作业只执行一次requeue，结果关联每个受影响节点；K8s按(namespace,job,pod)去重，同一Job的不同明确Pod分别保留UID/owner/node核验和eviction。节点没有关联作业时标记 `no_affected_jobs`，不会被误报为wrong-node失败；显式引用实际不在本批受影响节点则是失败。metadata/ownership核验失败而无法确定可信节点的结果保留 `nodes=[]`，作为批次可见错误附在所有requeue结果中，不静默忽略。K8s的替代Pod必须不在本批任何已知故障节点（包括仅有drain action的节点）才确认重新入队。

## 官方接口依据与本地验证

- [NVIDIA DCGM exporter](https://docs.nvidia.com/datacenter/dcgm/latest/gpu-telemetry/dcgm-exporter.html)、[默认XID gauge定义](https://github.com/NVIDIA/dcgm-exporter/blob/main/etc/default-counters.csv)
- [Prometheus instant API](https://prometheus.io/docs/prometheus/latest/querying/api/)
- [Prometheus文本标签转义协议](https://prometheus.io/docs/instrumenting/exposition_formats/)
- [Slurm scontrol（JSON、drain、requeue）](https://slurm.schedmd.com/scontrol.html)
- [K8s API Eviction](https://kubernetes.io/docs/concepts/scheduling-eviction/api-eviction/)、[Job替代Pod语义](https://kubernetes.io/docs/concepts/workloads/controllers/job/)

运行 `.venv/bin/python -m pytest cluster/drain/tests/test_controller.py -q`。证据见 `docs/evidence/drain-local/` 与 `docs/evidence/drain.md`。这些是软件模拟回归，不能当作真实集群健康/性能证据。
