# Task3：健康自动化与 drain 本地证据

日期：2026-10-23。**未在真实多机上验证**。本任务只执行 CPU Python 回归与静态检查，没有启动或连接 DCGM、Prometheus、node_exporter、Slurm、K8s、SSH、云、GPU；未调度/驱逐/重排任何实际作业，未产生费用。

实现 `Controller.evaluate(metrics: dict[str, NodeMetrics]) -> list[Action]`：XID 或受控网卡 link-down 产生 drain/requeue intent；运行期 step time 严格超过同 run/config 的 fresh 健康节点中位数 × 1.2，连续3个不同训练采样才告警。默认比较集合至少3个节点（含自身）；样本120秒内有效。持续事件不重复drain，健康恢复后再复发仍有600秒冷却，未知样本不算恢复。配置与静态 burn-in 带宽门槛分开。

提供每50步发布一次的轻量 `StepTimeHook`、Prometheus DCGM/网卡/训练 scrape 模板与硬件/陈旧告警；没有虚构已接入训练脚本。Task4可按 [接入说明](../../cluster/drain/README.md) 导入。真实采样时间来自 `timestamp(raw_metric)` 及 hook 的显式 timestamp gauge，HTTP查询固定同一求值时刻；不是把 instant-vector 的求值时间当源采样。

Slurm/K8s 执行器默认 dry-run；测试通过注入 runner 核对命令及失败状态。Slurm 校验明确job ID、owner comment、节点及RUNNING后requeue，只有PENDING确认才成功。K8s只核对受控Job UID/owner label及明确Pod所属节点，cordon/NoSchedule taint后发带Pod UID precondition的Eviction；观察到新的受控替代Pod后才记重新入队，Pending并不代表已运行。失败/部分成功、原始stdout/stderr保留在执行结果，未确认重排不宣称成功。没有全节点驱逐或自动恢复。

状态原子持久化覆盖重启、采样水位、事件与告警 latch、连续计数和冷却；写失败不返回action/不提交内存。持久化的是 intent，与调度器之间不提供事务：崩溃或部分命令成功须人工根据证据reconcile，不盲目自动重放。部署须明确受控节点、作业、exporter inventory、调度器JSON版本/权限与时钟同步；没有部署daemon或服务。普通K8s Job适配不等同于Kueue/MPIJob/gang重排。

本地 TDD 与检查：

- [初始RED](drain-local/red.txt)：测试先于实现，缺少controller模块导致collection失败；包含四个brief指定测试名。
- [自审RED](drain-local/self-review-red.txt)：最旧健康采样分量时间未变化时，新XID一度被吞掉；增加回归后修复为fault判断不依赖最旧watermark推进，恢复仍要求新健康采样。
- [focused GREEN](drain-local/green-focused.txt)：40 passed，含阈值/去重/恢复/重启/写失败/未知数据、模拟Prometheus HTTP、textfile→source→controller→dry-run链路、两种调度器归属和命令失败回归。
- [静态检查](drain-local/ruff.txt)：`ruff check cluster/drain`，All checks passed。
- [提交前唯一一次全量Python测试](drain-local/green-full.txt) 与 [退出状态](drain-local/green-full-exit.json)。Go工作尚未到Task12，未作为此任务门禁。

模拟数据只证明本地软件契约与失败边界，不是实际集群健康、训练性能或多机验收证据。

最终结果：focused **40 passed in 0.06s**；唯一一次全量 `.venv/bin/python -m pytest -q` 为 **261 passed in 55.49s**，returncode=0；Ruff通过。

## 独立审查修复 round 1

独立审查发现：多节点同批故障按原action交错顺序执行，可能在其它坏节点尚未隔离时重排；全jobs逐node遍历造成wrong-node假失败；JSON字符串转义会对Unicode输出Prometheus文本协议不支持的 `\uXXXX`。

修复后所有故障节点先隔离，全部成功才逐作业核验并重排；任一隔离失败则本批重排全部 `failed`/`not_executed`、保留隔离命令证据且不执行requeue/eviction。结果按核验的实际节点关联，Slurm跨故障节点作业仅requeue一次，K8s同Job不同明确Pod分别核验且每Pod至多eviction一次。无法确定可信节点的metadata/ownership失败作为批次可见失败保留，不会被分区静默吞掉。替代Pod落在任何同批已知故障节点均不确认成功。UTF-8标签保留Unicode原文，使用协议反斜杠/引号/LF转义，其它控制字符和surrogate/非字符串在写文件前拒绝。

新增测试 `cluster/drain/tests/test_drain_review_fixes.py`：[修复前RED](drain-local/fix-round-1/red.txt) 为 **21 failed, 4 passed**；[额外已知坏节点RED](drain-local/fix-round-1/known-fault-red.txt) 为 **1 failed, 1 passed, 25 deselected**。原测试名 `test_review_fixes.py` 与既有health测试同名，首轮全量在collection失败；已改为唯一文件名，原始 [collection失败](drain-local/fix-round-1/full-collection-failed.txt) 和 [退出状态](drain-local/fix-round-1/full-collection-failed-exit.json) 保留。均只用fake runner/CPU，没有真实scheduler/GPU/SSH/云；**未在真实多机上验证**。最终focused、Ruff与全量检查输出在 [fix-round-1证据目录](drain-local/fix-round-1/)。

本轮最终验证：focused **67 passed in 0.07s**；Ruff通过。root批准改名后补跑的 `.venv/bin/python -m pytest -q` 为 **288 passed in 55.02s**，returncode=0，见 [全量GREEN](drain-local/fix-round-1/green-full.txt) 与 [退出状态](drain-local/fix-round-1/green-full-exit.json)。本轮共两次全量入口：首次只发生collection错误，第二次执行完整suite通过，没有重复已成功suite。
