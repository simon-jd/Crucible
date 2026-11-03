# Task12：分段阻塞等待的 CPU 验证

本阶段仅验证客户端等待候选。整个 Task12、原生单卡基线和项目验收仍未完成；原 full02v3 的三次 Burst 校准失败及协调证据保持原样。

## 改动与语义

客户端复用一个 timer，每段最多阻塞 100ms，每次唤醒后按原始绝对 due 重算剩余时间；重置前再次读取时钟，重置后才记录可选的中间唤醒计数。没有忙等、系统调优、重试求通过或阈值调整。请求 deadline 仍是 due + Timeout，SentMs 仍取完整请求体写入结束时刻，迟到门禁仍为超过 5ms 的请求占比不得超过 1%。

可选侧车升为版本 2：post_select/timer_stopped 保存最后一次实际唤醒及 Stop 完成时刻，或取消时刻；中间唤醒只累计有界计数，不占事件槽。若在 Stop 或后续处理期间跨过 due，最后唤醒时刻可以早于 due，返回发生在重新检查 due 之后。只有从未创建 timer 才使用 no_timer。计数达到 65,535 后继续增长会使观测无效，事件、写入和请求数量上限不变。

100ms 是本次实验候选，没有证明它是最优参数，也没有证明历史故障的内核根因。普通运行与诊断运行共享等待实现；观测本身可能影响时序。

## 聚焦验证

新增六项 Go 测试、15 个叶子用例覆盖分段上限、提前/迟到唤醒、重置前后延迟、取消、无 timer、计数溢出与并发快照、绝对 deadline 和原八参数发送入口。隔离复制源码只把上限改成 1 小时，八个分段断言如预期失败；维护源码没有切换。聚焦测试通过（replay 0.167s、CLI 0.005s），新增测试及实际唤醒/连接复用控制的 race 检查通过（replay 1.117s）。CLI 包在该 race 筛选下没有匹配测试，不据此声称 CLI race 覆盖。

精确 argv、环境、完整用例与输出见 [反例](raw/task12-wait-go-red.log)、[聚焦测试](raw/task12-wait-go-focused.log)、[race](raw/task12-wait-go-race.log)和[构建记录](raw/task12-wait-go-build.json)。固定源码及二进制身份见[单次方案](raw/task12-wait-driver-proposal.json)和[root 准入](raw/task12-wait-root-one-shot-gate.json)。

## 唯一一次 CPU 回环观测

原句柄 48046 退出 0。沿用固定的 23 条 Burst 请求、seed 17、倍率 1、timeout 60s 和 inflight 192；复用既有有界回环 fixture，没有加载模型、调用 GPU 或执行原生基线。

| 指标 | 本次结果 |
|---|---:|
| 完成 / 输出 token 数 | 23 / 1,556 |
| 超过 5ms 的发送迟到 | 0 / 23 |
| 最大发送迟到 | 1.508635ms |
| index1 迟到 / 中间唤醒 | 0.855395ms / 79 |
| index2 迟到 / 中间唤醒 | 0.777151ms / 229 |
| SenderValid / Aborted | true / false |
| 侧车版本 / 有效性 | 2 / true |
| RSS 样本 / 最大采样值 | 585 / 62,181,376 字节 |
| 采样跨度 | 59.001804s |

RSS 是采样最大值，不能视为连续峰值。[原始输出](raw/task12-wait-cpu-01/stdout.json)、[阶段侧车](raw/task12-wait-cpu-01/phases.json)、[终态](raw/task12-wait-cpu-01/terminal.json)及[root 独立复算](raw/task12-wait-root-result-check.json)保留。原始客户端退出 0，诊断有效与原生性能验收分别记录；native_performance_accepted 始终为 false。单次 CPU 通过不能证明长期可靠性、模型服务性能或历史 GPU 运行内部原因，不重复试验来挑选结果。

## 资源、清理与审查

Go 缓存和新二进制使用原 690MB 预留内最多 500MB 的临时借用，提前停止限额仍为 400MB。race 期间最大采样 367,130,967 字节触发 300MB 规划检查点；终态 233,076,045 字节，tmp 为空。检查后只准入一次复用缓存的构建，没有清缓存重试或提高上限，见[检查点](raw/task12-wait-race-checkpoint.json)和[裁定](raw/task12-wait-root-warm-build-gate.json)。

回放子进程自然退出并回收，fixture 线程为零、监听关闭、错误列表为空。root 在[主机权限下](raw/task12-wait-root-host-terminal.json)独立核验实际控制器与子进程均不存在，并成功绑定后关闭 31013；[首次沙盒 PID 核验](raw/task12-wait-root-host-audit-attempt.json)的命名空间范围错误保留并更正，编译进程仍以原监测器回收前的进程组证明为据。

root 独立验证完整 3,719 条路径和 3,453 个文件的身份/SHA 后，仅删除本轮临时树，回收 243,380,070 字节；新二进制随树删除属于原计划。完整[压缩库存](raw/task12-wait-loan-inventory.json.gz)、[清理记录](raw/task12-wait-loan-cleanup.json)和[资源预检](raw/resources-end-resume-26.json)保留。原二进制、共享模型、运行时与所有历史失败证据不变。独立规格与代码质量审查均为 Approved，无 Critical/Important/Minor。审查来源是子 agent `/root/review_task12_native`，报告原样保留。仅关闭本次 CPU 阶段，不关闭整个 Task12。
