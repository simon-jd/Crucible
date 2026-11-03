# 生产learner与rollout的匿名权重交接

这是原Task11单卡W2流水线的接入工作：首次CPU运行记录20项通过后被临时目录门禁停止，修正测试夹具后的另名v2完成其余46项。独立初审发现全陈旧批次丢弃数漏记，首轮修复`9989494`的三项定点检查已通过，同agent独立复审已确认I1 ADDRESSED、规格Compliant/质量Approved，仅关闭本CPU前置阶段。已有[串行训练状态CPU阶段](rl-serial-state-handoff.md)及其修复/复审保持原结论，不能替代本阶段或实际完整模型验收。

现有磁盘VersionStore完整导出一份Qwen2.5-0.5B BF16推理权重需至少988,065,536字节，超出当前项目可用份额；该数来自[静态状态估算](raw/task11-ram-state-static-context.json)，不是本轮测量。新接入采用独立RAM权重receipt和私有phase owner：在原learner退出/回收、训练状态被接受后，由CPU进程逐storage验证和转换政策权重，封存匿名FD；保留完整FP32策略、Adam状态和BF16 reference的训练状态作为下一步恢复依据。原磁盘同步接口及数值证据保留。

持有训练状态时，只能再持有一份当前或pending推理权重；失败和身份不明的预留也计入。旧rollout须退出并回收，父端核对FD/映射和最后句柄后退休旧权重，才能继续learner。新loader拟使用固定vLLM0.10.2的自定义BaseModelLoader在profiling前直接加载FD内容，并在实际packed参数、buffer、tied head及全部有效坐标核对后才ACK版本；实际vLLM/GPU分支仍未验证。CPU注入接收器仅验证共享转换/核对逻辑和真实进程/FD生命周期，不能证明该运行时分支工作。

所有匿名pagecache、RSS、存储暂存、元数据、未来权重和其他副本计入16GB，GPU另核对24GB及其他占用；不把FD关闭当作任意进程映射已消失。CPU exporter不初始化或探测CUDA硬件，CUDA RNG只能核对受信生产者来源及有界结构/字节；实际CUDA恢复语义仍是后续门禁。已知空批次或全陈旧批次应在破坏性交接前处理为no-op，不制造假策略版本；意外no-op导致的已消费间隙必须失败封闭。

本阶段源码、raw、报告、审查及root元数据留存上限2MB，CPU临时树1MB，均在原690MB未来预留内；本项目14GB/$0、共享GPU锁、不修改共享模型缓存和不公开发布限制不变。[开始资源记录30](raw/resources-start-resume-30.json)、[冻结/协调证据核对](raw/task11-bridge-root-preservation-before-source.json)及[源码准入裁决](raw/task11-bridge-root-source-authorization.json)已保存。实现来源为子agent `/root/task11_phase_bridge`，独立审查来源为子agent `/root/review_task11_bridge`，均不是人工标注或审批。

完整0.5B实际内存、CUDA/FSDP、真实训练/rollout时间分片、数据获取与留出质量、等预算外部基线和全部消融仍开放。原full02已实际失败、授权已使用；用户随后对冻结full03 v2另授精确单次权，见授权原始记录。本CPU阶段不消耗该权，也不改变发送门禁；原Goal保持active，整个项目未完成。

首次[CPU原始终态](raw/task11-ram-handoff-bridge.json)退出125，20项通过后因Pytest current别名触发严格scratch符号链接门禁；该次余46项没有完成结果。root保留[完整路径与生成文件内容](raw/task11-bridge-root-first-failure-inventory.json)，再次核对后完成[精确自有scratch清理](raw/task11-bridge-root-first-failure-scratch-cleanup.json)。修正仅涉及两份新测试的无符号链接临时目录夹具及另名v2选择，原监控不放宽。

[v2日志](raw/task11-ram-handoff-bridge-v2.log)记录46项通过、39.09秒，包括两个真实CPU训练/导出/rollout/消费循环及AdamW完整状态连续性、BF16 reference、packed权重与实际token/logprob返回。原20项未重复；这不是原66项整次运行成功。旧RPC清理控制产生一条Python3.12多线程fork弃用警告，原样保留并提交独立审查。不能将这些tiny fixture结果当作真实vLLM/GPU、完整0.5B内存或训练质量验收。

[v2终态](raw/task11-ram-handoff-bridge-v2.json)退出0，errors为空、cleanup已确认、无剩余自有子进程；采样RSS加8MiB预留最高1,502,236,672字节，不能推断连续峰值或硬配额。root实际核对25个执行源散列及51条临时路径的完整身份、4910字节内容后，只精确删除本次12个临时文件和39个空目录；[删除前证明及完整内容](raw/task11-bridge-root-v2-result-scratch-cleanup.json)留存。原失败、两次运行和全部协调证据均保留，本CPU前置阶段最终审查见末段。

独立初审为规格/质量Needs fixes：Important I1指出全陈旧批次在父端提前返回后绕过了learner的dropped_stale计数。首轮修复保留no-op、原状态和版本，须补充可由调度器读取的计数及空批次、边界、混合批次、重试和非法头部覆盖；随后三项定点GREEN及复审已完成。Minor M2为历史RPC fork警告，和原RAM交接的RNG Minor M1一同递延到最终整体审查，原始日志不改。

首轮修复保留全部准备失败：root给Ruff设置的8KiB进程级文件限额导致三份自有文件截断；[24576字节原样副本与恢复记录](raw/task11-bridge-root-fix1-format-failure-recovery.json)留存，恢复后原准备SHA/AST精确一致，冻结生产及394/100身份未变。随后CPU尝试47411在初始memfd写入触发EFBIG，未到达I1断言，不能称作有效RED；[原日志](raw/task11-ram-handoff-bridge-fix1-red.log)和[终态/删除前证明](raw/task11-bridge-root-fix1-resource-failure-cleanup.json)保留。源码审查确认缺口后，root准入窄修复及一次GREEN，未再跑RED或已通过套件。

修复通过`audit.noop_dropped_stale`累计合法全陈旧提交的样本数，先验证全部头部和严格整数目标版本，再一次递增；空批次为零、合法重提再次计数。混合批次只由实际learner的dropped_stale计数，调度器将两者各自增量加一次，不重复相加。[GREEN日志](raw/task11-ram-handoff-bridge-fix1-green.log)记录3项通过/12.47秒、无警告，包括真实AdamW混合批次和不中断参考状态一致、边界/重试/非法输入、no-op无child/消费且封存字节不变。原29321退出0、errors为空、cleanup已确认、无剩余自有子进程，采样RSS含8MiB预留最高1,477,726,208字节。父监控FSIZE软32768/硬131072、固定prlimit子进程131072，快照与父端输出分别受限；[终态](raw/task11-ram-handoff-bridge-fix1-green.json)与[root 25散列及21路径删除前证明](raw/task11-bridge-root-fix1-green-result-cleanup.json)留存。同agent定向复审已确认I1 ADDRESSED、规格Compliant/质量Approved，无新增Critical/Important/Minor。仅关闭此CPU前置阶段，实际vLLM/full-model/GPU与整个Task11/Goal继续开放。
