# W2完整训练状态的RAM交接前置条件

本阶段服务于Task11完整实验，不能替代真实异步训练、同步/陈旧度对照、等预算消融、外部基线、留出质量及报告。静态检查已通过，真实CPU状态交接和失败路径已有聚焦验证；首轮修复的独立复审已通过。原始收集与测试失败全部保留，未执行GPU。Task12 full03 v2的单次精确授权仍待，原full02失败及已使用授权不改。

原Task9已实现BF16推理导出，但不包含optimizer；当前14GB项目不能直接增加完整FP32策略和Adam checkpoint。新路径以父进程持有的Linux匿名RAM描述符保留完整状态：原learner写入并封存，原始子进程退出/回收后，新learner恢复、真实更新并将新完整状态写入第二个父进程持有的描述符。调用方须持续证明FD和消费者归属，旧slot保留到消费者退出；不能把关闭一个FD当作全部映射释放。失败learner不可恢复或发布，恢复过程中禁止step和状态访问，错误使新phase永久失效。

[RAM静态上下文](raw/task11-ram-state-static-context.json)只读固定Qwen2.5-0.5B的32,280字节safetensors头，没有读取权重载荷或加载模型。下列都是规划量，不能相加冒称同一时刻实测峰值：

| 项 | 字节 | 依据 |
|---|---:|---|
| FP32策略与两份FP32 Adam moments下限 | 5,928,393,216 | 494,032,768元素；不含元数据/RNG等 |
| 再包含BF16 reference的一份raw状态下限 | 6,916,458,752 | 新静态上下文 |
| 两份完整raw状态下限 | 13,832,917,504 | 尚不含RSS、映射、metadata或复制 |
| 最大FP32 storage候选 | 544,538,624 | embedding形状151936×896 |
| 历史真实learner最高host RSS | 4,128,903,168 | [Task8证据](learner.md)，不是当前峰值 |

真实完整模型能否满足16,000,000,000字节仍未证明。所有匿名载荷、映射与额外副本均在同一内存上限内核算，即使pagecache未体现在RSS也不能视为免费。最多两个checkpoint同样包括RAM描述符、pending输出及已有磁盘checkpoint；不写共享模型缓存，不把RAM状态当作未计费的磁盘替代品。

CPU验证将实际memfd载荷限于131,072字节，源码/记录/审查资料合计最多2MB、临时树最多1MB，沿用既有预留。先验证固定Torch2.8在sealed memfd上的weights_only/private-mmap兼容，再验证原始子进程退出、新进程恢复和不中断训练的下一步一致；参数/buffer/别名、Adam moments/step/options、版本/config和受支持RNG都需保留。save侧metadata另有独立硬界限，load不增加自定义unpickler或allowlist。CUDA RNG代码只允许已初始化上下文，实际CUDA/FSDP恢复仍需后续真实门禁。

实现来源为子agent `/root/task11_ram_handoff`；独立初审与首轮规格/质量复审见下文，不冒称真人审查。完整模型内存、更新推理权重的空间与rollout激活、训练质量、性能和外部基线仍未验收。

首次静态检查的12条lint错误保留，限定修正后Ruff通过。首次[CPU日志](raw/task11-ram-handoff-compatibility.log)收集0项、退出4：固定Python的`fcntl`未导出`F_SEAL_WRITE`。测试正文没有运行，不能据此判断内核封存或Torch mmap是否可用。[终态](raw/task11-ram-handoff-compatibility.json)确认子进程回收、无剩余自有子进程；采样RSS加匿名预留最高653,340,672字节，临时树文件0字节。[root核验](raw/task11-ram-root-fcntl-failure-check.json)记录本机Linux头文件中的六个常量、实际缺失属性及全部12条临时目录身份。修复将依据[Linux封存接口](https://man7.org/linux/man-pages/man2/F_ADD_SEALS.2const.html)，继续要求真实内核设置/读取封存；原失败记录保留，后续结果另记。


补齐常量后，[兼容性v2](raw/task11-ram-handoff-compatibility-v2.log)1项通过，实际封存及私有映射符合要求。[聚焦原始输出](raw/task11-ram-handoff-focused.log)为28通过、1失败：预算测试请求131,072字节单storage，先触发8,192字节storage上限，未到达预期RAM断言。只修正该输入，[单项回归](raw/task11-ram-handoff-cap-regression.log)1项通过；原focused结果仍记录失败，没有覆盖或重复已通过用例。总计29个新用例及1个原有phase guard均有通过记录。

实际TinyLM/autograd/AdamW连续性测试经过三个独立子进程：写入第一状态后退出、恢复并执行第二步后写第二状态并退出、新进程再次读取第二状态。参数、buffer、固定reference、非零moments、step/options、版本、RNG及下一步loss均与同seed不中断两步一致；新状态17,648字节、版本2，loss为-0.18994787335395813。另有真实world1 CPU/Gloo DTensor恢复，不能当作CUDA/FSDP实证。错误mapping在mutation前拒绝，超时/信号/不封存、错误FD/散列/截断、slot/foreign-owner、metadata/archive预算及失败phase/原始加cleanup异常均有聚焦检查。

四次CPU命令的监督器均确认清理完成、无剩余自有子进程；各次采样RSS加1MiB匿名预留最高分别为653,340,672、698,724,352、1,370,574,848、710,393,856字节，不是连续峰值或内核硬配额。全部临时树文件为0字节；root完整核对42条目录身份和路径集合后，仅用rmdir清理本轮空目录，见[删除前完整库存及结果](raw/task11-ram-root-scratch-cleanup.json)。全部原始日志、封存receipt和失败记录保留。


本地实现提交`d0e7420`的独立规格/质量初审来源为子agent `/root/review_task11_ram`，当前为Issues found/Needs fixes：Important I1为consume遗漏两份storage暂存预留，I2为ZIP中央目录在独立metadata界限前已由ZipFile分配。同一实现代理将做首轮聚焦修复，本前置阶段暂不关闭。Minor M1指出同seed夹具未主动改变RNG，覆盖加强已递延至最终整体审查；没有把代码已实现的RNG路径冒称覆盖充分或CUDA已验证。


首轮修复`4027883`统一consume/advance的恢复暂存公式，并在默认Torch loader之前以固定小读取预检EOCD/ZIP64、目录总字节/记录数和名称/extra/comment范围。保存计数阶段仅保留有界目录尾部，在任何输出FD写入前验证；metadata规划预留改为16×metadata_cap+135,168字节，测试额外匿名预留改为2MiB，总2GB CPU上限不变。[16项聚焦输出](raw/task11-ram-handoff-fix1.log)全部通过，含15项新边界与一条受变更影响的真实AdamW控制；没有实际RED执行主张或无关旧套件重跑。

本次监督器退出0、无cleanup错误或剩余自有子进程；采样RSS加2MiB最高1,462,386,688字节。root核对全部12条临时目录身份及7份执行源码散列后，仅清理本轮空目录，见[首轮结果及删除前库存](raw/task11-ram-root-fix1-result-cleanup.json)。原始20份日志/receipt、初始提交及两份审查包均保留；同一独立子agent的限定复审已确认I1/I2解决，规格与质量均Approved，无新增Critical/Important/Minor；仅关闭本CPU前置阶段。Minor M1仍递延，完整模型内存、GPU、训练质量和整个Task11/Goal仍未完成。


复审来源仍为子agent `/root/review_task11_ram`；初审16,285字节前缀保持原样，后附独立的首轮复审10,647字节。root负责并另存完整删除前库存/终态、实际源码与Git身份、主机归属、冻结输入、协调证据、资源和Azure检查。Minor M1继续递延至最终整体审查，未默认为解决。

恢复步骤：先核对最新STATUS、资源和冻结身份，再为完整0.5B的实际生命周期提供可计量16GB方案。两份raw状态已有13,832,917,504字节，下次准入必须包含实际learner基线、暂存、metadata和映射；本阶段的零基线算式和tiny CPU结果不能替代该证明。随后才可推进实际训练状态交接、推理导出/rollout激活及等预算实验；不得关闭FD就假定映射释放、重置Adam、借用磁盘超额或改写失败。Task12 full03 v2仍需新的可信精确单次执行授权和fresh门禁，原full02授权已用且失败。整个Task11及Goal保持开放。
