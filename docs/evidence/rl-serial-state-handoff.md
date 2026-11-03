# W2完整状态的串行内存交接

这是Task11的另一项前置实现。初版45项CPU检查、首轮修复12项聚焦回归及同一子agent的规格/质量独立复审均通过，仅关闭本CPU前置阶段；没有GPU或完整模型实验，整项Task11和原Goal仍开放。此前保留双快照的CPU路径及其独立复审见[原证据](rl-state-handoff.md)，该结论不扩大到本阶段。

一份Qwen2.5-0.5B完整训练状态的raw载荷下限为6,916,458,752字节，包含FP32策略、Adam moments和BF16 reference；两份为13,832,917,504字节，尚未计入进程、映射、暂存和元数据。[静态上下文](raw/task11-ram-state-static-context.json)和历史learner RSS均为规划依据，不能当作同时峰值或16GB准入证明。

新路径保留全部optimizer step/options、参数/buffer/别名、reference、配置、版本和受支持RNG，采用有界清单与原始张量记录。它在分配张量前检查完整尺寸、偏移、dtype、device及CPU/GPU总量，再按storage恢复和验证实际数值；不依赖FakeTensor返回对象图来约束pickle全部中间分配。原ZIP/mmap接口保留。

串行生命周期必须满足以下顺序：

1. 原写入者退出并回收，父端核验封存、身份、大小和散列，接受输入。
2. 新worker只取得输入和本地控制FD，完成恢复、复制和同步，关闭输入并等待。
3. 父端保持原子进程的归属锚点，实际核验输入FD和映射消失，核验父端只有最后一个输入句柄，然后关闭它并退休旧receipt。
4. 此后才创建并传递新输出FD；worker执行下一步、保存并封存，父端在同一原子进程正常退出并回收后验收输出。

第3步之后存在明确的不可恢复间隙。失败意味着没有已接受的新checkpoint，不能声称旧状态仍有效、重置Adam、自动重试或报告成功。释放不明时保留预留并使owner失效；FD整数可复用，清理和验收都必须核对实际身份。

16,000,000,000字节host上限包含匿名memfd/pagecache、所有映射和副本、CPU候选/目标、暂存、元数据及父子进程基线；VRAM另按单张24GB卡核算全部候选、目标和临时量。最多两个checkpoint包含RAM、pending、磁盘和未来推理权重；只有强制执行上述生命周期后，才可用串行阶段最大值计算。未来独立BF16推理状态也必须保留字节和slot预算。本阶段不能证明完整模型或训练/rollout流水线已满足这些条件。

本阶段留存源码、原始记录、审查和元数据合计上限2MB，CPU临时树1MB，均在既有690MB未来预留内；项目14GB不变，无下载、安装、云资源或GPU作业。[本轮开始资源](raw/resources-start-resume-29.json)与[冻结/协调证据核验](raw/task11-stream-root-preservation-before-source.json)已记录。实现来源为子agent `/root/task11_stream_handoff`，独立审查来源为子agent `/root/review_task11_stream`，不是人工标注/审查。

首次格式化因线程池初始化/分配失败退出SIGABRT，源码未变；单线程格式化随后成功，lint的6处清理异常捕获只增加局部解释，最终检查通过。全部失败和分次日志保留，同一Python3.12下的AST与原源码匹配；早期Python3.14与3.12的dump比较不是语义变化证据。Ruff线程数控制依据[Rayon接口说明](https://docs.rs/rayon/latest/rayon/struct.ThreadPoolBuilder.html#method.num_threads)。

[一次CPU原始输出](raw/task11-ram-handoff-stream.log)为45 passed / 13.64s，含44项新路径及1条受共用复制函数影响的原AdamW控制。真实TinyLM经过写入/退出、恢复/消费输入/更新/退出、再次读取三个独立子进程，完整状态与不中断两步的状态和loss一致。新stream状态28,830字节、版本2、SHA256为625c400e66c64087238d1328a751d6217995413012145c5c0f06c493f67d9cf9，下一步loss为-0.18994787335395813。原路径控制也通过，没有重跑无关旧套件。原Minor M1主动扰动RNG的覆盖加强仍递延整体审查。

这份tiny夹具的reference为FP32复制；BF16是已实现的支持类型，本轮没有提供BF16 reference或实际Qwen恢复证据。初审两项Important分别为：父端关闭源FD后、登记和标志提交前的异常可能留下健康owner或陈旧已接受项；初始输出memfd的创建/fstat位于追踪和清理区域外，失败时可能遗留未追踪能力。初版提交`cb84477`及初审原文保留，同一实现代理在`92744f5`补齐失败封闭的转换及定点回归，同一独立审查agent已在首轮复审确认两项解决。

实际事件记录显示输入child无源FD/映射，父端仅保留最后句柄，随后source-retired为0个已接受训练状态，才创建/授予新输出；未来权重slot为声明的夹具预留，此时总slot为2，不代表已实现推理权重导出。[终态](raw/task11-ram-handoff-stream.json)确认原监督器退出0、清理完成、无错误和剩余自有子进程。采样RSS加4MiB匿名预留最高1,419,063,296字节，不是连续峰值或内核硬配额，更不是完整模型准入。

root复核9份执行代码身份及全部11条目录路径/元数据后，只删除本轮空临时树，见[完整删除前库存与结果](raw/task11-stream-root-result-scratch-cleanup.json)。全部原始证据和失败保留。完整0.5B内存、CUDA/FSDP、推理权重更新、等预算训练和消融、留出质量、外部基线和报告仍未完成。Task12 full02实际发送门禁失败及已耗用授权保留，full03 v2需要新的可信精确单次授权。

首轮修复另有[12项CPU回归](raw/task11-ram-handoff-stream-fix1.log)通过/5.54s，含8项新生命周期反例、3条受影响控制和真实AdamW连续性。源退休前先登记不可恢复转移；新输出创建前先保留slot/容量，身份不明时保留预留并使owner失效。原执行句柄83761终态0；[监督记录](raw/task11-ram-handoff-stream-fix1.json)清理完成/无剩余子进程，采样RSS+4MiB最高1,363,451,904字节，不是连续峰值或完整模型准入。root独立核对10源码身份及11条空目录后精确清理。I1/I2均已由原独立审查agent确认解决，规格/质量均Approved，无新增阻断问题；M1继续递延整体审查。

本轮[结束身份核验](raw/task11-stream-root-fix1-preservation-end.json)确认Task12的394冻结身份、100份协调证据及原客户端/15目录身份不变；[主机只读检查](raw/task11-stream-root-fix1-end-host-audit.json)为精确项目CPU候选0，67项不可读/竞态条目保留，不证明全局GPU/锁状态。初次argv文本误匹配和CPU预检解析错误均另存，未启动额外实验。[结束Azure检查29](raw/azure-reaper-end-resume-29.json)仍为项目组不存在/actionnone，累计估算0美元。完整模型准入和原Goal继续开放。
