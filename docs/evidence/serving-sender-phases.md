# Task12 客户端发送阶段观测

唯一一次新 CPU 观测进一步定位了发送迟到所在阶段：index2 的完整写入迟到22.146044ms，其中21.048895ms在实际 `select` 返回的观测点之前已累积。该运行仍为 sender-invalid；观测完成不表示性能通过。原 full02v3、原客户端 CPU 复现及 epoll 未复现记录继续保留，见[此前证据](serving-sender-diagnosis.md)。

本阶段由实现子agent编写观测代码并解释结果，root独立复算原始记录；独立规格和质量审查已通过，无Critical/Important/Minor；结论仅覆盖本次CPU观测阶段。整个Task12与项目均未完成。

`Run` 与可选 `RunWithDiagnostics` 共享调度和发送实现；默认Config及CLI JSON格式保持一致。只释放了原冻结清单中三个Go源码的修改权限，旧客户端二进制保持原散列。未启用观测时不创建观测缓冲或读取新增观测时钟。实际时刻覆盖timer select返回、Stop结束、发送goroutine、连接回调和底层Write两端；Stop与Write的两端时刻均在写入记录锁前采集。完整写入返回时刻继续决定原SentMs。

每请求最多16对Write、48事件，最多256请求；缺失、重复、逆序或溢出使捕获失效。独立sidecar在序列化并确认不超过60KiB后独占发布，捕获有效性与SenderValid分开。

新增六项回放器测试和一项CLI测试、五项直接受影响的旧控制检查、针对新代码的race检查均通过；监测核算另有10例聚焦回归通过。旧版CLI在隔离源码副本上因缺少新参数产生预期RED。首次编译的核算误报和随后内部send签名不兼容的编译失败均保留；前者用独立非loan扫描修正，后者恢复兼容转发入口，未修改旧测试。详见[root构建核验](raw/task12-sender-phase-root-build-check.json)。

唯一计时运行使用原23条Burst请求、seed17、倍率1、60秒请求期限及192并发上限，复用原立即响应CPU夹具；没有模型或GPU。全部23条原始记录为ok、未abort，index1和index2迟到，客户端退出1；外层退出0只表示诊断完成，捕获有效。

| 请求index | 完整写入迟到(ms) | 计划时间至post-select迟到(ms) | post-select至完整写入(ms) |
|---|---:|---:|---:|
| 1 | 6.360352 | 5.971081 | 0.389271 |
| 2 | 22.146044 | 21.048895 | 1.097149 |

index2的Stop耗时0.000782ms、Stop至发送goroutine入口0.081290ms、ConnectStart至ConnectDone为0.341002ms、底层Write为0.092274ms。其余连接准备和回调间隔均单独保留。这里仅定位新同次运行的阶段；尚不能区分该观测点之前的timer唤醒、Go调度和主机干扰，更未证明具体内核机制或原GPU运行的内部原因。钩子本身也会扰动新运行。

原始[phase数据](raw/task12-sender-phase-cpu-01/phases.json)、[阶段结果](raw/task12-sender-phase-cpu-01/phase-terminal.json)、[夹具终态](raw/task12-sender-phase-cpu-01/terminal.json)及[root独立复算](raw/task12-sender-phase-root-result-check.json)保留全部请求与失败分母。

夹具耗时59.039194秒；582个合计RSS样本最大60,432,384字节，并非连续峰值。子进程自然退出1且已回收，线程归零、监听关闭；[主机检查](raw/task12-sender-phase-host-cleanup.json)另确认控制器和子进程不存在、31013端口可绑定后立即关闭。没有调节阈值、优先级、CPU亲和、timerslack或Go运行时设置，也没有重复运行求通过。

Go冷缓存、测试临时文件及10,154,096字节新二进制一直位于原690MB预留内的500MB临时借用树。运行前完整项目13,540,787,544字节，非loan加690MB投影13,988,003,649字节；14GB上限不变。清理前完整[压缩清单](raw/task12-sender-phase-loan-inventory.json.gz)含3,730条路径、3,464个文件，root独立逐项核对身份、字节数和散列；仅该新树获准回收，旧二进制、运行时、共享模型及全部原始证据保留。

临时树已按清单回收，完整690MB预留重新通过；最新[资源检查](raw/resources-end-resume-25.json)为项目13,300,451,354字节、共享模型缓存23,815,435,170字节。默认[只读Azure检查](raw/azure-reaper-end-resume-25.json)仍为rg-crucible-lab absent/action=none。累计估算云/API/GPU租赁费用0美元，无推送或公开发布。
