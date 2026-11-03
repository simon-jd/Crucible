# full03 v2 单次原生执行：失败并清理

同用户[新授权记录](raw/task12-full03-v2-human-authorization-20261025.json)绑定的proposal/freeze/6524字节摘要未改。ROOT交回CPU源码并本地提交124d28e后，[fresh门禁](raw/task12-full03-v2-root-fresh-preflight-20261025.json)实际核验183源码、207证据、4可执行文件、161包元数据、19运行时审计源、14模型实际文件和上游散列，以及冻结43+4环境、15目录身份/14空叶、端口/无重复实例和资源。全部通过后仅一次经冻结launch_full入口启动；自动审批未拒绝。历史XDG767仅为实验重放配置。

原始工具句柄60652、wrapper3061993、guard3062015、server3062912；[入口预提交](raw/task12-full03-v2-root-launch-state.json)、[原始句柄](raw/task12-full03-v2-root-original-handle.json)、[终态句柄](raw/task12-full03-v2-root-original-terminal-handle.json)和[launcher终态](raw/task12-baseline-full-03-launch.json)保留。原句柄退出1，已回收。[原始terminal](raw/task12-baseline-full-03/terminal.json)为aborted=true、baseline_accepted=false、run_completed=false；启动inspect→check_resources→NativePool严格扫描抛ValueError('nonregular pool payload')。全部30个trial均not_run；native smoke/warmup未验收。没有发送延迟、SLO或性能改进结论。本次授权已耗用，禁止自动重试，原full02失败/已用授权保持。

失败现场在启动前为空的TMPDIR留下tmp48m9d3ci和tmps86njvgs两个AF_UNIX socket；原异常未带路径，尚无每次系统调用或创建者归因。不能仅据此指定底层根因、放行所有特殊文件或修改5ms/>1%门槛。协调者独立观察两项uid/gid1000且原三个PID消失，与ROOT终态一致。后续仅授权有界CPU核对精确IPC来源、兼容修正及复审，不执行或排队新GPU；新候选就绪再提供精确摘要，不提前反复请求确认。

[清理前主机与完整28项库存](raw/task12-full03-v2-root-postflight-before-cleanup.json)记录原三个PID均不存在、项目候选0、31012可绑定、GPU无项目进程及394冻结身份吻合。首次非阻塞锁检查EAGAIN原样保留在[第一次观察](raw/task12-full03-v2-root-postflight-lock-busy-first.json)，仅说明当时锁忙，未归因原guard、未终止别人，也未要求其他项目停GPU；后续主机点位可取得共享锁并立即释放。服务TERM退出-15，无KILL，reaped/listener_closed；外层guard终态及最终锁/主机核验补足后代清理证据，点位不保证未来全局空闲。

ROOT完整核对全部28路径lstat及SHA，保留2355字节生成文件全文后，仅[删除本次12项临时产物](raw/task12-full03-v2-root-exact-temp-cleanup.json)：1普通文件、2残留socket和9目录。原客户端10,160,231字节、原15目录身份及14空叶保持；没有删运行时、旧失败、共享模型、协调证据或缓存来资助下一次实验。

6个资源样本最大自有RSS997,625,856字节，仅启动期采样，非连续峰值或完整模型fit证明。fresh项目13,317,846,205字节、含原预留投影13,997,685,974字节；清理后项目13,317,942,708字节、含既有输出credit的投影13,997,718,283字节。客户端仍在原500MB池/690MB总预留内，不能再重复加客户端或完整690MB。共享HF23,815,435,170字节未下载。$0/14GB/16GB/4小时含清理/离线/固定flock保持；[结束Azure](raw/azure-reaper-end-full03-v2-20261025.json)absent/actionnone。Task12、Task11及整个Goal仍开放。

清理统计更正：上文“1普通文件/9目录”为ROOT汇总笔误；原始逐路径清单未错。按完整28项库存与12项删除清单逐条复核，实际为5普通文件（4空文件及已保留全文的2355字节文件）、2残留socket、5目录，合计12项，现均不存在。保留原文和raw不改，见[精确更正](raw/task12-ipc-root-cleanup-count-correction.json)。
