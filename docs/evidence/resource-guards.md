# 资源安全入口：本地离线证据

日期：2026-10-23。Crucible 云/API/GPU费用上限 **0 美元**，仅允许本机 RTX 4090；本次未操作真实 Azure 资源、未连接远程 GPU、未运行 GPU 作业，未写入共享锁/模型缓存。**未在真实多机上验证**。累计费用保持 0 美元。

控制器后续补充的真实只读检查：默认 `bash scripts/azure_reaper.sh` exit 0，固定项目组返回 absent/action=none。原始输出见 [raw/azure-reaper-initial.json](raw/azure-reaper-initial.json)，详见 [核验记录](zero-budget-resource-check.md)。未带 --apply，没有实际删除或 GPU 工作负载。

## 入口和约束

- `scripts/azure_reaper.sh [--plan|--apply]`：默认只读，固定 `rg-crucible-lab`，没有资源组/订阅覆盖参数。只接受当前 Enabled 的 `Azure for Students` 账号，捕获其 UUID 并向每条后续 CLI 命令显式传 `--subscription`。仅考虑 ID 位于此订阅和此组、`project=crucible`、`owner=codex-crucible`、合法 UTC `ttl` 已过期且成本估计标签合法的资源。UTC 格式接受 `YYYY-MM-DDTHH:MM:SS[.fraction]Z` 或 `+00:00`；时间等于当前时刻不算已过期。非法/未知资源输出 skip，删除前重取 ID 和全部 tags，变化立即停止。
- `make azure-down`：只读整组计划；显式执行入口为 `make azure-down-apply`（底层 `scripts/azure_down.sh --apply`）。group 和资源列表均须具有完整合法标签及归属，执行前重新读取两者，检查 inventory/tag 变化。未来 ttl 也可整组下线；此处不要求到期，但绝不接受未知/混合归属。
- 两者仅在 `az group exists` 明确返回 JSON `false` 时以 absent/no-op 成功结束。CLI 缺失、未登录、命令失败、异常 JSON、无效订阅均输出错误且非零退出；不把查询失败解释为空资源。
- `scripts/with_gpu.sh -- foreground-command [args...]`：固定 `flock --no-fork /var/tmp/rtx4090.lock`，专用 Linux subreaper supervisor 持锁，任务子进程不继承锁 FD。正常可终止任务取得锁后执行和清理合计最多四小时，预留最多一秒执行 TERM/KILL 清理。正常结束、超时、INT/TERM/HUP/QUIT 都清理自己的子/收养后代，必须由内核确认没有任何子进程才释放锁；超时返回 124，终止信号返回 128+signal；发生归属/清理异常后安全收尾返回 125。无法确认退出则报告 `cleanup_unconfirmed` 保持锁隔离，可能超过四小时，不授权继续训练。没有生产 CLI/environment 的锁或时间上限覆盖；临时锁和短时间窗仅由内部 `_run_locked` / `_locked_command` 测试 seam 注入。
- GPU 命令必须前台运行；拒绝常见 daemon/detach/nohup/setsid 入口。支持 Linux 自己 fork/exec/spawn 的子进程树，已覆盖意外新 session/双 fork 后代的 CPU 收尾，不承诺外部服务管理器创建的作业或不可信代码的隔离。外部 SIGKILL supervisor/主机崩溃不在用户态保证范围。不得杀其他项目进程、改驱动、长期保留推理服务。vLLM `--gpu-memory-utilization 0.80` 由后续 Task7 入口实现。
- `make resource-preflight PREFLIGHT_ARGS='...'`：只读扫描整个项目（含环境/缓存、去重 hardlink、不跟随 symlink；不可读取目录则报错），加预计新增量校验项目 14 GB；校验共享 `~/.cache/huggingface` 和相关缓存环境变量，超过 5 GB 的下载要求至少 50 GB+下载大小的剩余空间，另为项目新增文件预留空间。checkpoint 总保留数最多 2，常驻服务合计计划峰值内存最多 16 GB。GB 一律为 **1,000,000,000 bytes**，避免放宽上限。

## TDD 和命令结果

| 检查 | 结果 | 原始输出 |
|---|---|---|
| 新模块导入 RED | 2 个预期 collection errors（模块未实现） | [tdd-red.txt](resource-guards/tdd-red.txt) |
| reaper 默认计划 RED | 1 个预期 failure（shell 入口未实现） | [azure-red.txt](resource-guards/azure-red.txt) |
| 自审补强 RED | 4 个预期 failure（不存在组、可选字段、wrapper TERM） | [fix-red.txt](resource-guards/fix-red.txt) |
| `make test-resource-guards` GREEN | 65 passed | [tdd-green.txt](resource-guards/tdd-green.txt) |
| `.venv/bin/python -m pytest`（本任务唯一一次全量） | 130 passed，10.77s | [full-tests.txt](resource-guards/full-tests.txt) |
| `.venv/bin/ruff check .` | All checks passed! | [ruff.txt](resource-guards/ruff.txt) |
| `bash -n scripts/azure_reaper.sh scripts/azure_down.sh scripts/with_gpu.sh` | exit 0 | 本地 shell 语法检查，无云调用 |
| 未修正继承环境的零新增 preflight | 正确拒绝非共享 HF_HOME/HF_HUB_CACHE，脚本 exit 1 | [preflight-rejected.txt](resource-guards/preflight-rejected.txt) |
| 单次命令配置共享缓存后的同一 preflight | ok=true，约 0.315 GB 项目文件，剩余空间见 JSON | [preflight-green.txt](resource-guards/preflight-green.txt) |

通过的只读 preflight 命令（仅单次环境前缀，无 mkdir、下载或全局环境修改）：

```bash
HF_HOME="$HOME/.cache/huggingface" HF_HUB_CACHE="$HOME/.cache/huggingface/hub" \
  make resource-preflight PREFLIGHT_ARGS='--planned-project-bytes 0 --download-bytes 0 --checkpoints 0 --service-memory-bytes 0'
```

测试 Azure 使用 `/tmp` 的 `az` shim/JSON fixtures，记录每次命令并断言删除范围；`--apply` 只针对 shim。GPU wrapper 测试只有 `/tmp` 临时锁和 Python CPU sleep 进程，覆盖排队、锁等待不扣执行时长、超时、TERM、背景后代清理及释放后再次获取锁；没有访问 `/var/tmp/rtx4090.lock`。preflight 只读取路径/容量信息，不下载模型、不清理 Docker 或其他项目。

## 官方参数核对与剩余前提

已于本日核对微软官方 [az resource](https://learn.microsoft.com/en-us/cli/azure/resource?view=azure-cli-latest)、[az group](https://learn.microsoft.com/en-us/cli/azure/group?view=azure-cli-latest)、[az account](https://learn.microsoft.com/en-us/cli/azure/account?view=azure-cli-latest)：`resource list --resource-group`、`resource show/delete --ids`、`group exists/show/delete --name`、`group delete --yes`、`account show` 以及 `--subscription/--output json/--only-show-errors`。没有真实调用 Azure CLI 来验收本实现。

真实运行需要本项目 `.venv`、Linux util-linux `flock`、Azure CLI 登录和清理权限（仅云清理时）、工具允许的共享路径权限（仅真实 GPU 任务时）。preflight 的 checkpoint 数/峰值内存/预计文件量是声明式检查，调用者需核实并遵守；不自动删除 checkpoint、不施加 cgroup 内存限制，不检查/清理共享缓存内容。共享缓存合计 40 GB 由 Keel/Crucible 协同管理。不同文件系统采用保守的重复余量检查。

Azure CLI 没有在这里提供跨查询/删除的原子比较操作；重新读取能拒绝已观察到的变化，但仍有最后一次读与删除之间的竞态窗口。整组删除只可在没有并发创建/修改的维护窗口执行。真实 Azure 删除、GPU 执行和真实多机验收保持未验证。

## Review fix round 1：实际 Hugging Face 缓存路径

基线 `c724aa7`。审查发现 `HF_HOME` 缺省时 preflight 忽略 `XDG_CACHE_HOME`，错误报告共享目录并检查错误文件系统，可能把实际非共享缓存放行。

修复依据官方 [HF_HOME / XDG_CACHE_HOME 文档](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome) 和 [constants.py 默认路径解析](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/constants.py)：显式 `HF_HOME` 优先，其次 `$XDG_CACHE_HOME/huggingface`，最后才是 `~/.cache/huggingface`；先展开用户目录、再展开环境变量。preflight 的 JSON `model_cache` 和容量检查现在使用实际解析路径。未创建任何缓存目录。

空 `HF_HOME`、或者 HF_HOME 未设置时的空 `XDG_CACHE_HOME` 明确报错，保守拒绝，不回退成默认值；有效的显式共享 `HF_HOME` 优先，未使用的 XDG 值（包括空值）不会影响它。hub/transformers 覆盖变量继续对照固定共享 `hub` 子目录，空值或冲突值继续拒绝。

聚焦命令 `.venv/bin/python -m pytest scripts/tests/test_resource_preflight.py -q`：新增 CLI 环境矩阵先 **3 failed, 26 passed**（[RED](resource-guards/xdg-fix-red.txt)），修复并补充实际容量查询路径回归后 **30 passed in 3.03s**（[GREEN](resource-guards/xdg-fix-green.txt)）。覆盖非共享/空 XDG、显式共享 HF_HOME 优先、两个默认路径、空缓存变量与冲突覆盖；容量回归另验证使用有效 XDG 所在最近已存在目录。仅对变更文件运行 ruff，**All checks passed!**（[输出](resource-guards/xdg-fix-ruff.txt)）。本轮没有重跑全量、调用 Azure/GPU、修改全局环境或写共享路径。

自审确认：只修改 preflight、其环境变量测试与直接相关文档；保留控制器既有默认只读 reaper 记录及其他资源入口。声明式资源检查和原有未完成真实验收边界保持不变。

## GPU 进程归属修复（基线 865c5c5）

问题：原 supervisor 的 `Popen.poll()` 可以先回收 leader，再对其旧数值 PGID 发信号，存在复用误伤风险；同时没有整个工作负载退出的内核证明。

修复仍在 `scripts/gpu_guard.py` 单文件内：

1. 专用、单线程、唯一 reaper 的 supervisor 将 SIGCHLD 设为 SIG_DFL，启用本进程 `PR_SET_CHILD_SUBREAPER`；启动前由 `waitid(P_ALL, ..., __WALL)` 的 ECHILD 证明没有既有子进程，不接受空 proc 快照作为启动授权。不增加 cgroup/主机权限依赖。
2. 通过 `waitid(P_PID, ..., WNOWAIT|WNOHANG|WEXITED|__WALL)` 观察 leader，保留活进程或 zombie 锚点，调用 `getpgid` 确认私有组；此期间不调用 `Popen.poll/wait`。只有锚点有效才允许 group 信号，失去归属立即永久禁用旧 PGID。
3. `/proc/self/task/<self>/children` 只用来发现自己的子进程，不扫描其他 PID，也不把快照当作完整性证明；每个候选 PID 须先被 waitid 确认为仍是自己的未回收子进程，才可发信号。subreaper 接收孤儿后代，包含意外换 session/双 fork 的 worker。
4. 按控制器确认的两阶段裁定，先永久撤销 group-signal 能力，再 retire/reap leader，整个阶段仍持有 flock。随后只处理 waitid 确认的 own/adopted PID，用 `P_ALL + __WALL` 逐个回收，最终仅 ECHILD 才能解锁；None 表示仍有活 child，绝不代表为空。失去 anchor 后若最终 ECHILD，也可非零结束隔离，不无理由永久锁。
5. 观察失败仍尝试对有效锚点完成 TERM/KILL 升级；超时/无法确认则报告 `cleanup_unconfirmed` 并保锁，诊断管道关闭也不能导致提前返回。不可终止内核状态可能使隔离超过四小时；这不是继续执行训练的许可。外部 SIGKILL/主机崩溃无法由用户态兜底。

核对的 Linux primary API 语义：[PR_SET_CHILD_SUBREAPER](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html) 描述孤儿重归属及 wait 能力；[wait/waitid](https://man7.org/linux/man-pages/man2/waitid.2.html) 描述 WNOWAIT、P_ALL、ECHILD 和 Linux 4.7+ 的 `__WALL`，后者避免忽略非 SIGCHLD clone child；[pidfd_open 的 zombie 前提](https://man7.org/linux/man-pages/man2/pidfd_open.2.html) 列明无自动回收/无其他 reaper 时 PID 不复用；[proc children](https://man7.org/linux/man-pages/man5/proc_tid_children.5.html) 明确并发退出会漏项，故不承担最终证明职责。

TDD 输出：

| 命令（前缀 `.venv/bin/python -m pytest scripts/tests/test_gpu_guard.py`） | 修复前结果 | 输出 |
|---|---|---|
| `-k ownership_shim -q` | 3 failed，旧 PGID/未确认返回 | [RED](resource-guards/gpu-ownership-red.txt) |
| `-k empty_snapshot -q` | 1 failed，proc 漏项不能解锁 | [RED](resource-guards/gpu-ownership-snapshot-red.txt) |
| `-k proc_error -q` | 1 failed，观察错误不得跳过有效锚点升级 | [RED](resource-guards/gpu-ownership-discovery-red.txt) |
| `-k existing_child -q` | 1 failed，启动须 kernel 无子进程证明 | [RED](resource-guards/gpu-ownership-startup-red.txt) |
| `-k broken_error_stream -q` | 1 failed，坏诊断管道不能放锁 | [RED](resource-guards/gpu-ownership-diagnostic-red.txt) |
| `-k non_sigchld -q` | 1 failed，kernel proof 须包括 clone child | [RED](resource-guards/gpu-ownership-clone-red.txt) |
| `-q` | **23 passed in 3.10s** | [GREEN](resource-guards/gpu-ownership-green.txt) |

`.venv/bin/ruff check scripts/gpu_guard.py scripts/tests/test_gpu_guard.py`：**All checks passed!**（[输出](resource-guards/gpu-ownership-ruff.txt)）。本轮只跑这个聚焦文件，没有重跑整仓。复用/失去归属/不确定清理均为 syscall shim，隔离循环由测试异常有界截停，没有长期持锁或真实挂死进程。实际 CPU 测试只使用私有临时锁和自身短时进程，覆盖排队、超时、信号、普通 worker、新 session/双 fork worker 回收后再解锁。没有运行 GPU、Azure、远程作业，也没有写共享路径、真实 cgroup 或主机配置。真实 vLLM/FSDP 的多进程/GPU 清理行为仍待授权实际验证。

## 独立审查结论

首次审查发现 XDG_CACHE_HOME 默认值遗漏；修复提交 c8de07c，30 项 preflight 聚焦测试通过。聚焦复审 Approved，无剩余 Critical/Important/Minor。Azure/GPU 入口的其他本地审查项已通过；实际只读 reaper 查询已另行记录，真实删除、GPU 工作负载及多机验收仍未发生。
