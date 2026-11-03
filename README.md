# Crucible

GPU 集群上线、异步 RL 后训练（GRPO）与 P/D 分离推理，外加一个 PyTorch custom op。

保留原始验收标准，真实环境验证未完成时明确标为未验证。

证据约定与实际结果：[docs/evidence/README.md](docs/evidence/README.md)。租卡前先填写 [实验检查单](docs/EXPERIMENT-CHECKLIST.md)。

本地开发（Linux amd64，先提供 uv）：

```bash
make bootstrap
.venv/bin/python -m pytest
```

bootstrap 只在本项目目录安装锁定的 Python 开发依赖与 Go 工具链，不开通云资源。本机 RTX 4090 已只读核验，GPU 作业需经批准的主机入口与共享锁运行；多机调度/RDMA 的真实验收尚未完成且已暂停。Go 模块实现后可运行 `make test` 同时检查 Python 与 Go。

后续 Torch 任务使用独立环境：`make bootstrap-ml` 安装锁定的 `.venv-ml`，先检查磁盘份额，不下载模型。`.venv-ml/bin/python scripts/check_torch_runtime.py` 检查 CPU；CUDA 检查必须通过 `bash scripts/with_gpu.sh -- .venv-ml/bin/python scripts/check_torch_runtime.py --cuda`。当前真实 CPU/RTX 4090 小矩阵与梯度检查已通过，见 [运行环境证据](docs/evidence/ml-runtime.md)。小模型 rollout/learner 的单卡功能验证及 [custom op 门禁](docs/evidence/fused-logprob.md)已通过；训练留出质量和完整 serving 基线仍待验收。

资源上限：Crucible 云/API/GPU费用 **0 美元**，仅使用本机 RTX 4090。不得租卡、连接远程 GPU 或修改 `rg-mlinfra-westus`。需要多机、235B 模型、P/D 分离和跨节点的真实实验暂停。

本地资源入口（需要本项目 `.venv`；详细边界和离线证据见 [resource-guards](docs/evidence/resource-guards.md)）：

```bash
make test-resource-guards      # Azure CLI shim + 临时锁 + CPU 进程测试
make azure-reaper              # 只读计划；需要 Azure CLI 登录，资源组不存在则 no-op
make azure-down                # 只读整组删除计划，拒绝混合/未知归属
# 真实删除的显式入口：bash scripts/azure_reaper.sh --apply 或 make azure-down-apply

# 下载/启动前申报；下例仅做零新增的只读检查，数字单位为字节
HF_HOME="$HOME/.cache/huggingface" HF_HUB_CACHE="$HOME/.cache/huggingface/hub" \
  make resource-preflight PREFLIGHT_ARGS='--planned-project-bytes 0 --download-bytes 0 --checkpoints 0 --service-memory-bytes 0'

# 后续获准的本机 GPU 任务必须前台通过此入口运行（此处没有运行 GPU 作业）
# bash scripts/with_gpu.sh -- .venv/bin/python <任务脚本> <参数>
```

`with_gpu.sh` 固定使用 `flock /var/tmp/rtx4090.lock` 排队，与 Keel 共用；等待锁的时间不计入四小时执行窗口，正常可终止作业的执行和清理合计最多四小时。专用 Linux supervisor 在自身进程启用 subreaper（不需要 cgroup 权限），保留 leader 归属锚点；group 信号能力一经撤销便不再使用旧 PGID。leader 退出后仍清理自己的收养后代，直到内核确认没有子进程才解锁，不能把 leader 退出或 `/proc` 空快照当作全部 worker 退出。清理无法确认时报告 `cleanup_unconfirmed` 并保持锁隔离，停止新工作；这是故障隔离，不是延长训练许可，此时不能承诺四小时内退出。

任务必须前台运行，禁止后台/daemon/脱离监督。入口支持由自身 fork/exec/spawn 的 Linux 子进程树，已用短 CPU 新 session/双 fork 回归验证收尾；真实 vLLM/FSDP 多进程 GPU 范围仍需实际验证，不覆盖外部服务管理器另起的作业。入口不是不可信程序的沙箱，外部 SIGKILL supervisor 或主机崩溃会破坏用户态清理保证。不得干预别的 GPU 进程或驱动，推理服务在一批评测后停止；vLLM 后续入口须设 `--gpu-memory-utilization 0.80`。

项目新增文件和缓存总量保守按 **40 GB** 上限检查；模型复用 `~/.cache/huggingface`，不要在项目内重复下载，共享缓存总份额为 80 GB（2026-10-27 翻倍，2026-10-29 项目份额提到 40 GB）。preflight 不创建共享路径；`HF_HOME` 必须指向上述路径，hub 缓存变量只能指向其 `hub` 子目录；已有 `HUGGINGFACE_HUB_CACHE` / `TRANSFORMERS_CACHE` 若冲突同样拒绝，应在该次任务命令中改正或 unset。超过 5 GB 的下载要求剩余空间至少为 **50 GB + 下载大小**，并为预计项目新增量额外预留；对不同文件系统的检查采用保守余量。checkpoint 总保留数最多 **2**，常驻服务合计峰值内存最多 **16 GB**。这些计数和计划内存需如实申报，preflight 不自动删 checkpoint，也不对服务设置 cgroup 限额；运行者还需核对现有服务占用。所有 GB 采用 **1,000,000,000 bytes**，即项目 40,000,000,000、共享缓存 80,000,000,000、下载阈值 5,000,000,000、余量 50,000,000,000、内存 16,000,000,000 bytes；项目扫描包含已有环境/缓存，排除外部 symlink。

离线入口测试之外，控制器已实际运行默认只读 reaper，确认项目资源组不存在（[现场记录](docs/evidence/zero-budget-resource-check.md)），并经共享锁完成一次本机小矩阵 CUDA 正确性检查。未执行真实 Azure 删除、模型/性能或真实多机实验；默认离线测试不算真实云/GPU验收，累计费用保持 0 美元。

缓存默认路径遵循 Hugging Face 的优先顺序：显式 `HF_HOME`，否则 `$XDG_CACHE_HOME/huggingface`，两者均未设置时才是 `~/.cache/huggingface`。preflight 会报告实际生效路径，并检查该路径所在文件系统的余量；非共享路径会拒绝。空 `HF_HOME` 或在其未设置时的空 `XDG_CACHE_HOME` 都报错，不能当作 unset；显式有效 `HF_HOME` 会忽略 XDG 值。其他缓存覆盖变量若为空或冲突也须在该次命令中改正或 unset。

技术报告现有[异步 RL 草稿](reports/w2-async-grpo.md)和[推理草稿](reports/w3-disaggregated-serving.md)，[复现附录](reports/repro/README.md)说明数字溯源检查命令。校验器要求数字字面量与同一行引用的证据章节原值一致；通过检查不代表实验真实性或完整验收，草稿会保留尚未测量的项目。
