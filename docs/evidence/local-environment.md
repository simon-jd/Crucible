# 本地开发环境

## 范围

仅为开发环境盘点，不是集群验收。原始命令及退出码见 [raw/local-environment.json](raw/local-environment.json)。

系统 Python 为 3.14.4；项目使用已有 CPython 3.12.13 创建 .venv。Python 开发依赖以 uv.lock 固定；缓存、虚拟环境和后续 Go 工具链均位于本项目内。安装入口为 `bash scripts/bootstrap.sh`。

沙箱内 nvidia-smi 无法连接 NVIDIA 驱动；这不代表物理主机驱动不可用。用户新增本机 GPU 约束后，经允许的主机只读查询确认 RTX 4090 可用（见下面更新）。Docker daemon 可查询，但已配置运行时中没有 runsc；尚不能执行 gVisor 代码奖励集成测试。未发现 Slurm、Kubernetes CLI、DCGM、IB perftest 或 Nsight 命令。

## 开发工具来源

Go 工具链按 [官方下载元数据](https://go.dev/dl/?mode=json) 的版本及 SHA256 写入 tools.lock.json；安装器校验摘要后仅解压至 .tools。实际安装结果后续追加，不把锁文件本身当作运行成功证据。

本次执行 `python3 scripts/install_go.py` 成功核对摘要并安装，随后 `.tools/go/bin/go version` 输出 `go version go1.27.3 linux/amd64`。此工具链用于后续 Go 回放器、路由和扩缩容测试。

## 用户资源更新后的主机核验

2026-10-23，按用户指定 RTX 4090 范围，通过批准的沙箱外只读 `nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,driver_version --format=csv,noheader` 查询成功：RTX 4090、总显存 24564 MiB、驱动 595.99.02。实际 CSV 见 [raw/host-gpu.csv](raw/host-gpu.csv)。显存使用量随其他作业变化，不能当作本项目容量预留。

此查询不是 GPU 性能测试，没有启动或停止 GPU 工作负载。后续 GPU 实验需通过 `/var/tmp/rtx4090.lock` 排队，与 Keel 共享，单次最多四小时；不得更改驱动、停止其他 GPU 进程或连接远程 GPU。H100 多机候选基线仍不适用于此主机。
