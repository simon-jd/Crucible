# 本机 PyTorch 运行环境

日期：2026-10-23。证据类型：真实 CPU 与本机 RTX 4090 正确性检查。未下载模型，未进行训练、serving 或性能比较；不替代后续 custom op 和模型任务验收。

## 可复现配置

`make bootstrap-ml` 先准备基础环境，再以 uv.lock 安装独立 `.venv-ml`。ml extra 固定 NumPy 2.2.6、PyTorch 2.8.0，间接锁定 Triton 3.4.0、CUDA 12.8 库和 NCCL 2.27.3；全部 wheel URL/SHA256 位于 uv.lock。基础 `.venv` 不安装这些 GPU 依赖。

PyTorch 版本选择参考 [vLLM 0.10.2 元数据](https://pypi.org/pypi/vllm/0.10.2/json) 与 [SGLang 0.5.3.post3 元数据](https://pypi.org/pypi/sglang/0.5.3.post3/json) 共同声明的 torch==2.8.0。这里只为后续代码提供候选公共 ABI；两个 serving 引擎尚未安装或验证，未来实现时仍需锁定完整兼容依赖。

安装前资源 preflight 通过，保守预留环境增长至十进制 10 GB、下载 6 GB；重跑时扣除既有环境大小，避免重复计数。临时 wheel 缓存不持久保存，UV_LINK_MODE=copy 避免跨文件系统 hardlink 警告。最终 `.venv-ml` 约 6.4 GiB，项目总字节数及余量见 [resource-post-install.json](ml-runtime/resource-post-install.json)，低于十进制 14 GB。无模型、checkpoint 或常驻服务。

## 实际命令与结果

- `bash scripts/bootstrap_ml.sh`：exit 0；再次执行为幂等检查，依赖无变化，日志无警告。见 [bootstrap.log](ml-runtime/bootstrap.log)。首次导入发现 NumPy 缺失警告，已补固定 NumPy 后重新验证，未将警告隐藏。
- `.venv-ml/bin/python scripts/check_torch_runtime.py`：exit 0，CPU FP32 前向/反向正确。见 [cpu-smoke.json](ml-runtime/cpu-smoke.json)。
- `bash scripts/with_gpu.sh -- .venv-ml/bin/python scripts/check_torch_runtime.py --cuda`：在批准的主机入口执行，固定 `/var/tmp/rtx4090.lock` 排队，exit 0。仅执行 16×16 BF16 矩阵乘及梯度校验，实际 RTX 4090、SM 8.9、驱动 595.99.02；PyTorch build CUDA 12.8、NCCL 2.27.3。见 [cuda-smoke.json](ml-runtime/cuda-smoke.json)。进程正常结束，无推理服务、模型或其他常驻任务。
- `ruff check scripts/check_torch_runtime.py` 与 `bash -n scripts/bootstrap_ml.sh`：通过。

实际主机 CUDA 命令在上述 GPU 入口前带有以下单次环境前缀，输出写入 cuda-smoke.json；它只修正当前命令的缓存配置，不修改用户全局设置：

```bash
env -u HUGGINGFACE_HUB_CACHE -u TRANSFORMERS_CACHE \
  HF_HOME="$HOME/.cache/huggingface" HF_HUB_CACHE="$HOME/.cache/huggingface/hub" \
  bash scripts/with_gpu.sh -- .venv-ml/bin/python scripts/check_torch_runtime.py --cuda
```

模型字段为 null，performance_claim=false。没有推理吞吐、训练质量、融合 kernel 性能或多机结论；**未在真实多机上验证**。本次没有租卡、访问远程 GPU、变更驱动或干预其他 GPU 进程，新增费用估算为零美元。

实现 `eee8ecd` 已通过 独立审查：spec compliant / quality approved，无 Critical、Important 或 Minor。审查仅核对代码、锁文件、安装元数据及已有证据，没有再次运行 GPU 或安装依赖。
