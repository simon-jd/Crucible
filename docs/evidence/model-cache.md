# W2 共享模型缓存准备

日期：2026-10-23。按资源约束选择 [Qwen2.5-0.5B-Instruct 官方仓库](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，固定 revision `7ae557604adf67be50417f59c2c2f167def9a775`，许可证 Apache-2.0、公开下载无需新增凭证。此次只下载并校验文件，**尚未加载模型或验证推理/训练**。

下载前 preflight：项目预留新增 10 MB、下载 1.1 GB、checkpoint=0、常驻服务=0，通过。使用临时 `huggingface-hub==0.36.0` 客户端，exit 0；不改变 `.venv`/`.venv-ml` 或全局环境配置：

```bash
env -u HUGGINGFACE_HUB_CACHE -u TRANSFORMERS_CACHE \
  HF_HOME="$HOME/.cache/huggingface" HF_HUB_CACHE="$HOME/.cache/huggingface/hub" \
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 \
  UV_CACHE_DIR="$PWD/.cache/uv" UV_NO_CACHE=1 \
  uvx --python "$PWD/.venv/bin/python" --from huggingface-hub==0.36.0 \
  hf download Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --include config.json generation_config.json merges.txt model.safetensors \
    tokenizer.json tokenizer_config.json vocab.json LICENSE README.md
```

九个文件共 999,602,607 字节，位于授权的 `~/.cache/huggingface`，不在项目内重复复制；核验时共享缓存唯一文件字节数同为该值，低于 40 GB。权重 SHA256 与固定 revision 的官方 LFS 元数据一致；其它文件按 Git blob ID 核对，另记录每个文件的 SHA256。见 [完整清单](models/qwen25-05b.json)。未来 Task 7 必须复用该 revision；客户端、tokenizer、模型加载、policy version 与实际生成仍需对应任务验证。

本次未使用 GPU、模型 API、远程节点或推理服务，费用估算 0 美元。
