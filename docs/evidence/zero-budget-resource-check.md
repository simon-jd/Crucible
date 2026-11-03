# 零美元约束后的只读资源核验

日期：2026-10-23。证据类型：实际 Azure 控制面只读查询；不是资源部署验收。

按用户新增预算规则执行：

```text
AZURE_CORE_COLLECT_TELEMETRY=0 AZURE_LOGGING_ENABLE_LOG_FILE=0 az group exists --name rg-crucible-lab --output tsv --only-show-errors
exit code: 0
stdout: false
```

该时点 Crucible 专属资源组不存在，未创建、修改或删除资源，没有连接远程 GPU 节点，没有访问受保护资源组 rg-mlinfra-westus。本项目累计估算费用仍为零美元。之后会话结束检查仍需通过项目 reaper 入口重新核对，不能用本次结果证明未来状态。

资源入口实现后，控制器实际执行默认只读 `bash scripts/azure_reaper.sh`（未带 --apply），exit 0；结果为 scope=rg-crucible-lab、mode=plan、event=absent、action=none。原始 JSONL 见 [raw/azure-reaper-initial.json](raw/azure-reaper-initial.json)。固定订阅查询成功，未产生删除调用；这只验证只读入口，不构成真实删除或 GPU 验收。

本次会话结束检查再次执行相同默认只读入口，exit 0，仍为 absent/action=none；原始 JSONL 见 [raw/azure-reaper-end.json](raw/azure-reaper-end.json)。费用台账保持零美元，无实际删除。
