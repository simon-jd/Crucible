# 公共交替实验汇总门禁

日期：2026-10-23；证据类型：本地 CPU 单元测试。代码：scripts/experiment_results.py；测试：scripts/tests/test_experiment_results.py。

## 验证

- RED：`.venv/bin/python -m pytest scripts/tests/test_experiment_results.py -q`，模块未实现，collection error（ModuleNotFoundError）。原输出见 [tdd-red.txt](experiment-summary/tdd-red.txt)。
- GREEN：相同命令，15 passed in 0.01s；ruff 检查上述两个文件通过。
- 测试仅使用明确的 fixture 数据，没有测量任何真实 CPU/GPU 性能。
- 独立审查通过 spec/quality，发现偶数轮极大值的中位数可能溢出。新增回归测试先确认 `inf != 1e308` 失败，再改用两个中间值的精确分数平均；16 项聚焦测试通过，ruff 通过，聚焦复审 Approved，无剩余发现。修复提交 `8cdeb17`。

## 行为与使用

`.venv/bin/python scripts/experiment_results.py /path/to/records.json` 校验按时间排列的 baseline/candidate JSON 记录；每轮有 round、variant、conditions、valid、fp8、metrics 字段。至少三个完整交替轮次，每个轮次的相同条件、指标集合及各 variant 的 FP8 配置保持一致，输出每项指标的 median/min/max。作废轮次、非有限数、FP8 缺 accuracy_delta 等均拒绝汇总。

conditions 必须由实验入口填入所有保持相同的条件；accuracy_delta 必须来自留出集。该工具只校验记录结构和比较约束，无法证明输入数据真实，也不替代原始日志、硬件清单、模型 revision 或质量评测。调用方仍负责完整记录这些来源。后续训练与 serving 实验应复用此门禁或提供覆盖相同规则的验证。
