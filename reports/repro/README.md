# 报告复现附录

两篇报告（单卡 GRPO、单卡推理服务）里的数字都来自下面这些单卡运行。每个提交同时包含驱动代码和原始输出；运行本身没有记录提交号。[evidence: docs/evidence/report-numbers.md#single-gpu-run-identities]

## 环境

- 硬件与运行时：一张 NVIDIA GeForce RTX 4090（和另一个项目共用），driver 595.99.02、CUDA 12.8、Torch 2.8.0、Triton 3.4.0、Transformers 4.55.4、vLLM 0.10.2、Python 3.12.13、Go 1.27.3。[evidence: docs/evidence/report-numbers.md#single-gpu-run-identities]
- 本机原生执行，没有容器，所以没有 image digest；用 `scripts/env_identity.py` 生成的环境哈希代替。[evidence: docs/evidence/report-numbers.md#single-gpu-run-identities]
- `.venv-ml`：146 distributions，sha256 `e4feddb69cdcb3572341a4edacd14410bb3f4e6053329ecb3c1924db774b8bef`。[evidence: docs/evidence/report-numbers.md#single-gpu-run-identities]
- `.venv`：10 distributions，sha256 `71a9efac2614738fe2f110a67bd0e8854a71a1bbdb660ac6d8cb40fa8a701e04`。[evidence: docs/evidence/report-numbers.md#single-gpu-run-identities]
- 环境哈希是在运行之后采集的，期间没有安装或升级任何包。[evidence: docs/evidence/report-numbers.md#single-gpu-run-identities]
- 没有 OFED、RDMA 或多节点身份：多机部分未在真实多机上验证。

## 模型

- `Qwen/Qwen2.5-0.5B-Instruct` rev `7ae557604adf67be50417f59c2c2f167def9a775`（GRPO 报告）。[evidence: docs/evidence/report-numbers.md#single-gpu-run-identities]
- `Qwen/Qwen3-8B-AWQ` rev `4da05a8edb55c6046cce958586c33b61da07bb79`（推理报告的基线）。[evidence: docs/evidence/report-numbers.md#single-gpu-run-identities]
- `Qwen/Qwen3-8B` rev `b968826d9c46dd6066d109eabc6255188de91218`（推理报告的精度对照）。[evidence: docs/evidence/report-numbers.md#single-gpu-run-identities]

## 运行、提交与 seed

| 任务 | 提交 | 命令 | seed |
|---|---|---|---|
| T12 基线 | `bc749482dfc1a4a92a01c79cdf20165784cb6349` | `bash scripts/run_serving_baseline_gpu.sh azure\|burst <out>` | 17 [evidence: docs/evidence/report-numbers.md#single-gpu-run-identities] |
| T11 GRPO | `b49326bee0f7e0afd17f46e84995e1bf50d4dba4` | `bash scripts/run_grpo_gpu.sh <out> [--staleness 0\|1 --kl X]` | 17 [evidence: docs/evidence/report-numbers.md#single-gpu-run-identities] |
| T13 精度对照 | `4f834c7babd964bfe6588276616549afd3635219` | `bash scripts/run_fp8_compare_gpu.sh <out> [--quality-only]` | 17 [evidence: docs/evidence/report-numbers.md#single-gpu-run-identities] |

- GPU 作业都经过 `scripts/with_gpu.sh`（flock 共享锁），一次只跑一个作业。
- 服务端参数记录在每次运行的 `summary.json` / `session.json` 里：`--gpu-memory-utilization 0.8 --max-model-len 8192 --max-num-seqs 64 --seed 17`。[evidence: docs/evidence/report-numbers.md#single-gpu-run-identities]
- GRPO 参数在 `rl-grpo/<run>/summary.json` 的 `args` 字段里。[evidence: docs/evidence/report-numbers.md#single-gpu-run-identities]
- T11 的数据准备：`python -m rl.local.prepare_data --root .cache/datasets --output .cache/t11/data.json`。[evidence: docs/evidence/report-numbers.md#single-gpu-run-identities] 数据集哈希在 `docs/evidence/rl-grpo/data-identity.json` 里。
- 自定义算子和权重同步的历史运行记录在[算子证据](../../docs/evidence/fused-logprob.md#reproduction-and-environment)和[同步证据](../../docs/evidence/weight-sync.md#reproduction-and-limits)里。

## 检查与 PDF

数字检查命令（在项目根目录运行）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B scripts/check_report_numbers.py reports/w2-async-grpo.md reports/w3-disaggregated-serving.md reports/repro/README.md # [evidence: docs/evidence/report-numbers.md#checker-command]
```

本地 PDF（文字版，可选中复制）：Markdown 先由 `scripts/build_report_html.py` 转成 HTML，再用 headless Chrome 的 `--print-to-pdf` 打印；每次用项目 `.cache/` 下新建的 `--user-data-dir` 并加 `--no-first-run`，不碰桌面用户的 Chrome 配置。脚本最后用 `scripts/check_pdf_text.py` 确认 PDF 里能提取文字，并且报告里的每个数字字面值都出现在 PDF 文字里。

```bash
bash scripts/build_report_pdfs.sh
```

之前的 Pillow 图片页渲染器 `scripts/build_report_pdf.py` 还在，但已不再用于 `reports/pdf/`。

## 确定的文本范围与限制

每个物理行的 ASCII 数字字面值必须由同一行的 `[evidence: docs/evidence/report-numbers.md#checker-runtime]` 这种精确格式（替换文件与锚点） 覆盖，且原样出现在该锚定章节。正负小数、科学计数、百分数、分组数字，以及含数字的模型名、版本、配置和提交标识都检查；不进行数值等价、单位转换或自行舍入。独立及逗号分组的有符号数字可带 ASCII 单位后缀，整个字面值（含符号、百分号和单位）不可拆分匹配。句末句号不进入字面值；点后还有 ASCII 字母、数字或下划线段时保留为连续版本/标识的一部分，不能靠回退丢掉符号或拆开分组数字。数字连续标识保留点与连字符；独立数字优先，但不能从标识内部抽取数字。每行相同字面值去重计数；多个引用可联合覆盖该行各字面值。

仅引用标记、链接目的地址、实际 Markdown 列表标记和围栏语法不是事实数字；可见链接文字、行内代码、围栏代码/配置正文和表格都检查。行内代码按完全相同长度的反引号串闭合，较短内嵌串不结束代码；反引号围栏开启行的信息串不能含反引号，同一行闭合的长反引号代码跨度仍检查正文；无法闭合的串保守保留为字面值。链接使用平衡括号的行内目的地址、自动链接或普通引用定义；无法识别的复杂语法保守保留检查。缩进代码不做额外解析，沿用物理行规则。证据使用完全相同的过滤与扫描，链接地址或引用中的数字不能充当支持值。

仅使用 ATX 标题：去掉强调/代码标记，小写，移除非字母数字/下划线/连字符/空白，再把每个空白变成连字符；同名重复锚点不自动编号，直接拒绝。章节包括子标题，止于下一个同级或更高级标题；围栏内的标题不划分章节。只有项目证据目录内的普通文件可引用；拒绝路径遍历、逃逸符号链接和证据根目录重定向。所有引用都验证，即使该行没有数字。

缺失、空输入或任何没有可检查数字的报告都失败。非 ASCII Unicode 数字类别的字符（含全角、阿拉伯数字、上标和罗马数字字形）显式失败；自然语言数量词和事实真伪不属于语义检查范围。严格范围可能要求改写复杂排版或补同一行引用；应改报告表达与来源，不能通过省略真实声明或降低科学验收绕过。

通过只证明字面值与锚点一致，不证明数据真实、来源可信、因果关系或实验验收。中文无数字事实可写，但仍由人工审查；后续真实实验必须保留原始失败、完整计划与源身份。
