# 单卡异步 GRPO：小模型数学后训练的真实结果

本文报告在一张消费级 GPU 上跑通的 GRPO 后训练：rollout 引擎和 FSDP2 learner 分时共用一张卡，权重每步在显存里同步，样本带策略版本号并按有界陈旧度过滤。原计划的大模型、多机和代码奖励都没有做（原因见“局限”）。小留出集上的变化在抽样误差以内，并抓到了一个真实的奖励漏洞。[evidence: docs/evidence/rl-grpo.md#setup]
把同一 async 训练放到全留出集上之后，合并提升超过 2 SE。[evidence: docs/evidence/rl-grpo.md#update-phase-5-2026-10-28-full-held-out-set-passes-2-se]

## 方法

模型是 Qwen2.5-0.5B-Instruct。[evidence: docs/evidence/rl-grpo.md#setup] 奖励只看最后一个 `\boxed{}` 是否与参考答案等价（1/0），没有格式、长度或 EOS 项。[evidence: docs/evidence/rl-grpo.md#setup]

训练数据是 1,024 道 GSM8K 训练题；留出集是 200 道 GSM8K 测试题和 100 道 MATH-500 题，后者只从 verifier 能判分的 377/500 里抽取。[evidence: docs/evidence/rl-grpo.md#setup] 去重和训练/留出重叠都是 0。[evidence: docs/evidence/rl-grpo.md#setup]

每次运行 100 步，每步 8 道题、每题 group 8 个样本，即 6,400 个训练样本；AdamW 学习率 3e-6，clip 0.2，优势按组内标准差缩放，温度 1.0，最多 384 个新 token，seed 17。[evidence: docs/evidence/rl-grpo.md#setup]

引擎：进程内 vLLM 0.10.2 只占 15% 显存，learner 用 FSDP2（FP32 主权重、BF16 计算、冻结 BF16 参考模型）。[evidence: docs/evidence/rl-grpo.md#setup] 每步把全部策略参数以 BF16 拷进 vLLM，并清空前缀缓存，全程不经过磁盘。[evidence: docs/evidence/rl-grpo.md#results]

评测：pass@1 是每题 n=4 个样本的 c/n 平均（温度 0.6，top-p 0.95，最多 512 token），每题固定 seed，训练前后与各次运行完全相同；SE 按每题配对差计算。[evidence: docs/evidence/rl-grpo.md#setup]

对比的三组设置样本预算相同：async k=1 加 KL 0.001（基准）、sync k=0 加 KL 0.001、async k=1 加 KL 0。[evidence: docs/evidence/rl-grpo.md#results]

## 结果

### 留出集准确率

| 运行 | GSM8K 训练前→后（Δ ± SE，pp） | MATH-500 训练前→后（Δ ± SE，pp） | 训练 reward 前/后 10 步 | 证据 [evidence: docs/evidence/rl-grpo.md#results] |
|---|---|---|---|---|
| async k=1，KL 0.001 | 0.426 → 0.444（+1.8 ± 2.2） | 0.288 → 0.310（+2.3 ± 3.0） | 0.41 → 0.55 | [evidence: docs/evidence/rl-grpo.md#results] |
| sync k=0，KL 0.001 | 0.426 → 0.408（−1.9 ± 2.5） | 0.288 → 0.288（0.0 ± 2.6） | 0.40 → 0.59 | [evidence: docs/evidence/rl-grpo.md#results] |
| async k=1，KL 0 | 0.426 → 0.438（+1.1 ± 2.4） | 0.288 → 0.320（+3.3 ± 3.1） | 0.38 → 0.65 | [evidence: docs/evidence/rl-grpo.md#results] |

三次运行的训练 reward 都上升，但留出集 pass@1 的变化最多 +3.3 pp，且都在 1.3 SE 以内，不能和零区分。[evidence: docs/evidence/rl-grpo.md#task-11-math-grpo-on-qwen25-05b-instruct-one-rtx-4090--2026-10-27]

### 每步开销

| 指标（中位数 [最小, 最大]） | async k=1 | sync k=0 | async KL 0 | 证据 [evidence: docs/evidence/rl-grpo.md#results] |
|---|---|---|---|---|
| 每步耗时（s） | 5.93 [5.33, 8.84] | 5.84 [5.58, 6.41] | 5.74 [5.25, 6.45] | [evidence: docs/evidence/rl-grpo.md#results] |
| rollout / learner（s） | 1.84 / 2.95 | 1.91 / 2.82 | 1.85 / 2.76 | [evidence: docs/evidence/rl-grpo.md#results] |
| 权重同步（s） | 0.019 [0.018, 0.202] | 0.018 [0.017, 0.022] | 0.018 [0.017, 0.021] | [evidence: docs/evidence/rl-grpo.md#results] |
| GPU 空闲占比 | 0.18 [0.11, 0.22] | 0.18 [0.16, 0.23] | 0.19 [0.17, 0.23] | [evidence: docs/evidence/rl-grpo.md#results] |
| 陈旧丢弃数 | 0 | 0 | 0 | [evidence: docs/evidence/rl-grpo.md#results] |
| 整次运行（s） | 752 | 731 | 705 | [evidence: docs/evidence/rl-grpo.md#results] |

GPU 空闲占比是 rollout、learner 和同步之外的时间，主要花在 CPU 子进程里的数学判分；峰值 CUDA 分配 14.8 GB。[evidence: docs/evidence/rl-grpo.md#results] async k=1 时每批正好陈旧一个版本，在允许范围内，所以丢弃数为 0 是预期的。单卡串行时 async 不能让 rollout 和学习重叠，只改变数据的陈旧度，所以每步耗时和 sync 相同。[evidence: docs/evidence/rl-grpo.md#results]

### 抓到的奖励漏洞

sync k=0 那次学会了“写出 boxed 答案后继续生成、不停”：每 64 个样本里的截断数从 15.3 涨到 55.4，平均完成长度从 297 涨到 370 token。[evidence: docs/evidence/rl-grpo.md#reward-hacking-caught] 训练后 GSM8K 评测有 703/800 条达到 512-token 上限被截断，其中 685 条已经含 `\boxed{}`。[evidence: docs/evidence/rl-grpo.md#reward-hacking-caught] 奖励只判最后一个 box、从不惩罚续写，所以 pass@1 几乎不变，但输出不可用。两次 async 的截断反而下降（16.6 → 3.1 与 17.2 → 1.9，每 64 个）。[evidence: docs/evidence/rl-grpo.md#reward-hacking-caught] 每种设置只有一个 seed，不能据此说 sync 是原因。

### 加 EOS 惩罚、多 seed 与消融（第四阶段）

奖励改成：达到 max tokens 还没输出 EOS 的完成记 0 分（`--reward eos`），其余设置不变，每次仍是 6,400 个训练样本。[evidence: docs/evidence/rl-grpo.md#update-phase-4-2026-10-2728-eos-penalty-3-seeds-group-and-clip-ablations]

| 设置（都加 EOS 惩罚） | seed | GSM8K Δ（合并 ± SE，pp） | MATH-500 Δ（合并 ± SE，pp） | 证据 [evidence: docs/evidence/rl-grpo.md#update-phase-4-2026-10-2728-eos-penalty-3-seeds-group-and-clip-ablations] |
|---|---|---|---|---|
| async k=1，group 8，clip 0.2 | 17/18/19 | +3.3 ± 2.0 | +3.0 ± 2.2 | [evidence: docs/evidence/rl-grpo.md#update-phase-4-2026-10-2728-eos-penalty-3-seeds-group-and-clip-ablations] |
| sync k=0，group 8，clip 0.2 | 17/18/19 | +0.8 ± 1.8 | +0.75 ± 2.3 | [evidence: docs/evidence/rl-grpo.md#update-phase-4-2026-10-2728-eos-penalty-3-seeds-group-and-clip-ablations] |
| async k=1，group 4 / group 16 | 17 | +0.5 / +4.9 | +0.75 / +1.75 | [evidence: docs/evidence/rl-grpo.md#update-phase-4-2026-10-2728-eos-penalty-3-seeds-group-and-clip-ablations] |
| async k=1，clip 0.1 / clip 0.3 | 17 | +1.1 / +1.25 | +3.75 / +0.5 | [evidence: docs/evidence/rl-grpo.md#update-phase-4-2026-10-2728-eos-penalty-3-seeds-group-and-clip-ablations] |

漏洞消失了：加惩罚后 sync k=0 训练后的 GSM8K 评测截断只有 9 / 9 / 1 条（之前 703 条）。[evidence: docs/evidence/rl-grpo.md#update-phase-4-2026-10-2728-eos-penalty-3-seeds-group-and-clip-ablations] async 的 3 个 seed 在两个留出集上都提升，但合并后只有 1.6 和 1.4 SE，仍不能算显著。[evidence: docs/evidence/rl-grpo.md#update-phase-4-2026-10-2728-eos-penalty-3-seeds-group-and-clip-ablations] 消融都只有一个 seed，参考设置自己的 seed 间就差 +2.0 到 +4.75 pp，所以不能说哪个 group 或 clip 更好。[evidence: docs/evidence/rl-grpo.md#update-phase-4-2026-10-2728-eos-penalty-3-seeds-group-and-clip-ablations]

### 全留出集（第五阶段）

同一 async k=1、EOS 惩罚、group 8、三个 seed，评测改成全部 1,319 道 GSM8K 测试题和全部 377 道可判分 MATH-500 题。合并后 GSM8K +3.71 ± 0.67 pp（5.51 SE），MATH-500 +3.65 ± 0.98 pp（3.72 SE），两个都超过预先规定的 2 SE，三个 seed 都为正。[evidence: docs/evidence/rl-grpo.md#update-phase-5-2026-10-28-full-held-out-set-passes-2-se] group 16 同样三个 seed，GSM8K +4.45 ± 0.71 pp（6.27 SE），MATH-500 +2.61 ± 0.92 pp（2.82 SE），也过 2 SE；训练 reward 几乎不涨。[evidence: docs/evidence/rl-grpo.md#update-phase-5-2026-10-28-full-held-out-set-passes-2-se]

### verl 对照（第七阶段）

第四阶段的同步、EOS 惩罚、seed 17 这一组，在单独的 verl 0.9.1 环境里重跑。模型、800 条训练 prompt 的顺序、奖励、100 步 × 8 prompts × 8 samples 都相同，两边训练后的模型用同一个评测脚本打分。[evidence: docs/evidence/t11-verl/README.md#setup] 剩下的差异（rollout seed、GRPO 标准差的 Bessel 校正、两个进程等）列在运行脚本里。[evidence: docs/evidence/t11-verl/README.md#differences-that-remain]

| | 我们的 grpo_run | verl 0.9.1 | 证据 [evidence: docs/evidence/t11-verl/README.md#results] |
|---|---|---|---|
| GSM8K pass@1 | 0.4475 | 0.44 | [evidence: docs/evidence/t11-verl/README.md#results] |
| MATH-500 pass@1 | 0.295 | 0.34 | [evidence: docs/evidence/t11-verl/README.md#results] |
| 每秒样本 | 11.07 | 4.51 | [evidence: docs/evidence/t11-verl/README.md#results] |
| 墙钟（s） | 698.1（含评测） | 1572（不含评测） | [evidence: docs/evidence/t11-verl/README.md#results] |

GSM8K 两边一致；MATH-500 上 verl 高 4.5 ± 2.47 pp，不到 2 SE，而且每边只有一个 seed，不能说明 verl 训得更好。[evidence: docs/evidence/t11-verl/README.md#results] verl 每个样本慢 2.45×，主要花在 fp32 参考模型前向和没有 remove-padding 的 SDPA 上。[evidence: docs/evidence/t11-verl/README.md#results] 峰值显存口径不同：我们是单进程 torch 分配器的 14,794,147,328 bytes，verl 是 nvidia-smi 全卡的 16,423 MiB。[evidence: docs/evidence/t11-verl/README.md#results]

### 自定义 log-prob 算子

融合分块 log-prob 算子注册为 torch.library custom op。隔离测量时，前向加反向的增量峰值显存从 119,538,688 bytes 降到 39,847,936 bytes，这包含必需的稠密梯度。[evidence: docs/evidence/fused-logprob.md#isolated-vocabulary-memory-result] 放进整个 learner 后，峰值都是 10,020,235,776 bytes，没有下降。[evidence: docs/evidence/fused-logprob.md#actual-learner-matched-measurements] 每步耗时中位数 234.800 ms 对 228.713 ms，两者的范围重叠，不能证明加速。[evidence: docs/evidence/fused-logprob.md#actual-learner-matched-measurements]

权重同步的数值确认比较了 12 行、1,823,232 个词表坐标，落在预先固定的误差界内。[evidence: docs/evidence/weight-sync.md#independent-numerical-confirmation--accepted-scoped-gate]

## 局限

- 第四阶段的 200/100 留出集不到 2 SE；第五阶段全留出集（1,319 / 377）上 group 8 和 group 16 都过了 2 SE。clip 消融仍只有第四阶段的单个 seed。 [evidence: docs/evidence/rl-grpo.md#update-phase-5-2026-10-28-full-held-out-set-passes-2-se]
- verl 对照只有一个 seed、一组设置，留出集也小，MATH-500 只评了其中一部分题。[evidence: docs/evidence/t11-verl/README.md#results]
- 代码奖励需要可用的沙箱（rootless gVisor 在本机失败，Warden 不可用），所以只做了数学。
- MATH-500 只用了 verifier 能判分的子集，这是公开的选择偏差。[evidence: docs/evidence/rl-grpo.md#not-done--limits]
- 单卡：多卡 NCCL 同步、原计划的大模型和多机部分都没有做，未在真实多机上验证。GPU 和另一个项目共用。
- 下一步：加大留出集或训练步数，争取 2 SE 以上的结论；消融补到多 seed。 [evidence: docs/evidence/rl-grpo.md#update-phase-4-2026-10-2728-eos-penalty-3-seeds-group-and-clip-ablations]

## 复现

运行命令、提交、模型修订、环境哈希和 seed 见[复现附录](repro/README.md)。原始输出在 `docs/evidence/rl-grpo/`：每次运行的 `steps.jsonl`、`summary.json`，以及训练前后每条评测样本的 `eval-*.jsonl.gz`。
