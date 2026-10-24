---
tags: [规划, 长期参考, 算法, scope, Laya]
---

# Jev 调参 · Laya multilingual 322M scope 微调（长期参考）

> 2026-10-23 首次撰写。本文是“Jev 风格本地判别器”用于 scope 分类的专项调参计划；
> 方向决策和早期 Jev 调研仍见 `2026-10-14-方向决策-本地LLM·微信·Jev.md`。

## 0. 决策摘要

| 事项 | 决定 |
|---|---|
| 当前基座 | `convaiinnovations/laya-multilingual`（mmBERT-base，322M） |
| 本地位置 | `.models/jev`，权重不入 Git |
| 模型版本 | revision `1720e3e3357cfe1e281542e223f8273b0890ca34`，SHA256 `9d628fd9...f204` |
| 运行时 | `laya==0.4.13` |
| 第一任务 | 只微调 scope 判别，不同时改检索、生成和引用校验 |
| 输出 | `in_scope` / `out_of_scope` / 低置信回退 LLM |
| 对外契约 | 保持 v2 不变；前端继续只依赖 `hotspot.api` |
| 上线策略 | zero-shot baseline → 微调 → 校准 → 影子 → 小流量 → 默认 |

旧 openJev 包已实测不适合当前任务：17 条样本中范围内召回 0/13，
短标签、英文问题、关闭校准均无法改善；旧权重和源码已删除，只保留两份
`scope_probe*.json`。当前 Laya 只证明了可加载和可推理，尚未证明零样本质量。

## 1. 目标与非目标

**目标**：在现有 `AskRequest` 契约不变的前提下，用本地 322M 判别模型替代大部分
高延迟的 LLM scope 调用，同时控制范围内误拒和 OOD 误放行。

**非目标**：

- 不同时微调检索、答案生成和引用校验；
- 不让模型生成自然语言；
- 不改变前端已经解析的年份、会议、来源和子领域硬过滤；
- 不把“低置信”当作“范围外”；低置信必须交给 LLM fallback。

scope 是第一优先任务，因为它在工具调用之前、输入短、输出只有离散类别，且当前
30 秒网络超时会直接阻塞整条链路。

## 2. 固定基线

训练前冻结三类基线，后续所有增益都相对它们计算：

| 基线 | 配置 | 用途 |
|---|---|---|
| `llm_default` | 当前 flash 模型 + `SCOPE_INSTRUCTIONS` | 生产基线，记录误拒、OOD 召回、P50/P95 和超时 |
| `laya_zero_shot` | 本地 Laya multilingual，不训练、不校准 | 判断微调是否真有收益 |
| `laya_choice_prompt_v1` | 固定两条选项描述和选项顺序 | 作为训练与推理的唯一 prompt v1 |

现有 16 题验收集和其他完整回归集只作测试，不参与训练或阈值选择。每次实验记录：

```text
model_revision / dataset_hash / prompt_version / temperature /
threshold / git_commit
```

## 3. 数据设计

首版只训练一个 `choice` 问题：

```text
question: 这条用户提问是否属于本地 AI/ML 论文库的研究范围？
choices:
  in_scope: 询问 AI/ML/RL/LLM/Agent/机器人学习/决策及相关方法；
            跨领域问题只有明确询问其中 ML/RL 方法时属于此类。
  out_of_scope: 询问医学、生物、物理、材料、土木、气候、航天等领域本身；
                或与本地 AI/ML 论文库无关。
```

`intent`（trend/fact/compare/relation/nonsense/damage 等）继续由独立模块处理，
不混入 scope 标签。年份、会议、来源和子领域约束由前端 `scope` 字段处理。

数据字段至少包含：

```text
id / question / label / domain / boundary_cue / language /
source / split / annotator / rationale / dataset_version
```

| 数据 | 目标 | 注意 |
|---|---|---|
| 范围内正例 | 从 22,747 篇论文的标题/摘要生成事实、比较、关系、趋势问法 | 不从题目直接抄整句；覆盖中英混合和缩写 |
| 明确 OOD | 医学、生物、量子、材料、土木、气候、航天等域内问题 | 避免只靠领域名词，补泛化问法 |
| 跨域边界正例 | “强化学习用于交通控制”“AI 辅助医学影像”等 | 必须明确询问 ML/RL 方法 |
| 跨域边界负例 | “医学影像有什么热点”“桥梁抗震最新研究” | 只问应用领域本身 |
| 困难负例 | `CT`、`SAC`、`policy` 等缩写/多义词、泛词、噪声问题 | 防止关键词式误判 |

首轮建议 600–1,000 条做 pilot（正例约 50%、OOD 约 40%、边界约 10%），
确认流程后扩到 3,000–5,000 条。按语义主题/来源做分组切分，不做随机句级切分：
`train 70% / calibration 15% / test 15%`。中文为主，加入英文和混合语言切片；
边界样本单独分层，不能只抽总平均。

标注采用“规则/论文元数据生成 silver 标签 + 分层人工审核”。优先审核全部边界题、
模型分歧题和每类随机样本；人工只裁决定义冲突，不逐题调规则。测试集冻结后，
任何 prompt、阈值和数据修正都不得回看测试结果迭代。

## 4. 训练路线

采用逐步解冻，避免 322M 模型一上来全参数训练：

| 阶段 | 做法 | 目的 | 通过条件 |
|---|---|---|---|
| T0 | 固定数据与 prompt，跑 `laya_zero_shot` | 知道预训练模型真实下限 | 三类指标和延迟都有记录 |
| T1 | 冻结 encoder，只训练 2 层 decision head | 低成本验证标签与输入格式 | calibration 优于 zero-shot |
| T2 | 解冻最后 2–4 层，或加 LoRA（运行库支持时） | 学习中文领域词汇和边界 | 范围内误拒、OOD 误放行同时下降 |
| T3 | 视收益决定全参数微调 | 追求最终精度 | 训练成本可承受，相对 T2 有稳定增益 |
| T4 | 可选 RLCD/GRPO | 优化 proper scoring rule 和选择性分类 | 仅在校准后 coverage-risk 优于 T2/T3 时保留 |

建议从短输入 `max_len=256` 开始，确认收益后再比较 512；正式推理默认保留到 1024，
避免长问题被截断。超参先取小范围搜索，不在测试集上固定：

| 参数 | 初始范围 |
|---|---|
| batch size | 1–8，按显存/内存调整 |
| gradient accumulation | 16–64 |
| learning rate | head `1e-4` 量级；解冻 encoder `1e-5`–`3e-5` |
| epochs | 2–4，早停看 calibration NLL |
| weight decay | 0.01 |
| 精度 | GPU 优先 bf16/fp16；CPU 用 fp32 |
| seeds | 至少 3 个，报告均值和标准差 |

模型选择看 **calibration split 的 NLL/Brier + 分组错误率**，不看单一 accuracy。
训练产物放 `.models/jev/ft/scope-v1/`（不入 Git），同时提交模型卡：
基座 revision、数据哈希、超参、指标、失败切片和回滚命令。

## 5. 校准与路由门控

Laya multilingual 出厂温度全为 1.0，不能直接相信其概率。训练后只用校准集拟合
scope 的单一温度（`choice`、2 个选项），测试集只做一次最终评估。

运行时采用三路结果，不是简单的“概率最高就采纳”：

```text
高置信 in_scope    → 本地直接进入 agent
高置信 out_of_scope → 直接拒答
其余/异常/超时      → 回退现有 LLM scope classifier
```

门控参数由 coverage-risk 曲线选出，至少同时约束：

- 范围内误拒率不高于 `llm_default`；
- OOD 误放行率不高于 `llm_default`；
- 本地可接管覆盖率足够高，能显著降低 30 秒超时；
- 两个语言切片和跨域边界切片都不能靠多数类掩盖。

初版验收目标暂定为：范围内误拒 `<=2%`、OOD 误放行 `<=5%`、高置信本地覆盖率
`>=60%`；最终值必须先测 `llm_default`，再冻结，不在测试题上逐题调。

## 6. 推理、超时与接口

模型必须由服务启动时预加载并做一次 warm-up，不能在首请求懒加载：当前本机 CPU
冷加载约 32.5 秒，热推理约 1.05 秒；冷加载如果落在请求内会直接耗尽 scope 预算。

内部配置建议：

```yaml
scope:
  backend: llm          # llm | laya_shadow | laya_gate
  model_path: .models/jev
  device: cpu
  preload: true
  min_confidence: null  # 校准后写入
  fallback_backend: llm
```

接口红线：

- 对外仍只有 `hotspot.api`；前端不 import Laya；
- `AskRequest` 和 `ScopeDecision` 的现有语义不因微调改变；
- 先只在 `meta` 中增加可选的 `scope_backend / scope_confidence /
  scope_fallback` 内部观测字段；
- 本地模型超时、加载失败或输出非法时，错误码仍是 `scope_timeout` /
  `scope_failed`，不得误标成 `out_of_scope`；
- 服务端保持无会话状态，模型实例只读共享。

## 7. 实施步骤与协作顺序

| 步骤 | 交付物 | 前置 | 验收 |
|---|---|---|---|
| D0 基线冻结 | `eval/scope_v1` 数据 schema、测试集、三基线报告 | 确认现有验收集不参与训练 | 可一键复现指标 |
| D1 数据构造 | train/calibration/test JSONL、数据哈希、边界题审计表 | D0 | 每类有足够样本，切分无泄漏 |
| D2 评测/训练骨架 | `scripts/train_scope_laya.py`、`scripts/calibrate_scope_laya.py`、`scripts/eval_scope_classifier.py` | D0 | 固定输入可复现 |
| D3a head-only | scope checkpoint + 训练日志 | D1、D2 | calibration 优于 zero-shot |
| D3b 解冻/LoRA | 第二个候选 checkpoint | D1、D2；可与 D3a 并行 | 分组指标稳定提升 |
| D4 校准门控 | temperature、阈值、coverage-risk 报告 | D3a/D3b | 达标且无语言/边界短板 |
| D5 影子运行 | `laya_shadow` adapter、Trace 字段 | D4 | 不影响线上结果，只记录本地与 LLM 分歧 |
| D6 小流量切换 | `laya_gate`、回滚开关 | D5 | 超时显著下降，误拒/OOD 不退化 |
| D7 默认切换 | 配置默认值与模型卡 | D6 稳定观察 | 一条配置即可回退 `llm` |

并行安排：

- 两名算法开发者可在 D1 后并行：一人做数据与 head-only，一人做评测/校准和
  fallback adapter；
- D2 的评测骨架在 D1 定下 schema 后立即并行开发，不必等训练完成；
- 前端继续按 v2 契约开发账号、会话和展示，不需要等待模型训练；只有 D6 若新增
  `meta` 观测字段，才需要前端做 additive 展示适配；
- 训练候选之间可并行，但校准阈值和最终测试必须串行，避免在同一测试集反复调参。

## 8. 最终验收与回滚

| 维度 | 必报指标 |
|---|---|
| 分类 | 范围内召回/误拒率、OOD 召回/误放行率、混淆矩阵 |
| 选择性分类 | 覆盖率、accepted-region accuracy、fallback 率、ECE/Brier |
| 分组 | 中文/英文/混合语言、跨域边界、缩写与多义词、长问题 |
| 稳定性 | 同输入重复一致性、选项顺序置换稳定性 |
| 性能 | 冷加载、热 P50/P95、内存、失败率 |
| 端到端 | `scope_timeout` 数、总时延、拒答正确率、现有验收集不退化 |

上线顺序只能是 `llm_default → laya_shadow → 10% canary → 50% → 100%`。
任一阶段出现范围内误拒上升、OOD 误放行上升或本地推理失败率异常，立即把
`scope.backend` 切回 `llm`；模型文件保留，不通过删权重回滚。

## 相关

- `2026-10-14-方向决策-本地LLM·微信·Jev.md`
- `2026-10-18-接口契约-前端与算法.md`
- `2026-10-21-检索智能体地基与演进方向.md`
- `2026-10-22-工具成熟度与离线基准.md`
