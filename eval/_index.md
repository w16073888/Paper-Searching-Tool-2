---
tags: [索引]
---

# eval · `_index`

## 1. 这个文件夹是什么

问题集和工具级 fixture。旧 G0–G3 执行脚本及不完整 retrieval benchmark 已删除。

## 2. 当前工作状态

- `questions*.json` 保留为 v2 行为用例来源。
- `tool_fixtures/` 保留 evidence / verify 的确定性测试数据。
- 新工具基准与评测执行器待 v2 工具接口确定后重建。
- `scope_v1/` 保存冻结的 scope 测试集与双基线评测结果；不参与训练或阈值选择。

## 3. 文件清单

**计划文件**：本 `_index.md`。
**数据文件**：问题集与 fixture（脚本读取，不整篇进上下文）。
执行入口在 `scripts/run_v2_eval.py`。

- `v2_acceptance_16.json`：local / compare / relation / OOD 共 16 题。
- `model_selection_v1.json`：24 条冻结模型选型题，用于 8B / 14B / 可选云 API 对照。
- `model_selection_v1.manifest.json`：选型集状态、哈希、禁止调优策略和候选模型。
- `model_selection_v1.standard.md`：自动 rubric、分流规则和 14B 替换门槛。
- `model_selection_v1.result_summary.json`：8B、14B、移动云 API 的性能、质量和
  端到端回归摘要，以及 14B 不晋升结论。
- `v2_comparison_holdout_v1.json`：10 道全新比较题冻结集，**不参与提示词、检索或模型调优**，
  仅用于最终泛化验收；对象与调优题不重叠。
- `v2_comparison_holdout_v1.manifest.json`：冻结状态、SHA256、禁止调优和最终验收口径。
- `v2_comparison_holdout_v1.standard.md`：正式自动 gate、单题判定和口径限制。
- `v2_comparison_holdout_v1.result.json`：首次且唯一一次运行的原始结果归档。
- `v2_comparison_holdout_v1.result_summary.json`：首次且唯一一次运行的冻结结果摘要；
  `run_count=1`，题集已 retired，禁止再次运行。

scope v1 测试目录：`scope_v1/`
- `test.jsonl`：冻结的 16 条独立 scope 测试样本。
- `smoke.jsonl`：仅用于 Laya 训练输入格式与 dry-run 的合成小样。
- `manifest.json`：数据口径、哈希和冻结策略。
- `results/`：脚本生成的双基线与算力探针结果（可重建）。
- `dataset/`：800 条 LLM silver scope 数据、Laya 训练格式、审计表和质量报告。
- `dataset_v1_1/`：完成 146 条复核的不可变版本，保留 v1 作为父版本。
- `dataset_v1_2/`：复核 29 条错误后加入英文 ML、技术词 OOD 和最小对比对，
  共 1000 条；新增样本只进 train/calibration。
- `results_internal_v1_1/`：125 条开发期 LLM / Laya zero-shot 对照。
- `results_trained_v1_1/`：head-only pilot 的 Laya 热推理与分类结果。
- `gate_report.json`：只由 calibration 选阈值的三路门控报告。
- `pilot_summary.md`：head-only pilot 结论与未通过项。
- `pilot_v1_2_summary.md`：v1.2 head-only 与最后 2 层解冻的对比及停止扩容依据。
- `error_attribution.json` / `error_attribution.md`：逐题翻转、语言/领域误差、
  置信度分离、门控失败原因和归因排序。
- `gate_report_v1_2_head_only.json`：v1.2 head-only 门控。
- `gate_report_v1_2_last2.json`：v1.2 最后 2 层解冻门控。
- `results_v12_last2/`：最后 2 层模型的热推理和分类结果。
- `gate_report_v1_2_last2_oodw2.json`：OOD 2:1 非对称损失单次门控实验。
- `results_v12_oodw2/`：非对称损失模型的热推理和分类结果。

qrels v2 计划目录：`qrels_v2/`
- `queries.jsonl`：冻结查询
- `pool.jsonl`：盲排候选池
- `silver.jsonl`：LLM 银标

## 脚本速查

当前无 `.py`。

## 4. 建议阅读顺序

1. `questions.example.json`

## 5. 相关

- 上位指针：`../README.md`
