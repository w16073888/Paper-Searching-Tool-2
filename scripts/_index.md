---
tags: [索引]
---

# scripts · `_index`

## 1. 这个文件夹是什么

运行入口、检查脚本和数据维护脚本。旧检索/大创算法专用脚本已删除。

## 2. 当前工作状态

- 服务入口、数据 embedding/周报更新和接口检查保留。
- 检索评测、G0–G3 评测和单一检索链脚本将在 v2 工具完成后再建。

## 3. 文件清单

**计划文件**：本 `_index.md`。
**数据文件**：无。

## 脚本速查

- `check_access.py` —— 验证导师库只读接入。
- `check_tree.py` —— 索引树与脚本速查自检。
- `check_api.py` —— 契约与 HTTP/SSE 接口自检。
- `check_tools_v2.py` —— LanceDB、论文读取与三条 trend snapshot 工具离线自检。
- `build_topic_snapshot.py` —— 从全库 BGE-M3 向量构建版本化主题快照。
- `check_topic_snapshot.py` —— 校验 topic snapshot 的覆盖、schema 和映射一致性。
- `run_v2_eval.py` —— v2 离线工具指标或可选真实 agent 端到端回归。
- `benchmark_model_candidates.py` —— 对 OpenAI 兼容端点测流式 TTFT、生成速度、
  长上下文、并发和稳定性。
- `run_model_selection.py` —— 用冻结模型选型集比较本地模型与可选云 API。
- `evaluate_comparison_holdout.py` —— 离线校验冻结比较题 holdout 的哈希、
  路由、证据支持率、引用与延迟阈值，不运行模型。
- `eval_scope_classifier.py` —— 当前 LLM 与 Laya zero-shot scope 双基线、延迟和训练 dry-run 探针。
- `build_scope_dataset.py` —— 用移动云模型构造 scope silver 数据、分组切分、审计和 Laya 格式导出。
- `finalize_scope_review.py` —— 复核 scope 样本、补 Laya 预检、生成带新哈希的不可变数据版本。
- `train_scope_laya.py` —— 参数化显式 train/calibration 入口，支持冻结/局部解冻与 OOD 非对称损失。
- `calibrate_scope_laya.py` —— 只用 calibration 选三路阈值，并对 internal_test 做一次固定门控评估。
- `analyze_scope_errors.py` —— 基于冻结预测做 base→微调翻转、语言/领域和门控误差归因。
- `augment_scope_dataset.py` —— 复核 29 条错误并生成 v1.2 定向补强数据、哈希和审计报告。
- `finalize_scope_training.py` —— 从已训练 checkpoint 恢复温度拟合、最终模型和 pilot 报告。
- `build_lance_index.py` —— 从现有 BGE-M3 向量分片构建 LanceDB 论文索引。
- `build_qrels_pool.py` —— 生成去分、盲排的 qrels 候选池。
- `auto_label_qrels.py` —— 用移动云 LLM 批量生成 silver qrels，支持缓存续跑。
- `run_server.py` —— HTTP 服务入口。
- `run_ui.py` —— Streamlit 入口。
- `daily_arxiv_sync.py` —— 云端 arXiv 日更总入口：抓取、校验、周绑定、
  增量向量化、LanceDB 更新和 harness 导出。
- `run_weekly_cron.sh` —— 云端 cron 包装器：固定配置与路径，转发日更或封卷参数。
- `build_weekly_harness_input.py` —— 从全局 arXiv 库生成按 ISO 周绑定的
  `harness_input.jsonl` 与 manifest。
- `check_weekly_pipeline.py` —— 周绑定、harness 契约和 week_id 检索过滤的离线自检。
- `cost_report.py` —— 成本报表。
- `export_arxiv_weekly.py` —— arXiv 周报向量与发布物生成。
- `update_arxiv_weekly.py` —— arXiv 周报增量更新。
- `summarize_trace.py` —— Trace JSONL 汇总，不输出问题/答案原文。
- `build_dist.py` —— 服务端源码打包。
- `build_release.py` —— 发布目录组装。

## 4. 建议阅读顺序

1. `check_api.py`
2. `run_server.py`
3. `check_topic_snapshot.py`
4. `check_tree.py`

## 5. 相关

- 上位指针：`../README.md`
- 重建前实现：Git commit `b517988`
