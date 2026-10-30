---
tags: [索引]
---

# hotspot/weekly · `_index`

## 1. 这个文件夹是什么

arXiv 周报与本项目之间的适配层：按 ISO 周绑定论文、生成 LLM harness 输入，
并计算新增论文的增量向量。

## 2. 当前工作状态

- 周归属唯一使用 `first_seen` 的 ISO 周。
- `week_id` 已进入 arXiv LanceDB 和 API 的 `scope.week_id`。
- `harness_input.jsonl` 使用 `weekly-harness-v1` 契约。
- 封卷要求周一至周五五个 `.done` 全部存在且无失败、停滞、旧 feed 或受限批次。
- 保留窗口由 `arxiv_weekly.max_retention_weeks` 控制：按当前 ISO 周向前滚动，
  同时清理旧 `papers.jsonl` 行、ISO 周目录和 arXiv 向量分片。
- 原周报程序保存在 `hotspot/vendor/arxiv_weekly/`，只改路径适配。
- 无法从 arXiv 当前 RSS 回补已经错过的历史日批次。

## 3. 文件清单

**计划文件**：本 `_index.md`。
**数据文件**：运行时数据在 `data/arxiv-weekly/`，不入库。

## 脚本速查

- `binding.py` —— ISO 周成员绑定、公告旁路合并、harness manifest、保留窗口清理。
- `vectors.py` —— 为缺少向量的 arXiv 论文生成增量分片，并删除过期向量。

## 4. 建议阅读顺序

1. `binding.py`
2. `vectors.py`
3. `../../scripts/daily_arxiv_sync.py`

## 5. 相关

- 上位指针：`../_index.md`
- 原始程序：`../vendor/arxiv_weekly/`
- 外部源：`D:\大学编程学习\arXiv周报`
