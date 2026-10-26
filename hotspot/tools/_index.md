---
tags: [索引]
---

# hotspot/tools · `_index`

## 1. 这个文件夹是什么

Agent 可选择调用的通用工具。工具不接收 `answer_mode`，也不包含最终回答逻辑。

## 2. 当前工作状态

- `search_papers`：LanceDB vector + FTS 混合召回，带年份/会议硬过滤。
- `fetch_papers`：按搜索返回 ID 读取摘要。
- `get_topic_overview`：读取版本化 topic snapshot；空结果由 Agent
  继续调用 `search_papers` 聚合证据。
- Evidence gate 与 citation verify 是强制 runtime 管线，不属于可选工具。

## 3. 文件清单

**计划文件**：本 `_index.md`。
**数据文件**：无。

## 脚本速查

- `search.py` —— 通用论文搜索工具。
- `papers.py` —— 论文详情读取工具。
- `topics.py` —— 读取离线 topic snapshot，不在线聚类。

（`__init__.py` 是包脚手架，不单独登记。）

## 4. 建议阅读顺序

1. `search.py`
2. `papers.py`

## 5. 相关

- 上位指针：`../_index.md`
