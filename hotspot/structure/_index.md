---
tags: [索引]
---

# hotspot/structure · `_index`

## 1. 这个文件夹是什么

离线论文表示、全量主题聚类和会议统计。

## 2. 当前工作状态

- `embed.py`：1024 维论文/查询 embedding，作为新检索栈的数据基础。
- `venue_matrix.py`：跨会扩散三态，热点口径的唯一客观统计。
- `topic_cluster.py`：PCA + 局部 kNN + Leiden，全量成员强制归属；
  2026-10-26 构建 22,747 篇、220 簇，构建耗时 20.6 秒。
- 生产快照为 `../data/topic_snapshot_v1.json.gz`，不依赖 `workspace/`。

## 3. 文件清单

**计划文件**：本 `_index.md`。
**数据文件**：`../data/topic_snapshot_v1.json.gz`，随代码部署，不手工编辑。

## 脚本速查

- `embed.py` —— BGE-M3 分块向量化，可续跑。
- `topic_cluster.py` —— 全量主题聚类并生成版本化 topic snapshot。
- `venue_matrix.py` —— 铺开 / 新进 / 单会三态计数。

（`__init__.py` 是包脚手架，不单独登记。）

## 4. 建议阅读顺序

1. `venue_matrix.py`
2. `embed.py`
3. `topic_cluster.py`

## 5. 相关

- 上位指针：`../_index.md`
