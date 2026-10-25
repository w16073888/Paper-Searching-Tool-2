---
tags: [索引]
---

# hotspot · `_index`

## 1. 这个文件夹是什么

论文研究助手的核心包。2026-10-22 起进入 v2 重建：旧检索、聚类、路由和大创算法
已移除；当前保留稳定接口、数据读取、embedding、证据判断与引用校验。

## 2. 当前工作状态

- 旧算法在 `b517988` 快照中可恢复，并已从当前分支删除。
- `hotspot.api` 保留稳定请求/响应形状；`Engine.ask` 暂返回重建提示。
- 保留 `structure/embed.py` 作为 1024 维论文表示入口。
- 保留 `reasoning/evidence.py`、`verify.py`、`answer.py` 作为领域校验基础。

## 3. 文件清单

**计划文件**：本 `_index.md`。
**数据文件**：无（可重建产物在 `workspace/`）。

**子包**：

| 子包 | 当前职责 | 索引 |
|---|---|---|
| `api/` | 对外稳定接口与 HTTP | [[hotspot/api/_index\|打开]] |
| `agent/` | PydanticAI controller 与 grounded output | [[hotspot/agent/_index\|打开]] |
| `core/` | 工具/agent/API 共用的领域模型 | [[hotspot/core/_index\|打开]] |
| `index/` | LanceDB paper index 与混合召回 | [[hotspot/index/_index\|打开]] |
| `observability/` | append-only request Trace | [[hotspot/observability/_index\|打开]] |
| `reasoning/` | 证据、回答结构、引用校验 | [[hotspot/reasoning/_index\|打开]] |
| `retrieval/` | 仅保留上下文装配，待新检索栈替换 | [[hotspot/retrieval/_index\|打开]] |
| `structure/` | embedding 与会议扩散统计 | [[hotspot/structure/_index\|打开]] |
| `tools/` | agent 可调用的 search/fetch 工具 | [[hotspot/tools/_index\|打开]] |
| `ui/` | Streamlit 适配 | [[hotspot/ui/_index\|打开]] |
| `weekly/` | arXiv 周绑定、harness 输入和增量向量 | [[hotspot/weekly/_index\|打开]] |

## 脚本速查

- `config.py` —— 配置加载与路径解析。
- `corpus.py` —— 会议/arXiv 语料读取与统计。
- `cost.py` —— 成本记账与报表。
- `llm.py` —— dasc LLM 客户端薄封装。
- `schema.py` —— 保留的内部领域数据结构。

（`__init__.py` 是包脚手架，不单独登记。）

## 4. 建议阅读顺序

1. `../规划/2026-10-21-检索智能体地基与演进方向.md`
2. `api/contract.py`
3. `corpus.py`、`structure/embed.py`
4. `reasoning/evidence.py`、`reasoning/verify.py`

## 5. 相关

- 上位指针：`../README.md`
- 重建前快照：Git commit `b517988`
