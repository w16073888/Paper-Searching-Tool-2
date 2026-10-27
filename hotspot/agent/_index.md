---
tags: [索引]
---

# hotspot/agent · `_index`

## 1. 这个文件夹是什么

PydanticAI 控制回路。Agent 只决定搜索与取论文，不执行数据库或索引操作。

## 2. 当前工作状态

- `search_tool` / `fetch_tool` 带调用预算。
- 相同 search/fetch 参数使用请求内缓存拦截，避免模型重复工具循环耗尽预算。
- Qwen 偶发的非对象 claim 条目会被安全丢弃；预算耗尽时只保留已 fetch evidence。
- 输出 validator 强制证据 ID 必须先经过 fetch。
- Evidence 与 citation verify 由 API runtime 在 agent 输出后强制执行。

## 3. 文件清单

**计划文件**：本 `_index.md`。
**数据文件**：无。

## 脚本速查

- `deps.py` —— request-scoped 搜索状态和依赖。
- `runtime.py` —— PydanticAI Agent、工具注册、预算和 grounded-output validator。

（`__init__.py` 是包脚手架，不单独登记。）

## 4. 建议阅读顺序

1. `deps.py`
2. `runtime.py`

## 5. 相关

- 上位指针：`../_index.md`
