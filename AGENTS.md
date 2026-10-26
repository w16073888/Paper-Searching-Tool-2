# 移动云杯 · 开发入口（AGENTS.md）

> **续接会话先读 `STATUS.md`**（只保留当前目标/进度/下一条命令），再查下面的任务表。
> 大改前先写清“只改文件 / 只读函数 / 验收命令”；未写清就不要扩展探索。
> 命令固定用 PowerShell 原生 cmdlet，不同时混用 bash/cmd 的引号与转义方式。
> 日常自检用 `python scripts/check_api.py --quiet`；Trace 先用
> `python scripts/summarize_trace.py <jsonl>` 汇总，不整段读原文。

> **给 AI 的第一句话：这个仓库自带地图。先查下面的表，只读表里列的文件。**
> 不要为了"找代码"启动子代理、不要 `grep -r` 整个仓库、不要整篇读 `engine.py`。
> 90% 的任务 2–4 个文件就够，读完就停。

## 0. 三行看懂这个项目

本地论文库（22,747 篇会议论文）的研究助手。v2 使用 LanceDB 论文工具、
全量主题聚类快照、证据门控与引用校验；旧固定检索/子聚类/路由链仅保留在 Git 历史。
对外**只有 2 个接口**：`hotspot/api` 的 `meta`（能力/参数）与 `ask`（问一次）。
三种消费者：Streamlit（`hotspot/ui/`）、HTTP 网页（`web/` + `scripts/run_server.py`）、脚本/评测（`scripts/`、`eval/`）。
**前端只准 import `hotspot.api`，算法内部随便改**——这条边界是 2026-10-18 定的，别再拆。

## 1. 按任务查表（先查表，再读文件）

| 我要… | 只读这些（按顺序） | 读完就停，别扩散 |
|---|---|---|
| 改接口 / 加返回字段 / 加参数 | `规划/2026-10-18-接口契约-前端与算法.md` §4–§5 → `hotspot/api/contract.py` → `hotspot/api/engine.py` 的 `Engine.ask` | 不用读 `ui/`、不用读算法内部 |
| 改前端页面 / 接账号系统 | `hotspot/api/_index.md` → 契约文档 §8 → `hotspot/ui/app.py`（Streamlit）或 `web/index.html`（HTTP） | 不用读 `subcluster`/`retrieval` |
| 恢复旧大创算法做对照 | Git commit `b517988` | 当前分支已删除，不用磁盘旧路径猜 |
| 重建/调优主题聚类 | `hotspot/structure/_index.md` → `structure/embed.py` → `structure/topic_cluster.py` → `scripts/build_topic_snapshot.py` | 不动 `ui/` 与契约 |
| 重建检索 / 路由 / 召回 | `规划/2026-10-21-...` → `hotspot/retrieval/_index.md` | 当前分支没有旧 hybrid/router，禁止恢复 `subcluster_adaptive` |
| 改引用校验（防幻觉） | `hotspot/reasoning/_index.md` → `verify.py` | 四态 `supported/unsupported/missing_id/no_citation` **不许合并** |
| 跑数据 / 评测 | `scripts/_index.md`、`eval/_index.md` → `eval/run_eval.py` | 不要读 `workspace/` 产物 |
| 排查「拒答 / 没结果」 | `hotspot/api/engine.py`（当前为重建壳） → `STATUS.md` | 旧拒答链已删除；先确认是否在新工具链接入前 |
| 写周报 / 日志 / 回写索引 | `.claude/skills/research-log`、`.claude/skills/folder-readme` → `日志/` 最新一篇 | 只读最近 1 篇，不要通读历史 |

## 2. 不要读（省 token，而且会误导判断）

| 路径 | 为什么 |
|---|---|
| `hotspot/vendor/dasc/**` | 第三方 vendored 副本，**179k 字符**。只有确认要改 dasc 内部才看 |
| `hotspot/vendor/dasc/_out/**` | LLM 缓存与出境审计（406 个 JSON，已从版本库移除）。grep 会被淹没 |
| `workspace/**` | 可重建产物（向量/卡片/metrics/ledger）。要数据就写脚本读，别进上下文 |
| `日志/` 除最新 1–2 篇 | 流水账，只在查"当时为什么这么定"时读 |
| `*.npz` `*.pkl` `*.jsonl` | 二进制/大文件，属于脚本的输入，不属于你的上下文 |

## 3. 找东西的正确姿势（别起子代理）

```bash
# 单文件定位函数/类（先看骨架，再读那一段）
grep -n "^class \|^    def \|^# ═" hotspot/api/engine.py

# 全仓找符号（带后缀过滤，避开 vendor 与产物）
grep -rn --include=*.py "refuse_reason" . | grep -v vendor

# 找某个文件/目录
git ls-files | grep -i "检索\|retrieval"
```

**只有一种情况用 Explore 子代理**：完全不知道在哪、且要跨 5 个以上目录。
用的时候把范围写死（例如"只看 `hotspot/retrieval/` 和 `hotspot/reasoning/`"），别让它自由探索。

两条实测踩过的坑：

- **数文件用 `git ls-files`，别用 `find .`**：本地还有未跟踪的 `workspace/` 和 `vendor/.../_out/`，
  `find` 会数出 633 个文件，而仓库实际只有 118 个。
- **优先 `rg`**（读 `.gitignore`，自动跳过产物）；没装再用
  `grep -rn --include=*.py <符号> . | grep -v vendor`。
- **检索语义**：`corpus.search()` 是**全词 AND**。长句或中文改写结果要先拆词再做并集
  （`Engine._search_union`），否则会出现"库里有 MARL，但问多智能体却答本库无相关内容"。

## 4. 大文件怎么读（不要整篇读）

| 文件 | 行数 | 先 grep 这些入口 |
|---|---|---|
| `hotspot/api/engine.py` | 重建壳 | `Engine.meta` / `Engine.ask` |
| `hotspot/api/contract.py` | 574 | `AskRequest` / `AskResponse` / `PARAM_SPEC` / `apply_params` |
| `hotspot/ui/app.py` | 397 | `_run_with_progress` / `_render_assistant` / `main` |
| `hotspot/schema.py` | 339 | `SpreadState`（三态红线）/ `Answer` / `Context` |

这些文件都有 `# ══════` 分节注释，`sed -n '360,470p'` 读一节即可。

## 5. 改完必须跑（30 秒，离线）

```bash
python scripts/check_api.py --quiet      # 只报失败项和总数
python scripts/check_tree.py --vault .   # 索引树：悬空链接 / 孤儿 md / 脚本速查
```

两个都绿再提交。新增/改名 `.py` 要回写对应 `_index.md` 的「脚本速查」，
否则 `check_tree.py` 会报；改完顺手把 `README.md` 那一行的「最后更新」改掉。

## 6. 接口红线（2026-10-18 起）

1. 前端只准 import `hotspot.api`（或走 HTTP），不准 import 算法内部模块。
2. **只增不改**：加字段必须带默认值；删/改语义必须换 `CONTRACT_VERSION` 并写迁移说明。
3. 加可调参数 → 只改 `contract.py` 的 `PARAM_SPEC` + `config.yaml` 默认值，前端设置页自动出现。
4. 引擎**无会话状态**：历史与每用户 `api_key` 都随请求带，别在服务层存状态。

## 7. 热点红线（本项目口径）

热点只能表述为三态：**铺开 / 新进 / 单会**；**永不输出「退潮 / 下降 / 衰退」**
（语料 24+25，但主口径选择对比跨会、不做跨年，2026-09-22 拍板）。

## 8. 权限

| 路径 | 可写 |
|---|---|
| 根散件、`规划/`、`日志/`、`hotspot/`、`scripts/`、`eval/`、`web/` | ✅ |
| `workspace/`（可重建产物） | ♻️ 重建，不提交 |
| `G:\rl_data24`、`G:\rl_data`、导师库 | ❌ 只读 |
| `hotspot/vendor/` | ⚠️ 尽量不改（改了要写原因） |

## 9. Skill 部署

- **唯一源**：`.claude/skills/`（Claude Code 项目级 skill 目录）。
- **Codex 兼容入口**：`.agents/skills` 在 Windows 上是指向 `.claude/skills` 的目录
  junction；不要把它当第二份源，也不要在 `.agents` 里单独改 skill。
- 新克隆或非 Windows 环境若 `.agents/skills` 变成普通目录，以 `.claude` 为准重建链接
  或同步副本；改完用 `python -X utf8 .../quick_validate.py .claude/skills/<name>` 校验。
