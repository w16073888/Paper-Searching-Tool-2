---
tags: [索引, 说明书]
---

# 移动云杯 · 本地论文库研究方向热点挖掘 · 总目录

> 给新会话的入口：先读本文档「本项目速览」+「§4 建议阅读顺序」即可开始；动手前先读相关文件夹的 `_index.md` / README。
> **浏览项目只读计划文件**——数据文件（`workspace/` 产物、skill 源）不读，要么可重建、要么该由脚本读。

## 1. 索引树

| 文件夹 | 干什么 | 它的 README | 最后更新 |
|---|---|---|---|
| `日志/` | 研究日志（**流水账**，一天一篇）——**直辖，无子 README，根只索引目录** | — | 2026-10-24 |
| `规划/` | 长期参考（导师建议 / 方向决策 / 评估结论 / **接口契约**）——**直辖，无子 README，根只索引目录** | — | 2026-10-23 |
| `hotspot/` | 核心代码（L1–L6 + **L6.5 服务层 `api/`**，脚本速查见其 `_index`） | [[hotspot/_index\|打开]] | 2026-10-22 |
| `web/` | 新前端静态页（最小参考实现，原生 JS；账号系统待接） | — | 2026-10-18 |
| `scripts/` | 验证/自检脚本 | [[scripts/_index\|打开]] | 2026-10-24 |
| `eval/` | 评测集与评测脚本 | [[eval/_index\|打开]] | 2026-10-24 |

## 2. 直辖文件夹

- **`日志/`**：研究日志（**流水账**，一天一篇，按文件名日期排序）——没有子 README。根索引只追踪这个文件夹，不维护逐篇清单；需要具体经历时现场按日期查。格式见 `.claude/skills/research-log/SKILL.md`。
- **`规划/`**：长期参考（导师建议 / 方向决策 / 评估结论 / 迁移备忘），文件名以首次撰写日期开头——没有子 README。根索引同样只追踪这个文件夹，不维护逐篇清单；需要方案时按主题和文件名日期现场查。格式见 `.claude/skills/research-log/SKILL.md`。

## 3. 根目录的散件

- `README.md` —— 本文件（索引树总目录）
- `STATUS.md` —— 续接会话先读的短状态：当前目标、进度、验收命令
- `AGENTS.md` —— **AI 会话唯一入口**：按任务查表 + 禁读清单 + 大文件读法 + 自检 + 红线
- `CLAUDE.md` / `.cursorrules` —— 别的 AI 工具的入口别名，只指向 `AGENTS.md`（不重复内容）
- `config.yaml` —— 唯一配置入口

> 长期参考 md（方案/测试计划/规划/分工）已归入 `规划/`，见 §2。

## 4. 建议阅读顺序（按入口/目录）

> **AI 会话从这里开始**：先读 `AGENTS.md` 的「按任务查表」，按任务只读 2–4 个文件；
> 不要为找代码启动子代理。人读则按下面的顺序。

1. 本文档「本项目速览」+「§4」+「§脚本速查」
2. `AGENTS.md`（按任务查表，只读该任务需要的 2–4 个文件）
3. `规划/`（按主题和文件名日期选方案、决策或契约）
4. `hotspot/_index.md`（想了解代码结构时）
5. `日志/`（需要看历史进展时）

## 脚本速查

代码脚本**按目录登记在各目录的 `_index.md`**（`__init__.py` 是包脚手架不登记）：

| 目录 | 直接脚本（入口见各 _index） |
|---|---|
| `hotspot/` | config / corpus / llm / schema（主导包见其 §3 子包表） |
| `hotspot/agent/` | deps / runtime（PydanticAI controller） |
| `hotspot/core/` | models（工具与 agent 领域模型） |
| `hotspot/index/` | lance_store（LanceDB paper index） |
| `hotspot/observability/` | trace（request-scoped append-only） |
| `hotspot/structure/` | embed / venue_matrix |
| `hotspot/retrieval/` | assemble（新检索栈待建） |
| `hotspot/tools/` | search / papers |
| `hotspot/reasoning/` | answer / verify / evidence |
| `hotspot/api/` | contract / engine / http（**前端唯一接口面**） |
| `hotspot/ui/` | app / store |
| `scripts/` | check_access / check_tree / check_api / build_lance_index / run_server |
| `eval/` | 问题集与工具 fixture（执行器待 v2 重建） |

> 每条脚本一行功能取自其 **docstring 首句**（单一源，README 不另写全文）；`scripts/check_tree.py` 会**自检**这些清单与磁盘 `.py` 是否一致。

## 5. 不在树上的目录（每项写"为什么不列"）

| 目录 | 为什么不在树上 |
|---|---|
| `workspace/` | **数据产物**（vectors/clusters/metrics/cards/ledger），可重建，不读 |
| `hotspot/vendor/` | 第三方 vendored 副本（dasc，179k 字符）；`_out/` 是 LLM 缓存与审计日志，**2026-10-18 起不入库**（407 个 JSON，纯粹是搜索噪声） |
| `.claude/skills/` / `.agents/skills/` | skill 源：`.claude` 是唯一源；`.agents` 是 Codex 兼容入口（当前工作区为 junction，新克隆见 `AGENTS.md` §9） |
| 导师库 `D:\大学编程学习\obsidian\rl_acdemical_paper` | 外部只读，四道闸，本项目不写它一个字 |

> 代码目录（`hotspot/ scripts/ eval/`）不再列入「不在树上」——它们已登记进 §1 索引树，脚本速查在各 `_index.md`。

## 6. 维护本文件（树形回写）

改任何 md 之后，回写子 README 与根 README，**终点是根（本文件）**——回写规则见 `.claude/skills/folder-readme/SKILL.md` §4。

---

# 本项目速览

接入 LLM API 后，根据本地论文库和用户关注的方向挖掘研究热点。
目标产物是**交互式问答**：用户问"XX 方向现在什么最火"，系统给出**有引用、可溯源**的回答。

## 两条硬墙（决定了方案长什么样）

先读这两条，否则会以为方案漏了什么——**它不是漏了，是数据里没有。**

| 墙 | 实测 | 被打掉的设计 |
|---|---|---|
| **没有引用网络** | 全库只有 `citations` 一个 int，**50 / 13,119 非空**；无 references / cited_by / 共引边。OpenAlex 配额 100 请求/天，导师库主动搁置到 2026 年底 | 引文网络、PageRank、Leiden 社区 |
| **（原）没有时间序列** | 原四会全 2025；**2026-09-22 起语料扩为 24+25 两年**，跨年事实已具备 | Kleinberg burst、生命周期四象限、CAGR、年份回溯——因**主口径选跨会、不做跨年**仍不进（口径选择，非无数据） |

⇒ 在本语料下，「热点」**唯一**的客观依据是**跨会扩散**三态：
**`铺开`（≥3 会）/ `新进`（只末位会）/ `单会`（其余）**。

## ⚠️ 红线

`:class:`hotspot.schema.SpreadState`**只有三态**。任何输出**不得出现「退潮 / 下降 / 衰退」**。**主口径选择对比跨会、不做跨年**（语料 24+25，2026-09-22 拍板）——`单会` 是**不可知**，不是"不热"。

## 语料：G 盘 24+25 合并库，只读

`G:\rl_data24\` + `G:\rl_data\` 两份 `papers.jsonl` 经 `hotspot/corpus.MergedCorpus` 并成 **22,747 篇**
（24=9,628 / 25=13,119；`Paper.year` 保留，可分组）。产物全落 `workspace/`（gitignore，可全量重建）。

## 分层（唯一口径，详见方案）

| 层 | 名称 | 目录 | LLM |
|---|---|---|---|
| L1 | 语料层 | `hotspot/corpus.py` | ❌ |
| L2 | 结构层 | `hotspot/structure/`（embed/venue_matrix） | ❌ 0 token |
| L3 | 主题/候选工具 | v2 待建 | 待定 |
| L4 | 检索层 | `hotspot/retrieval/`（v2 待建） | 待定 |
| L5 | 推理层 | `hotspot/reasoning/` | ✅ |
| L6 | 交互层 | `hotspot/ui/app.py` | ❌ |

## 设计原则（贯穿全链）

> **不让 LLM 读原文，让它读"已被压缩且可溯源的结构化中间表示"，并用引用 ID 强制溯源。**

**离线批量（L2 索引、L3 卡片）几乎 0 token**；**在线问答允许轻量 LLM**（路由 + 查询改写，单次成本极小）提升使用舒适度；LLM 主要出现在"命名簇 / 路由与改写 / 带引用叙述 / 引用校验"。全程不读正文（LaTeX 宏未展、84% 吞词）。细节见 `规划/2026-09-21-热点挖掘系统-方案设计.md`。
