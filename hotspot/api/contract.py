"""前端 ↔ 算法 的**唯一**数据契约（API v1）。

改这里 = 改所有人的接口。**只增不改**：新增字段给默认值；删/改语义必须换
:data:`CONTRACT_VERSION` 并写迁移说明（沿用 `hotspot/schema.py` 的纪律）。

设计目标（写给接手的人）
------------------------
1. **前端不需要认识算法**：不需要 `corpus.get(pid)` 取标题、不需要知道
   `key_path` 字典长什么样、不需要知道簇卡片 JSON 的字段名。所有展示需要的
   东西都在响应里**摊平**好了（`citations` / `topics` / `graph`）。
2. **一条请求问完**：问题、数据源、追问历史、每用户的 API 配置、检索参数，
   全在 :class:`AskRequest` 里 —— 服务端**无会话状态**，多账号/多进程都安全。
3. **可自描述**：:class:`MetaResponse` 带着 `params` 参数表，前端据此**自动生成**
   设置页；加一个参数只改本文件的 :data:`PARAM_SPEC`，前端不用动。

字段总览见 `规划/2026-10-18-接口契约-前端与算法.md`。
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

#: 契约版本。不兼容改动必须换号。
CONTRACT_VERSION = "2"

#: 数据源枚举（与 `config.corpus` 一一对应，不新增实现）。
SOURCES: dict[str, str] = {"meeting": "会议 24+25", "arxiv": "arXiv 周报"}

#: 问答模式：`new`=重新检索；`followup`=基于历史（并把前文引用的论文并进上下文）。
MODES: tuple[str, ...] = ("new", "followup")

#: 进度阶段。**前端只认 key，不认中文名**——中文名改了不影响前端。
STAGES: tuple[tuple[str, str], ...] = (
    ("retrieve", "检索与证据"),
    ("answer", "生成回答"),
    ("verify", "校验引用"),
)

#: 算法内部（`verify.py` 等）报上来的中文阶段名 → 契约 stage key。
STAGE_KEY_FROM_NAME: dict[str, str] = {label: key for key, label in STAGES}

#: 错误码。前端按 code 分支，`message` 直接可展示。
E_BAD_REQUEST = "bad_request"
E_NO_API_KEY = "no_api_key"
E_CORPUS_UNAVAILABLE = "corpus_unavailable"
E_LLM_FAILED = "llm_failed"
E_INTERNAL = "internal_error"
E_INDEX_UNAVAILABLE = "index_unavailable"
E_SCOPE_TIMEOUT = "scope_timeout"
E_AGENT_TIMEOUT = "agent_timeout"
E_AGENT_CONNECTION = "agent_connection_failed"
E_AGENT_OUTPUT = "agent_output_invalid"
E_VERIFY_TIMEOUT = "verify_timeout"


def normalize_base_url(url: str) -> str:
    """接受 base URL 或完整 ``/chat/completions`` 端点，统一成 base URL。"""
    u = str(url or "").strip().rstrip("/")
    suffix = "/chat/completions"
    if u.endswith(suffix):
        u = u[: -len(suffix)].rstrip("/")
    return u


class ApiError(Exception):
    """契约级错误。`code` 稳定可判，`message` 面向用户，`status` 是 HTTP 状态码。"""

    def __init__(self, code: str, message: str, *, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message}


# ══════════════════════════════════════════════════════════════════════
# 请求
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Message:
    """一轮历史消息。`citation_ids` 只在 assistant 消息上有意义。

    前端存会话时把 `citation_ids` 一起存下来，追问就能把「上次引过的论文」
    重新并进上下文 —— 这正是组长要的「追问包含前文搜出的语料」，
    而且**不需要服务端记任何东西**。
    """

    role: str
    text: str
    citation_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "text": self.text, "citation_ids": list(self.citation_ids)}

    @classmethod
    def from_dict(cls, d: Any, *, index: int) -> "Message":
        if not isinstance(d, dict):
            raise ApiError(E_BAD_REQUEST, f"history[{index}] 必须是对象")
        role = str(d.get("role") or "").strip()
        if role not in ("user", "assistant"):
            raise ApiError(E_BAD_REQUEST, f"history[{index}].role 只能是 user / assistant")
        text = d.get("text")
        if not isinstance(text, str):
            raise ApiError(E_BAD_REQUEST, f"history[{index}].text 必须是字符串")
        ids = d.get("citation_ids") or []
        if not isinstance(ids, list):
            raise ApiError(E_BAD_REQUEST, f"history[{index}].citation_ids 必须是数组")
        return cls(role=role, text=text, citation_ids=[str(i) for i in ids])


@dataclass
class LLMConfig:
    """**每用户**的运行态 API 配置（前端账号系统存的东西）。

    优先级：本对象 > `config.yaml` 的 `llm`。`api_key` 只在本进程内存里待着，
    响应里**永不回显**，日志里只留末 4 位。
    """

    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"base_url": self.base_url, "model": self.model,
                "api_key": "***" if self.api_key else ""}

    @classmethod
    def from_dict(cls, d: Any) -> "LLMConfig":
        if d in (None, {}):
            return cls()
        if not isinstance(d, dict):
            raise ApiError(E_BAD_REQUEST, "llm 必须是对象 {base_url, model, api_key}")
        def _s(k: str) -> Optional[str]:
            v = d.get(k)
            if v in (None, ""):
                return None
            if not isinstance(v, str):
                raise ApiError(E_BAD_REQUEST, f"llm.{k} 必须是字符串")
            return v.strip() or None
        return cls(base_url=_s("base_url"), model=_s("model"), api_key=_s("api_key"))

    def apply_to(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """把运行态配置盖到 cfg 副本上。**不读环境变量、不写 config.yaml。**"""
        if not (self.base_url or self.model or self.api_key):
            return cfg
        llm = dict(cfg.get("llm") or {})
        if self.base_url:
            llm["base_url"] = normalize_base_url(self.base_url)
        if self.model:
            llm["model_flash"] = self.model
            llm["model_pro"] = self.model
        if self.api_key:
            llm["api_key"] = self.api_key
            # 有显式 key 时切断环境变量，避免请求 key 被服务端 key 覆盖。
            llm.pop("api_key_env", None)
        else:
            llm.pop("api_key", None)
            # 没有显式 key 时保留配置里的 api_key_env，让服务端 key
            # 与 meta.has_server_key 的语义一致。
        cfg["llm"] = llm
        return cfg


@dataclass
class AskRequest:
    """一次问答请求。字段全部可选（除 question），默认值即生产默认。

    Attributes:
        question: 用户问题原文。上限 2000 字（超了截断并给 warning）。
        source: `meeting` / `arxiv`。
        mode: `new` 重新检索；`followup` 基于 `history`（不重新检索）。
        history: 历史消息（只有 followup 用；服务端**不存会话**）。
        llm: 该用户的 API 配置。
        params: 检索/展示参数覆盖（白名单见 :data:`PARAM_SPEC`）。
        scope: 前端解析的硬过滤条件（years/venues/week_id）。
        dry_run: 不调真 LLM，只走离线链路（前端无 key 时也能把界面跑起来）。
        include_graph: 是否计算/返回 Streit-Garg 关键路径图。
    """

    question: str
    source: str = "meeting"
    mode: str = "new"
    history: list[Message] = field(default_factory=list)
    llm: LLMConfig = field(default_factory=LLMConfig)
    params: dict[str, Any] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False
    include_graph: bool = True
    #: 请求格式可解析、但当前算法不执行的字段提示。
    scope_warnings: list[str] = field(default_factory=list)
    #: 问题超过 :data:`MAX_QUESTION` 时被截断（服务层会给一条 warning）。
    question_truncated: bool = False
    #: 历史消息超过 :data:`MAX_HISTORY` 时只保留最近的消息。
    history_truncated: bool = False

    #: 问题长度上限（超了截断，不报错 —— 前端输入框也可能没设 maxlength）
    MAX_QUESTION = 2000
    #: 历史消息上限（防止一次塞几千条把上下文撑爆）
    MAX_HISTORY = 40

    @classmethod
    def from_dict(cls, d: Any) -> "AskRequest":
        if not isinstance(d, dict):
            raise ApiError(E_BAD_REQUEST, "请求体必须是 JSON 对象")
        q = d.get("question")
        if not isinstance(q, str) or not q.strip():
            raise ApiError(E_BAD_REQUEST, "question 不能为空")
        raw_q = q.strip()
        q = raw_q[: cls.MAX_QUESTION]
        truncated = len(raw_q) > cls.MAX_QUESTION

        source = str(d.get("source") or "meeting").strip()
        if source not in SOURCES:
            raise ApiError(E_BAD_REQUEST,
                           f"source 只能是 {' / '.join(SOURCES)}，收到 {source!r}")
        mode = str(d.get("mode") or "new").strip()
        if mode not in MODES:
            raise ApiError(E_BAD_REQUEST, f"mode 只能是 {' / '.join(MODES)}，收到 {mode!r}")

        raw_hist = d.get("history") or []
        if not isinstance(raw_hist, list):
            raise ApiError(E_BAD_REQUEST, "history 必须是数组")
        history = [Message.from_dict(m, index=i) for i, m in enumerate(raw_hist)]
        history_truncated = len(history) > cls.MAX_HISTORY
        if len(history) > cls.MAX_HISTORY:
            history = history[-cls.MAX_HISTORY:]

        params = d.get("params") or {}
        if not isinstance(params, dict):
            raise ApiError(E_BAD_REQUEST, "params 必须是对象")
        params = validate_params(params)

        raw_scope = d.get("scope") or {}
        if not isinstance(raw_scope, dict):
            raise ApiError(E_BAD_REQUEST, "scope 必须是对象")
        unknown_scope = sorted(
            set(raw_scope) - {"years", "years_explicit", "venues", "week_id"}
        )
        scope_warnings = []
        if unknown_scope:
            scope_warnings.append(
                f"scope 字段 {unknown_scope} 当前不支持，已忽略。"
            )
        scope: dict[str, Any] = {}
        raw_years = raw_scope.get("years") or []
        if not isinstance(raw_years, list):
            raise ApiError(E_BAD_REQUEST, "scope.years 必须是数组")
        try:
            scope["years"] = [int(y) for y in raw_years]
        except (TypeError, ValueError):
            raise ApiError(E_BAD_REQUEST, "scope.years 必须是整数数组") from None
        scope["years_explicit"] = bool(raw_scope.get("years_explicit", bool(raw_years)))
        raw_venues = raw_scope.get("venues") or []
        if not isinstance(raw_venues, list):
            raise ApiError(E_BAD_REQUEST, "scope.venues 必须是数组")
        scope["venues"] = [str(x) for x in raw_venues]
        raw_week = str(raw_scope.get("week_id") or "").strip()
        if raw_week:
            match = re.fullmatch(r"(\d{4})-W(\d{2})", raw_week)
            try:
                if not match:
                    raise ValueError
                _dt.date.fromisocalendar(int(match[1]), int(match[2]), 1)
            except ValueError:
                raise ApiError(
                    E_BAD_REQUEST,
                    "scope.week_id 必须是合法 ISO 周，例如 2026-W43",
                ) from None
            scope["week_id"] = raw_week

        return cls(
            question=q,
            source=source,
            mode=mode,
            history=history,
            llm=LLMConfig.from_dict(d.get("llm")),
            params=params,
            scope=scope,
            dry_run=bool(d.get("dry_run") or False),
            include_graph=bool(d.get("include_graph", True)),
            scope_warnings=scope_warnings,
            question_truncated=truncated,
            history_truncated=history_truncated,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question, "source": self.source, "mode": self.mode,
            "history": [m.to_dict() for m in self.history],
            "llm": self.llm.to_dict(), "params": dict(self.params),
            "scope": dict(self.scope),
            "dry_run": self.dry_run, "include_graph": self.include_graph,
            "scope_warnings": list(self.scope_warnings),
            "question_truncated": self.question_truncated,
            "history_truncated": self.history_truncated,
        }


# ══════════════════════════════════════════════════════════════════════
# 参数白名单（前端设置页由这份表自动生成）
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ParamSpec:
    """一个可调参数。`path` 是它在 `config.yaml` 里的位置。"""

    key: str
    path: tuple[str, ...]
    kind: str                 # "int" | "bool" | "float"
    label: str                # 中文名（前端直接显示）
    group: str                # 设置页分组
    help: str = ""
    default: Any = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None

    def coerce(self, value: Any) -> Any:
        """类型/范围校验 + 归一。**不合法就报错，不静默忽略**。"""
        if self.kind == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            if isinstance(value, str) and value.lower() in ("true", "false", "1", "0"):
                return value.lower() in ("true", "1")
            raise ApiError(E_BAD_REQUEST, f"参数 {self.key} 必须是布尔值")
        if self.kind in ("int", "float"):
            try:
                num = float(value) if not isinstance(value, bool) else float(value)
            except (TypeError, ValueError):
                raise ApiError(E_BAD_REQUEST, f"参数 {self.key} 必须是数字") from None
            if self.kind == "int":
                if num != int(num):
                    raise ApiError(E_BAD_REQUEST, f"参数 {self.key} 必须是整数")
                num = int(num)
            if self.minimum is not None and num < self.minimum:
                raise ApiError(E_BAD_REQUEST,
                               f"参数 {self.key} 不能小于 {self.minimum:g}")
            if self.maximum is not None and num > self.maximum:
                raise ApiError(E_BAD_REQUEST,
                               f"参数 {self.key} 不能大于 {self.maximum:g}")
            return num
        raise ApiError(E_INTERNAL, f"未知参数类型 {self.kind}")

    def current(self, cfg: dict[str, Any]) -> Any:
        """从 cfg 读当前值；没配则用默认值。"""
        node: Any = cfg
        for k in self.path:
            if not isinstance(node, dict) or k not in node:
                return self.default
            node = node[k]
        return node if node is not None else self.default

    def to_dict(self, cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return {
            "key": self.key, "type": self.kind, "label": self.label,
            "group": self.group, "help": self.help,
            "default": self.default,
            "value": self.current(cfg) if cfg is not None else self.default,
            "min": self.minimum, "max": self.maximum, "step": self.step,
        }


#: **唯一**的可调参数表。加参数只改这里；前端设置页自动跟着变。
PARAM_SPEC: tuple[ParamSpec, ...] = (
    ParamSpec("search_limit", ("agent", "search_limit"), "int",
              "单次搜索候选数", "Agent", "search_papers 单次最多返回的论文数",
              12, 3, 30, 1),
    ParamSpec("fetch_limit", ("agent", "fetch_limit"), "int",
              "单次取摘要论文数", "Agent", "fetch_papers 单次最多读取的摘要数",
              6, 1, 20, 1),
    ParamSpec("max_claims", ("agent", "max_claims"), "int",
              "答案 claim 上限", "回答", "最终最多保留的结构化证据 claim 数",
              6, 1, 12, 1),
    ParamSpec("agent_request_limit", ("agent", "request_limit"), "int",
              "模型请求上限", "Agent", "单个问题最多允许的模型请求数",
              10, 2, 20, 1),
    ParamSpec("agent_tool_limit", ("agent", "tool_limit"), "int",
              "工具调用上限", "Agent", "单个问题最多允许的工具调用数",
              10, 2, 20, 1),
    ParamSpec("scope_timeout_s", ("agent", "scope_timeout_s"), "int",
              "范围判断超时", "稳定性", "scope classifier 单次超时秒数",
              30, 5, 120, 5),
    ParamSpec("agent_timeout_s", ("agent", "agent_timeout_s"), "int",
              "Agent 超时", "稳定性", "主 agent 单次运行超时秒数",
              150, 30, 600, 10),
    ParamSpec("verify_timeout_s", ("agent", "verify_timeout_s"), "int",
              "引用校验超时", "稳定性", "整批引用校验硬超时秒数",
              30, 5, 120, 5),
)

PARAM_BY_KEY: dict[str, ParamSpec] = {p.key: p for p in PARAM_SPEC}


def validate_params(params: dict[str, Any]) -> dict[str, Any]:
    """白名单校验。未知键**报错**（不静默丢弃——否则前端以为生效了）。"""
    unknown = sorted(set(params) - set(PARAM_BY_KEY))
    if unknown:
        raise ApiError(E_BAD_REQUEST,
                       f"未知参数 {unknown}；可用参数：{sorted(PARAM_BY_KEY)}")
    return {k: PARAM_BY_KEY[k].coerce(v) for k, v in params.items()}


def apply_params(cfg: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """把参数覆盖写进 cfg 副本（就地改，调用方传 deepcopy）。"""
    for k, v in params.items():
        spec = PARAM_BY_KEY[k]
        node = cfg
        for seg in spec.path[:-1]:
            nxt = node.get(seg)
            if not isinstance(nxt, dict):
                nxt = {}
                node[seg] = nxt
            node = nxt
        node[spec.path[-1]] = v
    return cfg


# ══════════════════════════════════════════════════════════════════════
# 响应
# ══════════════════════════════════════════════════════════════════════


@dataclass
class Citation:
    """一条可展示的引用。**已经 deref 过**：前端不用再拿 id 去查语料。"""

    id: str
    kind: str                  # "paper" | "cluster"
    title: str = ""
    label: str = ""            # 簇卡片的主题名（kind=cluster 时有）
    venue: str = ""
    year: Optional[int] = None
    week_id: str = ""
    first_seen: str = ""
    abstract: str = ""
    arxiv_id: str = ""
    url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Verdict:
    """引用校验结果。`status` 四态：supported / unsupported / missing_id / no_citation。"""

    status: str
    text: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Topic:
    """一个主题簇（结构化卡片；渲染交给前端）。"""

    cluster_id: int
    label: str = ""
    core: str = ""
    size: int = 0
    venue_counts: dict[str, int] = field(default_factory=dict)
    year_counts: dict[str, int] = field(default_factory=dict)
    representative_id: str = ""
    representative_title: str = ""
    member_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class GraphNode:
    id: str
    title: str
    step: int
    cluster: Optional[int] = None
    new_cover: bool = False
    covered_so_far: float = 0.0      # 累积 ICF（覆盖度），不是篇数

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class GraphEdge:
    source: str
    target: str
    kind: str            # "chain"=依赖序主链；"cite"=真实引用依赖

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class GraphView:
    """Streit-Garg 关键路径图。**只给数据，不给图**——前端自己选 plotly/cytoscape/d3。"""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    algo: str = ""
    seed_count: int = 0
    value: float = 0.0
    delta: float = 0.0
    conn: float = 0.0
    covered_clusters: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d


@dataclass
class Usage:
    """本次请求的 token/成本/耗时。**实测值**（来自 LLM 客户端计数器），不是估的。"""

    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cache_hits: int = 0
    failures: int = 0
    cost_yuan: float = 0.0
    elapsed_s: float = 0.0
    stages: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class AskResponse:
    """一次问答的完整结果。字段表见 `规划/2026-10-18-接口契约-前端与算法.md`。"""

    request_id: str
    source: str
    mode: str
    route: str = "local"                  # global / local / followup / refuse / error
    refused: bool = False
    refuse_reason: str = ""
    text: str = ""
    n_filtered: int = 0
    verdicts: list[Verdict] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    topics: list[Topic] = field(default_factory=list)
    graph: Optional[GraphView] = None
    usage: Usage = field(default_factory=Usage)
    warnings: list[str] = field(default_factory=list)
    error: Optional[dict[str, Any]] = None
    #: 算法附加信号（rlRAG 排序、key_path 等）；前端可忽略，评测可用。
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "request_id": self.request_id,
            "source": self.source,
            "mode": self.mode,
            "route": self.route,
            "refused": self.refused,
            "refuse_reason": self.refuse_reason,
            "answer": {
                "text": self.text,
                "n_filtered": self.n_filtered,
                "verdicts": [v.to_dict() for v in self.verdicts],
                "citations": [c.to_dict() for c in self.citations],
            },
            "topics": [t.to_dict() for t in self.topics],
            "graph": self.graph.to_dict() if self.graph is not None else None,
            "usage": self.usage.to_dict(),
            "warnings": list(self.warnings),
            "error": self.error,
            "meta": dict(self.meta),
        }


@dataclass
class WeekInfo:
    """One selectable ISO week exposed by ``GET /api/meta``."""

    week_id: str
    status: str = "unknown"
    member_count: int = 0
    latest_day: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class SourceInfo:
    key: str
    label: str
    papers: int = 0
    indexed_papers: int = 0
    index_ready: bool = False
    new_this_week: int = 0
    note: str = ""
    weeks: list[WeekInfo] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class MetaResponse:
    """能力与参数自描述。前端拿它渲染设置页、判断哪些按钮该亮。"""

    sources: list[SourceInfo] = field(default_factory=list)
    params: list[dict[str, Any]] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "sources": [s.to_dict() for s in self.sources],
            "params": list(self.params),
            "capabilities": dict(self.capabilities),
            "llm": dict(self.llm),
            "warnings": list(self.warnings),
        }


# ══════════════════════════════════════════════════════════════════════
# 小工具
# ══════════════════════════════════════════════════════════════════════


def new_request_id() -> str:
    """请求 ID（日志/排障用）。**不要进 prompt**——会打破前缀缓存。"""
    return f"{int(time.time())}-{uuid.uuid4().hex[:8]}"


def stage_event(key: str, *, cur: Optional[int] = None,
                total: Optional[int] = None, elapsed: float = 0.0) -> dict[str, Any]:
    """一条进度事件。`cur/total` 都给了才是可量化进度（前端才画进度条）。"""
    label = dict(STAGES).get(key, key)
    return {"type": "stage", "stage": key, "label": label,
            "cur": cur, "total": total, "elapsed": round(elapsed, 3)}


def arxiv_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""


def as_list(x: Iterable[Any]) -> list[Any]:
    return list(x)
