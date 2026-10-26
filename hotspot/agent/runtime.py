"""PydanticAI controller for the paper-first research vertical slice."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import ModelConnectionError, UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from ..core.models import (
    AgentAnswer,
    PaperDetail,
    ScopeDecision,
    SearchHit,
    SearchResult,
    TopicOverview,
)
from ..tools.papers import fetch_papers
from ..tools.search import search_papers
from ..tools.topics import get_topic_overview
from .deps import AgentDeps


@dataclass(frozen=True)
class AgentRun:
    answer: AgentAnswer
    deps: AgentDeps
    scope: ScopeDecision
    usage: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


class ResearchAgentError(RuntimeError):
    def __init__(self, stage: str, code: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.code = code


INSTRUCTIONS = """你是本地论文库研究助手。只能依据工具返回的论文证据回答。

规则：
1. 问题涉及方法或论文时，先调用 search_papers；查询可用英文关键词。
2. 需要引用任何论文前，必须调用 fetch_papers 获取摘要；服务端已经注入的前文引用论文除外。
3. 比较两个对象时分别搜索两次，不要只搜一个合并字符串。
4. 问热门方向、研究趋势或领域概览时，先调用 get_topic_overview；它读取离线主题快照。
   若返回空 topics，必须改用 search_papers 按主题检索，再按年份/会议聚合；不要编造趋势。
5. 只能引用 fetched 中真实存在的 paper_id，不得编造 ID、标题或结论。
6. 每条 claim 必须带 evidence_ids；没有足够证据时返回空 claims，并填写 no_answer_reason。
7. 不回答本地论文库无法支持的外部知识问题。
8. 最多 4 次 search、3 次 fetch、2 次 topic overview；证据足够后停止。
9. 用户使用中文时，最终 claim 使用中文；论文标题保持原文。
10. 除非用户明确给出年份或会议名，否则不要猜测 years 或 venues 过滤条件。
11. 比较两个对象时，每个比较结论都必须能由对应对象的 fetched 摘要直接支持。
12. 不要把模型常识、论文标题推断或背景知识写进 claim；摘要没有写明的内容不要补充。
13. 方法列表按方法拆分 claim，每个 claim 只陈述一个可直接核验的方法要点。
14. 不要把多篇论文中互不相关的结论合并成一个 claim；每个 claim 只引用直接支持它的 1–2 个 ID。
"""

SCOPE_INSTRUCTIONS = """判断用户问题是否属于本地 AI/ML 论文库的研究范围。

范围内：强化学习、LLM 对齐与偏好优化、智能体、机器人学习、决策、相关机器学习。
范围外：医学/临床影像、生物信息、量子物理、材料、土木、大气、航天等核心领域问题。
跨领域应用只有在用户明确询问其中的 ML/RL 方法、模型或算法贡献时才判为范围内；
仅仅询问应用领域本身的热点、方法或进展时判为范围外。
只输出结构化判断，不要回答问题。
"""

AGENT_MODEL_SETTINGS = ModelSettings(
    temperature=0.1,
    top_p=0.8,
    seed=42,
)
HISTORY_ABSTRACT_CHARS = 1200
_COMPARISON_RE = re.compile(
    r"^(?P<left>[^，。？！]{1,80}?)\s*"
    r"(?:相比|与|和|vs\.?|versus|对比)\s*"
    r"(?P<right>[^，。？！]{1,120})",
    flags=re.IGNORECASE,
)
_COMPARISON_TAIL_RE = re.compile(
    r"\s*(?:在[^，。？！]{0,40}?上)?"
    r"(?:的)?(?:核心)?"
    r"(?:差异|区别|差别|共同来源|共同理论[^，。？！]*|"
    r"改了什么|是什么|有哪些|有什么|如何|怎么|为何).*$"
)


def comparison_terms(question: str) -> list[str]:
    """Extract two comparison subjects from a natural-language question."""
    match = _COMPARISON_RE.match(str(question or "").strip())
    if not match:
        return []
    parts = []
    for key in ("left", "right"):
        part = _COMPARISON_TAIL_RE.sub("", match.group(key)).strip(
            " \t，。？！?：:"
        )
        if part:
            parts.append(part)
    if len(parts) != 2:
        return []
    return list(dict.fromkeys(parts))


def merge_search_results(
    query: str,
    groups: list[SearchResult],
    *,
    limit: int,
) -> SearchResult:
    """Fuse ranked search lists with reciprocal-rank fusion."""
    scores: dict[str, float] = {}
    hits: dict[str, SearchHit] = {}
    order: list[str] = []
    for group in groups:
        for rank, hit in enumerate(group.results, start=1):
            paper_id = hit.paper_id
            if paper_id not in hits:
                hits[paper_id] = hit
                order.append(paper_id)
            scores[paper_id] = scores.get(paper_id, 0.0) + 1.0 / (60 + rank)
    rank_index = {paper_id: index for index, paper_id in enumerate(order)}
    ordered = sorted(
        scores,
        key=lambda paper_id: (-scores[paper_id], rank_index[paper_id]),
    )
    merged = [
        hits[paper_id].model_copy(update={"score": round(scores[paper_id], 6)})
        for paper_id in ordered[:limit]
    ]
    return SearchResult(
        query=query,
        results=merged,
        n_candidates=len(scores),
        truncated=len(scores) > limit,
    )


def build_history_context(
    history: list[dict],
    store,
    *,
    limit: int = 6,
) -> tuple[str, list[str]]:
    """Build follow-up context from prior assistant citation IDs."""
    citation_ids: list[str] = []
    for message in reversed(history):
        if message.get("role") != "assistant":
            continue
        for raw_id in message.get("citation_ids") or []:
            paper_id = str(raw_id).strip()
            if paper_id and paper_id not in citation_ids:
                citation_ids.append(paper_id)
    citation_ids = citation_ids[:max(1, min(int(limit), 20))]
    papers = store.get_papers(citation_ids) if citation_ids else []
    lines = []
    for paper in papers:
        abstract = str(paper.abstract or "").strip()
        if len(abstract) > HISTORY_ABSTRACT_CHARS:
            abstract = abstract[:HISTORY_ABSTRACT_CHARS].rstrip() + "…"
        lines.append(
            f"[P:{paper.paper_id}] {paper.title or ''}\n"
            f"摘要: {abstract}"
        )
    return "\n\n".join(lines), [paper.paper_id for paper in papers]


def _api_key(cfg: dict) -> str:
    llm = cfg.get("llm") or {}
    key = str(llm.get("api_key") or "").strip()
    env_name = str(llm.get("api_key_env") or "").strip()
    if not key and env_name:
        key = str(os.environ.get(env_name, "")).strip()
    if not key:
        raise RuntimeError("未配置 API key，无法启动算法 v2 agent")
    return key


def _normalize_base_url(url: str) -> str:
    value = str(url or "").strip().rstrip("/")
    suffix = "/chat/completions"
    return value[:-len(suffix)].rstrip("/") if value.endswith(suffix) else value


def _build_agent(cfg: dict) -> Agent[AgentDeps, AgentAnswer]:
    llm = cfg.get("llm") or {}
    base_url = _normalize_base_url(
        str(llm.get("base_url") or "https://api.deepseek.com/v1")
    )
    model_name = str(llm.get("model_pro") or llm.get("model_flash") or "deepseek-chat")
    provider = OpenAIProvider(base_url=base_url, api_key=_api_key(cfg))
    model = OpenAIChatModel(model_name, provider=provider)
    agent = Agent(
        model,
        deps_type=AgentDeps,
        output_type=AgentAnswer,
        instructions=INSTRUCTIONS,
        model_settings=AGENT_MODEL_SETTINGS,
        retries=1,
    )

    @agent.tool
    def search_tool(
        ctx: RunContext[AgentDeps],
        query: str,
        keywords: list[str] | None = None,
        years: list[int] | None = None,
        venues: list[str] | None = None,
        weeks: list[str] | None = None,
        limit: int = 0,
    ) -> SearchResult:
        """Search local papers. Use concise English queries and optional synonyms."""
        if ctx.deps.search_calls >= 4:
            raise ModelRetry("search budget exhausted; answer with current evidence")
        ctx.deps.search_calls += 1
        requested_years = ctx.deps.years or years or []
        requested_venues = ctx.deps.venues or venues or []
        requested_weeks = ctx.deps.weeks or weeks or []
        question_lower = ctx.deps.question.casefold()
        # LLMs sometimes invent metadata filters that eliminate every hit.
        # Hard filters are accepted only when the user explicitly wrote them.
        effective_years = [
            int(year) for year in requested_years
            if str(int(year)) in ctx.deps.question
        ]
        effective_venues = [
            str(venue) for venue in requested_venues
            if str(venue).strip()
            and str(venue).casefold() in question_lower
        ]
        effective_weeks = [
            str(week) for week in requested_weeks
            if str(week).strip()
            and (ctx.deps.weeks or str(week).casefold() in question_lower)
        ]
        model_limit = min(
            ctx.deps.search_limit,
            limit if limit and limit > 0 else ctx.deps.search_limit,
        )
        subqueries = comparison_terms(ctx.deps.question)
        effective_limit = (
            ctx.deps.search_limit if subqueries else model_limit
        )
        combined = search_papers(
            ctx.deps.store,
            query,
            keywords=keywords,
            years=effective_years,
            venues=effective_venues,
            weeks=effective_weeks,
            limit=effective_limit,
        )
        if subqueries:
            groups = [
                search_papers(
                    ctx.deps.store,
                    subquery,
                    keywords=[subquery],
                    years=effective_years,
                    venues=effective_venues,
                    weeks=effective_weeks,
                    limit=effective_limit,
                )
                for subquery in subqueries
            ]
            result = merge_search_results(
                query,
                [*groups, combined],
                limit=effective_limit,
            )
        else:
            result = combined
        ctx.deps.seen_ids.update(hit.paper_id for hit in result.results)
        ctx.deps.events.append({
            "tool": "search_papers",
            "query": query,
            "subqueries": subqueries,
            "keywords": list(keywords or []),
            "years": list(effective_years),
            "venues": list(effective_venues),
            "weeks": list(effective_weeks),
            "limit": effective_limit,
            "returned": len(result.results),
            "ids": [hit.paper_id for hit in result.results],
        })
        return result

    @agent.tool
    def fetch_tool(
        ctx: RunContext[AgentDeps],
        paper_ids: list[str],
        limit: int = 0,
    ) -> list[PaperDetail]:
        """Fetch title and abstract for paper IDs returned by search."""
        if ctx.deps.fetch_calls >= 3:
            raise ModelRetry("fetch budget exhausted; answer with fetched evidence")
        ctx.deps.fetch_calls += 1
        unknown = [pid for pid in paper_ids if pid not in ctx.deps.seen_ids]
        if unknown:
            raise ModelRetry(f"paper IDs were not returned by search: {unknown}")
        effective_limit = min(
            ctx.deps.fetch_limit,
            limit if limit and limit > 0 else ctx.deps.fetch_limit,
        )
        result = fetch_papers(ctx.deps.store, paper_ids, limit=effective_limit)
        ctx.deps.fetched.update(paper.paper_id for paper in result)
        ctx.deps.fetched_order.extend(
            paper.paper_id for paper in result
            if paper.paper_id not in ctx.deps.fetched_order
        )
        ctx.deps.events.append({
            "tool": "fetch_papers",
            "requested": len(paper_ids),
            "returned": len(result),
            "ids": [paper.paper_id for paper in result],
        })
        return result

    @agent.tool
    def topic_tool(
        ctx: RunContext[AgentDeps],
        query: str,
        limit: int = 5,
    ) -> TopicOverview:
        """Read precomputed topic clusters with venue/year spread for trend questions."""
        if ctx.deps.topic_calls >= 2:
            raise ModelRetry("topic overview budget exhausted")
        ctx.deps.topic_calls += 1
        result = get_topic_overview(ctx.deps.topics, query, limit=limit)
        ctx.deps.events.append({
            "tool": "get_topic_overview",
            "query": query,
            "returned": len(result.topics),
            "cluster_ids": [topic.cluster_id for topic in result.topics],
            "snapshot_source": ctx.deps.topics.source,
            "snapshot_error": ctx.deps.topics.error,
        })
        return result

    @agent.output_validator
    def validate_grounding(
        ctx: RunContext[AgentDeps],
        output: AgentAnswer,
    ) -> AgentAnswer:
        output.claims = [
            claim for claim in output.claims
            if any(str(pid).strip() for pid in claim.evidence_ids)
        ]
        if not output.claims:
            output.no_answer_reason = (
                output.no_answer_reason.strip()
                or "没有产生带可验证证据 ID 的 claim"
            )
            return output
        output.claims = output.claims[:ctx.deps.max_claims]
        used = {
            pid
            for claim in output.claims
            for pid in claim.evidence_ids
            if str(pid).strip()
        }
        if not used:
            raise ModelRetry("every claim must include at least one evidence ID")
        missing_fetch = sorted(used - ctx.deps.fetched)
        if missing_fetch:
            raise ModelRetry(
                "fetch these evidence IDs before answering: "
                + ", ".join(missing_fetch)
            )
        return output

    return agent


def _usage_dict(*usages) -> dict:
    names = (
        "requests", "tool_calls", "input_tokens", "output_tokens",
        "cache_read_tokens", "cache_write_tokens",
    )
    return {
        name: sum(int(getattr(usage, name, 0) or 0) for usage in usages)
        for name in names
    }


def _classify_scope(
    question: str,
    cfg: dict,
    *,
    timeout: float = 45.0,
) -> tuple[ScopeDecision, dict]:
    llm = cfg.get("llm") or {}
    provider = OpenAIProvider(
        base_url=_normalize_base_url(
            str(llm.get("base_url") or "https://api.deepseek.com/v1")
        ),
        api_key=_api_key(cfg),
    )
    model = OpenAIChatModel(
        str(llm.get("model_flash") or llm.get("model_pro") or "deepseek-chat"),
        provider=provider,
    )
    agent = Agent(
        model,
        output_type=ScopeDecision,
        instructions=SCOPE_INSTRUCTIONS,
        model_settings=AGENT_MODEL_SETTINGS,
        retries=0,
    )
    try:
        result = agent.run_sync(
            question,
            timeout=timeout,
            usage_limits=UsageLimits(request_limit=1),
        )
    except TimeoutError as exc:
        raise ResearchAgentError("scope", "scope_timeout", str(exc)) from exc
    except ModelConnectionError as exc:
        raise ResearchAgentError(
            "scope", "scope_connection_failed", str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise ResearchAgentError("scope", "scope_failed", str(exc)) from exc
    return result.output, _usage_dict(result.usage)


def run_research_agent(
    question: str,
    *,
    cfg: dict,
    source: str = "meeting",
    scope: dict | None = None,
    history: list[dict] | None = None,
    timeout: float = 180.0,
) -> AgentRun:
    scope = scope or {}
    deps = AgentDeps(
        cfg=cfg,
        question=question,
        source=source,
        years=[int(y) for y in (scope.get("years") or [])],
        venues=[str(v) for v in (scope.get("venues") or [])],
        weeks=[str(scope["week_id"])] if scope.get("week_id") else [],
    )
    if not deps.store.available:
        raise RuntimeError(
            "LanceDB index missing; run scripts/build_lance_index.py"
        )
    conversation = ""
    if history:
        conversation = "\n".join(
            f"{item.get('role')}: {item.get('text')}"
            for item in history[-6:]
            if item.get("text")
        )
    scope_prompt = question
    if conversation:
        scope_prompt = f"最近对话：\n{conversation}\n\n当前问题：{question}"

    history_context, history_ids = build_history_context(
        history or [],
        deps.store,
        limit=deps.fetch_limit,
    )
    deps.history_citation_ids = list(history_ids)
    if history_ids:
        deps.seen_ids.update(history_ids)
        deps.fetched.update(history_ids)
        deps.fetched_order.extend(
            paper_id for paper_id in history_ids
            if paper_id not in deps.fetched_order
        )
    agent_prompt = scope_prompt
    if history_context:
        agent_prompt += (
            "\n\n前文引用论文（服务端已读取摘要，可直接作为证据）：\n"
            + history_context
        )
    scope_started = time.perf_counter()
    scope_decision, scope_usage = _classify_scope(
        scope_prompt, cfg, timeout=deps.scope_timeout_s,
    )
    scope_elapsed = round(time.perf_counter() - scope_started, 3)
    if not scope_decision.in_scope:
        return AgentRun(
            answer=AgentAnswer(no_answer_reason=scope_decision.reason),
            deps=deps,
            scope=scope_decision,
            usage=scope_usage,
            timings={"scope_s": scope_elapsed},
        )

    agent_started = time.perf_counter()
    try:
        result = _build_agent(cfg).run_sync(
            agent_prompt,
            deps=deps,
            timeout=deps.agent_timeout_s,
            usage_limits=UsageLimits(
                request_limit=deps.request_limit,
                tool_calls_limit=deps.tool_limit,
            ),
        )
    except TimeoutError as exc:
        raise ResearchAgentError("agent", "agent_timeout", str(exc)) from exc
    except ModelConnectionError as exc:
        raise ResearchAgentError(
            "agent", "agent_connection_failed", str(exc),
        ) from exc
    except UnexpectedModelBehavior as exc:
        raise ResearchAgentError(
            "agent", "agent_output_invalid", str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise ResearchAgentError("agent", "agent_failed", str(exc)) from exc
    agent_elapsed = round(time.perf_counter() - agent_started, 3)
    combined_usage = _usage_dict(result.usage)
    for key, value in scope_usage.items():
        combined_usage[key] = int(combined_usage.get(key, 0)) + int(value)
    return AgentRun(
        answer=result.output,
        deps=deps,
        scope=scope_decision,
        usage=combined_usage,
        timings={"scope_s": scope_elapsed, "agent_s": agent_elapsed},
    )
