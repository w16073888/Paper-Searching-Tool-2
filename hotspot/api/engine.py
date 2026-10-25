"""Public API shell for the agent-v2 rebuild.

The legacy retrieval/orchestration implementation was removed. This module
keeps the stable ``meta``/``ask`` boundary importable while tools and the new
controller are assembled.
"""

from __future__ import annotations

import copy
import os
import threading
import time
from typing import Any, Optional

from ..agent.runtime import ResearchAgentError, run_research_agent
from ..config import load_config
from ..corpus import open_arxiv_corpus, open_corpus
from ..index.lance_store import LancePaperStore
from ..observability.trace import append_trace, sanitize_answer_text
from ..reasoning.grounding import (
    extractive_fallback,
    make_citations,
    verify_and_render,
)
from .contract import (
    E_CORPUS_UNAVAILABLE,
    E_INDEX_UNAVAILABLE,
    E_INTERNAL,
    E_NO_API_KEY,
    PARAM_SPEC,
    SOURCES,
    ApiError,
    AskRequest,
    AskResponse,
    MetaResponse,
    SourceInfo,
    Usage,
    apply_params,
    normalize_base_url,
    new_request_id,
    stage_event,
)

ProgressHook = Optional[Any]
_ENGINE: Optional["Engine"] = None


def _history_for_request(req: AskRequest) -> list[dict[str, Any]]:
    if req.mode != "followup":
        return []
    return [message.to_dict() for message in req.history]


class Engine:
    """Stable service shell; algorithm v2 will replace only the ask pipeline."""

    def __init__(self, cfg: dict | None = None):
        self._cfg = copy.deepcopy(cfg) if cfg is not None else load_config()
        self._corpora: dict[str, Any] = {}
        self._lock = threading.RLock()

    def _corpus(self, source: str):
        if source not in SOURCES:
            raise ApiError("bad_request", f"未知数据源：{source}")
        with self._lock:
            if source in self._corpora:
                return self._corpora[source]
            try:
                corpus = (
                    open_arxiv_corpus(self._cfg)
                    if source == "arxiv"
                    else open_corpus(self._cfg)
                )
            except Exception as exc:  # noqa: BLE001
                raise ApiError(
                    E_CORPUS_UNAVAILABLE,
                    f"数据源 {source!r} 打不开：{type(exc).__name__}: {exc}",
                    status=503,
                ) from exc
            self._corpora[source] = corpus
            return corpus

    def meta(self) -> MetaResponse:
        out = MetaResponse(params=[p.to_dict(self._cfg) for p in PARAM_SPEC])
        for key, label in SOURCES.items():
            info = SourceInfo(key=key, label=label)
            try:
                stats = self._corpus(key).stats()
                info.papers = int(stats.get("papers") or 0)
                store = LancePaperStore(self._cfg, source=key)
                info.index_ready = store.available
                if info.index_ready:
                    info.indexed_papers = store.count()
                    if info.indexed_papers < info.papers:
                        info.note = (
                            f"LanceDB 索引覆盖 {info.indexed_papers}/"
                            f"{info.papers} 篇"
                        )
                else:
                    info.note = "LanceDB 索引未构建，暂不可检索"
            except ApiError as exc:
                info.note = exc.message
                out.warnings.append(exc.message)
            out.sources.append(info)
        llm = self._cfg.get("llm") or {}
        key = str(llm.get("api_key") or "").strip()
        key_env = str(llm.get("api_key_env") or "").strip()
        if not key and key_env:
            key = str(os.environ.get(key_env) or "").strip()
        base_url = normalize_base_url(str(llm.get("base_url") or ""))
        model_flash = str(llm.get("model_flash") or "")
        model_pro = str(llm.get("model_pro") or "")
        out.llm = {
            "base_url": base_url,
            "model": model_pro or model_flash,
            "model_flash": model_flash,
            "model_pro": model_pro,
            "has_server_key": bool(key),
        }
        out.capabilities = {
            "algorithm_version": "2",
            "agent": "pydantic-ai",
            "retrieval": "lancedb-vector+fts",
            "topic_snapshot": True,
            "trace": True,
            "timeouts": {
                "scope_s": float(
                    ((self._cfg.get("agent") or {}).get("scope_timeout_s") or 30)
                ),
                "agent_s": float(
                    ((self._cfg.get("agent") or {}).get("agent_timeout_s") or 150)
                ),
                "verify_s": float(
                    ((self._cfg.get("agent") or {}).get("verify_timeout_s") or 30)
                ),
            },
            "stream": True,
            "dry_run": False,
            "sources": list(SOURCES),
        }
        return out

    def ask(
        self,
        request: AskRequest | dict[str, Any],
        on_event: ProgressHook = None,
    ) -> AskResponse:
        req = request if isinstance(request, AskRequest) else AskRequest.from_dict(request)
        started = time.perf_counter()
        cfg = copy.deepcopy(self._cfg)
        req.llm.apply_to(cfg)
        apply_params(cfg, req.params)
        request_warnings = []
        request_warnings.extend(req.scope_warnings)
        if req.question_truncated:
            request_warnings.append("问题超过 2000 字，已截断。")
        if req.history_truncated:
            request_warnings.append("历史消息超过 40 条，仅使用最近 40 条。")

        def finish(response: AskResponse, run=None) -> AskResponse:
            response.warnings = list(dict.fromkeys([
                *request_warnings,
                *response.warnings,
            ]))
            actions = list(run.deps.events) if run is not None else []
            append_trace(cfg, {
                "request_id": response.request_id,
                "source": response.source,
                "mode": response.mode,
                "question": req.question,
                "question_sha256": sanitize_answer_text(req.question)["sha256"],
                "plan": {"scope": dict(req.scope or {})},
                "scope_decision": (
                    run.scope.model_dump() if run is not None else None
                ),
                "timings": dict(run.timings) if run is not None else {},
                "actions": actions,
                "outcome": {
                    "route": response.route,
                    "refused": response.refused,
                    "refuse_reason": response.refuse_reason,
                    "answer": sanitize_answer_text(response.text),
                    "citation_ids": [citation.id for citation in response.citations],
                    "verdicts": [
                        {
                            "status": verdict.status,
                            "text": sanitize_answer_text(verdict.text),
                            "detail_sha256": sanitize_answer_text(verdict.detail)["sha256"],
                        }
                        for verdict in response.verdicts
                    ],
                    "warnings": list(response.warnings),
                    "error": response.error,
                    "elapsed_s": response.usage.elapsed_s,
                    "agent": (
                        {
                            "search_calls": run.deps.search_calls,
                            "fetch_calls": run.deps.fetch_calls,
                            "topic_calls": run.deps.topic_calls,
                        }
                        if run is not None else {}
                    ),
                    "verification_status": (
                        ((response.meta or {}).get("agent") or {}).get(
                            "verification_status"
                        )
                    ),
                },
            })
            return response

        corpus = self._corpus(req.source)
        store = LancePaperStore(cfg, source=req.source)
        if not store.available:
            raise ApiError(
                E_INDEX_UNAVAILABLE,
                f"数据源 {req.source!r} 的 LanceDB 索引不存在；"
                "先运行 scripts/build_lance_index.py",
                status=503,
            )
        if req.dry_run:
            message = "算法 v2 需要 LLM 与 LanceDB，dry-run 暂不生成回答。"
            return finish(AskResponse(
                request_id=new_request_id(),
                source=req.source,
                mode=req.mode,
                route="error",
                error={"code": E_NO_API_KEY, "message": message},
                warnings=[message],
                meta={"agent": {"dry_run": True}},
            ))

        if on_event:
            on_event(stage_event("retrieve", elapsed=0.0))
        try:
            run = run_research_agent(
                req.question,
                cfg=cfg,
                source=req.source,
                scope=req.scope,
                history=_history_for_request(req),
            )
        except ResearchAgentError as exc:
            elapsed = round(time.perf_counter() - started, 3)
            return finish(AskResponse(
                request_id=new_request_id(),
                source=req.source,
                mode=req.mode,
                route="error",
                error={"code": exc.code, "message": str(exc)},
                warnings=[f"{exc.stage} 阶段失败：{exc}"],
                usage=Usage(elapsed_s=elapsed),
                meta={"agent": {"failed_stage": exc.stage}},
            ))
        except RuntimeError as exc:
            if "API key" in str(exc):
                raise ApiError(E_NO_API_KEY, str(exc)) from exc
            return finish(AskResponse(
                request_id=new_request_id(),
                source=req.source,
                mode=req.mode,
                route="error",
                error={"code": E_INTERNAL, "message": str(exc)},
                warnings=[str(exc)],
            ))
        except Exception as exc:  # noqa: BLE001
            message = f"算法 v2 agent 失败：{type(exc).__name__}: {exc}"
            return finish(AskResponse(
                request_id=new_request_id(),
                source=req.source,
                mode=req.mode,
                route="error",
                error={"code": E_INTERNAL, "message": message},
                warnings=[message],
            ))
        try:
            if on_event:
                on_event(stage_event("answer", elapsed=time.perf_counter() - started))
            answer = run.answer
            if not answer.claims and run.deps.fetched_order:
                answer = extractive_fallback(corpus, run.deps.fetched_order)
            verify_started = time.perf_counter()
            grounded = verify_and_render(
                answer,
                corpus=corpus,
                cfg=cfg,
                timeout_s=run.deps.verify_timeout_s,
                fallback_ids=run.deps.fetched_order,
            )
            verify_elapsed = round(time.perf_counter() - verify_started, 3)
        except Exception as exc:  # noqa: BLE001
            message = f"算法 v2 处理失败：{type(exc).__name__}: {exc}"
            return finish(AskResponse(
                request_id=new_request_id(),
                source=req.source,
                mode=req.mode,
                route="error",
                error={"code": E_INTERNAL, "message": message},
                warnings=[message],
                meta={"agent": {"tool_events": run.deps.events}},
            ), run)
        if on_event:
            on_event(stage_event("verify", elapsed=time.perf_counter() - started))

        elapsed = round(time.perf_counter() - started, 3)
        return finish(AskResponse(
            request_id=new_request_id(),
            source=req.source,
            mode=req.mode,
            route="refuse" if grounded.refused else "local",
            refused=grounded.refused,
            refuse_reason=grounded.refuse_reason,
            text=grounded.text,
            n_filtered=sum(
                1 for verdict in grounded.verdicts
                if verdict.status != "supported"
            ),
            verdicts=list(grounded.verdicts),
            citations=make_citations(corpus, grounded.citation_ids),
            warnings=list(dict.fromkeys([
                *run.answer.limitations,
                *grounded.warnings,
            ])),
            usage=Usage(
                llm_calls=(
                    int(run.usage.get("requests") or 0)
                    + int((grounded.usage or {}).get("calls") or 0)
                ),
                tokens_in=(
                    int(run.usage.get("input_tokens") or 0)
                    + int((grounded.usage or {}).get("prompt_tokens") or 0)
                ),
                tokens_out=(
                    int(run.usage.get("output_tokens") or 0)
                    + int((grounded.usage or {}).get("completion_tokens") or 0)
                ),
                failures=int((grounded.usage or {}).get("failures") or 0),
                elapsed_s=elapsed,
                stages={
                    "scope": float(run.timings.get("scope_s") or 0.0),
                    "agent": float(run.timings.get("agent_s") or 0.0),
                    "verify": verify_elapsed,
                    "total": elapsed,
                },
            ),
            meta={
                "agent": {
                    "search_calls": run.deps.search_calls,
                    "fetch_calls": run.deps.fetch_calls,
                    "topic_calls": run.deps.topic_calls,
                    "history_citation_ids": list(
                        run.deps.history_citation_ids
                    ),
                    "seen_ids": len(run.deps.seen_ids),
                    "fetched_ids": len(run.deps.fetched),
                    "tool_events": run.deps.events,
                    "timings": dict(run.timings),
                    "verify_s": verify_elapsed,
                    "verification_status": grounded.verification_status,
                },
                "answer_claims": [item.model_dump() for item in run.answer.claims],
                "scope": run.scope.model_dump(),
            },
        ), run)


def get_engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine()
    return _ENGINE


def ask(request: AskRequest | dict[str, Any], on_event: ProgressHook = None) -> dict[str, Any]:
    return get_engine().ask(request, on_event=on_event).to_dict()


def meta() -> dict[str, Any]:
    return get_engine().meta().to_dict()
