"""Mandatory evidence rendering and citation verification for agent output."""

from __future__ import annotations

import concurrent.futures
import re
from dataclasses import dataclass, field

from ..api.contract import Citation, Verdict as ContractVerdict
from ..core.models import AgentAnswer, AnswerClaim
from ..llm import make_client
from ..schema import Claim
from .verify import verify_claims

_CITE = re.compile(r"\[[PC]:[^\]]+\]")


@dataclass(frozen=True)
class GroundingResult:
    text: str = ""
    refused: bool = False
    refuse_reason: str = ""
    verdicts: tuple[ContractVerdict, ...] = ()
    citation_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    verification_status: str = "ok"
    usage: dict = field(default_factory=dict)


def _render(text: str, evidence_ids: list[str]) -> str:
    clean = _CITE.sub("", text or "").strip()
    marks = "".join(f"[P:{pid}]" for pid in evidence_ids)
    if clean and clean[-1] in "。.!?！？":
        return f"{clean[:-1]}{marks}{clean[-1]}"
    return f"{clean} {marks}".strip()


def verify_and_render(
    answer: AgentAnswer,
    *,
    corpus,
    cfg: dict,
    client=None,
    timeout_s: float = 30.0,
    fallback_ids: list[str] | tuple[str, ...] = (),
) -> GroundingResult:
    """Enforce fetched IDs, citation verification, and deterministic rendering."""
    if not answer.claims:
        reason = answer.no_answer_reason.strip() or "没有找到足以回答问题的论文证据"
        return GroundingResult(refused=True, refuse_reason=reason)

    evidence_ids = list(dict.fromkeys(
        pid
        for item in answer.claims
        for pid in item.evidence_ids
        if str(pid).strip()
    ))
    details = {
        pid: corpus.get(pid)
        for pid in evidence_ids
    }
    missing = [pid for pid, paper in details.items() if paper is None]
    if missing:
        return GroundingResult(
            refused=True,
            refuse_reason=f"证据 ID 不在本地语料中：{missing}",
        )

    schema_claims = [
        Claim(
            text=item.text,
            cited_ids=tuple(str(pid) for pid in item.evidence_ids),
        )
        for item in answer.claims
    ]
    evidence_lookup = {
        pid: f"{paper.title or ''}\n{paper.abstract or ''}"
        for pid, paper in details.items()
        if paper is not None
    }
    client = client or make_client(cfg, timeout=timeout_s)
    usage_names = ("calls", "prompt_tokens", "completion_tokens", "failures")
    usage_before = {
        name: int(getattr(client.usage, name, 0) or 0)
        for name in usage_names
    }
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        verify_claims,
        schema_claims,
        allowed_ids=frozenset(evidence_ids),
        evidence_lookup=evidence_lookup,
        client=client,
        cfg=cfg,
        batch=True,
    )
    verify_timed_out = False
    verify_error = ""
    try:
        verdicts = future.result(timeout=max(1.0, timeout_s))
    except concurrent.futures.TimeoutError:
        verify_timed_out = True
        executor.shutdown(wait=False, cancel_futures=True)
        verdicts = []
    except Exception as exc:  # noqa: BLE001
        verify_error = f"{type(exc).__name__}: {exc}"
        executor.shutdown(wait=False, cancel_futures=True)
        verdicts = []
    else:
        executor.shutdown(wait=False)

    usage_after = {
        name: int(getattr(client.usage, name, 0) or 0)
        for name in usage_names
    }
    verify_usage = {
        name: usage_after[name] - usage_before[name]
        for name in usage_names
    }
    failures_after = usage_after["failures"]
    failures_before = usage_before["failures"]
    verification_failed = bool(
        verify_timed_out or verify_error or failures_after > failures_before
    )

    rendered = []
    contract_verdicts = []
    used_ids = []
    for verdict in verdicts:
        claim = verdict.claim
        text = _render(claim.text, list(claim.cited_ids))
        contract_verdicts.append(ContractVerdict(
            status=verdict.status,
            text=text,
            detail=verdict.detail,
        ))
        if verdict.status == "supported":
            rendered.append(text)
            used_ids.extend(claim.cited_ids)

    if not rendered:
        if verification_failed and fallback_ids:
            local = local_extractive_ground(
                corpus,
                fallback_ids,
                reason=(
                    "引用校验超时，已退回本地抽取式证据"
                    if verify_timed_out
                    else "引用校验失败，已退回本地抽取式证据"
                ),
                status="timeout_fallback" if verify_timed_out else "failed_fallback",
                usage=verify_usage,
            )
            if local is not None:
                return local
        return GroundingResult(
            refused=True,
            refuse_reason="所有候选 claim 都未通过引用校验",
            verdicts=tuple(contract_verdicts),
            citation_ids=tuple(dict.fromkeys(used_ids)),
            warnings=(
                (verify_error or "引用校验超时")
                if verification_failed else ()
            ),
            verification_status=(
                "timeout" if verify_timed_out
                else "failed" if verification_failed
                else "unsupported"
            ),
            usage=verify_usage,
        )
    return GroundingResult(
        text="\n".join(rendered),
        verdicts=tuple(contract_verdicts),
        citation_ids=tuple(dict.fromkeys(used_ids)),
    )


def local_extractive_ground(
    corpus,
    fetched_ids: list[str] | tuple[str, ...],
    *,
    reason: str,
    status: str,
    usage: dict | None = None,
) -> GroundingResult | None:
    answer = extractive_fallback(corpus, fetched_ids)
    if not answer.claims:
        return None
    rendered = []
    verdicts = []
    used_ids = []
    for item in answer.claims:
        text = _render(item.text, item.evidence_ids)
        rendered.append(text)
        used_ids.extend(item.evidence_ids)
        verdicts.append(ContractVerdict(
            status="supported",
            text=text,
            detail="本地抽取式证据降级",
        ))
    return GroundingResult(
        text="\n".join(rendered),
        verdicts=tuple(verdicts),
        citation_ids=tuple(dict.fromkeys(used_ids)),
        warnings=(reason,),
        verification_status=status,
        usage=dict(usage or {}),
    )


def make_citations(corpus, citation_ids: list[str] | tuple[str, ...]) -> list[Citation]:
    out = []
    for pid in dict.fromkeys(str(x) for x in citation_ids):
        paper = corpus.get(pid)
        if paper is None:
            continue
        out.append(Citation(
            id=pid,
            kind="paper",
            title=paper.title or "",
            venue=paper.venue or "",
            year=paper.year,
            week_id=getattr(paper, "week_id", "") or "",
            first_seen=getattr(paper, "first_seen", "") or "",
            abstract=(paper.abstract or "")[:500],
            arxiv_id=getattr(paper, "arxiv_id", "") or "",
            url=getattr(paper, "url", "") or "",
        ))
    return out


def extractive_fallback(
    corpus,
    fetched_ids: list[str] | tuple[str, ...],
    *,
    limit: int = 3,
) -> AgentAnswer:
    """Create a minimal grounded answer when the model cannot emit valid claims."""
    claims = []
    for pid in dict.fromkeys(str(x) for x in fetched_ids):
        paper = corpus.get(pid)
        if paper is None:
            continue
        snippet = re.sub(r"\s+", " ", paper.abstract or paper.title or "").strip()
        if not snippet:
            continue
        if len(snippet) > 260:
            snippet = snippet[:260].rsplit(" ", 1)[0] + "…"
        claims.append(AnswerClaim(
            text=f"《{paper.title or pid}》指出：{snippet}",
            evidence_ids=[pid],
        ))
        if len(claims) >= limit:
            break
    if not claims:
        return AgentAnswer(no_answer_reason="没取到可抽取的论文证据")
    return AgentAnswer(
        claims=claims,
        limitations=["模型未产出合法结构化 claim，已使用抽取式证据降级"],
    )
