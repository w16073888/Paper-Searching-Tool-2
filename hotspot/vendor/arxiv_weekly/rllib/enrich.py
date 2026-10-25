"""富化驱动：解析 arXiv ID 并写回库。

**来源变更说明（2026-09-13）**：原方案用 OpenAlex，但实测它已改为按量计费，
每天仅 100 次请求（``x-ratelimit-credits-required: 10`` × ``limit: 1000``），
3593 篇要跑 36 天。改用 **DataCite**（arXiv DOI 的注册机构）：免费、无需 key，
8 篇随机样本命中 7 篇且命中的标题相似度全为 1.00。

**引用数目前无来源**：OpenAlex 超限、Semantic Scholar 无 key 直接 429。
``citations`` 字段保留为 null，等接入可用的引用源再填——schema 不删字段，
下游按"可能为空"处理即可。

匹配纪律：宽召回 + 严判定，**宁可未匹配也绝不错配**。未匹配明细全量落盘。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .adapters import datacite
from .http import Fetcher
from .schema import CanonicalPaper, Provenance, normalize_doi
from .store import PaperStore

log = logging.getLogger(__name__)

#: 每处理这么多篇存一次盘。长任务分段落盘——中途被打断不至于全丢。
SAVE_EVERY = 50

#: 本任务认领的字段。落盘时只覆盖这些，磁盘上的其它字段以磁盘为准——
#: 这样它能和 backfill-abstracts 并行跑而不互相抹掉。见 :meth:`PaperStore.save`。
OWNED_FIELDS = (
    "arxiv_id", "doi", "enrich_attempted", "provenance", "last_updated",
    "citations", "citations_updated",
)


@dataclass
class EnrichReport:
    considered: int = 0
    matched: int = 0
    unmatched: int = 0
    no_candidates: int = 0
    got_arxiv: int = 0
    unmatched_detail: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "considered": self.considered,
            "matched": self.matched,
            "unmatched": self.unmatched,
            "match_rate": round(self.matched / self.considered, 3) if self.considered else 0.0,
            "no_candidates": self.no_candidates,
            "got_arxiv": self.got_arxiv,
        }


def needs_enrichment(
    paper: CanonicalPaper, retry_days: int = 30, refresh_citations: bool = False
) -> bool:
    """判断是否值得为这篇去查一次。

    两个作用：① 已有 arXiv ID 的跳过；② **上次尝试过但没匹配上的，在
    retry_days 内不再重查**——否则每周更新都会把注定匹配不上的论文再问一遍。
    之所以是"过 N 天再试"而非永久放弃，是因为 DOI 注册与收录都有滞后。
    """
    if paper.arxiv_id:
        return False
    if paper.enrich_attempted:
        try:
            attempted = date.fromisoformat(paper.enrich_attempted)
        except ValueError:
            return True
        if (date.today() - attempted).days < retry_days:
            return False
    return True


def reset_attempts(store: PaperStore) -> int:
    """清空所有"缺 arXiv ID 的论文"的 ``enrich_attempted``，让它们重回待查队列。

    什么时候需要：改了匹配逻辑之后（比如修了一个让命中率虚低的 bug），
    原来被判"试过了、没匹配上"的论文会被 ``retry_after_days`` 挡住 30 天，
    改了代码也吃不到效果。返回被重置的篇数。
    """
    n = 0
    for paper in store.papers.values():
        if not paper.arxiv_id and paper.enrich_attempted:
            paper.enrich_attempted = None
            n += 1
    return n


def apply_arxiv(paper: CanonicalPaper, arxiv_id: str, similarity: float, doi: str | None) -> None:
    """写回 arXiv ID。**不动 title/venue** —— 那些以会议源为准。"""
    if not paper.arxiv_id:
        paper.arxiv_id = arxiv_id
    if not paper.doi:
        d = normalize_doi(doi)
        if d:
            paper.doi = d
    if not any(p.source == datacite.SOURCE for p in paper.provenance):
        paper.provenance.append(
            Provenance(
                source=datacite.SOURCE,
                source_id=f"arxiv:{arxiv_id} (sim={similarity:.3f})",
            )
        )
    paper.last_updated = date.today().isoformat()


def enrich(
    store: PaperStore,
    fetcher: Fetcher,
    *,
    limit: int | None = None,
    retry_days: int = 30,
    refresh_citations: bool = False,
    report_path: Path | None = None,
) -> EnrichReport:
    report = EnrichReport()
    today = date.today().isoformat()

    targets = [
        p for p in store.papers.values()
        if needs_enrichment(p, retry_days, refresh_citations)
    ]
    targets.sort(key=lambda p: p.paper_id)
    if limit:
        targets = targets[:limit]

    total = len(targets)
    for i, paper in enumerate(targets, 1):
        report.considered += 1
        paper.enrich_attempted = today      # 无论成败都记下"试过"

        records = datacite.search(paper, fetcher)
        if not records:
            report.no_candidates += 1
            report.unmatched += 1
            report.unmatched_detail.append(
                {"paper_id": paper.paper_id, "reason": "no_candidates", "title": paper.title}
            )
        else:
            hit = datacite.pick_arxiv(records, paper)
            if hit is None:
                report.unmatched += 1
                report.unmatched_detail.append(
                    {
                        "paper_id": paper.paper_id,
                        "reason": "below_threshold",
                        "title": paper.title,
                        "candidates": len(records),
                    }
                )
            else:
                arxiv_id, sim = hit
                apply_arxiv(paper, arxiv_id, sim, _doi_of(records, arxiv_id))
                report.matched += 1
                report.got_arxiv += 1

        if i % SAVE_EVERY == 0:
            store.save(owned_fields=OWNED_FIELDS)

    store.save(owned_fields=OWNED_FIELDS)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {"summary": report.as_dict(), "unmatched": report.unmatched_detail},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return report


def _doi_of(records: list[dict], arxiv_id: str) -> str | None:
    for record in records:
        doi = (record.get("attributes") or {}).get("doi")
        if doi and datacite.extract_arxiv_id(doi) == arxiv_id:
            return doi
    return None
