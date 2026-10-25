"""引用数回补：走 OpenAlex 的**批量**标题检索。

**为什么必须批量**：OpenAlex 每次请求固定 10 credits，与查询大小无关
（单标题和 50 标题实测都是 10）；日额度 1000 credits → **100 请求/天**。
逐篇查全库是 13119 个请求 = 131 天，批量 50 篇一包是 263 个请求 = **2.6 天**。
这不是优化，是"不可行"与"可行"的差别。

**为什么不用 arXiv ID 直查**：实测 ``filter=doi:10.48550/arxiv.2503.07199``
返回 ``count=0``——OpenAlex 不收录 arXiv DOI，而库里 9798 个 DOI **全是**
arXiv DOI（与 ``arxiv_id`` 一一对应）。所以只有标题匹配这一条路。

**一个实测得到的好消息**：OpenAlex 把 arXiv 版和会议版**合并成一个 work**
（一篇 NeurIPS 论文的 ``locations`` 里 arXiv 与 ``10.52202/...`` 会议 DOI
同时在），所以引用数是跨版本合计的，不需要分别抓。

匹配纪律与 :mod:`rllib.enrich` 一致：**宁可标记"未匹配"，也绝不错配**。
未匹配明细全量落盘，不要靠读终端判断质量。

**收益预期要放低**：实测 RL 子集 1380 篇匹配上的里面 **81.2% 引用数为 0**，
非零仅 259 篇，最大 29，全子集引用总数 487。因为库里全是 2025 年的会议论文，
太新了。所以这一步现在的价值是"把 null 变成真实的 0"，不是"立刻排出重要性"。
非零的主要是 arXiv 放得早的那些（2023/2024 preprint）。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .adapters import openalex
from .http import Fetcher
from .schema import CanonicalPaper, Provenance
from .store import PaperStore

log = logging.getLogger(__name__)

#: 每处理这么多篇存一次盘。长任务分段落盘——中途被打断不至于全丢。
SAVE_EVERY = 200

#: 本任务认领的字段。落盘时只覆盖这些，其余以磁盘为准——这样它能和
#: ``enrich`` / ``backfill-abstracts`` / ``extract`` 并行跑而不互相抹掉。
#:
#: ⚠️ **刻意不认领 ``arxiv_id``**：那是 ``enrich`` 的地盘，两个任务同时拥有
#: 一个字段就是并行覆盖的前兆。匹配时顺带看到的 arXiv ID 只进报告，不写库。
OWNED_FIELDS = (
    "citations", "citations_updated", "citations_attempted",
    "provenance", "last_updated",
)


@dataclass
class CitationsReport:
    considered: int = 0
    batches: int = 0
    matched: int = 0
    unmatched: int = 0
    zero: int = 0
    nonzero: int = 0
    total_citations: int = 0
    max_citations: int = 0
    failed_batches: int = 0
    failed_papers: int = 0
    unmatched_detail: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "considered": self.considered,
            "batches": self.batches,
            "failed_batches": self.failed_batches,
            "failed_papers": self.failed_papers,
            "matched": self.matched,
            "unmatched": self.unmatched,
            "match_rate": (
                round(self.matched / self.considered, 3) if self.considered else 0.0
            ),
            "zero": self.zero,
            "nonzero": self.nonzero,
            "total_citations": self.total_citations,
            "max_citations": self.max_citations,
        }


def needs_citations(
    paper: CanonicalPaper, retry_days: int = 30, refresh: bool = False
) -> bool:
    """判断是否值得为这篇去查一次。

    * 默认（``refresh=False``）**只补空的**：已有引用数就不动。全量刷新走
      ``refresh=True``，那是"每周更新"该干的事，不是补缺口该干的。
    * ``citations_attempted`` 在 ``retry_days`` 内的一律跳过——否则每次更新
      都会把注定匹配不上的论文再问一遍，而配额只有 100 请求/天。
      不永久放弃是因为 OpenAlex 的收录是滞后的，过阵子可能就有了。
    """
    if paper.citations is not None and not refresh:
        return False
    if paper.citations_attempted:
        try:
            attempted = date.fromisoformat(paper.citations_attempted)
        except ValueError:
            return True
        if (date.today() - attempted).days < retry_days:
            return False
    return True


def apply_citations(
    paper: CanonicalPaper, work: dict, similarity: float
) -> None:
    """写回引用数。**不动 title/venue/arxiv_id** —— 那些各有各的归属。"""
    paper.citations = int(work.get("cited_by_count") or 0)
    paper.citations_updated = date.today().isoformat()
    paper.last_updated = paper.citations_updated

    # provenance 走并集合并（只增不减），所以这里必须去重：每周刷新一次、
    # 每次追加一条的话，几年下来这个字段会漫长得没法读。
    wid = str(work.get("id") or "")
    if not any(
        p.source == openalex.SOURCE and (p.source_id or "").startswith(wid)
        for p in paper.provenance
    ):
        paper.provenance.append(
            Provenance(
                source=openalex.SOURCE,
                source_id=f"{wid} (sim={similarity:.3f})",
            )
        )


def backfill(
    store: PaperStore,
    fetcher: Fetcher,
    *,
    scope: str = "rl",
    limit: int | None = None,
    retry_days: int = 30,
    refresh: bool = False,
    base_url: str = "https://api.openalex.org/works",
    match_cfg: openalex.MatchConfig | None = None,
    report_path: Path | None = None,
) -> CitationsReport:
    """按批补引用数。``scope`` 为 ``rl`` 只补 RL 子集，``all`` 补全库。

    ``rl`` = 32 个请求（一天内跑完）；``all`` = 263 个请求（约 2.6 天）。
    """
    cfg = match_cfg or openalex.MatchConfig()
    report = CitationsReport()
    today = date.today().isoformat()

    targets = [
        p for p in store.papers.values()
        if (scope == "all" or p.is_rl_effective)
        and needs_citations(p, retry_days, refresh)
    ]
    targets.sort(key=lambda p: p.paper_id)
    if limit:
        targets = targets[:limit]

    processed = 0
    for batch in openalex.chunk_titles(targets):
        report.batches += 1
        works = openalex.fetch_batch(batch, fetcher, base_url)

        if works is None:
            # 请求失败 = 这批**从未被查询过**。绝不能写 citations_attempted：
            # 那会让这些论文被 retry_days 锁死，故障表现成"查过了没匹配上"，
            # 而真相是一次都没查成。留白，下轮重来。
            report.failed_batches += 1
            report.failed_papers += len(batch)
            log.warning("批量检索失败，跳过 %d 篇（未计入尝试）：%s",
                        len(batch), batch[0].title[:60] if batch else "")
            continue

        for paper in batch:
            report.considered += 1
            paper.citations_attempted = today
            cand = openalex.pick_best(works, paper, cfg)

            if cand is None:
                report.unmatched += 1
                report.unmatched_detail.append(
                    {
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "venue": f"{paper.venue} {paper.year}",
                        "candidates": len(works),
                    }
                )
            else:
                apply_citations(paper, cand.work, cand.similarity)
                report.matched += 1
                c = paper.citations or 0
                report.total_citations += c
                report.max_citations = max(report.max_citations, c)
                if c == 0:
                    report.zero += 1
                else:
                    report.nonzero += 1

            processed += 1
            if processed % SAVE_EVERY == 0:
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
