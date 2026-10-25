"""摘要回补 —— 从会议源自己的单篇页取摘要。

**为什么必须单独做这一步**：三个索引页都不含摘要。
- PMLR 索引页：0 处 ``abstract``
- mlanthology 索引页：0 处 ``paper-abstract``
- 摘要只在**单篇页**上，所以这是逐篇请求

代价是约 12000 次请求（ICML/CoRL 3600 + NeurIPS/ICLR 9500），按 1 req/s
约 3.5 小时。但它是**免费的且零 token**，而且结果全量落盘缓存，
重跑不会重抓。对一次性建库完全可以接受。

替代方案是被否决的：OpenAlex 每天只给 100 次请求（见 datacite.py 的说明），
Semantic Scholar 无 key 直接 429。
"""

from __future__ import annotations

import logging
import re
from datetime import date

from bs4 import BeautifulSoup

from .http import Fetcher
from .store import PaperStore

log = logging.getLogger(__name__)

#: 每处理这么多篇存一次盘。长任务必须分段落盘——否则中途被打断就全丢了。
SAVE_EVERY = 100

#: 本任务认领的字段。落盘时只覆盖这些，磁盘上的其它字段以磁盘为准——
#: 这样它能和 enrich 并行跑而不互相抹掉。见 :meth:`rllib.store.PaperStore.save`。
OWNED_FIELDS = ("abstract", "provenance", "last_updated")


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\xa0", " "))
    return re.sub(r"^(Abstract|Summary)\s*[:\-—]?\s*", "", text.strip(), flags=re.I)


def _from_pmlr(html: str) -> str | None:
    """PMLR 单篇页：<div id="abstract" ...>正文</div>。

    页面上还有 Highwire 的 ``citation_abstract`` meta，但实测它是空的，
    真正的正文在 div 里。
    """
    soup = BeautifulSoup(html, "html.parser")
    el = soup.select_one("div#abstract") or soup.select_one("div.abstract")
    if el is None:
        return None
    # PMLR 在 div 里还放了一个 <h2>Abstract</h2>，_clean 会去掉
    return _clean(el.get_text()) or None


def _from_mlanthology(html: str) -> str | None:
    """mlanthology 单篇页：<div class=paper-abstract>正文</div>。"""
    soup = BeautifulSoup(html, "html.parser")
    el = soup.select_one("div.paper-abstract") or soup.select_one("[class*=paper-abstract]")
    if el is None:
        return None
    return _clean(el.get_text()) or None


def fetch_abstract(fetcher: Fetcher, page_url: str) -> str | None:
    """按 URL 自动分派到对应的解析器。"""
    if not page_url:
        return None
    html = fetcher.get_text(page_url)
    if html is None:
        return None
    if "proceedings.mlr.press" in page_url:
        return _from_pmlr(html)
    if "mlanthology.org" in page_url:
        return _from_mlanthology(html)
    return None


def backfill(
    store: PaperStore,
    fetcher: Fetcher,
    *,
    limit: int | None = None,
    save_path=None,
) -> dict[str, object]:
    """为缺少摘要的记录回补摘要。返回聚合统计。"""
    targets = [
        p for p in store.papers.values() if not p.abstract and p.url
    ]
    targets.sort(key=lambda p: p.paper_id)
    if limit:
        targets = targets[:limit]

    considered = len(targets)
    got = 0
    failed = 0
    no_url = sum(1 for p in store.papers.values() if not p.abstract and not p.url)

    for i, paper in enumerate(targets, 1):
        abstract = fetch_abstract(fetcher, paper.url)
        if abstract:
            paper.abstract = abstract
            paper.last_updated = date.today().isoformat()
            got += 1
        else:
            failed += 1

        # 分段落盘：跑 3 小时的任务不能只在结束时存一次
        if save_path is not None and i % SAVE_EVERY == 0:
            store.save(owned_fields=OWNED_FIELDS)

    if save_path is not None:
        store.save(owned_fields=OWNED_FIELDS)

    total = len(store.papers) or 1
    with_abstract = sum(1 for p in store.papers.values() if p.abstract)
    return {
        "considered": considered,
        "got": got,
        "failed": failed,
        "skipped_no_url": no_url,
        "abstract_coverage_pct": round(100 * with_abstract / total, 1),
        "cache": fetcher.stats,
    }
