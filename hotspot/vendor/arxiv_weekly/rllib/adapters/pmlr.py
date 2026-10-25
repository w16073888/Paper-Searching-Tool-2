"""PMLR 适配器 —— 覆盖 ICML 2025 (v267) 与 CoRL 2025 (v305)。

为什么这个源最省：**索引页 1 次请求就拿到全部论文**，不需要逐篇抓。
而且 PDF 链接指向 raw.githubusercontent.com（PMLR 的出版仓库），
绕开了 proceedings.mlr.press 自身的限速和一切反爬。

实测的条目结构（2026-09-13）：

    <div class="paper">
      <p class="title">TITLE</p>
      <p class="details">
        <span class="authors">A,&nbsp;B,&nbsp;C</span>;
        <span class="info">..., PMLR 267:1-26</span>
      </p>
      <p class="links">
        [<a href=".../a-mancisidor25a.html">abs</a>]
        [<a href="https://raw.githubusercontent.com/.../a-mancisidor25a.pdf">Download PDF</a>]
        [<a href="https://openreview.net/forum?id=jYmGi1175R">OpenReview</a>]
      </p>
    </div>

**坑（务必不要"顺手优化"掉）**：OpenReview 那个 <a> 的 onclick 属性里写的是
``forum?id=false``，只有 href 才是真的 ID。任何对整个标签做正则的写法都会把
全部论文的 openreview_id 抓成 "false"。所以这里坚持用属性解析。
"""

from __future__ import annotations

import re
from datetime import date
from typing import Iterator

from bs4 import BeautifulSoup

from ..http import Fetcher
from ..schema import CanonicalPaper, Provenance

SOURCE = "pmlr"

_ABS_KEY_RE = re.compile(r"/(?P<key>[A-Za-z0-9\-]+)\.html$")


def parse_index(html: str, *, venue: str, year: int, volume: str) -> Iterator[CanonicalPaper]:
    """解析 PMLR 卷索引页，产出 CanonicalPaper。纯函数，便于离线测试。"""
    soup = BeautifulSoup(html, "html.parser")

    for block in soup.select("div.paper"):
        title_el = block.select_one("p.title")
        if title_el is None:
            continue
        title = _clean(title_el.get_text())

        authors_el = block.select_one("span.authors")
        authors = _parse_authors(authors_el.get_text() if authors_el else "")

        source_key = None
        pdf_url = None
        page_url = None
        openreview_id = None

        for anchor in block.select("p.links a"):
            href = (anchor.get("href") or "").strip()   # 只信 href，不信 onclick
            if not href:
                continue
            if href.endswith(".html") and "proceedings.mlr.press" in href:
                page_url = href
                m = _ABS_KEY_RE.search(href)
                if m:
                    source_key = m.group("key")
            elif href.endswith(".pdf") and "raw.githubusercontent.com" in href:
                pdf_url = href
                if source_key is None:
                    # 兜底：从 .../assets/<key>/<key>.pdf 里取 key
                    parts = href.rsplit("/", 2)
                    if len(parts) >= 2 and parts[-1].endswith(".pdf"):
                        source_key = parts[-1][:-4]
            elif "openreview.net/forum" in href:
                m = re.search(r"[?&]id=([A-Za-z0-9_\-]+)", href)
                if m and m.group(1) != "false":
                    openreview_id = m.group(1)

        if not source_key:
            continue    # 没有稳定 key 就放弃，宁可少收也不要造脏数据

        # PMLR 不区分 poster/oral/spotlight，track 留空而不是瞎填。
        # 页码在 span.info 里（PMLR 267:1-26），但 schema 目前没有 pages 字段，
        # 需要时再加——扩展 schema 只允许新增可选字段。
        yield CanonicalPaper(
            source_key=source_key,
            title=title,
            authors=authors,
            venue=venue,
            year=year,
            track=None,
            openreview_id=openreview_id,
            url=page_url,
            pdf_url=pdf_url,
            provenance=[
                Provenance(source=SOURCE, source_id=f"v{volume}/{source_key}",
                           fetched=date.today().isoformat())
            ],
        )


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _parse_authors(text: str) -> list[str]:
    """PMLR 用 `,&nbsp;` 分隔作者。"""
    text = text.replace("\xa0", " ").strip().rstrip(";").strip()
    if not text:
        return []
    return [a.strip() for a in text.split(",") if a.strip()]


def fetch(volume: str, venue: str, year: int, url: str, fetcher: Fetcher) -> list[CanonicalPaper]:
    html = fetcher.get_text(url)
    if html is None:
        return []
    return list(parse_index(html, venue=venue, year=year, volume=volume))
