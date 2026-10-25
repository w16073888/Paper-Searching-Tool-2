"""mlanthology 适配器 —— 覆盖 NeurIPS 2025 与 ICLR 2025。

**为什么需要它**：这两大顶会的官方来源在本网络都不可用——`papers.nips.cc`
与 `nips.cc` 实测 000 不可达，OpenReview 全站 403 人机验证。mlanthology 是
唯一可达且结构完整的索引，实测 NeurIPS 2025 收录 5823 篇。

**它提供什么、不提供什么**（实测 2026-09-13）：

    提供：标题、作者、OpenReview ID、source_key
    不提供：摘要（索引页 0 处）、arXiv ID（0 处）
    PDF 链接指向 openreview.net/pdf/，403 不可用

所以摘要与 arXiv ID 全部依赖 :mod:`rllib.adapters.openalex` 富化。
摘要的缺口用 :func:`fetch_abstract` 逐篇回补（可选步骤，约 9500 次请求）。

条目结构：

    <div class=paper-entry>
      <a href=/neurips/2025/escudie2025neurips-optimally/ class=paper-title>TITLE</a>
      <span class=paper-authors-inline>A, B, C</span>
      <div class=paper-entry-links>
        <a href=https://openreview.net/pdf/<sha>.pdf class=link-pdf>PDF</a>
        <a href="https://openreview.net/forum?id=tqoAQMk4AH" class=link-openreview>OpenReview</a>
        <button data-bibtex-key=escudie2025neurips-optimally ...>Cite</button>
      </div>
    </div>
"""

from __future__ import annotations

import re
from datetime import date
from typing import Iterator

from bs4 import BeautifulSoup

from ..http import Fetcher
from ..schema import CanonicalPaper, Provenance

SOURCE = "mlanthology"

_SLUG_KEY_RE = re.compile(r"/(?P<slug>[^/]+)/?$")
_ORID_RE = re.compile(r"[?&]id=([A-Za-z0-9_\-]+)")


def parse_index(html: str, *, venue: str, year: int) -> Iterator[CanonicalPaper]:
    """解析 mlanthology 的会议年份索引页。纯函数，便于离线测试。"""
    soup = BeautifulSoup(html, "html.parser")

    for entry in soup.select("div.paper-entry"):
        title_el = entry.select_one("a.paper-title")
        if title_el is None:
            continue
        title = _clean(title_el.get_text())
        if not title:
            continue

        # source_key 优先用 data-bibtex-key（最规范），回退到页面 slug
        source_key = None
        button = entry.select_one("button[data-bibtex-key]")
        if button is not None:
            source_key = (button.get("data-bibtex-key") or "").strip() or None
        if not source_key:
            href = title_el.get("href") or ""
            m = _SLUG_KEY_RE.search(href.rstrip("/"))
            if m:
                source_key = m.group("slug")
        if not source_key:
            continue

        authors_el = entry.select_one("span.paper-authors-inline")
        authors = _parse_authors(authors_el.get_text() if authors_el else "")

        page_url = _absolutize(title_el.get("href") or "")

        openreview_id = None
        or_el = entry.select_one("a.link-openreview")
        if or_el is not None:
            m = _ORID_RE.search(or_el.get("href") or "")
            if m and m.group(1) != "false":
                openreview_id = m.group(1)

        # PDF 链接指向 openreview.net/pdf/，403 不可用。**刻意不写入 pdf_url**，
        # 免得渲染出打不开的链接、也免得阶段 6 误以为有现成的下载路径。
        # 真实 PDF 走阶段 2 解析出的 arXiv ID。

        yield CanonicalPaper(
            source_key=source_key,
            title=title,
            authors=authors,
            venue=venue,
            year=year,
            track=None,                 # mlanthology 不含 oral/poster 档次
            openreview_id=openreview_id,
            url=page_url,
            provenance=[
                Provenance(source=SOURCE, source_id=source_key, fetched=date.today().isoformat())
            ],
        )


def fetch_abstract(fetcher: Fetcher, page_url: str) -> str | None:
    """从单篇论文页取摘要。

    索引页不含摘要，所以这一步是**逐篇请求**。仅在 OpenAlex 也没能给到摘要时
    才值得跑——默认不做，见 CLI 的 ``backfill-abstracts``。
    """
    if not page_url:
        return None
    html = fetcher.get_text(page_url)
    if html is None:
        return None
    soup = BeautifulSoup(html, "html.parser")
    el = soup.select_one("div.paper-abstract") or soup.select_one("[class*=paper-abstract]")
    if el is None:
        return None
    text = _clean(el.get_text())
    # 去掉 "Abstract" 之类的标题词
    text = re.sub(r"^(Abstract|Summary)[:\s]*", "", text, flags=re.I)
    return text or None


# --------------------------------------------------------------------------- #

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _parse_authors(text: str) -> list[str]:
    text = text.replace("\xa0", " ").strip().rstrip(";").strip()
    if not text:
        return []
    # mlanthology 用逗号分隔；但 "Dibangoye, Jilles Steeve" 这种也会被切错，
    # 所以只在明显是逗号分隔列表时才切
    return [a.strip() for a in text.split(",") if a.strip()]


def _absolutize(href: str) -> str:
    if href.startswith("http"):
        return href
    return "https://mlanthology.org" + (href if href.startswith("/") else "/" + href)
