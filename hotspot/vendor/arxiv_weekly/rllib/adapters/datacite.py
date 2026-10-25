"""DataCite 适配器 —— arXiv ID 解析（替代 OpenAlex）。

**为什么换掉 OpenAlex**：2026-09-13 实测发现 OpenAlex 已改为按量计费，
响应头写着 ``x-ratelimit-limit: 1000`` / ``x-ratelimit-credits-required: 10``，
即**每天仅 100 次请求**，超出后 ``retry-after: 40554``（约 11 小时）。
3593 篇要跑 36 天，方案不可行。

**为什么用 DataCite**：arXiv 的 DOI 是 DataCite 注册的（不是 Crossref），
所以按标题查 DataCite 能直接拿到 ``10.48550/arxiv.XXXX``。实测 8 篇随机样本
命中 7 篇，且命中的 7 篇标题相似度**全部为 1.00**（零误配），免费、无需 key。

匹配纪律与 OpenAlex 路线相同：宽召回 + 严判定，**宁可未匹配也绝不错配**。
"""

from __future__ import annotations

import difflib
import re
import urllib.parse
from typing import Any

from ..http import Fetcher
from ..schema import CanonicalPaper, normalize_title

SOURCE = "datacite"

DOI_API = "https://api.datacite.org/dois"

#: 10.48550/arxiv.2506.03157  与老式 10.48550/arXiv.cs/0701001
_ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(.+)$", re.I)


def extract_arxiv_id(doi: str | None) -> str | None:
    if not doi:
        return None
    m = _ARXIV_DOI.match(doi.strip())
    if not m:
        return None
    ax = m.group(1).strip()
    # 现代格式形如 2506.03157（可能带 vN）；老格式形如 cs/0701001
    ax = re.sub(r"v\d+$", "", ax, flags=re.I)
    return ax or None


def _creators(attributes: dict[str, Any]) -> list[str]:
    """产出 ``"Given Family"`` 形式的作者名。

    **不要直接取 DataCite 的 ``name`` 字段**：它的格式是 ``"Family, Given"``，
    而 :func:`rllib.adapters.openalex._last_names` 是取**末词**当姓氏的。
    直接把 ``name`` 传进去，取到的是**名**，拿去和 PMLR 的 ``"Given Family"``
    求交集必然为空 —— 于是每一篇候选都被作者交叉校验否掉。

    实测这一处的代价：命中率从 70% 掉到 5%，而且**不报任何错**，只是安静地
    判定"标题相似度不够"，极难从聚合数字上看出来。
    """
    out: list[str] = []
    for c in attributes.get("creators") or []:
        family = (c.get("familyName") or "").strip()
        given = (c.get("givenName") or "").strip()
        if family:
            out.append(f"{given} {family}".strip())
            continue
        raw = (c.get("name") or "").strip()
        if "," in raw:                      # "Family, Given" → 反转
            fam, _, giv = raw.partition(",")
            raw = f"{giv.strip()} {fam.strip()}".strip()
        out.append(raw)
    return out


def _candidate_title(attributes: dict[str, Any]) -> str:
    titles = attributes.get("titles") or []
    return (titles[0].get("title") if titles else "") or ""


def search(
    paper: CanonicalPaper, fetcher: Fetcher, *, refresh: bool = False
) -> list[dict[str, Any]]:
    """按标题检索 DataCite 记录。

    **查询词必须只保留字母数字和空格**。DataCite 的 ``query`` 参数走的是
    Lucene 查询语法，``-`` 是取反、``:`` 是字段分隔、``()`` 是分组、``/`` 是
    正则定界——标题里这些标点一旦传进去，查询会被解析成完全不同的东西，
    结果多半是静默返回 0 条（HTTP 还是 200，看不出错）。
    实测：保留标点时命中 0/12，只留字母数字后命中率恢复正常。

    清洗只影响**召回**；判定用的是完整归一化标题的相似度，精度不受影响。

    ``refresh=True`` 绕过本地缓存重新请求（测真实延迟时要用，否则命中的
    都是缓存、测出来的不是网络耗时）。
    """
    q = re.sub(r"\\[a-zA-Z]+", " ", paper.title)   # 先去 LaTeX 命令
    q = re.sub(r"[^A-Za-z0-9 ]+", " ", q)          # 再只留字母数字与空格
    q = re.sub(r"\s+", " ", q).strip()

    url = f"{DOI_API}?query=titles.title:{urllib.parse.quote(q)}&page[size]=5"
    raw = fetcher.get_bytes(url, suffix=".json", refresh=refresh)
    if raw is None:
        return []
    import json

    try:
        return json.loads(raw).get("data") or []
    except json.JSONDecodeError:
        return []


def pick_arxiv(
    records: list[dict[str, Any]],
    paper: CanonicalPaper,
    *,
    threshold: float = 0.9,
    year_tolerance: int = 1,
) -> tuple[str, float] | None:
    """从候选中挑出置信度最高的 arXiv ID。

    只接受两类证据：① 标题高度相似；② 至少一位作者姓氏重合。
    返回 ``(arxiv_id, similarity)``，都不达标就返回 None。
    """
    from .openalex import _last_names   # 复用同一套姓氏提取逻辑

    paper_authors = _last_names(paper.authors)
    best: tuple[str, float] | None = None

    for record in records:
        attrs = record.get("attributes") or {}
        arxiv_id = extract_arxiv_id(attrs.get("doi"))
        if not arxiv_id:
            continue

        sim = difflib.SequenceMatcher(
            None, normalize_title(_candidate_title(attrs)), paper.title_norm
        ).ratio()
        if sim < threshold:
            continue

        year = attrs.get("publicationYear")
        if year and abs(int(year) - paper.year) > year_tolerance:
            continue

        if paper_authors and not (_last_names(_creators(attrs)) & paper_authors):
            continue

        if best is None or sim > best[1]:
            best = (arxiv_id, sim)

    return best
