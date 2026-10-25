"""OpenAlex 适配器 —— 富化源，不是发现源。

作用：标题 → arXiv ID + 摘要 + 引用数 + topics。

**为什么需要它**：NeurIPS / ICLR 的全文只能从 arXiv 取（OpenReview 403、
nips.cc 不可达），而 PMLR 和 mlanthology 都不提供 arXiv ID。OpenAlex 能
（实测：ICML 2025 的一篇论文成功解析到 arxiv.2506.00756）。

**为什么它也是全库最危险的环节**：OpenAlex 按标题模糊检索，一旦匹配错，
别人的论文会悄悄混进库里，而且**不会被任何下游流程发现**——不报错、不缺字段、
看起来一切正常。所以这里的匹配是刻意保守的：

原则：**宁可标记"未匹配"，也绝不错配。**
未匹配只是少一篇的 arXiv ID（可事后补），错配是往库里注入污染（几乎无法事后发现）。


检索方式（2026-09-15 实测重写，**不要改回 ``search=``**）
--------------------------------------------------------

两条路的代价差 50 倍，而且正确的那条还更准：

* ``?search=<标题>`` —— **这是全文检索，不是标题检索**。OpenAlex 的响应体自己
  写明了 ``oql: "works where full text has (...)"``：一篇论文的标题查下去返回
  ``meta.count = 14129``，前几条毫不相关。靠 ``pick_best`` 的 0.92 阈值从垃圾里
  捞，且 **1 篇 1 个请求**。
* ``?filter=title.search:<标题A>|<标题B>|...`` —— 标题检索，**支持 ``|`` 批量 OR**。
  实测 RL 子集 1585 篇 = 33 个请求 = 97 秒，匹配率 87.1%。

精度（拿「已有 arXiv ID」做独立第二证据：要求返回的 work 的 ``locations`` 里
能解析出同一个 arXiv ID）：180 篇抽样 → 可核对 153 篇 → **冲突 0**，
相似度 p10 = p50 = p90 = 1.000。

配额：**每次请求固定 10 credits，与查询大小无关**（单标题和 50 标题实测都是 10）。
日额度 1000 credits → **100 请求/天**。所以批量是唯一可行的形状：全库 13119 篇
= 263 个请求 ≈ 2.6 天；若按 1 篇 1 请求则是 131 天。
"""

from __future__ import annotations

import difflib
import json
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from ..http import Fetcher
from ..schema import CanonicalPaper, normalize_title

SOURCE = "openalex"

# 新旧两种 arXiv ID 格式
_ARXIV_MODERN = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_ARXIV_LEGACY = re.compile(r"arxiv\.org/abs/([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?", re.I)
_ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", re.I)


@dataclass
class MatchConfig:
    title_similarity_threshold: float = 0.92
    year_tolerance: int = 1
    require_author_overlap: bool = True


@dataclass
class Candidate:
    """一个候选工作及其匹配评分。保留评分是为了让"为什么没匹配上"可追溯。"""

    work: dict[str, Any]
    similarity: float
    year_delta: int
    author_overlap: int

    @property
    def title(self) -> str:
        return (self.work.get("title") or "").strip()


# --------------------------------------------------------------------------- #
# 归一化与评分
# --------------------------------------------------------------------------- #

def _last_names(authors: list[str]) -> set[str]:
    """取姓氏集合。用末词近似姓氏——中文名、复姓会有偏差，但用于**交叉校验**
    足够了：只要有一位作者重合就通过，不需要精确。
    """
    out: set[str] = set()
    for name in authors:
        parts = re.sub(r"[.,]", " ", name or "").split()
        if parts:
            out.add(parts[-1].lower())
    return out


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def score(work: dict[str, Any], paper: CanonicalPaper, cfg: MatchConfig) -> Candidate:
    cand_title = normalize_title(work.get("title") or "")
    cand_year = work.get("publication_year") or 0
    cand_authors = [
        (a.get("author") or {}).get("display_name") or ""
        for a in (work.get("authorships") or [])
    ]
    overlap = len(_last_names(cand_authors) & _last_names(paper.authors))
    return Candidate(
        work=work,
        similarity=_similarity(cand_title, paper.title_norm),
        year_delta=abs(cand_year - paper.year) if cand_year else 99,
        author_overlap=overlap,
    )


def is_acceptable(cand: Candidate, cfg: MatchConfig) -> bool:
    if cand.similarity < cfg.title_similarity_threshold:
        return False
    if cand.year_delta > cfg.year_tolerance:
        return False
    if cfg.require_author_overlap and cand.author_overlap == 0:
        return False
    return True


def pick_best(
    works: list[dict[str, Any]], paper: CanonicalPaper, cfg: MatchConfig
) -> Candidate | None:
    """返回最佳可接受候选；没有任何候选达标就返回 None（宁缺勿滥）。"""
    scored = [score(w, paper, cfg) for w in works]
    ok = [c for c in scored if is_acceptable(c, cfg)]
    if not ok:
        return None
    # 同分时优先作者重合多的，再优先年份接近的
    ok.sort(key=lambda c: (c.similarity, c.author_overlap, -c.year_delta), reverse=True)
    return ok[0]


# --------------------------------------------------------------------------- #
# 字段抽取
# --------------------------------------------------------------------------- #

def extract_arxiv_id(work: dict[str, Any]) -> str | None:
    for loc in work.get("locations") or []:
        for url in (loc.get("landing_page_url") or "", loc.get("pdf_url") or ""):
            for pattern in (_ARXIV_MODERN, _ARXIV_LEGACY):
                m = pattern.search(url)
                if m:
                    return m.group(1)
    m = _ARXIV_DOI.search(work.get("doi") or "")
    return m.group(1) if m else None


def extract_abstract(work: dict[str, Any]) -> str | None:
    """OpenAlex 的摘要是倒排索引（词 → 位置列表），需要还原。"""
    inv = work.get("abstract_inverted_index")
    if not inv:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            positions[i] = word
    if not positions:
        return None
    return " ".join(positions[i] for i in sorted(positions))


def extract_topics(work: dict[str, Any], limit: int = 5) -> list[str]:
    topics = work.get("topics") or []
    out = [
        t.get("display_name")
        for t in sorted(topics, key=lambda t: t.get("score") or 0, reverse=True)
        if t.get("display_name")
    ]
    if out:
        return out[:limit]
    # 回退到旧版 concepts 字段
    concepts = work.get("concepts") or []
    return [
        c["display_name"]
        for c in sorted(concepts, key=lambda c: c.get("score") or 0, reverse=True)[:limit]
        if c.get("display_name")
    ]


# --------------------------------------------------------------------------- #
# 取数
# --------------------------------------------------------------------------- #

#: 出现在标题里就会破坏 OpenAlex filter 语法的字符。**一废就是一整批。**
#:
#:   ``|``  同一 filter 内多个值的 OR 分隔符——我们正是靠它做批量，
#:          标题里再出现就把一篇标题劈成了两个词条
#:   ``,``  多个 filter 之间的分隔符
#:   ``"``  引号
#:   ``*`` ``?``  通配符。实测踩过：标题里带问号 → **整批 400**，报错文字是
#:          ``Wildcards (* or ?) require the exact (no-stem) field``——
#:          看着像语法写错，实际是数据里的问号。32 批里被它打废 8 批，
#:          匹配率从 87.1% 虚低到 40.5%，而失败只表现为"匹配率偏低"，不报错。
_FILTER_BREAKERS = (",", "|", '"', "*", "?")

#: 单个请求 URL 的字节上限。OpenAlex 硬上限实测 8190 字节；100 篇标题
#: （约 8.3 KB）直接 400。留余量到 7000，因为 URL 里还有 select / per-page。
_URL_BUDGET = 7000

#: 批量取数要的字段。``locations`` 用来反推 arXiv ID **并做精度的独立交叉验证**
#: （标题可能雷同，arXiv ID 不会）。字段要一次要齐——少一个就得为它再开一轮配额，
#: 而配额是每天 100 次请求。
_BATCH_SELECT = (
    "id,doi,title,publication_year,type,locations,authorships,"
    "cited_by_count,abstract_inverted_index,topics,concepts"
)


def sanitize_filter_value(title: str) -> str:
    """把标题变成能安全放进 ``filter=title.search:`` 的值。

    **只影响召回，不影响精度**：判定仍然拿完整归一化标题做相似度比较。
    实测 ``$\\pi_0.5$: a Vision-Language-Action Model...`` 这类标题直接搜会
    0 命中，去掉 ``$`` 和 ``\\pi`` 之后就能搜到。
    """
    t = re.sub(r"\\[a-zA-Z]+", " ", title or "")      # \pi \alpha 等命令
    t = re.sub(r"[${}\\^_~]", " ", t)                 # 数学定界符
    t = re.sub(f"[{re.escape(''.join(_FILTER_BREAKERS))}]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def chunk_titles(
    papers: Iterable[CanonicalPaper], budget: int = _URL_BUDGET
) -> Iterator[list[CanonicalPaper]]:
    """按 URL 字节预算切批。

    **不能按固定篇数切**：标题长短能差三倍，50 篇短标题过得去，
    50 篇长标题就撞上 8190 字节的硬上限了。
    """
    batch: list[CanonicalPaper] = []
    size = 0
    for p in papers:
        cost = len(urllib.parse.quote(sanitize_filter_value(p.title) or p.title)) + 1
        if batch and size + cost > budget:
            yield batch
            batch, size = [], 0
        batch.append(p)
        size += cost
    if batch:
        yield batch


def batch_url(papers: list[CanonicalPaper], base_url: str) -> str:
    """拼批量检索 URL。``base_url`` 形如 ``https://api.openalex.org/works``。"""
    q = "|".join(
        urllib.parse.quote(sanitize_filter_value(p.title) or p.title) for p in papers
    )
    return f"{base_url}?filter=title.search:{q}&select={_BATCH_SELECT}&per-page=200"


def fetch_batch(
    papers: list[CanonicalPaper], fetcher: Fetcher, base_url: str
) -> list[dict[str, Any]] | None:
    """取一批的候选 works。

    **返回整批共享的候选池，不做逐篇配对**——配对交给 :func:`pick_best`。
    宽召回 + 严判定才是这里该有的形状：多几个候选不会带来误配风险
    （0.92 相似度 + 年份窗口 + 作者重合三重门槛把关），漏掉候选才会。

    **返回 ``None`` 与返回 ``[]`` 是两件不同的事，调用方必须分开处理**：

    * ``None`` —— 请求失败（限额、超时、400）。这批**根本没查过**，
      调用方不得写"尝试过"标记，否则这些论文会被重试窗口锁死，
      表现为"查过了没匹配上"，而真相是一次都没查成。
    * ``[]`` —— 请求成功，OpenAlex 确实没有候选。这才是"未匹配"。
    """
    raw = fetcher.get_bytes(batch_url(papers, base_url), suffix=".json")
    if raw is None:
        return None
    try:
        return json.loads(raw).get("results") or []
    except json.JSONDecodeError:
        return None


def search(
    paper: CanonicalPaper, fetcher: Fetcher, base_url: str
) -> list[dict[str, Any]] | None:
    """单篇检索。**只在人工核对单篇时用**。

    批量走 :func:`fetch_batch`：每次请求固定 10 credits，查 1 篇和查 50 篇同价，
    单篇查是纯浪费配额（100 请求/天）。
    """
    return fetch_batch([paper], fetcher, base_url)
