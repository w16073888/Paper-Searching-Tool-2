"""检索。FTS5 只负责"哪些篇命中、排序如何"，命中段落回读正文现切。

**默认搜全量，不以 RL 为闸门。** 这一点是本工具存在的理由：

关键词筛选器只匹配标题+摘要，且词表是固定的 65 个强信号词。命中不了的子方向
对使用者就是**不存在**的——例如 RL for LLM reasoning 的标题写的是
"verifiable rewards"/"RLVR"（不在词表里），RL 理论写的是 "sample complexity"
（根本不出现 "reinforcement learning"）。

但全文检索不需要经过那道闸门。所以 ``is_rl`` 在这里是一个**筛选面（facet）**，
不是**闸门（gate）**——``--rl-only`` 是可选缩小，不是默认。这样筛选器的召回漏洞
就被检索补上了。这是「方向的完整性」在工具侧的落点。
"""

from __future__ import annotations

import gzip
import re
import shlex
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .. import ROOT
from ..paths import text_path

#: bm25 列权重：标题命中远比正文命中更能说明这篇讲的就是这个。
#: 不设权重的话，正文里顺带提一句的长论文会淹没标题就是该主题的论文。
BM25_WEIGHTS = (10.0, 3.0, 1.0)

#: 命中段落窗口（字符）
SNIPPET_RADIUS = 120


@dataclass
class Hit:
    paper_id: str
    title: str
    venue: str | None
    year: int | None
    subfields: list[str]
    is_rl: bool
    arxiv_id: str | None
    url: str | None
    chars: int
    rank: float
    snippet: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id, "title": self.title, "venue": self.venue,
            "year": self.year, "subfields": self.subfields, "is_rl": self.is_rl,
            "arxiv_id": self.arxiv_id, "url": self.url, "chars": self.chars,
            "rank": round(self.rank, 3), "snippet": self.snippet,
        }


# ---------------------------------------------------------------------- #
# 查询串解析
# ---------------------------------------------------------------------- #

def _fts_query(text: str, *, any_term: bool = False) -> str:
    """把用户输入变成安全的 FTS5 表达式。

    不能把用户输入直接塞进 MATCH：FTS5 的查询语法里 ``"`` ``*`` ``-`` ``:`` ``(``
    都有语义，标题里的标点会让查询**报错**（不是返回空，是抛
    ``sqlite3.OperationalError: fts5: syntax error``）。所以逐词加引号。

    引号内的转义规则是把 ``"`` 双写。词尾的 ``*`` 留在引号**外**——
    ``"optim"*`` 是合法前缀语法，而 ``"optim*"`` 会把星号当字面量。
    """
    try:
        # shlex 让用户能用 "direct preference optimization" 显式指定短语
        chunks = shlex.split(text, posix=False)
    except ValueError:
        chunks = text.split()

    terms: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip().strip('"').strip("'").strip()
        if not chunk:
            continue
        prefix = chunk.endswith("*")
        stem = chunk[:-1] if prefix else chunk
        if not stem:
            continue
        quoted = '"' + stem.replace('"', '""') + '"'
        terms.append(quoted + "*" if prefix else quoted)

    if not terms:
        return ""
    return (" OR " if any_term else " AND ").join(terms)


# ---------------------------------------------------------------------- #
# 命中段落
# ---------------------------------------------------------------------- #

def _terms_of(query_text: str) -> list[str]:
    try:
        chunks = shlex.split(query_text, posix=False)
    except ValueError:
        chunks = query_text.split()
    out = []
    for c in chunks:
        c = c.strip().strip('"').strip("'").strip().rstrip("*").strip()
        if len(c) >= 2:
            out.append(c)
    return out


def _snippet(text: str, terms: Iterable[str], radius: int = SNIPPET_RADIUS) -> str:
    """在正文里找**第一个**命中处，取前后一段。

    取第一个而不是"最好的"一个：这是给人快速判断相关性的，不是做摘要。
    挑"最好"的需要打分，而打分的依据（bm25 是整篇的）在段落层面并不成立。
    """
    if not text or not terms:
        return ""
    low = text.lower()
    best = -1
    for term in terms:
        pos = low.find(term.lower())
        if pos >= 0 and (best < 0 or pos < best):
            best = pos
    if best < 0:
        return ""
    start = max(0, best - radius)
    end = min(len(text), best + radius)
    # 掐到词边界，免得片段以半个单词开头
    if start > 0:
        cut = low.find(" ", start)
        if 0 <= cut < best:
            start = cut + 1
    if end < len(text):
        cut = low.rfind(" ", best, end)
        if cut > best:
            end = cut
    piece = re.sub(r"\s+", " ", text[start:end]).strip()
    return f"{'…' if start else ''}{piece}{'…' if end < len(text) else ''}"


def _read_body(text_dir: Path, paper_id: str) -> str:
    path = text_path(text_dir, paper_id)
    if not path.exists():
        return ""
    try:
        return gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
    except (OSError, EOFError):
        return ""


# ---------------------------------------------------------------------- #
# 检索
# ---------------------------------------------------------------------- #

def search(
    conn: sqlite3.Connection,
    query: str,
    *,
    text_dir: Path | None = None,
    venue: str | None = None,
    year: int | None = None,
    subfield: str | None = None,
    rl_only: bool = False,
    limit: int = 20,
    offset: int = 0,
    any_term: bool = False,
    snippets: bool = True,
) -> list[Hit]:
    """返回命中列表。``text_dir`` 为 None 时不算命中段落（快很多）。"""
    expr = _fts_query(query, any_term=any_term)
    if not expr:
        return []

    where = ["docs MATCH ?"]
    params: list[Any] = [expr]
    if venue:
        where.append("p.venue = ?")
        params.append(venue)
    if year:
        where.append("p.year = ?")
        params.append(year)
    if subfield:
        # 逗号分隔列里的精确成员匹配：前后补逗号再 LIKE，避免 "rl" 命中 "rlhf"
        where.append("(',' || p.subfields || ',') LIKE ?")
        params.append(f"%,{subfield},%")
    if rl_only:
        where.append("p.is_rl = 1")

    params.extend([limit, offset])
    # bm25 权重**必须内联**，不能走占位符：SELECT 在 WHERE 之前，占位符按 SQL 文本
    # 顺序绑定，三个权重参数会把 MATCH 的检索式挤到后面去，结果是
    # `fts5: syntax error near "."`——一个看不出真正原因的报错。
    weights = ", ".join(repr(float(w)) for w in BM25_WEIGHTS)
    sql = f"""
        SELECT p.paper_id, p.title, p.venue, p.year, p.subfields, p.is_rl,
               p.arxiv_id, p.url, p.chars,
               bm25(docs, {weights}) AS rank
        FROM docs JOIN papers p ON p.doc_rowid = docs.rowid
        WHERE {' AND '.join(where)}
        ORDER BY rank
        LIMIT ? OFFSET ?
    """
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return []

    terms = _terms_of(query) if snippets and text_dir else []
    hits: list[Hit] = []
    for r in rows:
        snip = ""
        if terms:
            snip = _snippet(_read_body(text_dir, r["paper_id"]), terms)
        hits.append(Hit(
            paper_id=r["paper_id"], title=r["title"], venue=r["venue"],
            year=r["year"],
            subfields=[s for s in (r["subfields"] or "").split(",") if s],
            is_rl=bool(r["is_rl"]), arxiv_id=r["arxiv_id"], url=r["url"],
            chars=r["chars"] or 0, rank=r["rank"], snippet=snip,
        ))
    return hits


def count(conn: sqlite3.Connection, query: str, **filters: Any) -> int:
    """命中总数（不算段落）。用于「只有前 N 条」时告诉用户总共多少条。"""
    hits = search(conn, query, limit=100000, snippets=False, **filters)
    return len(hits)


def facets(conn: sqlite3.Connection, query: str, **filters: Any) -> dict[str, Any]:
    """一次查询在各维度的分布——用于回答「这个方向主要发在哪几个会、哪一年」。

    这是"看趋势"最省事的形式：不需要图表，几个计数就能看出方向冷热。
    """
    hits = search(conn, query, limit=100000, snippets=False, **filters)
    if not hits:
        return {}
    from collections import Counter

    by_venue: Counter[str] = Counter()
    by_year: Counter[int] = Counter()
    by_sub: Counter[str] = Counter()
    for h in hits:
        by_venue[h.venue or "?"] += 1
        if h.year:
            by_year[h.year] += 1
        by_sub.update(h.subfields)
    return {
        "total": len(hits),
        "by_venue": dict(by_venue.most_common()),
        "by_year": dict(sorted(by_year.items())),
        "by_subfield": dict(by_sub.most_common(12)),
        "with_text": sum(1 for h in hits if h.chars),
    }
