"""派生导出：parquet 与 bib。

两者都是**产物**，可以随时从 ``data/papers.jsonl`` 重新生成，不承载任何独有信息。

**为什么要有 parquet**：下游的分类与趋势分析要跑在结构化表上。
让分析脚本直接读 jsonl 也行，但每次都要处理嵌套的 ``rl`` / ``provenance``
和 null 类型推断；导成扁平的强类型表之后，``pandas.read_parquet`` /
``duckdb`` 都能直接上手，不用再关心本库的内部结构。

**schema 是显式写死的，不靠推断。** 因为每周都在增量更新，类型推断会随
"这一列恰好全为空"而在 int64 / null / string 之间跳变，把下游查询搞崩。
显式 schema 保证任何一周导出的 parquet 类型都一致。
"""

from __future__ import annotations

import logging
from pathlib import Path
from string import ascii_lowercase
from typing import TYPE_CHECKING

from .schema import CanonicalPaper

if TYPE_CHECKING:
    from .store import PaperStore

log = logging.getLogger(__name__)


def flatten(paper: CanonicalPaper) -> dict:
    """把一篇论文压成一行扁平记录。

    ``rl`` 与 ``provenance`` 拆成带前缀的列，而不是留成结构体——
    分析时按列过滤比在嵌套结构里翻要顺手得多。
    """
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "title_norm": paper.title_norm,
        "authors": list(paper.authors),
        "n_authors": len(paper.authors),
        "venue": paper.venue,
        "year": paper.year,
        "track": paper.track,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "openreview_id": paper.openreview_id,
        "url": paper.url,
        "pdf_url": paper.pdf_url,
        "local_pdf": paper.local_pdf,
        "abstract": paper.abstract,
        "abstract_chars": len(paper.abstract or ""),
        "topics": list(paper.topics),
        "citations": paper.citations,
        "citations_updated": paper.citations_updated,
        "rl_is_rl": paper.rl.is_rl,
        "rl_score": paper.rl.score,
        "rl_reasons": list(paper.rl.reasons),
        "rl_subfield": list(paper.rl.subfield),
        "rl_manual_override": paper.rl.manual_override,
        "provenance_sources": [p.source for p in paper.provenance],
        "first_seen": paper.first_seen,
        "last_updated": paper.last_updated,
    }


def to_parquet(store: PaperStore, path: Path) -> dict:
    """导出 parquet。返回聚合计数。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [flatten(p) for _, p in sorted(store.papers.items())]

    # 显式 schema：见模块文档，靠推断会在周与周之间漂移。
    schema = pa.schema([
        ("paper_id", pa.string()),
        ("title", pa.string()),
        ("title_norm", pa.string()),
        ("authors", pa.list_(pa.string())),
        ("n_authors", pa.int32()),
        ("venue", pa.string()),
        ("year", pa.int32()),
        ("track", pa.string()),
        ("doi", pa.string()),
        ("arxiv_id", pa.string()),
        ("openreview_id", pa.string()),
        ("url", pa.string()),
        ("pdf_url", pa.string()),
        ("local_pdf", pa.string()),
        ("abstract", pa.string()),
        ("abstract_chars", pa.int32()),
        ("topics", pa.list_(pa.string())),
        ("citations", pa.int32()),
        ("citations_updated", pa.string()),
        ("rl_is_rl", pa.bool_()),
        ("rl_score", pa.float64()),
        ("rl_reasons", pa.list_(pa.string())),
        ("rl_subfield", pa.list_(pa.string())),
        ("rl_manual_override", pa.bool_()),
        ("provenance_sources", pa.list_(pa.string())),
        ("first_seen", pa.string()),
        ("last_updated", pa.string()),
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")

    return {
        "rows": table.num_rows,
        "columns": table.num_columns,
        "path": str(path),
        "bytes": path.stat().st_size,
        "rl_rows": sum(1 for r in rows if r["rl_is_rl"]),
    }


# --------------------------------------------------------------------------- #
# BibTeX
# --------------------------------------------------------------------------- #

#: 各会议在 bib 里的 booktitle。写成"Proceedings of ..."是通行做法，
#: 格式化成什么样式由你的引用工具（Zotero / Better BibTeX）决定。
_BOOKTITLE = {
    "ICML": "Proceedings of the 42nd International Conference on Machine Learning",
    "NeurIPS": "Advances in Neural Information Processing Systems 38",
    "ICLR": "International Conference on Learning Representations",
    "CoRL": "Conference on Robot Learning",
}


def _escape(field: str) -> str:
    """BibTeX 里花括号要转义，其余交给 LaTeX（标题里的公式本来就是 LaTeX）。

    不做引号/反斜杠转义——那会破坏 ``\\mathbb{X}`` 这类本来就合法的 LaTeX。
    """
    return field.replace("{", r"\{").replace("}", r"\}")


def bib_entry(paper: CanonicalPaper) -> str:
    fields: list[tuple[str, str]] = [
        ("title", paper.title),
        ("author", " and ".join(paper.authors)),
        ("booktitle", _BOOKTITLE.get(paper.venue, paper.venue)),
        ("year", str(paper.year)),
    ]
    if paper.track:
        fields.append(("note", paper.track))
    if paper.doi:
        fields.append(("doi", paper.doi))
    if paper.arxiv_id:
        fields.append(("eprint", paper.arxiv_id))
        fields.append(("archivePrefix", "arXiv"))
    if paper.url:
        fields.append(("url", paper.url))
    elif paper.pdf_url:
        fields.append(("url", paper.pdf_url))

    body = ",\n".join(f"  {k} = {{{_escape(v)}}}" for k, v in fields)
    return f"@inproceedings{{{paper.cite_key},\n{body}\n}}\n"


def to_bib(store: PaperStore, path: Path, *, rl_only: bool = False) -> dict:
    """导出 BibTeX。引用键用 :attr:`CanonicalPaper.cite_key`。

    ``rl_only=True`` 时只导出 RL 子集——把你的 Zotero 库灌成全库 13000 篇
    没什么意义。
    """
    papers = [p for _, p in sorted(store.papers.items())]
    if rl_only:
        papers = [p for p in papers if p.rl.is_rl]

    # 引用键可能撞车（同姓同年后同一个标题词），撞了就加后缀 a/b/c 区分，
    # 否则 bib 工具会报重复键并把两条合成一条。
    seen: dict[str, int] = {}
    entries: list[str] = []
    for paper in papers:
        key = paper.cite_key
        n = seen.get(key, 0) + 1
        seen[key] = n
        text = bib_entry(paper)
        if n > 1:
            suffix = ascii_lowercase[n - 2] if n - 2 < len(ascii_lowercase) else str(n)
            text = text.replace(f"{{{key},", f"{{{key}{suffix},", 1)
        entries.append(text)

    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "% 由 rllib 生成，勿手改。重新生成：python -m rllib export --bib\n"
        f"% 共 {len(entries)} 条{'（仅 RL）' if rl_only else ''}\n\n"
    )
    path.write_text(header + "\n".join(entries), encoding="utf-8")

    return {"entries": len(entries), "rl_only": rl_only, "path": str(path)}


def export_all(
    store: PaperStore, data_dir: Path, *, bib_rl_only: bool = True
) -> dict:
    """一次导出 parquet 与 bib。"""
    return {
        "parquet": to_parquet(store, data_dir / "papers.parquet"),
        "bib": to_bib(store, data_dir / "papers.bib", rl_only=bib_rl_only),
    }
