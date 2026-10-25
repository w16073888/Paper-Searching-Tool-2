"""LanceDB-backed paper index with vector + full-text hybrid retrieval."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa

from ..core.models import PaperDetail, SearchFilters, SearchHit, SearchResult
from ..config import load_config, paths_from
from ..corpus import open_arxiv_corpus, open_corpus
from ..structure.embed import embed_query, load_all

TABLE_NAME = "papers"
VECTOR_COLUMN = "vector"
TEXT_COLUMN = "search_text"


def _where(filters: SearchFilters | None) -> str:
    parts = []
    if filters and filters.years:
        years = [int(y) for y in filters.years]
        parts.append("year IN (" + ",".join(str(y) for y in years) + ")")
    if filters and filters.venues:
        escaped = [
            str(v).replace("'", "''") for v in filters.venues if str(v).strip()
        ]
        if escaped:
            parts.append(
                "venue IN (" + ",".join(f"'{v}'" for v in escaped) + ")"
            )
    return " AND ".join(parts)


class LancePaperStore:
    """One stable retrieval interface for tools; no answer-mode policy."""

    def __init__(
        self,
        cfg: dict | None = None,
        *,
        source: str = "meeting",
        db_path: Path | None = None,
    ):
        self.cfg = cfg or load_config()
        self.source = source
        if db_path is None:
            dirname = "lance_arxiv" if source == "arxiv" else "lance"
            db_path = paths_from(self.cfg)["workspace"] / dirname
        self.db_path = Path(db_path)

    @property
    def available(self) -> bool:
        return self.db_path.exists() and TABLE_NAME in lancedb.connect(
            str(self.db_path)
        ).table_names()

    def count(self) -> int:
        """Return the number of indexed rows, or zero when unavailable."""
        if not self.available:
            return 0
        return int(self._table().count_rows())

    def _table(self):
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"LanceDB index missing: {self.db_path}; run scripts/build_lance_index.py"
            )
        return lancedb.connect(str(self.db_path)).open_table(TABLE_NAME)

    def get_papers(self, paper_ids: list[str]) -> list[PaperDetail]:
        ids = [str(pid) for pid in dict.fromkeys(paper_ids) if str(pid).strip()]
        if not ids:
            return []
        table = self._table()
        rows = table.search().where(
            "paper_id IN (" + ",".join(f"'{pid}'" for pid in ids) + ")"
        ).limit(len(ids)).to_list()
        by_id = {str(row["paper_id"]): row for row in rows}
        return [
            PaperDetail(
                paper_id=pid,
                title=str(by_id.get(pid, {}).get("title") or ""),
                abstract=str(by_id.get(pid, {}).get("abstract") or ""),
                year=int(by_id[pid]["year"]) if pid in by_id and by_id[pid].get("year") else None,
                venue=str(by_id.get(pid, {}).get("venue") or ""),
                subfields=list(by_id.get(pid, {}).get("subfields") or []),
            )
            for pid in ids
            if pid in by_id
        ]

    def search(
        self,
        query: str,
        *,
        keywords: list[str] | None = None,
        filters: SearchFilters | None = None,
        limit: int = 20,
        vector_candidates: int = 100,
    ) -> SearchResult:
        query = str(query or "").strip()
        if not query:
            return SearchResult(query="", results=[])
        table = self._table()
        where = _where(filters)
        qv = embed_query(
            " ".join([query, *(keywords or [])]),
            model=str((self.cfg.get("embed") or {}).get("model") or "BAAI/bge-m3"),
        )

        vector_query = table.search(qv)
        if where:
            vector_query = vector_query.where(where)
        vector_rows = vector_query.limit(max(limit, vector_candidates)).to_list()

        fts_rows: list[dict[str, Any]] = []
        text_query = " OR ".join(
            dict.fromkeys(
                x.strip() for x in [query, *(keywords or [])] if x.strip()
            )
        )
        if text_query:
            try:
                fts_query = table.search(text_query, query_type="fts")
                if where:
                    fts_query = fts_query.where(where)
                fts_rows = fts_query.limit(max(limit, vector_candidates)).to_list()
            except Exception:  # noqa: BLE001 - FTS unavailable degrades to vector
                fts_rows = []

        ranked: dict[str, float] = defaultdict(float)
        for rank, row in enumerate(vector_rows, start=1):
            ranked[str(row["paper_id"])] += 1.0 / (60 + rank)
        for rank, row in enumerate(fts_rows, start=1):
            ranked[str(row["paper_id"])] += 1.0 / (60 + rank)
        by_id = {
            str(row["paper_id"]): row
            for row in [*vector_rows, *fts_rows]
        }
        ordered = sorted(ranked, key=ranked.get, reverse=True)[:limit]
        hits = []
        for pid in ordered:
            row = by_id[pid]
            abstract = str(row.get("abstract") or "")
            hits.append(SearchHit(
                paper_id=pid,
                title=str(row.get("title") or ""),
                year=int(row["year"]) if row.get("year") else None,
                venue=str(row.get("venue") or ""),
                score=round(float(ranked[pid]), 6),
                snippet=abstract[:280],
            ))
        return SearchResult(
            query=query,
            results=hits,
            n_candidates=len(ranked),
            truncated=len(ranked) > limit,
        )


def build_lance_index(
    cfg: dict | None = None,
    *,
    out_dir: Path | None = None,
    limit: int | None = None,
    source: str = "meeting",
) -> dict[str, Any]:
    """Offline full/small build from existing BGE-M3 vector shards."""
    cfg = cfg or load_config()
    ws = paths_from(cfg)["workspace"]
    vector_dir = ws / ("arxiv_vectors" if source == "arxiv" else "vectors")
    X, ids = load_all(vector_dir)
    if limit is not None:
        X, ids = X[:limit], ids[:limit]
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    corpus = open_arxiv_corpus(cfg) if source == "arxiv" else open_corpus(cfg)
    papers = {p.paper_id: p for p in corpus.list_papers(limit=None)}
    vector_row = {pid: i for i, pid in enumerate(ids)}
    kept = list(dict.fromkeys(pid for pid in ids if pid in papers))
    row_map = [vector_row[pid] for pid in kept]
    X = X[row_map]

    out = Path(out_dir) if out_dir else ws / (
        "lance_arxiv" if source == "arxiv" else "lance"
    )
    out.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(out))
    vector_type = pa.list_(pa.float32(), int(X.shape[1]))
    flat = pa.array(X.reshape(-1), type=pa.float32())
    vectors = pa.FixedSizeListArray.from_arrays(flat, int(X.shape[1]))
    table_data = pa.table({
        "paper_id": pa.array(kept, type=pa.string()),
        "title": pa.array([papers[p].title or "" for p in kept], type=pa.string()),
        "abstract": pa.array([papers[p].abstract or "" for p in kept], type=pa.string()),
        "search_text": pa.array([
            f"{papers[p].title or ''}\n{papers[p].abstract or ''}" for p in kept
        ], type=pa.string()),
        "year": pa.array([
            int(papers[p].year) if papers[p].year is not None else 0 for p in kept
        ], type=pa.int32()),
        "venue": pa.array([papers[p].venue or "" for p in kept], type=pa.string()),
        "subfields": pa.array([
            list(papers[p].subfields or ()) for p in kept
        ], type=pa.list_(pa.string())),
        VECTOR_COLUMN: vectors,
    })
    table = db.create_table(TABLE_NAME, data=table_data, mode="overwrite")
    table.create_fts_index(TEXT_COLUMN, replace=True)
    manifest = {
        "table": TABLE_NAME,
        "papers": len(kept),
        "dimension": int(X.shape[1]),
        "vector_column": VECTOR_COLUMN,
        "text_column": TEXT_COLUMN,
        "embedding_model": str(
            (cfg.get("embed") or {}).get("model") or "BAAI/bge-m3"
        ),
        "source": source,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
