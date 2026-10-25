"""Incremental vectorization for arXiv weekly-corpus updates."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_config, paths_from
from ..corpus import open_arxiv_corpus


def _existing_ids(vector_dir: Path) -> set[str]:
    ids: set[str] = set()
    for path in sorted(vector_dir.glob("vec_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            ids.update(str(value) for value in data["ids"])
    return ids


def embed_missing_arxiv(
    cfg: dict[str, Any] | None = None,
    *,
    limit: int = 0,
) -> dict[str, Any]:
    """Embed arXiv papers not present in any vec_*.npz shard."""
    cfg = cfg or load_config()
    workspace = paths_from(cfg)["workspace"]
    vector_dir = workspace / "arxiv_vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)

    existing = _existing_ids(vector_dir)
    corpus = open_arxiv_corpus(cfg)
    missing = [
        paper for paper in corpus.list_papers(limit=None)
        if paper.paper_id not in existing
    ]
    missing_before = len(missing)
    if limit:
        missing = missing[:limit]
    if not missing:
        return {
            "missing_before": 0,
            "embedded": 0,
            "output": None,
        }

    import md2vectors
    from dasc.cluster.embed import build_docs

    ids, docs = build_docs(missing)
    model_name = str(
        (cfg.get("embed") or {}).get("model") or "BAAI/bge-m3"
    )
    entries, dimension = md2vectors.embed(ids, docs, model_name)
    vectors = np.asarray(
        [entries[paper_id]["embedding"] for paper_id in ids],
        dtype=np.float32,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = vector_dir / f"vec_delta_{stamp}.npz"
    temp = output.with_suffix(".npz.tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(
            handle,
            vectors=vectors,
            ids=np.asarray(ids),
            model=np.asarray([model_name]),
            dim=np.asarray([dimension]),
            with_title=np.asarray([True]),
        )
    os.replace(temp, output)
    return {
        "missing_before": missing_before,
        "embedded": len(ids),
        "output": str(output),
        "dimension": int(dimension),
    }
