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


def prune_arxiv_vectors(
    cfg: dict[str, Any] | None = None,
    *,
    active_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Drop vectors whose papers no longer exist in the arXiv corpus.

    Each shard is rewritten atomically. An empty shard is removed. This keeps
    retention cleanup from leaving old paper vectors on disk even though the
    next LanceDB rebuild would already exclude them.
    """
    cfg = cfg or load_config()
    vector_dir = paths_from(cfg)["workspace"] / "arxiv_vectors"
    if not vector_dir.is_dir():
        return {
            "before_ids": 0,
            "after_ids": 0,
            "removed_ids": 0,
            "rewritten_shards": 0,
            "removed_shards": 0,
        }
    if active_ids is None:
        corpus = open_arxiv_corpus(cfg)
        active_ids = {
            paper.paper_id for paper in corpus.list_papers(limit=None)
        }

    before_ids = 0
    after_ids = 0
    removed_ids = 0
    rewritten_shards = 0
    removed_shards = 0
    for path in sorted(vector_dir.glob("vec_*.npz")):
        temp = path.with_suffix(path.suffix + ".tmp")
        try:
            with np.load(path, allow_pickle=False) as data:
                vectors = np.asarray(data["vectors"])
                ids = np.asarray(data["ids"])
                extras = {
                    key: data[key]
                    for key in data.files
                    if key not in {"vectors", "ids"}
                }
            if vectors.shape[0] != ids.shape[0]:
                raise ValueError(
                    f"{path}: vectors/ids length mismatch "
                    f"({vectors.shape[0]} != {ids.shape[0]})"
                )
            keep = np.asarray(
                [str(value) in active_ids for value in ids],
                dtype=bool,
            )
            shard_before = int(ids.size)
            shard_after = int(keep.sum())
            before_ids += shard_before
            after_ids += shard_after
            if shard_after == shard_before:
                continue
            removed_ids += shard_before - shard_after
            if shard_after == 0:
                path.unlink()
                removed_shards += 1
                continue
            with temp.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    vectors=vectors[keep],
                    ids=ids[keep],
                    **extras,
                )
            os.replace(temp, path)
            rewritten_shards += 1
        finally:
            temp.unlink(missing_ok=True)

    return {
        "before_ids": before_ids,
        "after_ids": after_ids,
        "removed_ids": removed_ids,
        "rewritten_shards": rewritten_shards,
        "removed_shards": removed_shards,
    }


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
