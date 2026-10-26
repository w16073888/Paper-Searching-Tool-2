"""Build the full-corpus topic snapshot consumed by trend questions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hotspot.config import load_config  # noqa: E402
from hotspot.corpus import open_corpus  # noqa: E402
from hotspot.structure.embed import load_all  # noqa: E402
from hotspot.structure.topic_cluster import (  # noqa: E402
    build_topic_snapshot,
    write_topic_snapshot,
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    cfg = load_config()
    topics_cfg = cfg.get("topics") or {}
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vectors",
        default=str((cfg.get("paths") or {}).get("workspace") or "workspace")
        + "/vectors",
    )
    parser.add_argument(
        "--out",
        default=str(
            topics_cfg.get("snapshot")
            or "hotspot/data/topic_snapshot_v1.json.gz"
        ),
    )
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--buckets", type=int, default=0)
    parser.add_argument("--max-bucket-size", type=int, default=600)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--min-cluster-size", type=int, default=20)
    parser.add_argument("--max-cluster-size", type=int, default=1_200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    vectors, ids = load_all(_resolve(args.vectors))
    if args.limit:
        vectors = vectors[: args.limit]
        ids = ids[: args.limit]
    corpus = open_corpus(cfg)
    papers = [corpus.get(paper_id) for paper_id in ids]
    missing = [paper_id for paper_id, paper in zip(ids, papers) if paper is None]
    if missing:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "vectors reference missing corpus papers",
                    "missing_count": len(missing),
                    "examples": missing[:5],
                },
                ensure_ascii=False,
            )
        )
        return 1

    snapshot = build_topic_snapshot(
        vectors,
        ids,
        papers,  # type: ignore[arg-type]
        pca_dim=args.pca_dim,
        n_neighbors=args.neighbors,
        n_buckets=args.buckets or None,
        max_bucket_size=args.max_bucket_size,
        resolution=args.resolution,
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size,
        seed=args.seed,
        spread_cfg=cfg.get("spread") or {},
    )
    output = None if args.no_write else write_topic_snapshot(snapshot, _resolve(args.out))
    print(
        json.dumps(
            {
                "ok": True,
                "snapshot": output,
                "corpus_size": snapshot["corpus_size"],
                "cluster_count": snapshot["cluster_count"],
                "metrics": snapshot["metrics"],
                "partial": bool(args.limit),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
