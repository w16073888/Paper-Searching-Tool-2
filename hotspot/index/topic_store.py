"""Hybrid topic-snapshot reader for trend questions."""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

import hotspot
from ..config import load_config, paths_from
from ..core.models import TopicHit, TopicOverview
from ..corpus import open_corpus

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "and", "for", "with", "from", "into", "using", "based",
    "what", "which", "how", "are", "is", "of", "to", "in", "on",
    "a", "an", "latest", "recent", "research", "method", "methods",
    "approach", "approaches", "paper", "papers", "topic", "topics",
}
_PACKAGED_CACHE: dict[str, tuple[float, int, dict]] = {}


def _terms(text: str) -> set[str]:
    return {
        word for word in _WORD.findall((text or "").lower())
        if len(word) > 2 and word not in _STOP
    }


def _year_counts(counts_by_year: dict) -> dict[int, int]:
    return {
        int(year): sum(int(count) for count in (values or {}).values())
        for year, values in (counts_by_year or {}).items()
    }


class TopicStore:
    """Read a packaged v1 snapshot, falling back to legacy workspace metrics."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        workspace = paths_from(self.cfg).get("workspace")
        if workspace is None:
            workspace = hotspot.HOTSPOT_ROOT / "workspace"
        self.metrics_path = workspace / "metrics" / "metrics.json"
        self.members_path = workspace / "metrics" / "cluster_members.json"
        configured = str(
            (self.cfg.get("topics") or {}).get("snapshot") or ""
        ).strip()
        self.snapshot_path = (
            Path(configured)
            if configured and Path(configured).is_absolute()
            else hotspot.HOTSPOT_ROOT / (
                configured or "hotspot/data/topic_snapshot_v1.json.gz"
            )
        )
        self.error = ""
        self._loaded_source = "unavailable"
        self._corpus = None
        self._rows: list[dict] | None = None
        self._pca_mean: np.ndarray | None = None
        self._pca_components: np.ndarray | None = None
        self._centroids: np.ndarray | None = None
        self._snapshot_meta: dict = {}

    @property
    def available(self) -> bool:
        if self.snapshot_path.is_file():
            return bool(self._load())
        return self.metrics_path.is_file() and self.members_path.is_file()

    @property
    def source(self) -> str:
        self._load()
        return self._loaded_source

    def _load_packaged(self) -> list[dict]:
        stat = self.snapshot_path.stat()
        cache_key = str(self.snapshot_path)
        cached = _PACKAGED_CACHE.get(cache_key)
        if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
            parsed = cached[2]
        else:
            with gzip.open(self.snapshot_path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
            if int(payload.get("schema_version") or 0) != 1:
                raise ValueError("unsupported topic snapshot schema")
            clusters = payload.get("clusters")
            if not isinstance(clusters, list) or not clusters:
                raise ValueError("topic snapshot has no clusters")
            pca = payload.get("pca") or {}
            pca_mean = np.asarray(pca.get("mean"), dtype=np.float32)
            pca_components = np.asarray(
                pca.get("components"), dtype=np.float32
            )
            if (
                pca_mean.ndim != 1
                or pca_components.ndim != 2
                or pca_components.shape[1] != pca_mean.shape[0]
            ):
                raise ValueError("invalid PCA transform in topic snapshot")
            centroids = np.asarray(
                [cluster["centroid"] for cluster in clusters],
                dtype=np.float32,
            )
            centroids = centroids / np.maximum(
                np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12
            )
            rows = []
            for cluster in clusters:
                venue = cluster.get("venue") or {}
                top_terms = [
                    str(term) for term in (cluster.get("top_terms") or [])
                ]
                representative_title = str(
                    cluster.get("representative_title") or ""
                )
                representative_ids = [
                    str(value)
                    for value in (cluster.get("representative_ids") or [])
                ]
                rows.append(
                    {
                        "cluster_id": int(cluster["cluster_id"]),
                        "label": str(
                            cluster.get("label")
                            or f"cluster {cluster['cluster_id']}"
                        ),
                        "size": int(cluster.get("size") or 0),
                        "spread_state": str(venue.get("state") or ""),
                        "venue_counts": dict(venue.get("counts") or {}),
                        "year_counts": _year_counts(
                            dict(venue.get("counts_by_year") or {})
                        ),
                        "representative_id": (
                            representative_ids[0] if representative_ids else ""
                        ),
                        "representative_title": representative_title,
                        "terms": _terms(
                            " ".join([*top_terms, representative_title])
                        ),
                    }
                )
            parsed = {
                "rows": rows,
                "pca_mean": pca_mean,
                "pca_components": pca_components,
                "centroids": centroids,
                "meta": {
                    "corpus_size": int(payload.get("corpus_size") or 0),
                    "cluster_count": int(
                        payload.get("cluster_count") or len(clusters)
                    ),
                    "coverage": float(payload.get("coverage") or 0.0),
                    "created_at": str(payload.get("created_at") or ""),
                    "metrics": dict(payload.get("metrics") or {}),
                },
            }
            _PACKAGED_CACHE[cache_key] = (
                stat.st_mtime,
                stat.st_size,
                parsed,
            )
        self._pca_mean = parsed["pca_mean"]
        self._pca_components = parsed["pca_components"]
        self._centroids = parsed["centroids"]
        self._snapshot_meta = parsed["meta"]
        return parsed["rows"]

    def _load_legacy(self) -> list[dict]:
        metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        members = json.loads(self.members_path.read_text(encoding="utf-8"))
        self._corpus = open_corpus(self.cfg)
        rows = []
        for metric in metrics.get("metrics") or []:
            cluster_id = int(metric["cluster_id"])
            member_ids = [str(x) for x in members.get(str(cluster_id), [])]
            titles = []
            for paper_id in member_ids[:8]:
                paper = self._corpus.get(paper_id)
                if paper is not None and paper.title:
                    titles.append(paper.title)
            representative_ids = [
                str(x) for x in (metric.get("representative_ids") or [])
            ]
            representative_title = ""
            if representative_ids:
                representative = self._corpus.get(representative_ids[0])
                representative_title = representative.title if representative else ""
            venue = metric.get("venue") or {}
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "label": representative_title or f"cluster {cluster_id}",
                    "size": int(metric.get("size") or 0),
                    "spread_state": str(venue.get("state") or ""),
                    "venue_counts": dict(venue.get("counts") or {}),
                    "year_counts": _year_counts(dict(venue.get("counts_by_year") or {})),
                    "representative_id": representative_ids[0] if representative_ids else "",
                    "representative_title": representative_title,
                    "terms": _terms(" ".join([representative_title, *titles])),
                }
            )
        return rows

    def _load(self) -> list[dict]:
        if self._rows is not None:
            return self._rows
        self.error = ""
        try:
            if self.snapshot_path.is_file():
                self._rows = self._load_packaged()
                self._loaded_source = "packaged-v1"
                return self._rows
            if self.metrics_path.is_file() and self.members_path.is_file():
                self._rows = self._load_legacy()
                self._loaded_source = "legacy-workspace"
                return self._rows
            self._rows = []
            self._loaded_source = "unavailable"
            self.error = "topic snapshot unavailable"
            return self._rows
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            if self.metrics_path.is_file() and self.members_path.is_file():
                try:
                    self._rows = self._load_legacy()
                    self._loaded_source = "legacy-workspace"
                    return self._rows
                except Exception as fallback_exc:  # noqa: BLE001
                    self.error += (
                        f"; legacy fallback: {type(fallback_exc).__name__}: "
                        f"{fallback_exc}"
                    )
            self._rows = []
            self._loaded_source = "unavailable"
            return self._rows

    def _vector_scores(self, query: str, count: int) -> np.ndarray | None:
        if self._pca_mean is None or self._pca_components is None or self._centroids is None:
            return None
        try:
            from ..structure.embed import embed_query

            vector = np.asarray(embed_query(query), dtype=np.float32)
            reduced = (vector - self._pca_mean) @ self._pca_components.T
            reduced = reduced / max(float(np.linalg.norm(reduced)), 1e-12)
            scores = self._centroids[:count] @ reduced
            return np.asarray(scores, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            self.error = f"topic vector fallback: {type(exc).__name__}: {exc}"
            return None

    def _search_packaged(self, query: str, rows: list[dict], limit: int) -> TopicOverview:
        query_terms = _terms(query)
        vector_scores = self._vector_scores(query, len(rows))
        scored = []
        for index, row in enumerate(rows):
            row_terms = row["terms"]
            lexical = (
                len(query_terms & row_terms) / len(query_terms)
                if query_terms else 0.0
            )
            vector = (
                float(vector_scores[index])
                if vector_scores is not None
                else 0.0
            )
            if vector_scores is not None:
                score = 0.72 * vector + 0.28 * lexical
            else:
                score = lexical
            # A weak vector-only match is not evidence for a trend. Strong
            # lexical overlap remains valid when embedding similarity is low.
            if (
                vector_scores is None and lexical > 0
            ) or vector >= 0.30 or lexical >= 0.50:
                scored.append((score, row))
        scored.sort(
            key=lambda item: (item[0], item[1]["size"]),
            reverse=True,
        )
        topics = []
        for score, row in scored[:max(1, min(limit, 10))]:
            if score < 0.08:
                continue
            topics.append(
                TopicHit(
                    cluster_id=row["cluster_id"],
                    label=row["label"],
                    size=row["size"],
                    spread_state=row["spread_state"],
                    venue_counts=row["venue_counts"],
                    year_counts=row["year_counts"],
                    representative_id=row["representative_id"],
                    representative_title=row["representative_title"],
                    score=round(float(score), 4),
                )
            )
        return TopicOverview(query=query, topics=topics)

    def _search_legacy(self, query: str, rows: list[dict], limit: int) -> TopicOverview:
        query_terms = _terms(query)
        if not query_terms:
            return TopicOverview(query=query, topics=[])
        scored = []
        for row in rows:
            overlap = len(query_terms & row["terms"])
            if overlap:
                scored.append((overlap / len(query_terms), row))
        scored.sort(
            key=lambda item: (item[0], item[1]["size"]),
            reverse=True,
        )
        topics = [
            TopicHit(
                cluster_id=row["cluster_id"],
                label=row["label"],
                size=row["size"],
                spread_state=row["spread_state"],
                venue_counts=row["venue_counts"],
                year_counts=row["year_counts"],
                representative_id=row["representative_id"],
                representative_title=row["representative_title"],
                score=round(float(score), 4),
            )
            for score, row in scored[:max(1, min(limit, 10))]
        ]
        return TopicOverview(query=query, topics=topics)

    def search(self, query: str, *, limit: int = 5) -> TopicOverview:
        rows = self._load()
        if not rows:
            return TopicOverview(query=query, topics=[])
        if self.source == "packaged-v1":
            return self._search_packaged(query, rows, limit)
        return self._search_legacy(query, rows, limit)
