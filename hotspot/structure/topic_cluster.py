"""Build a full-coverage, versioned topic snapshot from paper embeddings."""

from __future__ import annotations

import gzip
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import igraph as ig
import leidenalg
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

from ..corpus import Paper
from .venue_matrix import venue_profile

__all__ = [
    "SNAPSHOT_SCHEMA_VERSION",
    "build_topic_snapshot",
    "write_topic_snapshot",
]

SNAPSHOT_SCHEMA_VERSION = 1


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _remap_labels(labels: np.ndarray) -> np.ndarray:
    unique = np.unique(labels)
    mapping = {int(old): new for new, old in enumerate(unique)}
    return np.asarray([mapping[int(value)] for value in labels], dtype=np.int32)


def _bucket_indices(
    reduced: np.ndarray,
    *,
    n_buckets: int,
    max_bucket_size: int,
    seed: int,
) -> list[np.ndarray]:
    """Partition the corpus into small buckets for exact local kNN."""
    count = min(max(2, n_buckets), len(reduced))
    model = MiniBatchKMeans(
        n_clusters=count,
        batch_size=min(4096, max(256, len(reduced) // 8)),
        n_init=3,
        random_state=seed,
        max_iter=100,
        reassignment_ratio=0.01,
    )
    labels = model.fit_predict(reduced)

    def split(indices: np.ndarray) -> Iterable[np.ndarray]:
        if len(indices) <= max_bucket_size:
            yield indices
            return
        parts = max(2, math.ceil(len(indices) / max_bucket_size))
        local = MiniBatchKMeans(
            n_clusters=parts,
            batch_size=min(2048, max(128, len(indices) // 4)),
            n_init=3,
            random_state=seed,
            max_iter=100,
        ).fit_predict(reduced[indices])
        for part in range(parts):
            child = indices[local == part]
            if len(child):
                yield from split(child)

    buckets: list[np.ndarray] = []
    for label in range(count):
        indices = np.flatnonzero(labels == label)
        if len(indices):
            buckets.extend(split(indices))
    return buckets


def _build_knn_graph(
    reduced: np.ndarray,
    *,
    n_neighbors: int,
    n_buckets: int,
    max_bucket_size: int,
    seed: int,
) -> tuple[ig.Graph, dict]:
    """Build an approximate kNN graph using exact neighbors inside small buckets."""
    buckets = _bucket_indices(
        reduced,
        n_buckets=n_buckets,
        max_bucket_size=max_bucket_size,
        seed=seed,
    )
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    bucket_sizes = []
    for bucket in buckets:
        bucket_sizes.append(int(len(bucket)))
        if len(bucket) < 2:
            continue
        k = min(n_neighbors + 1, len(bucket))
        nn = NearestNeighbors(
            n_neighbors=k,
            metric="cosine",
            algorithm="brute",
            n_jobs=1,
        )
        nn.fit(reduced[bucket])
        distances, positions = nn.kneighbors(reduced[bucket])
        for row, global_index in enumerate(bucket):
            for position, distance in zip(positions[row][1:], distances[row][1:]):
                neighbor = int(bucket[int(position)])
                if neighbor == int(global_index):
                    continue
                edges.append((int(global_index), neighbor))
                weights.append(max(1e-4, 1.0 - float(distance)))

    graph = ig.Graph(n=len(reduced), edges=edges, directed=False)
    graph.es["weight"] = weights
    graph.simplify(combine_edges={"weight": "sum"})
    return graph, {
        "buckets": len(buckets),
        "bucket_size_min": min(bucket_sizes) if bucket_sizes else 0,
        "bucket_size_median": int(np.median(bucket_sizes)) if bucket_sizes else 0,
        "bucket_size_max": max(bucket_sizes) if bucket_sizes else 0,
        "edges": int(graph.ecount()),
        "mean_degree": round(
            (2.0 * graph.ecount() / max(1, len(reduced))), 3
        ),
    }


def _communities(
    graph: ig.Graph,
    *,
    resolution: float,
    seed: int,
) -> np.ndarray:
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=3,
        seed=seed,
    )
    return _remap_labels(np.asarray(partition.membership, dtype=np.int32))


def _split_large_clusters(
    labels: np.ndarray,
    reduced: np.ndarray,
    *,
    max_cluster_size: int,
    seed: int,
) -> np.ndarray:
    labels = labels.copy()
    next_label = int(labels.max()) + 1 if len(labels) else 0
    for old_label in np.unique(labels):
        members = np.flatnonzero(labels == old_label)
        if len(members) <= max_cluster_size:
            continue
        parts = max(2, math.ceil(len(members) / max_cluster_size))
        local = MiniBatchKMeans(
            n_clusters=parts,
            batch_size=min(4096, max(256, len(members) // 4)),
            n_init=5,
            random_state=seed,
            max_iter=150,
        ).fit_predict(reduced[members])
        for part in range(parts):
            child = members[local == part]
            if not len(child):
                continue
            labels[child] = next_label
            next_label += 1
    return _remap_labels(labels)


def _merge_small_clusters(
    labels: np.ndarray,
    reduced: np.ndarray,
    *,
    min_cluster_size: int,
) -> np.ndarray:
    labels = labels.copy()
    unique = np.unique(labels)
    centroids = {
        int(label): _normalize_rows(reduced[labels == label].mean(axis=0, keepdims=True))[0]
        for label in unique
    }
    large = [label for label in unique if np.sum(labels == label) >= min_cluster_size]
    if not large:
        return np.zeros(len(labels), dtype=np.int32)
    large_array = np.asarray(large, dtype=np.int32)
    large_centroids = np.vstack([centroids[int(label)] for label in large])
    for label in unique:
        if label in large:
            continue
        similarity = large_centroids @ centroids[int(label)]
        target = int(large_array[int(np.argmax(similarity))])
        labels[labels == label] = target
    return _remap_labels(labels)


def _cluster_terms(
    papers: list[Paper],
    labels: np.ndarray,
    *,
    top_n: int,
) -> tuple[dict[int, list[str]], float]:
    documents = [
        f"{paper.title} {(paper.abstract or '')[:1200]}"
        for paper in papers
    ]
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=20_000,
        token_pattern=r"(?u)\b[A-Za-z][A-Za-z0-9_-]{2,}\b",
    )
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return {int(label): [] for label in np.unique(labels)}, 0.0
    vocabulary = np.asarray(vectorizer.get_feature_names_out(), dtype=object)
    cluster_ids = np.unique(labels)
    cluster_counts = np.vstack(
        [np.asarray(matrix[labels == label].sum(axis=0)).ravel() for label in cluster_ids]
    ).astype(np.float64)
    totals = np.maximum(cluster_counts.sum(axis=1, keepdims=True), 1.0)
    frequencies = cluster_counts / totals
    document_frequency = np.maximum((cluster_counts > 0).sum(axis=0), 1)
    idf = np.log(1.0 + len(cluster_ids) / document_frequency)
    scores = frequencies * idf
    vocabulary_hits = float((cluster_counts.sum(axis=0) > 0).mean()) if len(vocabulary) else 0.0
    output: dict[int, list[str]] = {}
    for row, label in enumerate(cluster_ids):
        top = np.argsort(scores[row])[::-1][:top_n]
        output[int(label)] = [
            str(vocabulary[index]) for index in top if scores[row, index] > 0
        ]
    return output, vocabulary_hits


def _cluster_records(
    papers: list[Paper],
    labels: np.ndarray,
    reduced: np.ndarray,
    *,
    spread_cfg: dict,
    top_terms: int,
    representatives: int,
) -> tuple[list[dict], dict[int, list[str]], float]:
    terms_by_cluster, vocabulary_hits = _cluster_terms(
        papers,
        labels,
        top_n=top_terms,
    )
    records: list[dict] = []
    members_by_cluster: dict[int, list[str]] = {}
    venue_order = tuple(spread_cfg.get("venue_order") or ())
    spread_min = int(spread_cfg.get("spread_min") or 3)
    off_axis = tuple(spread_cfg.get("venue_off_axis") or ())

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        centroid = _normalize_rows(
            reduced[indices].mean(axis=0, keepdims=True)
        )[0]
        similarities = reduced[indices] @ centroid
        representative_indices = indices[
            np.argsort(similarities)[::-1][:representatives]
        ]
        member_ids = [papers[index].paper_id for index in indices]
        members_by_cluster[int(label)] = member_ids

        counts: Counter[str] = Counter()
        counts_by_year: dict[int, Counter[str]] = {}
        for index in indices:
            paper = papers[int(index)]
            venue = paper.venue or "unknown"
            counts[venue] += 1
            if paper.year is not None:
                counts_by_year.setdefault(int(paper.year), Counter())[venue] += 1
        profile = venue_profile(
            dict(counts),
            venue_order=venue_order,
            spread_min=spread_min,
            off_axis=off_axis,
            counts_by_year={
                year: dict(values) for year, values in counts_by_year.items()
            },
        )
        terms = terms_by_cluster.get(int(label), [])
        representative_papers = [papers[int(index)] for index in representative_indices]
        representative_title = (
            representative_papers[0].title if representative_papers else ""
        )
        records.append(
            {
                "cluster_id": int(label),
                "label": " / ".join(term.title() for term in terms[:4])
                or representative_title
                or f"cluster {int(label)}",
                "top_terms": terms,
                "size": int(len(indices)),
                "representative_ids": [p.paper_id for p in representative_papers],
                "representative_title": representative_title,
                "venue": profile.to_dict(),
                "centroid": [round(float(value), 7) for value in centroid],
            }
        )

    records.sort(key=lambda item: (-item["size"], item["cluster_id"]))
    return records, members_by_cluster, vocabulary_hits


def build_topic_snapshot(
    vectors: np.ndarray,
    ids: list[str],
    papers: list[Paper],
    *,
    pca_dim: int = 128,
    n_neighbors: int = 20,
    n_buckets: int | None = None,
    max_bucket_size: int = 600,
    resolution: float = 1.0,
    min_cluster_size: int = 20,
    max_cluster_size: int = 1_200,
    seed: int = 42,
    top_terms: int = 8,
    representatives: int = 5,
    spread_cfg: dict | None = None,
) -> dict:
    """Cluster all supplied papers and return a JSON-serializable snapshot."""
    if len(vectors) != len(ids):
        raise ValueError(
            f"vectors/ids length mismatch: {len(vectors)} != {len(ids)}"
        )
    if len(papers) != len(ids):
        raise ValueError(
            f"papers/ids length mismatch: {len(papers)} != {len(ids)}"
        )
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate paper ids in vectors")
    if not len(ids):
        raise ValueError("cannot build a topic snapshot from an empty corpus")

    started = time.perf_counter()
    original_dim = int(vectors.shape[1])
    dimensions = min(max(2, pca_dim), original_dim, len(ids) - 1)
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix = _normalize_rows(matrix)

    pca = PCA(
        n_components=dimensions,
        svd_solver="randomized" if dimensions < min(matrix.shape) else "full",
        random_state=seed,
    )
    reduced = pca.fit_transform(matrix).astype(np.float32)
    reduced = _normalize_rows(reduced)

    if n_buckets is None:
        n_buckets = min(512, max(64, round(math.sqrt(len(ids)) * 1.5)))
    graph, graph_stats = _build_knn_graph(
        reduced,
        n_neighbors=n_neighbors,
        n_buckets=n_buckets,
        max_bucket_size=max_bucket_size,
        seed=seed,
    )
    labels = _communities(graph, resolution=resolution, seed=seed)
    communities_before = int(len(np.unique(labels)))
    labels = _split_large_clusters(
        labels,
        reduced,
        max_cluster_size=max_cluster_size,
        seed=seed,
    )
    clusters_after_split = int(len(np.unique(labels)))
    labels = _merge_small_clusters(
        labels,
        reduced,
        min_cluster_size=min_cluster_size,
    )
    records, members_by_cluster, vocabulary_hits = _cluster_records(
        papers,
        labels,
        reduced,
        spread_cfg=spread_cfg or {},
        top_terms=top_terms,
        representatives=representatives,
    )
    member_counts = [item["size"] for item in records]
    elapsed = time.perf_counter() - started
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_size": len(ids),
        "coverage": 1.0,
        "cluster_count": len(records),
        "parameters": {
            "pca_dim": dimensions,
            "original_dim": original_dim,
            "n_neighbors": n_neighbors,
            "n_buckets": n_buckets,
            "max_bucket_size": max_bucket_size,
            "resolution": resolution,
            "min_cluster_size": min_cluster_size,
            "max_cluster_size": max_cluster_size,
            "seed": seed,
        },
        "metrics": {
            "elapsed_s": round(elapsed, 3),
            "communities_before_postprocess": communities_before,
            "clusters_after_split": clusters_after_split,
            "cluster_size_min": min(member_counts),
            "cluster_size_median": int(np.median(member_counts)),
            "cluster_size_max": max(member_counts),
            "vocabulary_coverage": round(vocabulary_hits, 4),
            **graph_stats,
        },
        "pca": {
            "mean": [round(float(value), 7) for value in pca.mean_],
            "components": [
                [round(float(value), 7) for value in row]
                for row in pca.components_
            ],
            "explained_variance": round(
                float(np.sum(pca.explained_variance_ratio_)), 6
            ),
        },
        "paper_to_cluster": {
            paper_id: int(labels[index]) for index, paper_id in enumerate(ids)
        },
        "clusters": records,
        "members": members_by_cluster,
    }


def write_topic_snapshot(snapshot: dict, path: Path) -> dict:
    """Write a deterministic gzip snapshot and return its provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        snapshot,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    with gzip.open(path, "wb", compresslevel=9) as handle:
        handle.write(payload)
    import hashlib

    payload_digest = hashlib.sha256(payload).hexdigest()
    file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "uncompressed_bytes": len(payload),
        "payload_sha256": payload_digest,
        "file_sha256": file_digest,
    }
