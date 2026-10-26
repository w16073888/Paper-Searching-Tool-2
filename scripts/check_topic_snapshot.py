"""Validate coverage and schema invariants of the packaged topic snapshot."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hotspot.config import load_config  # noqa: E402
from hotspot.index.topic_store import TopicStore  # noqa: E402


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    cfg = load_config()
    default = str(
        (cfg.get("topics") or {}).get("snapshot")
        or "hotspot/data/topic_snapshot_v1.json.gz"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=default)
    args = parser.parse_args()
    path = _resolve(args.snapshot)
    failures: list[str] = []
    if not path.is_file():
        print(f"✗ snapshot missing: {path}")
        return 1
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        snapshot = json.load(handle)

    clusters = snapshot.get("clusters") or []
    mapping = snapshot.get("paper_to_cluster") or {}
    corpus_size = int(snapshot.get("corpus_size") or 0)
    cluster_ids = {int(cluster["cluster_id"]) for cluster in clusters}
    member_total = sum(int(cluster.get("size") or 0) for cluster in clusters)
    centroid_dims = {
        len(cluster.get("centroid") or [])
        for cluster in clusters
    }
    empty_terms = [
        cluster["cluster_id"] for cluster in clusters
        if not (cluster.get("top_terms") or [])
    ]

    if int(snapshot.get("schema_version") or 0) != 1:
        failures.append("schema_version != 1")
    if corpus_size <= 0:
        failures.append("corpus_size <= 0")
    if float(snapshot.get("coverage") or 0.0) != 1.0:
        failures.append("coverage != 1.0")
    if len(clusters) != int(snapshot.get("cluster_count") or 0):
        failures.append("cluster_count mismatch")
    if member_total != corpus_size:
        failures.append(f"member total {member_total} != corpus {corpus_size}")
    if len(mapping) != corpus_size:
        failures.append(f"paper mapping {len(mapping)} != corpus {corpus_size}")
    if set(int(value) for value in mapping.values()) != cluster_ids:
        failures.append("paper mapping references unknown clusters")
    if len(centroid_dims) != 1 or 0 in centroid_dims:
        failures.append(f"centroid dimensions invalid: {sorted(centroid_dims)}")
    if empty_terms:
        failures.append(f"clusters without top_terms: {len(empty_terms)}")

    store = TopicStore(cfg)
    if store.source != "packaged-v1":
        failures.append(f"TopicStore source is {store.source}, expected packaged-v1")
    missing_store = TopicStore(
        {
            "paths": {"workspace": str(path.parent / "__missing_workspace__")},
            "topics": {"snapshot": str(path.parent / "__missing_snapshot__.json.gz")},
        }
    )
    if missing_store.available:
        failures.append("missing snapshot incorrectly reports available")
    if missing_store.search("RLHF").topics:
        failures.append("missing snapshot incorrectly returns topics")
    if not missing_store.error:
        failures.append("missing snapshot does not expose an error")

    print(
        f"{'✓' if not failures else '✗'} "
        f"snapshot={path.name} corpus={corpus_size} "
        f"clusters={len(clusters)} centroid_dim={sorted(centroid_dims)} "
        f"source={store.source}"
    )
    for failure in failures:
        print(f"- {failure}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
