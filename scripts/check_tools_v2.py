"""Offline checks for the LanceDB search and fetch tools.

    python scripts/check_tools_v2.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=SyntaxWarning)

from hotspot.config import load_config
from hotspot.corpus import open_corpus
from hotspot.core.models import SearchFilters
from hotspot.index.lance_store import LancePaperStore
from hotspot.index.topic_store import TopicStore
from hotspot.tools.papers import fetch_papers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--trend-limit", type=int, default=3)
    parser.add_argument("--min-trend-rate", type=float, default=0.8)
    args = parser.parse_args()
    cfg = load_config()
    store = LancePaperStore(cfg)
    corpus = open_corpus(cfg)
    if not store.available:
        print("✗ LanceDB index missing")
        return 1

    questions = json.loads(
        (ROOT / "eval" / "retrieval_questions.json").read_text(encoding="utf-8")
    )
    probes = [
        pid
        for item in questions
        for pid in item.get("relevant_ids", [])
        if corpus.get(pid) is not None
    ][:max(1, args.limit)]

    hits = 0
    failures = []
    for pid in probes:
        paper = corpus.get(pid)
        result = store.search(paper.title, limit=10)
        returned = [item.paper_id for item in result.results]
        if pid in returned:
            hits += 1
        else:
            failures.append(pid)

    fetched = fetch_papers(store, probes[:3])
    filtered = store.search(
        "reinforcement learning",
        filters=SearchFilters(years=[2025]),
        limit=10,
    )
    years_ok = all(hit.year == 2025 for hit in filtered.results)
    topic_store = TopicStore(cfg)
    trend_questions = json.loads(
        (ROOT / "eval" / "questions.json").read_text(encoding="utf-8")
    )
    trend_questions = [
        item["question"] for item in trend_questions
        if item.get("type") == "global"
    ][:max(1, args.trend_limit)]
    topic_hits = []
    for question in trend_questions:
        topics = topic_store.search(question, limit=3)
        topic_hits.append(len(topics.topics))
    trend_rate = (
        sum(bool(hit) for hit in topic_hits) / len(topic_hits)
        if topic_hits else 0.0
    )
    tool_ok = (
        not failures
        and bool(fetched)
        and years_ok
        and trend_rate >= args.min_trend_rate
    )
    print(
        f"{'✓' if tool_ok else '✗'} "
        f"tool_v2 probes={len(probes)} title_hits={hits} "
        f"fetched={len(fetched)} years_ok={years_ok} "
        f"topics={sum(bool(hit) for hit in topic_hits)}/{len(trend_questions)} "
        f"rate={trend_rate:.3f} "
        f"snapshot={topic_store.source}"
    )
    if not tool_ok:
        print("misses:", failures)
        print("trend_hits:", topic_hits)
        if topic_store.error:
            print("topic_error:", topic_store.error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
