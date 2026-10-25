"""Evaluate v2 tool retrieval offline or the agent end-to-end.

    python scripts/run_v2_eval.py
    python scripts/run_v2_eval.py --agent --live --limit 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hotspot.api import Engine  # noqa: E402
from hotspot.config import load_config, paths_from  # noqa: E402
from hotspot.corpus import open_corpus  # noqa: E402
from hotspot.index.lance_store import LancePaperStore  # noqa: E402


def _metrics(ids: list[str], relevant: set[str], k: int) -> dict:
    top = ids[:k]
    hit = [1.0 if pid in relevant else 0.0 for pid in top]
    n = len(relevant)
    gains = [1.0 if pid in relevant else 0.0 for pid in ids[:10]]
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n, 10)))
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    reciprocal = next(
        (1.0 / (i + 1) for i, pid in enumerate(ids) if pid in relevant),
        0.0,
    )
    return {
        "recall": sum(hit) / n if n else None,
        "ndcg10": dcg / idcg if idcg else None,
        "mrr": reciprocal,
    }


def tool_eval(
    questions_path: Path,
    *,
    limit: int,
    question_type: str | None = None,
    qrels_path: Path | None = None,
) -> dict:
    cfg = load_config()
    corpus = open_corpus(cfg)
    store = LancePaperStore(cfg)
    items = json.loads(questions_path.read_text(encoding="utf-8"))
    if question_type:
        items = [item for item in items if item.get("type") == question_type]
    items = items[:limit or None]
    qrels: dict[str, set[str]] = {}
    if qrels_path and qrels_path.exists():
        for line in qrels_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("grade") is not None and int(row["grade"]) > 0:
                qrels.setdefault(str(row["query_id"]), set()).add(
                    str(row["paper_id"])
                )
    rows = []
    for item in items:
        relevant = qrels.get(
            str(item["id"]),
            set(item.get("relevant_ids") or []),
        )
        started = time.perf_counter()
        result = store.search(
            str(item["query"]),
            keywords=item.get("keywords") or [],
            limit=50,
        )
        elapsed = round(time.perf_counter() - started, 3)
        ids = [hit.paper_id for hit in result.results]
        rows.append({
            "id": item["id"],
            "n": len(ids),
            "elapsed_s": elapsed,
            **({"recall10": _metrics(ids, relevant, 10)["recall"],
                "recall50": _metrics(ids, relevant, 50)["recall"],
                "ndcg10": _metrics(ids, relevant, 10)["ndcg10"],
                "mrr": _metrics(ids, relevant, 10)["mrr"]}
               if relevant else {}),
        })

    probes = []
    for item in items:
        for pid in item.get("relevant_ids") or []:
            paper = corpus.get(pid)
            if paper is None:
                continue
            result = store.search(paper.title, limit=10)
            probes.append(pid in {hit.paper_id for hit in result.results})

    def avg(name: str):
        vals = [row[name] for row in rows if row.get(name) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "n": len(rows),
        "recall10": avg("recall10"),
        "recall50": avg("recall50"),
        "ndcg10": avg("ndcg10"),
        "mrr": avg("mrr"),
        "elapsed_s": avg("elapsed_s"),
        "title_probes": len(probes),
        "title_probe_hits": sum(probes),
        "rows": rows,
    }


def agent_eval(
    questions_path: Path,
    *,
    limit: int,
    question_type: str | None = None,
) -> dict:
    cfg = load_config()
    engine = Engine(cfg=cfg)
    items = json.loads(questions_path.read_text(encoding="utf-8"))
    if question_type:
        items = [item for item in items if item.get("type") == question_type]
    items = items[:limit or None]
    rows = []
    for item in items:
        resp = engine.ask({
            "question": item["question"],
            "source": item.get("source", "meeting"),
        })
        statuses = [v.status for v in resp.verdicts]
        rows.append({
            "id": item["id"],
            "route": resp.route,
            "refused": resp.refused,
            "citations": [c.id for c in resp.citations],
            "claims": len(statuses),
            "supported": statuses.count("supported"),
            "error": resp.error,
            "elapsed_s": resp.usage.elapsed_s,
        })
        print(
            f"{item['id']} | {resp.route} | refused={resp.refused} | "
            f"claims={len(statuses)} | citations={len(resp.citations)}"
        )
    return {"n": len(rows), "rows": rows}


def holdout_is_retired(questions_path: Path) -> bool:
    manifest = questions_path.with_name(
        f"{questions_path.stem}.manifest.json"
    )
    if not manifest.exists():
        return False
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return bool(
        data.get("do_not_run_again")
        or data.get("status") == "retired"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--questions", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--type", default=None, help="只跑指定问题类型")
    parser.add_argument("--out", default=None)
    parser.add_argument("--qrels", default=None)
    args = parser.parse_args()
    cfg = load_config()
    ws = paths_from(cfg)["workspace"]
    if args.agent:
        if not args.live:
            parser.error("--agent requires --live")
        path = Path(args.questions or ROOT / "eval" / "questions_small.json")
        if holdout_is_retired(path):
            parser.error(
                "this holdout is retired and must not be run again; "
                "use a new versioned holdout for future acceptance"
            )
        result = agent_eval(
            path,
            limit=args.limit,
            question_type=args.type,
        )
        out = Path(args.out) if args.out else ws / "ledger" / "v2_agent_eval.json"
    else:
        path = Path(args.questions or ROOT / "eval" / "retrieval_questions.json")
        if holdout_is_retired(path):
            parser.error(
                "this holdout is retired and must not be run again; "
                "use a new versioned holdout for future acceptance"
            )
        result = tool_eval(
            path,
            limit=args.limit,
            question_type=args.type,
            qrels_path=Path(args.qrels) if args.qrels else None,
        )
        out = Path(args.out) if args.out else ws / "ledger" / "v2_tool_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.agent:
        print(
            f"v2 tools | n={result['n']} recall10={result['recall10']} "
            f"recall50={result['recall50']} ndcg10={result['ndcg10']} "
            f"mrr={result['mrr']} elapsed={result['elapsed_s']} "
            f"titles={result['title_probe_hits']}/{result['title_probes']} "
            f"qrels={args.qrels or 'seed'} "
            f"out={out}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
