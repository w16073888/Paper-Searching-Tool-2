"""Score a frozen comparison holdout result without running the model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile_nearest_rank(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return round(float(ordered[index]), 3)


def evaluate(
    questions_path: Path,
    manifest_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))

    actual_hash = sha256_file(questions_path)
    if manifest.get("questions_sha256") != actual_hash:
        raise ValueError("holdout question hash does not match manifest")
    if manifest.get("used_for_tuning") is not False:
        raise ValueError("holdout manifest must set used_for_tuning=false")

    rows = result.get("rows") or []
    by_id = {str(row["id"]): row for row in rows}
    expected_ids = [str(item["id"]) for item in questions]
    if set(by_id) != set(expected_ids):
        raise ValueError("result IDs do not match holdout questions")
    if len(rows) != len(expected_ids):
        raise ValueError("result contains duplicate or missing rows")

    per_question = []
    answered = 0
    errors = 0
    claims_total = 0
    supported_total = 0
    fully_grounded = 0
    latencies = []
    citation_counts = []
    for item in questions:
        row = by_id[str(item["id"])]
        claims = int(row.get("claims") or 0)
        supported = int(row.get("supported") or 0)
        citations = len(row.get("citations") or [])
        has_error = row.get("route") == "error" or row.get("error") is not None
        ok = (
            not has_error
            and row.get("route") == "local"
            and row.get("refused") is False
            and supported > 0
            and citations > 0
        )
        answered += int(ok)
        errors += int(has_error)
        claims_total += claims
        supported_total += supported
        fully_grounded += int(claims > 0 and supported == claims)
        latencies.append(float(row.get("elapsed_s") or 0.0))
        citation_counts.append(citations)
        per_question.append({
            "id": item["id"],
            "route": row.get("route"),
            "refused": bool(row.get("refused")),
            "claims": claims,
            "supported": supported,
            "filtered": max(claims - supported, 0),
            "citations": citations,
            "elapsed_s": round(float(row.get("elapsed_s") or 0.0), 3),
            "operational_pass": ok,
        })

    total = len(questions)
    answer_rate = answered / total if total else 0.0
    supported_rate = supported_total / claims_total if claims_total else 0.0
    full_rate = fully_grounded / total if total else 0.0
    p95 = percentile_nearest_rank(latencies, 0.95)
    thresholds = {
        "minimum_answer_rate": 1.0,
        "minimum_supported_claim_rate": 0.85,
        "minimum_full_grounding_rate": 0.70,
        "maximum_p95_seconds": 60.0,
        "maximum_errors": 0,
    }
    gates = {
        "answer_rate": answer_rate >= thresholds["minimum_answer_rate"],
        "supported_claim_rate":
            supported_rate >= thresholds["minimum_supported_claim_rate"],
        "full_grounding_rate":
            full_rate >= thresholds["minimum_full_grounding_rate"],
        "p95_latency": p95 <= thresholds["maximum_p95_seconds"],
        "errors": errors <= thresholds["maximum_errors"],
    }
    return {
        "name": manifest.get("name"),
        "version": manifest.get("version"),
        "questions_sha256": actual_hash,
        "result_file": str(result_path),
        "run_count": int(manifest.get("run_count") or 1),
        "n_questions": total,
        "answered": answered,
        "answer_rate": round(answer_rate, 4),
        "errors": errors,
        "claims_total": claims_total,
        "supported_total": supported_total,
        "filtered_total": max(claims_total - supported_total, 0),
        "supported_claim_rate": round(supported_rate, 4),
        "fully_grounded_questions": fully_grounded,
        "full_grounding_rate": round(full_rate, 4),
        "citation_min": min(citation_counts) if citation_counts else 0,
        "citation_max": max(citation_counts) if citation_counts else 0,
        "avg_seconds": round(sum(latencies) / total, 3) if total else 0.0,
        "p95_seconds": p95,
        "thresholds": thresholds,
        "gates": gates,
        "automatic_gate_passed": all(gates.values()),
        "limitations": [
            "The automatic gate checks routing, citation-backed claims, and latency; "
            "it does not establish human-judged semantic correctness."
        ],
        "per_question": per_question,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        default=str(ROOT / "eval" / "v2_comparison_holdout_v1.json"),
    )
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "eval" / "v2_comparison_holdout_v1.manifest.json"),
    )
    parser.add_argument("--result", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    summary = evaluate(
        Path(args.questions),
        Path(args.manifest),
        Path(args.result),
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["automatic_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
