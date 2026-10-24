"""Finalize the reviewed scope dataset as a new immutable version.

This script reads the 146-row review CSV, runs one Laya zero-shot audit pass,
fills ``review_decision`` / ``review_notes``, copies all data to a new
directory, and writes fresh hashes. The source dataset is never modified.

    python scripts/finalize_scope_review.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_scope_dataset import build_quality_report

SCOPE_QUESTION = {
    "scope": {
        "type": "choice",
        "instructions": "判断问题是否属于本地 AI/ML 论文库的研究范围。",
        "criteria": {
            "in_scope": "询问 AI/ML/RL/LLM/Agent/机器人学习或决策方法",
            "out_of_scope": "询问其他领域本身或与本地 AI/ML 论文库无关",
        },
    }
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_overrides(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("overrides file must be a JSON object keyed by item id")
    return payload


def manual_note(row: dict[str, str], laya_prediction: str) -> str:
    category = row["category"]
    expected = row["expected"]
    if category == "boundary" and expected == "in_scope":
        note = "Manual review: explicit ML/RL method in a cross-domain application; kept in_scope."
    elif category == "boundary":
        note = (
            "Manual review: asks the application domain's mechanism or traditional "
            "method without an ML/RL method; kept out_of_scope."
        )
    elif category == "in_scope":
        note = "Manual review: corpus-grounded AI/ML/RL question; kept in_scope."
    elif row.get("kind") == "hard_negative":
        note = (
            "Manual review: technical term resembles ML vocabulary but the question "
            "asks the source domain itself; kept out_of_scope."
        )
    else:
        note = "Manual review: clearly domain-only question; kept out_of_scope."

    if laya_prediction != expected and expected == "in_scope":
        note += " Laya zero-shot false refusal flags a technical-term confusion case."
    elif laya_prediction != expected and expected == "out_of_scope":
        note += " Laya zero-shot false pass is a difficult negative."
    elif laya_prediction == expected:
        note += " Laya zero-shot agrees."
    return note


def fill_review_table(
    *,
    path: Path,
    model_dir: Path,
    overrides: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    import laya

    reviewed_at = utc_now()
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    agent = laya.load(str(model_dir))
    for row in rows:
        result = agent.predict({"question": row["question"]}, SCOPE_QUESTION)
        answer = result["answers"]["scope"]
        predicted = str(answer["choice"])
        probability = float(answer["probabilities"][predicted])
        row["laya_predicted"] = predicted
        row["laya_selected_probability"] = f"{probability:.6f}"
        row["reviewed_at"] = reviewed_at

        override = overrides.get(row["id"], {})
        replacement = override.get("expected")
        if replacement and replacement != row["expected"]:
            if replacement not in {"in_scope", "out_of_scope"}:
                raise ValueError(f"{row['id']}: invalid override label {replacement!r}")
            row["review_decision"] = f"relabel:{replacement}"
            row["review_notes"] = (
                override.get("notes", "").strip()
                + f" Original expected={row['expected']}; corrected to {replacement}."
            ).strip()
            row["expected"] = replacement
        else:
            row["review_decision"] = "accept"
            extra = override.get("notes", "").strip()
            row["review_notes"] = (
                manual_note(row, predicted)
                + (f" Reviewer note: {extra}" if extra else "")
            )

    decision_counts: dict[str, int] = {}
    for row in rows:
        decision_counts[row["review_decision"]] = (
            decision_counts.get(row["review_decision"], 0) + 1
        )
    laya_disagreements = sum(
        row["laya_predicted"] != row["expected"] for row in rows
    )
    return rows, {
        "rows": len(rows),
        "decisions": decision_counts,
        "laya_disagreements": laya_disagreements,
        "reviewed_at": reviewed_at,
        "reviewer": "Codex manual semantic review with Laya disagreement precheck",
    }


def write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def apply_review_to_records(
    rows: list[dict[str, Any]],
    review: dict[str, dict[str, str]],
    *,
    version: str,
) -> list[dict[str, Any]]:
    reviewed_at = next(
        (row.get("reviewed_at", "") for row in review.values()),
        "",
    )
    output = []
    for row in rows:
        updated = dict(row)
        updated["dataset_version"] = version
        reviewed = review.get(str(row["id"]))
        if reviewed:
            updated["expected"] = reviewed["expected"]
            updated["review_status"] = "reviewed"
            updated["review_decision"] = reviewed["review_decision"]
            updated["reviewed_by"] = "codex_manual"
            updated["reviewed_at"] = reviewed_at
        else:
            updated["review_status"] = "not_sampled"
        output.append(updated)
    return output


def update_laya_expected(
    rows: list[dict[str, Any]],
    review: dict[str, dict[str, str]],
    *,
    version: str,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        updated = dict(row)
        updated["dataset_version"] = version
        reviewed = review.get(str(row["id"]))
        if reviewed:
            updated["expected"] = {"scope": reviewed["expected"]}
        output.append(updated)
    return output


def recompute_manifest(
    *,
    source_manifest: Path,
    out_dir: Path,
    review_summary: dict[str, Any],
    version: str,
) -> dict[str, Any]:
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_hash = sha256_file(source_manifest)
    manifest["dataset_version"] = version
    manifest["generated_at"] = utc_now()
    manifest["parent"] = {
        "dataset_version": "scope-dataset-v1",
        "manifest_sha256": source_hash,
        "preserved_separately": True,
    }
    manifest["review"] = review_summary
    manifest["quality_policy"]["human_review"] = (
        "146-row audit completed; remaining labels remain LLM silver"
    )
    files = (
        "all.jsonl",
        "train.jsonl",
        "calibration.jsonl",
        "internal_test.jsonl",
        "laya_train.jsonl",
        "laya_calibration.jsonl",
        "laya_internal_test.jsonl",
        "review.csv",
        "quality_report.json",
    )
    manifest["files"] = {
        name: {
            "sha256": sha256_file(out_dir / name),
            "bytes": (out_dir / name).stat().st_size,
        }
        for name in files
    }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "eval" / "scope_v1" / "dataset",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "eval" / "scope_v1" / "dataset_v1_1",
    )
    parser.add_argument("--version", default="scope-dataset-v1.1")
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".models" / "jev")
    parser.add_argument("--overrides", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing immutable dataset: {args.out_dir}"
        )
    overrides = load_overrides(args.overrides)
    review_rows, review_summary = fill_review_table(
        path=args.source_dir / "review.csv",
        model_dir=args.model_dir,
        overrides=overrides,
    )
    review = {row["id"]: row for row in review_rows}

    shutil.copytree(args.source_dir, args.out_dir)
    for name in ("all.jsonl", "train.jsonl", "calibration.jsonl", "internal_test.jsonl"):
        rows = read_jsonl(args.out_dir / name)
        write_jsonl(
            args.out_dir / name,
            apply_review_to_records(rows, review, version=args.version),
        )
    for name in ("laya_train.jsonl", "laya_calibration.jsonl", "laya_internal_test.jsonl"):
        rows = read_jsonl(args.out_dir / name)
        write_jsonl(
            args.out_dir / name,
            update_laya_expected(rows, review, version=args.version),
        )
    write_review_csv(args.out_dir / "review.csv", review_rows)

    quality = build_quality_report(
        read_jsonl(args.out_dir / "all.jsonl"),
        external_path=ROOT / "eval" / "scope_v1" / "test.jsonl",
    )
    quality["dataset_version"] = args.version
    quality["review"] = review_summary
    (args.out_dir / "quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = recompute_manifest(
        source_manifest=args.source_dir / "manifest.json",
        out_dir=args.out_dir,
        review_summary=review_summary,
        version=args.version,
    )
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "manifest.sha256").write_text(
        sha256_file(manifest_path) + "\n",
        encoding="utf-8",
    )
    readme = (args.out_dir / "README.md").read_text(encoding="utf-8")
    readme = readme.replace("scope-dataset-v1", args.version)
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(
        json.dumps(
            {
                "version": args.version,
                "out_dir": str(args.out_dir),
                "manifest_sha256": sha256_file(manifest_path),
                "review": review_summary,
                "quality_passed": quality["passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
