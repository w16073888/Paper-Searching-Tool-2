"""Select a two-threshold scope gate on calibration and evaluate it once.

The gate has three outcomes:

* high-confidence in_scope -> local Laya;
* high-confidence out_of_scope -> local Laya;
* otherwise -> fallback LLM.

Thresholds are selected only on calibration. The internal test set is loaded
after thresholds are frozen and is used for development-period evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from laya.common import uses_parallel_layout
from laya.train import (
    calibration_records,
    evaluate_records,
    items_from_rows,
    load_checkpoint,
    read_data,
    resolve_device,
)

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(line_buffering=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def source_label(source: dict[str, Any]) -> str:
    expected = source.get("expected")
    if isinstance(expected, dict):
        return str(expected.get("scope", ""))
    return str(expected or "")


def softmax(logits: Any, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    values -= values.max()
    exp = np.exp(values)
    return exp / exp.sum()


def attach_sources(items: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    if len(items) != len(rows):
        raise ValueError(f"items={len(items)} rows={len(rows)}")
    for item, row in zip(items, rows):
        item["_source"] = row


def prediction_rows(
    *,
    records: list[tuple[int, Any, Any, int]],
    items: list[dict[str, Any]],
    temperature: float,
) -> list[dict[str, Any]]:
    output = []
    for record, item in zip(records, items):
        _qtype, logits_raw, _target, k = record
        logits = np.asarray(logits_raw, dtype=float)[:k]
        probabilities = softmax(logits, temperature)
        options = list(item["q"]["crit"].keys())
        in_index = options.index("in_scope")
        p_in = float(probabilities[in_index])
        p_out = float(1.0 - p_in)
        predicted = "in_scope" if p_in >= p_out else "out_of_scope"
        source = item["_source"]
        output.append(
            {
                "id": source["id"],
                "expected": source_label(source),
                "category": source.get("category", ""),
                "group": (
                    "OOD"
                    if source.get("category") == "ood"
                    else "boundary"
                    if source.get("category") == "boundary"
                    else source.get("kind", "")
                ),
                "language": source.get("language", ""),
                "predicted": predicted,
                "p_in_scope": p_in,
                "p_out_of_scope": p_out,
            }
        )
    return output


def confusion_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("predicted") in {"in_scope", "out_of_scope"}]
    positives = [row for row in valid if row["expected"] == "in_scope"]
    negatives = [row for row in valid if row["expected"] == "out_of_scope"]
    tp = sum(row["predicted"] == "in_scope" for row in positives)
    fn = sum(row["predicted"] == "out_of_scope" for row in positives)
    tn = sum(row["predicted"] == "out_of_scope" for row in negatives)
    fp = sum(row["predicted"] == "in_scope" for row in negatives)
    return {
        "items": len(rows),
        "evaluated": len(valid),
        "errors": len(rows) - len(valid),
        "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        "accuracy": round((tp + tn) / max(1, tp + tn + fp + fn), 4),
        "in_scope_recall": round(tp / max(1, tp + fn), 4),
        "in_scope_false_refusal_rate": round(fn / max(1, tp + fn), 4),
        "out_of_scope_recall": round(tn / max(1, tn + fp), 4),
        "ood_false_pass_rate": round(fp / max(1, tn + fp), 4),
    }


def grouped_metrics(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field) or "unknown")].append(row)
    return {
        key: confusion_metrics(group)
        for key, group in sorted(grouped.items())
    }


def select_thresholds(
    rows: list[dict[str, Any]],
    *,
    max_false_refusal: float,
    max_false_pass: float,
) -> dict[str, Any]:
    positives = [row for row in rows if row["expected"] == "in_scope"]
    negatives = [row for row in rows if row["expected"] == "out_of_scope"]
    candidates = [round(value, 3) for value in np.arange(0.50, 1.0, 0.005)]
    candidates.append(1.0)
    best: dict[str, Any] | None = None
    for tau_in in candidates:
        for tau_out in candidates:
            local_in = [
                row
                for row in negatives
                if row["predicted"] == "in_scope"
                and row["p_in_scope"] >= tau_in
            ]
            local_out = [
                row
                for row in positives
                if row["predicted"] == "out_of_scope"
                and row["p_out_of_scope"] >= tau_out
            ]
            false_pass = len(local_in) / max(1, len(negatives))
            false_refusal = len(local_out) / max(1, len(positives))
            if false_pass > max_false_pass or false_refusal > max_false_refusal:
                continue

            decisions = []
            for row in rows:
                if row["predicted"] == "in_scope" and row["p_in_scope"] >= tau_in:
                    decisions.append({**row, "gated_predicted": "in_scope", "route": "laya"})
                elif row["predicted"] == "out_of_scope" and row["p_out_of_scope"] >= tau_out:
                    decisions.append({**row, "gated_predicted": "out_of_scope", "route": "laya"})
                else:
                    decisions.append({**row, "gated_predicted": None, "route": "llm_fallback"})
            coverage = sum(row["route"] == "laya" for row in decisions) / max(1, len(rows))
            accepted = [row for row in decisions if row["route"] == "laya"]
            accepted_accuracy = (
                sum(row["gated_predicted"] == row["expected"] for row in accepted)
                / max(1, len(accepted))
            )
            candidate = {
                "tau_in_scope": tau_in,
                "tau_out_of_scope": tau_out,
                "coverage": round(coverage, 4),
                "local_high_confidence_false_refusal_rate": round(false_refusal, 4),
                "local_high_confidence_false_pass_rate": round(false_pass, 4),
                "accepted_accuracy": round(accepted_accuracy, 4),
                "decisions": decisions,
            }
            rank = (
                candidate["coverage"],
                candidate["accepted_accuracy"],
                candidate["tau_in_scope"] + candidate["tau_out_of_scope"],
            )
            if best is None or rank > best["rank"]:
                best = {**candidate, "rank": rank}
    if best is None:
        raise RuntimeError("no threshold pair satisfies the requested risk constraints")
    best.pop("rank", None)
    return best


def apply_gate(
    rows: list[dict[str, Any]],
    *,
    tau_in: float,
    tau_out: float,
    fallback: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row["predicted"] == "in_scope" and row["p_in_scope"] >= tau_in:
            output.append({**row, "route": "laya", "final_predicted": "in_scope", "error": None})
        elif row["predicted"] == "out_of_scope" and row["p_out_of_scope"] >= tau_out:
            output.append({**row, "route": "laya", "final_predicted": "out_of_scope", "error": None})
        else:
            llm = fallback.get(row["id"], {})
            output.append(
                {
                    **row,
                    "route": "llm_fallback",
                    "final_predicted": llm.get("predicted"),
                    "error": llm.get("error"),
                }
            )
    return output


def gate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = confusion_metrics(
        [
            {**row, "predicted": row.get("final_predicted")}
            for row in rows
        ]
    )
    accepted = [row for row in rows if row["route"] == "laya"]
    result["laya_coverage"] = round(len(accepted) / max(1, len(rows)), 4)
    result["fallback_rate"] = round(1.0 - result["laya_coverage"], 4)
    result["laya_accepted_accuracy"] = round(
        sum(row["final_predicted"] == row["expected"] for row in accepted)
        / max(1, len(accepted)),
        4,
    )
    return result


def load_llm_predictions(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in payload["rows"]}


def load_rich_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {str(row["id"]): row for row in rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / ".models" / "jev" / "ft" / "scope-v1-head-only-seed0",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "eval" / "scope_v1" / "dataset_v1_1",
    )
    parser.add_argument(
        "--llm-results",
        type=Path,
        default=ROOT / "eval" / "scope_v1" / "results_internal_v1_1" / "llm_default.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "eval" / "scope_v1" / "gate_report.json",
    )
    parser.add_argument("--max-false-refusal", type=float, default=0.02)
    parser.add_argument("--max-false-pass", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model, tokenizer, cfg = load_checkpoint(str(args.model))
    device = resolve_device(args.device)
    model.to(device)
    max_len = int(cfg.get("max_len") or 256)
    head_max_len = int(cfg.get("head_max_len") or 256)
    temperature = list(cfg.get("temperature") or [1.0, 1.0, 1.0])
    temp_by_options = cfg.get("temperature_by_options") or {}
    parallel = uses_parallel_layout(cfg)

    def records_for(filename: str, rich_filename: str):
        rows = read_data(str(args.data_dir / filename))
        rich_rows = load_rich_rows(args.data_dir / rich_filename)
        items, skipped = items_from_rows(
            tokenizer,
            rows,
            max_len,
            head_max_len,
        )
        if skipped:
            raise RuntimeError(f"{filename}: skipped rows {skipped}")
        attach_sources(
            items,
            [rich_rows.get(str(row["id"]), row) for row in rows],
        )
        records = calibration_records(
            model,
            tokenizer,
            items,
            device,
            max_len,
            head_max_len,
            parallel=parallel,
        )
        predictions = prediction_rows(
            records=records,
            items=items,
            temperature=float(temperature[0]),
        )
        metrics = evaluate_records(records, temperature, temp_by_options)
        return predictions, metrics

    calibration_rows, calibration_metrics = records_for(
        "laya_calibration.jsonl",
        "calibration.jsonl",
    )
    selected = select_thresholds(
        calibration_rows,
        max_false_refusal=args.max_false_refusal,
        max_false_pass=args.max_false_pass,
    )
    calibration_decisions = selected.pop("decisions")
    calibration_accepted = [
        row for row in calibration_decisions if row["route"] == "laya"
    ]
    calibration_gate = confusion_metrics(
        [
            {**row, "predicted": row["gated_predicted"]}
            for row in calibration_accepted
        ]
    )
    calibration_gate["accepted_items"] = len(calibration_accepted)

    test_rows, test_metrics = records_for(
        "laya_internal_test.jsonl",
        "internal_test.jsonl",
    )
    llm_rows = load_llm_predictions(args.llm_results)
    test_gate_rows = apply_gate(
        test_rows,
        tau_in=float(selected["tau_in_scope"]),
        tau_out=float(selected["tau_out_of_scope"]),
        fallback=llm_rows,
    )
    test_gate = gate_metrics(test_gate_rows)
    test_gate["by_group"] = grouped_metrics(test_gate_rows, "group")
    test_gate["by_language"] = grouped_metrics(test_gate_rows, "language")
    test_gate["by_category"] = grouped_metrics(test_gate_rows, "category")
    target_checks = {
        "in_scope_false_refusal_le_0_02": (
            test_gate["in_scope_false_refusal_rate"] <= 0.02
        ),
        "ood_false_pass_le_0_05": test_gate["ood_false_pass_rate"] <= 0.05,
        "laya_coverage_ge_0_60": test_gate["laya_coverage"] >= 0.60,
    }

    report = {
        "generated_at": utc_now(),
        "model": display_path(args.model),
        "dataset": display_path(args.data_dir),
        "temperature": temperature,
        "thresholds": {
            "in_scope": selected["tau_in_scope"],
            "out_of_scope": selected["tau_out_of_scope"],
            "selected_only_on_calibration": True,
            "max_false_refusal": args.max_false_refusal,
            "max_false_pass": args.max_false_pass,
        },
        "calibration": {
            "laya_metrics": calibration_metrics,
            "gate_selection": {
                key: value
                for key, value in selected.items()
                if key != "decisions"
            },
            "gate_metrics": calibration_gate,
        },
        "internal_test": {
            "laya_metrics": test_metrics,
            "gate_metrics": test_gate,
            "target_checks": target_checks,
            "targets_met": all(target_checks.values()),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
