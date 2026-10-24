"""Finalize an already-trained Laya checkpoint after an interrupted report step.

This utility recomputes base and trained calibration, fits temperature, saves the
calibrated checkpoint, and writes the same pilot artifacts as train_scope_laya.py.
It does not train additional epochs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from laya.calibrate import fit_abstention_thresholds
from laya.common import uses_parallel_layout
from laya.train import (
    TrainConfig,
    calibration_records,
    evaluate_records,
    fit_temperature_map,
    items_from_rows,
    load_checkpoint,
    prior_collapse_message,
    read_data,
    resolve_device,
    save_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(line_buffering=True)

from train_scope_laya import (
    attach_sources,
    classification_metrics,
    display_path,
    sha256_file,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="The trained checkpoint_latest to finalize.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Final calibrated checkpoint directory.",
    )
    parser.add_argument("--base", type=Path, default=ROOT / ".models" / "jev")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--calibration-file", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--head-max-len", type=int, default=256)
    parser.add_argument("--target-error", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=2.5e-5)
    parser.add_argument("--loss", choices=("soft-ce", "rlcd"), default="soft-ce")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    device = resolve_device(args.device)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    train_rows = read_data(str(args.train_file))
    calibration_rows = read_data(str(args.calibration_file))
    versions = {
        str(row.get("dataset_version"))
        for row in train_rows + calibration_rows
        if row.get("dataset_version")
    }
    dataset_version = next(iter(versions)) if len(versions) == 1 else "unknown"

    started = time.perf_counter()
    base_model, base_tokenizer, base_cfg = load_checkpoint(str(args.base))
    base_model.to(device)
    base_items, base_skipped = items_from_rows(
        base_tokenizer,
        calibration_rows,
        args.max_len,
        args.head_max_len,
    )
    if base_skipped:
        raise RuntimeError(f"base preprocessing skipped: {base_skipped}")
    attach_sources(base_items, calibration_rows)
    base_parallel = uses_parallel_layout(base_cfg)
    base_temperature = list(base_cfg.get("temperature") or [1.0, 1.0, 1.0])
    base_temp_by_options = base_cfg.get("temperature_by_options") or {}
    base_records = calibration_records(
        base_model,
        base_tokenizer,
        base_items,
        device,
        args.max_len,
        args.head_max_len,
        parallel=base_parallel,
    )
    base_eval = evaluate_records(
        base_records,
        base_temperature,
        base_temp_by_options,
    )
    base_classification, base_predictions = classification_metrics(
        base_records,
        base_items,
        temperature=float(base_temperature[0]),
    )
    del base_model
    gc.collect()

    model, tokenizer, base_cfg = load_checkpoint(str(args.checkpoint))
    model.to(device)
    calibration_items, calibration_skipped = items_from_rows(
        tokenizer,
        calibration_rows,
        args.max_len,
        args.head_max_len,
    )
    if calibration_skipped:
        raise RuntimeError(f"checkpoint preprocessing skipped: {calibration_skipped}")
    attach_sources(calibration_items, calibration_rows)
    parallel = uses_parallel_layout(base_cfg)
    records = calibration_records(
        model,
        tokenizer,
        calibration_items,
        device,
        args.max_len,
        args.head_max_len,
        parallel=parallel,
    )
    fitted = fit_temperature_map(records)
    trained_eval = evaluate_records(
        records,
        fitted["temperature"],
        fitted["temperature_by_options"],
    )
    trained_classification, trained_predictions = classification_metrics(
        records,
        calibration_items,
        temperature=float(fitted["temperature"][0]),
    )
    collapse = prior_collapse_message(records)
    try:
        abstention = fit_abstention_thresholds(
            records,
            temperature=fitted["temperature"],
            temperature_by_options=fitted["temperature_by_options"],
            target_error=args.target_error,
            min_bucket_n=10,
        )
    except Exception as exc:  # noqa: BLE001
        abstention = {"error": str(exc)}

    train_config = TrainConfig(
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        loss=args.loss,
        shuffle_options=("choice",),
        freeze_encoder=True,
        seed=args.seed,
        max_len=args.max_len,
        head_max_len=args.head_max_len,
        target_error=args.target_error,
    )
    epochs_path = args.out / "epochs.jsonl"
    epoch_rows = read_jsonl(epochs_path) if epochs_path.exists() else []
    elapsed_s = round(time.perf_counter() - started, 2)
    report = {
        "status": "ok" if collapse is None else "collapsed",
        "generated_at": utc_now(),
        "recovered_from_checkpoint": display_path(args.checkpoint),
        "elapsed_finalize_s": elapsed_s,
        "device": str(device),
        "dataset": {
            "version": dataset_version,
            "manifest": display_path(args.dataset_manifest),
            "manifest_sha256": sha256_file(args.dataset_manifest),
        },
        "layer_policy": {
            "policy": "frozen",
            "unfrozen_layers": 0,
        },
        "train": {
            "path": display_path(args.train_file),
            "sha256": sha256_file(args.train_file),
            "items": len(train_rows),
            "epoch_loss": [row["mean_loss"] for row in epoch_rows],
            "epochs": epoch_rows,
        },
        "calibration": {
            "path": display_path(args.calibration_file),
            "sha256": sha256_file(args.calibration_file),
            "items": len(calibration_items),
        },
        "base": {
            "metrics": base_eval,
            "classification": base_classification,
        },
        "trained": {
            "metrics": trained_eval,
            "classification": trained_classification,
            "temperature": fitted["temperature"],
            "temperature_by_options": fitted["temperature_by_options"],
            "n_by_bucket": fitted["n_by_bucket"],
            "abstention_thresholds": abstention,
            "collapse_warning": collapse,
        },
        "training_config": asdict(train_config),
        "internal_test_used": False,
        "checkpoint": display_path(args.out),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "calibration_base_predictions.jsonl", base_predictions)
    write_jsonl(args.out / "calibration_predictions.jsonl", trained_predictions)
    (args.out / "pilot_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    out_cfg = dict(
        base_cfg,
        max_len=args.max_len,
        head_max_len=args.head_max_len,
        fine_tuned=True,
        temperature=fitted["temperature"],
    )
    out_cfg.pop("temperature_by_options", None)
    if fitted["temperature_by_options"]:
        out_cfg["temperature_by_options"] = fitted["temperature_by_options"]
    out_cfg["training"] = {
        "scope_training": asdict(train_config),
        "scope_layer_policy": report["layer_policy"],
        "scope_dataset": report["dataset"],
        "scope_pilot_report": report,
    }
    save_checkpoint(model, tokenizer, out_cfg, str(args.out))
    questions = {}
    for row in train_rows:
        questions.update(row.get("questions") or {})
    for target in (args.out, args.out / "checkpoint_latest"):
        if target.exists():
            (target / "questions.json").write_text(
                json.dumps(questions, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (target / "train_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (target / "rl_agent_config.json").write_text(
                json.dumps(out_cfg, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    print(
        json.dumps(
            {
                "status": report["status"],
                "base": base_classification,
                "trained": trained_classification,
                "temperature": fitted["temperature"],
                "collapse_warning": collapse,
                "checkpoint": display_path(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if collapse is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
