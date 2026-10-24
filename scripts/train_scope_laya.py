"""Train the Laya scope head with explicit train/calibration files.

Unlike ``laya.train_cli``, this entrypoint does not carve calibration rows out of
the training file. It trains only on ``train.jsonl`` and fits temperature on
``calibration.jsonl``. The internal test set is intentionally not evaluated here.

    python scripts/train_scope_laya.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import laya.train as laya_train

from laya.calibrate import fit_abstention_thresholds
from laya.common import uses_parallel_layout
from laya.agent import _fix_tokenizer_config
from laya.train import (
    TrainConfig,
    calibration_records,
    evaluate_records,
    fit_temperature_map,
    items_from_rows,
    load_checkpoint,
    optimizer_updates,
    prior_collapse_message,
    read_data,
    resolve_checkpoint_dir,
    resolve_device,
    save_checkpoint,
    train_model,
)
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(line_buffering=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def configure_trainable_layers(
    model: Any,
    *,
    freeze_encoder: bool,
    unfreeze_last_layers: int,
) -> dict[str, Any]:
    if unfreeze_last_layers < 0:
        raise ValueError("unfreeze-last-layers must be >= 0")
    if freeze_encoder and unfreeze_last_layers:
        raise ValueError("freeze-encoder and unfreeze-last-layers are mutually exclusive")
    if freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        return {
            "policy": "frozen",
            "unfrozen_layers": 0,
            "trainable_encoder_parameters": 0,
        }
    if unfreeze_last_layers == 0:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(True)
        return {
            "policy": "full_encoder",
            "unfrozen_layers": "all",
            "trainable_encoder_parameters": sum(
                parameter.numel()
                for parameter in model.encoder.parameters()
                if parameter.requires_grad
            ),
        }

    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    candidates = (
        ("encoder", "layer"),
        ("encoder", "layers"),
        ("layers",),
        ("model", "layers"),
        ("model", "encoder", "layer"),
    )
    layers = None
    selected_path = ""
    parent = model.encoder
    for path in candidates:
        candidate_parent = parent
        try:
            for name in path:
                candidate_parent = getattr(candidate_parent, name)
        except AttributeError:
            continue
        if isinstance(candidate_parent, torch.nn.ModuleList):
            layers = candidate_parent
            selected_path = "encoder." + ".".join(path)
            break
    if layers is None:
        raise RuntimeError("could not locate encoder transformer layers")
    if unfreeze_last_layers > len(layers):
        raise ValueError(
            f"cannot unfreeze {unfreeze_last_layers} layers from {len(layers)}"
        )
    for layer in layers[-unfreeze_last_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    return {
        "policy": f"last_{unfreeze_last_layers}_layers",
        "layer_path": selected_path,
        "total_layers": len(layers),
        "unfrozen_layers": unfreeze_last_layers,
        "trainable_encoder_parameters": sum(
            parameter.numel()
            for parameter in model.encoder.parameters()
            if parameter.requires_grad
        ),
    }


def asymmetric_soft_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    ood_weight: float,
    ood_index: int,
) -> torch.Tensor:
    logp = torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
    per_item = -(target * logp * mask).sum(-1)
    weights = 1.0 + (float(ood_weight) - 1.0) * target[:, ood_index]
    return (per_item * weights).mean()


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    values = values - values.max()
    exp = np.exp(values)
    return exp / exp.sum()


def classification_metrics(
    records: list[tuple[int, Any, Any, int]],
    items: list[dict[str, Any]],
    *,
    temperature: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for record, item in zip(records, items):
        _qtype, logits_raw, _target, k = record
        logits = np.asarray(logits_raw, dtype=float)[:k]
        probabilities = softmax(logits, temperature)
        options = list(item["q"]["crit"].keys())
        in_index = options.index("in_scope")
        predicted = "in_scope" if int(probabilities.argmax()) == in_index else "out_of_scope"
        source_expected = item["_source"]["expected"]
        expected = (
            str(source_expected.get("scope", ""))
            if isinstance(source_expected, dict)
            else str(source_expected)
        )
        row = {
            "expected": expected,
            "category": str(item["_source"].get("category", "")),
            "kind": str(item["_source"].get("kind", "")),
            "language": str(item["_source"].get("language", "")),
            "domain": str(item["_source"].get("domain", "")),
            "predicted": predicted,
            "p_in_scope": round(float(probabilities[in_index]), 6),
            "p_out_of_scope": round(float(1.0 - probabilities[in_index]), 6),
        }
        rows.append(row)

    positives = [row for row in rows if row["expected"] == "in_scope"]
    negatives = [row for row in rows if row["expected"] == "out_of_scope"]
    tp = sum(row["predicted"] == "in_scope" for row in positives)
    fn = sum(row["predicted"] == "out_of_scope" for row in positives)
    tn = sum(row["predicted"] == "out_of_scope" for row in negatives)
    fp = sum(row["predicted"] == "in_scope" for row in negatives)
    return {
        "items": len(rows),
        "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        "accuracy": round((tp + tn) / max(1, len(rows)), 4),
        "in_scope_recall": round(tp / max(1, tp + fn), 4),
        "in_scope_false_refusal_rate": round(fn / max(1, tp + fn), 4),
        "out_of_scope_recall": round(tn / max(1, tn + fp), 4),
        "ood_false_pass_rate": round(fp / max(1, tn + fp), 4),
    }, rows


def attach_sources(
    items: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> None:
    if len(items) != len(source_rows):
        raise ValueError(
            f"preprocessing produced {len(items)} items for {len(source_rows)} rows"
        )
    for item, source in zip(items, source_rows):
        item["_source"] = source


def evaluate_split(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
    max_len: int,
    head_max_len: int,
    parallel: bool,
    temperature: list[float],
    temperature_by_options: dict[str, float] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[int, Any, Any, int]]]:
    items, skipped = items_from_rows(tokenizer, rows, max_len, head_max_len)
    attach_sources(items, rows)
    records = calibration_records(
        model,
        tokenizer,
        items,
        device,
        max_len,
        head_max_len,
        parallel=parallel,
    )
    report = evaluate_records(records, temperature, temperature_by_options)
    choice_temperature = float(temperature[0] if temperature else 1.0)
    classification, predictions = classification_metrics(
        records,
        items,
        temperature=choice_temperature,
    )
    return {
        **report,
        "classification": classification,
        "skipped": skipped,
    }, predictions, records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "eval" / "scope_v1" / "dataset_v1_2",
    )
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--base", type=Path, default=ROOT / ".models" / "jev")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / ".models" / "jev" / "ft" / "scope-v12-head-only-seed0",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=2.5e-5)
    parser.add_argument("--loss", choices=("soft-ce", "rlcd"), default="soft-ce")
    parser.add_argument("--ood-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--freeze-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--unfreeze-last-layers", type=int, default=0)
    parser.add_argument(
        "--shuffle-options",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--head-max-len", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-error", type=float, default=0.10)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--strict-manifest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out.exists() and not args.dry_run:
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {args.out}")
    train_path = args.train_file or args.data_dir / "laya_train.jsonl"
    calibration_path = (
        args.calibration_file or args.data_dir / "laya_calibration.jsonl"
    )
    if not train_path.exists() or not calibration_path.exists():
        raise FileNotFoundError("missing explicit train/calibration files")

    started_at = utc_now()
    started = time.perf_counter()
    train_rows = read_data(str(train_path))
    calibration_rows = read_data(str(calibration_path))
    dataset_versions = {
        str(row.get("dataset_version"))
        for row in train_rows + calibration_rows
        if row.get("dataset_version")
    }
    dataset_version = (
        next(iter(dataset_versions)) if len(dataset_versions) == 1 else "unknown"
    )
    manifest_path = args.dataset_manifest or args.data_dir / "manifest.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_counts = (manifest.get("counts") or {}).get("split") or {}
        if args.strict_manifest:
            if len(train_rows) != int(split_counts.get("train", -1)):
                raise RuntimeError(
                    f"train rows {len(train_rows)} != manifest {split_counts.get('train')}"
                )
            if len(calibration_rows) != int(split_counts.get("calibration", -1)):
                raise RuntimeError(
                    f"calibration rows {len(calibration_rows)} != "
                    f"manifest {split_counts.get('calibration')}"
                )
    elif args.strict_manifest and args.train_file is None:
        raise FileNotFoundError(f"missing dataset manifest: {manifest_path}")

    if args.dry_run:
        checkpoint_dir = resolve_checkpoint_dir(str(args.base))
        _fix_tokenizer_config(checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(checkpoint_dir, "tokenizer")
        )
        train_items, train_skipped = items_from_rows(
            tokenizer,
            train_rows,
            args.max_len,
            args.head_max_len,
        )
        calibration_items, calibration_skipped = items_from_rows(
            tokenizer,
            calibration_rows,
            args.max_len,
            args.head_max_len,
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "dataset_version": dataset_version,
                    "train_rows": len(train_rows),
                    "train_items": len(train_items),
                    "train_skipped": train_skipped,
                    "calibration_rows": len(calibration_rows),
                    "calibration_items": len(calibration_items),
                    "calibration_skipped": calibration_skipped,
                    "optimizer_updates": optimizer_updates(
                        len(train_items),
                        args.micro_batch,
                        args.grad_accum,
                        args.epochs,
                    ),
                    "training_args": {
                        "base": str(args.base),
                        "out": str(args.out),
                        "epochs": args.epochs,
                        "micro_batch": args.micro_batch,
                        "grad_accum": args.grad_accum,
                        "loss": args.loss,
                        "ood_loss_weight": args.ood_loss_weight,
                        "freeze_encoder": args.freeze_encoder,
                        "unfreeze_last_layers": args.unfreeze_last_layers,
                        "shuffle_options": args.shuffle_options,
                        "seed": args.seed,
                        "max_len": args.max_len,
                        "head_max_len": args.head_max_len,
                        "device": args.device,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    device = resolve_device(args.device)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f"device={device} torch_threads={torch.get_num_threads()}", flush=True)

    model, tokenizer, base_cfg = load_checkpoint(str(args.base))
    model.to(device)
    layer_policy = configure_trainable_layers(
        model,
        freeze_encoder=args.freeze_encoder,
        unfreeze_last_layers=args.unfreeze_last_layers,
    )
    train_items, train_skipped = items_from_rows(
        tokenizer,
        train_rows,
        args.max_len,
        args.head_max_len,
    )
    calibration_items, calibration_skipped = items_from_rows(
        tokenizer,
        calibration_rows,
        args.max_len,
        args.head_max_len,
    )
    if train_skipped or calibration_skipped:
        raise RuntimeError(
            f"preprocessing skipped rows: train={train_skipped}, calibration={calibration_skipped}"
        )
    if not train_items or not calibration_items:
        raise RuntimeError("train and calibration must both produce items")
    attach_sources(train_items, train_rows)
    attach_sources(calibration_items, calibration_rows)

    parallel = uses_parallel_layout(base_cfg)
    base_temperature = list(base_cfg.get("temperature") or [1.0, 1.0, 1.0])
    base_temp_by_options = base_cfg.get("temperature_by_options") or {}

    base_calibration_records = calibration_records(
        model,
        tokenizer,
        calibration_items,
        device,
        args.max_len,
        args.head_max_len,
        parallel=parallel,
    )
    base_calibration_eval = evaluate_records(
        base_calibration_records,
        base_temperature,
        base_temp_by_options,
    )
    base_classification, base_predictions = classification_metrics(
        base_calibration_records,
        calibration_items,
        temperature=float(base_temperature[0]),
    )
    print(
        f"base calibration: acc={base_classification['accuracy']} "
        f"in_recall={base_classification['in_scope_recall']} "
        f"ood_recall={base_classification['out_of_scope_recall']}",
        flush=True,
    )

    freeze_encoder = args.freeze_encoder and args.unfreeze_last_layers == 0
    config = TrainConfig(
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        loss=args.loss,
        shuffle_options=("choice",) if args.shuffle_options else (),
        freeze_encoder=freeze_encoder,
        seed=args.seed,
        max_len=args.max_len,
        head_max_len=args.head_max_len,
        target_error=args.target_error,
        log_every=args.log_every,
    )
    config.validate()
    if args.ood_loss_weight < 1.0:
        raise ValueError("ood-loss-weight must be >= 1.0")
    loss_policy = {
        "base_loss": args.loss,
        "ood_weight": args.ood_loss_weight,
        "in_scope_weight": 1.0,
        "asymmetric": args.ood_loss_weight > 1.0,
    }
    if args.ood_loss_weight > 1.0:
        if args.loss != "soft-ce":
            raise ValueError("ood-loss-weight currently requires soft-ce")
        ood_indices = {
            list(item["q"]["crit"].keys()).index("out_of_scope")
            for item in train_items
        }
        if len(ood_indices) != 1:
            raise RuntimeError(f"inconsistent OOD option index: {ood_indices}")
        ood_index = next(iter(ood_indices))
        laya_train.soft_ce_loss = (
            lambda logits, target, mask: asymmetric_soft_ce_loss(
                logits,
                target,
                mask,
                ood_weight=args.ood_loss_weight,
                ood_index=ood_index,
            )
        )
        loss_policy["ood_index"] = ood_index
        loss_policy["implementation"] = "ood_asymmetric_soft_ce_v1"
        print(f"loss policy: {json.dumps(loss_policy, ensure_ascii=False)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=False)

    epoch_rows: list[dict[str, Any]] = []

    def on_epoch_end(epoch: int, mean_loss: float) -> None:
        elapsed = round(time.perf_counter() - started, 2)
        epoch_rows.append(
            {"epoch": epoch + 1, "mean_loss": round(float(mean_loss), 6), "elapsed_s": elapsed}
        )
        save_checkpoint(
            model,
            tokenizer,
            dict(base_cfg, max_len=args.max_len, head_max_len=args.head_max_len),
            str(args.out / "checkpoint_latest"),
        )
        write_jsonl(args.out / "epochs.jsonl", epoch_rows)
        print(f"epoch {epoch + 1} done loss={mean_loss:.6f} elapsed={elapsed}s", flush=True)

    history = train_model(
        model,
        tokenizer,
        train_items,
        config,
        device,
        args.max_len,
        args.head_max_len,
        on_epoch_end=on_epoch_end,
        parallel=parallel,
    )

    calibration_after_records = calibration_records(
        model,
        tokenizer,
        calibration_items,
        device,
        args.max_len,
        args.head_max_len,
        parallel=parallel,
    )
    fitted = fit_temperature_map(calibration_after_records)
    calibrated_eval = evaluate_records(
        calibration_after_records,
        fitted["temperature"],
        fitted["temperature_by_options"],
    )
    calibrated_classification, calibrated_predictions = classification_metrics(
        calibration_after_records,
        calibration_items,
        temperature=float(fitted["temperature"][0]),
    )
    collapse = prior_collapse_message(calibration_after_records)
    try:
        abstention = fit_abstention_thresholds(
            calibration_after_records,
            temperature=fitted["temperature"],
            temperature_by_options=fitted["temperature_by_options"],
            target_error=args.target_error,
            min_bucket_n=10,
        )
    except Exception as exc:  # noqa: BLE001
        abstention = {"error": str(exc)}

    elapsed_s = round(time.perf_counter() - started, 2)
    report = {
        "status": "ok" if collapse is None else "collapsed",
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_s": elapsed_s,
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "dataset": {
            "version": dataset_version,
            "versions_in_files": sorted(dataset_versions),
            "manifest": display_path(manifest_path) if manifest_path.exists() else None,
            "manifest_sha256": (
                sha256_file(manifest_path) if manifest_path.exists() else None
            ),
        },
        "layer_policy": layer_policy,
        "loss_policy": loss_policy,
        "train": {
            "path": display_path(train_path),
            "sha256": sha256_file(train_path),
            "items": len(train_items),
            "epoch_loss": history,
        },
        "calibration": {
            "path": display_path(calibration_path),
            "sha256": sha256_file(calibration_path),
            "items": len(calibration_items),
        },
        "base": {
            "metrics": base_calibration_eval,
            "classification": base_classification,
        },
        "trained": {
            "metrics": calibrated_eval,
            "classification": calibrated_classification,
            "temperature": fitted["temperature"],
            "temperature_by_options": fitted["temperature_by_options"],
            "n_by_bucket": fitted["n_by_bucket"],
            "abstention_thresholds": abstention,
            "collapse_warning": collapse,
        },
        "training_config": asdict(config),
        "internal_test_used": False,
        "checkpoint": display_path(args.out),
    }
    write_jsonl(args.out / "calibration_base_predictions.jsonl", base_predictions)
    write_jsonl(args.out / "calibration_predictions.jsonl", calibrated_predictions)
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
        "scope_training": asdict(config),
        "scope_layer_policy": layer_policy,
        "scope_loss_policy": loss_policy,
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
    print(
        json.dumps(
            {
                "status": report["status"],
                "elapsed_s": elapsed_s,
                "base": base_classification,
                "trained": calibrated_classification,
                "temperature": fitted["temperature"],
                "collapse_warning": collapse,
                "checkpoint": str(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if collapse is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
