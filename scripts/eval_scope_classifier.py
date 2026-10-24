"""Evaluate the current LLM scope classifier and zero-shot Laya.

This script only evaluates the frozen ``eval/scope_v1/test.jsonl`` set. It does
not train on those examples or select a production threshold.

    python scripts/eval_scope_classifier.py --backend all --train-dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(line_buffering=True)

from hotspot.agent.runtime import (
    SCOPE_INSTRUCTIONS,
    ResearchAgentError,
    _classify_scope,
)
from hotspot.config import load_config

LABELS = ("in_scope", "out_of_scope")
MODEL_REVISION = "1720e3e3357cfe1e281542e223f8273b0890ca34"
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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            rows.append(row)
    return rows


def validate_dataset(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("scope dataset is empty")
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        missing = {"id", "question", "expected"} - set(row)
        if missing:
            raise ValueError(f"row {index} missing fields: {sorted(missing)}")
        item_id = str(row["id"])
        if item_id in seen:
            raise ValueError(f"duplicate id: {item_id}")
        seen.add(item_id)
        if row["expected"] not in LABELS:
            raise ValueError(f"{item_id}: invalid expected label {row['expected']!r}")
        if not str(row["question"]).strip():
            raise ValueError(f"{item_id}: empty question")


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 4)
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(value, 4)


def problem_group(row: dict[str, Any]) -> str:
    category = str(row.get("category") or "")
    if category == "ood":
        return "OOD"
    if category == "boundary":
        return "boundary"
    return str(row.get("kind") or category or "unknown")


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [row for row in rows if row.get("error") is None]
    labels = [row for row in evaluated if row.get("predicted") in LABELS]
    positives = [row for row in labels if row["expected"] == "in_scope"]
    negatives = [row for row in labels if row["expected"] == "out_of_scope"]
    tp = sum(row["predicted"] == "in_scope" for row in positives)
    fn = sum(row["predicted"] == "out_of_scope" for row in positives)
    tn = sum(row["predicted"] == "out_of_scope" for row in negatives)
    fp = sum(row["predicted"] == "in_scope" for row in negatives)

    return {
        "items": len(rows),
        "evaluated": len(evaluated),
        "errors": sum(row.get("error") is not None for row in rows),
        "completion_rate": round(len(evaluated) / max(1, len(rows)), 4),
        "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        "accuracy": round((tp + tn) / max(1, tp + tn + fp + fn), 4),
        "in_scope_recall": round(tp / max(1, tp + fn), 4),
        "in_scope_false_refusal_rate": round(fn / max(1, tp + fn), 4),
        "out_of_scope_recall": round(tn / max(1, tn + fp), 4),
        "ood_false_pass_rate": round(fp / max(1, tn + fp), 4),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = metrics(rows)
    evaluated = [row for row in rows if row.get("error") is None]
    latencies = [
        float(row["latency_s"])
        for row in evaluated
        if row.get("latency_s") is not None
    ]
    result.update({
        "latency_s": {
            "mean": round(statistics.fmean(latencies), 4) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": round(max(latencies), 4) if latencies else None,
        },
        "by_kind": {},
        "by_group": {},
        "by_language": {},
        "by_category": {},
    })
    for kind in sorted({str(row.get("kind") or "unknown") for row in rows}):
        group = [row for row in rows if str(row.get("kind") or "unknown") == kind]
        result["by_kind"][kind] = {
            "items": len(group),
            "correct": sum(row.get("correct") is True for row in group),
            "errors": sum(row.get("error") is not None for row in group),
        }
    for field, target in (
        ("group", "by_group"),
        ("language", "by_language"),
        ("category", "by_category"),
    ):
        keys = sorted({str(row.get(field) or "unknown") for row in rows})
        for key in keys:
            subset = [row for row in rows if str(row.get(field) or "unknown") == key]
            result[target][key] = metrics(subset)
    return result


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "gpu": [],
    }
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            info["gpu"] = [
                line.strip() for line in proc.stdout.splitlines() if line.strip()
            ]
    except OSError:
        pass
    return info


def base_report(backend: str, dataset: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_path = dataset.resolve()
    try:
        dataset_label = str(dataset_path.relative_to(ROOT))
    except ValueError:
        dataset_label = str(dataset_path)
    return {
        "generated_at": utc_now(),
        "git_commit": git_commit(),
        "backend": backend,
        "dataset": dataset_label,
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_items": len(rows),
        "labels": list(LABELS),
        "frozen_revision": MODEL_REVISION,
    }


def run_llm(
    rows: list[dict[str, Any]],
    *,
    config_path: Path,
    limit: int,
    concurrency: int,
    dataset_path: Path,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    timeout = float((cfg.get("agent") or {}).get("scope_timeout_s") or 30)
    items = rows[:limit]

    def evaluate_one(index: int, item: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        started = time.perf_counter()
        row = {
            "id": item["id"],
            "question": item["question"],
            "expected": item["expected"],
            "kind": item.get("kind", ""),
            "category": item.get("category", ""),
            "group": problem_group(item),
            "language": item.get("language", ""),
            "domain": item.get("domain", ""),
            "predicted": None,
            "correct": False,
            "error": None,
        }
        try:
            decision, usage = _classify_scope(
                str(item["question"]),
                cfg,
                timeout=timeout,
            )
            row.update(
                {
                    "predicted": (
                        "in_scope" if decision.in_scope else "out_of_scope"
                    ),
                    "domain": decision.domain,
                    "reason": decision.reason,
                    "usage": usage,
                }
            )
            row["correct"] = row["predicted"] == item["expected"]
        except ResearchAgentError as exc:
            row["error"] = {"code": exc.code, "message": str(exc)}
        except Exception as exc:  # noqa: BLE001
            row["error"] = {"code": "unexpected", "message": str(exc)}
        row["latency_s"] = round(time.perf_counter() - started, 4)
        status = row["predicted"] or (row["error"] or {}).get("code", "error")
        print(f"llm [{index}/{len(items)}] {item['id']}: {status}")
        return index, row

    results_by_index: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [
            pool.submit(evaluate_one, index, item)
            for index, item in enumerate(items, start=1)
        ]
        for future in as_completed(futures):
            index, row = future.result()
            results_by_index[index] = row
    results = [results_by_index[index] for index in sorted(results_by_index)]

    report = base_report("llm_default", dataset_path, rows)
    report.update(
        {
            "scope_timeout_s": timeout,
            "model": str(
                (cfg.get("llm") or {}).get("model_flash")
                or (cfg.get("llm") or {}).get("model")
                or ""
            ),
            "prompt_version": "runtime-scope-instructions-2026-10-23",
            "prompt_sha256": sha256_text(SCOPE_INSTRUCTIONS),
            "llm_concurrency": max(1, concurrency),
            "summary": summarize(results),
            "rows": results,
        }
    )
    return report


def run_laya(
    rows: list[dict[str, Any]],
    *,
    model_dir: Path,
    repeat: int,
    dataset_path: Path,
) -> dict[str, Any]:
    import laya

    load_started = time.perf_counter()
    agent = laya.load(str(model_dir))
    load_s = round(time.perf_counter() - load_started, 4)

    warm_started = time.perf_counter()
    agent.predict({"question": rows[0]["question"]}, SCOPE_QUESTION)
    warmup_s = round(time.perf_counter() - warm_started, 4)

    first_pass: list[dict[str, Any]] = []
    latency_runs: list[float] = []
    for pass_index in range(max(1, repeat)):
        for row_index, item in enumerate(rows, start=1):
            started = time.perf_counter()
            result = agent.predict(
                {"question": item["question"]},
                SCOPE_QUESTION,
            )
            latency_s = round(time.perf_counter() - started, 4)
            latency_runs.append(latency_s)
            answer = result["answers"]["scope"]
            predicted = str(answer["choice"])
            row = {
                "id": item["id"],
                "question": item["question"],
                "expected": item["expected"],
                "kind": item.get("kind", ""),
                "category": item.get("category", ""),
                "group": problem_group(item),
                "language": item.get("language", ""),
                "domain": item.get("domain", ""),
                "predicted": predicted,
                "correct": predicted == item["expected"],
                "error": None,
                "latency_s": latency_s,
                "probabilities": answer.get("probabilities", {}),
                "confidence": answer.get("confidence"),
                "answer_confidence": answer.get("answer_confidence"),
                "usage": result.get("usage", {}),
            }
            if pass_index == 0:
                first_pass.append(row)
            if pass_index == 0:
                print(f"laya [{row_index}/{len(rows)}] {item['id']}: {predicted}")

    report = base_report("laya_zero_shot", dataset_path, rows)
    model_config_path = (
        model_dir / "config.json"
        if (model_dir / "config.json").exists()
        else model_dir / "rl_agent_config.json"
    )
    report.update(
        {
            "model_dir": str(model_dir),
            "prompt_version": "scope-choice-v1",
            "prompt_sha256": sha256_text(
                json.dumps(SCOPE_QUESTION, ensure_ascii=False, sort_keys=True)
            ),
            "model_config": json.loads(
                model_config_path.read_text(encoding="utf-8")
            ),
            "load_s": load_s,
            "warmup_s": warmup_s,
            "timed_inference_calls": len(latency_runs),
            "summary": summarize(first_pass),
            "warm_summary": {
                "latency_s": {
                    "mean": round(statistics.fmean(latency_runs), 4),
                    "p50": percentile(latency_runs, 0.50),
                    "p95": percentile(latency_runs, 0.95),
                    "max": round(max(latency_runs), 4),
                }
            },
            "rows": first_pass,
        }
    )
    return report


def run_train_dry_run(
    *,
    model_dir: Path,
    smoke_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "laya.train_cli",
        "--data",
        str(smoke_path),
        "--base",
        str(model_dir),
        "--out",
        str(output_dir),
        "--freeze-encoder",
        "--loss",
        "soft-ce",
        "--device",
        "cpu",
        "--max-len",
        "256",
        "--head-max-len",
        "256",
        "--seed",
        "0",
        "--dry-run",
    ]
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )
    return {
        "command": command,
        "returncode": proc.returncode,
        "elapsed_s": round(time.perf_counter() - started, 4),
        "stdout": proc.stdout[-12000:],
        "stderr": proc.stderr[-12000:],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "eval" / "scope_v1" / "test.jsonl",
    )
    parser.add_argument(
        "--backend",
        choices=("none", "llm", "laya", "all"),
        default="all",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--model-dir", type=Path, default=ROOT / ".models" / "jev")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "eval" / "scope_v1" / "results",
    )
    parser.add_argument("--llm-limit", type=int, default=16)
    parser.add_argument("--llm-concurrency", type=int, default=4)
    parser.add_argument("--laya-repeat", type=int, default=3)
    parser.add_argument("--train-dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.dataset)
    validate_dataset(rows)
    reports: list[dict[str, Any]] = []

    if args.backend in {"llm", "all"}:
        llm_report = run_llm(
            rows,
            config_path=args.config,
            limit=min(max(1, args.llm_limit), len(rows)),
            concurrency=max(1, args.llm_concurrency),
            dataset_path=args.dataset,
        )
        write_json(args.out_dir / "llm_default.json", llm_report)
        reports.append(llm_report)

    if args.backend in {"laya", "all"}:
        laya_report = run_laya(
            rows,
            model_dir=args.model_dir,
            repeat=max(1, args.laya_repeat),
            dataset_path=args.dataset,
        )
        write_json(args.out_dir / "laya_zero_shot.json", laya_report)
        reports.append(laya_report)

    result_paths = {
        "llm_default": args.out_dir / "llm_default.json",
        "laya_zero_shot": args.out_dir / "laya_zero_shot.json",
    }
    combined_reports: dict[str, dict[str, Any]] = {}
    for name, path in result_paths.items():
        if path.exists():
            combined_reports[name] = json.loads(path.read_text(encoding="utf-8"))
    for report in reports:
        combined_reports[report["backend"]] = report

    previous_compute_path = args.out_dir / "compute_probe.json"
    previous_compute: dict[str, Any] = {}
    if previous_compute_path.exists():
        previous_compute = json.loads(
            previous_compute_path.read_text(encoding="utf-8")
        )
    current_laya = next(
        (
            {
                "load_s": report["load_s"],
                "warmup_s": report["warmup_s"],
                "timed_inference_calls": report["timed_inference_calls"],
                "warm_summary": report["warm_summary"],
            }
            for report in reports
            if report["backend"] == "laya_zero_shot"
        ),
        None,
    )
    if current_laya is None and "laya_zero_shot" in combined_reports:
        stored_laya = combined_reports["laya_zero_shot"]
        current_laya = {
            "load_s": stored_laya.get("load_s"),
            "warmup_s": stored_laya.get("warmup_s"),
            "timed_inference_calls": stored_laya.get("timed_inference_calls"),
            "warm_summary": stored_laya.get("warm_summary"),
        }
    compute: dict[str, Any] = {
        "generated_at": utc_now(),
        "git_commit": git_commit(),
        "hardware": hardware_info(),
        "laya": current_laya or previous_compute.get("laya"),
    }
    if args.train_dry_run:
        compute["laya_train_dry_run"] = run_train_dry_run(
            model_dir=args.model_dir,
            smoke_path=ROOT / "eval" / "scope_v1" / "smoke.jsonl",
            output_dir=ROOT / ".models" / "jev" / "ft" / "scope-v1-dryrun",
        )
    elif previous_compute.get("laya_train_dry_run"):
        compute["laya_train_dry_run"] = previous_compute["laya_train_dry_run"]
    write_json(args.out_dir / "compute_probe.json", compute)

    summary = {
        "generated_at": utc_now(),
        "dataset_items": len(rows),
        "reports": {
            name: report["summary"]
            for name, report in combined_reports.items()
        },
        "compute_probe": display_path(args.out_dir / "compute_probe.json"),
    }
    write_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
