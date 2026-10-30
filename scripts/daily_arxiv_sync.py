"""Daily arXiv fetch, week binding, incremental indexing, and harness export."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hotspot.config import load_config  # noqa: E402
from hotspot.index.lance_store import build_lance_index  # noqa: E402
from hotspot.weekly import (  # noqa: E402
    DEFAULT_MAX_RETENTION_WEEKS,
    build_week,
    iso_week_id,
    prune_weekly_corpus,
    retention_cutoff,
    weekly_paper_ids,
)
from hotspot.weekly.vectors import (  # noqa: E402
    embed_missing_arxiv,
    prune_arxiv_vectors,
)

PROGRAM_ROOT = ROOT / "hotspot" / "vendor" / "arxiv_weekly"
DAILY_SCRIPT = PROGRAM_ROOT / "scripts" / "daily_arxiv_fetch.py"


def default_data_root() -> Path:
    return Path(os.environ.get(
        "ARXIV_WEEKLY_DATA",
        str(ROOT / "data" / "arxiv-weekly"),
    ))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _acquire_lock(state_dir: Path) -> Path:
    lock_dir = state_dir / "daily.lock.d"
    try:
        lock_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        age = time.time() - lock_dir.stat().st_mtime
        if age < 8 * 60 * 60:
            raise RuntimeError(f"another weekly sync is running: {lock_dir}")
        shutil.rmtree(lock_dir)
        lock_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(lock_dir / "owner.json", {
        "pid": os.getpid(),
        "started_at": int(time.time()),
    })
    return lock_dir


def _release_lock(lock_dir: Path) -> None:
    shutil.rmtree(lock_dir, ignore_errors=True)


def _last_json_line(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _done_summary(data_root: Path, day: str) -> dict[str, Any]:
    path = data_root / iso_week_id(day) / f"{day}.done"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _done_is_complete(summary: dict[str, Any]) -> bool:
    args = summary.get("args") or {}
    return (
        int(summary.get("exit_code") or 0) == 0
        and not (summary.get("failed_feeds") or [])
        and not (summary.get("feed_build_stale") or [])
        and not (summary.get("feed_stalled") or [])
        and not int(args.get("limit") or 0)
        and not int(args.get("license_limit") or 0)
    )


def _resolve_retention_weeks(
    cfg: dict[str, Any],
    override: int | None,
) -> int:
    raw = override
    if raw is None:
        raw = (cfg.get("arxiv_weekly") or {}).get(
            "max_retention_weeks",
            DEFAULT_MAX_RETENTION_WEEKS,
        )
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValueError("max_retention_weeks must be a positive integer")
    # Reuse the weekly helper's upper-bound validation.
    retention_cutoff(date.today(), raw)
    return raw


def _run_daily(
    data_root: Path,
    cache_dir: Path,
    *,
    day: str | None,
    limit: int,
    license_limit: int,
    refresh: bool,
) -> tuple[int, str, str]:
    env = dict(os.environ)
    env["ARXIV_WEEKLY_DATA"] = str(data_root)
    env["ARXIV_WEEKLY_CACHE"] = str(cache_dir)
    cmd = [sys.executable, str(DAILY_SCRIPT)]
    if day:
        cmd.extend(["--date", day])
    if limit:
        cmd.extend(["--limit", str(limit)])
    if license_limit:
        cmd.extend(["--license-limit", str(license_limit)])
    if refresh:
        cmd.append("--refresh")
    proc = subprocess.run(
        cmd,
        cwd=PROGRAM_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=8 * 60 * 60,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _preflight(
    data_root: Path,
    cache_dir: Path,
    *,
    max_retention_weeks: int,
    day: str,
) -> dict[str, Any]:
    if not DAILY_SCRIPT.is_file():
        raise FileNotFoundError(f"missing vendored daily script: {DAILY_SCRIPT}")
    proc = subprocess.run(
        [sys.executable, str(DAILY_SCRIPT), "--help"],
        cwd=PROGRAM_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        env={
            **os.environ,
            "ARXIV_WEEKLY_DATA": str(data_root),
            "ARXIV_WEEKLY_CACHE": str(cache_dir),
        },
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "daily script preflight failed")
    return {
        "ok": True,
        "program_root": str(PROGRAM_ROOT),
        "data_root": str(data_root),
        "cache_dir": str(cache_dir),
        "papers_file": str(data_root / "papers.jsonl"),
        "max_retention_weeks": max_retention_weeks,
        "retention_cutoff_week": retention_cutoff(
            day,
            max_retention_weeks,
        ),
    }


def _previous_run_complete(state_dir: Path, day: str) -> bool:
    path = state_dir / "last_run.json"
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return value.get("date") == day and value.get("status") == "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--license-limit", type=int, default=0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seal-week", action="store_true")
    parser.add_argument(
        "--week",
        default=None,
        help="with --seal-week, publish this ISO week instead of today's",
    )
    parser.add_argument("--no-bind", action="store_true")
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument("--embed-limit", type=int, default=0)
    parser.add_argument("--no-index", action="store_true")
    parser.add_argument(
        "--max-retention-weeks",
        type=int,
        default=None,
        help="override arxiv_weekly.max_retention_weeks (includes current week)",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="disable retention cleanup for this run",
    )
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.week and not args.seal_week:
        parser.error("--week requires --seal-week")
    if args.seal_week and args.no_bind:
        parser.error("--seal-week cannot be combined with --no-bind")

    day = args.date or date.today().isoformat()
    week_id = args.week or iso_week_id(day)
    cfg = load_config()
    try:
        max_retention_weeks = _resolve_retention_weeks(
            cfg,
            args.max_retention_weeks,
        )
    except ValueError as exc:
        parser.error(str(exc))
    data_root = default_data_root()
    cache_dir = Path(
        os.environ.get("ARXIV_WEEKLY_CACHE", str(data_root / "cache"))
    )
    state_dir = data_root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    if args.preflight:
        print(json.dumps(
            _preflight(
                data_root,
                cache_dir,
                max_retention_weeks=max_retention_weeks,
                day=day,
            ),
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    lock_dir = _acquire_lock(state_dir)
    started = time.time()
    result: dict[str, Any] = {
        "date": day,
        "week_id": week_id,
        "data_root": str(data_root),
        "max_retention_weeks": max_retention_weeks,
        "status": "failed",
    }
    log_path = state_dir / "logs" / f"{day}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.seal_week:
            result["mode"] = "seal"
            retention: dict[str, Any] | None = None
            if not args.no_prune:
                retention = prune_weekly_corpus(
                    data_root,
                    max_retention_weeks=max_retention_weeks,
                    as_of=day,
                )
                result["retention"] = retention
                if retention["removed_papers"]:
                    result["vector_prune"] = prune_arxiv_vectors(
                        cfg,
                        active_ids=weekly_paper_ids(data_root),
                    )
            result["week"] = build_week(
                data_root,
                week_id,
                seal=True,
                historical=True,
            )
            if (
                retention is not None
                and retention["removed_papers"]
                and not args.no_index
            ):
                result["index"] = build_lance_index(
                    cfg,
                    source="arxiv",
                )
            result["status"] = "ok"
            return _finish(result, state_dir, started)

        previous_run_complete = _previous_run_complete(state_dir, day)
        summary = _done_summary(data_root, day)
        skipped = bool(summary and _done_is_complete(summary)) and not args.force
        if skipped:
            result["fetch"] = {"skipped": True, "reason": "complete .done exists"}
        else:
            code, stdout, stderr = _run_daily(
                data_root,
                cache_dir,
                day=day,
                limit=args.limit,
                license_limit=args.license_limit,
                refresh=args.refresh,
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(stdout)
                if stderr:
                    handle.write(stderr)
            result["fetch"] = {
                "exit_code": code,
                "summary": _last_json_line(stdout),
            }
            if code != 0:
                result["status"] = "waiting" if code in (3, 4) else "failed"
                return _finish(result, state_dir, started)
            summary = _done_summary(data_root, day)

        if not _done_is_complete(summary):
            result["error"] = "today's .done is missing or not a complete run"
            return _finish(result, state_dir, started, rc=2)

        retention: dict[str, Any] | None = None
        retention_changed = False
        if not args.no_prune:
            retention = prune_weekly_corpus(
                data_root,
                max_retention_weeks=max_retention_weeks,
                as_of=day,
            )
            result["retention"] = retention
            retention_changed = bool(
                retention["removed_papers"]
                or retention["removed_weeks"]
            )
            if retention["removed_papers"]:
                result["vector_prune"] = prune_arxiv_vectors(
                    cfg,
                    active_ids=weekly_paper_ids(data_root),
                )

        if previous_run_complete and not retention_changed and not args.force:
            result["status"] = "skipped"
            result["reason"] = "daily pipeline already completed"
            return _finish(result, state_dir, started)

        if not args.no_bind:
            result["week"] = build_week(
                data_root,
                week_id,
                seal=args.seal_week,
                historical=False,
            )

        if not args.no_embed:
            result["vectors"] = embed_missing_arxiv(
                cfg,
                limit=args.embed_limit,
            )
        if not args.no_index:
            result["index"] = build_lance_index(
                cfg,
                source="arxiv",
            )

        result["status"] = "ok"
        return _finish(result, state_dir, started)
    finally:
        _release_lock(lock_dir)


def _finish(
    result: dict[str, Any],
    state_dir: Path,
    started: float,
    *,
    rc: int = 0,
) -> int:
    result["elapsed_s"] = round(time.time() - started, 3)
    _atomic_json(state_dir / "last_run.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
