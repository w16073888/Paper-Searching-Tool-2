"""Bind the append-only arXiv store into immutable weekly harness inputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

HARNESS_SCHEMA_VERSION = "weekly-harness-v1"
_WEEK_RE = re.compile(r"^\d{4}-W\d{2}$")


def iso_week_id(value: str | date) -> str:
    day = date.fromisoformat(value) if isinstance(value, str) else value
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )


def _done_is_valid(row: dict[str, Any]) -> bool:
    args = row.get("args") or {}
    return (
        int(row.get("exit_code") or 0) == 0
        and not (row.get("failed_feeds") or [])
        and not (row.get("feed_build_stale") or [])
        and not (row.get("feed_stalled") or [])
        and not int(args.get("limit") or 0)
        and not int(args.get("license_limit") or 0)
    )


def _expected_weekdays(week_id: str) -> list[str]:
    year, week = week_id.split("-W")
    return [
        date.fromisocalendar(int(year), int(week), weekday).isoformat()
        for weekday in range(1, 6)
    ]


def _load_announcements(week_dir: Path) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    events = []
    by_arxiv_id: dict[str, set[str]] = {}
    for path in sorted(week_dir.glob("*.announce.jsonl")):
        event_date = path.name.removesuffix(".announce.jsonl")
        for row in _read_jsonl(path):
            key = str(row.get("key") or "").strip()
            announce_type = str(row.get("announce_type") or "").strip()
            if not key or not announce_type:
                continue
            events.append({
                "week_id": week_dir.name,
                "date": event_date,
                "paper_id": f"arxiv:{key}",
                "arxiv_id": key,
                "announce_type": announce_type,
            })
            by_arxiv_id.setdefault(key, set()).add(announce_type)
    events.sort(key=lambda row: (row["date"], row["arxiv_id"], row["announce_type"]))
    return events, by_arxiv_id


def build_week(
    data_root: Path | str,
    week_id: str,
    *,
    seal: bool = False,
    historical: bool = False,
) -> dict[str, Any]:
    """Materialize one ISO week and return its manifest.

    Membership is determined only by ``first_seen``. Revisions remain in the
    original week and are represented separately as announcement events.
    """
    if not _WEEK_RE.fullmatch(week_id):
        raise ValueError(f"invalid ISO week_id: {week_id!r}")
    root = Path(data_root)
    week_dir = root / week_id
    if not week_dir.is_dir():
        raise FileNotFoundError(f"week directory does not exist: {week_dir}")

    papers_path = root / "papers.jsonl"
    all_papers = _read_jsonl(papers_path)
    members = [
        row for row in all_papers
        if row.get("first_seen")
        and iso_week_id(str(row["first_seen"])) == week_id
    ]
    members.sort(key=lambda row: (
        str(row.get("first_seen") or ""),
        str(row.get("paper_id") or ""),
    ))

    daily_files = sorted(
        path for path in week_dir.glob("*.jsonl")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.jsonl", path.name)
    )
    done_files = sorted(week_dir.glob("*.done"))
    done_by_date = {}
    for path in done_files:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        done_by_date[path.stem] = json.loads(raw)

    announcements, announce_types = _load_announcements(week_dir)
    source_sha = _sha256(papers_path)
    harness_rows = []
    for row in members:
        arxiv_id = str(row.get("arxiv_id") or row.get("source_key") or "")
        topics = list(row.get("topics") or [])
        rl = row.get("rl") or {}
        harness_rows.append({
            "schema_version": HARNESS_SCHEMA_VERSION,
            "week_id": week_id,
            "paper_id": row.get("paper_id") or "",
            "title": row.get("title") or "",
            "abstract": row.get("abstract") or "",
            "year": row.get("year"),
            "arxiv_id": arxiv_id,
            "url": row.get("url") or "",
            "topics": topics,
            "subfields": list(rl.get("subfield") or []),
            "is_rl": bool(rl.get("is_rl")),
            "first_seen": row.get("first_seen") or "",
            "last_updated": row.get("last_updated") or "",
            "announce_types": sorted(announce_types.get(arxiv_id, set())),
            "license": row.get("license"),
            "source_sha256": source_sha,
        })

    expected_days = _expected_weekdays(week_id)
    missing_days = [
        day for day in expected_days if day not in done_by_date
    ]
    valid_done = (
        not missing_days
        and all(_done_is_valid(row) for row in done_by_date.values())
    )
    observed_days = sorted(
        path.name.removesuffix(".jsonl") for path in daily_files
    )
    current_week = iso_week_id(date.today())
    if seal:
        if not valid_done:
            reasons = []
            if missing_days:
                reasons.append("missing daily runs: " + ", ".join(missing_days))
            invalid_days = sorted(
                day for day, row in done_by_date.items()
                if not _done_is_valid(row)
            )
            if invalid_days:
                reasons.append(
                    "incomplete daily runs: " + ", ".join(invalid_days)
                )
            raise ValueError(
                "cannot seal week: " + "; ".join(reasons)
            )
        status = "sealed"
    elif historical and week_id < current_week:
        status = "partial"
    else:
        status = "open"

    member_path = week_dir / "member_papers.jsonl"
    announce_path = week_dir / "announcements.jsonl"
    harness_path = week_dir / "harness_input.jsonl"
    _atomic_write(member_path, _jsonl_text(members))
    _atomic_write(announce_path, _jsonl_text(announcements))
    _atomic_write(harness_path, _jsonl_text(harness_rows))

    manifest = {
        "schema_version": HARNESS_SCHEMA_VERSION,
        "week_id": week_id,
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "member_count": len(members),
        "announcement_count": len(announcements),
        "observed_days": observed_days,
        "expected_days": expected_days,
        "missing_days": missing_days,
        "completed_days": sorted(done_by_date),
        "all_completed_days_valid": valid_done,
        "source_papers": str(papers_path),
        "source_papers_sha256": source_sha,
        "files": {
            "member_papers": member_path.name,
            "member_papers_sha256": _sha256(member_path),
            "announcements": announce_path.name,
            "announcements_sha256": _sha256(announce_path),
            "harness_input": harness_path.name,
            "harness_input_sha256": _sha256(harness_path),
        },
    }
    _atomic_write(
        week_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return manifest
