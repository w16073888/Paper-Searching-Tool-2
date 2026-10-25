"""Offline checks for weekly binding, harness input, and week filtering."""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hotspot.core.models import SearchFilters  # noqa: E402
from hotspot.index.lance_store import _where  # noqa: E402
from hotspot.weekly import build_week, iso_week_id  # noqa: E402


def _paper(
    paper_id: str,
    first_seen: str,
    arxiv_id: str,
    title: str,
) -> dict:
    return {
        "paper_id": paper_id,
        "source_key": arxiv_id,
        "title": title,
        "year": 2026,
        "arxiv_id": arxiv_id,
        "abstract": f"{title} abstract",
        "topics": ["cs.LG"],
        "rl": {"is_rl": True, "subfield": ["RL"]},
        "first_seen": first_seen,
        "last_updated": first_seen,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    failures = []
    if iso_week_id("2026-10-26") != "2026-W44":
        failures.append("ISO week conversion")
    if _where(SearchFilters(weeks=["2026-W44"])) != (
        "week_id IN ('2026-W44')"
    ):
        failures.append("LanceDB week filter")

    temp_root = ROOT / "workspace"
    temp_root.mkdir(parents=True, exist_ok=True)
    root = temp_root / f"weekly-check-{uuid.uuid4().hex}"
    root.mkdir()
    try:
        week = root / "2026-W44"
        week.mkdir()
        papers = [
            _paper("arxiv:1", "2026-10-26", "1", "Weekly paper"),
            _paper("arxiv:old", "2026-10-18", "old", "Old paper"),
        ]
        _write_jsonl(root / "papers.jsonl", papers)
        _write_jsonl(week / "2026-10-26.jsonl", [papers[0]])
        _write_jsonl(week / "2026-10-26.announce.jsonl", [
            {"key": "1", "announce_type": "new"},
        ])
        (week / "2026-10-26.done").write_text(json.dumps({
            "exit_code": 0,
            "failed_feeds": [],
            "feed_build_stale": [],
            "feed_stalled": [],
            "args": {"limit": 0, "license_limit": 0},
        }), encoding="utf-8")

        manifest = build_week(root, "2026-W44", seal=True)
        if manifest["status"] != "sealed" or manifest["member_count"] != 1:
            failures.append("weekly member binding")
        harness = [
            json.loads(line)
            for line in (
                week / "harness_input.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line
        ]
        if (
            len(harness) != 1
            or harness[0]["week_id"] != "2026-W44"
            or harness[0]["announce_types"] != ["new"]
        ):
            failures.append("harness contract")

        (week / "2026-10-26.done").write_text(json.dumps({
            "exit_code": 0,
            "failed_feeds": ["cs.LG"],
            "feed_build_stale": [],
            "feed_stalled": [],
            "args": {"limit": 0, "license_limit": 0},
        }), encoding="utf-8")
        try:
            build_week(root, "2026-W44", seal=True)
            failures.append("invalid day seal blocked")
        except ValueError:
            pass
    finally:
        shutil.rmtree(root, ignore_errors=True)

    result = {"checks": 6, "failures": failures}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        print("FAIL:", ", ".join(failures), file=sys.stderr)
        return 1
    print("✓ weekly pipeline checks=6 failures=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
