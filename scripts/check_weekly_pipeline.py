"""Offline checks for weekly binding, harness input, and week filtering."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hotspot.core.models import SearchFilters  # noqa: E402
from hotspot.index.lance_store import _where  # noqa: E402
from hotspot.weekly import (  # noqa: E402
    build_week,
    iso_week_id,
    prune_weekly_corpus,
    retention_cutoff,
)
from hotspot.weekly.vectors import prune_arxiv_vectors  # noqa: E402


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
    if retention_cutoff("2026-10-28", 2) != "2026-W43":
        failures.append("retention cutoff")
    if retention_cutoff("2026-01-01", 2) != "2025-W52":
        failures.append("retention cutoff across ISO year")
    try:
        retention_cutoff("2026-10-28", 0)
        invalid_retention_rejected = False
    except ValueError:
        invalid_retention_rejected = True
    if not invalid_retention_rejected:
        failures.append("invalid retention rejected")
    try:
        retention_cutoff("2026-10-28", 521)
        oversized_retention_rejected = False
    except ValueError:
        oversized_retention_rejected = True
    if not oversized_retention_rejected:
        failures.append("oversized retention rejected")

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
        for day_number in range(26, 31):
            (week / f"2026-10-{day_number}.done").write_text(json.dumps({
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
            or manifest["expected_days"] != [
                "2026-10-26",
                "2026-10-27",
                "2026-10-28",
                "2026-10-29",
                "2026-10-30",
            ]
            or manifest["missing_days"]
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
        (week / "2026-10-26.done").write_text(json.dumps({
            "exit_code": 0,
            "failed_feeds": [],
            "feed_build_stale": [],
            "feed_stalled": [],
            "args": {"limit": 0, "license_limit": 0},
        }), encoding="utf-8")
        (week / "2026-10-27.done").unlink()
        try:
            build_week(root, "2026-W44", seal=True)
            failures.append("missing weekday seal blocked")
        except ValueError:
            pass
        (week / "2026-10-27.done").write_text(json.dumps({
            "exit_code": 0,
            "failed_feeds": [],
            "feed_build_stale": [],
            "feed_stalled": [],
            "args": {"limit": 1, "license_limit": 0},
        }), encoding="utf-8")
        try:
            build_week(root, "2026-W44", seal=True)
            failures.append("limited run seal blocked")
        except ValueError:
            pass

        (week / "2026-10-27.done").write_text(json.dumps({
            "exit_code": 0,
            "failed_feeds": [],
            "feed_build_stale": [],
            "feed_stalled": [],
            "args": {"limit": 0, "license_limit": 0},
        }), encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "daily_arxiv_sync.py"),
                "--seal-week",
                "--week",
                "2026-W44",
                "--no-prune",
            ],
            cwd=ROOT,
            env={**os.environ, "ARXIV_WEEKLY_DATA": str(root)},
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        if (
            proc.returncode != 0
            or json.loads(proc.stdout or "{}").get("mode") != "seal"
        ):
            failures.append("CLI seal path")

        retention_root = root / "retention"
        for week_id in ("2026-W42", "2026-W43", "2026-W44"):
            (retention_root / week_id).mkdir(parents=True)
        retained = _paper(
            "arxiv:1",
            "2026-10-26",
            "1",
            "Weekly paper",
        )
        expired = _paper(
            "arxiv:old",
            "2026-10-12",
            "old",
            "Old paper",
        )
        unbound = _paper(
            "arxiv:unknown",
            "not-a-date",
            "unknown",
            "Unbound paper",
        )
        _write_jsonl(
            retention_root / "papers.jsonl",
            [retained, expired, unbound],
        )
        retention = prune_weekly_corpus(
            retention_root,
            max_retention_weeks=2,
            as_of="2026-10-28",
        )
        remaining = [
            json.loads(line)
            for line in (
                retention_root / "papers.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line
        ]
        if not (
            retention["cutoff_week"] == "2026-W43"
            and retention["removed_papers"] == 1
            and retention["unbound_papers_retained"] == 1
            and not (retention_root / "2026-W42").exists()
            and (retention_root / "2026-W43").is_dir()
            and {row["paper_id"] for row in remaining}
            == {"arxiv:1", "arxiv:unknown"}
        ):
            failures.append("weekly retention cleanup")
        retained_manifest = build_week(
            retention_root,
            "2026-W44",
            historical=False,
        )
        papers_sha256 = hashlib.sha256(
            (retention_root / "papers.jsonl").read_bytes()
        ).hexdigest()
        if not (
            retained_manifest["member_count"] == 1
            and retained_manifest["source_papers_sha256"] == papers_sha256
        ):
            failures.append("retention manifest source hash")

        vector_dir = root / "workspace" / "arxiv_vectors"
        vector_dir.mkdir(parents=True)
        vector_path = vector_dir / "vec_0000.npz"
        with vector_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
                ids=np.asarray(["arxiv:1", "arxiv:old"]),
            )
        vector_prune = prune_arxiv_vectors(
            {"paths": {"workspace": str(root / "workspace")}},
            active_ids={"arxiv:1"},
        )
        with np.load(vector_path, allow_pickle=False) as data:
            vector_ids = [str(value) for value in data["ids"]]
        if not (
            vector_prune["removed_ids"] == 1
            and vector_prune["rewritten_shards"] == 1
            and vector_ids == ["arxiv:1"]
        ):
            failures.append("weekly vector cleanup")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    result = {"checks": 15, "failures": failures}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        print("FAIL:", ", ".join(failures), file=sys.stderr)
        return 1
    print("✓ weekly pipeline checks=15 failures=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
