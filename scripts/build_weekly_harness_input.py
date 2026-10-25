"""Materialize ISO-week paper bindings and LLM harness inputs."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hotspot.weekly import build_week, iso_week_id  # noqa: E402

_WEEK_RE = re.compile(r"^\d{4}-W\d{2}$")


def default_data_root() -> Path:
    return Path(
        os.environ.get(
            "ARXIV_WEEKLY_DATA",
            str(ROOT / "data" / "arxiv-weekly"),
        )
    )


def available_weeks(data_root: Path) -> list[str]:
    return sorted(
        path.name for path in data_root.iterdir()
        if path.is_dir() and _WEEK_RE.fullmatch(path.name)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(default_data_root()))
    parser.add_argument("--week", default=None, help="ISO week, e.g. 2026-W44")
    parser.add_argument("--all-existing", action="store_true")
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not (data_root / "papers.jsonl").is_file():
        parser.error(f"missing {data_root / 'papers.jsonl'}")
    if args.seal and not args.week:
        parser.error("--seal requires one --week")
    if not args.week and not args.all_existing:
        parser.error("pass --week or --all-existing")

    weeks = [args.week] if args.week else available_weeks(data_root)
    current_week = iso_week_id(_dt.date.today())
    manifests = []
    for week_id in weeks:
        manifests.append(build_week(
            data_root,
            week_id,
            seal=bool(args.seal),
            historical=week_id < current_week,
        ))
    print(json.dumps({
        "weeks": len(manifests),
        "sealed": sum(1 for item in manifests if item["status"] == "sealed"),
        "partial": sum(1 for item in manifests if item["status"] == "partial"),
        "open": sum(1 for item in manifests if item["status"] == "open"),
        "manifests": [
            {
                "week_id": item["week_id"],
                "status": item["status"],
                "member_count": item["member_count"],
                "harness_input_sha256":
                    item["files"]["harness_input_sha256"],
            }
            for item in manifests
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
