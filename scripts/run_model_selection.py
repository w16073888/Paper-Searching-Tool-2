"""Run the frozen model-selection v1 set against an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


SYSTEM_PROMPT = (
    "你是强化学习论文研究助手。中文题回答 2-3 个要点，总字数不超过 220 字；"
    "英文题回答 2-3 个要点，总长度不超过 180 词。"
    "题目超出机器学习论文范围时，只回复“超出范围”。"
    "不要编造论文、数字或引用。"
)
REFUSAL_MARKERS = ("超出范围", "out of scope", "无法回答", "cannot answer")


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def _score(row: dict[str, Any], content: str) -> dict[str, Any]:
    normalized = _normalize(content)
    groups = []
    for alternatives in row.get("must_include") or []:
        matched = [
            alternative
            for alternative in alternatives
            if _normalize(alternative) in normalized
        ]
        groups.append({"matched": matched, "ok": bool(matched)})
    refused = any(marker in normalized for marker in REFUSAL_MARKERS)
    expected_refusal = bool(row.get("expected_refusal"))
    refusal_ok = refused == expected_refusal
    length_ok = len(content.strip()) <= int(row.get("max_chars") or 10_000)
    coverage_ok = all(group["ok"] for group in groups)
    return {
        "passed": coverage_ok and refusal_ok and length_ok,
        "coverage_ok": coverage_ok,
        "refusal_ok": refusal_ok,
        "length_ok": length_ok,
        "refused": refused,
        "groups": groups,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    source_sha = hashlib.sha256(
        Path(args.questions).read_bytes()
    ).hexdigest()
    timeout = httpx.Timeout(args.timeout, connect=15)
    results = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for row in rows:
            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["question"]},
                ],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": args.max_tokens,
            }
            started = time.perf_counter()
            error = None
            content = ""
            usage: dict[str, Any] = {}
            try:
                response = await client.post(
                    args.base_url.rstrip("/") + "/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {args.api_key}"},
                )
                response.raise_for_status()
                data = response.json()
                content = (
                    data["choices"][0]["message"].get("content") or ""
                )
                usage = data.get("usage") or {}
            except Exception as exc:  # noqa: BLE001 - errors are results
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            result = {
                "id": row["id"],
                "category": row["category"],
                "language": row["language"],
                "question": row["question"],
                "content": content,
                "seconds": round(elapsed, 3),
                "usage": usage,
                "error": error,
                "automatic": (
                    _score(row, content)
                    if not error
                    else {
                        "passed": False,
                        "coverage_ok": False,
                        "refusal_ok": False,
                        "length_ok": False,
                        "refused": False,
                        "groups": [],
                    }
                ),
            }
            results.append(result)
            print(
                f"{row['id']} | {'PASS' if result['automatic']['passed'] else 'FAIL'} "
                f"| {result['seconds']:.3f}s"
            )

    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({row["category"] for row in results}):
        subset = [row for row in results if row["category"] == category]
        passed = sum(bool(row["automatic"]["passed"]) for row in subset)
        by_category[category] = {
            "passed": passed,
            "total": len(subset),
            "rate": round(passed / len(subset), 4),
        }
    passed = sum(bool(row["automatic"]["passed"]) for row in results)
    return {
        "schema_version": "model-selection-result-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.base_url,
        "model": args.model,
        "questions": args.questions,
        "questions_sha256": source_sha,
        "conditions": {
            "temperature": 0,
            "top_p": 1,
            "max_tokens": args.max_tokens,
        },
        "summary": {
            "passed": passed,
            "total": len(results),
            "rate": round(passed / len(results), 4) if results else 0,
            "errors": sum(bool(row["error"]) for row in results),
            "by_category": by_category,
        },
        "rows": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument(
        "--questions",
        default="eval/model_selection_v1.json",
    )
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    args.api_key = os.environ.get(args.api_key_env, "")
    if not args.api_key:
        parser.error(f"missing environment variable: {args.api_key_env}")
    result = asyncio.run(run(args))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "model": args.model,
        "summary": result["summary"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))
    return 0 if result["summary"]["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
