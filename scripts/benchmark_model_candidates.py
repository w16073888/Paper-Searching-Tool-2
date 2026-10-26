"""Benchmark OpenAI-compatible model endpoints with fixed workloads.

The script measures streaming TTFT, decode throughput, long-context latency,
and bounded concurrency. It does not decide which model to deploy.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return round(ordered[index], 3)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["seconds"]) for row in rows]
    ttfts = [
        float(row["ttft_s"])
        for row in rows
        if row.get("ttft_s") is not None
    ]
    rates = [
        float(row["decode_tokens_per_s"])
        for row in rows
        if row.get("decode_tokens_per_s") is not None
    ]
    return {
        "count": len(rows),
        "latency_p50_s": _percentile(latencies, 0.50),
        "latency_p95_s": _percentile(latencies, 0.95),
        "ttft_p50_s": _percentile(ttfts, 0.50),
        "ttft_p95_s": _percentile(ttfts, 0.95),
        "decode_tokens_per_s_p50": _percentile(rates, 0.50),
        "decode_tokens_per_s_p95": _percentile(rates, 0.95),
        "errors": sum(1 for row in rows if row.get("error")),
    }


async def _tokenize_count(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    text: str,
) -> int | None:
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        response = await client.post(
            root + "/tokenize",
            json={"model": model, "prompt": text},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        return int(response.json().get("count") or 0)
    except Exception:  # noqa: BLE001 - tokenizer is optional outside vLLM
        return None


async def _build_context(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    target_tokens: int,
) -> tuple[str, int | None]:
    marker = f"CTX-{target_tokens}-BEGIN"
    end = f"CTX-{target_tokens}-END"
    paragraph = (
        "这是用于长上下文检索性能测试的独立占位段落，编号仅用于定位，"
        "不包含需要推理的关系。"
    )
    header = f"标记 {marker}。"
    footer = f"标记 {end}。\n请只输出开头标记和结尾标记，用换行分隔，不要解释。"
    body: list[str] = []
    index = 1
    while True:
        body.append(f"段落 {index:04d}：{paragraph}")
        text = "\n".join([header, *body, footer])
        count = await _tokenize_count(
            client, base_url, api_key, model, text
        )
        if count is None:
            chars_per_token = 2.0
            while len(text) < target_tokens * chars_per_token:
                index += 1
                body.append(f"段落 {index:04d}：{paragraph}")
                text = "\n".join([header, *body, footer])
            return text, None
        if count >= target_tokens - 24:
            return text, count
        index += 1


async def _request(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    ignore_eos: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True

    started = time.perf_counter()
    first_token: float | None = None
    content: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason = None
    try:
        async with client.stream(
            "POST",
            base_url.rstrip("/") + "/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                event = json.loads(raw)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        if first_token is None:
                            first_token = time.perf_counter()
                        content.append(str(piece))
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        ended = time.perf_counter()
        completion_tokens = int(usage.get("completion_tokens") or 0)
        decode_seconds = (
            ended - first_token
            if first_token is not None
            else ended - started
        )
        rate = (
            max(completion_tokens - 1, 0) / decode_seconds
            if decode_seconds > 0
            else None
        )
        return {
            "seconds": round(ended - started, 3),
            "ttft_s": (
                round(first_token - started, 3)
                if first_token is not None
                else None
            ),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": completion_tokens,
            "decode_tokens_per_s": round(rate, 2) if rate else None,
            "finish_reason": finish_reason,
            "content_sha256": hashlib.sha256(
                "".join(content).encode("utf-8")
            ).hexdigest(),
            "content_preview": "".join(content)[:160],
        }
    except Exception as exc:  # noqa: BLE001 - errors are benchmark data
        return {
            "seconds": round(time.perf_counter() - started, 3),
            "ttft_s": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "decode_tokens_per_s": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def _concurrency(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    width: int,
    rounds: int,
) -> dict[str, Any]:
    samples = []
    for round_number in range(1, rounds + 1):
        prompts = [
            (
                f"并发轮次 {round_number}，请求 {request_number}："
                "用一句话说明模型量化推理的显存收益。"
            )
            for request_number in range(1, width + 1)
        ]
        started = time.perf_counter()
        rows = await asyncio.gather(*[
            _request(
                client,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                prompt=prompt,
                max_tokens=64,
                ignore_eos=True,
            )
            for prompt in prompts
        ])
        wall = time.perf_counter() - started
        total_tokens = sum(row["completion_tokens"] for row in rows)
        samples.append({
            "round": round_number,
            "wall_seconds": round(wall, 3),
            "aggregate_decode_tokens_per_s": round(
                total_tokens / wall, 2
            ) if wall else None,
            "requests": rows,
        })
    return {
        "width": width,
        "rounds": rounds,
        "samples": samples,
        "aggregate_decode_tokens_per_s_p50": _percentile([
            float(row["aggregate_decode_tokens_per_s"])
            for row in samples
            if row.get("aggregate_decode_tokens_per_s") is not None
        ], 0.50),
        "wall_seconds_p95": _percentile([
            float(row["wall_seconds"]) for row in samples
        ], 0.95),
        "errors": sum(
            bool(request.get("error"))
            for sample in samples
            for request in sample["requests"]
        ),
    }


async def _soak(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.soak_seconds <= 0:
        return {"enabled": False}
    deadline = time.monotonic() + args.soak_seconds
    rows = []
    long_prompt, _ = await _build_context(
        client, args.base_url, args.api_key, args.model, 4096
    )
    while time.monotonic() < deadline:
        number = len(rows) + 1
        prompt = (
            long_prompt
            if number % 10 == 0
            else f"稳定性请求 {number}：用一句话解释离线强化学习的核心风险。"
        )
        rows.append(await _request(
            client,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            prompt=prompt,
            max_tokens=48,
            ignore_eos=False,
        ))
    return {
        "enabled": True,
        "duration_s": args.soak_seconds,
        **_summary(rows),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    timeout = httpx.Timeout(args.timeout, connect=15)
    async with httpx.AsyncClient(timeout=timeout) as client:
        warmup = await _request(
            client,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            prompt="只回复 READY",
            max_tokens=8,
            ignore_eos=False,
        )

        short = [
            await _request(
                client,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                prompt="请用三点解释强化学习与监督学习的区别，每点不超过30字。",
                max_tokens=128,
                ignore_eos=True,
            )
            for _ in range(3)
        ]
        medium = await _request(
            client,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            prompt="用一个简短例子说明 PPO 中 clip 操作为什么能限制策略更新幅度。",
            max_tokens=512,
            ignore_eos=True,
        )

        context_rows = []
        for requested_tokens in [1024, 2048, 4096, 7600]:
            prompt, tokenized = await _build_context(
                client,
                args.base_url,
                args.api_key,
                args.model,
                requested_tokens,
            )
            row = await _request(
                client,
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                prompt=prompt,
                max_tokens=32,
                ignore_eos=False,
            )
            row["requested_context_tokens"] = requested_tokens
            row["tokenizer_count_before_chat_template"] = tokenized
            row["prompt_sha256"] = hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest()
            context_rows.append(row)

        concurrency = [
            await _concurrency(client, args, width, args.concurrent_rounds)
            for width in [1, 2, 3]
        ]
        soak = await _soak(client, args)

    return {
        "schema_version": "model-benchmark-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.base_url,
        "model": args.model,
        "conditions": {
            "temperature": 0,
            "top_p": 1,
            "stream": True,
            "ignore_eos_for_throughput": True,
            "timeout_s": args.timeout,
            "concurrent_rounds": args.concurrent_rounds,
            "soak_seconds": args.soak_seconds,
        },
        "warmup": warmup,
        "short": _summary(short),
        "medium": medium,
        "long_context": context_rows,
        "concurrency": concurrency,
        "soak": soak,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--api-key-env",
        default="QWEN_LOCAL_API_KEY",
        help="environment variable containing the bearer token",
    )
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--concurrent-rounds", type=int, default=5)
    parser.add_argument("--soak-seconds", type=int, default=0)
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
        "model": result["model"],
        "short": result["short"],
        "medium": {
            key: result["medium"].get(key)
            for key in [
                "seconds",
                "ttft_s",
                "prompt_tokens",
                "completion_tokens",
                "decode_tokens_per_s",
                "error",
            ]
        },
        "long_context": [
            {
                key: row.get(key)
                for key in [
                    "requested_context_tokens",
                    "prompt_tokens",
                    "seconds",
                    "ttft_s",
                    "error",
                ]
            }
            for row in result["long_context"]
        ],
        "concurrency": [
            {
                key: row.get(key)
                for key in [
                    "width",
                    "aggregate_decode_tokens_per_s_p50",
                    "wall_seconds_p95",
                    "errors",
                ]
            }
            for row in result["concurrency"]
        ],
        "soak": result["soak"],
        "out": str(out),
    }, ensure_ascii=False, indent=2))
    return 0 if not result["short"]["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
