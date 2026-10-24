"""Build the silver scope-classification dataset with Mobile Cloud MaaS.

The generator combines corpus-grounded in-scope questions with generated OOD
and cross-domain boundary questions. The API key is read only through
``config.yaml`` -> ``llm.api_key_env`` and is never written to disk.

Smoke run:

    python scripts/build_scope_dataset.py --paper-count 10 --ood-count 4 --boundary-count 2

Full pilot:

    python scripts/build_scope_dataset.py
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(line_buffering=True)

from hotspot.config import load_config
from hotspot.corpus import Paper, open_corpus

PROMPT_VERSION = "scope-dataset-v1"
DATASET_VERSION = "scope-dataset-v1"
DEFAULT_SEED = 20261023
LABELS = ("in_scope", "out_of_scope")
LANGUAGES = ("zh", "zh_en", "en")

OOD_DOMAINS = (
    "clinical_medicine",
    "medical_imaging",
    "genomics",
    "molecular_biology",
    "pharmacology",
    "neuroscience",
    "quantum_physics",
    "semiconductor_materials",
    "chemistry",
    "geology",
    "materials_science",
    "civil_engineering",
    "mechanical_engineering",
    "electrical_power",
    "petroleum_engineering",
    "aerospace",
    "astronomy",
    "atmospheric_science",
    "climate_science",
    "marine_science",
    "ecology",
    "agriculture",
    "food_science",
    "architecture",
    "urban_planning",
    "transportation_engineering",
    "economics",
    "finance",
    "law",
    "history",
    "archaeology",
    "general_knowledge",
)

BOUNDARY_DOMAINS = (
    "ml_for_medical_imaging",
    "ml_for_genomics",
    "ml_for_drug_discovery",
    "ml_for_quantum_control",
    "ml_for_materials_discovery",
    "ml_for_civil_monitoring",
    "ml_for_climate_forecasting",
    "ml_for_energy_forecasting",
    "ml_for_agriculture",
    "ml_for_finance",
    "ml_for_education",
    "ml_for_transport_control",
    "ml_for_chemistry",
    "physics_informed_ml",
    "ml_for_neuroscience",
    "abbreviation_and_polysemy",
)


class InScopeQuestion(BaseModel):
    variant: int = Field(ge=1, le=2)
    question: str
    language: Literal["zh", "zh_en", "en"]
    kind: Literal["local", "compare", "relation", "trend", "followup"]
    domain: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class InScopePaper(BaseModel):
    paper_id: str
    questions: list[InScopeQuestion]


class InScopeBatch(BaseModel):
    papers: list[InScopePaper]


class OodQuestion(BaseModel):
    domain: str
    question: str
    language: Literal["zh", "zh_en", "en"]
    kind: Literal["ood", "hard_negative"]
    boundary_cue: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class OodBatch(BaseModel):
    questions: list[OodQuestion]


class BoundaryQuestion(BaseModel):
    domain: str
    question: str
    expected: Literal["in_scope", "out_of_scope"]
    language: Literal["zh", "zh_en", "en"]
    boundary_cue: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class BoundaryBatch(BaseModel):
    questions: list[BoundaryQuestion]


IN_SCOPE_INSTRUCTIONS = """你是本地 AI/ML 论文库 scope 数据构造员。给定论文标题、摘要和元数据，
为每篇论文构造恰好 2 个用户可能提出的问题。

问题必须：
1. 保持论文 ID 原样；问题应能由该论文或同主题论文回答。
2. variant=1 使用中文；variant=2 使用中英混合或英文，但不要重复 variant=1。
3. 覆盖事实、比较、关系、趋势等不同问法；不要以“请总结摘要”开头。
4. 不得把标签词 in_scope/out_of_scope 写进问题，不得编造论文中不存在的方法名。
5. 问题必须是 AI/ML/RL/LLM/Agent/机器人学习/决策方向，不能只问应用领域本身。
6. 每篇论文必须返回 exactly 2 条，所有输入 paper_id 都必须覆盖一次。

这是 silver 数据，必须返回结构化结果。"""

OOD_INSTRUCTIONS = """你负责构造本地 AI/ML 论文库的明确域外问题和少量困难负例。
输入给出域名和每个领域的问题数。严格按数量返回，domain 原样使用。

要求：
1. out-of-scope 问题只询问医学、生物、物理、材料、土木、气候、航天、化学、
   经济、法律、历史等领域本身的问题，不得同时询问其中的机器学习/强化学习方法。
2. 中文为主，加入中英混合和英文问法；避免模板化重复。
3. general_knowledge 领域要包含缩写、多义词、泛词等 hard_negative。
4. 不得把 in_scope/out_of_scope 写进问题。
5. 每个问题给出 boundary_cue 和简短 rationale，confidence 表示域外标签置信度。
6. 每个输入 domain 必须返回指定数量，不能遗漏或换名。"""

BOUNDARY_INSTRUCTIONS = """你负责构造跨领域 scope 边界题。输入给出应用领域和每个领域的问题数，
严格按数量返回，domain 原样使用。

判定口径：
- 只有用户明确询问其中的机器学习、强化学习、模型、算法或 AI 方法时，才是 in_scope。
- 只询问应用领域本身的热点、机制、工艺、实验或传统方法时，是 out_of_scope。
- 每个领域必须同时包含正例和负例，问法不能只靠领域关键词区分。

全局规则：
1. 中文为主，加入中英混合和英文。
2. 问题中不得出现 in_scope/out_of_scope 标签词。
3. 给出 boundary_cue 和 rationale；confidence 表示标签置信度。
4. 每个输入 domain 必须返回指定数量，不能遗漏或换名。"""


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


def stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def api_key(cfg: dict[str, Any]) -> str:
    llm = cfg.get("llm") or {}
    key = str(llm.get("api_key") or "").strip()
    env_name = str(llm.get("api_key_env") or "").strip()
    if not key and env_name:
        key = str(os.environ.get(env_name, "")).strip()
    if not key:
        raise RuntimeError(
            f"missing API key: set environment variable {env_name or '<configured name>'}"
        )
    return key


def model_name(cfg: dict[str, Any]) -> str:
    llm = cfg.get("llm") or {}
    return str(
        llm.get("model_flash") or llm.get("model_pro") or "deepseek-chat"
    )


def make_agent(cfg: dict[str, Any], output_type: type[BaseModel], instructions: str) -> Agent:
    llm = cfg.get("llm") or {}
    provider = OpenAIProvider(
        base_url=str(llm.get("base_url") or "").rstrip("/"),
        api_key=api_key(cfg),
    )
    model = OpenAIChatModel(model_name(cfg), provider=provider)
    return Agent(
        model,
        output_type=output_type,
        instructions=instructions,
        retries=1,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["cache_key"]): row
        for row in read_jsonl(path)
        if row.get("cache_key")
    }


def cache_key(kind: str, payload: dict[str, Any], model: str, seed: int) -> str:
    raw = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "kind": kind,
            "model": model,
            "seed": seed,
            "payload": payload,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return sha256_text(raw)


def sample_papers(
    corpus: Any,
    count: int,
    seed: int,
) -> list[Paper]:
    papers = [
        paper
        for paper in corpus.list_papers()
        if paper.paper_id and paper.title.strip()
    ]
    if len(papers) < count:
        raise RuntimeError(f"corpus has only {len(papers)} usable papers; need {count}")

    buckets: dict[str, list[Paper]] = defaultdict(list)
    for paper in papers:
        primary = paper.subfields[0] if paper.subfields else "unclassified"
        buckets[primary].append(paper)
    for bucket in buckets.values():
        bucket.sort(key=lambda paper: stable_rank(paper.paper_id, seed))
    bucket_keys = sorted(
        buckets,
        key=lambda key: stable_rank(f"bucket:{key}", seed),
    )

    selected: list[Paper] = []
    cursor = 0
    while len(selected) < count:
        key = bucket_keys[cursor % len(bucket_keys)]
        bucket = buckets[key]
        offset = cursor // len(bucket_keys)
        if offset < len(bucket):
            selected.append(bucket[offset])
        cursor += 1
    return selected


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", "", question).strip().lower()


def validate_question_text(question: str) -> None:
    value = question.strip()
    if not (5 <= len(value) <= 360):
        raise ValueError(f"question length out of range: {value!r}")
    if "\n" in value or "\r" in value:
        raise ValueError(f"question contains newline: {value!r}")
    if "in_scope" in value or "out_of_scope" in value:
        raise ValueError(f"question leaks label token: {value!r}")


def validate_in_scope_batch(
    output: InScopeBatch,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_ids = [str(row["paper_id"]) for row in payload["papers"]]
    by_id = {row.paper_id: row for row in output.papers}
    if set(by_id) != set(expected_ids):
        raise ValueError(
            f"in-scope paper coverage mismatch: expected={expected_ids}, got={list(by_id)}"
        )
    rows: list[dict[str, Any]] = []
    paper_meta = {str(row["paper_id"]): row for row in payload["papers"]}
    for paper_id in expected_ids:
        item = by_id[paper_id]
        if sorted(question.variant for question in item.questions) != [1, 2]:
            raise ValueError(f"{paper_id}: expected variants 1 and 2")
        for question in item.questions:
            validate_question_text(question.question)
            rows.append(
                {
                    "group_id": f"paper:{paper_id}",
                    "category": "in_scope",
                    "question": question.question.strip(),
                    "expected": "in_scope",
                    "kind": question.kind,
                    "language": question.language,
                    "domain": question.domain.strip() or "ai_ml",
                    "boundary_cue": "",
                    "rationale": question.rationale.strip(),
                    "silver_confidence": float(question.confidence),
                    "paper_id": paper_id,
                    "paper_title": paper_meta[paper_id]["title"],
                    "venue": paper_meta[paper_id].get("venue", ""),
                    "year": paper_meta[paper_id].get("year"),
                }
            )
    return rows


def validate_ood_batch(
    output: OodBatch,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = {
        str(row["domain"]): int(row["count"])
        for row in payload["domains"]
    }
    counts = Counter(row.domain for row in output.questions)
    for domain, count in expected.items():
        if counts[domain] < count:
            raise ValueError(
                f"OOD domain count too small for {domain}: expected={count}, got={counts[domain]}"
            )
    rows = []
    used: Counter[str] = Counter()
    for item in output.questions:
        if item.domain not in expected:
            continue
        if used[item.domain] >= expected[item.domain]:
            continue
        used[item.domain] += 1
        validate_question_text(item.question)
        rows.append(
            {
                "group_id": f"ood:{item.domain}",
                "category": "ood",
                "question": item.question.strip(),
                "expected": "out_of_scope",
                "kind": item.kind,
                "language": item.language,
                "domain": item.domain.strip(),
                "boundary_cue": item.boundary_cue.strip(),
                "rationale": item.rationale.strip(),
                "silver_confidence": float(item.confidence),
                "paper_id": "",
                "paper_title": "",
                "venue": "",
                "year": None,
            }
        )
    return rows


def validate_boundary_batch(
    output: BoundaryBatch,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = {
        str(row["domain"]): int(row["count"])
        for row in payload["domains"]
    }
    counts = Counter(row.domain for row in output.questions)
    for domain, count in expected.items():
        if counts[domain] < count:
            raise ValueError(
                f"boundary domain count too small for {domain}: "
                f"expected={count}, got={counts[domain]}"
            )
    rows = []
    used: Counter[str] = Counter()
    for item in output.questions:
        if item.domain not in expected:
            continue
        if used[item.domain] >= expected[item.domain]:
            continue
        used[item.domain] += 1
        validate_question_text(item.question)
        rows.append(
            {
                "group_id": f"boundary:{item.domain}",
                "category": "boundary",
                "question": item.question.strip(),
                "expected": item.expected,
                "kind": (
                    "boundary_positive"
                    if item.expected == "in_scope"
                    else "boundary_negative"
                ),
                "language": item.language,
                "domain": item.domain.strip(),
                "boundary_cue": item.boundary_cue.strip(),
                "rationale": item.rationale.strip(),
                "silver_confidence": float(item.confidence),
                "paper_id": "",
                "paper_title": "",
                "venue": "",
                "year": None,
            }
        )
    return rows


async def generate_one(
    *,
    cfg: dict[str, Any],
    kind: str,
    payload: dict[str, Any],
    output_type: type[BaseModel],
    instructions: str,
    validator: Any,
    timeout: float,
    seed: int,
) -> dict[str, Any]:
    prompt = json.dumps(payload, ensure_ascii=False)
    for attempt in range(1, 3):
        agent = make_agent(cfg, output_type, instructions)
        try:
            result = await agent.run(
                prompt,
                timeout=timeout,
                usage_limits=UsageLimits(request_limit=2),
            )
            rows = validator(result.output, payload)
            return {
                "rows": rows,
                "attempts": attempt,
                "usage": {
                    "requests": getattr(result.usage(), "requests", None)
                    if callable(getattr(result, "usage", None))
                    else None,
                },
            }
        except Exception:
            if attempt == 2:
                raise
    raise RuntimeError("unreachable")


async def generate_batches(
    *,
    cfg: dict[str, Any],
    specs: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    concurrency: int,
    timeout: float,
    seed: int,
    split_missing_batches: bool,
) -> None:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_spec(spec: dict[str, Any]) -> None:
        key = cache_key(spec["kind"], spec["payload"], model_name(cfg), seed)
        if key in cache:
            print(f"cache [{spec['kind']}] {spec['label']}")
            return
        async with semaphore:
            started = asyncio.get_running_loop().time()
            async def generate_payload(
                payload: dict[str, Any],
                depth: int = 0,
            ) -> dict[str, Any]:
                domains = payload.get("domains") or []
                can_split = (
                    split_missing_batches
                    and spec["kind"] in {"ood", "boundary"}
                    and len(domains) > 1
                    and depth < 4
                )
                if can_split:
                    middle = len(domains) // 2
                    left = {**payload, "domains": domains[:middle]}
                    right = {**payload, "domains": domains[middle:]}
                    print(
                        f"split missing [{spec['kind']}] {spec['label']} "
                        f"depth={depth + 1} sizes={len(left['domains'])},{len(right['domains'])}"
                    )
                    subresults = await asyncio.gather(
                        generate_payload(left, depth + 1),
                        generate_payload(right, depth + 1),
                    )
                    return {
                        "rows": [
                            row
                            for subresult in subresults
                            for row in subresult["rows"]
                        ],
                        "attempts": sum(
                            int(subresult.get("attempts") or 0)
                            for subresult in subresults
                        ),
                        "usage": {"requests": None},
                    }
                return await generate_one(
                    cfg=cfg,
                    kind=spec["kind"],
                    payload=payload,
                    output_type=spec["output_type"],
                    instructions=spec["instructions"],
                    validator=spec["validator"],
                    timeout=timeout,
                    seed=seed,
                )

            result = await generate_payload(spec["payload"])
            record = {
                "cache_key": key,
                "kind": spec["kind"],
                "label": spec["label"],
                "model": model_name(cfg),
                "prompt_version": PROMPT_VERSION,
                "seed": seed,
                "payload": spec["payload"],
                "attempts": result["attempts"],
                "usage": result["usage"],
                "rows": result["rows"],
                "generated_at": utc_now(),
            }
            cache[key] = record
            append_jsonl(cache_path, record)
            elapsed = round(asyncio.get_running_loop().time() - started, 2)
            print(
                f"generated [{spec['kind']}] {spec['label']} "
                f"rows={len(record['rows'])} elapsed={elapsed}s"
            )

    results = await asyncio.gather(
        *(run_spec(spec) for spec in specs),
        return_exceptions=True,
    )
    failures = [
        (spec["kind"], spec["label"], result)
        for spec, result in zip(specs, results)
        if isinstance(result, Exception)
    ]
    if failures:
        summary = "; ".join(
            f"{kind}/{label}: {type(exc).__name__}: {exc}"
            for kind, label, exc in failures
        )
        raise RuntimeError(f"{len(failures)} batch(es) failed: {summary}")


def allocate_group_splits(
    groups: list[str],
    seed: int,
) -> dict[str, str]:
    ordered = sorted(groups, key=lambda group: stable_rank(group, seed))
    n = len(ordered)
    n_train = round(n * 0.70)
    n_calibration = round(n * 0.15)
    assignments: dict[str, str] = {}
    for group in ordered[:n_train]:
        assignments[group] = "train"
    for group in ordered[n_train:n_train + n_calibration]:
        assignments[group] = "calibration"
    for group in ordered[n_train + n_calibration:]:
        assignments[group] = "internal_test"
    return assignments


def build_rows(
    cached_rows: list[dict[str, Any]],
    *,
    seed: int,
    external_path: Path,
) -> list[dict[str, Any]]:
    external = read_jsonl(external_path)
    blocked = {normalize_question(str(row["question"])) for row in external}
    seen = set(blocked)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cached_rows:
        question = str(row["question"]).strip()
        normalized = normalize_question(question)
        if normalized in seen:
            continue
        seen.add(normalized)
        by_group[str(row["group_id"])].append(dict(row))

    expected_groups = sorted(by_group)
    groups_by_category: dict[str, list[str]] = defaultdict(list)
    for group_id, rows in by_group.items():
        groups_by_category[str(rows[0]["category"])].append(group_id)
    split_map: dict[str, str] = {}
    for category_groups in groups_by_category.values():
        split_map.update(allocate_group_splits(category_groups, seed))
    output: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for group_id in expected_groups:
        rows = sorted(
            by_group[group_id],
            key=lambda row: stable_rank(str(row["question"]), seed),
        )
        for index, row in enumerate(rows, start=1):
            prefix = {
                "in_scope": "ins",
                "ood": "ood",
                "boundary": "bnd",
            }[str(row["category"])]
            group_slug = sha256_text(group_id)[:10]
            item_id = f"{prefix}_{group_slug}_{index:02d}"
            counters[item_id] += 1
            if counters[item_id] > 1:
                item_id = f"{item_id}_{counters[item_id]}"
            row.update(
                {
                    "id": item_id,
                    "split": split_map[group_id],
                    "dataset_version": DATASET_VERSION,
                    "annotator": "mobile_cloud_llm_silver",
                    "label_source": "llm_generated_silver",
                    "source": "mobile_trash_generation",
                }
            )
            output.append(row)
    return sorted(output, key=lambda row: row["id"])


def laya_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scope_question = {
        "scope": {
            "type": "choice",
            "instructions": "判断问题是否属于本地 AI/ML 论文库的研究范围。",
            "criteria": {
                "in_scope": "询问 AI/ML/RL/LLM/Agent/机器人学习或决策方法",
                "out_of_scope": "询问其他领域本身或与本地 AI/ML 论文库无关",
            },
        }
    }
    output = []
    for row in rows:
        output.append(
            {
                "id": row["id"],
                "state": {"question": row["question"]},
                "questions": scope_question,
                "expected": {"scope": row["expected"]},
                "split": row["split"],
                "category": row["category"],
                "group_id": row["group_id"],
            }
        )
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_review_csv(path: Path, rows: list[dict[str, Any]], seed: int) -> None:
    boundary = [row for row in rows if row["category"] == "boundary"]
    samples = [
        row
        for row in rows
        if row["category"] != "boundary"
        and int(stable_rank(str(row["id"]), seed)[:8], 16) % 10 == 0
    ]
    priority = sorted(
        boundary + samples,
        key=lambda row: (row["category"] != "boundary", row["id"]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "question",
        "expected",
        "category",
        "kind",
        "domain",
        "language",
        "boundary_cue",
        "rationale",
        "review_decision",
        "review_notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in priority:
            writer.writerow(
                {
                    key: row.get(key, "")
                    for key in fields
                }
            )


def build_quality_report(
    rows: list[dict[str, Any]],
    *,
    external_path: Path,
) -> dict[str, Any]:
    required = (
        "id",
        "question",
        "expected",
        "category",
        "kind",
        "language",
        "domain",
        "group_id",
        "split",
        "rationale",
    )
    missing_fields = sum(
        any(field not in row or row[field] in (None, "") for field in required)
        for row in rows
    )
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[str(row["group_id"])].add(str(row["split"]))
    crossing_groups = sum(len(splits) > 1 for splits in group_splits.values())

    paper_groups: dict[str, int] = Counter(
        str(row["group_id"])
        for row in rows
        if row["category"] == "in_scope"
    )
    paper_group_sizes_bad = sum(count != 2 for count in paper_groups.values())

    boundary_labels: dict[str, set[str]] = defaultdict(set)
    boundary_sizes: Counter[str] = Counter()
    for row in rows:
        if row["category"] == "boundary":
            boundary_labels[str(row["domain"])].add(str(row["expected"]))
            boundary_sizes[str(row["domain"])] += 1
    boundary_both = sum(len(labels) == 2 for labels in boundary_labels.values())

    category_label_bad = sum(
        (
            row["category"] == "in_scope"
            and row["expected"] != "in_scope"
        )
        or (
            row["category"] == "ood"
            and row["expected"] != "out_of_scope"
        )
        or row["expected"] not in LABELS
        for row in rows
    )
    external = {
        normalize_question(str(row["question"]))
        for row in read_jsonl(external_path)
    }
    external_overlap = sum(
        normalize_question(str(row["question"])) in external
        for row in rows
    )

    def grams(value: str) -> set[str]:
        normalized = normalize_question(value)
        n = 3
        if len(normalized) <= n:
            return {normalized}
        return {normalized[index:index + n] for index in range(len(normalized) - n + 1)}

    split_grams = {
        split: [
            (row["id"], grams(str(row["question"])))
            for row in rows
            if row["split"] == split
        ]
        for split in ("train", "calibration", "internal_test")
    }

    def similar(left: set[str], right: set[str]) -> float:
        union = left | right
        return len(left & right) / len(union) if union else 1.0

    same_split_pairs = 0
    same_split_examples: list[dict[str, Any]] = []
    for split_rows in split_grams.values():
        for left_index, (left_id, left_grams) in enumerate(split_rows):
            for right_id, right_grams in split_rows[left_index + 1:]:
                score = similar(left_grams, right_grams)
                if score >= 0.90:
                    same_split_pairs += 1
                    if len(same_split_examples) < 10:
                        same_split_examples.append(
                            {"left": left_id, "right": right_id, "similarity": round(score, 4)}
                        )

    cross_split_pairs = 0
    train_rows = split_grams["train"]
    for split in ("calibration", "internal_test"):
        for test_id, test_grams in split_grams[split]:
            for train_id, train_grams in train_rows:
                if similar(test_grams, train_grams) >= 0.90:
                    cross_split_pairs += 1

    lengths = [len(str(row["question"])) for row in rows]
    confidences = [
        float(row["silver_confidence"])
        for row in rows
        if row.get("silver_confidence") is not None
    ]
    checks = {
        "required_fields": {"passed": missing_fields == 0, "failures": missing_fields},
        "group_split_isolation": {
            "passed": crossing_groups == 0,
            "crossing_groups": crossing_groups,
        },
        "paper_group_size": {
            "passed": paper_group_sizes_bad == 0,
            "bad_groups": paper_group_sizes_bad,
        },
        "boundary_both_labels": {
            "passed": boundary_both == len(boundary_labels),
            "domains_with_both": boundary_both,
            "domains": len(boundary_labels),
        },
        "category_label_consistency": {
            "passed": category_label_bad == 0,
            "failures": category_label_bad,
        },
        "external_exact_overlap": {
            "passed": external_overlap == 0,
            "overlaps": external_overlap,
        },
    }
    return {
        "generated_at": utc_now(),
        "items": len(rows),
        "checks": checks,
        "near_duplicates": {
            "threshold": 0.90,
            "same_split_pairs": same_split_pairs,
            "cross_split_pairs": cross_split_pairs,
            "same_split_examples": same_split_examples,
        },
        "question_length": {
            "min": min(lengths) if lengths else None,
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else None,
            "max": max(lengths) if lengths else None,
        },
        "silver_confidence": {
            "min": round(min(confidences), 4) if confidences else None,
            "mean": round(sum(confidences) / len(confidences), 4)
            if confidences
            else None,
            "max": round(max(confidences), 4) if confidences else None,
        },
        "passed": all(check["passed"] for check in checks.values()),
    }


def build_manifest(
    *,
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    dataset_dir: Path,
    external_path: Path,
    seed: int,
    generation_args: dict[str, Any],
) -> dict[str, Any]:
    files = (
        "all.jsonl",
        "train.jsonl",
        "calibration.jsonl",
        "internal_test.jsonl",
        "laya_train.jsonl",
        "laya_calibration.jsonl",
        "review.csv",
        "quality_report.json",
    )
    counts = {
        "total": len(rows),
        "split": dict(Counter(row["split"] for row in rows)),
        "label": dict(Counter(row["expected"] for row in rows)),
        "category": dict(Counter(row["category"] for row in rows)),
        "language": dict(Counter(row["language"] for row in rows)),
        "kind": dict(Counter(row["kind"] for row in rows)),
        "groups": {
            "total": len({row["group_id"] for row in rows}),
            "by_split": {
                split: len(
                    {
                        row["group_id"]
                        for row in rows
                        if row["split"] == split
                    }
                )
                for split in ("train", "calibration", "internal_test")
            },
        },
    }
    return {
        "dataset_version": DATASET_VERSION,
        "generated_at": utc_now(),
        "seed": seed,
        "generator": {
            "provider": "mobile_cloud_maas",
            "model": model_name(cfg),
            "api_key_env": str((cfg.get("llm") or {}).get("api_key_env") or ""),
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": {
                "in_scope": sha256_text(IN_SCOPE_INSTRUCTIONS),
                "ood": sha256_text(OOD_INSTRUCTIONS),
                "boundary": sha256_text(BOUNDARY_INSTRUCTIONS),
            },
            "generation_args": generation_args,
        },
        "counts": counts,
        "external_test": {
            "path": str(external_path.relative_to(ROOT)),
            "sha256": sha256_file(external_path),
            "training_allowed": False,
        },
        "quality_policy": {
            "label_source": "llm_generated_silver",
            "gold": False,
            "group_split": True,
            "exact_duplicate_filter": True,
            "external_test_overlap_filter": True,
            "human_review": "pending in review.csv",
            "threshold_selection_allowed": False,
        },
        "files": {
            name: {
                "sha256": sha256_file(dataset_dir / name),
                "bytes": (dataset_dir / name).stat().st_size,
            }
            for name in files
        },
    }


def make_specs(
    *,
    papers: list[Paper],
    paper_batch_size: int,
    ood_domains: list[str],
    ood_batch_size: int,
    ood_per_domain: int,
    boundary_domains: list[str],
    boundary_batch_size: int,
    boundary_per_domain: int,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for index, batch in enumerate(chunked(papers, paper_batch_size), start=1):
        payload = {
            "papers": [
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "abstract": paper.abstract[:1000],
                    "subfields": list(paper.subfields),
                    "venue": paper.venue,
                    "year": paper.year,
                }
                for paper in batch
            ],
            "questions_per_paper": 2,
        }
        specs.append(
            {
                "kind": "in_scope",
                "label": f"papers-{index:02d}",
                "payload": payload,
                "output_type": InScopeBatch,
                "instructions": IN_SCOPE_INSTRUCTIONS,
                "validator": validate_in_scope_batch,
            }
        )
    for index, batch in enumerate(chunked(ood_domains, ood_batch_size), start=1):
        payload = {
            "domains": [
                {"domain": domain, "count": ood_per_domain}
                for domain in batch
            ]
        }
        specs.append(
            {
                "kind": "ood",
                "label": f"ood-{index:02d}",
                "payload": payload,
                "output_type": OodBatch,
                "instructions": OOD_INSTRUCTIONS,
                "validator": validate_ood_batch,
            }
        )
    for index, batch in enumerate(
        chunked(boundary_domains, boundary_batch_size),
        start=1,
    ):
        payload = {
            "domains": [
                {"domain": domain, "count": boundary_per_domain}
                for domain in batch
            ]
        }
        specs.append(
            {
                "kind": "boundary",
                "label": f"boundary-{index:02d}",
                "payload": payload,
                "output_type": BoundaryBatch,
                "instructions": BOUNDARY_INSTRUCTIONS,
                "validator": validate_boundary_batch,
            }
        )
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "eval" / "scope_v1" / "dataset",
    )
    parser.add_argument(
        "--external-test",
        type=Path,
        default=ROOT / "eval" / "scope_v1" / "test.jsonl",
    )
    parser.add_argument("--paper-count", type=int, default=200)
    parser.add_argument("--paper-batch-size", type=int, default=10)
    parser.add_argument("--ood-count", type=int, default=32)
    parser.add_argument("--ood-batch-size", type=int, default=4)
    parser.add_argument("--ood-per-domain", type=int, default=10)
    parser.add_argument("--boundary-count", type=int, default=16)
    parser.add_argument("--boundary-batch-size", type=int, default=2)
    parser.add_argument("--boundary-per-domain", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--split-missing-batches", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.paper_count > len(OOD_DOMAINS) * args.ood_per_domain * 100:
        raise ValueError("paper count is implausibly large")
    if args.ood_count > len(OOD_DOMAINS):
        raise ValueError(f"ood-count cannot exceed {len(OOD_DOMAINS)}")
    if args.boundary_count > len(BOUNDARY_DOMAINS):
        raise ValueError(
            f"boundary-count cannot exceed {len(BOUNDARY_DOMAINS)}"
        )
    if args.paper_count % args.paper_batch_size:
        raise ValueError("paper-count must be divisible by paper-batch-size")
    if args.ood_count % args.ood_batch_size:
        raise ValueError("ood-count must be divisible by ood-batch-size")
    if args.boundary_count % args.boundary_batch_size:
        raise ValueError(
            "boundary-count must be divisible by boundary-batch-size"
        )

    cfg = load_config()
    _ = api_key(cfg)  # fail before loading the corpus; never print the key
    corpus = open_corpus(cfg)
    papers = sample_papers(corpus, args.paper_count, args.seed)
    ood_domains = list(OOD_DOMAINS[:args.ood_count])
    boundary_domains = list(BOUNDARY_DOMAINS[:args.boundary_count])
    specs = make_specs(
        papers=papers,
        paper_batch_size=args.paper_batch_size,
        ood_domains=ood_domains,
        ood_batch_size=args.ood_batch_size,
        ood_per_domain=args.ood_per_domain,
        boundary_domains=boundary_domains,
        boundary_batch_size=args.boundary_batch_size,
        boundary_per_domain=args.boundary_per_domain,
    )
    print(
        f"planned batches={len(specs)} papers={len(papers)} "
        f"ood_domains={len(ood_domains)} boundary_domains={len(boundary_domains)}"
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "first_paper": {
                        "paper_id": papers[0].paper_id,
                        "title": papers[0].title,
                    },
                    "ood_domains": ood_domains,
                    "boundary_domains": boundary_domains,
                    "expected_items": (
                        len(papers) * 2
                        + len(ood_domains) * args.ood_per_domain
                        + len(boundary_domains) * args.boundary_per_domain
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "generation_cache.jsonl"
    cache = load_cache(cache_path)
    if not args.skip_generation:
        asyncio.run(
            generate_batches(
                cfg=cfg,
                specs=specs,
                cache=cache,
                cache_path=cache_path,
                concurrency=args.concurrency,
                timeout=args.timeout,
                seed=args.seed,
                split_missing_batches=args.split_missing_batches,
            )
        )

    selected_keys = {
        cache_key(spec["kind"], spec["payload"], model_name(cfg), args.seed)
        for spec in specs
    }
    missing = sorted(selected_keys - set(cache))
    if missing:
        raise RuntimeError(
            f"generation cache is incomplete: missing {len(missing)} batches"
        )
    generated_rows = [
        row
        for key in sorted(selected_keys)
        for row in cache[key]["rows"]
    ]
    rows = build_rows(
        generated_rows,
        seed=args.seed,
        external_path=args.external_test,
    )
    expected_total = (
        args.paper_count * 2
        + args.ood_count * args.ood_per_domain
        + args.boundary_count * args.boundary_per_domain
    )
    if len(rows) != expected_total:
        raise RuntimeError(
            f"deduplication removed items: expected={expected_total}, got={len(rows)}"
        )

    by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "calibration", "internal_test")
    }
    write_jsonl(args.out_dir / "all.jsonl", rows)
    for split, split_rows in by_split.items():
        write_jsonl(args.out_dir / f"{split}.jsonl", split_rows)
        write_jsonl(
            args.out_dir / f"laya_{split}.jsonl",
            laya_rows(split_rows),
        )
    write_review_csv(args.out_dir / "review.csv", rows, args.seed)
    quality_report = build_quality_report(
        rows,
        external_path=args.external_test,
    )
    (args.out_dir / "quality_report.json").write_text(
        json.dumps(quality_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = build_manifest(
        rows=rows,
        cfg=cfg,
        dataset_dir=args.out_dir,
        external_path=args.external_test,
        seed=args.seed,
        generation_args={
            "paper_count": args.paper_count,
            "paper_batch_size": args.paper_batch_size,
            "ood_count": args.ood_count,
            "ood_batch_size": args.ood_batch_size,
            "ood_per_domain": args.ood_per_domain,
            "boundary_count": args.boundary_count,
            "boundary_batch_size": args.boundary_batch_size,
            "boundary_per_domain": args.boundary_per_domain,
            "concurrency": args.concurrency,
        },
    )
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "total": len(rows),
                "split": manifest["counts"]["split"],
                "label": manifest["counts"]["label"],
                "category": manifest["counts"]["category"],
                "language": manifest["counts"]["language"],
                "groups": manifest["counts"]["groups"],
                "quality_passed": quality_report["passed"],
                "near_duplicates": quality_report["near_duplicates"],
                "review_rows": sum(
                    1 for _ in (args.out_dir / "review.csv").open(encoding="utf-8-sig")
                )
                - 1,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
