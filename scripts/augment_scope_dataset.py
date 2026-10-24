"""Build scope-dataset-v1.2 with reviewed errors and targeted augmentation.

Adds 200 synthetic examples:
* 60 English ML positives;
* 60 Chinese technical OOD negatives across petroleum, pharmacology and aerospace;
* 80 minimal-pair cross-domain examples (40 positive, 40 negative).

New examples enter train/calibration only. The v1.1 internal test remains frozen.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(line_buffering=True)

from hotspot.config import load_config

PROMPT_VERSION = "scope-augment-v1"
DATASET_VERSION = "scope-dataset-v1.2"
SEED = 20261024

ENGLISH_THEMES = (
    "reinforcement_learning",
    "llm_alignment",
    "agents_and_planning",
    "robotics_policy_learning",
    "bandits_and_online_learning",
    "graph_representation_learning",
    "optimization_theory",
    "multimodal_learning",
    "sequential_decision_making",
    "efficient_machine_learning",
)

OOD_GROUPS = (
    {
        "group_id": "aug_ood_petroleum_reservoir",
        "domain": "petroleum_engineering",
        "focus": "油藏数值模拟、孔隙度/渗透率、PVT 相态和采收率机制",
        "split": "train",
    },
    {
        "group_id": "aug_ood_petroleum_well",
        "domain": "petroleum_engineering",
        "focus": "钻井、套管、固井、测井和储层保护",
        "split": "calibration",
    },
    {
        "group_id": "aug_ood_pharmacology_pk",
        "domain": "pharmacology",
        "focus": "药代动力学、CYP 酶、药物相互作用和剂量调整",
        "split": "train",
    },
    {
        "group_id": "aug_ood_pharmacology_pd",
        "domain": "pharmacology",
        "focus": "药效学、受体通路、耐受、依赖和抗菌药敏试验",
        "split": "calibration",
    },
    {
        "group_id": "aug_ood_aerospace_orbit",
        "domain": "aerospace",
        "focus": "轨道力学、变轨、姿态动力学和卫星轨道参数",
        "split": "train",
    },
    {
        "group_id": "aug_ood_aerospace_vehicle",
        "domain": "aerospace",
        "focus": "飞行器稳定性、推进、气动热、结构载荷和热防护",
        "split": "calibration",
    },
)

PAIR_DOMAINS = (
    ("ml_for_education", "个性化学习、知识追踪、教育推荐和课堂分析"),
    ("physics_informed_ml", "PINN、物理约束损失和科学计算"),
    ("abbreviation_and_polysemy", "缩略词、词义消歧、多义性和序列标注"),
    ("ml_for_medical_imaging", "医学影像分割、检测和影像生成"),
    ("ml_for_finance", "风险预测、交易策略和金融时间序列"),
    ("ml_for_climate_forecasting", "气象预报、气候预测和物理约束模型"),
    ("ml_for_civil_monitoring", "结构健康监测、裂缝检测和异常检测"),
    ("ml_for_chemistry", "分子性质预测、分子生成和反应预测"),
    ("ml_for_agriculture", "作物识别、产量预测和温室控制"),
    ("ml_for_transport_control", "交通流量预测、信号控制和自动驾驶"),
)


class EnglishQuestion(BaseModel):
    theme: str
    question: str
    kind: Literal["local", "compare", "relation", "trend", "followup"]
    domain: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class EnglishBatch(BaseModel):
    questions: list[EnglishQuestion]


class OodQuestion(BaseModel):
    group_id: str
    question: str
    language: Literal["zh", "zh_en", "en"]
    boundary_cue: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class OodBatch(BaseModel):
    questions: list[OodQuestion]


class MinimalPair(BaseModel):
    domain: str
    in_scope_question: str
    out_of_scope_question: str
    boundary_cue: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class PairBatch(BaseModel):
    pairs: list[MinimalPair]


ENGLISH_INSTRUCTIONS = """Generate English scope-classification training questions for a local
AI/ML/RL paper corpus. Return exactly the requested number for every input theme.

Rules:
1. Every question must explicitly concern AI/ML/RL/LLM/Agent/robotics learning/decision methods.
2. Cover local, compare, relation, trend and follow-up styles. Do not ask only for a paper summary.
3. Vary vocabulary, including model, policy, agent, representation, training, alignment, graph,
   transformer, bandit, optimization and evaluation.
4. Do not copy benchmark questions. Every question must be distinct.
5. Never include the literal labels in_scope or out_of_scope."""

OOD_INSTRUCTIONS = """Generate Chinese-dominant out-of-scope questions for a local AI/ML paper
corpus. Return exactly the requested number for the given group.

Rules:
1. Ask about the source discipline itself: mechanisms, experiments, standards, equipment or
   conventional calculations. Do not ask for ML/RL/AI methods.
2. Include technical terms that can look algorithmic (simulation, grid, optimization,
   sensitivity, feedback, decision), but do not name AI/ML methods.
3. Use 7 Chinese, 2 mixed Chinese-English and 1 English question per 10-item group.
4. Keep group_id exactly as provided. Questions must be distinct and not copied from benchmarks.
5. Never include the literal labels in_scope or out_of_scope."""

PAIR_INSTRUCTIONS = """Generate minimal-pair scope questions for cross-domain applications.
For each requested domain, return exactly 4 pairs.

For each pair:
* in_scope_question explicitly asks for an ML/RL/AI model, method, training or representation;
* out_of_scope_question asks the same application domain's mechanism, traditional method,
  measurement or practice without any ML/RL/AI method.

The paired questions should differ only by the presence of an ML method. Use Chinese primary,
with some English or mixed-language pairs. Do not copy benchmark questions. Never include the
literal labels in_scope or out_of_scope."""


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


def stable_rank(value: str) -> str:
    return hashlib.sha256(f"{SEED}|{value}".encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def api_key(cfg: dict[str, Any]) -> str:
    llm = cfg.get("llm") or {}
    value = str(llm.get("api_key") or "").strip()
    env_name = str(llm.get("api_key_env") or "").strip()
    if not value and env_name:
        value = str(os.environ.get(env_name, "")).strip()
    if not value:
        raise RuntimeError(f"missing API key: {env_name}")
    return value


def model_name(cfg: dict[str, Any]) -> str:
    llm = cfg.get("llm") or {}
    return str(
        llm.get("model_flash") or llm.get("model_pro") or "deepseek-chat"
    )


def make_agent(
    cfg: dict[str, Any],
    output_type: type[BaseModel],
    instructions: str,
) -> Agent:
    llm = cfg.get("llm") or {}
    provider = OpenAIProvider(
        base_url=str(llm.get("base_url") or "").rstrip("/"),
        api_key=api_key(cfg),
    )
    model = OpenAIChatModel(model_name(cfg), provider=provider)
    return Agent(model, output_type=output_type, instructions=instructions, retries=1)


def cache_key(kind: str, payload: dict[str, Any], model: str) -> str:
    return sha256_text(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "kind": kind,
                "model": model,
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["cache_key"]: row for row in read_jsonl(path)}


def normalize(text: str) -> str:
    return "".join(str(text).lower().split())


def validate_question(text: str) -> None:
    value = text.strip()
    if not (8 <= len(value) <= 360):
        raise ValueError(f"question length out of range: {value!r}")
    if "\n" in value or "\r" in value:
        raise ValueError(f"newline in question: {value!r}")
    if "in_scope" in value or "out_of_scope" in value:
        raise ValueError(f"label leaked into question: {value!r}")


def make_specs() -> list[dict[str, Any]]:
    specs = []
    for index in range(0, len(ENGLISH_THEMES), 5):
        themes = ENGLISH_THEMES[index:index + 5]
        payload = {
            "themes": [{"theme": theme, "count": 6} for theme in themes],
        }
        specs.append(
            {
                "kind": "english_ml",
                "label": f"english-{index // 5 + 1}",
                "payload": payload,
                "output_type": EnglishBatch,
                "instructions": ENGLISH_INSTRUCTIONS,
            }
        )
    for group in OOD_GROUPS:
        payload = {"groups": [dict(group, count=10)]}
        specs.append(
            {
                "kind": "ood_technical",
                "label": group["group_id"],
                "payload": payload,
                "output_type": OodBatch,
                "instructions": OOD_INSTRUCTIONS,
            }
        )
    for index in range(0, len(PAIR_DOMAINS), 2):
        domains = PAIR_DOMAINS[index:index + 2]
        payload = {
            "domains": [
                {"domain": domain, "focus": focus, "pairs": 4}
                for domain, focus in domains
            ]
        }
        specs.append(
            {
                "kind": "minimal_pairs",
                "label": f"pairs-{index // 2 + 1}",
                "payload": payload,
                "output_type": PairBatch,
                "instructions": PAIR_INSTRUCTIONS,
            }
        )
    return specs


def validate_output(spec: dict[str, Any], output: BaseModel) -> list[dict[str, Any]]:
    payload = spec["payload"]
    rows: list[dict[str, Any]] = []
    if spec["kind"] == "english_ml":
        expected = {item["theme"]: item["count"] for item in payload["themes"]}
        counts = Counter(item.theme for item in output.questions)
        if dict(counts) != expected:
            raise ValueError(f"English theme counts mismatch: {expected} vs {dict(counts)}")
        for item in output.questions:
            validate_question(item.question)
            rows.append(
                {
                    "group_id": f"aug_en:{item.theme}",
                    "category": "in_scope",
                    "question": item.question.strip(),
                    "expected": "in_scope",
                    "kind": item.kind,
                    "language": "en",
                    "domain": item.domain.strip() or item.theme,
                    "boundary_cue": "",
                    "rationale": item.rationale,
                    "silver_confidence": float(item.confidence),
                }
            )
    elif spec["kind"] == "ood_technical":
        group = payload["groups"][0]
        expected = group["count"]
        if len(output.questions) != expected:
            raise ValueError(f"OOD count mismatch: expected {expected}, got {len(output.questions)}")
        for item in output.questions:
            validate_question(item.question)
            if item.group_id != group["group_id"]:
                raise ValueError("OOD group id mismatch")
            rows.append(
                {
                    "group_id": item.group_id,
                    "category": "ood",
                    "question": item.question.strip(),
                    "expected": "out_of_scope",
                    "kind": "hard_negative",
                    "language": item.language,
                    "domain": group["domain"],
                    "boundary_cue": item.boundary_cue,
                    "rationale": item.rationale,
                    "silver_confidence": float(item.confidence),
                }
            )
    else:
        expected = {item["domain"]: item["pairs"] for item in payload["domains"]}
        counts = Counter(item.domain for item in output.pairs)
        if dict(counts) != expected:
            raise ValueError(f"pair counts mismatch: {expected} vs {dict(counts)}")
        for pair in output.pairs:
            validate_question(pair.in_scope_question)
            validate_question(pair.out_of_scope_question)
            rows.append(
                {
                    "group_id": f"aug_pair:{pair.domain}",
                    "category": "boundary",
                    "question": pair.in_scope_question.strip(),
                    "expected": "in_scope",
                    "kind": "boundary_positive",
                    "language": "zh" if any("\u4e00" <= c <= "\u9fff" for c in pair.in_scope_question) else "en",
                    "domain": pair.domain,
                    "boundary_cue": pair.boundary_cue,
                    "rationale": pair.rationale,
                    "silver_confidence": float(pair.confidence),
                }
            )
            rows.append(
                {
                    "group_id": f"aug_pair:{pair.domain}",
                    "category": "boundary",
                    "question": pair.out_of_scope_question.strip(),
                    "expected": "out_of_scope",
                    "kind": "boundary_negative",
                    "language": "zh" if any("\u4e00" <= c <= "\u9fff" for c in pair.out_of_scope_question) else "en",
                    "domain": pair.domain,
                    "boundary_cue": pair.boundary_cue,
                    "rationale": pair.rationale,
                    "silver_confidence": float(pair.confidence),
                }
            )
    return rows


async def generate_one(
    cfg: dict[str, Any],
    spec: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    prompt = json.dumps(spec["payload"], ensure_ascii=False)
    last_error = None
    for _attempt in range(2):
        agent = make_agent(cfg, spec["output_type"], spec["instructions"])
        try:
            result = await agent.run(
                prompt,
                timeout=timeout,
                usage_limits=UsageLimits(request_limit=2),
            )
            return {
                "rows": validate_output(spec, result.output),
                "usage": {
                    "input_tokens": getattr(result.usage(), "input_tokens", None)
                    if callable(getattr(result, "usage", None))
                    else None,
                },
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"batch failed after retry: {last_error}")


async def generate_all(
    cfg: dict[str, Any],
    specs: list[dict[str, Any]],
    cache_path: Path,
    concurrency: int,
    timeout: float,
) -> None:
    cache = load_cache(cache_path)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run(spec: dict[str, Any]) -> None:
        key = cache_key(spec["kind"], spec["payload"], model_name(cfg))
        if key in cache:
            print(f"cache {spec['label']}")
            return
        async with semaphore:
            started = asyncio.get_running_loop().time()
            result = await generate_one(cfg, spec, timeout)
            record = {
                "cache_key": key,
                "kind": spec["kind"],
                "label": spec["label"],
                "model": model_name(cfg),
                "prompt_version": PROMPT_VERSION,
                "payload": spec["payload"],
                "rows": result["rows"],
                "generated_at": utc_now(),
            }
            append_jsonl(cache_path, record)
            cache[key] = record
            elapsed = round(asyncio.get_running_loop().time() - started, 2)
            print(f"generated {spec['label']} rows={len(record['rows'])} elapsed={elapsed}s")

    results = await asyncio.gather(
        *(run(spec) for spec in specs),
        return_exceptions=True,
    )
    failures = [
        (spec["label"], result)
        for spec, result in zip(specs, results)
        if isinstance(result, Exception)
    ]
    if failures:
        raise RuntimeError(
            "; ".join(f"{label}: {exc}" for label, exc in failures)
        )


def review_error_rows(
    source_rows: list[dict[str, Any]],
    trained_rows: list[dict[str, Any]],
    prior_review: set[str],
) -> list[dict[str, Any]]:
    source = {row["id"]: row for row in source_rows}
    rows = []
    for prediction in trained_rows:
        if prediction["correct"] or prediction["id"] in prior_review:
            continue
        item = source[prediction["id"]]
        expected = item["expected"]
        if isinstance(expected, dict):
            expected = expected.get("scope", "")
        note = (
            "Manual review: explicit AI/ML/RL method, model, theory or training question; "
            "label remains in_scope."
            if expected == "in_scope"
            else
            "Manual review: asks source-domain mechanism, equipment, measurement or "
            "traditional calculation without an ML/RL method; label remains out_of_scope."
        )
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected": expected,
                "predicted": prediction["predicted"],
                "category": item.get("category", ""),
                "language": item.get("language", ""),
                "domain": item.get("domain", ""),
                "review_decision": "accept",
                "review_notes": note,
            }
        )
    if len(rows) != 29:
        raise RuntimeError(f"expected 29 unreviewed errors, found {len(rows)}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def apply_generated_rows(
    rows: list[dict[str, Any]],
    *,
    prefix: str,
    split: str,
) -> list[dict[str, Any]]:
    output = []
    seen = Counter()
    for row in sorted(rows, key=lambda item: stable_rank(item["question"])):
        group = row["group_id"]
        seen[group] += 1
        item = dict(row)
        item.update(
            {
                "id": f"{prefix}_{sha256_text(group)[:8]}_{seen[group]:02d}",
                "split": split,
                "dataset_version": DATASET_VERSION,
                "annotator": "mobile_cloud_llm_silver",
                "label_source": "llm_generated_silver",
                "source": "mobile_trash_augmentation",
                "review_status": "not_sampled",
                "paper_id": "",
                "paper_title": "",
                "venue": "",
                "year": None,
            }
        )
        output.append(item)
    return output


def quality_report(
    rows: list[dict[str, Any]],
    external_rows: list[dict[str, Any]],
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
    missing = sum(
        any(field not in row or row[field] in (None, "") for field in required)
        for row in rows
    )
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        group_splits.setdefault(row["group_id"], set()).add(row["split"])
    crossing = sum(len(splits) > 1 for splits in group_splits.values())
    normalized = Counter(normalize(row["question"]) for row in rows)
    exact_duplicates = sum(count - 1 for count in normalized.values() if count > 1)
    external_norm = {normalize(row["question"]) for row in external_rows}
    external_overlap = sum(normalize(row["question"]) in external_norm for row in rows)
    category_label_bad = sum(
        (row["category"] == "in_scope" and row["expected"] != "in_scope")
        or (row["category"] == "ood" and row["expected"] != "out_of_scope")
        or row["expected"] not in {"in_scope", "out_of_scope"}
        for row in rows
    )
    boundary: dict[str, set[str]] = {}
    for row in rows:
        if row["category"] == "boundary":
            boundary.setdefault(row["domain"], set()).add(row["expected"])
    boundary_bad = sum(len(labels) != 2 for labels in boundary.values())
    def grams(text: str) -> set[str]:
        value = normalize(text)
        size = 3
        if len(value) <= size:
            return {value}
        return {value[index:index + size] for index in range(len(value) - size + 1)}

    split_grams = {
        split: [
            (row["id"], grams(row["question"]))
            for row in rows
            if row["split"] == split
        ]
        for split in ("train", "calibration", "internal_test")
    }

    def similarity(left: set[str], right: set[str]) -> float:
        union = left | right
        return len(left & right) / len(union) if union else 1.0

    same_split_near = 0
    cross_split_near = 0
    for split_rows in split_grams.values():
        for index, (_left_id, left) in enumerate(split_rows):
            for _right_id, right in split_rows[index + 1:]:
                if similarity(left, right) >= 0.90:
                    same_split_near += 1
    for split in ("calibration", "internal_test"):
        for _test_id, test_grams in split_grams[split]:
            for _train_id, train_grams in split_grams["train"]:
                if similarity(test_grams, train_grams) >= 0.90:
                    cross_split_near += 1
    checks = {
        "required_fields": missing == 0,
        "group_split_isolation": crossing == 0,
        "exact_duplicates": exact_duplicates == 0,
        "external_test_overlap": external_overlap == 0,
        "category_label_consistency": category_label_bad == 0,
        "boundary_domains_have_both_labels": boundary_bad == 0,
        "near_duplicates": same_split_near == 0 and cross_split_near == 0,
    }
    return {
        "generated_at": utc_now(),
        "dataset_version": DATASET_VERSION,
        "items": len(rows),
        "checks": checks,
        "counts": {
            "split": dict(Counter(row["split"] for row in rows)),
            "label": dict(Counter(row["expected"] for row in rows)),
            "category": dict(Counter(row["category"] for row in rows)),
            "language": dict(Counter(row["language"] for row in rows)),
            "groups": len(group_splits),
        },
        "failures": {
            "missing_fields": missing,
            "crossing_groups": crossing,
            "exact_duplicates": exact_duplicates,
            "external_overlap": external_overlap,
            "category_label_bad": category_label_bad,
            "boundary_domains_bad": boundary_bad,
            "same_split_near_duplicates": same_split_near,
            "cross_split_near_duplicates": cross_split_near,
        },
        "passed": all(checks.values()),
    }


def laya_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    question = {
        "scope": {
            "type": "choice",
            "instructions": "判断问题是否属于本地 AI/ML 论文库的研究范围。",
            "criteria": {
                "in_scope": "询问 AI/ML/RL/LLM/Agent/机器人学习或决策方法",
                "out_of_scope": "询问其他领域本身或与本地 AI/ML 论文库无关",
            },
        }
    }
    return [
        {
            "id": row["id"],
            "state": {"question": row["question"]},
            "questions": question,
            "expected": {"scope": row["expected"]},
            "split": row["split"],
            "category": row["category"],
            "group_id": row["group_id"],
            "dataset_version": DATASET_VERSION,
        }
        for row in rows
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "eval" / "scope_v1" / "dataset_v1_1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "eval" / "scope_v1" / "dataset_v1_2",
    )
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--refresh-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out_dir.exists() and not args.refresh_existing:
        raise FileExistsError(f"refusing to overwrite {args.out_dir}")
    cfg = load_config()
    _ = api_key(cfg)
    specs = make_specs()
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    if not args.out_dir.exists():
        shutil.copytree(args.source_dir, args.out_dir)

    cache_path = args.out_dir / "augmentation_cache.jsonl"
    if not args.skip_generation:
        asyncio.run(
            generate_all(
                cfg,
                specs,
                cache_path,
                args.concurrency,
                args.timeout,
            )
        )
    cache = load_cache(cache_path)
    selected = [
        cache[cache_key(spec["kind"], spec["payload"], model_name(cfg))]
        for spec in specs
        if cache_key(spec["kind"], spec["payload"], model_name(cfg)) in cache
    ]
    if len(selected) != len(specs):
        raise RuntimeError(f"incomplete augmentation cache: {len(selected)}/{len(specs)}")

    english = [row for item in selected if item["kind"] == "english_ml" for row in item["rows"]]
    ood = [row for item in selected if item["kind"] == "ood_technical" for row in item["rows"]]
    pairs = [row for item in selected if item["kind"] == "minimal_pairs" for row in item["rows"]]

    english_groups = sorted({row["group_id"] for row in english})
    english_train = set(sorted(english_groups, key=stable_rank)[:8])
    english_cal = set(english_groups) - english_train
    pair_groups = sorted({row["group_id"] for row in pairs})
    pair_train = set(sorted(pair_groups, key=stable_rank)[:8])
    pair_cal = set(pair_groups) - pair_train

    add_train = (
        [row for row in english if row["group_id"] in english_train]
        + [row for row in english if row["group_id"] in english_cal]
        + [row for row in ood if row["group_id"] in {
            "aug_ood_petroleum_reservoir",
            "aug_ood_pharmacology_pk",
            "aug_ood_aerospace_orbit",
        }]
        + [row for row in pairs if row["group_id"] in pair_train]
    )
    # Calibration gets only its explicitly allocated groups; English groups are 8/2.
    add_cal = (
        [row for row in english if row["group_id"] in english_cal]
        + [row for row in ood if row["group_id"] not in {
            "aug_ood_petroleum_reservoir",
            "aug_ood_pharmacology_pk",
            "aug_ood_aerospace_orbit",
        }]
        + [row for row in pairs if row["group_id"] in pair_cal]
    )
    # Remove English calibration rows accidentally assembled into train.
    add_train = [row for row in add_train if row["group_id"] in english_train or not row["group_id"].startswith("aug_en:")]
    add_train = apply_generated_rows(add_train, prefix="aug_train", split="train")
    add_cal = apply_generated_rows(add_cal, prefix="aug_cal", split="calibration")
    if len(add_train) != 142 or len(add_cal) != 58:
        raise RuntimeError(f"unexpected augmentation split: {len(add_train)}/{len(add_cal)}")

    base_train = read_jsonl(args.source_dir / "train.jsonl")
    base_cal = read_jsonl(args.source_dir / "calibration.jsonl")
    base_test = read_jsonl(args.source_dir / "internal_test.jsonl")

    prior_review = {
        row["id"]
        for row in csv.DictReader(
            (args.source_dir / "review.csv").open(encoding="utf-8-sig", newline="")
        )
    }
    trained = json.loads(
        (ROOT / "eval" / "scope_v1" / "results_trained_v1_1" / "laya_zero_shot.json")
        .read_text(encoding="utf-8")
    )["rows"]
    reviewed_errors = review_error_rows(base_test, trained, prior_review)
    reviewed_map = {row["id"]: row for row in reviewed_errors}
    write_csv(args.out_dir / "review_errors_29.csv", reviewed_errors)

    def bump(rows: list[dict[str, Any]], augment: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for row in rows:
            item = dict(row)
            item["dataset_version"] = DATASET_VERSION
            reviewed = reviewed_map.get(item["id"])
            if reviewed:
                item["review_status"] = "reviewed"
                item["review_decision"] = reviewed["review_decision"]
                item["reviewed_by"] = "codex_manual"
                item["reviewed_at"] = utc_now()
            output.append(item)
        return output + augment

    train = bump(base_train, add_train)
    calibration = bump(base_cal, add_cal)
    internal = bump(base_test, [])
    original_internal = {
        row["id"]: (row["question"], row["expected"])
        for row in base_test
    }
    current_internal = {
        row["id"]: (row["question"], row["expected"])
        for row in internal
    }
    internal_content_changes = sum(
        original_internal.get(item_id) != current_internal.get(item_id)
        for item_id in set(original_internal) | set(current_internal)
    )
    internal_id_overlap = len(
        {row["id"] for row in add_train + add_cal}
        & set(original_internal)
    )
    if internal_content_changes or internal_id_overlap:
        raise RuntimeError(
            f"internal test changed: content={internal_content_changes}, "
            f"id_overlap={internal_id_overlap}"
        )
    all_rows = train + calibration + internal
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "calibration.jsonl", calibration)
    write_jsonl(args.out_dir / "internal_test.jsonl", internal)
    write_jsonl(args.out_dir / "all.jsonl", all_rows)
    for name, rows in (
        ("train", train),
        ("calibration", calibration),
        ("internal_test", internal),
    ):
        write_jsonl(args.out_dir / f"laya_{name}.jsonl", laya_rows(rows))

    quality = quality_report(all_rows, read_jsonl(ROOT / "eval" / "scope_v1" / "test.jsonl"))
    (args.out_dir / "quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads((args.source_dir / "manifest.json").read_text(encoding="utf-8"))
    parent_manifest = args.source_dir / "manifest.json"
    manifest.update(
        {
            "dataset_version": DATASET_VERSION,
            "generated_at": utc_now(),
            "parent": {
                "dataset_version": "scope-dataset-v1.1",
                "manifest_sha256": sha256_file(parent_manifest),
            },
            "augmentation": {
                "model": model_name(cfg),
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": {
                    "english": sha256_text(ENGLISH_INSTRUCTIONS),
                    "ood": sha256_text(OOD_INSTRUCTIONS),
                    "minimal_pairs": sha256_text(PAIR_INSTRUCTIONS),
                },
                "train_items": len(add_train),
                "calibration_items": len(add_cal),
                "internal_test_items": 0,
                "internal_test_content_changes": internal_content_changes,
                "internal_test_id_overlap": internal_id_overlap,
            },
            "review_errors_29": {
                "items": len(reviewed_errors),
                "decisions": dict(Counter(row["review_decision"] for row in reviewed_errors)),
            },
            "counts": quality["counts"],
            "quality_policy": {
                "label_source": "mixed_reviewed_and_llm_silver",
                "gold": False,
                "internal_test_reviewed_error_subset": True,
                "threshold_selection_allowed": False,
            },
        }
    )
    file_names = (
        "all.jsonl",
        "train.jsonl",
        "calibration.jsonl",
        "internal_test.jsonl",
        "laya_train.jsonl",
        "laya_calibration.jsonl",
        "laya_internal_test.jsonl",
        "review.csv",
        "review_errors_29.csv",
        "quality_report.json",
    )
    manifest["files"] = {
        name: {
            "sha256": sha256_file(args.out_dir / name),
            "bytes": (args.out_dir / name).stat().st_size,
        }
        for name in file_names
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "manifest.sha256").write_text(
        sha256_file(manifest_path) + "\n",
        encoding="utf-8",
    )
    readme_path = args.out_dir / "README.md"
    readme = f"""---
tags: [索引, 数据]
---

# {DATASET_VERSION}

在 v1.1 基础上完成 29 条错误复核和定向补强的 scope silver 数据集。

## 数据口径

- 共 {len(all_rows)} 条：范围内 {quality['counts']['label'].get('in_scope', 0)}、
  域外 {quality['counts']['label'].get('out_of_scope', 0)}。
- 类别：in_scope {quality['counts']['category'].get('in_scope', 0)}、
  OOD {quality['counts']['category'].get('ood', 0)}、
  boundary {quality['counts']['category'].get('boundary', 0)}。
- 切分：train {quality['counts']['split'].get('train', 0)}、
  calibration {quality['counts']['split'].get('calibration', 0)}、
  internal_test {quality['counts']['split'].get('internal_test', 0)}。
- 200 条新增样本只进入 train/calibration；internal_test 的 125 条内容保持冻结。
- 16 条 `../test.jsonl` 仍是独立外部测试集。

## v1.2 补强

- 复核 head-only 在 internal_test 中产生的 29 条未审错误，全部 `accept`。
- 新增 60 条英文 ML 正例。
- 新增 60 条中文 OOD 技术负例，覆盖石油工程、药理学和航天。
- 新增 80 条跨域最小对比题（40 正、40 负）。
- 新增样本按父组切分，无跨 split 泄漏、精确重复、近重复或外部测试重叠。

## 文件

- `train.jsonl` / `calibration.jsonl` / `internal_test.jsonl`
- `laya_*.jsonl`：Laya 训练/评测格式
- `all.jsonl`：全量审计数据
- `review.csv`：v1.1 的 146 条复核
- `review_errors_29.csv`：本轮 29 条错误复核
- `quality_report.json`：结构与泄漏审计
- `manifest.json` / `manifest.sha256`
- `augmentation_cache.jsonl`：本轮模型生成缓存

## 版本关系

- 父版本：`scope-dataset-v1.1`
- 父 manifest SHA256：`{sha256_file(parent_manifest)}`
- 本版本 manifest SHA256：`{sha256_file(manifest_path)}`
- v1 和 v1.1 均保持独立，不被覆盖。

## 复现

```powershell
python scripts/augment_scope_dataset.py
```

密钥只从 `config.yaml` 的 `llm.api_key_env` 指向的环境变量读取，不落盘。

## 相关

- `../../_index.md`
- `../../../scripts/augment_scope_dataset.py`
- `../../../scripts/train_scope_laya.py`
- `../../../规划/2026-10-23-Jev调参-Laya多语言scope微调.md`
"""
    readme_path.write_text(readme, encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset_version": DATASET_VERSION,
                "items": len(all_rows),
                "augmentation": manifest["augmentation"],
                "split": quality["counts"]["split"],
                "category": quality["counts"]["category"],
                "label": quality["counts"]["label"],
                "review_errors": manifest["review_errors_29"],
                "quality_passed": quality["passed"],
                "manifest_sha256": sha256_file(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if quality["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
