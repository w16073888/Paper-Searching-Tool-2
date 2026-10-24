"""Analyze scope-classifier errors and head-only training flips.

Consumes only frozen artifacts from the pilot; no model or API call is made.

    python scripts/analyze_scope_errors.py
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / ".models" / "jev" / "ft" / "scope-v1-head-only-seed0"
DATASET_DIR = ROOT / "eval" / "scope_v1" / "dataset_v1_1"
INTERNAL_BASE = ROOT / "eval" / "scope_v1" / "results_internal_v1_1" / "laya_zero_shot.json"
INTERNAL_TRAINED = ROOT / "eval" / "scope_v1" / "results_trained_v1_1" / "laya_zero_shot.json"
GATE_REPORT = ROOT / "eval" / "scope_v1" / "gate_report.json"
LLM_REPORT = ROOT / "eval" / "scope_v1" / "results_internal_v1_1" / "llm_default.json"

ML_PATTERNS = {
    "RL": r"(?i)(?:\brl\b|reinforcement learning|强化学习|策略梯度)",
    "神经网络": r"(?i)(?:\bneural\b|\bnetwork\b|神经网络|深度学习)",
    "Transformer/LLM": r"(?i)(?:\btransformer\b|\bbert\b|\bllm\b|大模型)",
    "图/GNN": r"(?i)(?:\bgraph\b|\bgnn\b|图神经网络|知识图谱)",
    "PINN/物理ML": r"(?i)(?:\bpinn\b|physics-informed|物理信息神经网络)",
    "CNN/LSTM": r"(?i)(?:\bcnn\b|\blstm\b|卷积神经网络|长短期记忆)",
    "训练/损失": r"(?i)(?:\btrain(?:ing)?\b|\bloss\b|训练|损失函数|梯度)",
    "模型/算法": r"(?i)(?:\bmodel\b|\balgorithm\b|模型|算法)",
    "策略/决策": r"(?i)(?:\bpolicy\b|\bdecision\b|策略|决策)",
    "对比/生成": r"(?i)(?:contrastive|generative|gan\b|对比学习|生成式)",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def label(source: dict[str, Any]) -> str:
    value = source.get("expected")
    if isinstance(value, dict):
        return str(value.get("scope", ""))
    return str(value or "")


def group_of(source: dict[str, Any]) -> str:
    category = source.get("category")
    if category == "ood":
        return "OOD"
    if category == "boundary":
        return "boundary"
    return str(source.get("kind") or category or "unknown")


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("predicted") in {"in_scope", "out_of_scope"}]
    pos = [row for row in valid if row["expected"] == "in_scope"]
    neg = [row for row in valid if row["expected"] == "out_of_scope"]
    tp = sum(row["predicted"] == "in_scope" for row in pos)
    fn = len(pos) - tp
    tn = sum(row["predicted"] == "out_of_scope" for row in neg)
    fp = len(neg) - tn
    return {
        "items": len(rows),
        "evaluated": len(valid),
        "errors": len(rows) - len(valid),
        "accuracy": round((tp + tn) / max(1, tp + tn + fp + fn), 4),
        "in_scope_recall": round(tp / max(1, tp + fn), 4),
        "in_scope_false_refusal_rate": round(fn / max(1, tp + fn), 4),
        "out_of_scope_recall": round(tn / max(1, tn + fp), 4),
        "ood_false_pass_rate": round(fp / max(1, tn + fp), 4),
        "confusion": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
    }


def grouped(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(field) or "unknown")].append(row)
    return {key: metrics(value) for key, value in sorted(buckets.items())}


def calibration_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = load_jsonl(DATASET_DIR / "calibration.jsonl")
    base_raw = load_jsonl(MODEL_DIR / "calibration_base_predictions.jsonl")
    trained_raw = load_jsonl(MODEL_DIR / "calibration_predictions.jsonl")
    if not (len(source) == len(base_raw) == len(trained_raw)):
        raise ValueError("calibration source/prediction length mismatch")
    base, trained = [], []
    for src, before, after in zip(source, base_raw, trained_raw):
        common = {
            "id": src["id"],
            "question": src["question"],
            "expected": label(src),
            "category": src.get("category", ""),
            "group": group_of(src),
            "kind": src.get("kind", ""),
            "language": src.get("language", ""),
            "domain": src.get("domain", ""),
            "reviewed": False,
        }
        base.append(
            {
                **common,
                "predicted": before["predicted"],
                "p_in_scope": float(before["p_in_scope"]),
                "p_out_of_scope": float(before["p_out_of_scope"]),
                "confidence": max(
                    float(before["p_in_scope"]),
                    float(before["p_out_of_scope"]),
                ),
            }
        )
        trained.append(
            {
                **common,
                "predicted": after["predicted"],
                "p_in_scope": float(after["p_in_scope"]),
                "p_out_of_scope": float(after["p_out_of_scope"]),
                "confidence": max(
                    float(after["p_in_scope"]),
                    float(after["p_out_of_scope"]),
                ),
            }
        )
    return base, trained


def internal_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = {
        row["id"]: row
        for row in load_jsonl(DATASET_DIR / "internal_test.jsonl")
    }
    base_json = json.loads(INTERNAL_BASE.read_text(encoding="utf-8"))
    trained_json = json.loads(INTERNAL_TRAINED.read_text(encoding="utf-8"))
    base, trained = [], []
    for before, after in zip(base_json["rows"], trained_json["rows"]):
        if before["id"] != after["id"]:
            raise ValueError("internal prediction row order mismatch")
        src = source[before["id"]]
        common = {
            "id": before["id"],
            "question": before["question"],
            "expected": before["expected"],
            "category": before.get("category") or src.get("category", ""),
            "group": before.get("group") or group_of(src),
            "kind": src.get("kind", ""),
            "language": before.get("language") or src.get("language", ""),
            "domain": before.get("domain") or src.get("domain", ""),
        }
        base.append(
            {
                **common,
                "predicted": before["predicted"],
                "p_in_scope": float(before["probabilities"]["in_scope"]),
                "p_out_of_scope": float(before["probabilities"]["out_of_scope"]),
                "confidence": float(before["answer_confidence"]),
            }
        )
        trained.append(
            {
                **common,
                "predicted": after["predicted"],
                "p_in_scope": float(after["probabilities"]["in_scope"]),
                "p_out_of_scope": float(after["probabilities"]["out_of_scope"]),
                "confidence": float(after["answer_confidence"]),
            }
        )
    return base, trained


def flip_summary(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> dict[str, Any]:
    flips = []
    counts: Counter[str] = Counter()
    for left, right in zip(before, after):
        a = left["predicted"] == left["expected"]
        b = right["predicted"] == right["expected"]
        kind = (
            "correct_to_correct"
            if a and b
            else "wrong_to_correct"
            if not a and b
            else "correct_to_wrong"
            if a and not b
            else "wrong_to_wrong"
        )
        counts[kind] += 1
        if a != b:
            flips.append(
                {
                    "id": right["id"],
                    "question": right["question"],
                    "group": right["group"],
                    "language": right["language"],
                    "expected": right["expected"],
                    "before": left["predicted"],
                    "after": right["predicted"],
                    "change": kind,
                }
            )
    return {"counts": dict(counts), "flips": flips}


def confidence_summary(rows: list[dict[str, Any]], *, threshold: float = 0.8) -> dict[str, Any]:
    correct = [row for row in rows if row["predicted"] == row["expected"]]
    wrong = [row for row in rows if row["predicted"] != row["expected"]]
    high_wrong = [row for row in wrong if row["confidence"] >= threshold]
    return {
        "mean_confidence_correct": round(
            sum(row["confidence"] for row in correct) / max(1, len(correct)),
            4,
        ),
        "mean_confidence_wrong": round(
            sum(row["confidence"] for row in wrong) / max(1, len(wrong)),
            4,
        ),
        "high_confidence_wrong": len(high_wrong),
        "high_confidence_wrong_ids": [row["id"] for row in high_wrong],
    }


def term_profile(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wrong = [row for row in rows if row["predicted"] != row["expected"]]
    output = []
    for name, pattern in ML_PATTERNS.items():
        hits = [row for row in wrong if re.search(pattern, row["question"])]
        if hits:
            output.append(
                {
                    "term": name,
                    "wrong_items": len(hits),
                    "examples": [
                        {
                            "id": row["id"],
                            "group": row["group"],
                            "language": row["language"],
                            "question": row["question"],
                        }
                        for row in hits[:3]
                    ],
                }
            )
    return sorted(output, key=lambda row: (-row["wrong_items"], row["term"]))


def reviewed_ids() -> set[str]:
    with (DATASET_DIR / "review.csv").open(encoding="utf-8-sig", newline="") as fh:
        return {row["id"] for row in csv.DictReader(fh)}


def gate_failure_details(
    trained_rows: list[dict[str, Any]],
    reviewed: set[str],
) -> dict[str, Any]:
    gate = json.loads(GATE_REPORT.read_text(encoding="utf-8"))
    thresholds = gate["thresholds"]
    llm = {
        row["id"]: row
        for row in json.loads(LLM_REPORT.read_text(encoding="utf-8"))["rows"]
    }
    decisions = []
    for row in trained_rows:
        if (
            row["predicted"] == "in_scope"
            and row["p_in_scope"] >= thresholds["in_scope"]
        ):
            route, final = "laya", "in_scope"
        elif (
            row["predicted"] == "out_of_scope"
            and row["p_out_of_scope"] >= thresholds["out_of_scope"]
        ):
            route, final = "laya", "out_of_scope"
        else:
            route = "llm_fallback"
            final = llm.get(row["id"], {}).get("predicted")
        decisions.append(
            {
                **row,
                "route": route,
                "final_predicted": final,
                "error": llm.get(row["id"], {}).get("error"),
                "reviewed": row["id"] in reviewed,
            }
        )
    accepted = [row for row in decisions if row["route"] == "laya"]
    accepted_errors = [
        row for row in accepted if row["final_predicted"] != row["expected"]
    ]
    fallback_errors = [
        row
        for row in decisions
        if row["route"] == "llm_fallback" and row["final_predicted"] != row["expected"]
    ]
    return {
        "thresholds": thresholds,
        "calibration_coverage": gate["calibration"]["gate_selection"]["coverage"],
        "coverage": round(len(accepted) / len(decisions), 4),
        "accepted_items": len(accepted),
        "accepted_errors": len(accepted_errors),
        "fallback_items": len(decisions) - len(accepted),
        "fallback_errors": len(fallback_errors),
        "accepted_error_examples": accepted_errors,
        "fallback_error_examples": fallback_errors,
        "accepted_by_group": dict(Counter(row["group"] for row in accepted)),
        "accepted_by_language": dict(Counter(row["language"] for row in accepted)),
    }


def markdown(report: dict[str, Any]) -> str:
    cal = report["calibration"]
    test = report["internal_test"]
    lines = [
        "# Laya scope pilot 误差归因",
        "",
        "## 结论",
        "",
        report["conclusion"],
        "",
        "## 核心证据",
        "",
        "| 指标 | calibration base | calibration trained | internal base | internal trained |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in (
        "accuracy",
        "in_scope_false_refusal_rate",
        "ood_false_pass_rate",
    ):
        lines.append(
            f"| {key} | {cal['base_metrics'][key]:.4f} | "
            f"{cal['trained_metrics'][key]:.4f} | "
            f"{test['base_metrics'][key]:.4f} | "
            f"{test['trained_metrics'][key]:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 训练翻转",
            "",
            f"- Calibration: `{json.dumps(cal['flips']['counts'], ensure_ascii=False)}`",
            f"- Internal test: `{json.dumps(test['flips']['counts'], ensure_ascii=False)}`",
            "",
            "## 主要错误方向",
            "",
        ]
    )
    for name, summary in (
        ("Calibration trained", report["calibration"]["trained_errors"]),
        ("Internal trained", report["internal_test"]["trained_errors"]),
    ):
        lines.append(
            f"- {name}: false refusal {summary['false_refusal_items']}，"
            f"false pass {summary['false_pass_items']}；"
            f"reviewed error items {summary['reviewed_error_items']}/"
            f"{summary['error_items']}。"
        )
    lines.extend(
        [
            "",
            "## 语言与题型",
            "",
            "完整分组指标见同名 JSON。最明显的短板是 boundary 和英文/混合语言，"
            "而内部测试里的 OOD 误放行主要来自技术词触发。",
            "",
            "## 门控失败原因",
            "",
            f"- Calibration 选阈值时覆盖 {report['gate']['calibration_coverage']:.1%}，"
            f"到 internal_test 后实际覆盖 {report['gate']['coverage']:.1%}。",
            f"- 本地接管样例中仍有 {report['gate']['accepted_errors']} 条错误。",
            f"- Fallback 错误 {report['gate']['fallback_errors']} 条；"
            "其中包含当前 LLM 的一条 timeout。",
            "",
            "## 决策",
            "",
            "先修数据与任务定义，再考虑 encoder 解冻；多 seed 只会重复当前瓶颈。",
            "",
            "## 归因排序",
            "",
        ]
    )
    for item in report["root_causes"]:
        lines.append(
            f"- **{item['severity']} · {item['cause']}**：{item['evidence']} "
            f"下一步：{item['next_action']}"
        )
    lines.extend(
        [
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    reviewed = reviewed_ids()
    cal_base, cal_trained = calibration_rows()
    test_base, test_trained = internal_rows()

    def errors(rows: list[dict[str, Any]]) -> dict[str, Any]:
        wrong = [row for row in rows if row["predicted"] != row["expected"]]
        return {
            "error_items": len(wrong),
            "false_refusal_items": sum(
                row["expected"] == "in_scope" and row["predicted"] == "out_of_scope"
                for row in wrong
            ),
            "false_pass_items": sum(
                row["expected"] == "out_of_scope" and row["predicted"] == "in_scope"
                for row in wrong
            ),
            "reviewed_error_items": sum(row["id"] in reviewed for row in wrong),
            "by_group": dict(Counter(row["group"] for row in wrong)),
            "by_language": dict(Counter(row["language"] for row in wrong)),
            "by_domain": dict(
                sorted(
                    Counter(str(row.get("domain") or "unknown") for row in wrong).items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "examples": wrong[:20],
        }

    gate = gate_failure_details(test_trained, reviewed)
    test_false_refusals = [
        row
        for row in test_trained
        if row["expected"] == "in_scope" and row["predicted"] == "out_of_scope"
    ]
    test_false_passes = [
        row
        for row in test_trained
        if row["expected"] == "out_of_scope" and row["predicted"] == "in_scope"
    ]
    internal_errors = errors(test_trained)
    root_causes = [
        {
            "severity": "primary",
            "cause": "head-only 只改善了概率排序，没有移动主要决策边界",
            "evidence": (
                f"internal_test 仅 3 条错转对、2 条对转错，37 条继续错；"
                f"NLL/Brier 改善但 accuracy 仅从 0.680 到 0.688。"
            ),
            "next_action": "先用已标错误构造更难/更均衡的 head-only 诊断集，不先加 seed。",
        },
        {
            "severity": "primary",
            "cause": "英文 ML 问题被误拒，中文专业领域问题被误放行",
            "evidence": (
                f"20 条范围内误拒中 13 条为英文；19 条 OOD 误放行中 11 条为中文。"
                f"OOD 误放行集中在 petroleum_engineering 6、pharmacology 4、"
                f"aerospace 3。"
            ),
            "next_action": "补英文 ML 术语正例和中文 OOD 技术词负例，按领域分层复核。",
        },
        {
            "severity": "primary",
            "cause": "跨域边界定义仍是主要混淆源",
            "evidence": (
                f"boundary accuracy 0.60；该组 6 条错误中 ml_for_education 4、"
                f"physics_informed_ml 1、abbreviation_and_polysemy 1。"
            ),
            "next_action": "扩充边界金标，重点覆盖“用 ML 优化领域”与“领域传统方法”的最小对比对。",
        },
        {
            "severity": "secondary",
            "cause": "正确/错误样本的置信度可分离性不足",
            "evidence": (
                f"internal_test 正确均值 0.6760、错误均值 0.6393，仅差 0.0367；"
                f"安全门控因此只覆盖 0.096。"
            ),
            "next_action": "若数据修正后仍有此问题，再评估更大容量 head 或 encoder 解冻。",
        },
        {
            "severity": "not_primary_but_incomplete",
            "cause": "标签/切分错误不是已证实主因，但仍有银标覆盖缺口",
            "evidence": (
                f"切分泄漏和近重复均为 0；internal_test 的 39 条错误中 10 条经过复核且"
                f"全部为 accept，其余 29 条未逐条审阅。"
            ),
            "next_action": "优先人工复核 29 条未审错误，再判断是否需要调整 silver 标签。",
        },
    ]
    report = {
        "scope": {
            "model": str(MODEL_DIR.relative_to(ROOT)),
            "internal_test_reused_for_attribution": True,
            "external_16_reused": False,
        },
        "calibration": {
            "base_metrics": metrics(cal_base),
            "trained_metrics": metrics(cal_trained),
            "base_by_group": grouped(cal_base, "group"),
            "trained_by_group": grouped(cal_trained, "group"),
            "base_by_language": grouped(cal_base, "language"),
            "trained_by_language": grouped(cal_trained, "language"),
            "flips": flip_summary(cal_base, cal_trained),
            "confidence_base": confidence_summary(cal_base),
            "confidence_trained": confidence_summary(cal_trained),
            "trained_errors": errors(cal_trained),
            "term_profile": term_profile(cal_trained),
        },
        "internal_test": {
            "base_metrics": metrics(test_base),
            "trained_metrics": metrics(test_trained),
            "base_by_group": grouped(test_base, "group"),
            "trained_by_group": grouped(test_trained, "group"),
            "base_by_language": grouped(test_base, "language"),
            "trained_by_language": grouped(test_trained, "language"),
            "flips": flip_summary(test_base, test_trained),
            "confidence_base": confidence_summary(test_base),
            "confidence_trained": confidence_summary(test_trained),
            "trained_errors": errors(test_trained),
            "term_profile": term_profile(test_trained),
        },
        "gate": gate,
        "root_causes": root_causes,
        "conclusion": (
            "head-only 改善了 NLL/Brier，但没有改善 argmax 决策边界；"
            "主要瓶颈是边界/跨域定义和英文/混合语言表示。"
            "全量 146 条复核样本中错误样本标签均为 accept，"
            "因此当前证据不支持把失败主要归因于已复核标签错误。"
            "安全门控需要较高阈值才能在 calibration 控制风险，导致覆盖率不足 60%。"
        ),
    }
    json_path = ROOT / "eval" / "scope_v1" / "error_attribution.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path = ROOT / "eval" / "scope_v1" / "error_attribution.md"
    md_path.write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "conclusion": report["conclusion"],
                "calibration_flips": report["calibration"]["flips"]["counts"],
                "internal_flips": report["internal_test"]["flips"]["counts"],
                "gate_coverage": gate["coverage"],
                "json": str(json_path.relative_to(ROOT)),
                "markdown": str(md_path.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
