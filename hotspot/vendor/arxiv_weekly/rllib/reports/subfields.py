"""子方向覆盖报告 —— 「方向的完整性」的常设检查。

这份报告存在的理由，是别的报告查不出来的那类问题：

**一个漏掉的子方向不会表现为"一条错误记录"，而表现为它根本不在那里。**
没有任何质量指标会报警——命中率、异常率、字段完整率全都正常。唯一能把它变成
一个可见数字的办法，是拿一个**外部**的子方向骨架去套，然后看哪一格的计数是 0。

因此报告有两块：

1. **分布**——每个子方向在各会议上的篇数。看方向冷热，也是趋势分析的原料。
2. **盲区**——有专有子方向词命中、但 ``is_rl`` 判为否的论文。
   这一块是**召回漏洞的告警**：某个子方向的盲区数接近或超过命中数时，
   说明筛选器基本没覆盖这个方向，而不是"漏了几篇"。

盲区只报计数与少量样本标题，**不自动改判**。改判走 ``rl.manual_override``，
那是人工区（见 CLAUDE.md 铁律 2）。

跑在 jsonl 上，不解析 Markdown（铁律 1）。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from .. import ROOT
from ..filters.subfields import SubfieldTagger
from ..store import PaperStore

REPORT_DIR = "data/reports"

#: 每个子方向最多留几条盲区样本标题。留标题不是为了读——是为了让人**能去核对**
#: 词表是不是误伤。30 条足够看出模式，也不会把报告撑成论文列表。
BLIND_SAMPLES = 30

#: 低于这个篇数就认为该子方向"覆盖不足"，在报告里顶出来。
#: 不是硬性阈值，是提示：一个子方向在 13k 篇里只有个位数，值得看一眼是真空缺
#: 还是筛选/词表的问题。
THIN_COVERAGE = 15

#: 盲区占比达到多少就顶出来。0.4 意味着「这个子方向的论文里，筛选器连六成都够不着」——
#: 到这个程度就不能算「漏了几篇」，而是这个方向的覆盖本身不可信。
BLIND_RATIO = 0.4


def _subfield_table(papers, tagger: SubfieldTagger) -> dict[str, Any]:
    by_sub: dict[str, Counter] = defaultdict(Counter)
    rl_by_sub: Counter[str] = Counter()
    for paper in papers:
        for key in paper.rl.subfield:
            by_sub[key][paper.venue] += 1
            if paper.is_rl_effective:
                rl_by_sub[key] += 1
    rows = []
    for rule in tagger.rules:
        counts = by_sub.get(rule.key, Counter())
        rows.append({
            "key": rule.key,
            "label": rule.label,
            "total": sum(counts.values()),
            "rl": rl_by_sub.get(rule.key, 0),
            "by_venue": dict(counts.most_common()),
        })
    rows.sort(key=lambda r: r["total"], reverse=True)
    return {"rows": rows}


def _blind_section(papers, tagger: SubfieldTagger) -> dict[str, Any]:
    from ..filters.subfields import _haystack  # noqa: PLC2701  同包内部工具

    per_sub: dict[str, list] = defaultdict(list)
    for paper in papers:
        if paper.is_rl_effective:
            continue
        hay = _haystack(paper)
        for rule in tagger.rules:
            if rule.strong is not None and rule.strong.search(hay):
                per_sub[rule.key].append(paper)

    rows = []
    for key, group in per_sub.items():
        rows.append({
            "key": key,
            "label": tagger.labels.get(key, key),
            "papers": len(group),
            "by_venue": dict(Counter(p.venue for p in group).most_common()),
            "samples": [
                {"venue": p.venue, "year": p.year, "title": p.title}
                for p in group[:BLIND_SAMPLES]
            ],
        })
    rows.sort(key=lambda r: r["papers"], reverse=True)
    return {"rows": rows, "total_papers": sum(r["papers"] for r in rows)}


def build(store: PaperStore, cfg: dict[str, Any]) -> dict[str, Any]:
    papers = list(store.papers.values())
    tagger = SubfieldTagger.from_yaml(
        ROOT / cfg["filter"].get("subfields_file", "data/rl_subfields.yaml")
    )
    table = _subfield_table(papers, tagger)
    blind = _blind_section(papers, tagger)

    total_rl = sum(1 for p in papers if p.is_rl_effective)

    # 把盲区数并进分布表，算出**盲区占比**。
    #
    # 为什么用占比而不是「盲区数 ≥ 命中数」：后者是个 50% 处的硬悬崖，
    # 72% 盲区的子方向和 48% 的会得到同样的"未告警"。占比是个连续的灰度，
    # 读者自己看梯度，比一个二元标志诚实。
    blind_by_key = {r["key"]: r["papers"] for r in blind["rows"]}
    rows = []
    for row in table["rows"]:
        b = blind_by_key.get(row["key"], 0)
        denom = b + row["rl"]
        rows.append({
            **row,
            "blind": b,
            "blind_ratio": round(b / denom, 3) if denom else 0.0,
        })
    rows.sort(key=lambda r: r["blind_ratio"], reverse=True)

    thin = []
    for row in rows:
        if row["total"] < THIN_COVERAGE:
            thin.append({**row, "why": "计数过低"})
        elif row["blind_ratio"] >= BLIND_RATIO:
            thin.append({**row, "why": f"盲区占比 {row['blind_ratio']:.0%}"})

    return {
        "generated": date.today().isoformat(),
        "total": len(papers),
        "rl_effective": total_rl,
        "subfields": len(tagger.rules),
        "tagged": sum(1 for p in papers if p.rl.subfield),
        "table": rows,
        "blind": blind["rows"],
        "blind_total": blind["total_papers"],
        "thin_coverage": thin,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RL 子方向覆盖报告",
        "",
        f"生成日期：**{report['generated']}**　全库 **{report['total']}** 篇，"
        f"其中 RL 判定为真 **{report['rl_effective']}** 篇；"
        f"有子方向标签 **{report['tagged']}** 篇（{report['subfields']} 个子方向）。",
        "",
        "> 这份报告的作用是**发现库里缺了什么**。一个漏掉的子方向不会表现为一条错误记录，",
        "> 而表现为它根本不在这里。所以下面的计数为 0 或异常低的格子，比任何错误都更值得看。",
        "",
        "## 一、子方向分布",
        "",
        "| 子方向 | 说明 | 合计 | RL 判定为真 | 盲区 | 盲区占比 | 按会议 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in report["table"]:
        venues = " ".join(f"{k} {v}" for k, v in row["by_venue"].items()) or "—"
        lines.append(
            f"| `{row['key']}` | {row['label']} | {row['total']} | {row['rl']} "
            f"| {row['blind']} | {row['blind_ratio']:.0%} | {venues} |"
        )

    lines += [
        "",
        f"## 二、覆盖不足的子方向（{len(report['thin_coverage'])} 个）",
        "",
        f"判据：计数低于 {THIN_COVERAGE}，**或盲区占比 ≥ {BLIND_RATIO:.0%}**。后者意味着",
        "筛选器基本没覆盖这个方向，而不是「漏了几篇」——这两种情况的处理方式完全不同。",
        "",
        "| 子方向 | 合计 | RL 命中 | 盲区 | 盲区占比 | 提示 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    if report["thin_coverage"]:
        for row in report["thin_coverage"]:
            lines.append(
                f"| `{row['key']}` | {row['total']} | {row['rl']} | {row['blind']} "
                f"| {row['blind_ratio']:.0%} | {row['why']} |"
            )
    else:
        lines.append("| — | — | — | — | — | 无 |")

    lines += [
        "",
        f"## 三、筛选盲区（共 {report['blind_total']} 篇）",
        "",
        "有**专有**子方向词命中，但 `rl.is_rl` 判为否。这些是召回漏洞的候选：",
        "可能是筛选器漏了的 RL 论文，也可能是词表误伤（缩写尤其容易）。",
        "",
        "**不自动改判。** 核对后要改，走 `rl.manual_override`（人工区）。",
        "",
        "| 子方向 | 盲区篇数 | 按会议 |",
        "|---|---:|---|",
    ]
    for row in report["blind"]:
        venues = " ".join(f"{k} {v}" for k, v in row["by_venue"].items()) or "—"
        lines.append(f"| `{row['key']}` | {row['papers']} | {venues} |")

    lines += ["", "### 盲区样本（用于核对词表是否误伤）", ""]
    for row in report["blind"][:6]:
        lines.append(f"**`{row['key']}`**（{row['papers']} 篇）")
        lines.append("")
        for s in row["samples"][:12]:
            lines.append(f"- [{s['venue']} {s['year']}] {s['title']}")
        lines.append("")

    lines += [
        "---",
        "",
        "## 怎么用这份报告",
        "",
        "1. **先看第二节。** 覆盖不足的子方向要么是真空缺（值得做），要么是词表/筛选的问题",
        "   （值得修）。两种情况都需要人看一眼，报告只负责把它顶出来。",
        "2. **再看第三节。** 盲区数大的子方向，说明 `data/rl_keywords.yaml` 缺了这类词。",
        "   增补词表后重跑 `python -m rllib filter --rl` 即可，**不需要重新抓取**。",
        "3. **最后看第一节。** 分布本身是趋势分析的原料；跨周对比这份报告就能看出方向冷热变化。",
        "",
        "> 检索工具（`python -m rllib search`）**默认搜全量、不以 RL 筛选为闸门**，",
        "> 所以即使某个子方向被筛选器漏掉，也仍然搜得到。筛选器在这里是筛选面，不是闸门。",
    ]
    return "\n".join(lines) + "\n"


def write(report: dict[str, Any], cfg: dict[str, Any]) -> tuple[Path, Path]:
    out = ROOT / cfg["paths"]["data"] / "reports"
    out.mkdir(parents=True, exist_ok=True)
    jp, mp = out / "subfields.json", out / "subfields.md"
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    mp.write_text(render_markdown(report), encoding="utf-8")
    return jp, mp


def run(store: PaperStore, cfg: dict[str, Any]) -> dict[str, Any]:
    report = build(store, cfg)
    jp, mp = write(report, cfg)
    return {
        "report": str(jp.relative_to(ROOT)),
        "markdown": str(mp.relative_to(ROOT)),
        "subfields": report["subfields"],
        "tagged": report["tagged"],
        "blind_total": report["blind_total"],
        "thin_coverage": len(report["thin_coverage"]),
        "thin_keys": [r["key"] for r in report["thin_coverage"]][:12],
    }
