"""自动化质量报告（阶段 6）。

**这是唯一被允许回答"抽取质量怎么样"的东西。** CLAUDE.md 的 token 纪律第 4 条：
质量校验靠报告，不靠读论文样本；只有异常率 > 5% 才取 2–3 篇有目的地定位。

报告分三块：

1. **元数据**：arXiv ID / 摘要 / DOI 覆盖率——决定能取到全文的比例。
2. **全文**：抽了多少、哪些来源、字符数分布、异常率。
3. **缺口**：明确列出拿不到全文的论文，按 venue 计数。

产出两个文件：

* ``data/reports/quality.json`` —— 机器读，供后续流程和对比不同周次
* ``data/reports/quality.md`` —— 人读，Obsidian 里能直接打开

两个都是**派生产物**，随时可重建。报告不做任何修改数据的动作：
异常样本只标记不删除，宁可多留可疑数据，也不要静默丢掉好数据。
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import ROOT
from ..paths import text_path
from ..store import PaperStore
from ..extract import text as text_mod

log = logging.getLogger(__name__)

#: 异常率上限。超过就该停下来看具体样本，而不是继续往下跑。
ANOMALY_BUDGET = 0.05


def _metadata_section(store: PaperStore) -> dict[str, Any]:
    total = len(store.papers)
    by_venue: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "arxiv": 0, "abstract": 0, "doi": 0, "rl": 0}
    )
    counts = Counter()
    for p in store.papers.values():
        v = by_venue[p.venue]
        v["total"] += 1
        if p.arxiv_id:
            v["arxiv"] += 1
            counts["arxiv"] += 1
        if p.abstract:
            v["abstract"] += 1
            counts["abstract"] += 1
        if p.doi:
            v["doi"] += 1
            counts["doi"] += 1
        if p.is_rl_effective:
            v["rl"] += 1
            counts["rl"] += 1

    def rate(n: int) -> float:
        return round(n / total, 4) if total else 0.0

    return {
        "total": total,
        "arxiv": counts["arxiv"],
        "arxiv_rate": rate(counts["arxiv"]),
        "abstract": counts["abstract"],
        "abstract_rate": rate(counts["abstract"]),
        "doi": counts["doi"],
        "rl": counts["rl"],
        "by_venue": {k: dict(v) for k, v in sorted(by_venue.items())},
    }


def text_on_disk(store: PaperStore, text_dir: Path) -> set[str]:
    """**磁盘上真实存在全文**的 paper_id 集合。

    不能拿 manifest 当这个集合：它每 200 篇才落盘一次（``extract/text.py``），
    崩溃或重启后必然落后于磁盘——而"崩溃恢复之后"恰恰是最需要跑这份报告的
    时刻。按清单算会把已经抽到的算成没抽到：覆盖率虚低、缺口清单虚长，
    而且不报任何错。所以以磁盘为准，清单只用来取 chars/pages 这些元数据。

    遍历 ``store.papers`` 而不是反向解析文件名：``paths.safe_name`` 的净化是
    有损的（``:`` 和 ``/`` 都变成 ``-``），反解回 paper_id 会撞名。
    """
    return {
        p.paper_id for p in store.papers.values()
        if text_path(text_dir, p.paper_id).exists()
    }


def _text_section(
    store: PaperStore, cfg: dict[str, Any], on_disk: set[str]
) -> dict[str, Any]:
    text_dir = ROOT / cfg["paths"]["text"]
    manifest = text_mod.load_manifest(text_dir)

    sources = Counter()
    chars: list[int] = []
    short: list[str] = []
    suspect: list[str] = []
    total_bytes = 0
    # 清单说抽过、文件却没了的（被手删 / 同步盘没同步完）。必须单独计数——
    # 否则报告的覆盖率会虚高，而实际读不到内容。
    dangling = [pid for pid in manifest if pid not in on_disk]
    # 反过来：文件在、清单里还没有（崩溃 / 未 flush）。**计入覆盖率**，
    # 但要显式报出来，否则这种漂移会一直悄悄存在。
    unaccounted: list[str] = []

    for pid in sorted(on_disk):
        rec = manifest.get(pid)
        if rec is None:
            unaccounted.append(pid)
            total_bytes += text_path(text_dir, pid).stat().st_size
            n = len(text_mod.read_text(text_dir, pid))
            sources["?"] += 1
        else:
            total_bytes += int(rec.get("bytes") or 0)
            n = int(rec.get("chars") or 0)
            sources[rec.get("source", "?")] += 1
        chars.append(n)
        if n < text_mod.MIN_CHARS:
            short.append(pid)
        pages = int((rec or {}).get("pages") or 0)
        if pages and n / pages < text_mod.SUSPECT_CHARS_PER_PAGE:
            # PDF 抽出来的文字密度过低：多半是扫描版，或正文其实是图片
            suspect.append(pid)

    chars.sort()

    def pct(q: float) -> int:
        if not chars:
            return 0
        return chars[min(int(len(chars) * q), len(chars) - 1)]

    # ``with_text`` **不含** too_short：它和 gaps 的 no_text_total 一起必须
    # 恰好凑齐全库——``with_text + too_short + no_text_total == len(store.papers)``。
    # 这里曾经写成 ``len(chars) + len(short)``，而 short 是 chars 的子集，
    # 于是全库总数多出 2×|short|。没有任何一处断言会因此变红，只能靠这条恒等式发现。
    return {
        "with_text": len(chars) - len(short),
        "files_on_disk": len(chars),
        "unaccounted_in_manifest": len(unaccounted),
        "unaccounted_sample": unaccounted[:10],
        "sources": dict(sources),
        "dangling_manifest": len(dangling),
        "dangling_sample": dangling[:10],
        "total_mb": round(total_bytes / 1048576, 1),
        "chars": {
            "min": chars[0] if chars else 0,
            "p10": pct(0.10),
            "p50": pct(0.50),
            "p90": pct(0.90),
            "max": chars[-1] if chars else 0,
        },
        "too_short": len(short),
        "too_short_sample": short[:10],
        "suspect_low_density": len(suspect),
        "suspect_sample": suspect[:10],
    }


def _pdf_section(store: PaperStore, cfg: dict[str, Any]) -> dict[str, Any]:
    pdfs_dir = ROOT / cfg["paths"]["pdfs"]
    n_records = 0
    n_files = 0
    n_missing = 0
    total_bytes = 0
    by_venue: dict[str, int] = defaultdict(int)

    for p in store.papers.values():
        if not p.local_pdf:
            continue
        n_records += 1
        path = ROOT / p.local_pdf
        if path.exists() and path.stat().st_size > 1024:
            n_files += 1
            total_bytes += path.stat().st_size
            by_venue[p.venue] += 1
        else:
            n_missing += 1

    # 目录里没有对应记录的文件：多半是 paper_id 变了或手动拷进来的。
    # 不参与覆盖率计算，但要说出来——说明磁盘上躺着没人认领的数据。
    tracked = {p.local_pdf for p in store.papers.values() if p.local_pdf}
    orphans = 0
    if pdfs_dir.exists():
        for f in pdfs_dir.glob("*.pdf"):
            if f.relative_to(ROOT).as_posix() not in tracked:
                orphans += 1

    return {
        "records": n_records,
        "files": n_files,
        "missing_files": n_missing,
        "orphan_files": orphans,
        "total_mb": round(total_bytes / 1048576, 1),
        "by_venue": dict(sorted(by_venue.items())),
    }


def _gaps_section(
    store: PaperStore, cfg: dict[str, Any], on_disk: set[str]
) -> dict[str, Any]:
    """明确列出拿不到全文的论文。**这个清单必须显式存在**——
    缺口一旦不可见，就会在下游被当成"这些论文不存在"，而不是"我们没取到"。"""
    # 与 _text_section 共用同一个集合（build() 里只扫一次）。各扫一次的话，
    # extract 正好在两段之间写下一个文件，恒等式就会差 1——那种偶发的不一致
    # 比固定的错更难查。
    have = on_disk
    no_arxiv: dict[str, int] = defaultdict(int)
    no_text_by_venue: dict[str, int] = defaultdict(int)
    rl_no_text: list[str] = []

    for p in store.papers.values():
        if not p.arxiv_id:
            no_arxiv[p.venue] += 1
        if p.paper_id not in have:
            no_text_by_venue[p.venue] += 1
            if p.is_rl_effective:
                rl_no_text.append(p.paper_id)

    return {
        "no_arxiv_id": dict(sorted(no_arxiv.items())),
        "no_arxiv_id_total": sum(no_arxiv.values()),
        "no_text_by_venue": dict(sorted(no_text_by_venue.items())),
        "no_text_total": sum(no_text_by_venue.values()),
        # RL 子集的缺口优先——这是用户实际会去读的部分
        "rl_without_text": len(rl_no_text),
        "rl_without_text_sample": sorted(rl_no_text)[:10],
    }


def build(store: PaperStore, cfg: dict[str, Any]) -> dict[str, Any]:
    meta = _metadata_section(store)
    # 只扫一次磁盘，两个 section 共用，恒等式才成立（见 text_on_disk 的注释）。
    on_disk = text_on_disk(store, ROOT / cfg["paths"]["text"])
    txt = _text_section(store, cfg, on_disk)
    pdf = _pdf_section(store, cfg)
    gaps = _gaps_section(store, cfg, on_disk)

    attempted = txt["with_text"] + txt["too_short"] + gaps["no_text_total"]
    anomalies = txt["too_short"] + txt["suspect_low_density"]
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "extractor_version": text_mod.VERSION,
        "metadata": meta,
        "text": txt,
        "pdf": pdf,
        "gaps": gaps,
        "anomaly": {
            "attempted": attempted,
            "too_short": txt["too_short"],
            "low_density": txt["suspect_low_density"],
            "total": anomalies,
            "rate": round(anomalies / attempted, 4) if attempted else 0.0,
            "budget": ANOMALY_BUDGET,
            "ok": (anomalies / attempted) <= ANOMALY_BUDGET if attempted else True,
        },
        # 恒等式：每篇论文恰好落在"有全文 / 过短 / 没有全文"三者之一。
        # 它曾经不成立（with_text 把 short 重复计了一次），而报告里没有任何
        # 一处会因此变红——只有把恒等式本身写进来，漂移才会当场暴露。
        #
        # 第二条恒等式同源：**磁盘上每一份 PDF 都要有记录认领**（orphan == 0）。
        # 2026-09-16 实测踩过：pdf.py 中止时收尾落盘不执行，4 个 PDF 躺在
        # pdfs/ 里而 local_pdf 为空。当时 orphan_files 这个数**已经在打**，
        # 却不进任何 ok ——于是报告全绿着把这件事放了过去。
        # 数出来但不接红灯，等于没有检查。
        "accounting": {
            "papers": len(store.papers),
            "accounted": attempted,
            "ok": attempted == len(store.papers),
            "pdf_files": pdf["files"],
            "pdf_orphans": pdf["orphan_files"],
            "pdf_ok": pdf["orphan_files"] == 0,
            # 反方向同源：**每条记录都要指向一个真实存在的文件**。
            # 铁律 7 的原话是「两个方向都要查。只查一个方向等于默认另一个方向
            # 不会发生」——pdf_ok 只管"磁盘有、库里没有"，缺口这一半此前无人接红灯：
            # ``missing_files`` 一直在打这个数，却和 pdf_ok 之前一样不进任何 ok。
            "pdf_missing": pdf["missing_files"],
            "pdf_missing_ok": pdf["missing_files"] == 0,
        },
    }
    return report


def render_markdown(r: dict[str, Any]) -> str:
    m, t, p, g, a = r["metadata"], r["text"], r["pdf"], r["gaps"], r["anomaly"]

    def rows(d: dict[str, Any], keys: list[str]) -> str:
        out = ["| venue | " + " | ".join(keys) + " |",
               "|---|" + "---|" * len(keys)]
        for venue, v in d.items():
            out.append(f"| {venue} | " + " | ".join(str(v.get(k, 0)) for k in keys) + " |")
        return "\n".join(out)

    lines = [
        "# 数据质量报告",
        "",
        f"生成时间：{r['generated']}　抽取器版本：`{r['extractor_version']}`",
        "",
        "> 本文件由 `rllib report quality` 生成，是派生产物，可以随时重新生成。",
        "",
        "## 结论",
        "",
        f"- **异常率 {a['rate']:.2%}**（上限 {a['budget']:.0%}）"
        f"　{'通过' if a['ok'] else '**超出，需要人工看样本**'}",
        f"- 可抽取 {a['attempted']} 篇，其中过短 {a['too_short']} 篇、"
        f"文字密度可疑 {a['low_density']} 篇",
        f"- 记账恒等式（有全文 + 过短 + 无全文 = 全库）："
        f"{r['accounting']['accounted']} / {r['accounting']['papers']}"
        f"　{'通过' if r['accounting']['ok'] else '**对不上，报告口径有 bug**'}",
        "",
        "## 元数据覆盖率",
        "",
        f"- 总计 **{m['total']}** 篇，RL 子集 **{m['rl']}** 篇",
        f"- arXiv ID：{m['arxiv']}（{m['arxiv_rate']:.1%}）—— **这直接决定能取到多少全文**",
        f"- 摘要：{m['abstract']}（{m['abstract_rate']:.1%}）",
        f"- DOI：{m['doi']}",
        "",
        rows(m["by_venue"], ["total", "rl", "arxiv", "abstract", "doi"]),
        "",
        "## 全文",
        "",
        f"- 已抽出 **{t['with_text']}** 篇，合计 {t['total_mb']} MB",
        f"- 来源分布：{t['sources'] or '（空）'}",
        f"- 字符数 min/p10/p50/p90/max："
        f"{t['chars']['min']} / {t['chars']['p10']} / {t['chars']['p50']} / "
        f"{t['chars']['p90']} / {t['chars']['max']}",
    ]
    if t["dangling_manifest"]:
        lines += [
            "",
            f"⚠️ **清单里有 {t['dangling_manifest']} 条记录在磁盘上找不到对应文件。**",
            "这些论文会被算进覆盖率但实际读不到，需要重跑 `rllib extract` 补齐。",
            f"示例：`{'`, `'.join(t['too_short_sample'])}`",
        ]
    lines += [
        "",
        "## PDF",
        "",
        f"- 已下载 **{p['files']}** 篇，合计 {p['total_mb']} MB",
        f"- 记录了但文件丢失：{p['missing_files']}"
        f"　{'通过' if r['accounting']['pdf_missing_ok'] else '**有记录指向不存在的文件**'}",
        f"- 磁盘上无人认领的 PDF：{p['orphan_files']}"
        f"　{'通过' if r['accounting']['pdf_ok'] else '**有文件没记账，磁盘与真相源对不上**'}",
        "",
        # ``_pdf_section`` 的 ``by_venue`` 是 ``{venue: 篇数}``，而 ``rows`` 要的是
        # ``{venue: {列: 值}}``（``metadata`` 那节就是后者）。形状不同，包一层再喂。
        # 别改成让 ``rows`` 容忍裸整数——那是把形状错误咽下去，换来的静默比崩溃更难查。
        # 2026-09-17 实测：PDF 下满之后这个分支**第一次**被走到，直接 AttributeError；
        # 此前 ``pdf_files=0``，``by_venue`` 为空，一直走 ``_（无）_`` 那一支。
        rows({v: {"files": n} for v, n in p["by_venue"].items()}, ["files"])
        if p["by_venue"]
        else "_（无）_",
        "",
        "## 缺口",
        "",
        f"- **无 arXiv ID：{g['no_arxiv_id_total']} 篇。**"
        "这些论文拿不到源码，只能靠本地 PDF；而 PDF 只下了 RL 子集。",
        f"- 无全文：{g['no_text_total']} 篇，其中 **RL 子集占 {g['rl_without_text']} 篇**"
        "（RL 子集是实际会去读的部分，缺口优先看这个数字）",
        "",
    ]
    if g["no_text_by_venue"]:
        lines += [
            "| venue | 无全文 | 无 arXiv ID |",
            "|---|---|---|",
        ]
        for venue, n in g["no_text_by_venue"].items():
            lines.append(f"| {venue} | {n} | {g['no_arxiv_id'].get(venue, 0)} |")
    lines += [
        "",
        "## 本周新增缺口与变化",
        "",
        "> 与上一份报告对比。首次生成时为空。",
        "",
    ]
    return "\n".join(lines) + "\n"


def write(report: dict[str, Any], cfg: dict[str, Any]) -> tuple[Path, Path]:
    # 走 cfg 而不是写死 `ROOT / "data" / "reports"`：`--store weekly` 下
    # 写死的路径会把周报的质量报告盖到主库头上，而两者口径完全不同
    # （主库 13,119 篇会议论文 / 周报是 arXiv 日更流）。
    out_dir = ROOT / cfg["paths"]["data"] / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "quality.json"
    md_path = out_dir / "quality.md"

    prev = None
    if json_path.exists():
        try:
            prev = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = None
    if prev:
        report["delta"] = _delta(prev, report)

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    md = render_markdown(report)
    if report.get("delta"):
        md = md.replace("> 与上一份报告对比。首次生成时为空。", _delta_md(report["delta"]))
    md_path.write_text(md, encoding="utf-8", newline="\n")
    return json_path, md_path


def _delta(prev: dict[str, Any], cur: dict[str, Any]) -> dict[str, Any]:
    """和上一份报告比。**周更的意义全在这里**——
    绝对值说明不了"这周有没有进展"，只有增量能。"""
    keys = [
        ("metadata", "total"), ("metadata", "arxiv"), ("metadata", "abstract"),
        ("text", "with_text"), ("text", "too_short"), ("pdf", "files"),
        ("gaps", "no_text_total"), ("gaps", "rl_without_text"),
    ]
    out: dict[str, Any] = {}
    for section, key in keys:
        before = (prev.get(section) or {}).get(key)
        after = (cur.get(section) or {}).get(key)
        if before is None or after is None:
            continue
        out[f"{section}.{key}"] = {"before": before, "after": after, "delta": after - before}
    return out


def _delta_md(delta: dict[str, Any]) -> str:
    lines = ["| 指标 | 上次 | 本次 | 变化 |", "|---|---|---|---|"]
    for key, v in delta.items():
        sign = "+" if v["delta"] > 0 else ""
        lines.append(f"| `{key}` | {v['before']} | {v['after']} | {sign}{v['delta']} |")
    return "\n".join(lines)


def run(store: PaperStore, cfg: dict[str, Any]) -> dict[str, Any]:
    report = build(store, cfg)
    json_path, md_path = write(report, cfg)
    summary = {
        "report": json_path.relative_to(ROOT).as_posix(),
        "markdown": md_path.relative_to(ROOT).as_posix(),
        "papers": report["metadata"]["total"],
        "rl": report["metadata"]["rl"],
        "arxiv_rate": report["metadata"]["arxiv_rate"],
        "abstract_rate": report["metadata"]["abstract_rate"],
        "text_files": report["text"]["with_text"],
        "text_unaccounted": report["text"]["unaccounted_in_manifest"],
        "text_mb": report["text"]["total_mb"],
        "pdf_files": report["pdf"]["files"],
        "pdf_mb": report["pdf"]["total_mb"],
        "no_text": report["gaps"]["no_text_total"],
        "rl_without_text": report["gaps"]["rl_without_text"],
        "anomaly_rate": report["anomaly"]["rate"],
        "anomaly_ok": report["anomaly"]["ok"],
        "accounting_ok": report["accounting"]["ok"],
        "pdf_accounting_ok": report["accounting"]["pdf_ok"],
        "pdf_missing_ok": report["accounting"]["pdf_missing_ok"],
    }
    if report.get("delta"):
        summary["delta"] = report["delta"]
    return summary
