"""覆盖范围声明（阶段 8.3）。

**这一页回答的问题只有一个：这个库的边界在哪。**

它存在的理由是 PLAN 第十节定下的目标函数之一——"方向的完整性"。而完整性
最危险的失败方式不是"库里有一条错的"，是"库里**缺**了一整块，而没人知道"。
缺掉的东西不会表现为错误，只表现为它不在。所以边界必须显式写出来：

    一个诚实的边界比一个假装完整的库有用。

设计上分两半，**不要把它们混起来**：

* **库内算得出来的**（覆盖率、缺 arXiv ID、无全文、citations 全空）——
  这里现算。写死就会过期，而这份页面的全部价值就在于它可信。
* **外部知识**（世界上还有哪些源没进来）——`data/coverage.yaml`，人维护。
  用库内数据归纳不出来：只能看见已经进来的东西（同 rl_subfields.yaml 的陷阱）。
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from .. import ROOT
from ..store import PaperStore
from .quality import text_on_disk

log = logging.getLogger(__name__)

#: 外部维护的"没纳入的源"清单。与词表同规矩：外置、可随时改。
ABSENT_YAML = "data/coverage.yaml"


def load_absent(path: Path | None = None) -> list[dict[str, Any]]:
    """读缺源清单。**文件不存在时返回空表并告警，不抛异常**——

    这一页是文档，不是数据管道。缺了 yaml 就少一段，不该让整个 `render` 挂掉。
    """
    p = Path(path) if path else ROOT / ABSENT_YAML
    if not p.exists():
        log.warning("覆盖范围声明：找不到 %s，缺源清单这一节会是空的", p)
        return []
    with p.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return [dict(row) for row in (data.get("absent") or [])]


def build(store: PaperStore, cfg: dict[str, Any]) -> dict[str, Any]:
    """库内事实 + 外部缺源清单。全是计数，不含论文正文。"""
    papers = list(store.papers.values())
    total = len(papers)

    by_venue: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "rl": 0, "arxiv": 0, "text": 0}
    )
    for p in papers:
        v = by_venue[p.venue]
        v["total"] += 1
        if p.is_rl_effective:
            v["rl"] += 1
        if p.arxiv_id:
            v["arxiv"] += 1

    # 全文覆盖走磁盘，不读 manifest（铁律 7）。
    on_disk = text_on_disk(store, ROOT / cfg["paths"]["text"])
    for p in papers:
        if p.paper_id in on_disk:
            by_venue[p.venue]["text"] += 1

    n_arxiv = sum(1 for p in papers if p.arxiv_id)
    n_text = len(on_disk)
    n_rl = sum(1 for p in papers if p.is_rl_effective)
    # ``citations`` 基本为空（见 coverage.yaml 里那条缺源）。这里现算而不是写死，
    # 将来接入来源之后页面会自动改口——写死就会撒谎。
    #
    # 判据是"**够不够用来排序**"，不是"是不是恰好 0"。写成 ``n_cit == 0`` 曾经让
    # 整段静默消失：库里已经有 4 篇带引用数，于是页面不再提这件事——
    # 而 4/13119 和 0 一样不能用来排序，缺口一点没变。这正是本页要防的那种
    # "因为不精确所以悄悄不再报警"。用比例判，不用零判。
    n_cit = sum(1 for p in papers if p.citations is not None)
    cit_pct = round(n_cit / total * 100, 1) if total else 0.0
    years = Counter(p.year for p in papers)

    return {
        "generated": date.today().isoformat(),
        "total": total,
        "rl": n_rl,
        "rl_pct": round(n_rl / total * 100, 1) if total else 0.0,
        "years": [{"year": y, "n": n} for y, n in sorted(years.items())],
        "single_year": len(years) == 1,
        "arxiv": n_arxiv,
        "arxiv_pct": round(n_arxiv / total * 100, 1) if total else 0.0,
        "text": n_text,
        "text_pct": round(n_text / total * 100, 1) if total else 0.0,
        # 能抓到全文的上限就是"有 arXiv ID"的那些——分母用它才有意义。
        "text_of_arxiv_pct": round(n_text / n_arxiv * 100, 1) if n_arxiv else 0.0,
        "citations": n_cit,
        "citations_pct": cit_pct,
        # 覆盖不到一半就认为"不能用来排序"。阈值是判断力，不是数学——
        # 重点是这个判断必须由**比例**触发，不能由"恰好为零"触发。
        "citations_thin": cit_pct < 50.0,
        "by_venue": [
            {"venue": v, **counts}
            for v, counts in sorted(by_venue.items(), key=lambda kv: -kv[1]["total"])
        ],
        # 缺源清单按 cfg 定位，不写死 ABSENT_YAML：它回答的是"**这个库**还有哪些
        # RL 源没进来"，而周报库（arXiv 单一源）的答案与主库完全不同，
        # 共用一份会让页面替另一个库答话。文件不存在时 load_absent 告警返回空表。
        "absent": load_absent(ROOT / cfg["paths"]["data"] / "coverage.yaml"),
    }
