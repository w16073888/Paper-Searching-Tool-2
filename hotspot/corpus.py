"""24+25 合并语料源（方案 A，2026-09-24）。

读 G 盘两份 `papers.jsonl`（`G:\\rl_data24\\资源` + `G:\\rl_data\\资源`），并成
一套 22,747 篇的 `MergedCorpus`，接口与 `dasc.Corpus` 兼容（`list_papers` / `get`
/ `search` / `stats` + no-op `lib`），供 embedding、工具和 UI 使用。

**保留 `year` 分组能力**：每篇 `Paper.year`（2024/2025）都在；`list_papers(year=...)`
可按年筛；簇的跨会三态默认为**跨年混合**，但 `VenueProfile.counts_by_year` 保留
分年计数（见 `structure/venue_matrix.py`）。主口径仍跨会、不做跨年，24 仅辅助验证。

**为什么不复制 dasc**：继续复用 dasc 的聚类/LLM 等设施；这里只做一个"合并+读 jsonl"
的薄适配器，字段映射到现有 `Paper` 契约。
"""

from __future__ import annotations

import json
import re
import datetime as _dt
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .config import load_config
from . import HOTSPOT_ROOT  # noqa: E402 —— 项目根，供相对语料路径解析

__all__ = ["Paper", "MergedCorpus", "arXivCorpus", "open_corpus", "open_arxiv_corpus"]

#: 英文子串检索用（大小写不敏感，按词匹配以略提升召回）
_WORD = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class Paper:
    """与 dasc 的 Paper 契约对齐的最小实现（保留 `year`）。

    额外字段（周报 arXiv 语料专用，会议库为空/默认值）：
    `topics`=arXiv 分类（cs.LG/stat.ML/cs.AI）、`first_seen`=首次入库日期 `YYYY-MM-DD`。
    """

    paper_id: str
    title: str = ""
    abstract: str = ""
    year: Optional[int] = None
    venue: str = ""
    arxiv_id: str = ""
    is_rl: bool = False
    subfields: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    first_seen: str = ""

    @property
    def key(self) -> str:
        return self.paper_id

    @property
    def week_id(self) -> str:
        if not self.first_seen:
            return ""
        try:
            year, week, _ = _dt.date.fromisoformat(self.first_seen).isocalendar()
        except ValueError:
            return ""
        return f"{year}-W{week:02d}"


def _load_source(path: Path) -> list[Paper]:
    out: list[Paper] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rl = r.get("rl") or {}
            sub = tuple(s for s in (rl.get("subfield") or []) if s)
            out.append(Paper(
                paper_id=r.get("paper_id") or "",
                title=r.get("title") or "",
                abstract=r.get("abstract") or "",
                year=r.get("year"),
                venue=r.get("venue") or "",
                arxiv_id=r.get("arxiv_id") or "",
                is_rl=bool(rl.get("is_rl")),
                subfields=sub,
                topics=tuple(t for t in (r.get("topics") or []) if t),
                first_seen=r.get("first_seen") or "",
            ))
    return out


class MergedCorpus:
    """把 24+25 两份 jsonl 并成一套语料。`paper_id` 并集（交集为 0，去重兜底）。"""

    def __init__(self, sources: Iterable[str | Path] = (), *, fallback: bool = True):
        # `fallback=False` 时（arXivCorpus）不给配置兜底——避免空 sources 误回退到会议库。
        if not sources and fallback:
            cfg = load_config()
            sources = (cfg.get("corpus") or {}).get("sources") or []
        by_id: dict[str, Paper] = {}
        for s in sources:
            for p in _load_source(Path(s)):
                by_id.setdefault(p.paper_id, p)  # 先 24 后 25；冲突保留先来
        self._by_id = by_id
        self._papers = list(by_id.values())

    # ---- 查询 ---------------------------------------------------------- #

    def list_papers(self, limit: Optional[int] = None, pool: str | None = None,
                    rl_only: bool = False, year: int | None = None) -> list[Paper]:
        out = self._papers
        if rl_only:
            out = [p for p in out if p.is_rl]
        if year is not None:
            out = [p for p in out if p.year == year]
        if pool:
            key = _norm(pool)
            out = [p for p in out if any(_norm(s) == key for s in p.subfields)]
        if limit is not None:
            out = out[:limit]
        return out

    def get(self, paper_id: str) -> Paper | None:
        return self._by_id.get(paper_id)

    def search(self, query: str, k: int = 50) -> list[str]:
        """轻重：标题/摘要**词级匹配**（22,747 篇上每问 O(n)，够用）。

        ⚠️ 必须是「词级」而非子串：`"rloo" in hay` 会把 `overlook` 里的 `rloo`
        误当独立词 RLOO（实测 798 篇误报，真 RLOO 仅 4 篇）。这里对 hay 做
        `_WORD` 词切分后判子集，杜绝被长词吞并。
        """
        terms = [t for t in _WORD.findall(query.lower()) if t]
        if not terms:
            return []  # 无检索词（如纯中文方向）不匹配任何篇，而不是全命中
        term_set = set(terms)
        hits: list[str] = []
        for p in self._papers:
            hay = (p.title + " " + (p.abstract or "")).lower()
            if term_set <= set(_WORD.findall(hay)):
                hits.append(p.paper_id)
        return hits[:k]

    def stats(self) -> dict:
        return {
            "papers": len(self._papers),
            "is_rl": sum(1 for p in self._papers if p.is_rl),
            "pools": {s for p in self._papers for s in p.subfields},
        }

    def new_this_week(self, anchor: Optional[str] = None) -> list[Paper]:
        """基类兜底：会议库无 `first_seen`（周报字段），恒返回空。

        `arXivCorpus` 覆盖此方法做真实「本周新增」判定。
        """
        return []

    # ---- 兼容 `with c.lib`（no-op，读 jsonl 无写面） ---------------------- #

    @contextmanager
    def lib(self) -> Iterator[None]:
        yield


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def open_corpus(cfg=None) -> MergedCorpus:
    """从 config 的 `corpus.sources` 打开合并语料。相对路径以项目根(HOTSPOT_ROOT)为基准。"""
    sources = ((cfg or load_config()).get("corpus") or {}).get("sources") or []
    resolved = [str(Path(s) if Path(s).is_absolute() else HOTSPOT_ROOT / s)
                for s in sources]
    return MergedCorpus(resolved)


class arXivCorpus(MergedCorpus):
    """arXiv 周报连续流语料（独立源，共享引擎，2026-10-02-规划-接入arXiv周报与发布.md）。

    读周报 `数据/papers.jsonl`。`venue` 恒为 `"arXiv"`，另带 `topics`（arXiv 分类）
    与 `first_seen`（入库日期）。**不参与跨会三态**——热点轴换成分类/周窗口。
    检索/统计与 MergedCorpus 同构可复用。
    """

    def __init__(self, path=None):
        if not path:
            path = ((load_config().get("corpus") or {}).get("arxiv_source"))
        # 用 `[path] if path else []` 当 path 为空串时给空 sources；
        # `fallback=False` 使 MergedCorpus 绝不回退到会议库 sources（否则空参会误并成 22,747）。
        super().__init__([path] if path else [], fallback=False)

    def new_this_week(self, anchor: Optional[str] = None) -> list[Paper]:
        """`first_seen` 落在 `anchor`（默认今天）所在 ISO 周的论文。

        用于「本周新增」标注——周报每周 append，这是它相对老库的演进增量。
        """
        import datetime as _dt
        if anchor is None:
            anchor = _dt.date.today().isoformat()
        try:
            y, w, _ = _dt.date.fromisoformat(anchor).isocalendar()
        except ValueError:
            return []
        out = []
        for p in self._papers:
            if not p.first_seen:
                continue
            try:
                py, pw, _ = _dt.date.fromisoformat(p.first_seen).isocalendar()
            except ValueError:
                continue
            if (py, pw) == (y, w):
                out.append(p)
        return out


def open_arxiv_corpus(cfg=None) -> arXivCorpus:
    """从 config 的 `corpus.arxiv_source` 打开周报语料。相对路径以项目根为基准。"""
    src = ((cfg or load_config()).get("corpus") or {}).get("arxiv_source")
    if src:
        src = str(Path(src) if Path(src).is_absolute() else HOTSPOT_ROOT / src)
    return arXivCorpus(src)
