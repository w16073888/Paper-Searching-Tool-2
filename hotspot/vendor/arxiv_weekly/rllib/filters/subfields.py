"""RL 子方向打标 —— 「方向的完整性」的落地机制。

写进 ``paper.rl.subfield``，与 ``rl.is_rl`` 同属一个字段对象，所以由
``rllib filter --rl`` 一并重算、一并落盘（见 :func:`apply`）。这样不新增命令，
也不引入第二个写 ``rl`` 的长任务。

为什么必须有这个东西
--------------------

关键词筛选器漏掉的子方向**不会表现为一条错误记录，而是表现为它根本不在那里**——
没有任何指标会报警。子方向计数是唯一能把这种缺失变成可见数字的手段。

闸门（gate）
------------

打标**不**以 ``is_rl`` 为前提。若以它为前提，筛选器漏掉的 RL 论文会被一并挡在
门外，而找出这些漏网论文恰恰是本模块的首要目的。

闸门定义：命中任意一条 ``keywords``（专有词）**或** ``is_rl`` 为真。
``weak``（通用词，如 "planning"/"scheduling"）本身不开闸门，只在闸门已开时计入——
否则 ``rl_for_systems`` 会收进几千篇非 RL 的系统论文，趋势曲线变成噪声。
详见 ``data/rl_subfields.yaml`` 顶部的说明。

于是「有专有词命中但 ``is_rl`` 为假」的论文自动浮现为**筛选盲区候选**，
由 ``rllib report subfields`` 汇总。这是本设计的核心产出。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..schema import CanonicalPaper
from .rl import term_pattern


def _phrase_pattern(terms: list[str]) -> re.Pattern[str] | None:
    """把一组词编译成一个 alternation。长词优先，避免 ``\\birl\\b`` 抢在别的词前。"""
    terms = [t.strip().lower() for t in terms if t and t.strip()]
    if not terms:
        return None
    terms.sort(key=len, reverse=True)
    return re.compile("|".join(term_pattern(t).pattern for t in terms))


def _haystack(paper: CanonicalPaper) -> str:
    """只扫标题 + 摘要。

    **不扫全文。** 正文里一句 "we compare against PPO" 会把一大批不相干的论文
    拉进来，把子方向计数彻底稀释掉。摘要才是作者自报的定位。
    """
    return f"{paper.title} {paper.abstract or ''}".lower()


@dataclass(frozen=True)
class SubfieldRule:
    key: str
    label: str
    strong: re.Pattern[str] | None   # 专有词：命中即开闸门
    weak: re.Pattern[str] | None     # 通用词：仅在闸门已开时计入


@dataclass
class SubfieldTagger:
    rules: list[SubfieldRule] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> SubfieldTagger:
        with Path(path).open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        block = data.get("subfields") or {}
        rules = [
            SubfieldRule(
                key=key,
                label=(spec or {}).get("label", key),
                strong=_phrase_pattern((spec or {}).get("keywords", [])),
                weak=_phrase_pattern((spec or {}).get("weak", [])),
            )
            for key, spec in block.items()
        ]
        return cls(rules=rules)

    @property
    def labels(self) -> dict[str, str]:
        return {r.key: r.label for r in self.rules}

    # ------------------------------------------------------------------ #

    def tag(self, paper: CanonicalPaper) -> list[str]:
        """返回该篇命中的子方向 key（已排序）。空列表表示未命中任何子方向。"""
        hay = _haystack(paper)
        strong_hits: list[str] = []
        weak_hits: list[str] = []

        for rule in self.rules:
            if rule.strong is not None and rule.strong.search(hay):
                strong_hits.append(rule.key)      # 专有词：同时开闸门
            elif rule.weak is not None and rule.weak.search(hay):
                # 通用词先记下，闸门可能由**后面的**规则打开，所以这时还不能丢
                weak_hits.append(rule.key)

        if not strong_hits and not paper.is_rl_effective:
            return []                              # 闸门未开，通用词一律丢弃
        return sorted(set(strong_hits) | set(weak_hits))

    def blind_spots(self, papers) -> list[CanonicalPaper]:
        """有专有子方向词命中、但 ``is_rl`` 判为否的论文——筛选器的漏网候选。

        「候选」是严谨的说法：命中的可能确实是 RL 论文，也可能是词表误伤
        （例如 "dpo" 这类缩写）。所以这个方法只负责**把候选捞出来给人看**，
        不自动改判。改判走 ``rl.manual_override``，那是人工区。
        """
        out = []
        for paper in papers:
            if paper.is_rl_effective:
                continue
            hay = _haystack(paper)
            if any(r.strong is not None and r.strong.search(hay) for r in self.rules):
                out.append(paper)
        return out


def apply(store, tagger: SubfieldTagger) -> dict[str, object]:
    """对全库重打子方向标签。**只输出聚合计数**（CLAUDE.md token 纪律）。

    与 ``rl.is_rl`` 同属 ``rl`` 字段，所以必须和 :func:`rllib.filters.rl.apply`
    在**同一次 save** 里落盘，否则两者会互相覆盖。
    """
    from collections import Counter

    dist: Counter[str] = Counter()
    blind: Counter[str] = Counter()
    blind_papers = 0
    tagged = 0

    for paper in store.papers.values():
        keys = tagger.tag(paper)
        if keys != paper.rl.subfield:
            # 就地改，调用方最后统一 save(owned_fields=("rl",))
            paper.rl = paper.rl.model_copy(update={"subfield": keys})
        if keys:
            tagged += 1
            dist.update(keys)
            if not paper.is_rl_effective:
                blind_papers += 1
                blind.update(keys)

    return {
        "tagged": tagged,
        "subfields": len(tagger.rules),
        "distribution": dict(dist.most_common()),
        # blind_papers 是**论文数**，blind_by_subfield 是 (论文,子方向) 对数——
        # 后者总和会大于前者，别把两个数混着读。
        "blind_spot_papers": blind_papers,
        "blind_spot_by_subfield": dict(blind.most_common()),
    }
