"""RL 相关性筛选。

**刻意不使用 LLM**。按 1500 篇命中的量级、每篇约 1000 token 估算，LLM 分类
一次要 150 万 token，是整个代码库成本的十倍；而且你每调一次判定标准就要
重跑一遍，成本线性叠加。关键词打分一次全库只要几秒、零成本，且结果可解释
（``rl.reasons`` 逐条记录命中了什么）。

什么时候才值得上 LLM：等你确信用关键词筛出的边界情况太多、且愿意为一次
全量重跑付那份钱。那时再加 ``rllib classify --llm``，让它显式存在，
而不是悄悄烧在每次更新里。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from ..schema import CanonicalPaper, RLAnnotation

#: 单篇最多保留多少条命中理由。不加限制的话，命中 30 个词会把 jsonl 撑肿。
_MAX_REASONS = 8

#: 命中强信号词的分值（只用于排序，判定不依赖它）
_STRONG_SCORE = 2.0
_MID_SCORE = 0.5
_TOPIC_SCORE = 2.0
_SCORE_CAP = 9.9


@lru_cache(maxsize=4096)
def term_pattern(term: str) -> re.Pattern[str]:
    """词边界匹配。短语（含空格）按 ``\\s+`` 匹配，所以 "offline rl" 也命中 "offline  RL"。

    必须用边界而不是裸子串：缩写词尤其危险——``ppo`` 会命中 "su**ppo**rt"，
    ``sac`` 会命中 "sa**c**rifice"，``a2c``/``dqn`` 同理。这类误判不会报错，
    只会让一大批无关论文悄悄被打上 RL 标签。

    ``subfields.py`` 复用本函数（见那里对两级词表的说明），所以改这里要两边都想一遍。
    """
    parts = term.lower().split()
    body = r"\s+".join(re.escape(p) for p in parts)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


@dataclass
class RLFilter:
    strong: list[str] = field(default_factory=list)
    mid: list[str] = field(default_factory=list)
    negative: list[str] = field(default_factory=list)
    topic_strong: list[str] = field(default_factory=list)
    mid_threshold: float = 2.0

    @classmethod
    def from_yaml(cls, path: Path, mid_threshold: float = 2.0) -> RLFilter:
        with Path(path).open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(
            strong=[t.lower() for t in data.get("strong", [])],
            mid=[t.lower() for t in data.get("mid", [])],
            negative=[t.lower() for t in data.get("negative", [])],
            topic_strong=[t.lower() for t in data.get("topic_strong", [])],
            mid_threshold=mid_threshold,
        )

    # ------------------------------------------------------------------ #

    def score(self, paper: CanonicalPaper) -> RLAnnotation:
        # 人工改判优先。这是唯一能覆盖自动判定的东西。
        if paper.rl.manual_override is not None:
            ann = paper.rl.model_copy(deep=True)
            ann.is_rl = bool(paper.rl.manual_override)
            ann.reasons = [*ann.reasons, "manual_override"]
            return ann

        haystack = f"{paper.title} {paper.abstract or ''}".lower()
        topics_lower = {t.lower() for t in paper.topics}

        score = 0.0
        reasons: list[str] = []
        hit_negative: str | None = None
        hit_strong = False

        for term in self.negative:
            if term_pattern(term).search(haystack):
                hit_negative = term
                break

        if hit_negative:
            return RLAnnotation(
                is_rl=False,
                score=0.0,
                reasons=[f"neg:{hit_negative}"],
                subfield=paper.rl.subfield,
                manual_override=paper.rl.manual_override,
            )

        for term in self.strong:
            if term_pattern(term).search(haystack):
                hit_strong = True
                score = min(score + _STRONG_SCORE, _SCORE_CAP)
                reasons.append(f"strong:{term}")

        for term in self.topic_strong:
            if term in topics_lower:
                hit_strong = True
                score = min(score + _TOPIC_SCORE, _SCORE_CAP)
                reasons.append(f"topic:{term}")

        for term in self.mid:
            if term_pattern(term).search(haystack):
                score = min(score + _MID_SCORE, _SCORE_CAP)
                reasons.append(f"mid:{term}")

        is_rl = hit_strong or score >= self.mid_threshold
        return RLAnnotation(
            is_rl=is_rl,
            score=round(score, 2),
            reasons=reasons[:_MAX_REASONS],
            subfield=paper.rl.subfield,
            manual_override=paper.rl.manual_override,
        )


def apply(store, filt: RLFilter) -> dict[str, object]:
    """对全库重新打分。返回聚合统计——不回显任何论文内容。"""
    before = sum(p.rl.is_rl for p in store.papers.values())
    changed = 0
    for paper in store.papers.values():
        new = filt.score(paper)
        if new.model_dump() != paper.rl.model_dump():
            paper.rl = new
            changed += 1
    after = sum(p.rl.is_rl for p in store.papers.values())
    total = len(store.papers) or 1
    return {
        "total": len(store.papers),
        "rl_before": before,
        "rl_after": after,
        "rescored": changed,
        "rl_pct": round(100 * after / total, 1),
    }
