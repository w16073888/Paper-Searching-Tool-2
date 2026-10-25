"""周报卡片范围 —— 一条判定，两个调用方。

范围定义
--------

``is_rl`` **有效地**为真 **或** 命中 RL 子方向专有词（``rl.subfield`` 非空）。

这正是 ``rllib/filters/subfields.py`` 里打标的**闸门**（那里的原话是
「命中任意一条 ``keywords``（专有词）**或** ``is_rl`` 为真」）。区别只在于：
这里读的是 ``filter --rl`` **落盘之后的存量字段**，不重新打标 —— 所以词表改了、
``filter`` 重跑之后，这里自动跟着变，不存在两份判据漂移的问题。

⚠️ 用 ``is_rl_effective`` 而不是裸 ``is_rl``：``manual_override``（人工区）优先。
   规则本体在 ``schema.CanonicalPaper.is_rl_effective``，那里写着「下游一律用这个
   属性，免得各处理解不一致」—— 本模块是下游，照办。

数字来历（2026-09-17 实测，周报库 537 篇）
-----------------------------------------

============  =====
``is_rl``     47
``subfield``  60
交集          34
**并集**      **73**
============  =====

用户当日拍「**73 吧**」（日志 §14.9.8）。

为什么不能只用 ``is_rl``
------------------------

那 26 篇盲区论文是**用本行行话写**的 —— 摘要里根本不出现 "reinforcement
learning"，所以 ``is_rl`` 判假。它们的分数 **0.0～0.5**、离阈值 2.0 隔着整条沟，
**不是擦边，是离词表**。于是「没命中」在这里等于「**没被看见**」，
正是本库签名病（「没有」vs「没读到」）的位置。

为什么单独一个模块
------------------

这条判定有**两个**调用方：抽取的增量调度（``scripts/weekly_sections.py``）与
卡片渲染。本库最贵的一次教训就是「**同一条判定有两处各写一份**」
（probe v2→v3 修的根因，见 ``rllib/extract/gap.py`` 开头）——
所以它只写在这里，两个调用方都 import 这一个。
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["BOTH", "IS_RL", "OUT", "SUBFIELD", "in_scope", "scope_of"]

#: ``scope_of`` 的四个返回值。``OUT`` 不是"缺省"、是**明确的"不在范围内"**。
BOTH = "both"
IS_RL = "is_rl"
SUBFIELD = "subfield"
OUT = ""


def _rl(paper: Any) -> Mapping[str, Any]:
    """取 ``rl`` 字段。接受 ``papers.jsonl`` 的 dict 行，也接受 ``CanonicalPaper``。

    两种形状都要收，是因为两个调用方拿到的不是同一种东西：脚本读 jsonl 得 dict，
    渲染器走 ``PaperStore`` 得 pydantic 模型。**在这里收口，别让调用方各自解包。**
    """
    rl = paper.get("rl") if isinstance(paper, Mapping) else getattr(paper, "rl", None)
    if rl is None:
        return {}
    if isinstance(rl, Mapping):
        return rl
    if hasattr(rl, "model_dump"):          # pydantic v2
        return rl.model_dump()
    return vars(rl)


def scope_of(paper: Any) -> str:
    """这篇**为什么**在范围内。``OUT``（空串）表示不在范围内。

    返回值直接可以进卡片的「来源」栏与汇总行 —— 让"为什么收它"这件事**可见**，
    而不是只有一个布尔。
    """
    rl = _rl(paper)
    # ``manual_override`` 优先，规则同 ``CanonicalPaper.is_rl_effective``
    override = rl.get("manual_override")
    effective = bool(override) if override is not None else bool(rl.get("is_rl"))
    sub = bool(rl.get("subfield"))
    if effective and sub:
        return BOTH
    if effective:
        return IS_RL
    if sub:
        return SUBFIELD
    return OUT


def in_scope(paper: Any) -> bool:
    """这篇要不要出卡片。"""
    return scope_of(paper) != OUT
