r"""周报 concern 人眼验收页 —— 每周一份 ``<周>/review.md``（周报库内）。

这一页回答什么
--------------

**「抽出来的这段，读起来能不能回答『这篇还能往下做什么』。」**

这是本库唯一一条**机器判据证明不了**的验收项（那句话的原文在
``scripts/weekly_sections.py`` 的模块 docstring 里）。卡片本身就是人眼面，但
一百多片按 ``arxiv_id`` 平铺**不告诉你先看哪片**—— 这一页就是那个排序。

按可疑度排，不按对错排
----------------------

机器判据在**两件事**上会出错，而它自己不会说：

1. **长度带外**（``outcome=degenerate``）—— 多半是排版残留，或吞掉了下级小节；
2. **挑中的标题里没有缺口/展望词** —— 挑中的是 ``Conclusion`` / ``Discussion``
   这类**泛总结**节。

⚠️ 第 2 条**是筛选信号，不是判据**：作者在 ``Conclusions`` 里写 future work 的
不少（实测口径见 ``ROADMAP.md`` 6.4 W）。判据仍然是那句
「**缺口**或**展望**都合格，**纯总结**不合格」。分组只决定**先看哪一片**，
不决定**哪一片不合格**。

为什么以磁盘为准
----------------

真相源是 ``sections.jsonl``（周报库内），**不是某一次跑的内存**。这一页取代的
``sections_review.md`` 由 ``weekly_sections.py`` 从 ``new_rows`` 写出，只覆盖
**最近一次跑**的那些篇——09-17 那一轮的 47 篇就永远不在里面（那是"少而无人知"
的形状）。这一页每次从盘上重算**全部** ⇒ 重跑逐字节幂等。

这一页是**产物**，不是笔记
--------------------------

每次重跑**整份覆盖**。批注写在**卡片**上：卡片走 ``upsert_note``，机器区/人工区
隔离，读者写的东西**永不丢**（见 ``rllib/render/weekly_card.py``）。**别把这一页
当笔记本**——它不是 ``upsert_note`` 写的，你写在上面的字下一次跑就没了。

**Token 纪律**：终端只打一行聚合 JSON，绝不回显标题或正文。
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

from rllib.atomic import write_atomic
from rllib.extract import gap
from rllib.extract.latex import has_gap_word
from rllib.render import weekly_card as wc
from rllib.weekly_scope import in_scope

__all__ = ["REVIEW_NAME", "group_of", "render_review"]

REVIEW_NAME = "review.md"

#: 四个分组。**常量值就是页面上第几组**——顺序即优先级（先看哪一组）。
G_DEGENERATE = 1
G_SUSPECT = 2
G_OK = 3
G_NO_TEXT = 4
GROUP_ORDER = (G_DEGENERATE, G_SUSPECT, G_OK, G_NO_TEXT)

GROUP_TITLE = {
    G_DEGENERATE: "长度异常",
    G_SUSPECT: "可疑：标题里没有缺口/展望词",
    G_OK: "标题带缺口/展望词",
    G_NO_TEXT: "没抽出正文（一行一篇）",
}

#: 每组开头那句话。**改措辞只动这里**，分组逻辑在 :func:`group_of`。
GROUP_SAY = {
    G_DEGENERATE:
        "`outcome=degenerate` = 长度落在合理区间外，多半是排版残留或吞掉了下级小节。"
        "**这一段到底是不是缺口，机器判不了——得你读。**",
    G_SUSPECT:
        "挑中的是 `Conclusion` / `Discussion` 这类**泛总结**节。"
        "⚠️ **这一组不是「不合格」清单**——作者在 `Conclusions` 里写 future work 的不少，"
        "它只说明**挑错的概率更高，所以先看**。",
    G_OK:
        "挑中的标题里带 `limitation` / `future` / `open problem`。"
        "**标题对不保证正文对**（也可能只有一句空话），只是先看上面两组。",
    G_NO_TEXT:
        "这些篇**没有正文可读**。⚠️ 表里那句话**分两拨，别读成一回事**："
        "`no_section` 是**关于论文的结论**（作者确实没写），"
        "`no_source` / `no_text` / `failed` 是**我们自己的缺口**（没拿到 / 没切出来）。",
}

#: 页首那段判据。**一个字都不是程序生成的**，就是那条人眼判据本身。
CRITERION = """\
> **判据**：缺口**或**展望都合格，**纯总结不合格**。
> 机器判不了这一条 —— 只有人能回答：**这段读起来，能不能说明「这篇还能往下做什么」**。
>
> 下面按**可疑度**排，**不按对错排**。第 2 组（标题里没有缺口/展望词）**不是**
> 不合格清单：作者在 `Conclusions` / `Discussion` 里写 future work 的不少，它只决定
> **先看哪一片**。反过来，第 3 组标题带缺口词，也不保证正文一定合格。
>
> 这一页是**产物**：每次重跑整份覆盖。**批注请写在卡片上**
> （`<周>/concern/<arxiv_id>.md` 的 `<!-- AUTO:END -->` 之后），
> 卡片是机器区/人工区隔离的，重渲不会抹掉你写的东西。"""

#: 读回核对的报错里最多列几个 key。**满了不影响对错**，只是别把一行 JSON 刷爆。
_MISSING_KEPT = 5


def group_of(sec: dict | None) -> int:
    """这一篇归哪一组。**全覆盖**：认不出来的 outcome 一律进第 4 组。

    ⚠️ 第 4 组的表里用的是 :data:`weekly_card.OUTCOME_SAY`，**没登记的 outcome
    在那儿会自带一行告警**——所以"新加了一个桶"不会静默混进表里。
    """
    outcome = (sec or {}).get("outcome", "")
    if outcome == "degenerate":
        return G_DEGENERATE
    if outcome == "ok":
        # 标题只是**筛选信号**（见模块 docstring）：没有缺口词 ⇒ 排在前面。
        return G_OK if has_gap_word((sec or {}).get("picked") or "") else G_SUSPECT
    return G_NO_TEXT


def _cell(text: str) -> str:
    """表格单元格。⚠️ 论文标题里**可以**有 ``|``，不转义会把一整行劈成两格。"""
    return text.replace("|", "\\|").replace("\n", " ")


def _entry(paper: dict, sec: dict | None, latest: str) -> list[str]:
    """一组里的一片：标题行 + 来源标签 + **正文全文**。

    ⚠️ 正文**绝不截断**——判据是"读起来能不能回答"，截断就判不了。
    """
    key = paper.get("arxiv_id") or ""
    picked = (sec or {}).get("picked") or ""
    star = " ★" if (sec or {}).get("extracted") == latest else ""
    return [
        f"### {key} · {picked}（{(sec or {}).get('picked_len') or 0} 字）{star}",
        "",
        f"- 标题：{paper.get('title') or ''}",
        # 来源标签**复用卡片那一份**，不另写一句：读者在卡片与这一页上看到的
        # 必须是同一句话，否则"我该信哪个"就会变成一个新的问题。
        f"- {wc._provenance_line(sec)}",
        "",
        ((sec or {}).get("text") or "").strip(),
        "",
        "---",
        "",
    ]


def _table(rows: list[tuple[dict, dict | None]]) -> list[str]:
    """第 4 组：一行一篇。没有正文，只交代**为什么没有**。"""
    out = ["| arxiv_id | outcome | 说明 | 标题 |", "|---|---|---|---|"]
    for paper, sec in rows:
        outcome = (sec or {}).get("outcome", "")
        say = wc.OUTCOME_SAY.get(
            outcome, f"⚠️ 未登记的 outcome `{outcome}` —— 渲染器不认识它。")
        out.append(f"| {_cell(paper.get('arxiv_id') or '')} | `{_cell(outcome)}` "
                   f"| {_cell(say)} | {_cell(paper.get('title') or '')} |")
    out.append("")
    return out


def _render_week(week: str, groups: dict[int, list], total: int,
                 latest: str, generated: str) -> str:
    counts = " / ".join(str(len(groups[g])) for g in GROUP_ORDER)
    lines = [
        f"# 周报 concern 人眼验收 · {week}",
        "",
        f"生成 {generated} · 本周在范围内 **{total}** 篇 · 分组 **{counts}**（= {total}）",
    ]
    if latest:
        lines.append(f"最近一次抽取 {latest}（下面标 ★ 的那些）")
    lines += ["", CRITERION, ""]

    for g in GROUP_ORDER:
        rows = groups[g]
        lines += [f"## 第 {g} 组 · {GROUP_TITLE[g]} · {len(rows)} 片", "", GROUP_SAY[g], ""]
        if not rows:
            # 空组**必须写出来**。少一节和"这一组没有"在页面上长得一模一样，
            # 而前者读起来像是渲染器坏了。
            lines += ["*（本组这一周没有。）*", ""]
        elif g == G_NO_TEXT:
            lines += _table(rows)
        else:
            for paper, sec in rows:
                lines += _entry(paper, sec, latest)
    return "\n".join(lines) + "\n"


def render_review(weekly_dir: Path, *, generated: str | None = None) -> dict:
    """写 ``<周>/review.md``（周报库内）。只返回聚合计数，**不打印任何论文内容**。

    ``generated`` 可传入（默认今天）：把它钉住才能让不变量检查断言"跑两次逐字节
    相同"。日期**必须**留在页面上——这一页最大的失败模式就是"看着像刚算的账"。

    排序 = ``(outcome, arxiv_id)``，一个**全序**，所以重跑逐字节幂等。
    第 1/2/3 组内 outcome 是常量，于是这一条退化成"按 arxiv_id 升序"；
    第 4 组保留 outcome 优先，让同一原因的篇相邻（那张表有「说明」一列）。
    """
    papers, sections, _announce = wc.load_inputs(weekly_dir)
    generated = generated or date.today().isoformat()

    in_week: dict[str, list[tuple[dict, dict | None]]] = {}
    no_week: list[str] = []
    for paper in papers:
        if not in_scope(paper):
            continue
        key = paper.get("arxiv_id") or ""
        week = wc.week_of(paper.get("first_seen") or "")
        if not week:
            # ⚠️ 与 `weekly_render.py` 同一条：不知道写进哪一周就**不写**，
            # 但必须报出来——这一篇会**不在任何一页上**，而页面上看不出来。
            no_week.append(key)
            continue
        in_week.setdefault(week, []).append((paper, sections.get(key)))

    latest = max((r.get("extracted") or "" for r in sections.values()), default="")
    outcomes: Counter[str] = Counter()
    by_week: dict[str, dict] = {}

    for week in sorted(in_week):
        rows = in_week[week]
        groups: dict[int, list] = {g: [] for g in GROUP_ORDER}
        for paper, sec in rows:
            groups[group_of(sec)].append((paper, sec))
        for g in GROUP_ORDER:
            groups[g].sort(key=lambda ps: ((ps[1] or {}).get("outcome", ""),
                                           ps[0].get("arxiv_id") or ""))
            outcomes.update((sec or {}).get("outcome", "") for _, sec in groups[g])

        page = _render_week(week, groups, len(rows), latest, generated)
        path = weekly_dir / week / REVIEW_NAME
        write_atomic(path, page.encode("utf-8"))
        by_week[week] = {
            "rows": len(rows),
            "groups": {str(g): len(groups[g]) for g in GROUP_ORDER},
            "bytes": len(page.encode("utf-8")),
        }

    # 恒等式（铁律 7）：每一篇落进**恰好一个**组。`group_of` 是全定义的，所以这条
    # 现在几乎必然成立——**它防的是以后有人在分组里加一句 `continue`**（那会让某篇
    # 两边都不出现，而页面上只是少一片）。
    # ⚠️ 真正能变红的是下面两条：`unknown_outcomes`（加了新桶没登记进 `gap.OUTCOMES`）
    # 与 `no_week`（这篇不在任何一页上）。**别把恒等式当唯一防线。**
    identity_groups_ok = all(sum(v["groups"].values()) == v["rows"] for v in by_week.values())
    unknown = sorted(set(outcomes) - set(gap.OUTCOMES))
    # 盘上有抽取记录、但它对应的篇不在这一页上（词表改过之后会这样）。**不为零不是错**，
    # 只是这一页不会写它——报出来是为了让"少了一片"这件事有个出处，而不是靠猜。
    on_page = {p.get("arxiv_id") or "" for rows in in_week.values() for p, _ in rows}
    orphan = len(set(sections) - on_page)
    return {
        "generated": generated,
        "weeks": sorted(by_week),
        "by_week": by_week,
        "rows": sum(v["rows"] for v in by_week.values()),
        "groups": {str(g): sum(v["groups"][str(g)] for v in by_week.values())
                   for g in GROUP_ORDER},
        "latest_extracted": latest,
        "identity_groups_ok": identity_groups_ok,
        "unknown_outcomes": unknown,
        "no_week": no_week,
        "orphan_sections": orphan,
        "sections_total": len(sections),
        "data_dir": str(weekly_dir),
    }
