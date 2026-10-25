"""从全文里抽**作者自述的展望 / 缺口** —— 周报 concern 那一栏的产线实现。

为什么要有这个模块
------------------

这段逻辑原来**只活在探针里**（`scripts/probe_sections_v3.py` 的 `probe_one`）。
探针的产物是**原始证据**，按本库规矩冻结（v1/v2 一个字没动，v3 是第三个），
所以产线**不能去 import 探针**——那会让"改一个证据脚本"和"改生产行为"
变成同一件事，而那正是冻结要防的。

但也不能把 `probe_one` 抄一份过来：**v2→v3 那一轮修的根因，恰好就是
「同一条判定有两处各写一份」**（算不算缺口节、长度带怎么定）。抄一份等于
把那次的 bug 重新种一遍，而且是静默的。

所以分工是：
**判定本体留在 `latex.py`**（`GAP_SECTIONS` / `pick_gap_section` / `BODY_MIN` /
`BODY_MAX`，探针本来就调的是它），**编排收进这里一份**，产线调这里。

⚠️ 由此留下一份**有意的重复**：`probe_one` 与 `extract_one` 的编排同源。
不消除它，是因为探针必须冻结。**下次动这里的桶语义时要知道探针里还有一份旧的**
——它是证据，不是待同步的代码。

分桶（每一个桶都必须能把目标数分干净，铁律 7）
----------------------------------------------

==============  ==========================================================
``ok``          挑到了长度在带内的缺口 / 展望节
``degenerate``  挑到了，但长度在带外（过短＝排版残留 / 过长＝吞了下级小节）
``no_section``  正文里没有可用的缺口节标题
``no_text``     ``/src/`` 返回的是 PDF（周报线已放弃 PDF 路径）
``no_source``   arXiv 明说没有源码包（``/src/`` 终态 4xx）
``failed``      网络层没抓到 / 源码里没有可用正文
==============  ==========================================================

⚠️ ``no_source`` 与 ``failed`` **必须分开**：两者的返回值都是 ``None``，
只有按 ``Fetcher.stats["not_found"]`` 的增量才分得开。混成一个桶就是本库
反复栽的「**没有** vs **没读到**」——实测 ``2605.06850`` 的 ``/src/`` 是 404，
被记成"下载失败"记了两轮。

⚠️ ``no_section`` 与 ``no_text`` **也必须分开**：前者是「真没有」（作者没写展望），
后者是「没读到」（拿到 PDF、切不出标题）。卡片上这两句话完全不同——
一句是关于论文的结论，一句是关于我们自己的缺口。
"""

from __future__ import annotations

from ..http import Fetcher
from . import latex as latex_mod
from .text import arxiv_src_url

#: 全部结果桶。**它是判据不是描述**：任何一个 ``extract_one`` 的返回值，
#: ``outcome`` 必须落在这里面——落到外面说明有人加了新桶却没登记。
OUTCOMES: tuple[str, ...] = (
    "ok", "degenerate", "no_section", "no_text", "no_source", "failed",
)

#: ``no_section`` 时记多少个标题——用来区分「真没有」与「没读到」。
HEADINGS_KEPT = 40

#: 识别得出的排版宏。**只用来给残留打标签，不参与任何判定。**
#:
#: ⚠️ **这又是一张小清单，它必然有洞**——所以它绝不能被当成闸门用
#: （那正是 v2 那张 6 项清单翻车的方式）。落进 ``unexplained`` 的宏
#: **不代表它是错的**，只代表"这张小表不认识它"：作者自定义宏（``\Sstate``）
#: 与 ``latex._SYMBOLS`` 未覆盖的标准符号（``\top`` ``\star``）会混在同一栏。
_LAYOUT_MACROS = frozenset({
    "textcolor", "par", "newline", "pagebreak", "href", "fakepar",
    "ninept", "tt", "em", "vspace", "hspace", "noindent", "centering",
    "hrule", "smallskip", "text", "texttt", "textbf", "textit", "emph",
})


def macro_inventory(body: str) -> dict[str, int]:
    """正文里剩下的**每一个未展开的 ``\\宏``** → 出现次数。

    判据是「**还剩没剩下**」，不是「某几个宏在不在」。v2 用的是后者，
    于是拿到一个「42 篇残留全 0」的假绿灯——真实是 17 篇。
    这两者的区别就是铁律 8 那句话：**一份本该穷举的清单有个洞，而洞不报错。**

    不用正则：正则里数反斜杠要写四层转义，改起来容易再错一层（已踩过一次）。
    """
    out: dict[str, int] = {}
    i, n = 0, len(body)
    while i < n:
        if body[i] == "\\" and i + 1 < n and body[i + 1].isalpha():
            j = i + 1
            while j < n and body[j].isalpha():
                j += 1
            name = body[i + 1:j]
            out[name] = out.get(name, 0) + 1
            i = j
        else:
            i += 1
    return out


def residue_kind(name: str) -> str:
    """这个宏名归哪一类。**只用于汇总，不参与判定**（见 ``_LAYOUT_MACROS``）。"""
    return "known_layout" if name in _LAYOUT_MACROS else "unexplained"


def extract_one(paper: dict, fetcher: Fetcher) -> dict:
    """一篇 → 一行结果。**不抛异常**：单篇失败要落进 ``failed`` 桶，不是中断整批。"""
    aid = paper.get("arxiv_id") or ""
    row: dict = {"arxiv_id": aid, "title": (paper.get("title") or "")[:90]}

    nf_before = fetcher.stats["not_found"]
    data = fetcher.get_bytes(arxiv_src_url(aid), suffix=".src")
    if not data:
        if fetcher.stats["not_found"] > nf_before:
            row.update(outcome="no_source", note="arXiv 没有源码包（/src/ 终态 4xx）")
        else:
            row.update(outcome="failed", note="源码包下载失败（网络层）")
        return row
    row["src_bytes"] = len(data)

    if data[:5] == b"%PDF-":
        # 周报线**已放弃 PDF 路径**（2026-09-17 用户决策）：PDF 里没有 markdown
        # 标题可切，实测那两篇拿到的就是"0 个标题 / 只剩 `Params`"。
        # 写一套按字号的定位法不划算。**这里只记「全文未取到」，不做回退。**
        #
        # ⚠️ 这是**周报线**的决定。主库 `extract/text.py` 里那个同名回退是给
        # 人啃全文用的，**与这里无关，不许顺势删**。
        row.update(outcome="no_text", note="src 返回 PDF；周报线已放弃 PDF 路径")
        return row

    text = latex_mod.source_to_text(data)
    if not text:
        row.update(outcome="failed", note="源码里没找到可用正文")
        return row

    secs = latex_mod.extract_sections(text, latex_mod.GAP_SECTIONS)
    row["candidates"] = {t: len(b) for t, b in sorted(secs.items(), key=lambda x: -len(x[1]))}
    picked = latex_mod.pick_gap_section(secs)
    if picked is None:
        # 记下它**实际有哪些标题**：这是区分「真没有」与「没读到」的唯一入口。
        heads = latex_mod.list_headings(text)
        row.update(outcome="no_section", headings=heads[:HEADINGS_KEPT],
                   n_headings=len(heads))
        return row

    heading, body = picked
    in_band = latex_mod.BODY_MIN <= len(body) <= latex_mod.BODY_MAX
    # 长度在带外的段落是一种**独立的结果**，不是 ok：它多半是排版残留或
    # 吞掉下级小节的整块，交出去等于骗下游说"这篇有缺口可挖"。
    row.update(
        outcome="ok" if in_band else "degenerate",
        picked=heading,
        picked_len=len(body),
    )
    if not in_band:
        row["note"] = ("过短（疑似排版残留/空节）" if len(body) < latex_mod.BODY_MIN
                       else "过长（疑似吞掉下级小节）")
    # 记下**被让路的**候选：判据里说"优先级本身没错"，这一栏就是那句话的证据面
    # ——人眼要能看见我们跳过了谁、跳过的那个有多长。
    row["passed_over"] = {
        t: len(b) for t, b in secs.items()
        if t != heading and latex_mod.BODY_MIN <= len(b) <= latex_mod.BODY_MAX
    }
    # 残留清点。**如实记录，不作判定、不拦**——残留算"内容"还是"噪声"要靠人眼定。
    row["macro_residue"] = macro_inventory(body)
    row["text"] = body
    return row
