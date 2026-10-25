"""这条 feed 到底有没有往前滚？——**纯逻辑，不碰盘、不碰网**。

为什么需要它（2026-09-22 实测事故）
----------------------------------

``cs.AI`` 那条 feed **把周一那 214 条原样又发了一遍**，只把每一条的
``<pubDate>`` 重戳成新日期（214 条里只有 4 条真有版本号变化）。
而当天**每一个现成的判据都是绿的**：

===========================  ==========================================
``feed_build``               当天（``Tue, 22 Sep 2026 04:00:00 +0000``）
``http``                     ``3 fetched / 0 failed``
``raw_unchanged_feeds``      ``[]``——它只盖"与前一天**逐字节相同**"
``upsert``                   ``unchanged: 241``，看着像正常的重公告
退出码                        ``0``
===========================  ==========================================

根因是**结构性**的：这个流水线用的每一个"新鲜度"信号，**都是源自己提供的、
且源可以改写**——``<lastBuildDate>`` 是它写的，每一条 ``<pubDate>`` 也是它写的。
**你没法用一个被检查对象控制的仪器去测这个对象是不是陈的。**

判据：**重发条数**（不是最大 ID，不是字节）
------------------------------------------

"今天标为 ``new``/``cross`` 的条目，有多少**上一份 raw 里已经出现过**"。

``new`` / ``cross`` 是**首次公告**的两种视角，重复出现违背源自身的语义；
``replace`` / ``replace-cross`` 是旧稿重新公告，**重复出现是它的本分**，
所以不参与计数。实测（本库盘上全部可用的日对）：

===================  ==============  ==================
观察                 ``new``+``cross``  其中"昨天已有"
===================  ==============  ==================
9 个正常日（3×3）      903            **0**
**09-22 cs.AI（事故）**  125            **125（100%）**
09-22 cs.LG（正常）     426            0
09-22 stat.ML（正常）    47             0
===================  ==============  ==================

按 Fisher 精确检验的正态上界，正常日的重发率 ≲ 0.3%。**二值可分**。

为什么**不是**"最大 arXiv ID 是否前进"
--------------------------------------

这条判据一度是本模块的初稿，被 Plan 代理在真实数据上实测推翻，三条理由：

1. **对部分停摆是瞎的。** 只要今天有一条 ID 更高的新条目，最大值就前进了，
   而那批里一半是昨天的货它看不出来。重发计数直接数那"一半"。
2. **不依赖单调性假设。** 初稿的等值判据会把"最大值**下降**"（对一个只增不减的
   滚动窗口来说逻辑上不可能）**静默判成健康**。重发计数不需要任何单调性。
3. **最大 ID 那个槽位常常不是新投稿，而是 ``cross``。** 实测 12 份非空 raw 里有
   **6 份**的最大 ID 条目 ``announce_type`` 是 ``cross``——而 cross 条目**按定义**
   就是别处已公告过的。且三个分类的最大值是**相关的**（09-17 cs.AI ``2609.19145``
   对 cs.LG ``2609.19144``，相邻），所以"三条里两条前进了"不是三条独立证据。

⇒ 最大 ID 只作**并列报告的佐证数字**（``latest_id`` / ``latest_id_prev`` / ``delta``），
给人眼一个"``cs.AI`` 卡在 ``2609.22086`` 没动"的直接读数，**不参与判定**。
⇒ ``delta`` **带符号**，让"下降"作为负数**显形**，而不是被等值判据静默吞掉。

⚠️ 基线**只能是前一天的 ``raw``**，不能拿 ``store`` 当基线：``CATEGORIES`` 是
``("cs.LG", "stat.ML", "cs.AI")`` 按此序处理，轮到 ``cs.AI`` 时它当天的 cross 条目
**已经被 ``cs.LG`` 入库了**——每一个当天的交叉列表都会看着像"已知"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Mapping

from .schema import CanonicalPaper

#: 新式 ID 的形状，**必须整串 ``fullmatch``**。
#:
#: ⚠️ **不要用"含不含点"判别旧式 ID。** :data:`rllib.adapters.arxiv_rss._ARXIV_ID_RE`
#: 刻意接受旧式 ID ``[a-z-]+(?:\.[A-Za-z]{2})?/\d{7}``，其中那个可选组**就是一个点**——
#: ``math.GT/0601001`` 含点，而 ``int("math")`` 抛 ``ValueError``。它跑在**每分类循环里**
#: 和**最多 7 层/分类的基线回退里**，一条就能把整趟日更炸掉。
#: 实测 15 份 raw 里旧式 ID 计数**全为 0**，所以这是一颗**潜伏**的雷。
_NEW_ID_RE = re.compile(r"(\d{4})\.(\d{4,5})")

#: 序号位宽（``NNNN`` 或 ``NNNNN`` → 统一乘这个数）。跨月、跨年都不重号：
#: ``2610.00001`` → 261000001 > ``2609.99999`` → 260999999（序号 ≤ 99999 < 倍率）。
_SEQ_SCALE = 100_000

#: 判"重发"的公告类型：**首次公告**的两种视角。
#: ``replace`` / ``replace-cross`` 是旧稿重新公告，**刻意不在内**——重复是它的本分。
_FIRST_ANNOUNCE = frozenset({"new", "cross"})

#: RFC-822 月份名。**硬编码，不用 ``strptime("%b")``**：那个走 locale，
#: 本机 Windows 是中文 locale，`"Tue, 22 Sep 2026 …"` 会解析失败。
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

_RFC822_DAY_RE = re.compile(r"\b(\d{1,2}) ([A-Z][a-z]{2}) (\d{4})\b")


def arxiv_seq(arxiv_id: str | None) -> int | None:
    """新式 arXiv ID → **可比大小**的整数；旧式或取不到 → ``None``。

    ``2609.22086`` → ``260922086``。旧式（``cs/0601001``、``math.GT/0601001``）
    一律 ``None``——它们的序号与 ``YYMM`` 不同源，混进来比大小是无意义的。
    """
    m = _NEW_ID_RE.fullmatch((arxiv_id or "").strip())
    if m is None:
        return None
    return int(m.group(1)) * _SEQ_SCALE + int(m.group(2))


def max_arxiv_id(papers: Iterable[CanonicalPaper]) -> str | None:
    """feed 里**投稿序号最大**的那个 ID 原串；一个都比不了就 ``None``。

    返回**原串**而不是序号，因为它要进汇总行给人看（"``cs.AI`` 卡在
    ``2609.22086`` 没动"）。要比大小请再过一遍 :func:`arxiv_seq`。
    """
    best_id: str | None = None
    best_seq: int | None = None
    for p in papers:
        aid = getattr(p, "arxiv_id", None)
        s = arxiv_seq(aid)
        if s is None:
            continue
        if best_seq is None or s > best_seq:
            best_id, best_seq = aid, s
    return best_id


def build_date_matches(build_date: str | None, day: str) -> bool:
    """feed 自己写的构建时刻，是不是 ``day`` 这一天。

    **为什么需要这一条**：``raw/<日期>-<分类>.xml`` 的文件名记的是 ``--date``，
    而里面装的是**那一趟真抓到的东西**。这两者**在本库已经分家过一次**——
    2026-09-21 的 09:02 那趟把 ``feed_build: Sun, 20 Sep 2026`` 写进了名为
    ``2026-09-21-cs.*.xml`` 的文件里。⇒ **文件名在这个库里撒过谎**，
    所以基线必须自带出处，否则一份回填出来的文件会静默变成一个"最大 ID 更高的前一天"。

    ``build_date`` 缺席时返回 ``False``（当作对不上、要人看一眼）。
    """
    if not build_date:
        return False
    try:
        d = date.fromisoformat(day)
    except ValueError:
        return False
    m = _RFC822_DAY_RE.search(build_date)
    if m is None:
        return False
    return (int(m.group(1)), m.group(2), int(m.group(3))) == (d.day, _MONTHS[d.month - 1], d.year)


@dataclass(frozen=True)
class FeedSample:
    """**某一天、某一个 feed** 的样本。三个字段都从 ``parse_feed`` 的结果直接来。"""

    #: 该 feed 当天全部 arXiv ID（含 ``replace``；用来判"昨天有没有它"）。
    ids: frozenset[str] = field(default_factory=frozenset)
    #: 其中 ``announce_type`` 是 ``new`` / ``cross`` 的那些（**首次公告**）。
    new_cross: frozenset[str] = field(default_factory=frozenset)
    max_id: str | None = None

    @classmethod
    def of(cls, papers: Iterable[CanonicalPaper],
           announce_by_key: Mapping[str, str]) -> FeedSample:
        """从解析结果造一个样本。

        ⚠️ 调用方传进来的 ``papers`` **必须是 ``ParseResult.papers`` 全量**，
        绝不能是 ``--limit`` 截断后的那份：feed 的顺序**不是 ID 顺序**，
        截断后算出的最大值是**一个错的数**，而不是一个"缺的数"——前者更坏。
        """
        papers = list(papers)
        ids: set[str] = set()
        first: set[str] = set()
        for p in papers:
            aid = p.arxiv_id
            if not aid:
                continue
            ids.add(aid)
            if announce_by_key.get(aid or p.title_norm) in _FIRST_ANNOUNCE:
                first.add(aid)
        return cls(frozenset(ids), frozenset(first), max_arxiv_id(papers))


@dataclass(frozen=True)
class Baseline:
    """用作对照的那一天。**自带出处**：文件名 + feed 自己写的构建时刻。"""

    #: 文件名里的日期（``2026-09-21``）——**不是** ``--date``，也不是构建日。
    day: str
    #: 文件名（``2026-09-21-cs.AI.xml``）。进汇总行，回答"跟谁比的"。
    name: str
    sample: FeedSample
    #: feed 自己写的 ``<lastBuildDate>`` 原样字符串。
    build_date: str | None = None
    #: 上面那个是不是真等于 ``day``。**为 ``False`` 不中止判定，只标记**——
    #: 原始 ``build_date`` 无条件进汇总行，人眼可查。
    build_date_matches_name: bool = True


def _entry(reason: str | None = None) -> dict[str, object]:
    """一条读数的**空壳**。``reason`` 非空 = 没比成，值就是"为什么没比成"。

    所有键**一开始就都在**（值为 ``None``），这样"没比成"与"比出来是 0"
    在打印出来的那一行里也分得开——``null`` 与 ``0`` 不是一回事。
    """
    return {
        "checked": False,
        "unchecked_reason": reason,
        "baseline": None, "baseline_day": None, "baseline_build": None,
        "baseline_build_matches_name": None,
        "items": None, "new_cross": None,
        "latest_id": None, "latest_id_prev": None, "delta": None, "reissued": None,
    }


def feed_report(today: Mapping[str, FeedSample | None],
                prev: Mapping[str, Baseline]) -> dict[str, object]:
    """把"今天"与"基线"比成一个可以直接并进汇总行的字典。**纯函数。**

    ``today`` 的值允许是 ``None``，表示**这一趟压根没拿到那个 feed**（抓取失败）。
    那种分类必须照样出现在 ``feed_advance`` 里：如果让它以"键不存在"的形式退场，
    它在汇总行里就**只剩缺席**，而缺席与"没停摆"长得一模一样。

    返回四个键：

    * ``feed_advance`` —— 分类 → 该分类的全部读数（``checked`` / ``unchecked_reason`` /
      基线出处 / 最大 ID 与带符号的 ``delta`` / ``new_cross`` / ``reissued``）。
      **人眼读这个。**
    * ``feed_stalled`` —— ``reissued > 0`` 的分类。**这是判据。**
    * ``feed_advance_unchecked`` —— **没比成**的分类（含抓失败的）。
    * ``feed_baseline_mismatch`` —— 基线自己说它那天不是它那天。

    ⚠️ **``feed_stalled: []`` 必须与"没检查"分得开。** 有三条路径会让某分类根本不进
    比较（抓取失败、基线回退找不到、首跑），那时 ``feed_stalled`` 里的空缺**长得和
    "没停摆"一模一样**——那正是本库反复栽的那个病（「没有」与「没读到」）。
    所以有独立的 ``feed_advance_unchecked`` 和每条上的 ``unchecked_reason``。
    """
    advance: dict[str, dict[str, object]] = {}
    for cat, t in today.items():
        if t is None:
            advance[cat] = _entry("feed 没抓到")
            continue

        entry = _entry()
        entry.update({
            "items": len(t.ids), "new_cross": len(t.new_cross), "latest_id": t.max_id,
        })
        b = prev.get(cat)
        if b is None:
            # 刻意**不写回退深度**：那个常数在调用方（``_BASELINE_LOOKBACK``），
            # 在这里再写一遍就是第二处定义，迟早对不上。
            entry["unchecked_reason"] = "往回找遍了也没有一份有内容的 raw"
        else:
            entry.update({
                "checked": True,
                "baseline": b.name,
                "baseline_day": b.day,
                "baseline_build": b.build_date,
                "baseline_build_matches_name": b.build_date_matches_name,
                "latest_id_prev": b.sample.max_id,
                "reissued": len(t.new_cross & b.sample.ids),
            })
            ts, ps = arxiv_seq(t.max_id), arxiv_seq(b.sample.max_id)
            # 带符号：负数 = 最大值**退回去了**，对只增不减的滚动窗口逻辑上不可能，
            # 必须**显形**而不是被一个等值判据静默判成健康。
            entry["delta"] = (ts - ps) if (ts is not None and ps is not None) else None
        advance[cat] = entry

    return {
        "feed_advance": advance,
        "feed_stalled": [c for c, e in advance.items() if e["reissued"]],
        "feed_advance_unchecked": [c for c, e in advance.items() if not e["checked"]],
        "feed_baseline_mismatch": [
            c for c, e in advance.items()
            if e["checked"] and e["baseline_build_matches_name"] is False
        ],
    }
