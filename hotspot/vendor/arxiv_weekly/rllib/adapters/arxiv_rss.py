"""arXiv 每日公告 RSS 适配器。

与 :mod:`rllib.adapters.pmlr` / :mod:`rllib.adapters.mlanthology` 的**根本区别**：
那两个源给的是一**套会议论文集**（一次性、有界、每篇只出现一次），这个源给的是
一**条每日公告流**（滚动窗口、无界、同一篇会在多个分类里重复出现）。因此本模块
有三条与别的适配器不同的约定：

1. **不去重。** 契约仍是"产出记录、不写盘"，跨分类与跨天的重复一律交给
   :meth:`rllib.store.PaperStore.upsert` 的多键匹配吸收（``identity_keys()`` 里的
   ``arxiv:<id>``）。在适配器里去重等于把"同一篇今天在几个分类里出现过"这个
   事实丢掉，而那是日更脚本唯一能用来判断自己抓全了没有的东西。
2. **保留公告类型。** ``<arxiv:announce_type>`` 回答"今天这几百条里有多少是
   **真新**的"。**这个信息不可事后补**——RSS 是滚动窗口，今天的 ``new`` 列表
   下周就滚没了。
   ⚠️ 实测（2026-09-17）是**四个**取值：``new`` / ``cross`` / ``replace`` /
   ``replace-cross``，**不是** ``cross-list``——本模块初稿按记忆写成三种，
   被当天的数据当场纠正。所以 :class:`ParseResult` 收的是**整张 Counter**、
   汇总行打的也是整张表：arXiv 哪天再加一个取值，它会作为新键自己冒出来，
   而写死成三元组的话新取值只会静默并进"其它"，谁都发现不了。
3. **不碰 ``announce_type`` 之外的扩展字段。** ``CanonicalPaper`` 是冻结契约
   （铁律 5），而往 ``content_hash()`` 里加字段会让**主库 13,119 篇的笔记
   全部重渲染**。所以公告类型只回到调用方，不进 schema。

实测（2026-09-17，详见 ``MEASUREMENTS.md``）：cs.LG / stat.ML / cs.AI 三个分类均
HTTP 200，每个 ``<item>`` 都带 title / description（**摘要全文**）/ guid（**含
arXiv ID**）/ link / dc:creator / category / arxiv:announce_type / pubDate。
"一次给标题+摘要+ID"正是 ``STATUS.md`` §七 预登记的 D0 判据——探针此前只数过
``<item>`` 的条数，摘要与 ID 从未验过，2026-09-17 才第一次实测。
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from ..schema import CanonicalPaper, Provenance

#: 抓取面。**改这里就要同步改 ``MEASUREMENTS.md`` 的实测行**（条数是窗口值）。
CATEGORIES: tuple[str, ...] = ("cs.LG", "stat.ML", "cs.AI")

RSS_URL = "https://arxiv.org/rss/{category}"

#: ``venue`` 的取值。刻意**不是** ``Venue`` 那个 Literal 里的会议名——
#: 预印本流不是会议论文集，混进 ``by_venue`` 里会让人误以为 2026 有会议数据了。
VENUE = "arXiv"

#: 新版 ID（``2609.17560``）与旧版 ID（``cs/0601001``）两种形状都认。
#: 旧版 2007 年就停用了，RSS 里基本见不到——但认不出 ID 的后果是**静默降级成
#: 标题哈希当主键**，所以宁可多写一条正则。
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Za-z]{2})?/\d{7})")

#: ⚠️ **"ID 形状 → 数字"的换算不在这里，在 :mod:`rllib.feedwatch`。**
#: （``arxiv_seq`` 判"投稿序号大小"、``build_date_matches`` 判构建日。）
#: 放在那边是因为它只有"feed 有没有往前滚"一个用途，而**两次实现同一个 ID 语义
#: 是漂移的来源**——注意上面那条正则里的 ``(?:\.[A-Za-z]{2})?`` 那个可选组
#: **就是一个点**，所以 ``math.GT/0601001`` 这种旧式 ID **是含点的**：
#: feedwatch 因此必须 ``fullmatch(r"\d{4}\.\d{4,5}")``，不能靠"含不含点"判别。

#: ``2609.17560v1`` → ``2609.17560``。**必须剥版本号**：``_mint_paper_id()``
#: 的第一条分支就是 ``arxiv:<id>``，带版本号的话同一篇的 v2 会被铸成**第二篇论文**
#: （``arxiv:2609.17560v2`` 不等于 ``arxiv:2609.17560v1``），而 paper_id 一铸永不变。
_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)

#: ``<description>`` 的固定前缀。实测形如::
#:
#:     arXiv:2609.17560v1 Announce Type: new Abstract: We study ...
_DESC_PREFIX_RE = re.compile(
    r"^\s*arXiv:\S+\s+Announce Type:\s*\S+\s*Abstract:\s*", re.IGNORECASE
)

_OAI_PREFIX = "oai:arxiv.org:"


def _localname(tag: str) -> str:
    """剥掉 XML 命名空间：``{http://purl.org/dc/elements/1.1/}creator`` → ``creator``。

    按**局部名**匹配，不按 ``dc:`` / ``arxiv:`` 前缀：``xml.etree`` 展开后前缀就没了，
    而那两个命名空间的 URI 我们并不想写死在代码里——写死了，arXiv 哪天换 URI
    就会**静默解析出空作者、空公告类型**，而不是报错。
    """
    return tag.rsplit("}", 1)[-1]


def strip_version(value: str) -> str:
    """去掉 arXiv 版本号后缀。"""
    return _VERSION_RE.sub("", (value or "").strip())


def extract_arxiv_id(guid: str | None, link: str | None) -> str | None:
    """从 ``<guid>`` 或 ``<link>`` 里抠出裸 arXiv ID。

    两个都试，因为两者都可能缺席，而 ID 是这篇论文**唯一稳定**的身份
    （标题会改、作者顺序会改，ID 不会）。
    """
    for raw in (guid or "", link or ""):
        s = raw.strip()
        if s.lower().startswith(_OAI_PREFIX):
            s = s[len(_OAI_PREFIX):]
        m = _ARXIV_ID_RE.search(s)
        if m:
            return strip_version(m.group(1))
    return None


def clean_abstract(description: str | None) -> str | None:
    """从 ``<description>`` 剥出干净摘要。

    **剥不掉就返回原文，不返回 None。** 宁可多留一小段前缀，也不能把摘要整条丢掉：
    摘要缺席会让 ``filters/rl.py`` 的 haystack 只剩标题（``f"{title} {abstract}"``），
    ``is_rl`` 判假**且不报任何错**——库里 1,585 篇的召回就是这么来的。
    """
    if not description:
        return None
    text = re.sub(r"\s+", " ", description).strip()
    text = _DESC_PREFIX_RE.sub("", text)
    return text or None


def _year_from_arxiv_id(arxiv_id: str) -> int | None:
    """从 ID 推投稿年份：``2609.17560`` → 2026（YYMM 开头）。"""
    m = re.match(r"(\d{2})\d{2}\.", arxiv_id)
    if not m:
        m = re.match(r"[a-z-]+(?:\.[A-Za-z]{2})?/(\d{2})", arxiv_id)
    if not m:
        return None
    yy = int(m.group(1))
    return 2000 + yy if yy < 90 else 1900 + yy


def _year_from_pubdate(pub_date: str) -> int | None:
    m = re.search(r"\b(19|20)\d{2}\b", pub_date or "")
    return int(m.group(0)) if m else None


@dataclass
class ParseResult:
    """一份 RSS 的解析结果。三个计数是**自检**，不是装饰：

    ``no_id`` / ``malformed`` 不为零就说明源或解析器变了，而这两种变化都会
    **静默降级**（降成标题哈希当主键 / 直接少几篇），不在当天当场报出来，
    之后没有任何东西会发现它们。
    """

    papers: list[CanonicalPaper] = field(default_factory=list)
    announce_types: Counter[str] = field(default_factory=Counter)
    #: ``arxiv_id or title_norm`` → 公告类型（**逐条**）。
    #:
    #: 与上面的 ``announce_types`` 是两个用途，别混：那个回答"今天几百条里各占多少"，
    #: 这个回答"**这一篇**是新投的，还是旧稿被重新公告的"。后者必须能落到每一篇上——
    #: 实测 2026-09-17 那 537 条里混着 ``year=2023`` 的老预印本（被 replace/cross 推上来），
    #: 不逐条记就没法在卡片上区分"本周新工作"与"三年前的老论文又被公告了一次"。
    #:
    #: ⚠️ **不进 ``CanonicalPaper``**：那是冻结契约（铁律 5），而这条信息只有周报要。
    #: 键与 ``scripts/daily_arxiv_fetch.py`` 的去重键**同源**（同一个 ``title_norm``
    #: 字段，不是各算各的），两边因此不可能错位。
    announce_by_key: dict[str, str] = field(default_factory=dict)
    #: feed 自己的 ``<lastBuildDate>``，**原样字符串、不解析成 date**。
    #:
    #: 它是"**这份字节是哪一刻构建的**"的唯一载体，而那是本库签名的那个病
    #: （分不清「没有」与「没读到」）在这一环上唯一分得开的东西：2026-09-18 那次
    #: 日更静默事故里，三个 raw 的 `lastBuildDate` 全都**停在前一天**；
    #   2026-09-19（周六、feed 里 0 条）它却是**当天** 04:00——同样是"零篇"，
    #   一个是故障、一个是正常，**只有这一行分得开**。
    #:
    #: ⚠️ **刻意不解析成 ``date``**：那等于替调用方决定时区与取值语义（feed 给的是
    #: UTC，而判"是不是当天"要拿什么当"当天"是调用方的事）。这里只负责如实上报。
    build_date: str | None = None
    no_id: int = 0        # 有标题但取不到 arXiv ID
    malformed: int = 0    # 连标题都没有，被丢弃


def parse_feed(xml_text: str, *, category: str) -> ParseResult:
    """解析一份 arXiv 分类 RSS。**不去重、不写盘。**

    ``category`` 会写进每条记录的 ``topics``——记的是**哪个 feed 看见了它**，
    不是论文自己的分类。两者不同：论文的 ``<category>`` 是它的学科归类（同一篇
    固定不变），而这里要回答的是"同一个分类我抓了两遍会不会重复记账"。
    论文自己的分类留在 ``<周>/raw/*.xml``（周报库内） 里，需要时再解析。
    """
    out = ParseResult()
    root = ET.fromstring(xml_text)

    # 先取 `<lastBuildDate>`（在 `<channel>` 里，与 `<item>` 平级）。理由见
    # ``ParseResult.build_date``：**零条 item 的 feed 也是合法 feed**（周末就是），
    # 而没有这一行的话，"零篇"这个结果在"我们没读到"与"今天真没有"之间没有区别。
    for el in root.iter():
        if _localname(el.tag) == "lastBuildDate":
            out.build_date = (el.text or "").strip() or None
            break

    for item in root.iter():
        if _localname(item.tag) != "item":
            continue

        # 同名元素只取第一个（title/link/guid/description/pubDate 都只有一个），
        # 但 <category> 会重复出现，单独收。
        fields: dict[str, str] = {}
        item_cats: list[str] = []
        for child in item:
            name = _localname(child.tag)
            text = (child.text or "").strip()
            if name == "category":
                if text:
                    item_cats.append(text)
                continue
            fields.setdefault(name, text)

        title = re.sub(r"\s+", " ", fields.get("title", "")).strip()
        if not title:
            out.malformed += 1
            continue

        announce_type = fields.get("announce_type", "").strip()
        if announce_type:
            out.announce_types[announce_type] += 1

        arxiv_id = extract_arxiv_id(fields.get("guid"), fields.get("link"))
        if not arxiv_id:
            out.no_id += 1

        year = None
        if arxiv_id:
            year = _year_from_arxiv_id(arxiv_id)
        if year is None:
            year = _year_from_pubdate(fields.get("pubDate", ""))
        if year is None:
            # 推不出年份也不能丢——`year` 是必填字段，而这条记录的其他部分
            # （标题、摘要、ID）都是好的。取当年的代价是可能偏一年，比丢一篇好。
            year = date.today().year

        # dc:creator 是**一条**字符串装着全部作者（逗号分隔），不是多个元素。
        authors = [a.strip() for a in (fields.get("creator") or "").split(",") if a.strip()]

        paper = CanonicalPaper(
            source_key=arxiv_id or fields.get("guid") or None,
            title=title,
            authors=authors,
            venue=VENUE,
            year=year,
            arxiv_id=arxiv_id,
            url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else fields.get("link") or None,
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None,
            abstract=clean_abstract(fields.get("description")),
            topics=[category],
            # source_id 记 feed 分类；provenance 的合并语义是**按 source 去重**
            # （store._merge），所以多 feed 重复出现时只留第一条——那没关系，
            # "被几个 feed 看见过"由 topics 的并集回答。
            provenance=[Provenance(source="arxiv", source_id=category)],
        )
        out.papers.append(paper)
        # 公告类型逐条记。**取键放在构造之后**，为的是用模型自己算出的 ``title_norm``
        # ——在这里另写一份归一化就会和日更脚本的去重键各算各的，将来某次改动让两份
        # 归一化分叉时，旁路文件会**静默地**对不上号。
        if announce_type:
            out.announce_by_key[paper.arxiv_id or paper.title_norm] = announce_type

    return out
