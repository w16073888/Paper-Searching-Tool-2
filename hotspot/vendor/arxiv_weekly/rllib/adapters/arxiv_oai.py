"""arXiv OAI-PMH 适配器 —— 取**再分发许可**（``<license>``）。

为什么非走这个源
----------------

许可**不在搜索 API 的 schema 里**（arXiv 官方原文，见 ``MEASUREMENTS.md`` §四c #24），
也不在每日 RSS 里。OAI-PMH 是唯一能取到它的地方。这是本模块存在的全部理由。

**只实现 ``GetRecord``，刻意不实现 ``ListRecords`` 窗口采。**
按 datestamp 窗口整日整日地拉确实只要几个请求，但它输在三点，且都被实测过：

1. **带宽。** 实测（2026-09-18）``ListRecords&from=2026-09-17&until=2026-09-17``
   单页 **1300 条 / 3,385,969 字节 / 140.8 秒**（约 0.024 MB/s，正是 ``CLAUDE.md``
   警告过的那种链路）。一天约 2,565 条 ⇒ 两页、约 7 MB、四五分钟。而当天真正要
   许可的只是新入库的百来篇，其余两千多篇是白下的。
2. **缓存是死的。** 那一页的内容只对**那一天**有用，第二天要的是新窗口，旧页永远
   不再需要，却会一直占着 ``cache/``（按 7 MB/天算是 2.5 GB/年）。而 ``GetRecord``
   的 URL 是**按 id 唯一且永久**的——同一篇**一辈子只抓一次**，重跑、重渲染、
   断点续跑全是缓存命中、零网络；每篇响应实测 2,790 字节。
3. **窗口语义会造假。** ``from``/``until`` 按 datestamp 切，而本机日期（UTC+8）与
   arXiv 的 datestamp（UTC）在凌晨差一天，``replace`` 拉上来的旧论文其 datestamp
   也不总等于公告日。**第 3 条的后果是实测过的**：#24 里那 18 篇「未采到」
   （RL 子集 3 篇）就是这么来的——窗口没罩住，却会被读成接近「许可未知」。
   ``GetRecord`` **按 id 问**，没有这个问题：每一篇要么拿到许可，要么源明说这条
   记录里没有 ``<license>`` 元素，要么这次没读到——**三者必须分开报**。

代价是请求数：一篇一请求，``HOST_DELAYS["oaipmh.arxiv.org"]`` 下约 4 秒/篇。
所以调用方必须靠**缓存**与**只补没有许可的那些**两条把总量压住——
见 :func:`rllib.weekly_license.fill_missing`。

实测记录形状（2026-09-18，``GetRecord&metadataPrefix=arXivRaw``，2,790 字节）
---------------------------------------------------------------------------

``<arXivRaw>`` 的**直接子元素**共 10 个，顺序为::

    <id> <submitter> <version> <version> <title> <authors>
    <categories> <comments> <license> <abstract>

三个必须知道的事实：

* ``<id>`` 是**裸 arXiv ID**（``2006.12694``），与库里 ``arxiv_id`` 同格式，
  不带版本号。
* ``<license>`` 的值是**一串 URL**，实测
  ``http://arxiv.org/licenses/nonexclusive-distrib/1.0/``。**不是**
  「arXiv 非独占 1.0」这种可读名——#24 那张分布表里的短名是当时的测量脚本
  翻译出来的。本模块**原样返回**，翻译留给展示层（理由见
  :attr:`rllib.schema.CanonicalPaper.license`）。
* ``<version>`` **会出现多次**（一版一个），所以取字段只能看 ``<arXivRaw>`` 的
  **直接子元素**，不能用 ``.iter()`` 往下递归——递归会把将来可能新增的同名
  子元素一起卷进来，而且不报错。

⚠️ OAI-PMH 的报错是 **HTTP 200 里带 ``<error code="...">``**，不是 4xx/5xx。
``Fetcher`` 那边看着一切正常，所以**必须由本模块把 ``<error>`` 找出来**——
否则一次 ``slowDown`` 会表现成「这一天没有任何许可」，那正是本库反复栽的那个病
（「没有」与「没读到」不分）。:class:`ParseResult` 因此把 ``errors`` 单列一栏，
**不与 ``no_license`` 合并**。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from urllib.parse import quote

from ..http import Fetcher
from .arxiv_rss import _localname

#: OAI-PMH 入口。实测（2026-09-18）**直连可用**（HTTP 200）。
#:
#: 另一个等价入口是 ``https://export.arxiv.org/oai2``：实测也能用，但它 302 到
#: 本行这个地址，白多一个来回。⚠️ ``export.arxiv.org`` 的 ``/api/`` 是出了名的
#: 429（见 ``CLAUDE.md`` 数据源现状），虽然同一个 host 下的 ``/oai2`` 实测正常，
#: 但既然有直连地址就没必要去走它。
OAI_BASE = "https://oaipmh.arxiv.org/oai"

METADATA_PREFIX = "arXivRaw"

#: OAI 标识符的前缀。**大写 A**——源站实测回的是 ``oai:arXiv.org:2006.12694``。
#:
#: ⚠️ ``arxiv_rss._OAI_PREFIX`` 写的是小写 ``oai:arxiv.org:``。那个常量只用于
#: **匹配我们自己接到的 guid**，两边都不出错，所以它没暴露过问题；但照着它做
#: 这里的 ``removeprefix`` 会一条都匹配不上，然后把整串 OAI 标识当成 arXiv ID
#: 存进库——**一个不报错、只是永远匹配不上任何论文的键**。见 :func:`_id_from_header`。
OAI_ID_PREFIX = "oai:arXiv.org:"


@dataclass
class ParseResult:
    """一次 OAI 响应的解析结果。**不回显任何记录内容**（token 纪律）。"""

    #: ``arxiv_id -> license 原样值``。
    licenses: dict[str, str] = field(default_factory=dict)
    #: ``<record>`` 条数（含被删的）。
    records: int = 0
    #: ``<header status="deleted">`` 的条数。它没有 ``<metadata>``，
    #: **不是**「这条没有许可」——所以单独一栏。
    deleted: int = 0
    #: 记录在、但里面**没有 ``<license>`` 元素**的条数。这是**关于源的事实**。
    no_license: int = 0
    #: ``(code, 前 120 字)``。**HTTP 200 里的 ``<error>``**，必须让调用方看见。
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """这次响应能不能用来下结论。有 ``<error>`` 就不能。"""
        return not self.errors


def _id_from_header(header_id: str) -> str:
    """``oai:arXiv.org:2006.12694`` → ``2006.12694``。

    大小写不敏感地剥前缀（理由见 :data:`OAI_ID_PREFIX`）；剥不掉就原样返回，
    让调用方拿到一个明显不对劲的键，而不是一个悄悄错位的键。
    """
    text = (header_id or "").strip()
    if text.lower().startswith(OAI_ID_PREFIX.lower()):
        return text[len(OAI_ID_PREFIX):]
    return text


def _direct_child(el: ET.Element | None, name: str) -> ET.Element | None:
    """**只找直接子元素**，不递归。理由见模块开头「<version> 会出现多次」那条。"""
    if el is None:
        return None
    for child in el:
        if _localname(child.tag) == name:
            return child
    return None


def _direct_child_text(el: ET.Element | None, name: str) -> str:
    child = _direct_child(el, name)
    return (child.text or "").strip() if child is not None else ""


def parse_response(xml_text: str) -> ParseResult:
    """解一页 ``ListRecords`` **或**一条 ``GetRecord`` 的响应。

    两者共用同一套 ``<record>`` 结构，所以一个函数就够——分成两个只会让
    ``<error>`` 的处理写两遍，而那种重复迟早会有一边漏掉。

    ``xml.etree.ElementTree.ParseError`` 由调用方接住（源返回了非 XML，
    比如一个 HTML 错误页）。
    """
    res = ParseResult()
    root = ET.fromstring(xml_text)

    # 先收错误：OAI-PMH 的报错是 200 里的 <error>，不先看它就会把
    # "源拒绝了这次请求"读成"这次一条记录都没有"。
    for el in root.iter():
        if _localname(el.tag) != "error":
            continue
        code = el.attrib.get("code", "?")
        res.errors.append((code, " ".join((el.text or "").split())[:120]))

    for rec in root.iter():
        if _localname(rec.tag) != "record":
            continue
        res.records += 1

        header = metadata = None
        for child in rec:
            name = _localname(child.tag)
            if name == "header":
                header = child
            elif name == "metadata":
                metadata = child

        if header is not None and header.attrib.get("status") == "deleted":
            res.deleted += 1
            continue

        # arXivRaw 是本模块要的格式；arXiv（简版）只是兜底——它同样带 <license>。
        raw = _direct_child(metadata, "arXivRaw") or _direct_child(metadata, "arXiv")
        if raw is None:
            # 结构变了（arXiv 换了 metadataPrefix 的实现）。**绝不能当成
            # "这条没有许可"**——那会把"我们没读懂"伪装成"源没给"，
            # 而且下次仍然读不懂，永远修不好。记成 error 让汇总行变红。
            res.errors.append(("no_metadata", "record 里既没有 <arXivRaw> 也没有 <arXiv>"))
            continue

        aid = _direct_child_text(raw, "id") or _id_from_header(
            _direct_child_text(header, "identifier"))
        if not aid:
            res.errors.append(("no_id", "record 里没有 <id>，header 里也没有 <identifier>"))
            continue

        lic = _direct_child_text(raw, "license")
        if lic:
            res.licenses[aid] = lic
        else:
            res.no_license += 1

    return res


def get_record(fetcher: Fetcher, arxiv_id: str, *, refresh: bool = False) -> ParseResult:
    """按 id 取一条记录。**一篇一请求，但 URL 永久稳定 ⇒ 一篇一辈子只抓一次。**

    ⚠️ 版本号要剥掉再传（库里存的就是剥过的），旧式 id（``cs/0601001``）里的
    斜杠是 **id 的一部分**，所以 ``quote`` 时必须放行 ``/``。
    """
    url = (
        f"{OAI_BASE}?verb=GetRecord&metadataPrefix={METADATA_PREFIX}"
        f"&identifier={OAI_ID_PREFIX}{quote(arxiv_id.strip(), safe='/')}"
    )
    res = ParseResult()
    xml = fetcher.get_text(url, refresh=refresh)
    if xml is None:
        # 网络层失败或终态 4xx。⚠️ 这条**不是**"源说这篇没有许可"。
        res.errors.append(("http", "取不到（网络层失败，或该 URL 返回 4xx）"))
        return res
    try:
        return parse_response(xml)
    except ET.ParseError as exc:
        res.errors.append(("bad_xml", f"响应不是 XML（疑似错误页）：{exc}"[:120]))
        return res
