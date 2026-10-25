"""给周报库里**还没有许可**的预印本逐条补许可。**日更脚本与补库脚本共用这一份。**

为什么要有这一层
----------------

取许可的实际代价全在「要不要发这个请求」上，不在解析——一条 ``GetRecord`` 实测
2,790 字节。所以这一层的全部工作就是把**目标集压到最小**，而目标集的判据只有一条：

    ``arxiv_id`` 非空 **且** ``license`` 为空

这一条判据同时买到三样东西：

1. **幂等。** 已经采到的永远不再请求。这是因为许可**不可撤销**（arXiv 官方规定，
   见 ``MEASUREMENTS.md`` §四c #24）——先到的就是最终值，不需要复查。
2. **断点续跑。** 中途崩了重跑只补剩下的；而 ``GetRecord`` 的 URL 按 id 唯一且永久，
   ``Fetcher`` 的缓存因此让"已经抓过的那些"零网络（「重解析不重抓」在这里的落点）。
3. **它同时就是补库。** 首次运行时"还没有许可的那些"正好是整个存量库，
   所以 ``scripts/backfill_license.py`` **不是另写一套逻辑**，而是调同一个函数。

三个数不是一回事，必须分开报
----------------------------

本库的签名病是分不清「没有」与「没读到」，而这里正好凑齐了三种：

======================================  ===================================
``license_filled``                      取到了
``license_none``                        **源明说**这条记录里没有 ``<license>``
``license_failed``                      **我们没读到**（网络 / 非 XML / 源报错）
======================================  ===================================

后两者在库里都是 ``license = None``，但含义相反：前者是关于**源**的事实，
后者是关于**我们**的缺口。**合并成一个数就等于把「没读到」写成「没有」**——
而这一次的读者会把它读成一篇论文的版权判决。

⚠️ ``license_none`` 的那些会**留在目标集里**，下一轮再问一次。这是有意的：
它靠 ``cache/`` 变成零网络的重问，代价远小于为它编一个哨兵值——哨兵值会把
「源没给」伪装成一个许可名，正是上面那条病。实测（2026-09-18）这种记录是 0 条
（1300 条里有 1300 条带 ``<license>``），所以它出现的量本身就是个信号。
"""

from __future__ import annotations

from collections import Counter

from .adapters import arxiv_oai
from .http import Fetcher
from .schema import CanonicalPaper
from .store import PaperStore

__all__ = ["fill_missing", "targets"]


def targets(store: PaperStore) -> list[CanonicalPaper]:
    """库里还需要问一次的论文，**顺序确定**（按 arXiv ID 排）。

    确定性不是洁癖：顺序不稳会让"跑到一半崩了，下一轮从哪继续"变得不可复现，
    而这一层的全部价值就是可复现地续跑。
    """
    out = [p for p in store.papers.values() if p.arxiv_id and not p.license]
    out.sort(key=lambda p: (p.arxiv_id or "", p.paper_id))
    return out


def fill_missing(
    store: PaperStore,
    fetcher: Fetcher,
    *,
    refresh: bool = False,
    limit: int | None = None,
) -> dict[str, object]:
    """给 ``store`` 里所有「有 arXiv ID 但没有许可」的论文逐条取许可。

    只返回聚合计数，**不回显任何论文内容**（token 纪律）。

    ``limit`` 只影响**这一轮问几篇**（开发期冒烟用），不影响判据——
    没问的那些下一轮还在目标集里，不会静默消失。

    ⚠️ **本函数只改内存，不落盘。** 落盘由调用方按自己的
    ``owned_fields`` 决定（铁律 3）。所以中途被 ``HostBlocked`` / ``DiskLow``
    打断时，已取到的许可**不在盘上**——但它们**在 ``cache/`` 里**，
    重跑是零网络命中（``cache/`` 被搬到 G: 之后这条不再成立，那时就是重抓）。
    """
    todo = targets(store)
    attempted = todo[:limit] if limit else todo

    before = dict(fetcher.stats)
    by_license: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    filled = no_license = failed = mismatched = 0

    for paper in attempted:
        res = arxiv_oai.get_record(fetcher, paper.arxiv_id or "", refresh=refresh)
        for code, _detail in res.errors:
            errors[code] += 1
        if not res.ok:
            failed += 1
            continue

        lic = res.licenses.get(paper.arxiv_id or "")
        if lic:
            paper.license = lic
            by_license[lic] += 1
            filled += 1
        elif res.licenses:
            # 源回了一条记录，但 id 不是我们要的那个。**只可能是我们把键搞错了**
            # （比如 OAI 标识符的大小写，见 arxiv_oai.OAI_ID_PREFIX）。
            # 它不会自己变好，所以单列一栏让它变红。
            mismatched += 1
        elif res.no_license:
            no_license += 1
        else:
            # 记录在、也没报错，但既没有 license 也不属于上面任何一类
            # （正常 GetRecord 不该走到这里）。算没读到，不猜。
            failed += 1

    after = fetcher.stats
    remaining = sum(1 for p in store.papers.values() if p.arxiv_id and not p.license)

    return {
        # —— 目标集 ——
        # `targets` 是本轮开始前"没有许可"的总数（**存量**），
        # `attempted` 是本轮真去问的（受 limit 限制）。
        # 两个数一起看才分得清"库本来就有这么多没补"与"这次只补了一部分"。
        "license_targets": len(todo),
        "license_attempted": len(attempted),
        "license_filled": filled,
        "license_none": no_license,          # 源明说没有 → 关于源的事实
        "license_failed": failed,            # 没读到 → 关于我们的缺口
        "license_id_mismatch": mismatched,   # 必须为 0；非零说明键搞错了
        "license_remaining": remaining,
        # 本轮新增的许可分布，**原样 URL**（不翻译、不归类）。
        # 打整张表而不是固定几个键：arXiv 哪天加一种许可，它会作为新键自己冒出来。
        "license_by_url": dict(by_license),
        "license_errors": dict(errors),
        # OAI 这一段的网络计数（**与 RSS 那段分开**——合起来看的话，
        # "重跑同一天应该是 3 次全 hit" 这条判据就失效了）。
        "http_oai": {k: after.get(k, 0) - before.get(k, 0) for k in after},
    }
