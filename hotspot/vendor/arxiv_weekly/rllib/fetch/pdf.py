"""PDF 下载（阶段 6）。

**只下 RL 子集。** 全量 13119 篇 × 5–8MB ≈ 70–100GB；RL 子集 668 篇约 4–8GB。
全文抽取覆盖全量，但它走 LaTeX 源码，不需要 PDF。

来源按 venue 分两条路（2026-09-13 实测确认库内实际情况）：

* PMLR 论文（ICML 3593 篇，CoRL 263 篇）有 ``pdf_url``，**全部**指向
  ``raw.githubusercontent.com``——PMLR 官方在 GitHub 上的镜像，无任何反爬，
  且拿到的是已出版的 camera-ready 版，比 arXiv 预印本权威。
* NeurIPS / ICLR（9526 篇）在库里**没有任何 pdf_url**，只能走
  ``arxiv.org/pdf/<arxiv_id>``。因此它们的覆盖率完全受 arXiv ID 回填进度限制
  （DataCite 命中率约 70%，见 CLAUDE.md）。

产物落 ``pdfs/``，**不进 cache/**：PDF 本身就是最终产物，先落 cache 再拷一份
会白占一倍磁盘（理由见 ``Fetcher.stream_to``）。

``local_pdf`` 记**相对仓库根目录**的路径，不是绝对路径——绝对路径换台机器就
全失效，而这个库的预期寿命以年计。

**可重复运行**：已存在且非空的 PDF 一律跳过，所以 enrich 还在跑的时候可以先
把 PMLR 那部分下掉，等 arXiv ID 回填完再跑一次补齐剩下的。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from .. import ROOT
from ..http import Fetcher
from ..paths import pdf_path
from ..schema import CanonicalPaper
from ..store import PaperStore

log = logging.getLogger(__name__)

#: 落盘时声明拥有的字段。PDF 只碰 local_pdf，
#: 这样和 enrich / backfill-abstracts 并行跑也不会互相覆盖。
OWNED_FIELDS = ("local_pdf", "provenance", "last_updated")

#: 每多少篇落一次盘。长任务必须分段保存，否则中断即全丢。
SAVE_EVERY = 50

#: PDF 魔数。站点出错时返回的多是 HTTP 200 的 HTML 错误页，不校验就会存出
#: 一堆名叫 ``.pdf`` 的 HTML，而且要到真去读的时候才发现。
PDF_MAGIC = b"%PDF-"

#: 单篇上限。正常论文 1–20MB；超过这个数基本是扫描版或附件包。
MAX_PDF_BYTES = 80 * 1024 * 1024

#: 已确认取不到 PDF 的 host。openreview 是人机验证 403，nips.cc 在本网络 000
#: 不可达（见 CLAUDE.md）。列出来是为了**不发这次注定失败的请求**——
#: 每篇省一次往返，全量就是几小时。
DEAD_HOSTS = frozenset({
    "openreview.net", "www.openreview.net", "papers.nips.cc", "nips.cc",
})

#: 优先级最高的 host：PMLR 的官方 GitHub 镜像，拿到的是已出版版本。
PREFERRED_HOSTS = frozenset({"raw.githubusercontent.com"})


def arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}"


def candidates(paper: CanonicalPaper) -> list[tuple[str, str]]:
    """该论文的候选 ``(url, 来源标签)``，按优先级排序。

    判据是"哪一版更可信"，而不是"哪个站更快"：

    * PMLR 论文优先 GitHub raw —— 覆盖 100% 且是 camera-ready 版。
    * 其余论文优先 arXiv —— 它们本来也没有别的 ``pdf_url`` 可用。

    死 host 直接剔除，不发注定 403 的请求。

    返回列表而非单个 URL，是为了让调用方在首选源失败时**顺次回退**：
    GitHub raw 偶尔会有单篇缺失，此时 arXiv 版本仍能补上。
    """
    pmlr: list[tuple[str, str]] = []
    arxiv: list[tuple[str, str]] = []

    url = paper.pdf_url or ""
    if url and urlparse(url).hostname not in DEAD_HOSTS:
        pmlr.append((url, "pmlr-gh"))
    if paper.arxiv_id:
        arxiv.append((arxiv_pdf_url(paper.arxiv_id), "arxiv"))

    if urlparse(url).hostname in PREFERRED_HOSTS:
        return pmlr + arxiv
    return arxiv + pmlr


def select(store: PaperStore, *, scope: str = "rl", venue: str | None = None) -> list[CanonicalPaper]:
    """挑出待下载的论文，按 paper_id 排序（保证每次运行顺序一致）。"""
    papers = list(store.papers.values())
    if scope == "rl":
        papers = [p for p in papers if p.is_rl_effective]
    if venue:
        papers = [p for p in papers if p.venue.lower() == venue.lower()]
    # 有候选来源的排前面：先把能下的下完，别让无源的论文阻塞进度
    papers = [p for p in papers if candidates(p)]
    return sorted(papers, key=lambda p: p.paper_id)


def _store_pdf(fetcher: Fetcher, paper: CanonicalPaper, pdfs_dir: Path) -> tuple[str, int]:
    """下载单篇。返回 ``(状态, 字节数)``。

    状态取值：``ok`` / ``exists`` / ``failed``。
    """
    dest = pdf_path(pdfs_dir, paper.paper_id)
    if dest.exists() and dest.stat().st_size > 1024:
        return "exists", dest.stat().st_size

    for url, label in candidates(paper):
        size = fetcher.stream_to(
            url, dest, magic=PDF_MAGIC, max_bytes=MAX_PDF_BYTES
        )
        if size:
            _remember_source(paper, label)
            return "ok", size
        # 首选源失败就换下一个候选（GitHub raw 偶有单篇缺失，arXiv 版可补）
    return "failed", 0


def _remember_source(paper: CanonicalPaper, label: str) -> None:
    """把 PDF 的实际来源记进 provenance。

    为什么值得单独记：``local_pdf`` 只说明"本地有文件"，不说明它来自哪一版。
    GitHub raw 是 camera-ready、arXiv 是预印本，两者内容可能有实质差异；
    日后发现某篇正文与正式版对不上时，这是唯一的线索。
    """
    from ..schema import Provenance

    for prov in paper.provenance:
        if prov.source == label:
            return
    paper.provenance.append(Provenance(source=label, source_id=paper.arxiv_id or paper.source_key))


def run(
    store: PaperStore,
    fetcher: Fetcher,
    cfg: dict[str, Any],
    *,
    scope: str = "rl",
    venue: str | None = None,
    limit: int | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """批量下载。**只输出聚合计数**，不回显任何论文内容（CLAUDE.md token 纪律）。"""
    pdfs_dir = ROOT / cfg["paths"]["pdfs"]
    papers = select(store, scope=scope, venue=venue)
    if limit:
        papers = papers[:limit]
    if refresh:
        for paper in papers:
            pdf_path(pdfs_dir, paper.paper_id).unlink(missing_ok=True)

    stats: dict[str, Any] = {
        "scope": scope, "venue": venue or "all",
        "targets": len(papers),
        "ok": 0, "exists": 0, "failed": 0,
        "bytes": 0,
        "no_source": 0,
    }
    # 无候选来源的单独计数：这是缺口清单要显式列出的东西，
    # 不能混进 failed —— failed 意味着"试过但没成"，两者要采取的行动不同。
    pool = list(store.papers.values())
    if scope == "rl":
        pool = [p for p in pool if p.is_rl_effective]
    if venue:
        pool = [p for p in pool if p.venue.lower() == venue.lower()]
    stats["no_source"] = sum(1 for p in pool if not candidates(p))

    changed = 0
    try:
        for i, paper in enumerate(papers, 1):
            status, size = _store_pdf(fetcher, paper, pdfs_dir)
            stats[status] = stats.get(status, 0) + 1
            stats["bytes"] += size
            # ``exists`` 与 ``ok`` 同等认领。文件已经在磁盘上，就是"这篇下好了"
            # 的证据；只认 ``ok`` 的话，中止遗留的孤儿文件永远补不上记录——
            # 磁盘上有、库里没有，而且 ``exists`` 每次都从它旁边走过去。
            if status in ("ok", "exists"):
                rel = pdf_path(pdfs_dir, paper.paper_id).relative_to(ROOT).as_posix()
                if paper.local_pdf != rel:
                    paper.local_pdf = rel
                    changed += 1
            if i % SAVE_EVERY == 0:
                store.save(owned_fields=OWNED_FIELDS)
                log.info("PDF %d/%d  已有 %d 新下 %d 失败 %d", i, len(papers),
                         stats["exists"], stats["ok"], stats["failed"])
    except BaseException:
        # 熔断（HostBlocked）与磁盘红线（DiskLow）都是**有意中止**，不是崩溃
        # （退出码 3 / 4）。它们从 _store_pdf 里穿出来时，下面那次收尾落盘
        # 本来不会执行——上一个保存点之后已下好的 PDF 就只剩磁盘上的文件，
        # 库里一条记录都没有。这里补上。
        # 但落盘再失败也不能盖掉中止原因：那会把"被限流"报成"磁盘炸了"，
        # 排查方向直接指错。
        try:
            store.save(owned_fields=OWNED_FIELDS)
        except Exception:
            log.exception("中止前落盘失败：本批已下好的 PDF 可能未记账")
        raise
    store.save(owned_fields=OWNED_FIELDS)
    stats["recorded"] = changed
    stats["mb"] = round(stats["bytes"] / 1048576, 1)
    del stats["bytes"]
    stats["http"] = dict(fetcher.stats)
    return stats
