"""命令行入口。

用法（在仓库根目录）：``python -m rllib sync --venue icml2025 --limit 5``

**Token 纪律**（见 CLAUDE.md）：所有命令只输出**聚合摘要**，绝不回显论文列表、
HTML 或响应体。需要看具体数据时去读 ``data/papers.jsonl`` 或渲染后的笔记。

开发新适配器时一律先 ``--limit 5``，跑通再去掉。全量跑一次几千篇，
一旦把中间结果打出来到终端，上下文就被污染了，之后每轮对话都在为它付费。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import typer

from . import ROOT
from .store import PaperStore, load_config


from .console import use_utf8_console

use_utf8_console()          # 必须在任何输出之前调用，否则第一条中文就可能崩

app = typer.Typer(
    add_completion=False,
    help="本地 RL 论文库管理工具。所有命令只输出聚合摘要。",
)


# --------------------------------------------------------------------------- #
# 数据落点 profile（`--store`）
# --------------------------------------------------------------------------- #

#: 当前 profile。由 app 级的 ``--store`` 旗标设置，:func:`_setup` 读它。
#:
#: **为什么必须是 CLI 旗标，不能是环境变量**：``scripts/resume.py`` 靠
#: **枚举进程命令行**判断某个长任务是否已经在跑（``Task.marker`` 是命令行子串，
#: 见 `resume.py` 顶部那段理由）。环境变量不进命令行，于是主库的 ``filter``
#: 和周报库的 ``filter --store weekly`` 会被判成**同一个任务**——恢复逻辑要么
#: 漏启动一个、要么重复启动一个，两种都不报错。这类"看起来在跑"的静默失败
#: 正是 resume.py 存在的唯一理由，所以 profile 必须在命令行上可见。
_STORE_PROFILE = "main"

#: profile 的落点改写表：键是 ``config.yaml`` 的 ``paths`` 键，值是**绝对路径**。
#:
#: **为什么是绝对路径**：2026-09-20 这套程序从导师库
#: ``D:\大学编程学习\obsidian\rl_acdemical_paper\`` 搬来本库
#: ``D:\大学编程学习\arXiv周报\程序\``，而**数据刻意留在它的上一级**
#: ``D:\大学编程学习\arXiv周报\数据``（与 ``交付\`` 并列）。下游二十余处一律写
#: ``ROOT / cfg["paths"][...]``，而 ROOT = ``程序\`` ⇒ 写相对路径就得写 ``..\数据``，
#: 不如下面写绝对路径稳。pathlib 的 ``/`` 遇到绝对右操作数会
#: **丢弃左操作数**（本机 Python 3.13.9 实测），所以绝对路径能原样穿过去，
#: 一处签名都不用改。
#: ⚠️ **别改回相对路径**——改了不会报错，只是落点悄悄变成 ``程序\数据``，
#: 而这个目录**不存在**（真数据是中文的 ``数据``）⇒ 表现为"库是空的"，不报错。
#: 正是铁律 8 那个形状。
#:
#: **只改 data / text / pdfs，`cache` 刻意不动**——`Fetcher` 的缓存按 URL 定位，
#: 两个库抓同一个 URL 时共享一份缓存是**对的**（少一次请求），不会串数据。
_WEEKLY_ROOT = Path(
    os.environ.get("ARXIV_WEEKLY_DATA", str(ROOT / "data"))
).resolve()
_WEEKLY_PATHS = {
    "data": str(_WEEKLY_ROOT),
    "text": str(_WEEKLY_ROOT / "text"),
    "pdfs": str(_WEEKLY_ROOT / "pdfs"),
    "cache": str(
        Path(
            os.environ.get("ARXIV_WEEKLY_CACHE", str(_WEEKLY_ROOT / "cache"))
        ).resolve()
    ),
}


@app.callback()
def _global_options(
    store: str = typer.Option(
        "main",
        "--store",
        help="数据落点：main = 2025 会议主库（默认）；weekly = arXiv 周报库，完全隔离（见 ROADMAP 6.4 W）",
    ),
) -> None:
    """全局选项，必须写在子命令**之前**：``python -m rllib --store weekly stats``"""
    global _STORE_PROFILE
    if store not in ("main", "weekly"):
        typer.secho(f"--store 只支持 main / weekly，收到 {store!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    _STORE_PROFILE = store


def _apply_profile(cfg: dict, profile: str | None = None) -> dict:
    """按 profile 重写 ``cfg`` 里的落点。``main`` 原样返回。

    **改 cfg，而不是给每个模块加一个路径参数**：下游二十余处一律写
    ``ROOT / cfg["paths"][...]``，改这一处它们全部自动跟随；加参数则要逐个
    改签名，漏掉的那个会**静默写主库**——本库已经因为这个形状栽过两次
    （铁律 8）。

    ``profile`` 省略时取命令行那个全局开关。显式传参是给**不经 CLI 的入口**用的
    （``scripts/daily_arxiv_fetch.py`` 走 :func:`_setup(False, "weekly")`）。

    ⚠️ **2026-09-20 更正**：这里原本写着「落点只有这一处定义，脚本与 CLI 因此
    不可能各说各话」——**那句话在写下的时候就是假的**。``scripts/export_weekly.py``
    一直绕过本函数，自己拼 ``ROOT / "data" / "weekly"``（该文件 :183），
    而且它从不调 :func:`_setup` 的 ``weekly`` 剖面。**已实测，不是猜测。**
    ⇒ 落点有**两处**定义：本表，和 ``export_weekly.py`` 里那一行。
    待办：把 export 改成跟随本函数。登记在 ``D:\\大学编程学习\\arXiv周报\\README.md`` §2。
    """
    if (profile or _STORE_PROFILE) == "main":
        return cfg

    cfg = dict(cfg)                     # 浅拷贝：不原地改 load_config 的返回值
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"].update(_WEEKLY_PATHS)
    cfg["render"] = dict(cfg["render"])
    cfg["render"]["vault_dir"] = "vault/weekly"
    # `config.yaml` **没有** `search:` 段，而 `search/index.py` 的回落是
    # `data/index.sqlite`——不注入这一项，建周报索引会去覆盖主库那个
    # 250 MB 的全文索引（它派生自 13,119 篇的 jsonl + text/，重建要重灌全文）。
    # ⚠️ 这个值**必须与 `_WEEKLY_PATHS["data"]` 同源**——它也是绝对路径，理由同上。
    # ⚠️ 2026-09-20 搬到 `arXiv周报\程序\` 后**值不变**：数据仍在 `程序\` 的上一级
    # （`arXiv周报\数据\`），这两项本来就是对的，搬家用不着改它们。
    # 实测该文件当前不存在（周报索引从未建过），所以这一行今天不落盘；
    # 但它是"哪天建索引时不覆盖主库那个 250 MB 索引"的唯一保证，不能留成相对路径。
    cfg["search"] = {"index_file": str(_WEEKLY_ROOT / "index.sqlite")}
    return cfg


def _setup(verbose: bool, profile: str | None = None) -> tuple[dict, Fetcher]:
    from .http import DEFAULT_CLIENT_MAX_AGE, Fetcher

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # 上面把根级别压在 WARNING，而**全库仅有的两条进度行都是 log.info**
    # （`extract/text.py` 与 `fetch/pdf.py` 各一条）——于是它们从来没打印过。
    # 长任务因此完全没有进度可见性：2026-09-16 实测 fetch-pdf 跑了 53 分钟，
    # 日志只字未增，只能靠翻 `pdfs/` 的 mtime 猜它是不是卡住了。
    # 把 `rllib` 自己抬到 INFO，**第三方库仍留在 WARNING**（否则 httpx/urllib3
    # 会把终端刷满），这样进度看得见，又不淹掉别的命令的输出。
    logging.getLogger("rllib").setLevel(logging.DEBUG if verbose else logging.INFO)

    cfg = _apply_profile(load_config(ROOT / "config.yaml"), profile)
    cache_dir = ROOT / cfg["paths"]["cache"]
    fetcher = Fetcher(
        cache_dir,
        timeout=cfg["http"].get("timeout", 60),
        max_retries=cfg["http"].get("max_retries", 3),
        min_free_gb=cfg.get("disk", {}).get("min_free_gb", 1.0),
        # 连接池轮换周期（秒）。见 `rllib/http.py` 的 DEFAULT_CLIENT_MAX_AGE。
        # config.yaml 里没有这个键就走默认值——默认就是为实测的劣化点设的。
        client_max_age=cfg["http"].get("client_max_age", DEFAULT_CLIENT_MAX_AGE),
    )
    return cfg, fetcher


def _store(cfg: dict) -> PaperStore:
    return PaperStore(ROOT / cfg["paths"]["data"] / "papers.jsonl").load()


# --------------------------------------------------------------------------- #

@app.command()
def sync(
    venue: str = typer.Option(..., "--venue", "-v", help="config.yaml 里的 venue 键，如 icml2025"),
    limit: int | None = typer.Option(None, "--limit", "-n", help="只处理前 N 篇（开发期必用）"),
    refresh: bool = typer.Option(False, "--refresh", help="忽略缓存重新抓取"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """从某个会议源同步论文元数据到 papers.jsonl。"""
    cfg, fetcher = _setup(verbose)

    spec = cfg["venues"].get(venue)
    if spec is None:
        typer.secho(f"未知 venue: {venue}", fg=typer.colors.RED, err=True)
        typer.echo(f"可选: {', '.join(sorted(cfg['venues']))}")
        raise typer.Exit(1)

    html = fetcher.get_text(spec["url"], refresh=refresh)
    if html is None:
        typer.secho(f"抓取失败: {spec['url']}", fg=typer.colors.RED, err=True)
        raise typer.Exit(3)

    adapter = spec["adapter"]
    if adapter == "pmlr":
        from .adapters import pmlr
        papers = list(
            pmlr.parse_index(
                html, venue=spec["venue"], year=spec["year"], volume=spec["volume"]
            )
        )
    elif adapter == "mlanthology":
        from .adapters import mlanthology
        papers = list(
            mlanthology.parse_index(html, venue=spec["venue"], year=spec["year"])
        )
    else:
        typer.secho(f"adapter '{adapter}' 尚未实现", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(2)

    fetched = len(papers)
    if limit:
        papers = papers[:limit]

    store = _store(cfg)
    counts = {"new": 0, "updated": 0, "unchanged": 0}
    for paper in papers:
        counts[store.upsert(paper)] += 1
    store.save()

    # 只输出聚合摘要
    typer.echo({
        "venue": venue,
        "fetched": fetched,
        "processed": len(papers),
        "new": counts["new"],
        "updated": counts["updated"],
        "unchanged": counts["unchanged"],
        "store_total": len(store),
        "cache": fetcher.stats,
    })


@app.command()
def enrich(
    limit: int | None = typer.Option(
        None, "--limit", "-n", help="只处理前 N 篇（开发期必用；全量约 3600 篇需 8 分钟）"
    ),
    refresh_citations: bool = typer.Option(
        False, "--citations", help="刷新已有记录的引用数（会让全部笔记 frontmatter 变动）"
    ),
    reset_attempts: bool = typer.Option(
        False,
        "--reset-attempts",
        help="先清空 enrich_attempted，让所有缺 arXiv ID 的论文重新进队列（改了匹配逻辑后用）",
    ),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """用 DataCite 解析论文的 arXiv ID。

    这是全库最容易静默污染数据的环节，所以匹配策略刻意保守：
    宁可留空，也不错配。未匹配明细落盘供人工核对——
    **不要靠读终端输出来判断质量**。

    注：引用数不走这里，走 ``rllib citations``——它用 OpenAlex 的**批量**标题
    检索，和本命令的逐篇 DataCite 查询是两套形状（配额只有 100 请求/天）。
    """
    from . import enrich as enrich_mod

    cfg, fetcher = _setup(verbose)
    store = _store(cfg)

    reset = 0
    if reset_attempts:
        reset = enrich_mod.reset_attempts(store)
        store.save(owned_fields=enrich_mod.OWNED_FIELDS)

    report_path = ROOT / cfg["paths"]["data"] / "reports" / "arxiv-unmatched.json"
    report = enrich_mod.enrich(
        store,
        fetcher,
        limit=limit,
        retry_days=cfg["enrich"].get("retry_after_days", 30),
        refresh_citations=refresh_citations,
        report_path=report_path,
    )

    summary = report.as_dict()
    summary["reset_attempts"] = reset
    summary["cache"] = fetcher.stats
    summary["unmatched_report"] = str(report_path)
    typer.echo(summary)


@app.command()
def citations(
    scope: str = typer.Option(
        "rl", "--scope",
        help="rl=只补 RL 子集（32 个请求，一天跑得完）；all=全库（263 个请求，约 2.6 天）",
    ),
    limit: int | None = typer.Option(
        None, "--limit", "-n", help="只处理前 N 篇（开发期必用）"
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="已有引用数的也重查（每周更新用）；默认只补空的"
    ),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """从 OpenAlex 补引用数。

    **配额是硬约束：每天 100 个请求**（每次请求固定 10 credits，日额度 1000）。
    所以必须批量：50 篇一包，全库 263 个请求；逐篇查是 131 天。

    路由：标题 → OpenAlex ``filter=title.search:`` 批量 OR。**不能用 arXiv ID**
    ——OpenAlex 不收录 arXiv DOI（实测 ``filter=doi:10.48550/…`` 返回 count=0），
    而库里 9798 个 DOI 全是 arXiv DOI。

    预期：RL 子集匹配率约 87%，但**其中八成引用数为 0**（库里全是 2025 年会议
    论文，太新）。这一步是把 null 变成真实的 0，不是立刻排出重要性。

    匹配策略与 ``enrich`` 一致：宁可未匹配，也绝不错配。
    """
    if scope not in ("rl", "all"):
        typer.secho(
            f"--scope 只能是 rl 或 all，收到 {scope!r}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(2)

    from . import citations as citations_mod
    from .adapters import openalex

    cfg, fetcher = _setup(verbose)
    store = _store(cfg)

    m = cfg.get("matching", {})
    ccfg = cfg.get("citations", {})
    report_path = ROOT / cfg["paths"]["data"] / "reports" / "citations-unmatched.json"
    report = citations_mod.backfill(
        store,
        fetcher,
        scope=scope,
        limit=limit,
        retry_days=ccfg.get("retry_after_days", 30),
        refresh=refresh,
        base_url=ccfg.get("openalex_base_url", "https://api.openalex.org/works"),
        match_cfg=openalex.MatchConfig(
            title_similarity_threshold=m.get("title_similarity_threshold", 0.92),
            year_tolerance=m.get("year_tolerance", 1),
            require_author_overlap=m.get("require_author_overlap", True),
        ),
        report_path=report_path,
    )

    summary = report.as_dict()
    summary["cache"] = fetcher.stats
    summary["unmatched_report"] = str(report_path)
    typer.echo(summary)


@app.command("backfill-abstracts")
def backfill_abstracts(
    limit: int | None = typer.Option(
        None, "--limit", "-n", help="只处理前 N 篇（开发期必用；全量约 12000 篇，数小时）"
    ),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """从会议源的单篇页回补摘要。

    三个索引页都不含摘要，摘要只在单篇页上，所以这是**逐篇请求**。
    约 12000 次请求、数小时，但免费、零 token，且结果全量落盘缓存。
    建议放后台跑，并可随时中断——每 100 篇落一次盘。
    """
    from . import abstracts as abstracts_mod

    cfg, fetcher = _setup(verbose)
    store = _store(cfg)
    result = abstracts_mod.backfill(
        store, fetcher, limit=limit, save_path=ROOT / cfg["paths"]["data"] / "papers.jsonl"
    )
    typer.echo(result)


@app.command()
def filter(  # noqa: A001  与内置 filter 同名，但作为 CLI 命令名更自然
    rl: bool = typer.Option(False, "--rl", help="重新计算 RL 相关性标注"),
) -> None:
    """对全库重新打标。零网络开销，几秒完成。

    改动 ``词表/rl_keywords.yaml`` 或 ``词表/rl_subfields.yaml`` 后跑这个即可，
    **不需要重新抓取**。

    ⚠️ **周报库要写成 ``python -m rllib --store weekly filter --rl``**——
    ``--store weekly`` **必须在子命令之前**。放到 ``filter`` 后面会被当成
    ``filter`` 的未知参数而报错；而**漏掉它则更糟**：会去给主库打标，
    两边都不报错，只是打错了库。

    两件事在**同一次 save** 里落盘：RL 相关性打分（``rl.is_rl``/``score``/``reasons``）
    与子方向打标（``rl.subfield``）。它们同属 ``rl`` 字段，分两次 save 会互相覆盖。
    """
    if not rl:
        typer.secho("请指定 --rl（目前只有这一种过滤器）", fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    from .filters import rl as rl_filter
    from .filters import subfields as subfield_filter

    cfg, _ = _setup(False)
    kw_path = ROOT / cfg["filter"]["keywords_file"]
    if not kw_path.exists():
        typer.secho(f"词表不存在: {kw_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    sub_path = ROOT / cfg["filter"].get(
        "subfields_file", "data/rl_subfields.yaml"
    )
    if not sub_path.exists():
        typer.secho(f"子方向词表不存在: {sub_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    filt = rl_filter.RLFilter.from_yaml(
        kw_path, mid_threshold=cfg["filter"].get("mid_signal_threshold", 2.0)
    )
    tagger = subfield_filter.SubfieldTagger.from_yaml(sub_path)

    store = _store(cfg)
    result = rl_filter.apply(store, filt)
    # 顺序要紧：先打分（定 is_rl），再打子方向——后者的闸门要用到前者的结果。
    result["subfields"] = subfield_filter.apply(store, tagger)
    store.save(owned_fields=("rl",))
    typer.echo(result)


@app.command()
def render(
    scope: str = typer.Option("rl", "--scope", help="rl=只渲染 RL 论文；all=全库 13000 篇"),
    limit: int | None = typer.Option(None, "--limit", "-n", help="只渲染前 N 篇（开发期用）"),
    force: bool = typer.Option(
        False, "--force", help="忽略内容 hash 强制重写（只在你手动改过受管区时用）"
    ),
    no_index: bool = typer.Option(False, "--no-index", help="不重新生成 _index/ 索引页"),
) -> None:
    """把 papers.jsonl 渲染成 Obsidian 笔记。

    默认只渲染 RL 子集（约 700 篇）——全库 13000 篇笔记会让 Obsidian 变卡，
    而且绝大多数与你的用途无关。要全量加 ``--scope all``。

    **受管区之外的内容永不改动**，详见 rllib/render/obsidian.py 的模块文档。
    """
    from .render import obsidian as render_mod

    if scope not in ("rl", "all"):
        typer.secho(f"--scope 只能是 rl 或 all，收到 {scope!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    cfg, _ = _setup(False)
    store = _store(cfg)
    result = render_mod.render(
        store, cfg, scope=scope, limit=limit, force=force, render_indices=not no_index
    )
    typer.echo(result)


@app.command()
def export(
    parquet: bool = typer.Option(True, "--parquet/--no-parquet", help="导出 data/papers.parquet"),
    bib: bool = typer.Option(True, "--bib/--no-bib", help="导出 data/papers.bib"),
    bib_all: bool = typer.Option(
        False, "--bib-all", help="bib 导出全库（默认只导 RL 子集，约为 13000 条 vs 700 条）"
    ),
) -> None:
    """导出机器可读的派生格式。

    ``parquet`` 是给下游分析用的扁平强类型表——分类、趋势概括直接读它，
    不用再处理 jsonl 里的嵌套结构和 null 类型。schema 是写死的，跨周不漂移。

    ``bib`` 是给 Zotero / Better BibTeX 用的，默认只导 RL 子集。

    两者都是产物，随时可重新生成。
    """
    from . import export as export_mod

    cfg, _ = _setup(False)
    store = _store(cfg)
    data_dir = ROOT / cfg["paths"]["data"]

    result: dict = {}
    if parquet:
        result["parquet"] = export_mod.to_parquet(store, data_dir / "papers.parquet")
    if bib:
        result["bib"] = export_mod.to_bib(
            store, data_dir / "papers.bib", rl_only=not bib_all
        )
    typer.echo(result)


@app.command("fetch-pdf")
def fetch_pdf(
    scope: str = typer.Option("rl", "--scope", help="rl（默认，约 4–8GB）或 all（约 70–100GB）"),
    venue: str | None = typer.Option(None, "--venue", "-v", help="只看某个会议，如 icml"),
    limit: int | None = typer.Option(None, "--limit", "-n", help="只处理前 N 篇（开发期必用）"),
    refresh: bool = typer.Option(False, "--refresh", help="删掉已有 PDF 重新下载"),
    verbose: bool = typer.Option(False, "--verbose", help="打印 DEBUG 日志"),
) -> None:
    """下载 PDF 到 ``pdfs/``。

    默认只下 RL 子集：全量约 70–100GB，而全文抽取走 LaTeX 源码，不需要 PDF。

    **可重复运行**：已有且非空的 PDF 一律跳过（``--refresh`` 除外），所以
    arXiv ID 还在回填时可以先把 PMLR 那部分下掉，之后再跑一次补齐剩下的。
    """
    from .fetch import pdf as pdf_mod

    cfg, fetcher = _setup(verbose)
    store = _store(cfg)
    try:
        typer.echo(pdf_mod.run(
            store, fetcher, cfg, scope=scope, venue=venue, limit=limit, refresh=refresh
        ))
    finally:
        fetcher.close()


@app.command()
def extract(
    scope: str = typer.Option("all", "--scope", help="all（默认）或 rl"),
    venue: str | None = typer.Option(None, "--venue", "-v", help="只看某个会议"),
    limit: int | None = typer.Option(None, "--limit", "-n", help="只处理前 N 篇（开发期必用）"),
    refresh: bool = typer.Option(False, "--refresh", help="重抽已有文本"),
    verbose: bool = typer.Option(False, "--verbose", help="打印 DEBUG 日志"),
) -> None:
    """抽取全文到 ``data/text/<paper_id>.txt.gz``。

    **LaTeX 源码优先，PDF 回退。** 源码从 ``arxiv.org/src/`` 取，公式完整保留；
    PDF 抽取会把贝尔曼方程毁成无法检索的碎片，所以只在拿不到源码时用。

    默认覆盖全语料（约 125MB gz）。源码包不进 ``cache/``——13000 篇会占 35GB，
    而重抽的代价只是重新下载一遍。
    """
    from .extract import text as text_mod

    cfg, fetcher = _setup(verbose)
    store = _store(cfg)
    try:
        typer.echo(text_mod.run(
            store, fetcher, cfg, scope=scope, venue=venue, limit=limit, refresh=refresh
        ))
    finally:
        fetcher.close()


@app.command()
def report(
    what: str = typer.Argument("quality", help="quality | subfields"),
) -> None:
    """生成自动化报告。

    **这是唯一被允许回答"抽取质量怎么样"的东西**——质量校验靠报告，
    不靠读论文样本。产出 ``data/reports/<name>.json``（机器读）与
    ``.md``（人读，Obsidian 里能直接打开）。

    * ``quality``   —— 元数据/全文/PDF 的异常率与缺口清单
    * ``subfields`` —— 子方向分布与**筛选盲区**。「方向的完整性」的常设检查：
      一个漏掉的子方向不会表现为错误记录，而表现为它根本不在这里，
      只有拿外部骨架去套才能把这种缺失变成一个可见数字。
    """
    from .reports import quality, subfields

    runners = {"quality": quality.run, "subfields": subfields.run}
    if what not in runners:
        typer.echo({"error": f"未知报告类型 {what!r}", "available": sorted(runners)},
                   err=True)
        raise typer.Exit(2)

    cfg, _ = _setup(False)
    store = _store(cfg)
    typer.echo(runners[what](store, cfg))


@app.command()
def reindex() -> None:
    """重算全部记录的派生字段（title_norm 等）与跨源索引。

    改动 :func:`rllib.schema.normalize_title` 之后必须跑一次——否则老记录的
    title_norm 会停留在旧算法下，按标题去重/匹配的行为在库内不一致。
    不碰人工区，也不改变 paper_id。
    """
    from .schema import normalize_title

    cfg, _ = _setup(False)
    store = PaperStore(ROOT / cfg["paths"]["data"] / "papers.jsonl").load()
    changed = 0
    for paper in store.papers.values():
        new = normalize_title(paper.title)
        if new != paper.title_norm:
            paper.title_norm = new
            changed += 1

    # 索引可能因为标题变化而需要重建：重新落盘再加载一次最省心
    store.save(owned_fields=("title_norm",))
    store = PaperStore(ROOT / cfg["paths"]["data"] / "papers.jsonl").load()
    typer.echo({
        "total": len(store),
        "title_norm_updated": changed,
        "identity_keys": len(store._index),  # noqa: SLF001
    })


@app.command()
def stats() -> None:
    """输出全库聚合统计。不回显任何论文内容。"""
    cfg, _ = _setup(False)
    store = _store(cfg)
    summary = store.summary()
    typer.echo(summary)


@app.command()
def show(
    limit: int = typer.Option(5, "--limit", "-n", help="最多显示 N 条（务必保持小）"),
    venue: str | None = typer.Option(None, "--venue", "-v"),
) -> None:
    """抽样查看记录。每条压成一行，用于确认字段是否合理。"""
    cfg, _ = _setup(False)
    store = _store(cfg)
    shown = 0
    for paper in sorted(store.papers.values(), key=lambda p: p.paper_id):
        if venue and f"{paper.venue}{paper.year}".lower() != venue.lower():
            continue
        typer.echo(
            f"{paper.paper_id:<24} {paper.venue}{paper.year} "
            f"arxiv={paper.arxiv_id or '-':<12} "
            f"orid={(paper.openreview_id or '-')[:12]:<12} "
            f"pdf={'Y' if paper.pdf_url else 'N'} "
            f"{paper.title[:60]}"
        )
        shown += 1
        if shown >= limit:
            break
    if not shown:
        typer.echo("(无匹配记录)")


@app.command()
def index(
    rebuild: bool = typer.Option(False, "--rebuild", help="整库重建（改了分词器或 DDL 后需要）"),
) -> None:
    """构建 / 增量更新全文检索索引（SQLite FTS5）。

    正文抽取是分批完成的，所以这个命令会被反复跑；增量判据是内容 hash，
    重复跑只处理变化的那几篇。索引本身是产物，随时可删可重建。
    """
    from .search import index as search_index

    cfg, _ = _setup(False)
    store = _store(cfg)
    typer.echo(search_index.build(store, cfg, rebuild=rebuild))
    typer.echo(search_index.stats(cfg))


@app.command()
def search(
    query: str = typer.Argument(..., help='检索式。短语加引号，前缀加 *，如 \'"direct preference" optim*\'"'),
    venue: str = typer.Option(None, "--venue", help="只搜某个会议（ICML/NeurIPS/ICLR/CoRL）"),
    year: int = typer.Option(None, "--year", help="只搜某一年"),
    subfield: str = typer.Option(None, "--subfield", help="按子方向收窄，如 offline / rlhf"),
    rl_only: bool = typer.Option(False, "--rl-only", help="只用 RL 筛选命中集（默认搜全量）"),
    any_term: bool = typer.Option(False, "--any", help="任一词命中即可（默认要求全部命中）"),
    facts: bool = typer.Option(False, "--facets", help="只看分布统计，不列论文"),
    n: int = typer.Option(15, "-n", "--limit", help="返回条数"),
    offset: int = typer.Option(0, "--offset", help="跳过前 N 条"),
) -> None:
    """全文检索。**默认搜全量，不以 RL 筛选为闸门**——见 search/query.py 顶部说明。"""
    from .search import index as search_index
    from .search import query as search_query

    cfg, _ = _setup(False)
    path = search_index.index_path(cfg)
    if not path.exists():
        typer.secho("索引不存在，先跑 `python -m rllib index`", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    conn = search_index.connect(path, create=False)
    try:
        filters = dict(venue=venue, year=year, subfield=subfield, rl_only=rl_only,
                       any_term=any_term)
        if facts:
            typer.echo(search_query.facets(conn, query, **filters))
            return
        hits = search_query.search(
            conn, query, text_dir=ROOT / cfg["paths"]["text"],
            limit=n, offset=offset, **filters,
        )
        typer.echo({"query": query, "shown": len(hits), "offset": offset, **{
            k: v for k, v in filters.items() if v not in (None, False)}})
        for i, h in enumerate(hits, 1 + offset):
            tags = "/".join(h.subfields) or "-"
            rl = "RL" if h.is_rl else "  "
            body = f"{h.chars // 1000}k字" if h.chars else "无全文"
            typer.echo(f"\n[{i}] {rl} {(h.venue or '?')} {h.year} [{tags}] {body}")
            typer.echo(f"    {h.title}")
            if h.snippet:
                typer.echo(f"    … {h.snippet}")
            if h.arxiv_id:
                typer.echo(f"    arXiv:{h.arxiv_id}")
    finally:
        conn.close()


@app.command()
def check() -> None:
    """自检：schema 能否实例化、配置能否加载、路径是否存在。"""
    from .schema import CanonicalPaper, normalize_title

    p = CanonicalPaper(title="A Test Paper", venue="ICML", year=2025, source_key="test25a")
    assert p.title_norm == "a test paper", p.title_norm
    assert p.paper_id == "", "paper_id 应由 store 铸造，schema 不自行赋值"
    assert normalize_title("(How) Do LLMs Work?") == "how do llms work"

    cfg = load_config(ROOT / "config.yaml")
    typer.echo({
        "schema_ok": True,
        "cite_key": p.cite_key,
        "identity_keys": p.identity_keys(),
        "venues": sorted(cfg["venues"]),
        "root": str(ROOT),
    })


if __name__ == "__main__":
    app()
