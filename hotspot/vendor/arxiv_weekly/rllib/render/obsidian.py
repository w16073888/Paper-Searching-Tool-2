"""把 :class:`CanonicalPaper` 渲染成 Obsidian 笔记。

## 受管区规则（这个渲染器唯一需要记住的事）

每篇笔记从**文件开头到 ``<!-- AUTO:END -->`` 那一行**是受管区，程序会整体重写；
**这一行之后的所有内容永远属于你**，程序在任何情况下都不会读、不会改、不会删。

    ---                        ← 受管
    frontmatter（自动生成）
    <!-- AUTO:BEGIN ... -->
    正文表格 / 摘要 / 溯源
    <!-- AUTO:END -->          ← 受管区到此为止
                               ── 下面全是你的 ──
    ## 我的笔记
    随手写的东西、#read 标签、[[]] 双链……

## frontmatter 里"你的字段"

程序生成的字段（见 :data:`MANAGED_KEYS`）每次重渲染都会刷新。
**不在那个名单里的键会被原样保留** —— 你可以在 frontmatter 里加
``rating: 5``、``read: true``、``project: 我的课题``，Dataview 直接可查，
重渲染不会丢。这是"机器区/人工区隔离"在笔记侧的落地：
真相源永远是 JSONL，笔记里的私有字段只活在笔记里。

## 为什么按内容 hash 跳过重写

全量 13000 篇笔记每跑一次都重写，git diff 会直接爆炸、Obsidian 也会重扫。
所以只重写**受管区内容真的变了**的笔记，状态记在 ``data/render_state.json``。

## 为什么不删笔记

论文可能因为词表调整而不再算 RL。此时**不删它的笔记**，只是 frontmatter
里的 ``rl`` 变成 false，于是自动从各索引页里掉出去。理由：删除是不可逆的，
而你可能已经在里面写了批注。让"掉出视图"代替"删文件"。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Iterable, Literal

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .. import ROOT, atomic
from ..paths import safe_name
from ..schema import CanonicalPaper
from ..store import PaperStore
from ..reports import coverage as coverage_mod

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"

#: 程序生成的 frontmatter 键。**不在此列的键一律视为你自己的，逐字保留。**
MANAGED_KEYS = frozenset({
    "title", "authors", "venue", "year", "paper_id", "cite_key", "aliases",
    "doi", "arxiv_id", "openreview_id", "url", "pdf_url", "local_pdf",
    "citations", "citations_updated", "abstract_available",
    "rl", "rl_score", "rl_reasons", "rl_subfield",
    "tags", "first_seen", "last_updated", "provenance",
})

#: Windows 文件名禁用字符远多于 POSIX；``:`` 尤其致命——paper_id 形如 ``orid:c1Ng``。
#: 规则集中在 :mod:`rllib.paths`，因为 PDF 和全文文件也必须用同一套，
#: 否则同一篇论文在磁盘上会有两个对不上的文件名。
#: 受管区之后新建笔记时追加的手工区骨架。
MANUAL_SCAFFOLD = """

## 我的笔记

<!-- 上面到 AUTO:END 为止是程序生成的，重跑会覆盖；这一行以下永远是你的。
     可以随手写批注，也可以用 #read 之类的标签，索引页里的 Dataview 查得到。 -->
"""

#: paper_id → 别名表。渲染前由 :func:`assign_aliases` 一次性填好。
#: cite_key 会撞车（同姓同年后同一个标题词），撞了就加 ``-2``/``-3``。
#: 别名在**全量集合**上按 paper_id 排序分配，所以增量渲染时结果一致。
_ALIASES: dict[str, list[str]] = {}


# --------------------------------------------------------------------------- #
# 路径与状态
# --------------------------------------------------------------------------- #

def _data_dir(cfg: dict) -> Path:
    return ROOT / cfg["paths"]["data"]


def _vault_dir(cfg: dict) -> Path:
    return ROOT / cfg["render"]["vault_dir"]


def note_filename(paper: CanonicalPaper) -> str:
    """笔记文件名 = 净化后的 paper_id。

    用 paper_id 而不是标题或 cite_key，是因为它**永不改变**：改标题、换源、
    重跑都不会让笔记改名，Obsidian 里的 ``[[双链]]`` 也就不会断。
    可读性交给 frontmatter 里的 aliases（含 cite_key），Obsidian 支持按别名链接。
    """
    return safe_name(paper.paper_id, ".md")


def load_render_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_render_state(path: Path, state: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# frontmatter
# --------------------------------------------------------------------------- #

def build_frontmatter(paper: CanonicalPaper) -> dict:
    """生成受管的 frontmatter。值一律取 JSONL，笔记侧不可信。"""
    tags = [f"venue/{paper.venue.lower()}", f"year/{paper.year}"]
    if paper.rl.is_rl:
        tags.insert(0, "rl")
    for sub in paper.rl.subfield:
        tags.append(f"rl/{sub}")

    return {
        "title": paper.title,
        "authors": paper.authors,
        "venue": paper.venue,
        "year": paper.year,
        "paper_id": paper.paper_id,
        "cite_key": paper.cite_key,
        "aliases": _ALIASES.get(paper.paper_id, [paper.cite_key]),
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "openreview_id": paper.openreview_id,
        "url": paper.url,
        "pdf_url": paper.pdf_url,
        "local_pdf": paper.local_pdf,
        "citations": paper.citations,
        "citations_updated": paper.citations_updated,
        "abstract_available": bool(paper.abstract),
        "rl": paper.rl.is_rl,
        "rl_score": paper.rl.score,
        "rl_reasons": paper.rl.reasons,
        "rl_subfield": paper.rl.subfield,
        "tags": tags,
        "first_seen": paper.first_seen,
        "last_updated": paper.last_updated,
        "provenance": [p.source for p in paper.provenance],
    }


def _existing_frontmatter(path: Path) -> dict:
    """读出已存在笔记的 frontmatter。读不动就当空的——宁可少保留，不可崩。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    try:
        data = yaml.safe_load(text[3:end])
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _dump_frontmatter(fm: dict) -> str:
    return yaml.safe_dump(
        fm,
        allow_unicode=True,      # 中文标题/作者原样写，成对引号由 yaml 自己加
        sort_keys=False,         # 保持可读顺序，别按字母乱排
        default_flow_style=False,
        width=10**6,             # 不折行：折了会让长标题在多行间断开
    )


def assign_aliases(papers: Iterable[CanonicalPaper]) -> None:
    global _ALIASES
    used: dict[str, int] = {}
    _ALIASES = {}
    for paper in sorted(papers, key=lambda p: p.paper_id):
        key = paper.cite_key
        n = used.get(key, 0) + 1
        used[key] = n
        _ALIASES[paper.paper_id] = [key] if n == 1 else [key, f"{key}-{n}"]


# --------------------------------------------------------------------------- #
# 受管区拼接
# --------------------------------------------------------------------------- #

def make_head(path: Path, paper: CanonicalPaper, body: str, begin: str, end: str) -> str:
    """拼出「frontmatter + 受管块」。返回值以 ``end`` 标记收尾，不带换行。

    这里会把你写在 frontmatter 里的私有键（不在 :data:`MANAGED_KEYS` 的那些）
    原样并回来——所以 ``rating``、``read`` 这类字段重渲染不会丢。
    """
    fm = build_frontmatter(paper)
    for key, value in _existing_frontmatter(path).items():
        if key not in MANAGED_KEYS:
            fm[key] = value
    # BEGIN 标记**必须**写进去：upsert_note 靠它判断"这是笔记还是你的私人文件"，
    # 少了它所有笔记都会被判成 unmanaged 而永远不再更新。
    return f"---\n{_dump_frontmatter(fm)}---\n{begin}\n{body}\n{end}"


def _managed_end_pos(text: str, begin: str, end: str) -> int:
    """定位受管区的结束位置，返回紧跟在结束标记之后的偏移；找不到返回 -1。

    **必须按整行匹配，不能用 ``text.find(end)``。** 生成的正文里会引用这个
    标记本身——「使用说明」页就在教你这套规则，正文里明晃晃写着
    ``<!-- AUTO:END -->``。按子串找会命中正文里那个引用，于是从那里把文件
    截断、再把 head 拼一遍，表现为：那两个页面**每次重渲染都报 updated，
    文件一次比一次长**。

    注意调用方把整行匹配之外的 ``\\\\s*`` 收窄成了 ``[ \\\\t]*``：只容忍行尾
    空格/制表符，不接受跨行，免得匹配到别的东西。
    """
    if begin not in text:
        return -1
    pattern = re.compile(rf"^{re.escape(end)}[ \t]*$", re.MULTILINE)
    m = pattern.search(text, text.index(begin))
    return m.end() if m else -1


def upsert_note(path: Path, head: str, begin: str, end: str) -> str:
    """把受管区写进笔记，保留其后的用户内容。

    返回 ``created`` / ``updated`` / ``unchanged`` / ``unmanaged``。
    ``unmanaged`` 表示文件存在但找不到受管块标记——**这种情况一律不碰**，
    多半是你手动删掉了标记，那就该按你的意思来。

    **必须原子写。** 这里是整个项目里崩溃代价最高的一处：``new`` 里带着
    文件尾部的人工区，一次写了一半就被断电/强杀，丢掉的不只是受管区，
    还有你自己写的批注——那是唯一没有副本、也重算不出来的东西。
    先写 pid 唯一的临时文件再替换，任何时刻读到的都是完整的某一版。
    （``newline`` 不传，沿用 ``write_text`` 的默认换行转换，避免整库 diff 抖动。）
    """
    if not path.exists():
        _write_note(path, head + MANUAL_SCAFFOLD)
        return "created"

    old = path.read_text(encoding="utf-8")
    cut = _managed_end_pos(old, begin, end)
    if cut == -1:
        return "unmanaged"

    new = head + old[cut:]                 # 只换受管区，尾巴原封不动
    if new == old:
        return "unchanged"
    _write_note(path, new)
    return "updated"


def _write_note(path: Path, text: str) -> None:
    """笔记的原子写。想复用这一步的话从这里走，不要直接 ``write_text``。"""
    tmp = atomic.tmp_for(path)
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(text)
        atomic.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _managed_hash(paper: CanonicalPaper, cfg: dict) -> str:
    """只对**受管内容**取 hash。

    刻意排除 ``last_updated`` / ``citations`` / 用户的额外键：前者每周都在变，
    放进 hash 就等于"每周全量重写一次"；后者是你自己改的，不该触发程序覆盖。
    """
    payload = {
        "title": paper.title,
        "authors": paper.authors,
        "venue": paper.venue,
        "year": paper.year,
        "track": paper.track,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "openreview_id": paper.openreview_id,
        "url": paper.url,
        "pdf_url": paper.pdf_url,
        "abstract": paper.abstract or "",
        "rl": paper.rl.model_dump(),
        "provenance": [p.source for p in paper.provenance],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

Scope = Literal["rl", "all"]

_ENV = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    undefined=StrictUndefined,           # 模板里写错变量名要立刻炸，不要静默留空
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)


def env() -> Environment:
    """给**别的渲染器**用的环境。

    ⚠️ **本库目前没有调用方**：唯一的那个（周报卡片 ``weekly_card.py``）2026-09-20
    随周报线搬到了 ``D:/大学编程学习/arXiv周报/程序/rllib/render/``。函数保留——
    它是**共享**的，别因为"暂时没人用"就删。

    开这个口子而不是让调用方自建一个 ``Environment``：那 4 行配置里
    ``StrictUndefined`` 是判据不是偏好——模板里写错变量名必须当场炸，
    抄漏一行就会变成静默留空，而静默正是本库最防的东西。
    """
    return _ENV


#: ``_dump_frontmatter`` / ``_existing_frontmatter`` 的公开别名，理由同上：
#: 调用方要**同一套** yaml 选项（``allow_unicode`` / ``sort_keys=False`` / 不折行），
#: 自建一份就会漂。周报卡片渲染器拿它做「读者自己的键逐字保留」。
dump_frontmatter = _dump_frontmatter
existing_frontmatter = _existing_frontmatter


def _ctx(paper: CanonicalPaper) -> dict:
    """准备模板上下文。

    **表格行在这里拼好，不留给模板的 if 块。** 原因很实际：Jinja 开了
    ``trim_blocks``，块标签所在行的换行会被吃掉，模板里写 if 块很容易把
    表格前的空行挤掉——markdown 表格缺了前置空行就不渲染成表格，而且
    在源码里看不出来。逻辑放 Python 里，模板只管排版。
    """
    arxiv_abs = f"https://arxiv.org/abs/{paper.arxiv_id}" if paper.arxiv_id else None

    links: list[str] = []
    if paper.url:
        links.append(f"[会议页]({paper.url})")
    if paper.pdf_url:
        links.append(f"[PDF]({paper.pdf_url})")
    if arxiv_abs:
        links.append(f"[arXiv PDF](https://arxiv.org/pdf/{paper.arxiv_id})")
    if paper.openreview_id:
        links.append(f"[OpenReview](https://openreview.net/forum?id={paper.openreview_id})")

    rows: list[tuple[str, str]] = [
        ("会议", f"{paper.venue} {paper.year}" + (f" · {paper.track}" if paper.track else "")),
        ("RL 相关", f"{'是' if paper.rl.is_rl else '否'} · 分数 {paper.rl.score}"),
    ]
    if paper.rl.reasons:
        rows.append(("命中理由", " · ".join(f"`{r}`" for r in paper.rl.reasons)))
    if arxiv_abs:
        rows.append(("arXiv", f"[{paper.arxiv_id}]({arxiv_abs})"))
    if paper.doi:
        rows.append(("DOI", f"`{paper.doi}`"))
    if paper.citations is not None:
        rows.append(("引用", f"{paper.citations}（{paper.citations_updated or '日期未知'}）"))
    rows.append(("入库", paper.first_seen))
    rows.append(("链接", " · ".join(links) or "（无外部链接）"))

    return {
        "paper": paper,
        "authors_line": ", ".join(paper.authors),
        "rows": rows,
        "provenance": paper.provenance,
    }


def render(
    store: PaperStore,
    cfg: dict,
    *,
    scope: Scope = "rl",
    limit: int | None = None,
    force: bool = False,
    render_indices: bool = True,
) -> dict:
    """渲染笔记与索引页。返回聚合计数——**不回显任何论文内容**。"""
    rcfg = cfg["render"]
    papers_dir = _vault_dir(cfg) / rcfg["papers_subdir"]
    papers_dir.mkdir(parents=True, exist_ok=True)

    begin, end = rcfg["auto_begin"], rcfg["auto_end"]

    all_papers = list(store.papers.values())
    assign_aliases(all_papers)               # 别名在全量集合上分配，保证稳定

    targets = all_papers if scope == "all" else [p for p in all_papers if p.rl.is_rl]
    targets.sort(key=lambda p: p.paper_id)
    if limit:
        targets = targets[:limit]

    state_path = _data_dir(cfg) / "render_state.json"
    state = {} if force else load_render_state(state_path)

    counts = {"created": 0, "updated": 0, "unchanged": 0, "unmanaged": 0, "fresh": 0}
    template = _ENV.get_template("paper.md.j2")
    live_ids: set[str] = set()

    for paper in targets:
        path = papers_dir / note_filename(paper)
        live_ids.add(paper.paper_id)

        digest = _managed_hash(paper, cfg)
        if not force and state.get(paper.paper_id) == digest and path.exists():
            counts["fresh"] += 1                  # 内容没变，连读都不读
            continue

        body = template.render(**_ctx(paper))
        result = upsert_note(path, make_head(path, paper, body, begin, end), begin, end)
        counts[result] += 1
        if result != "unmanaged":
            state[paper.paper_id] = digest

    # 掉出渲染集合的笔记：**不删文件**，只清状态，下次再进集合会重新渲染
    dropped = [pid for pid in state if pid not in live_ids]
    for pid in dropped:
        del state[pid]

    save_render_state(state_path, state)

    result = {
        "scope": scope,
        "targets": len(targets),
        "notes": counts,
        "state_entries": len(state),
        "state_dropped": len(dropped),
    }
    if render_indices:
        result["indices"] = render_index_pages(store, cfg, scope=scope)
    return result


# --------------------------------------------------------------------------- #
# 索引页
# --------------------------------------------------------------------------- #

INDEX_PAGES = [
    ("RL 论文总览.md", "index_rl.md.j2"),
    ("按会议.md", "index_venue.md.j2"),
    ("我的评分与已读.md", "index_user.md.j2"),
    ("缺口清单.md", "index_gaps.md.j2"),
    ("覆盖范围声明.md", "index_coverage.md.j2"),
    ("使用说明.md", "index_help.md.j2"),
]


def render_index_pages(store: PaperStore, cfg: dict, *, scope: Scope) -> dict:
    idx_dir = _vault_dir(cfg) / cfg["render"]["index_subdir"]
    idx_dir.mkdir(parents=True, exist_ok=True)
    begin, end = cfg["render"]["auto_begin"], cfg["render"]["auto_end"]

    stats = corpus_stats(list(store.papers.values()))
    # 覆盖范围声明要的字段与 corpus_stats 不同（年份分布、全文覆盖、缺源清单），
    # 单独算一份。它要扫一遍磁盘数全文，所以只在渲染索引页时做一次。
    coverage = coverage_mod.build(store, cfg)
    written: dict[str, str] = {}
    for filename, template_name in INDEX_PAGES:
        body = _ENV.get_template(template_name).render(
            stats=stats, scope=scope, coverage=coverage
        )
        head = (
            f"---\ntitle: {json.dumps(filename[:-3], ensure_ascii=False)}\n"
            f"tags: [rl-index]\n---\n{begin}\n{body}\n{end}"
        )
        written[filename] = upsert_note(idx_dir / filename, head, begin, end)
    return written


def corpus_stats(papers: list[CanonicalPaper]) -> dict:
    """索引页要用的聚合数字。全是计数，不含论文正文。"""
    from collections import Counter

    rl = [p for p in papers if p.rl.is_rl]
    n_rl = len(rl)
    total_by = Counter((p.venue, p.year) for p in papers)
    rl_by = Counter((p.venue, p.year) for p in rl)
    # 结构化列出各会议，模板里直接用 venue/year 两个字段做 Dataview 过滤，
    # 不要拼成 "ICML2025" 再靠隐式类型转换比——那在不同 Dataview 版本上行为不一致。
    venues = [
        {"key": f"{v}{y}", "venue": v, "year": y, "total": n, "rl": rl_by.get((v, y), 0)}
        for (v, y), n in sorted(total_by.items())
    ]
    return {
        "generated": date.today().isoformat(),
        "total": len(papers),
        "rl_total": n_rl,
        "rl_pct": round(n_rl / len(papers) * 100, 1) if papers else 0.0,
        "venues": venues,
        "by_venue": {v["key"]: v["total"] for v in venues},
        "rl_by_venue": {v["key"]: v["rl"] for v in venues},
        "with_abstract": sum(1 for p in papers if p.abstract),
        "with_arxiv": sum(1 for p in papers if p.arxiv_id),
        "with_doi": sum(1 for p in papers if p.doi),
        "rl_no_arxiv": sum(1 for p in rl if not p.arxiv_id),
        "rl_no_arxiv_pct": round(sum(1 for p in rl if not p.arxiv_id) / n_rl * 100, 1) if n_rl else 0.0,
        "rl_no_abstract": sum(1 for p in rl if not p.abstract),
        "rl_no_abstract_pct": round(sum(1 for p in rl if not p.abstract) / n_rl * 100, 1) if n_rl else 0.0,
    }
