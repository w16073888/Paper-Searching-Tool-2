r"""周报卡片渲染 —— 一篇一片 md，落在 ``<周>/concern/``（周报库内）。

卡片是什么
----------

一篇论文一片，回答一件事：**这篇说了什么、作者自己说还差什么**。
交付契约在 ``rl周报/README.md`` §一：「元数据 + 摘要 + 作者自述的展望/缺口」，
**没有正文、没有 PDF**（全文层已在 2026-09-18 因版权实测关掉，见
``MEASUREMENTS.md`` §四c #24）。

与主库渲染器（``obsidian.py``）的关系
--------------------------------------

**复用三样，其余各自独立**：

- ``upsert_note`` —— 机器区/人工区隔离。**必须复用**，不能抄：它里面的
  ``_managed_end_pos`` 有个「结束标记必须整行匹配」的坑，按子串找会截断文件
  并反复拼接。抄一份就是把那个坑重种一遍。
- ``MANUAL_SCAFFOLD`` —— 人工区骨架。
- ``env()`` —— Jinja 环境（``StrictUndefined``：模板里写错变量名当场炸）。

不共用的是 **frontmatter 的键集**：卡片的字段是周报自己的（周、公告类型、
为什么在范围里），与主库那套（会议 / 年份 / cite_key）不是一回事，所以卡片
自己维护 :data:`CARD_KEYS`。

「没有」vs「没读到」—— 卡片上最要紧的一段话
--------------------------------------------

``concern`` 那一栏必须**如实交代它是怎么来的**。``rllib/extract/gap.py`` 的六个
outcome 里有**两组必须分开说**：

============================  ============================================
``no_section``                **作者真的没写**展望/缺口节 → 关于论文的结论
``no_source`` / ``failed``    **我们没拿到**东西 → 关于我们自己的缺口
``no_text``                   **拿到了但切不出来** → 关于我们自己的缺口
============================  ============================================

名片上这几句话完全不同，**绝不能合并成一句「未提取到」** —— 那正是本库的
签名病（「没有」与「没读到」不分）被写进产品。全库最贵的一次教训就是
这个混淆（probe v2→v3 的根因），交付物里再犯一次，代价是被别人信以为真。

未展开的 LaTeX 宏同理：卡片要**自己说出来**，不能让读者自己发现 ``\Sstate``。

三种笔记，三套受管键
--------------------

本模块产**三类**笔记，frontmatter 的受管键集**各是一套**：

============================  ====================  ==========================
``<周>/concern/<id>.md``      :data:`CARD_KEYS`     一篇一片卡片
``<周>/report.md``            :data:`INDEX_KEYS`    周索引（这一周的**账**）
``<周>/digest.md``            :data:`DIGEST_KEYS`  摘要合订本（**喂模型的原料**）
============================  ====================  ==========================

⚠️ **给 ``_head()`` 传错集合不会报错**，只会让本该由程序生成的那个键被当成读者的键、
**永远停在第一次写的值**——2026-09-18 实测：索引页那两个计数就这么冻了整整一轮，
正文一直是新的、frontmatter 一直是旧的，**退出码 0、没有任何一行变红**。
⇒ **新增一种笔记、或给已有的加一个受管键时，必须同时加进对应的那个 frozenset**，
并且那个笔记的 ``render_*()`` 要带**读回逐键核对**（见 :data:`INDEX_KEYS`）。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

from rllib.atomic import write_atomic
from rllib.render.obsidian import (
    dump_frontmatter,
    env,
    existing_frontmatter,
    upsert_note,
)
from rllib.weekly_scope import BOTH, IS_RL, OUT, SUBFIELD, in_scope, scope_of

__all__ = ["CARD_KEYS", "DIGEST_KEYS", "INDEX_KEYS", "build_digest", "card_digest",
           "load_inputs", "render", "render_digest", "render_index", "week_of"]

TEMPLATE = "concern_card.md.j2"
INDEX_TEMPLATE = "weekly_report.md.j2"
DIGEST_TEMPLATE = "weekly_digest.md.j2"
INDEX_NAME = "report.md"
DIGEST_NAME = "digest.md"
CARD_DIRNAME = "concern"
STATE_NAME = "render_state.json"
FILENAME_SAFE = "-_."                       # 白名单：字母数字 + 这三个

#: 卡片 frontmatter 里**由程序生成**的键。不在此列的一律视为读者自己的，逐字保留。
#: 与主库的 ``MANAGED_KEYS`` 是**两个集合**，因为两类笔记的字段不同——共用一份
#: 会让卡片把 ``rating`` 当成受管键删掉，或者反过来把 ``week`` 当成读者的键留下。
CARD_KEYS = frozenset({
    "title", "authors", "arxiv_id", "week", "announce", "announce_date",
    "in_scope_because", "rl_subfield", "topics", "url", "pdf_url",
    "abstract_available", "concern_outcome", "tags", "first_seen",
})

#: **周索引页** frontmatter 里由程序生成的键。与 :data:`CARD_KEYS` 是**两套**，
#: 理由同 CARD_KEYS 与主库 ``MANAGED_KEYS`` 不共用——而这次是**实测撞出来的**：
#:
#: 2026-09-18：``_head()`` 原先一律按 ``CARD_KEYS`` 判"这是不是读者的键"，而索引页
#: 的 ``papers_in_scope`` / ``concern_pending`` / ``generated`` 一个都不在里面
#: ⇒ 它们被当成读者的键**原样并回来**，于是 frontmatter **从第一次渲染起就冻住**，
#: 之后每次重渲都把第一次的旧值写回去，**不报错、不退出非零**。
#: 现场：同一份 ``report.md`` 里 frontmatter 写着 ``73 / 24``，而正文（**由同一次
#: ``build_index()`` 产生**）写着 ``162 / 0``。
#:
#: ⚠️ 所以**加一个新的索引页 frontmatter 键时，必须同时加进这里**——漏进来不会
#: 报错，只会让那个数永远停在第一次。``render_index()`` 里有一条读回核对
#: （``index_fm_ok``）专门探这件事。
INDEX_KEYS = frozenset({
    "tags", "week", "generated", "papers_in_scope", "concern_pending",
})

#: **摘要合订本** frontmatter 里由程序生成的键。**第三套**，理由与 :data:`INDEX_KEYS`
#: 同源——键集必须**按笔记种类**给。⚠️ 漏一个不会报错，只会让它冻在第一次写的值上。
#:
#: ``library_total`` 是**全库**篇数（不只是范围内的），它和 ``papers_in_digest``
#: 一起构成那一行**覆盖口径**——合订本最要紧的一句话就是"没进来的是什么"，
#: 只报分子不报分母，读它的模型会把「范围内」读成「全部」。
DIGEST_KEYS = frozenset({
    "tags", "week", "generated", "days_covered", "papers_in_digest",
    "library_total",
})

#: ``render_index()`` / ``render_digest()`` 写完后**读回逐键核对**的那几个键。
#: **提到模块级是因为各有两个使用者**：两个 ``render_*()`` 自己，以及
#: ``scripts/check_weekly_render.py`` 的负向验证——它要拿**同一份清单**去证明
#: "传错键集时这套检查真的会红"。⚠️ **同一份判断写两处是本库最贵的一课**
#: （probe v2→v3 的根因）：写在两处的清单，改了其中一处**不会报错**。
INDEX_TRACKED = ("papers_in_scope", "concern_pending", "generated")
DIGEST_TRACKED = ("days_covered", "papers_in_digest", "library_total", "generated")

#: 公告类型 → 人话。契约里的「本周新论文」= ``new`` **或** ``cross``
#: （``cross`` 是跨分类重复公告，**不是旧稿**，见 ``rl周报/README.md`` §四）。
ANNOUNCE_LABEL = {
    "new": "本周新论文",
    "cross": "本周新论文（跨分类）",
    "replace": "修订版",
    "replace-cross": "修订版（跨分类）",
}

#: ``scope_of`` 的返回值 → 卡片上「为什么收它」。空串不是"缺省"，是"不在范围内"。
SCOPE_LABEL = {
    BOTH: "RL 命中 + 子方向专有词",
    IS_RL: "RL 命中",
    SUBFIELD: "子方向专有词命中（RL 词表没看见它）",
    OUT: "",
}

#: 没抽出正文时，卡片上那句**必须分清主体**的话。``None`` = 正文就是答案。
#: ⚠️ 改这里的措辞前先读模块 docstring 的「没有 vs 没读到」。
OUTCOME_SAY = {
    "ok": None,
    "degenerate": "⚠️ **这段长度异常**，可能是排版残留或吞掉了下级小节，读时留意。",
    "no_section": "作者在正文里**没有**写展望或缺口一节。"
                  "（这是**关于论文的结论**——作者确实没交代后续方向。）",
    "no_text": "arXiv 给的是 PDF，我们从里面切不出正文标题。"
               "**这是我们的缺口，不是论文没有。**",
    "no_source": "arXiv 上**没有**这篇的 LaTeX 源码包（``/src/`` 终态 4xx）。",
    "failed": "这次**没抓到**（网络层）。下次增量跑会重试。",
    "": "**还没抽**（这篇刚进范围）。下次跑 ``weekly_sections.py`` 会补上。",
}

#: 索引表里那一列的**短标签**。用词与 :data:`OUTCOME_SAY` 同一套判断，
#: 只是压到一格宽：``no_text`` 必须读得出"是我们的锅"，``no_section`` 必须读得出
#: "是论文的结论"。**别把两者压成同一个词**（见模块 docstring）。
OUTCOME_LABEL = {
    "ok": "抽到",
    "degenerate": "⚠️ 抽到但长度异常",
    "no_section": "作者没写（关于论文）",
    "no_text": "没切出来（关于我们）",
    "no_source": "arXiv 无源码包（关于我们）",
    "failed": "这次没抓到（关于我们）",
    "": "还没抽",
}


def week_of(datestr: str) -> str:
    """``2026-09-17`` → ``2026-W38``。空/坏值给 ``""``，由调用方决定怎么办。

    ⚠️ ``date.fromisoformat`` 对 ``""`` 抛 ValueError，而 ``first_seen`` 理论上
    可能是空的（字段可选）。**宁可返回空串让上游看见，也不要静默归到某一周**——
    归错了会让一篇论文出现在它不属于的那一周的索引里，且没有任何东西会报警。
    """
    if not datestr:
        return ""
    y, m, d = (int(x) for x in datestr.split("-"))
    iso = date(y, m, d).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def safe_name(key: str) -> str:
    """文件名净化。``arxiv_id`` 本就安全，但契约不依赖这一点。"""
    return "".join(c if (c.isalnum() or c in FILENAME_SAFE) else "_" for c in key)


def _rl(paper: dict) -> dict:
    rl = paper.get("rl") or {}
    return rl if isinstance(rl, dict) else {}


# --------------------------------------------------------------------------- #
# 读入
# --------------------------------------------------------------------------- #

def _read_jsonl(path: Path) -> list[dict]:
    """读 jsonl。**损坏要炸**——截断的文件会让下游静默少一半卡片。"""
    if not path.exists():
        return []
    out = []
    for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except ValueError:
            raise SystemExit(f"{path} 第 {i} 行不是合法 JSON —— 拒绝在损坏的输入上渲染") from None
    return out


def load_inputs(weekly_dir: Path) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    """三份输入：``papers.jsonl`` + ``sections.jsonl`` + 各日 ``*.announce.jsonl``。

    公告旁路是**按日分文件**的（``<周>/<日期>.announce.jsonl``），取并集；
    同 key 后出现的日期覆盖先出现的 —— 一份论文可能被公告两次（6 月 new、
    7 月 replace），我们要的是**最近一次**的说法。
    """
    papers = _read_jsonl(weekly_dir / "papers.jsonl")
    sections = {r["arxiv_id"]: r for r in _read_jsonl(weekly_dir / "sections.jsonl")
                if r.get("arxiv_id")}

    announce: dict[str, dict] = {}
    for f in sorted(weekly_dir.glob("*/*.announce.jsonl")):
        day = f.name.split(".")[0]
        for row in _read_jsonl(f):
            key = row.get("key")
            if key:
                announce[key] = {"announce_type": row.get("announce_type", ""), "date": day}
    return papers, sections, announce


# --------------------------------------------------------------------------- #
# 造卡片
# --------------------------------------------------------------------------- #

def _authors_line(authors: list[str], keep: int = 8) -> str:
    if not authors:
        return "*（元数据里没有作者）*"
    if len(authors) <= keep:
        return ", ".join(authors)
    return ", ".join(authors[:keep]) + f" 等 {len(authors)} 人"


def _provenance_line(sec: dict | None) -> str:
    """conern 那一栏的**来源标签** —— 让读者知道这段话是从哪儿、什么时候、怎么来的。"""
    if not sec:
        return "来源：尚无抽取记录。"
    outcome = sec.get("outcome", "")
    parts = [f"``outcome={outcome}``"]
    if sec.get("picked"):
        parts.append(f"取自《{sec['picked']}》")
    if sec.get("picked_len"):
        parts.append(f"{sec['picked_len']} 字")
    if sec.get("extracted"):
        parts.append(f"{sec['extracted']} 抽取")
    if sec.get("passed_over"):
        # 抽取是在**若干候选**里挑了一个（`gap.py` 按标题与长度排序）。把落选的列出来，
        # 是因为「挑得对不对」正是那条机器判据证明不了、只能人眼过的验收项
        # （见 `weekly_sections.py` 模块 docstring）。**读者看不到选择，就无从质疑选择。**
        items = sorted(sec["passed_over"].items(), key=lambda x: -x[1])[:3]
        parts.append("另有候选未选：" + "、".join(f"《{k}》{v} 字" for k, v in items))
    if sec.get("macro_residue"):
        n = sum(sec["macro_residue"].values())
        top = "、".join(f"`{k}`×{v}" for k, v in
                        sorted(sec["macro_residue"].items(), key=lambda x: -x[1])[:3])
        parts.append(f"⚠️ 含 {n} 个未展开的 LaTeX 宏（最多：{top}）——**这是我们的解析缺口**")
    return " · ".join(parts)


def _concern_block(sec: dict | None) -> tuple[str, str]:
    """→ (正文, 来源标签)。**两者必须一起看**，正文说结论、标签说来源。"""
    outcome = (sec or {}).get("outcome", "")
    say = OUTCOME_SAY.get(outcome, f"⚠️ 未登记的 outcome ``{outcome}`` —— 渲染器不认识它。")
    if say is None:                                   # ok / degenerate：正文就是答案
        return (sec.get("text") or "").strip(), _provenance_line(sec)
    if outcome == "degenerate":
        text = (sec.get("text") or "").strip()
        return f"{say}\n\n{text}", _provenance_line(sec)
    return say, _provenance_line(sec)


def build_card(paper: dict, sec: dict | None, ann: dict | None, week: str) -> tuple[dict, str]:
    """→ (受管 frontmatter, 正文)。**返回两者而不是整串**，好让指纹只覆盖这两个
    （读者自己的 frontmatter 键不参与指纹，否则他一改批注就触发重渲染）。"""
    key = paper.get("arxiv_id") or paper.get("paper_id") or ""
    ann = ann or {}
    authors = paper.get("authors") or []

    rows = [
        ("本周", week),
        ("公告", ANNOUNCE_LABEL.get(ann.get("announce_type", ""), ann.get("announce_type", "—"))
         + (f"（{ann['date']}）" if ann.get("date") else "")),
        ("arXiv", key),
        ("为什么收这篇", SCOPE_LABEL.get(scope_of(paper), "")),
        ("子方向", "、".join(_rl(paper).get("subfield") or []) or "—"),
        ("分类", "、".join(paper.get("topics") or []) or "—"),
        ("链接", f"[abs]({paper.get('url') or ''}) · [pdf]({paper.get('pdf_url') or ''})"),
    ]
    concern_text, concern_src = _concern_block(sec)
    abstract = (paper.get("abstract") or "").strip()

    fm = {
        "title": paper.get("title") or "",
        "authors": authors,
        "arxiv_id": key,
        "week": week,
        "announce": ann.get("announce_type", ""),
        "announce_date": ann.get("date", ""),
        "in_scope_because": scope_of(paper),
        "rl_subfield": _rl(paper).get("subfield") or [],
        "topics": paper.get("topics") or [],
        "url": paper.get("url") or "",
        "pdf_url": paper.get("pdf_url") or "",
        "abstract_available": bool(abstract),
        "concern_outcome": (sec or {}).get("outcome", ""),
        "first_seen": paper.get("first_seen") or "",
        "tags": ["周报", "论文"] + (_rl(paper).get("subfield") or []),
    }
    body = env().get_template(TEMPLATE).render(
        title=fm["title"],
        authors_line=_authors_line(authors),
        rows=rows,
        abstract=abstract or "*（元数据里没有摘要）*",
        concern_text=concern_text,
        concern_src=concern_src,
    )
    return fm, body


def card_digest(fm: dict, body: str, begin: str, end: str) -> str:
    """卡片的**受管内容**指纹。

    覆盖 frontmatter 全部受管键 **+ 正文全文**。不覆盖读者自己的键（否则他改一句
    批注就触发全量重渲染）—— 与主库 ``_managed_hash`` 同一条规矩。

    ⚠️ 正文**必须**进指纹。``sections.jsonl`` 是**边跑边长的**：今天 ``outcome=""``
    （还没抽），明天同一篇变成 ``ok`` 加一段正文，而 ``papers.jsonl`` 那一行
    **一个字都没变**。指纹只看元数据的话，这篇的卡片会**永远停在"还没抽"**
    （铁律 6 的形状，这里的具体后果）。
    """
    payload = json.dumps({"fm": fm, "body": body, "begin": begin, "end": end},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _head(path: Path, fm: dict, body: str, begin: str, end: str, *,
          managed: frozenset[str] = CARD_KEYS) -> str:
    """frontmatter + 受管块。**读者自己的键（不在 ``managed`` 里的）原样并回来。**

    ⚠️ ``managed`` 是**按笔记种类**给的，不是全局一个：卡片走 :data:`CARD_KEYS`，
    周索引页走 :data:`INDEX_KEYS`。用错集合不会报错，只会让**本该由程序生成的那个键
    被当成读者的键、永远停在第一次写的值**——索引页那两个计数就这么冻了整整一轮
    （见 :data:`INDEX_KEYS`）。

    ⚠️ ``path`` 是必需的，不是多余参数：读者的键存在**那一片卡片**里，不给路径就
    无从取——早先这里写成 ``existing_frontmatter(Path())`` 是死代码，它会静默返回空
    并把批注全抹掉，**而且不报错**。保留逻辑与主库 ``make_head`` 同一条。
    """
    merged = dict(fm)
    for k, v in existing_frontmatter(path).items():
        if k not in managed:
            merged[k] = v
    return f"---\n{dump_frontmatter(merged)}---\n{begin}\n{body}\n{end}"


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def render(weekly_dir: Path, cfg: dict, *, force: bool = False,
           limit: int | None = None) -> dict:
    """把范围内每一篇渲成一片卡片。只返回聚合计数，**不打印任何论文内容**。"""
    rcfg = cfg["render"]
    begin, end = rcfg["auto_begin"], rcfg["auto_end"]
    template_ok = (Path(__file__).parent / "templates" / TEMPLATE).exists()
    if not template_ok:
        raise SystemExit(f"模板缺失：{TEMPLATE}")

    papers, sections, announce = load_inputs(weekly_dir)
    targets = [p for p in papers if in_scope(p)]
    targets.sort(key=lambda p: p.get("arxiv_id") or "")
    if limit:
        targets = targets[:limit]

    state_path = weekly_dir / STATE_NAME
    state = {} if force else _load_state(state_path)

    counts: Counter[str] = Counter()
    by_week: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    no_week: list[str] = []

    for paper in targets:
        key = paper.get("arxiv_id") or ""
        week = week_of(paper.get("first_seen") or "")
        if not week:
            # ⚠️ **不落盘、不静默**：没有周就不知道往哪个目录写，猜一个等于把论文
            # 放进它不属于的那一周。收集起来在摘要里报出来（`no_week`）。
            no_week.append(key)
            continue
        sec = sections.get(key)
        fm, body = build_card(paper, sec, announce.get(key), week)
        outcomes[(sec or {}).get("outcome", "")] += 1

        digest = card_digest(fm, body, begin, end)
        path = weekly_dir / week / CARD_DIRNAME / f"{safe_name(key)}.md"
        if not force and state.get(key) == digest and path.exists():
            counts["fresh"] += 1
            by_week[week] += 1
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        result = upsert_note(path, _head(path, fm, body, begin, end), begin, end)
        counts[result] += 1
        by_week[week] += 1
        if result != "unmanaged":                  # unmanaged = 读者删了标记，按他的意思来
            state[key] = digest

    live = {p.get("arxiv_id") or "" for p in targets}
    dropped = [k for k in state if k not in live]
    for k in dropped:
        del state[k]

    write_atomic(state_path,
                 json.dumps(state, ensure_ascii=False, sort_keys=True,
                            indent=0).encode("utf-8"))

    return {
        "targets_scope": len(targets),
        "cards": dict(counts),
        "by_week": dict(sorted(by_week.items())),
        "concern_outcomes": dict(sorted(outcomes.items())),
        "state_entries": len(state),
        "state_dropped": len(dropped),
        "no_week": no_week,                        # 必须为空；非空 = first_seen 有洞
        "data_dir": str(weekly_dir),
    }


def _load_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        raise SystemExit(f"{path} 不是合法 JSON —— 拒绝在损坏的渲染状态上续跑") from None


# --------------------------------------------------------------------------- #
# 周索引
# --------------------------------------------------------------------------- #

#: 「本周新论文」= ``new`` **或** ``cross``。⚠️ ``cross`` **不是旧稿**，
#: 它是本周新论文、只是主分类不在我们抓的三个里（``rl周报/README.md`` §四）。
NEW_ANNOUNCE = frozenset({"new", "cross"})


def _short(title: str, n: int = 88) -> str:
    title = (title or "").replace("|", "\\|")
    return title if len(title) <= n else title[: n - 1] + "…"


def _card_link(key: str, title: str) -> str:
    """卡片链接用**相对 markdown**，不用 ``[[wikilink]]``。

    因为 `rl周报/` 有两个读者面：Obsidian（两种都认）和 GitHub 仓库（只认 markdown）。
    wikilink 在 GitHub 上是**一段死文本**——而这一页对外的主要形态正是那个仓库。
    """
    return f"[{_short(title)}]({CARD_DIRNAME}/{safe_name(key)}.md)"


def build_index(paper_rows: list[dict], sections: dict, announce: dict,
                week: str, generated: str, library_total: int = 0) -> tuple[dict, str]:
    """一周的索引。→ (frontmatter, 正文)。"""
    groups = {"new": [], "replace": []}
    outcomes: Counter[str] = Counter()
    subs: Counter[str] = Counter()

    for p in paper_rows:
        key = p.get("arxiv_id") or ""
        sec = sections.get(key)
        outcome = (sec or {}).get("outcome", "")
        outcomes[outcome] += 1
        for s in (_rl(p).get("subfield") or []):
            subs[s] += 1
        atype = (announce.get(key) or {}).get("announce_type", "")
        bucket = "new" if atype in NEW_ANNOUNCE else "replace"
        groups[bucket].append({
            "link": _card_link(key, p.get("title") or key),
            "sub": "、".join(_rl(p).get("subfield") or []) or "—",
            "concern": OUTCOME_LABEL.get(outcome, outcome or "还没抽"),
            "arxiv": key,
        })
    for rows in groups.values():
        rows.sort(key=lambda r: r["arxiv"])

    pending = outcomes.get("", 0) + outcomes.get("failed", 0)
    n_scope = len(paper_rows)
    # 两条恒等式**在这里算、在页面上打印**（铁律 7）。合起来就是：每一篇在范围内的
    # 论文，要么在「新论文」组、要么在「修订版」组，且恰好有一个 outcome。
    # 不成立就说明上面那个分桶把某篇漏了或重了——而漏掉的那篇会**静默地不出现**。
    identity = {
        "total": n_scope,
        "grouped": len(groups["new"]) + len(groups["replace"]),
        "by_outcome": sum(outcomes.values()),
    }
    identity["ok"] = identity["grouped"] == n_scope == identity["by_outcome"]
    fm = {
        "tags": ["周报", "索引"],
        "week": week,
        "generated": generated,
        "papers_in_scope": n_scope,
        "concern_pending": pending,
    }
    body = env().get_template(INDEX_TEMPLATE).render(
        week=week, generated=generated,
        library_total=library_total, n_scope=n_scope,
        n_new=len(groups["new"]), n_rev=len(groups["replace"]),
        rows_new=groups["new"], rows_rev=groups["replace"],
        outcomes=[[OUTCOME_LABEL.get(k, k or "还没抽"), v]
                  for k, v in sorted(outcomes.items(), key=lambda x: -x[1])],
        subs=sorted(subs.items(), key=lambda x: -x[1]),
        pending=pending, identity=identity,
    )
    return fm, body


def render_index(weekly_dir: Path, cfg: dict, *, generated: str | None = None) -> dict:
    """把每周的 ``report.md`` 写出来。**没有指纹**：一页周索引，重渲一遍比维护
    指纹更便宜，而指纹漏字段（铁律 6）在这里的代价是索引停在旧口径上且不报错。

    ⚠️ 前一句里"停在旧口径"**真的发生过**（见 :data:`INDEX_KEYS`）：正文一路是新的，
    frontmatter 那两个计数冻在第一次。所以这里写完**读回来核一遍**——"写出去的值 ==
    我算出来的值"是本页唯一能证明自己没在说旧话的判据。
    """
    rcfg = cfg["render"]
    begin, end = rcfg["auto_begin"], rcfg["auto_end"]
    generated = generated or date.today().isoformat()

    papers, sections, announce = load_inputs(weekly_dir)
    by_week: dict[str, list[dict]] = {}
    for p in papers:
        if not in_scope(p):
            continue
        w = week_of(p.get("first_seen") or "")
        if w:
            by_week.setdefault(w, []).append(p)

    counts: Counter[str] = Counter()
    #: 恒等式（铁律 7）：写回磁盘的 frontmatter 必须等于本次算出来的 fm。
    #: **键要逐个核**，不能只核"文件变没变"——冻住的写回恰好是"文件一个字没变"。
    fm_checked = 0
    fm_mismatch: list[str] = []
    tracked = INDEX_TRACKED
    for week, rows in sorted(by_week.items()):
        fm, body = build_index(rows, sections, announce, week, generated,
                               library_total=len(papers))
        path = weekly_dir / week / INDEX_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        outcome = upsert_note(path, _head(path, fm, body, begin, end,
                                          managed=INDEX_KEYS), begin, end)
        counts[outcome] += 1
        if outcome == "unmanaged":
            # 读者删掉了受管块标记 ⇒ 整页按他的意思来，我们一个字没写。
            # ⚠️ 这一支必须显式跳过，否则"读者手改过的页"会永远报 mismatch ——
            # **一个会误报的检查比没有检查更糟**（CLAUDE.md 已知风险那条）。
            continue
        written = existing_frontmatter(path)
        fm_checked += 1
        for k in tracked:
            if written.get(k) != fm.get(k):
                fm_mismatch.append(f"{week}.{k}: 盘上 {written.get(k)!r} != 算出 {fm.get(k)!r}")
    return {"weeks": sorted(by_week), "index_pages": dict(counts),
            "index_targets": sum(len(v) for v in by_week.values()),
            "index_fm_checked": fm_checked,
            "index_fm_mismatch": fm_mismatch,
            "index_fm_ok": not fm_mismatch}


# --------------------------------------------------------------------------- #
# 摘要合订本
# --------------------------------------------------------------------------- #
#
# 与周索引的分工：**索引页是「账」，合订本是「原料」。** 索引页回答"本周有哪几篇、
# 每篇落在哪个 outcome"（给人和检查用）；合订本把标题 + 摘要攒成**一份可以整份喂给
# 模型**的文本（读者是大模型，不是人）。
#
# ⚠️ 这个用途决定了三处与另两份产物不同的取舍，**都不是随手写的**：
#
# 1. **不做 markdown 锚点**（见 :func:`_digest_entry`）——模型不点链接，而锚点由标题
#    生成，论文标题里全是锚点杀手。目录与正文的对应改用 ``arxiv_id``。
# 2. **口径必须写在正文里**，不能只写 frontmatter——frontmatter 会被下游管道剥掉，
#    而"这份覆盖什么"一旦跟着剥掉，读它的模型会把 162 篇读成"全部"。
# 3. **"不能回答什么"要写在页面上**——摘要只写"我们做了什么"，里面没有"没做"的
#    信息。不写这一句，模型会替你把「哪个方向还没人做」答了，**而那个答案是编的**。

def _digest_entry(p: dict) -> tuple[str, str]:
    """→ (目录行, 正文块)，**两处都以同一个 ``arxiv_id`` 逐字打头**。

    ⚠️ **刻意不做 markdown 锚点。** 锚点由标题生成，而论文标题里的
    ``: ? * [ ] ^ |``、上下标、公式在**不同渲染器里规则不同**（Obsidian 与 GitHub
    就不一样），且标题可能重复。``arxiv_id`` **唯一、稳定、复制到哪儿都还认得出来**
    ——按 id 定位在"可点"和"不可点"（微信里贴文本、整份喂给模型）两种环境里都成立。

    ⚠️ **标题与摘要原样输出，不做 markdown 转义。** 本文件的读者是模型，
    加反斜杠是噪声；而 ``*`` ``_`` 被渲染成斜体只是**观感**问题，不影响取词。
    （与卡片那边不同：卡片给人读，所以它转义 ``|``。两处口径不同是有意的。）
    """
    key = p.get("arxiv_id") or p.get("paper_id") or ""
    title = (p.get("title") or "").strip() or "（元数据里没有标题）"
    head = f"{key} · {title}"
    authors = _authors_line(p.get("authors") or [], keep=3)
    links = " · ".join(x for x in (
        f"[abs]({p['url']})" if p.get("url") else "",
        f"[pdf]({p['pdf_url']})" if p.get("pdf_url") else "",
    ) if x)
    abstract = (p.get("abstract") or "").strip()
    block = "\n".join([
        f"### {head}", "",
        authors + (f" · {links}" if links else ""), "",
        abstract or "*（元数据里没有摘要）*",
    ])
    return f"- {head}", block


def build_digest(papers: list[dict], week: str, generated: str,
                 library_total: int = 0) -> tuple[dict, str, dict]:
    """一周的**摘要合订本**。→ (frontmatter, 正文, 自检信息)。

    ⚠️ **分日按 ``first_seen``（我们把它入库的那天），不按公告日。** 实测这两个口径
    **结果不同**（2026-09-18：16 篇的公告日晚于 ``first_seen``，即被 arXiv 二次公告），
    而**两张表看上去都正常** ⇒ 口径必须印在页面上（见模板）。
    按 ``first_seen`` 分的好处是**每篇恰好落在一天**，各日之和 == 总数这条恒等式
    才**平凡成立**；按公告日分会出现同一篇在两个日文件里，那条恒等式就变成需要解释的东西。
    """
    dated: dict[str, list[dict]] = {}
    undated: list[str] = []
    for p in papers:
        day = p.get("first_seen") or ""
        if day:
            dated.setdefault(day, []).append(p)
        else:
            # ⚠️ **不静默丢**：没有日期就不知道该印在哪一天，猜一个等于把它放在错的日子下。
            # 收起来由调用方报出来（`digest_undated`），它非空就是一个非零退出码。
            undated.append(p.get("arxiv_id") or p.get("paper_id") or "")

    days = sorted(dated)
    toc: list[str] = []
    body: list[str] = []
    printed = 0
    for day in days:
        rows = sorted(dated[day], key=lambda p: p.get("arxiv_id") or "")
        toc.append(f"**{day}（{len(rows)} 篇）**")
        toc.append("")
        body.append(f"## {day}（{len(rows)} 篇）")
        body.append("")
        for p in rows:
            toc_line, block = _digest_entry(p)
            toc.append(toc_line)
            body.append(block)
            body.append("")
            printed += 1
        toc.append("")

    n = len(papers)
    # 恒等式（铁律 7）：**印出来的条目数 == 范围内总篇数**。不写下来的恒等式等于没有检查，
    # 而这里漏一篇的症状是"它就那样不在"，目录和正文都看不见它。
    identity = {"printed": printed, "total": n, "ok": printed == n}
    days_human = (f"覆盖 **{len(days)}** 天（{days[0]} ~ {days[-1]}）"
                  if days else "**一天都没有**")
    fm = {
        "tags": ["周报", "合订本"],
        "week": week,
        "generated": generated,
        "days_covered": days,
        "papers_in_digest": n,
        "library_total": library_total,
    }
    text = env().get_template(DIGEST_TEMPLATE).render(
        week=week, generated=generated, days_human=days_human,
        n=n, library_total=library_total,
        toc="\n".join(toc).strip(),
        body_block="\n".join(body).strip(),
        identity=identity,
    )
    # ⚠️ 返回给调用方的 `info` **必须带上 ok**：调用方拿它决定退出码，
    # 而 `identity` 是给模板用的。两份**不能合成一份**——`identity` 里塞了
    # `total` 这类只给读者看的字段，调用方按它判会判错。
    return fm, text, {"printed": printed, "total": n, "ok": identity["ok"],
                      "undated": undated}


def render_digest(weekly_dir: Path, cfg: dict, *, generated: str | None = None) -> dict:
    """把每周的 ``digest.md`` 写出来。**没有指纹**，理由同 :func:`render_index`——
    重渲一遍比维护指纹便宜，而指纹漏字段（铁律 6）在这里的代价是合订本停在旧口径上
    且不报错，那正是这一页存在的意义所在。

    ⚠️ **仍然走 ``upsert_note``**，虽然合订本整份是一次写的：受管区之外**一个字节都不动**，
    所以将来有人在下面写自己的分析，重渲不会抹掉它。复用还换来结束标记的**整行匹配**
    （抄一份就会把"截断文件再拼接"那个坑重种一遍）。

    ⚠️ 写完**读回来逐键核**（同 :func:`render_index`）：键集给错时的症状是那个数
    **冻结在第一次写的值**，而"文件一个字都没变"恰好是它最正常的表象。
    """
    rcfg = cfg["render"]
    begin, end = rcfg["auto_begin"], rcfg["auto_end"]
    generated = generated or date.today().isoformat()

    papers, _sections, _announce = load_inputs(weekly_dir)
    by_week: dict[str, list[dict]] = {}
    for p in papers:
        if not in_scope(p):
            continue
        w = week_of(p.get("first_seen") or "")
        if w:
            by_week.setdefault(w, []).append(p)

    counts: Counter[str] = Counter()
    fm_checked = 0
    fm_mismatch: list[str] = []
    identity_bad: list[str] = []
    undated_total = 0
    tracked = DIGEST_TRACKED
    for week, rows in sorted(by_week.items()):
        fm, body, info = build_digest(rows, week, generated, library_total=len(papers))
        path = weekly_dir / week / DIGEST_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        outcome = upsert_note(path, _head(path, fm, body, begin, end,
                                          managed=DIGEST_KEYS), begin, end)
        counts[outcome] += 1
        undated_total += len(info["undated"])
        if not info["ok"]:
            identity_bad.append(
                f"{week}: 印出 {info['printed']} != 范围内 {len(rows)} 篇")
        if outcome == "unmanaged":
            # 读者删掉了受管块标记 ⇒ 整页按他的意思来，我们一个字没写。
            # ⚠️ 显式跳过核对，否则"读者手改过的页"会永远报红——
            # **一个会误报的检查比没有检查更糟**（CLAUDE.md 已知风险那条）。
            continue
        written = existing_frontmatter(path)
        fm_checked += 1
        for k in tracked:
            if written.get(k) != fm.get(k):
                fm_mismatch.append(f"{week}.{k}: 盘上 {written.get(k)!r} != 算出 {fm.get(k)!r}")
    return {"digest_weeks": sorted(by_week), "digest_pages": dict(counts),
            "digest_entries": sum(len(v) for v in by_week.values()),
            "digest_undated": undated_total,
            "digest_identity_bad": identity_bad,
            "digest_fm_checked": fm_checked,
            "digest_fm_mismatch": fm_mismatch,
            "digest_fm_ok": not fm_mismatch}
