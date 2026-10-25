"""全语料全文抽取（阶段 6）。

**LaTeX 源码优先，PDF 回退。** 这不是洁癖：RL 论文公式密度极高，PDF 抽取会把
贝尔曼方程毁成 ``r(s,a) + γ max Q(s ,a )``——上下标变独立 token、分式彻底散架，
做语义检索时等于废掉一半信息。源码走 :mod:`rllib.extract.latex`，公式完整保留。

覆盖范围是**全语料**，但成本比想象的低：源码包平均约 2.7MB，解包取完文本就丢掉，
不进 ``cache/``（13000 篇会占 35GB，而重抽的代价只是重新下载一遍）。
落盘的只有 ``data/text/<id>.txt.gz``，全量约 125MB。

哪些论文拿得到全文，取决于有没有 arXiv ID（DataCite 命中率约 70%，见 CLAUDE.md）：

* 有 arXiv ID → 源码；源码其实是 PDF（早期投稿常见）→ 同一份字节给 PyMuPDF
* 无 arXiv ID 但有本地 PDF → PyMuPDF（PDF 只下了 RL 子集，见 :mod:`rllib.fetch.pdf`）
* 都没有 → 记为 ``no_source``，进缺口清单

**不写 schema 字段。** 文本是派生产物，覆盖率由 :mod:`rllib.reports.quality` 扫
目录得出。少一个字段就少一处 jsonl 变更，也少一次与 enrich 的字段所有权协商。
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .. import ROOT, atomic
from ..http import Fetcher
from ..paths import text_path
from ..schema import CanonicalPaper
from ..store import PaperStore
from . import latex as latex_mod

log = logging.getLogger(__name__)

#: 抽取器版本，和 :data:`rllib.extract.latex.VERSION` 一起构成"这份文本是哪套
#: 规则产出的"标识。改了解析逻辑就该 +1，这样质量报告能指出哪些是旧规则的结果。
VERSION = "1"

#: 短于这个字符数视为抽取失败。正常论文正文 2 万字符以上；
#: 只有几百字符基本是"源码包里的主文件选错了"或"PDF 只有图没有文字层"。
MIN_CHARS = 2000

#: arXiv 源码包的体积上限。正常 1–10MB，个别带大量图片的能到 40MB；
#: 超过这个数基本是附了数据集，下完也没用。
MAX_SOURCE_BYTES = 60 * 1024 * 1024

#: 正文里公式/符号占比高到一定程度就说明抽取毁掉了结构。
#: 只用于质量报告标记，不做自动删除——宁可多留可疑样本，也别悄悄丢掉好数据。
SUSPECT_CHARS_PER_PAGE = 400


@dataclass
class Outcome:
    """单篇抽取结果。

    ``text`` 直接挂在对象上而不是让调用方重抽一遍。它可能几十万字符，所以
    **任何日志和统计都不得触碰它**——只读 ``chars``。多一个字段换掉一次
    完整的重复解析（含 tarball 解包），值得。
    """

    status: str                      # ok | too_short | no_source | failed
    source: str = ""                 # latex | pdf-arxiv | pdf-local
    chars: int = 0
    pages: int = 0
    note: str = ""
    text: str = ""


def arxiv_src_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/src/{arxiv_id}"


# --------------------------------------------------------------------------- #
# PDF 抽取（回退路径）
# --------------------------------------------------------------------------- #

def pdf_to_text(data: bytes) -> tuple[str, int]:
    """PyMuPDF 抽正文。返回 ``(文本, 页数)``；失败返回 ``("", 0)``。

    用默认的 ``"text"`` 模式而不是 ``sort=True``：PyMuPDF 的块顺序对双栏排版
    通常已经正确，而 ``sort=True`` 按上下左右强制重排，反而会把双栏的左栏尾部
    和右栏开头交错在一起。这是回退路径，够用就行。
    """
    import pymupdf

    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            pages = doc.page_count
            parts = [doc.load_page(i).get_text("text") for i in range(pages)]
    except Exception as exc:                      # noqa: BLE001 —— 坏 PDF 形态太多
        log.debug("PyMuPDF 抽取失败: %s", exc)
        return "", 0
    return "\n".join(parts).strip(), pages


# --------------------------------------------------------------------------- #
# 单篇
# --------------------------------------------------------------------------- #

def _from_latex(paper: CanonicalPaper, fetcher: Fetcher) -> Outcome:
    data = fetcher.get_bytes(arxiv_src_url(paper.arxiv_id or ""), suffix=".src")
    if not data:
        return Outcome("failed", note="源包下载失败")
    if data[:5] == b"%PDF-":
        # arXiv 对早期投稿的 /src/ 直接返回 PDF。同一份字节喂给 PyMuPDF，
        # 不额外再下一次 /pdf/。
        text, pages = pdf_to_text(data)
        if not text:
            return Outcome("failed", note="src 返回 PDF 且抽取失败")
        return Outcome(_ok_or_short(text), "pdf-arxiv", len(text), pages, text=text)
    text = latex_mod.source_to_text(data)
    if not text:
        return Outcome("failed", note="源码里没找到可用正文")
    return Outcome(_ok_or_short(text), "latex", len(text), text=text)


def _from_local_pdf(paper: CanonicalPaper, cfg: dict[str, Any]) -> Outcome:
    path = ROOT / (paper.local_pdf or "")
    if not paper.local_pdf or not path.exists():
        return Outcome("no_source")
    try:
        data = path.read_bytes()
    except OSError as exc:
        return Outcome("failed", note=f"读本地 PDF 失败: {exc}")
    text, pages = pdf_to_text(data)
    if not text:
        return Outcome("failed", note="本地 PDF 抽取失败")
    return Outcome(_ok_or_short(text), "pdf-local", len(text), pages, text=text)


def _ok_or_short(text: str) -> str:
    return "ok" if len(text) >= MIN_CHARS else "too_short"


def extract_one(paper: CanonicalPaper, fetcher: Fetcher, cfg: dict[str, Any]) -> Outcome:
    if paper.arxiv_id:
        return _from_latex(paper, fetcher)
    return _from_local_pdf(paper, cfg)


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #

def write_text(text_dir: Path, paper_id: str, text: str) -> int:
    """gzip 落盘，返回压缩后字节数。

    ``mtime=0`` 让同样的输入产出**字节完全相同**的文件。否则每次重抽 gzip 头里
    的时间戳都会变，git 里满屏都是"文件被改了但内容没变"——这个库要靠 git
    记录每周增量，那种噪声会让 diff 失去意义。
    """
    path = text_path(text_dir, paper_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = gzip.compress(text.encode("utf-8"), compresslevel=6, mtime=0)
    tmp = path.with_suffix(".txt.gz.tmp")
    tmp.write_bytes(blob)
    atomic.replace(tmp, path)
    return len(blob)


def read_text(text_dir: Path, paper_id: str) -> str:
    path = text_path(text_dir, paper_id)
    if not path.exists():
        return ""
    return gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# 清单
# --------------------------------------------------------------------------- #

MANIFEST = "manifest.jsonl"


def load_manifest(text_dir: Path) -> dict[str, dict[str, Any]]:
    path = Path(text_dir) / MANIFEST
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[rec["paper_id"]] = rec
    return out


def save_manifest(text_dir: Path, entries: dict[str, dict[str, Any]]) -> None:
    """写清单。一行一篇，按 paper_id 排序——顺序稳定 git diff 才读得懂。

    清单是**派生产物**，随时可以从 ``data/text/`` 扫出来重建，
    所以它不该成为任何流程的唯一真相源。
    """
    path = Path(text_dir) / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for pid in sorted(entries):
            fh.write(json.dumps(entries[pid], ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")
    atomic.replace(tmp, path)


# --------------------------------------------------------------------------- #
# 批量
# --------------------------------------------------------------------------- #

def select(
    store: PaperStore,
    *,
    scope: str = "all",
    venue: str | None = None,
    only_missing: bool = True,
    text_dir: Path | None = None,
) -> list[CanonicalPaper]:
    papers = list(store.papers.values())
    if scope == "rl":
        papers = [p for p in papers if p.is_rl_effective]
    if venue:
        papers = [p for p in papers if p.venue.lower() == venue.lower()]
    # 有取全文的路子的才排队：有 arXiv ID，或有已下好的本地 PDF
    papers = [p for p in papers if p.arxiv_id or (p.local_pdf and (ROOT / p.local_pdf).exists())]
    if only_missing and text_dir is not None:
        papers = [p for p in papers if not text_path(text_dir, p.paper_id).exists()]
    return sorted(papers, key=lambda p: p.paper_id)


def run(
    store: PaperStore,
    fetcher: Fetcher,
    cfg: dict[str, Any],
    *,
    scope: str = "all",
    venue: str | None = None,
    limit: int | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """批量抽取。**只输出聚合计数**，不回显任何正文（CLAUDE.md token 纪律）。"""
    text_dir = ROOT / cfg["paths"]["text"]
    text_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(text_dir)

    pool = list(store.papers.values())
    if scope == "rl":
        pool = [p for p in pool if p.is_rl_effective]
    if venue:
        pool = [p for p in pool if p.venue.lower() == venue.lower()]

    papers = select(store, scope=scope, venue=venue, only_missing=not refresh, text_dir=text_dir)
    if limit:
        papers = papers[:limit]
    if refresh:
        for paper in papers:
            text_path(text_dir, paper.paper_id).unlink(missing_ok=True)

    stats: dict[str, Any] = {
        "scope": scope, "venue": venue or "all",
        "pool": len(pool), "targets": len(papers),
        "ok": 0, "too_short": 0, "failed": 0,
        "by_source": {}, "chars": 0, "mb": 0.0,
    }
    stats["no_source"] = sum(
        1 for p in pool if not (p.arxiv_id or (p.local_pdf and (ROOT / p.local_pdf).exists()))
    )

    today = date.today().isoformat()
    for i, paper in enumerate(papers, 1):
        out = extract_one(paper, fetcher, cfg)
        stats[out.status] = stats.get(out.status, 0) + 1
        if out.status not in ("ok", "too_short"):
            continue
        nbytes = write_text(text_dir, paper.paper_id, out.text)
        stats["chars"] += out.chars
        stats["mb"] += nbytes / 1048576
        stats["by_source"][out.source] = stats["by_source"].get(out.source, 0) + 1
        manifest[paper.paper_id] = {
            "paper_id": paper.paper_id,
            "source": out.source,
            "chars": out.chars,
            "pages": out.pages,
            "bytes": nbytes,
            "extractor": VERSION,
            "latex_pipeline": latex_mod.VERSION,
            "extracted": today,
        }
        if i % 200 == 0:
            save_manifest(text_dir, manifest)
            log.info("抽取 %d/%d  成功 %d 过短 %d 失败 %d",
                     i, len(papers), stats["ok"], stats["too_short"], stats["failed"])

    save_manifest(text_dir, manifest)
    stats["mb"] = round(stats["mb"], 1)
    stats["manifest_entries"] = len(manifest)
    stats["http"] = dict(fetcher.stats)
    return stats
