"""构建 / 增量更新 SQLite FTS5 检索索引。

为什么是 SQLite FTS5
--------------------

* **零新依赖**。``sqlite3`` 是标准库。这是个要维护几年的本地库，每加一个依赖
  都是未来某次 ``pip install`` 失败的债。
* **增量天然**。周更 = INSERT 新论文，索引自己更新，不需要重建。
* **快**。13k 篇毫秒级。
* **可 join**。venue / year / subfield / is_rl 当筛选面。

关于正文存放（一个刻意的取舍）
------------------------------

索引用**无内容（contentless）FTS5 表**：

* 只存倒排索引，不存正文 → 约 200MB 而不是 800MB+。
* 代价是 FTS5 自带的 ``snippet()`` 不可用。所以命中段落由
  :func:`rllib.search.query.search` **回读 ``data/text/<id>.txt.gz`` 现切**。
  这其实更好：切窗逻辑完全可控，而且正文仍然只有一份、仍以文件为准
  （与「jsonl 是唯一真相源、其余皆产物」同构）。

``data/index.sqlite`` 本身也是**产物**，随时可删可重建，不进版本控制。
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from .. import ROOT
from ..paths import text_path
from ..store import PaperStore

log = logging.getLogger(__name__)

INDEX_NAME = "data/index.sqlite"

#: 索引结构版本。改动 DDL 或分词器就 +1，旧索引会被整体重建。
#: 不这么做的话，改了分词器而没重建，检索会静默地按旧规则跑——没人会发现。
SCHEMA_VERSION = "1"

_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS papers (
    paper_id   TEXT PRIMARY KEY,
    doc_rowid  INTEGER NOT NULL,
    title      TEXT NOT NULL,
    venue      TEXT,
    year       INTEGER,
    subfields  TEXT NOT NULL DEFAULT '',   -- 逗号分隔，便于 LIKE 粗筛
    is_rl      INTEGER NOT NULL DEFAULT 0,
    citations  INTEGER,
    arxiv_id   TEXT,
    doi        TEXT,
    url        TEXT,
    first_seen TEXT,                       -- 入库日期；预印本的时效排序靠它
    chars      INTEGER NOT NULL DEFAULT 0,
    body_hash  TEXT NOT NULL DEFAULT ''    -- 增量判定：正文变了才重灌索引
);
CREATE INDEX IF NOT EXISTS idx_papers_venue ON papers(venue, year);
CREATE INDEX IF NOT EXISTS idx_papers_rl    ON papers(is_rl);

-- 无内容表：正文只在灌库时过一遍，之后丢弃，索引留用。
-- contentless_delete=1 是增量更新的前提（默认不允许 DELETE）。
CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(
    title, abstract, body,
    content='', contentless_delete=1,
    tokenize='porter unicode61'
);
"""


def index_path(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg.get("search", {}).get("index_file", INDEX_NAME)


def connect(path: Path, *, create: bool = True) -> sqlite3.Connection:
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # 建索引是纯写，检索是纯读；WAL 让周更灌库时检索不被阻塞。
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    if create:
        conn.executescript(_DDL)
    return conn


def _meta(conn: sqlite3.Connection) -> dict[str, str]:
    try:
        return {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM meta")}
    except sqlite3.OperationalError:
        return {}


def _body_for(text_dir: Path, paper_id: str) -> tuple[str, int]:
    """读回正文。缺文件返回 ``("", 0)``——**不报错**，没有全文的论文照样该被检索到
    （靠标题和摘要）。忽略它才会让库看起来比实际更全。
    """
    path = text_path(text_dir, paper_id)
    if not path.exists():
        return "", 0
    try:
        raw = gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
    except (OSError, EOFError) as exc:
        log.warning("正文解压失败 %s: %s", path.name, exc)
        return "", 0
    return raw, len(raw)


def _hash(*parts: str) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(part.encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _digest(paper, body: str) -> str:
    """索引行的内容指纹。

    **必须覆盖索引里存着的每一个字段，而不只是正文。** 否则：改了
    ``rl_subfields.yaml`` 重跑 ``filter --rl`` 之后，``is_rl``/``subfields``
    变了而指纹没变，这一行就会被跳过、静默保留旧值——检索的筛选面与 jsonl
    对不上，且不报任何错。（实测踩过：改完子方向词表重建索引，仍按旧子方向筛出论文。）

    **刻意不含 ``last_updated`` / ``enrich_attempted``**：那两个字段每次富化都会变，
    含进来就等于每周把 13000 篇全量重灌一遍索引，换不到任何东西。

    注意：改动本函数的定义会让所有存量指纹失配，下次 ``index`` 会全量重灌一次
    （约 14 秒）。这是自愈的，不必手工清库。
    """
    return _hash(
        paper.title, paper.abstract or "", body, paper.venue, str(paper.year),
        ",".join(paper.rl.subfield), str(int(paper.is_rl_effective)),
        str(paper.citations), paper.arxiv_id or "", paper.doi or "",
        paper.url or "", paper.first_seen,
    )


def build(
    store: PaperStore,
    cfg: dict[str, Any],
    *,
    rebuild: bool = False,
) -> dict[str, Any]:
    """建索引或增量更新。**只输出聚合计数**（CLAUDE.md token 纪律）。

    增量判据是 :func:`_digest`（覆盖索引里存的全部字段）：正文抽取和词表调整
    都是分批进行的，所以这个命令会被反复跑；没有这个判据，每次都要把
    13000 篇重灌一遍。
    """
    path = index_path(cfg)
    if rebuild and path.exists():
        for suffix in ("", "-wal", "-shm"):
            Path(str(path) + suffix).unlink(missing_ok=True)

    text_dir = ROOT / cfg["paths"]["text"]
    conn = connect(path)
    try:
        if _meta(conn).get("schema_version") not in (None, SCHEMA_VERSION):
            conn.close()
            return build(store, cfg, rebuild=True)

        conn.execute("BEGIN")
        conn.execute(
            "INSERT INTO meta(key,value) VALUES('schema_version',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (SCHEMA_VERSION,),
        )

        known = {
            r["paper_id"]: (r["doc_rowid"], r["body_hash"])
            for r in conn.execute("SELECT paper_id, doc_rowid, body_hash FROM papers")
        }
        seen: set[str] = set()
        added = updated = unchanged = 0
        with_text = 0

        for paper in store.papers.values():
            seen.add(paper.paper_id)
            body, chars = _body_for(text_dir, paper.paper_id)
            if chars:
                with_text += 1
            digest = _digest(paper, body)

            prior = known.get(paper.paper_id)
            if prior is not None and prior[1] == digest:
                unchanged += 1
                continue

            # 正文只在灌索引这一刻过一遍，之后 FTS5 丢弃它。
            if prior is None:
                cur = conn.execute(
                    "INSERT INTO docs(title, abstract, body) VALUES(?,?,?)",
                    (paper.title, paper.abstract or "", body),
                )
                rowid = cur.lastrowid
                added += 1
            else:
                rowid = prior[0]
                conn.execute("DELETE FROM docs WHERE rowid=?", (rowid,))
                conn.execute(
                    "INSERT INTO docs(rowid, title, abstract, body) VALUES(?,?,?,?)",
                    (rowid, paper.title, paper.abstract or "", body),
                )
                updated += 1

            conn.execute(
                """
                INSERT INTO papers(paper_id, doc_rowid, title, venue, year, subfields,
                                   is_rl, citations, arxiv_id, doi, url, first_seen,
                                   chars, body_hash)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    doc_rowid=excluded.doc_rowid, title=excluded.title,
                    venue=excluded.venue, year=excluded.year,
                    subfields=excluded.subfields, is_rl=excluded.is_rl,
                    citations=excluded.citations, arxiv_id=excluded.arxiv_id,
                    doi=excluded.doi, url=excluded.url, first_seen=excluded.first_seen,
                    chars=excluded.chars, body_hash=excluded.body_hash
                """,
                (
                    paper.paper_id, rowid, paper.title, paper.venue, paper.year,
                    ",".join(paper.rl.subfield), int(paper.is_rl_effective),
                    paper.citations, paper.arxiv_id, paper.doi, paper.url,
                    paper.first_seen,
                    chars, digest,
                ),
            )

        # 库里已删掉的论文（人工清理过），索引里也不该留着
        for pid in set(known) - seen:
            conn.execute("DELETE FROM docs WHERE rowid=?", (known[pid][0],))
            conn.execute("DELETE FROM papers WHERE paper_id=?", (pid,))
        dropped = len(set(known) - seen)

        conn.execute("COMMIT")
        conn.execute("INSERT INTO docs(docs) VALUES('optimize')")
        conn.commit()

        return {
            "index": str(path.relative_to(ROOT)),
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "total": len(store.papers),
            "added": added,
            "updated": updated,
            "unchanged": unchanged,
            "dropped": dropped,
            "with_text": with_text,
            "with_text_pct": round(100 * with_text / (len(store.papers) or 1), 1),
        }
    finally:
        conn.close()


def stats(cfg: dict[str, Any]) -> dict[str, Any]:
    path = index_path(cfg)
    if not path.exists():
        return {"exists": False}
    conn = connect(path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) n, SUM(chars>0) w, SUM(is_rl) r FROM papers"
        ).fetchone()
        return {
            "exists": True,
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "docs": row["n"],
            "with_text": row["w"] or 0,
            "rl": row["r"] or 0,
        }
    finally:
        conn.close()


def dumps(obj: Any) -> str:
    """调试用：把查询结果序列化成紧凑 JSON（**不进终端**）。"""
    return json.dumps(obj, ensure_ascii=False, indent=1)
