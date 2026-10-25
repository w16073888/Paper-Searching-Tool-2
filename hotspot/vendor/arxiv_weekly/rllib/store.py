"""JSONL 真相源 + 跨源去重索引。

铁律：``data/papers.jsonl`` 是**唯一真相源**，parquet / bib / Markdown 全是产物。

写盘用**全量排序重写**而非 append：
- 12000 条 × ~2KB ≈ 24MB，内存与耗时都可忽略
- 排序后输出是确定性的 → git diff 只反映真实变化，不会被插入顺序污染

合并语义的关键约定（三条都不能破）：
1. ``paper_id`` 以**已存在记录**为准，永不因新数据而改变。
2. ``user`` 整个字段是人工区，**只读不写**。
3. ``rl.manual_override`` 一旦被人工设置，任何重跑都不得覆盖。
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Iterable, Literal

from . import atomic
from .schema import CanonicalPaper, Provenance

log = logging.getLogger(__name__)

UpsertResult = Literal["new", "updated", "unchanged"]

#: 空值就填、非空不覆盖的字段。
#: 这些是"外部世界的事实"，先到的源未必更权威，但覆盖只会带来噪声。
_FILL_IF_EMPTY = (
    "source_key", "doi", "arxiv_id", "openreview_id",
    "url", "pdf_url", "local_pdf", "abstract", "track",
    # license：**不可撤销**（arXiv 官方规定，见 MEASUREMENTS.md §四c #24），
    # 所以"空值才填、非空不覆盖"在这里不是妥协，而是**正确语义**——
    # 后来的值不可能比先到的更新。反过来它同时挡住了那类事故：
    # 某次 OAI-PMH 抓取失败（返回值是 None）不会把已经采到的许可抹成未知。
    "license",
)

#: 锁文件多久没动就算**僵尸锁**。因为锁只在 save() 内部持有（约 1 秒），
#: 正常不可能超过这个数；超了就是上次写入中途进程被杀，留下的空壳。
_LOCK_STALE_SECONDS = 120.0


class _FileLock:
    """跨进程互斥锁，用 ``O_CREAT|O_EXCL`` 抢锁文件实现。

    存在的理由很具体：``enrich`` 和 ``backfill-abstracts`` 这类任务会跑几小时、
    期间每几十篇落一次盘。两个这样的进程同时写 ``papers.jsonl``，**后写的那个
    会把先写的整片覆盖**——因为它们各自在启动时载入了全库快照。静默丢几小时
    的成果，而且不报任何错。

    锁只在 :meth:`PaperStore.save` 内部短暂持有，所以正常竞争窗口约一秒。
    """

    def __init__(self, path: Path, timeout: float = 120.0, poll: float = 0.25) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self.poll = poll

    def __enter__(self) -> _FileLock:
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._is_stale():
                    log.warning("清理僵尸锁（上次写入进程可能被杀）: %s", self.path)
                    self._unlink_quietly()
                    continue
                if time.monotonic() > deadline:
                    # 硬闯：宁可冒一次覆写的风险，也不让周更管道永久卡死。
                    # 真走到这里说明锁的清理逻辑出了问题，日志里有记录可查。
                    log.error("等锁超时，强行写入: %s", self.path)
                    return self
                time.sleep(self.poll)
                continue
            try:
                os.write(fd, str(os.getpid()).encode())
            finally:
                os.close(fd)
            return self

    def __exit__(self, *exc: object) -> None:
        self._unlink_quietly()

    def _is_stale(self) -> bool:
        try:
            return (time.time() - self.path.stat().st_mtime) > _LOCK_STALE_SECONDS
        except OSError:
            return False

    def _unlink_quietly(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


def _would_downgrade(field: str, disk: dict, mine: dict) -> bool:
    """这次合并落盘是不是"把已知抹成未知"。

    :data:`_FILL_IF_EMPTY` 那组字段的语义是「空值才填、非空不覆盖」，但那只在
    **内存合并**那一层成立。走合并落盘时它有两个相反的坑，都会丢数据：

    * **没认领** → 磁盘一律赢，于是"内存里刚填上的值"被静默丢掉（填了等于没填）。
    * **认领了** → 本进程这次**没取到**（值是 ``None``）时会把磁盘上的值一起写掉，
      把已经采到的事实抹成"没读到"。

    所以判据放在这里：**本进程对这个字段无话可说（空）时，不许改动磁盘上的非空值。**
    空是"没读到"，不等于"没有"——这是本库反复栽的那个病；而这一条让 `daily_arxiv_fetch`
    这类"填缺"任务可以放心认领自己的字段，不必担心并行时抹掉别人或自己的成果。
    """
    if field not in _FILL_IF_EMPTY:
        return False
    return disk.get(field) not in (None, "") and mine.get(field) in (None, "")


def _union_provenance(disk: list[dict], mine: list[dict]) -> list[dict]:
    """:attr:`CanonicalPaper.provenance` 的合并语义是**并集**，不是覆盖。

    这一条是特例，因为 provenance 是"只增不减的账本"：没人会删溯源条目，
    多源合并的整个意义就在于把所有来源都记下来。而 ``enrich`` 和
    ``backfill-abstracts`` 都会往里追加自己那条——两个进程并行时若用
    覆盖语义，后写的那个就会把对方的溯源记录抹掉，正好毁掉这个字段的用途。

    其它列表字段（如 ``authors``）不这么做：它们有"哪个源更权威"的问题，
    老老实实按所有权覆盖反而更可预期。
    """
    seen = {(p.get("source"), p.get("source_id")) for p in disk}
    out = list(disk)
    for entry in mine:
        key = (entry.get("source"), entry.get("source_id"))
        if key not in seen:
            seen.add(key)
            out.append(entry)
    return out


class PaperStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.papers: dict[str, CanonicalPaper] = {}
        self._index: dict[str, str] = {}          # identity key -> paper_id
        self.stats_counter: Counter[str] = Counter()

    # ------------------------------------------------------------------ #
    # 读写
    # ------------------------------------------------------------------ #

    def load(self) -> PaperStore:
        self.papers.clear()
        self._index.clear()
        if not self.path.exists():
            return self
        with self.path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    paper = CanonicalPaper.model_validate_json(line)
                except Exception as exc:                       # noqa: BLE001
                    # 单条坏行不该毁掉整个库。记下来，由质量报告暴露。
                    log.warning("跳过损坏记录 %s:%d (%s)", self.path.name, lineno, exc)
                    self.stats_counter["corrupt_lines"] += 1
                    continue
                self.papers[paper.paper_id] = paper
                self._index_paper(paper)
        return self

    def _index_paper(self, paper: CanonicalPaper) -> None:
        for key in paper.identity_keys():
            self._index.setdefault(key, paper.paper_id)

    def save(self, *, owned_fields: Iterable[str] | None = None) -> None:
        """确定性写出：按 paper_id 排序，一行一条。

        ``owned_fields`` 是**字段所有权声明**，用来防止并行任务互相抹掉数据：

        * ``None`` —— "我拥有全部字段"，整库以本进程内存为准。
          适用于秒级、单发、不与长任务并行的命令（``sync`` / ``reindex``）。
        * 给定字段名 —— 走**合并落盘**：在锁内重新读一遍磁盘现状，
          只把这些字段从内存覆盖上去，其余字段一律以磁盘为准。
          ``enrich`` 认领 ``arxiv_id`` 系字段、``backfill-abstracts`` 认领
          ``abstract``、``filter`` 认领 ``rl``，三者并行也互不干扰。

        为什么必须有这个：这些任务各自在启动时载入全库快照，跑几小时。
        没有合并的话，最后落盘的那个会把另一个的成果整片覆盖掉。

        合并落盘时另有一条**不许把已知抹成未知**的闸：本进程对
        :data:`_FILL_IF_EMPTY` 里的字段给出空值时，磁盘上的非空值原地保留。
        见 :func:`_would_downgrade`。
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _FileLock(self.path.with_suffix(".lock")):
            if owned_fields is None:
                records = {
                    pid: paper.model_dump(mode="json")
                    for pid, paper in self.papers.items()
                }
            else:
                records = self._read_raw()
                fields = tuple(owned_fields)
                for pid, paper in self.papers.items():
                    mine = paper.model_dump(mode="json")
                    disk = records.get(pid)
                    if disk is None:                 # 本进程新增的论文
                        records[pid] = mine
                    else:
                        for field in fields:
                            if field == "provenance":
                                disk[field] = _union_provenance(
                                    disk.get(field) or [], mine.get(field) or []
                                )
                            elif _would_downgrade(field, disk, mine):
                                continue
                            else:
                                disk[field] = mine[field]
            self._write_raw(records)

    def _read_raw(self) -> dict[str, dict]:
        """把磁盘上的 jsonl 读成原始 dict，供合并落盘用。

        坏行直接跳过——合并的前提是"尽力保住双方的数据"，
        不能因为一行脏数据就让整次落盘失败。
        """
        out: dict[str, dict] = {}
        if not self.path.exists():
            return out
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = rec.get("paper_id")
                if pid:
                    out[pid] = rec
        return out

    def _write_raw(self, records: dict[str, dict]) -> None:
        """先写临时文件再原子替换——中途被杀不会留下半个文件。

        替换走 :func:`rllib.atomic.replace`，因为 Windows 上**任何进程打开着
        目标文件都会让 ``os.replace`` 直接失败**。实测：``enrich`` 与
        ``backfill-abstracts`` 并行跑两小时后，落盘在 ``tmp.replace(...)``
        处以 ``PermissionError: [WinError 5]`` 崩掉，几千条成果卡在临时文件里。
        文件锁挡不住这种外部读者——见 :mod:`rllib.atomic` 的说明。
        """
        tmp = atomic.tmp_for(self.path)
        try:
            with tmp.open("w", encoding="utf-8", newline="\n") as fh:
                for pid in sorted(records):
                    fh.write(json.dumps(records[pid], ensure_ascii=False,
                                        separators=(",", ":")))
                    fh.write("\n")
            atomic.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #

    def find(self, incoming: CanonicalPaper) -> CanonicalPaper | None:
        """用全部 identity key 反查已有记录。任一命中即视为同一篇。"""
        for key in incoming.identity_keys():
            pid = self._index.get(key)
            if pid is not None:
                return self.papers.get(pid)
        return None

    def __len__(self) -> int:
        return len(self.papers)

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #

    def upsert(self, incoming: CanonicalPaper) -> UpsertResult:
        existing = self.find(incoming)
        if existing is None:
            self._insert(incoming)
            return "new"

        if self._merge(existing, incoming):
            return "updated"
        return "unchanged"

    def _insert(self, paper: CanonicalPaper) -> None:
        paper.paper_id = paper.paper_id or _mint_paper_id(paper)
        self.papers[paper.paper_id] = paper
        self._index_paper(paper)

    def _merge(self, dst: CanonicalPaper, src: CanonicalPaper) -> bool:
        """把 src 的新信息并进 dst。返回是否真的产生了变化。

        ``dst.user`` 与 ``dst.rl`` 全程不参与——前者是人工区，后者由本库自己的
        筛选器产出，都不该被数据源覆盖。
        """
        changed = False

        for field in _FILL_IF_EMPTY:
            if getattr(dst, field) in (None, "") and getattr(src, field) not in (None, ""):
                setattr(dst, field, getattr(src, field))
                changed = True

        # 作者：只在原来为空时采纳（不同源的作者拼写可能不同，覆盖会引入抖动）
        if not dst.authors and src.authors:
            dst.authors = list(src.authors)
            changed = True

        # topics 取并集，保持稳定顺序
        if src.topics:
            merged = list(dict.fromkeys([*dst.topics, *src.topics]))
            if merged != dst.topics:
                dst.topics = merged
                changed = True

        # 摘要只在原来为空时采纳，避免不同源截断版本互相覆盖
        if not dst.abstract and src.abstract:
            dst.abstract = src.abstract
            changed = True

        # 引用数与时间戳由 enrich 显式驱动，此处不碰
        if src.provenance:
            existing_sources = {p.source for p in dst.provenance}
            for prov in src.provenance:
                if prov.source not in existing_sources:
                    dst.provenance.append(prov)
                    changed = True

        if changed:
            dst.last_updated = date.today().isoformat()
        return changed

    # ------------------------------------------------------------------ #
    # 统计（只输出聚合，不回显数据 —— 见 CLAUDE.md token 纪律）
    # ------------------------------------------------------------------ #

    def summary(self) -> dict[str, object]:
        by_venue: Counter[str] = Counter()
        missing = Counter()
        rl_hits = 0
        for p in self.papers.values():
            by_venue[f"{p.venue}{p.year}"] += 1
            rl_hits += bool(p.rl.is_rl)
            for field in ("abstract", "arxiv_id", "doi", "pdf_url"):
                if not getattr(p, field):
                    missing[field] += 1
        total = len(self.papers) or 1
        return {
            "total": len(self.papers),
            "by_venue": dict(sorted(by_venue.items())),
            "rl_matched": rl_hits,
            "missing_pct": {k: round(100 * v / total, 1) for k, v in sorted(missing.items())},
            "corrupt_lines": self.stats_counter.get("corrupt_lines", 0),
        }


def _mint_paper_id(paper: CanonicalPaper) -> str:
    """首次插入时确定 paper_id，此后永不重算。

    优先级反映"这个标识有多不可能变"：arXiv ID 与 DOI 是全局唯一且永久的，
    source_key 在会议内唯一且稳定，标题哈希只是最后兜底。
    """
    from .schema import normalize_doi, normalize_title, sha1_16

    if paper.arxiv_id:
        return f"arxiv:{paper.arxiv_id.lower()}"
    doi = normalize_doi(paper.doi)
    if doi:
        return f"doi:{doi}"
    if paper.openreview_id:
        return f"orid:{paper.openreview_id}"
    if paper.source_key:
        return f"src:{paper.venue.lower()}{paper.year}:{paper.source_key}"
    return f"title:{sha1_16(normalize_title(paper.title))}"


def load_config(path: Path) -> dict:
    import yaml

    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def iter_provenance(papers: Iterable[CanonicalPaper]) -> Iterable[Provenance]:
    for p in papers:
        yield from p.provenance
