"""验证「这条 feed 有没有往前滚」这条判据——**离线、只读、不联网**。

为什么需要它
------------

2026-09-22 实测事故：``cs.AI`` 那条 feed **把周一那 214 条原样又发了一遍**，
只把每一条的 ``<pubDate>`` 重戳成新日期。而当天**每一个现成的判据都是绿的**——
``feed_build`` 是当天、``http`` 是 ``3 fetched / 0 failed``、``raw_unchanged_feeds``
是 ``[]``、退出码 ``0``。根因是结构性的：**新鲜度信号全是源自己提供的、且源可以改写**。

判据换成"**今天标 ``new``/``cross`` 的条目里，有多少上一份 raw 里已经有了**"
（理由与实测数字全在 :mod:`rllib.feedwatch` 的模块 docstring 里）。

⚠️ **这个脚本最容易写成"什么都证明不了"的样子**，所以每个用例都钉住一个**具体的
反例**，而不是复述实现：

* 用例 3（部分停摆）钉的是**换判据的理由**——它让"最大值前进了"与"停摆"同时成立，
  即**初稿那条 max-ID 判据会漏掉的那一例**。
* 用例 5 用的是 ``math.GT/0601001``，**不是** ``cs/0601001``。这两个都是旧式 ID，
  但只有前者**含点**，而初稿的判别法正是"含不含点"⇒ 只测后者的话这个用例**恒绿**、
  一点用都没有。
* 用例 11 分两次调用，因为 ``_archive_superseded`` 有两个分支（字节不同 / 相同），
  而 **09-22 那次事故走的是"相同"那条**（三跑 raw 逐字节一样）——只测"不同"会让人
  以为它在 09-22 上救过场，那是**假证据**。

⚠️ **全程在临时目录里跑，绝不写真实库**，也**不发任何网络请求**。
唯一碰真实库的是 ``--replay``，而它**只读**。

用法::

    python scripts/check_feed_advance.py
    python scripts/check_feed_advance.py --replay   # 另加：在真实档案上重放 09-22 事故
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import daily_arxiv_fetch as daf  # noqa: E402  （同目录；模块级无副作用，main() 有 __main__ 保护）
from rllib import feedwatch  # noqa: E402
from rllib.adapters import arxiv_rss  # noqa: E402
from rllib.console import use_utf8_console  # noqa: E402

use_utf8_console()

CAT = "cs.AI"
CATS = ("cs.LG", "stat.ML", "cs.AI")

#: 真实周报库根。**只有 ``--replay`` 会读它，且只读。**
WEEKLY = Path(r"D:\大学编程学习\arXiv周报\数据")

_WD = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MON = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def fail(msg: str) -> None:
    print(f"失败: {msg}")
    sys.exit(1)


# --------------------------------------------------------------------------
# 合成 feed
# --------------------------------------------------------------------------

def _item(arxiv_id: str | None, announce_type: str = "new", *,
          title: str | None = None, guid: bool = True, link: bool = True) -> str:
    """一个 ``<item>``。``guid=False`` / ``link=False`` 用来造"取不到 ID"的条目。"""
    head = f"<title>{title or (f'Paper {arxiv_id}' if arxiv_id else 'Untitled')}</title>"
    parts = [head]
    if arxiv_id and guid:
        parts.append(f"<guid>oai:arXiv.org:{arxiv_id}v1</guid>")
    if arxiv_id and link:
        parts.append(f"<link>https://arxiv.org/abs/{arxiv_id}</link>")
    if announce_type:
        parts.append(f"<arxiv:announce_type>{announce_type}</arxiv:announce_type>")
    parts.append("<description>arXiv:2609.00001v1 Announce Type: new Abstract: x</description>")
    return f"<item>{''.join(parts)}</item>"


def _feed(items: list[str], build: str = "Tue, 22 Sep 2026 04:00:00 +0000") -> str:
    return (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<rss xmlns:arxiv='http://arxiv.org/schemas/atom' version='2.0'><channel>"
        f"<lastBuildDate>{build}</lastBuildDate>" + "".join(items) +
        "</channel></rss>"
    )


def _build_for(day: str) -> str:
    """给某一天编一个像样的 ``<lastBuildDate>``（星期名不影响比对，但别写错）。"""
    d = date.fromisoformat(day)
    return f"{_WD[d.weekday()]}, {d.day:02d} {_MON[d.month - 1]} {d.year} 04:00:00 +0000"


def _sample(items: list[tuple[str, str]], category: str = CAT) -> feedwatch.FeedSample:
    res = arxiv_rss.parse_feed(
        _feed([_item(a, t) for a, t in items]), category=category)
    return feedwatch.FeedSample.of(res.papers, res.announce_by_key)


def _baseline(day: str, items: list[tuple[str, str]], *, category: str = CAT,
              build: str | None = None, name: str | None = None) -> feedwatch.Baseline:
    res = arxiv_rss.parse_feed(
        _feed([_item(a, t) for a, t in items], build or _build_for(day)), category=category)
    build = res.build_date
    return feedwatch.Baseline(
        day=day, name=name or f"{day}-{category}.xml",
        sample=feedwatch.FeedSample.of(res.papers, res.announce_by_key),
        build_date=build,
        build_date_matches_name=feedwatch.build_date_matches(build, day),
    )


# --------------------------------------------------------------------------
# 真实档案重放（--replay）
# --------------------------------------------------------------------------

def replay() -> dict[str, object]:
    """在**真实档案**上重放 09-22 那次事故。**只读。**

    两天都从 ``raw-archive/`` 里读，**不读 ``raw/``**：``raw/2026-09-22-*.xml``
    是流水线**有权覆盖**的文件，一次 ``--date 2026-09-22`` 的回填会把它换成一份
    新的抓取（滚动窗口里已经没有那次重发了），那时本函数会在一次**完全合法**的
    重跑上失败。⇒ **绝不要把它接进任何常跑的门禁**，它只能是一个显式的开关。
    """
    week = WEEKLY / "2026-W39"
    pairs = {                       # 分类 → (前一天那份, 当天那份)
        cat: (week / "raw-archive" / "2026-09-21" / f"2026-09-21-{cat}.xml",
              week / "raw-archive" / "2026-09-22-1201-killed" / f"2026-09-22-{cat}.xml")
        for cat in CATS
    }
    missing = [str(p) for pr in pairs.values() for p in pr if not p.exists()]
    if missing:
        print(json.dumps({"replay": "skipped", "missing": missing}, ensure_ascii=False))
        return {"replay": "skipped"}

    samples: dict[str, feedwatch.FeedSample] = {}
    baselines: dict[str, feedwatch.Baseline] = {}
    for cat, (prev_path, today_path) in pairs.items():
        # 与 daily_arxiv_fetch._load_baseline 同一套读法：字节 → utf-8，不走 locale。
        prev_res = arxiv_rss.parse_feed(prev_path.read_bytes().decode("utf-8"), category=cat)
        today_res = arxiv_rss.parse_feed(today_path.read_bytes().decode("utf-8"), category=cat)
        samples[cat] = feedwatch.FeedSample.of(today_res.papers, today_res.announce_by_key)
        baselines[cat] = feedwatch.Baseline(
            day="2026-09-21", name=prev_path.name,
            sample=feedwatch.FeedSample.of(prev_res.papers, prev_res.announce_by_key),
            build_date=prev_res.build_date,
            build_date_matches_name=feedwatch.build_date_matches(prev_res.build_date, "2026-09-21"),
        )

    report = feedwatch.feed_report(samples, baselines)
    advance = report["feed_advance"]

    if report["feed_stalled"] != ["cs.AI"]:
        fail(f"--replay: 09-22 应只把 cs.AI 判为停摆，实得 {report['feed_stalled']}")
    ai = advance["cs.AI"]                       # type: ignore[index]
    if ai["reissued"] != ai["new_cross"] or not ai["reissued"]:
        fail(f"--replay: cs.AI 应是 100% 重发，实得 {ai['reissued']}/{ai['new_cross']}")
    if ai["delta"] != 0:
        fail(f"--replay: cs.AI 的最大 ID 应原地不动（delta 0），实得 {ai['delta']}")
    for cat in ("cs.LG", "stat.ML"):
        e = advance[cat]                        # type: ignore[index]
        if e["reissued"] != 0:
            fail(f"--replay: {cat} 是正常的，却被判重发 {e['reissued']} 条")
        if not e["delta"] or e["delta"] <= 0:
            fail(f"--replay: {cat} 的最大 ID 应前进，实得 delta={e['delta']}")
    if report["feed_baseline_mismatch"]:
        fail(f"--replay: 基线文件名应与构建日一致，实得对不上 {report['feed_baseline_mismatch']}")

    return {
        "replay": "ok",
        "stalled": report["feed_stalled"],
        "readings": {c: {"latest_id": e["latest_id"], "latest_id_prev": e["latest_id_prev"],
                         "delta": e["delta"], "new_cross": e["new_cross"],
                         "reissued": e["reissued"]}
                     for c, e in advance.items()},      # type: ignore[union-attr]
    }


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="校验 feed 停摆判据（离线、只读）。")
    ap.add_argument("--replay", action="store_true",
                    help="另加：在真实 raw-archive 上重放 2026-09-22 那次事故（只读）")
    args = ap.parse_args()

    cases = 0

    # ---- 1. 正常推进 --------------------------------------------------------
    prev = {CAT: _baseline("2026-09-21", [("2609.22086", "new"), ("2609.22000", "cross")])}
    r = feedwatch.feed_report({CAT: _sample([("2609.24985", "new"), ("2609.24900", "cross")])}, prev)
    e = r["feed_advance"][CAT]                  # type: ignore[index]
    if r["feed_stalled"] or e["reissued"] or not e["checked"]:
        fail(f"1 正常推进: {r['feed_stalled']} / reissued={e['reissued']}")
    if e["delta"] != 260924985 - 260922086:
        fail(f"1 正常推进: delta 应为 {260924985 - 260922086}，实得 {e['delta']}")
    cases += 1

    # ---- 2. 逐字节重发（09-22 事故重演）------------------------------------
    prev = {CAT: _baseline("2026-09-21", [("2609.22086", "new"), ("2609.22000", "cross")])}
    r = feedwatch.feed_report({CAT: _sample([("2609.22086", "new"), ("2609.22000", "cross")])}, prev)
    e = r["feed_advance"][CAT]                  # type: ignore[index]
    if r["feed_stalled"] != [CAT] or e["reissued"] != 2:
        fail(f"2 原样重发: {r['feed_stalled']} / reissued={e['reissued']}")
    if e["delta"] != 0 or e["latest_id"] != e["latest_id_prev"]:
        fail(f"2 原样重发: 最大 ID 应原地不动，实得 {e['latest_id']} vs {e['latest_id_prev']}")
    cases += 1

    # ---- 3. **部分停摆**——换判据的理由就在这里 -----------------------------
    # 今天的最大值**前进了**（有两条新 ID），所以"最大 ID 是否前进"这条判据会说"健康"；
    # 而其中两条是昨天原样重发的。⇒ 这一例是 max-ID 判据**结构上抓不到**的。
    prev = {CAT: _baseline("2026-09-21", [("2609.22086", "new"), ("2609.22000", "new"),
                                          ("2609.21900", "new"), ("2609.21800", "new")])}
    r = feedwatch.feed_report({CAT: _sample([("2609.24985", "new"), ("2609.24900", "new"),
                                             ("2609.22086", "new"), ("2609.22000", "new")])}, prev)
    e = r["feed_advance"][CAT]                  # type: ignore[index]
    if e["delta"] <= 0:
        fail(f"3 部分停摆: 这一例的最大值**必须**前进（否则它证明不了什么），实得 {e['delta']}")
    if r["feed_stalled"] != [CAT] or e["reissued"] != 2:
        fail(f"3 部分停摆: 应报出停摆且 reissued=2，实得 {r['feed_stalled']} / {e['reissued']}")
    cases += 1

    # ---- 4. 周末那份空 feed 不是停摆 ----------------------------------------
    prev = {CAT: _baseline("2026-09-21", [("2609.22086", "new")])}
    r = feedwatch.feed_report({CAT: _sample([])}, prev)
    e = r["feed_advance"][CAT]                  # type: ignore[index]
    if r["feed_stalled"] or e["reissued"] != 0 or e["items"] != 0 or not e["checked"]:
        fail(f"4 空 feed: {r['feed_stalled']} / {e}")
    cases += 1

    # ---- 4b. 没有基线 ⇒ **没比成**，不是"没停摆" ---------------------------
    r = feedwatch.feed_report({CAT: _sample([("2609.24985", "new")])}, {})
    e = r["feed_advance"][CAT]                  # type: ignore[index]
    if e["checked"] or not e["unchecked_reason"] or r["feed_advance_unchecked"] != [CAT]:
        fail(f"4b 无基线: {e}")
    if r["feed_stalled"]:
        fail("4b 无基线: 没比成不能报成停摆")
    cases += 1

    # ---- 4c. 抓失败的 feed 也要在表里有一席之地 ----------------------------
    r = feedwatch.feed_report({CAT: None}, {CAT: _baseline("2026-09-21", [("2609.22086", "new")])})
    e = r["feed_advance"][CAT]                  # type: ignore[index]
    if e["checked"] or e["unchecked_reason"] != "feed 没抓到" or e["latest_id"] is not None:
        fail(f"4c 抓失败: {e}")
    if r["feed_advance_unchecked"] != [CAT]:
        fail("4c 抓失败: 必须出现在 feed_advance_unchecked 里，否则它只剩'缺席'")
    cases += 1

    # ---- 5. 旧式 ID：**必须测含点的那个**（math.GT/ 而不是 cs/）------------
    res = arxiv_rss.parse_feed(
        _feed([_item("2609.24985", "new"), _item("math.GT/0601001", "new"),
               _item("hep-th/9901001", "new"), _item("cs/0601001", "new")]), category=CAT)
    for old in ("math.GT/0601001", "hep-th/9901001", "cs/0601001"):
        if feedwatch.arxiv_seq(old) is not None:
            fail(f"5 旧式 ID: {old} 的序号应为 None")
    if feedwatch.max_arxiv_id(res.papers) != "2609.24985":
        fail(f"5 旧式 ID: 最大值应只由新式条目决定，实得 {feedwatch.max_arxiv_id(res.papers)}")
    if len(res.papers) != 4:
        fail(f"5 旧式 ID: 四条都该被解析出来，实得 {len(res.papers)}")
    cases += 1

    # ---- 6. 连 ID 都取不到的条目不能把判据炸掉 ------------------------------
    res = arxiv_rss.parse_feed(
        _feed([_item("2609.24985", "new"), _item(None, "new", guid=False, link=False)]),
        category=CAT)
    if res.no_id != 1:
        fail(f"6 无 ID 条目: no_id 应为 1，实得 {res.no_id}")
    if feedwatch.max_arxiv_id(res.papers) != "2609.24985":
        fail("6 无 ID 条目: 不该影响最大值")
    if feedwatch.arxiv_seq(None) is not None:
        fail("6 无 ID 条目: arxiv_seq(None) 应为 None")
    cases += 1

    # ---- 7. 同日重跑：`_recent_raws` 必须排除今天 ---------------------------
    # 这是 `_prev_raw_path` 原 docstring 钉住的不变量（"跟自己比 = 恒真"）。
    with tempfile.TemporaryDirectory() as td:
        weekly = Path(td) / "weekly"
        raw = weekly / "2026-W39" / "raw"
        raw.mkdir(parents=True)
        today_xml = _feed([_item("2609.24985", "new")])
        (raw / "2026-09-21-cs.AI.xml").write_text(today_xml, encoding="utf-8")
        (raw / "2026-09-22-cs.AI.xml").write_text(today_xml, encoding="utf-8")
        got = [d for d, _ in daf._recent_raws(weekly, "2026-09-22", "cs.AI")]
        if got != ["2026-09-21"]:
            fail(f"7 同日重跑: 应只回 2026-09-21，实得 {got}")
        if daf._prev_raw_path(weekly, "2026-09-22", "cs.AI").name != "2026-09-21-cs.AI.xml":
            fail("7 同日重跑: _prev_raw_path 没排除今天")
        cases += 1

        # ---- 10. `raw-archive/` 里的同名文件**不是**基线候选 -----------------
        # 守住 `*/raw/*` 模式里那个 `raw` 段——`2026-09-22-1201-killed/` 里装的
        # 正是**停摆的**那份 feed，它若变成候选，判据就被喂了错药。
        arch = weekly / "2026-W39" / "raw-archive" / "2026-09-22-1201-killed"
        arch.mkdir(parents=True)
        (arch / "2026-09-23-cs.AI.xml").write_text(today_xml, encoding="utf-8")
        got = [d for d, _ in daf._recent_raws(weekly, "2026-09-24", "cs.AI")]
        if got != ["2026-09-22", "2026-09-21"]:
            fail(f"10 归档不当基线: 应只回 raw/ 下那两份，实得 {got}")
        cases += 1

    # ---- 8. 基线的 build_date 与文件名日期不符 ⇒ 标记出来 -------------------
    # 真实动因：2026-09-21 的 09:02 那趟把 `Sun, 20 Sep 2026` 写进了名为
    # `2026-09-21-*.xml` 的文件——**文件名在这个库里撒过谎**。
    prev = {CAT: _baseline("2026-09-21", [("2609.22086", "new")],
                           build="Sun, 20 Sep 2026 04:00:00 +0000")}
    r = feedwatch.feed_report({CAT: _sample([("2609.24985", "new")])}, prev)
    if r["feed_baseline_mismatch"] != [CAT]:
        fail(f"8 构建日对不上: {r['feed_baseline_mismatch']}")
    if r["feed_advance"][CAT]["checked"] is not True:      # type: ignore[index]
        fail("8 构建日对不上: 对不上只该**标记**，不该中止判定")
    cases += 1

    # ---- 9. 全 replace 的 feed 不是停摆（不许假阳性）------------------------
    prev = {CAT: _baseline("2026-09-21", [("2305.16056", "new")])}
    r = feedwatch.feed_report({CAT: _sample([("2609.24985", "replace"),
                                             ("2609.24900", "replace-cross")])}, prev)
    e = r["feed_advance"][CAT]                  # type: ignore[index]
    if r["feed_stalled"] or e["new_cross"] != 0 or e["items"] != 2:
        fail(f"9 全 replace: {r['feed_stalled']} / {e}")
    cases += 1

    # ---- 11. 覆盖前自动留档（两个分支都要测）--------------------------------
    with tempfile.TemporaryDirectory() as td:
        week_dir = Path(td) / "2026-W39"
        raw = week_dir / "raw"
        raw.mkdir(parents=True)
        dst = raw / "2026-09-22-cs.AI.xml"
        dst.write_bytes(b"OLD")

        # (a) 字节**不同** ⇒ 归档，并且清单是裸文件名形
        out = daf._archive_superseded(dst, b"NEW", week_dir, "2026-09-22", "1337")
        if out is None or out.read_bytes() != b"OLD":
            fail("11a 留档: 旧字节应被原样留档")
        want = f"{hashlib.sha256(b'OLD').hexdigest()} *2026-09-22-cs.AI.xml\n"
        sums = out.parent / "SHA256SUMS.txt"
        if sums.read_text(encoding="utf-8") != want:
            fail(f"11a 留档: 清单应为裸文件名形，实得 {sums.read_text(encoding='utf-8')!r}")
        if not out.parent.name.endswith("-superseded"):
            fail(f"11a 留档: 目录名应以 -superseded 收尾，实得 {out.parent.name}")

        # (b) 字节**相同** ⇒ 不归档。**这一条才是 09-22 那天的真实情形**
        #     （三跑 raw 逐字节一样），只测 (a) 会让人以为它在 09-22 上救过场。
        dst.write_bytes(b"NEW")
        if daf._archive_superseded(dst, b"NEW", week_dir, "2026-09-22", "1339") is not None:
            fail("11b 留档: 字节相同就不该归档")
        if sums.read_text(encoding="utf-8") != want:
            fail("11b 留档: 不该多出一行清单")

        # (c) 目标不存在（首跑）⇒ 不归档，也不抛
        dst.unlink()
        if daf._archive_superseded(dst, b"NEW", week_dir, "2026-09-22", "1340") is not None:
            fail("11c 留档: 目标不存在时不该归档")
        cases += 1

    # ---- 13. 读盘那条路：`_load_baseline`（唯一真碰文件系统的函数）-----------
    # 上面 13 个用例全走 `FeedSample.of` / `feed_report`（纯内存）。**这个函数不是**：
    # 它 `read_bytes().decode("utf-8")`、跳过解析不了的、跳过空 feed、还要算
    # `build_date_matches_name`。一条都不测的话，"基线能读出来"这件事在本套件里
    # **没有任何证据**。
    with tempfile.TemporaryDirectory() as td:
        weekly = Path(td) / "weekly"
        raw = weekly / "2026-W39" / "raw"
        raw.mkdir(parents=True)

        # 往回走要跨过三种"不用"：读不动的、解析不了的、解析得动但 0 条的。
        (raw / "2026-09-20-cs.AI.xml").write_bytes(b"<rss><channel>")
        (raw / "2026-09-19-cs.AI.xml").write_text(_feed([]), encoding="utf-8")
        # 要返回的那份：**含非 ASCII**，且构建日**不是**文件名那天（09-21 那次的形状）。
        good = _feed([_item("2609.22086", "new", title="Über die 中文标题"),
                      _item("2609.22000", "cross")],
                     build="Sun, 20 Sep 2026 04:00:00 +0000")
        (raw / "2026-09-18-cs.AI.xml").write_text(good, encoding="utf-8")

        b = daf._load_baseline(weekly, "2026-09-21", "cs.AI")
        if b is None:
            fail("13 读盘: 09-18 那份是好的，应被取到")
        if b.day != "2026-09-18" or b.name != "2026-09-18-cs.AI.xml":
            fail(f"13 读盘: 应回退到 09-18，实得 {b.day}/{b.name}")
        if b.sample.ids != {"2609.22086", "2609.22000"}:
            fail(f"13 读盘: ID 集不对，实得 {sorted(b.sample.ids)}")
        # ⚠️ **`cross` 在 `new_cross` 里**（`_FIRST_ANNOUNCE = {"new", "cross"}`）——
        # 它和 `new` 都是**首次公告**的视角，只有 `replace*` 才是重新公告。
        # 这条断言一度被我写成"cross 不该进来"，被自己写的用例当场抓住。
        if b.sample.new_cross != {"2609.22086", "2609.22000"}:
            fail(f"13 读盘: new+cross 两条都该在，实得 {sorted(b.sample.new_cross)}")
        if b.sample.max_id != "2609.22086":
            fail(f"13 读盘: 最大值不对，实得 {b.sample.max_id}")
        if b.build_date_matches_name is not False:
            # 触发 `build_date_matches` 里那条硬编码月份表；若谁改成 strptime，
            # 本机中文 locale 下这里会抛而不是返回 False。
            fail("13 读盘: 构建日 09-20 ≠ 文件名 09-18，应标 False")

        # 非 ASCII 必须真的活下来——这一条守的是"别把 read_bytes().decode('utf-8')
        # 换成不带编码的 read_text()"。ID 是 ASCII，解码错了照样能算出正确的最大值，
        # 所以**只看 ids 是看不出 mojibake 的**，必须看标题。
        got = arxiv_rss.parse_feed((raw / "2026-09-18-cs.AI.xml").read_bytes()
                                   .decode("utf-8"), category=CAT)
        if "Über" not in got.papers[0].title:
            fail(f"13 读盘: 非 ASCII 标题没活下来，实得 {got.papers[0].title!r}")

        # 一整条往回都是空的/坏的 ⇒ `None`（首跑），而不是抛
        for p in raw.iterdir():
            p.unlink()
        (raw / "2026-09-19-cs.AI.xml").write_text(_feed([]), encoding="utf-8")
        if daf._load_baseline(weekly, "2026-09-21", "cs.AI") is not None:
            fail("13 读盘: 只有空 feed 时应返回 None")
        cases += 1

    # ---- 12. 真事故重放（可选）---------------------------------------------
    out: dict[str, object] = {"ok": True, "cases": cases}
    if args.replay:
        out.update(replay())

    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
