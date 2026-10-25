"""验证周报渲染器的「哪些 frontmatter 键归程序管」这个不变量。

为什么需要它
------------

2026-09-18 实测事故：**周索引页的计数从第一次渲染起就冻住了。**

``_head()`` 原本只有一个受管键集合（卡片的 :data:`CARD_KEYS`），索引页的
``papers_in_scope`` / ``concern_pending`` / ``generated`` 一个都不在里面
⇒ 它们被当成"读者的键"**原样并回来**，于是每次重渲都把第一次的旧值写回去。
现场：同一份 ``report.md`` 里 frontmatter 写着 ``73 / 24``，而正文
（**由同一次 ``build_index()`` 产生**）写着 ``162 / 0``。退出码 0、没有一行变红。

⚠️ **这个 bug 的形状是"文件一个字都没变"**——所以任何只比"渲染前后有没有变化"
的断言都抓不到它。必须**逐键比对"写出去的值 vs 算出来的值"**。

验四件事（三个方向 + 一个假阳性方向）
-------------------------------------

1. **正向**：索引页的计数**跟着新 fm 走**（防这次的 bug 复发）；
2. **保留**：卡片与索引页上**读者的键**（``rating`` / ``my_note``）**原样在**
   （防修过头，反向把读者的批注抹了）；
3. **反向**：卡片与索引页上**程序受管键仍然更新**，不被旧值粘住；
4. **不误报**：读者删掉了受管块标记（``unmanaged``）时，**一个字节都不许写**，
   且 ``index_fm_ok`` 必须照旧为真——否则"读者手改过的页"会永远报红，
   而**一个会误报的检查比没有检查更糟**（``CLAUDE.md`` 已知风险那条）。

2026-09-19 扩到**合订本**（``<周>/digest.md``，第三套受管键 :data:`DIGEST_KEYS`）：
同上面那四个方向，**外加一条负向验证**——故意用 ``CARD_KEYS`` 写一份合订本，
逐键比**必须报出不一致**。⚠️ 不加这一条，"这套检查能抓到传错键集"就只是一句
**没量过的话**（09-18 那次是手工改源码、验完再还原；这次做成脚本内的，可重复跑）。

⚠️ **全程在临时目录里跑，绝不写真实库**，也**不发任何网络请求**
（`CLAUDE.md`：「不变量检查里不许有真实外网请求」）。真实库是否被动过，收尾时
按目录 mtime 之外的办法核：本脚本只往 ``tempfile.TemporaryDirectory()`` 里写。

用法：``python scripts/check_weekly_render.py``
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllib import ROOT  # noqa: E402
from rllib.console import use_utf8_console  # noqa: E402
from rllib.render import weekly_card as wc  # noqa: E402
from rllib.store import load_config  # noqa: E402

use_utf8_console()

WEEK = "2026-W38"
DAY = "2026-09-17"
KEY = "2601.00001"

#: 读者自己写的键。**两条支路都要它**：索引页（表头的批注）与卡片（打分）。
READER_KEYS = {"rating": 5, "my_note": "我自己写的"}

#: 一份"上一轮留下的"索引页 frontmatter。三个受管键**都取了与真值不同的值**——
#: 撞上真值就等于那个键这一轮没被检查到（第一版 ``papers_in_scope`` 写成 1，
#: 恰好等于这一轮的篇数，于是反向验证时它没有变红）。
FM_STALE = "\n".join(
    ["tags:", "- 周报", "- 索引", f"week: {WEEK}", "generated: '2000-01-01'",
     "papers_in_scope: 999", "concern_pending: 99"])

#: 合订本的同一份"上一轮留下"。四个受管键**都取了与真值不同的值**（同上一条理由：
#: 撞上真值就等于那个键这一轮没被检查到）。``days_covered`` 给一天**不同的日期**，
#: 不然它和真值都是 ``[2026-09-17]``，这一项就永远是绿的。
DIGEST_STALE = "\n".join(
    ["tags:", "- 周报", "- 合订本", f"week: {WEEK}", "generated: '2000-01-01'",
     "days_covered:", "- 1999-01-01", "papers_in_digest: 999", "library_total: 999"])


def fail(msg: str) -> None:
    print(f"失败: {msg}")
    sys.exit(1)


def _fixture(root: Path) -> Path:
    """造一个最小的周报库。**只有 ``render_index`` 真正会读的那几列。**"""
    weekly = root / "weekly"
    (weekly / WEEK).mkdir(parents=True)
    papers = [{
        "arxiv_id": KEY,
        "title": "A Minimal Fixture Paper",
        "authors": ["A. Author"],
        "first_seen": DAY,
        "topics": ["cs.LG"],
        "url": f"https://arxiv.org/abs/{KEY}",
        "pdf_url": f"https://arxiv.org/pdf/{KEY}",
        "abstract": "这是一段摘要，用来验证合订本把它印出来了。",
        "abstract_available": True,
        "rl": {"is_rl": True, "subfield": ["exploration"]},
    }]
    (weekly / "papers.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in papers), encoding="utf-8")
    (weekly / "sections.jsonl").write_text(json.dumps({
        "arxiv_id": KEY, "title": papers[0]["title"], "outcome": "ok",
        "picked": "Limitations", "picked_len": 300, "candidates": {"Limitations": 300},
        "passed_over": {}, "macro_residue": {}, "text": "正文。", "extracted": DAY,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    (weekly / WEEK / f"{DAY}.announce.jsonl").write_text(
        json.dumps({"key": KEY, "announce_type": "new"}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return weekly


def _reader_block() -> str:
    return "".join(f"{k}: {json.dumps(v, ensure_ascii=False)}\n"
                   for k, v in READER_KEYS.items())


def _assert_seeded(path: Path, sentinel: str, value: object) -> None:
    """**fixture 自检**：种下去的 frontmatter 必须解析得出来，且带着那个假旧值。

    ⚠️ 2026-09-19 补。起因是一个**已经潜伏了一天**的缺陷：``FM_STALE`` /
    ``DIGEST_STALE`` 结尾没有换行，闭合的 ``---`` 被粘到最后一行值上
    （变成 ``papers_in_scope: 999---``）⇒ **整块 frontmatter 解析失败、返回 ``{}``**。
    凡是"盘上的旧值 vs 算出来的新值"这类断言，**在空字典上全部平凡成立** ——
    **恒绿，而它看起来和"刚刚核对过"一模一样**（本库的老毛病：撞上空值或真值，
    就等于那个键这一轮根本没被检查到）。⇒ **种完必须自己验一遍**，别指望下次发现。
    """
    got = wc.existing_frontmatter(path)
    if not got:
        fail(f"fixture 不合格：{path.name} 的 frontmatter 解析不出来"
             f"（闭合的 --- 是不是粘在了最后一行值上？）")
    if got.get(sentinel) != value:
        fail(f"fixture 不合格：{path.name} 的 {sentinel} = {got.get(sentinel)!r}，"
             f"应为假旧值 {value!r} —— 撞上真值就等于这个键这一轮没被检查到")


def _seed_index(weekly: Path, begin: str, end: str, *, reader: bool, markers: bool) -> Path:
    """把一份"上一轮留下的"索引页放进临时库。``markers=False`` = 读者删了标记。"""
    fm = FM_STALE + ("\n" + _reader_block() if reader else "")
    if markers:
        head = f"---\n{fm}\n---\n{begin}\n旧正文\n{end}\n"
    else:
        head = f"---\n{fm}\n---\n旧正文（受管块标记被读者删了）\n"
    path = weekly / WEEK / wc.INDEX_NAME
    path.write_text(head + "\n我的笔记：这是我写的。\n", encoding="utf-8")
    _assert_seeded(path, "papers_in_scope", 999)
    return path


def _seed_digest(weekly: Path, begin: str, end: str, *,
                 reader: bool, markers: bool = True) -> Path:
    """把一份"上一轮留下的"合订本放进临时库。``markers=False`` = 读者删了标记。"""
    fm = DIGEST_STALE + ("\n" + _reader_block() if reader else "")
    if markers:
        head = f"---\n{fm}\n---\n{begin}\n旧正文\n{end}\n"
    else:
        head = f"---\n{fm}\n---\n旧正文（受管块标记被读者删了）\n"
    path = weekly / WEEK / wc.DIGEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(head + "\n我的笔记：这是我写的。\n", encoding="utf-8")
    _assert_seeded(path, "papers_in_digest", 999)
    return path


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    begin, end = cfg["render"]["auto_begin"], cfg["render"]["auto_end"]

    real = ROOT / cfg["paths"]["data"]
    with tempfile.TemporaryDirectory(prefix="check_weekly_render_") as td:
        root = Path(td)
        if real.is_relative_to(root) or root.is_relative_to(real):
            fail(f"隔离失效：临时目录与真实库重叠（{root}）")

        # --- 1/2/3. 正常的索引页：计数跟着走、读者的键留住 -----------------------
        weekly = _fixture(root / "case1")
        path = _seed_index(weekly, begin, end, reader=True, markers=True)
        res = wc.render_index(weekly, cfg, generated="2026-09-18")

        if res["index_fm_mismatch"]:
            fail(f"索引页 frontmatter 与算出来的对不上：{res['index_fm_mismatch']}")
        if not res["index_fm_ok"] or res["index_fm_checked"] != 1:
            fail(f"读回核对没跑起来：checked={res['index_fm_checked']} ok={res['index_fm_ok']}")

        fm = wc.existing_frontmatter(path)
        # 1. 正向：程序受管的三个键必须是**这一次**算出来的值，不是上一轮的
        if fm.get("papers_in_scope") != 1:
            fail(f"papers_in_scope 没更新（{fm.get('papers_in_scope')!r}）—— 就是这次的 bug")
        if fm.get("concern_pending") != 0:
            fail(f"concern_pending 没更新（{fm.get('concern_pending')!r}）—— 就是这次的 bug")
        if fm.get("generated") != "2026-09-18":
            fail(f"generated 没更新（{fm.get('generated')!r}）—— 同上")
        # 2. 保留：读者写的键原样在
        for k, v in READER_KEYS.items():
            if fm.get(k) != v:
                fail(f"索引页上读者的键丢了：{k}={fm.get(k)!r}（应为 {v!r}）")
        # 4 的正面：读者的正文尾巴不许被覆盖
        if "我的笔记：这是我写的。" not in path.read_text(encoding="utf-8"):
            fail("索引页受管区之外的手写正文被覆盖了")

        # --- 3. 卡片支路：受管键更新、读者的键留住 -----------------------------
        card_fm = {"title": "新标题", "week": WEEK, "tags": ["周报"]}
        card_path = root / "card.md"
        card_path.write_text(
            "---\ntitle: 旧标题\nweek: 2026-W01\ntags:\n- 周报\n" + _reader_block()
            + f"---\n{begin}\n旧\n{end}\n", encoding="utf-8")
        head = wc._head(card_path, card_fm, "新正文", begin, end)   # managed 取默认 = CARD_KEYS
        card_path.write_text(head, encoding="utf-8")
        cfm = wc.existing_frontmatter(card_path)
        if cfm.get("title") != "新标题" or cfm.get("week") != WEEK:
            fail(f"卡片上的受管键没更新：title={cfm.get('title')!r} week={cfm.get('week')!r}")
        for k, v in READER_KEYS.items():
            if cfm.get(k) != v:
                fail(f"卡片上读者的键丢了：{k}={cfm.get(k)!r}（应为 {v!r}）")

        # --- 4. 假阳性方向：读者删了标记 ⇒ 一个字节都不许写，且不许报红 ----------
        weekly2 = _fixture(root / "case2")
        path2 = _seed_index(weekly2, begin, end, reader=False, markers=False)
        before = path2.read_bytes()
        res2 = wc.render_index(weekly2, cfg, generated="2026-09-18")
        if path2.read_bytes() != before:
            fail("读者删掉受管块标记后，索引页仍被改写了")
        if res2["index_fm_mismatch"]:
            fail(f"unmanaged 的页被误报成不一致（假阳性）：{res2['index_fm_mismatch']}")
        if res2["index_fm_checked"] != 0 or not res2["index_fm_ok"]:
            fail(f"unmanaged 那一支的计数不对：checked={res2['index_fm_checked']} "
                 f"ok={res2['index_fm_ok']}")
        if res2["index_pages"].get("unmanaged") != 1:
            fail(f"没被识别成 unmanaged：{res2['index_pages']}")

        # --- 5/6/7. 合订本：计数跟着走、读者的键留住、恒等式成立 -----------------
        weekly3 = _fixture(root / "case3")
        dpath = _seed_digest(weekly3, begin, end, reader=True)
        res3 = wc.render_digest(weekly3, cfg, generated="2026-09-18")

        if res3["digest_fm_mismatch"]:
            fail(f"合订本 frontmatter 与算出来的对不上：{res3['digest_fm_mismatch']}")
        if not res3["digest_fm_ok"] or res3["digest_fm_checked"] != 1:
            fail(f"合订本读回核对没跑起来：checked={res3['digest_fm_checked']} "
                 f"ok={res3['digest_fm_ok']}")
        # 恒等式：印出来的条目数 == 范围内篇数，且没有"没日期所以没印"的
        if res3["digest_identity_bad"] or res3["digest_undated"]:
            fail(f"合订本恒等式不成立：{res3['digest_identity_bad']} "
                 f"undated={res3['digest_undated']}")

        dfm = wc.existing_frontmatter(dpath)
        if dfm.get("papers_in_digest") != 1:
            fail(f"papers_in_digest 没更新（{dfm.get('papers_in_digest')!r}）")
        if dfm.get("library_total") != 1:
            fail(f"library_total 没更新（{dfm.get('library_total')!r}）")
        if dfm.get("generated") != "2026-09-18":
            fail(f"合订本 generated 没更新（{dfm.get('generated')!r}）")
        if dfm.get("days_covered") != [DAY]:
            fail(f"days_covered 没更新（{dfm.get('days_covered')!r}）")
        for k, v in READER_KEYS.items():
            if dfm.get(k) != v:
                fail(f"合订本上读者的键丢了：{k}={dfm.get(k)!r}（应为 {v!r}）")
        dtext = dpath.read_text(encoding="utf-8")
        if "我的笔记：这是我写的。" not in dtext:
            fail("合订本受管区之外的手写正文被覆盖了")
        # 目录与正文**必须以同一个 arxiv_id 打头**——这一份不做锚点，对应关系全靠它。
        if f"### {KEY} · " not in dtext or f"- {KEY} · " not in dtext:
            fail("合订本的目录与正文没有以同一个 arxiv_id 打头（对应关系断了）")

        # --- 8. 合订本的假阳性方向：读者删了标记 ⇒ 不写、不报红 ------------------
        weekly4 = _fixture(root / "case4")
        dpath2 = _seed_digest(weekly4, begin, end, reader=False, markers=False)
        before2 = dpath2.read_bytes()
        res4 = wc.render_digest(weekly4, cfg, generated="2026-09-18")
        if dpath2.read_bytes() != before2:
            fail("读者删掉受管块标记后，合订本仍被改写了")
        if res4["digest_fm_mismatch"]:
            fail(f"unmanaged 的合订本被误报成不一致（假阳性）：{res4['digest_fm_mismatch']}")
        if res4["digest_fm_checked"] != 0 or not res4["digest_fm_ok"]:
            fail(f"合订本 unmanaged 那一支的计数不对：{res4}")
        if res4["digest_pages"].get("unmanaged") != 1:
            fail(f"合订本没被识别成 unmanaged：{res4['digest_pages']}")

        # --- 9. 负向：证明上面那套读回核对**真的会红** ---------------------------
        # 09-18 验索引页时是手工改源码、验完再还原（一次性、容易漏还原）。
        # 这里直接在进程内**用错集合写一次**，再拿 render_digest 的同一份清单
        # `DIGEST_TRACKED` 逐键比 —— 不这么测，"这套检查能抓到传错键集"就是
        # 一句没量过的话，而**没量过的检查正是这个脚本存在的理由**。
        weekly5 = _fixture(root / "case5")
        dpath3 = _seed_digest(weekly5, begin, end, reader=False)
        papers5, _s5, _a5 = wc.load_inputs(weekly5)
        dfm3, dbody3, _i3 = wc.build_digest(papers5, WEEK, "2026-09-18",
                                            library_total=len(papers5))
        # ↓ 故意传 CARD_KEYS（就是 2026-09-18 那个事故的姿势）
        dpath3.write_text(wc._head(dpath3, dfm3, dbody3, begin, end,
                                   managed=wc.CARD_KEYS), encoding="utf-8")
        written3 = wc.existing_frontmatter(dpath3)
        gap = [k for k in wc.DIGEST_TRACKED if written3.get(k) != dfm3.get(k)]
        if len(gap) != len(wc.DIGEST_TRACKED):
            fail(f"用 CARD_KEYS 写合订本之后，逐键比只报出 {gap} —— "
                 f"应报出全部 {list(wc.DIGEST_TRACKED)}；这套检查是假绿的")

    print(json.dumps({
        "index_counts_track_this_run": True,      # 1 正向
        "reader_keys_preserved_index": True,      # 2 保留
        "reader_keys_preserved_card": True,
        "managed_keys_refreshed_card": True,      # 3 反向
        "unmanaged_page_untouched": True,         # 4 不误报
        "index_fm_ok": True,
        "digest_counts_track_this_run": True,     # 5 合订本正向
        "digest_reader_keys_preserved": True,     # 6 合订本保留
        "digest_identity_ok": True,               # 7 合订本恒等式
        "digest_unmanaged_untouched": True,       # 8 合订本不误报
        "wrong_keyset_detected": True,            # 9 **负向**：传错键集必须报全
        "real_store_untouched": str(real),        # 本脚本只写 tempfile 目录
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
