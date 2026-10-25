"""把周报库导出成**可交付的包**，供上传到云端仓库。

⚠️ **2026-09-19 用户拍：这一步休眠。**
周报线的投放方式改成了「**发合订本 md 到微信群**」⇒ **不再有"上传到云端仓库"这个动作**，
而本脚本的落点清单是显式的（卡片 + 周索引 + 累积语料 + 公告），**本来就不含
``digest.md``**（见下面「落点」那一段）。靠山 ``weekly_sections.py`` / 卡片渲染同批休眠。
✅ **代码保留**——哪天恢复"发站/发仓库"，它接管的仍是同一件事；
但**恢复它之前先想清楚**：它导的是卡片与周索引，那两类现在没有 content 来源（``concern``）。

为什么单独一个脚本、而不是手工拷
-------------------------------

交付包里有两样东西**必须每次现算**，手工拷一定会漏：

1. **累积语料** `papers.jsonl` —— 逐周变大，忘了拷就是发了一份过期的。
2. **按周累积的公告类型** —— 库里的真相是 ``<周>/<日期>.announce.jsonl``（**一天一个文件**），
   而交付侧要的是 ``<周>/announce.jsonl``（**一周一个文件**，多一列 ``date``）。
   这个合并**不能靠 glob 拼**：拼错了不报错，只是少几百行。

⚠️ 它**刻意不进** `resume.py` 的 `POST_STEPS`：那条铁律管的是"**从
``data/papers.jsonl`` 派生**的产物"，而这里是**另一个库的导出**，
既不是派生链的一部分，也不该在每次"恢复长任务"时自动跑（交付是人工动作）。

判据（都在汇总行里，铁律 7）
------------------------------

- ``papers_written == papers_read``；
- ``announce_lines_written == announce_lines_read``（合并不能丢行）；
- ``cards_read == cards_verified``：每片卡片拷完**读回来算 sha** 比过；
- **``unmatched_keys`` 必须报出来**：交付包里 join 不上的 key 数。
  ⚠️ 这不是"顺带看一眼"的数——它的默认值必须是**报警**而不是 0，
  因为"沉默的缺口看起来和全都匹配上了"一模一样。
- 另外三个"沉默缺口"的不同形状，各自单列：``cards_mismatched``（拷了但字节不对）、
  ``cards_stale``（交付侧多出来的卡片）、``index_mismatched``。

交付出去的是什么
------------------

- ``papers.jsonl`` —— 元数据 + 摘要（arXiv 的元数据是 CC0）。
- ``<周>/concern/<id>.md`` —— 每篇一片卡片：元数据 + 摘要 + **作者自述的展望/缺口**。
  ⚠️ 那一段是**引文**，不是全文。实测 44 篇引文的中位长度 **1549 字符**、最长
  **5482**、全周合计 **7.8 万字符**。**全文层已于 2026-09-18 关闭**
  （arXiv 再分发授权实测，见 ``MEASUREMENTS.md`` §四c #24）——两者不冲突：
  关掉的是"整篇正文"，这里是"一段带出处的引文"。
  **改动本脚本时不要顺手把正文或 PDF 加进来。**
- ``<周>/report.md`` —— 周索引，含逐篇去向与本版覆盖声明。

落点（``--out``，默认 ``D:\\大学编程学习\\arXiv周报\\交付``）
-------------------------------------------------------------

.. code-block:: text

   <out>/
     papers.jsonl                 累积语料（copy，逐字节）
     <周>/
       announce.jsonl             {"key":..., "announce_type":..., "date":"YYYY-MM-DD"}
       report.md                  周索引
       concern/<arxiv_id>.md      每篇一片卡片

⚠️ 本脚本**只写不删**：它不会去清理交付目录里已经不需要的旧文件。
交付目录如果被手工改过，先在那边自己确认，**不要**让它顺手 rm。

**Token 纪律**：终端只打一行聚合 JSON。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllib import ROOT  # noqa: E402
from rllib.atomic import write_atomic  # noqa: E402
from rllib.console import use_utf8_console  # noqa: E402

PAPERS_NAME = "papers.jsonl"
ANNOUNCE_NAME = "announce.jsonl"
CARD_DIR = "concern"
INDEX_NAME = "report.md"

#: 周报库的真相源。**这是全库第二处落点定义**，与 ``rllib/cli.py::_WEEKLY_PATHS["data"]``
#: 同值——本脚本不调 ``_setup(..., "weekly")``，拿不到 profile 改写后的 cfg，只能自己拼。
#: ⚠️ **改一处就要改另一处**：两边不一致不会报错，只会**静默地读出旧位置**。
#: 待办是把本脚本改成跟随 ``_apply_profile``，登记在 ``D:\大学编程学习\arXiv周报\README.md``。
WEEKLY_SRC = Path(r"D:\大学编程学习\arXiv周报\数据")

#: 交付目录（``--out`` 的默认值）。2026-09-20 起**程序与数据同住 ``arXiv周报\``**，
#: 交付物在它的 ``交付\`` 子目录（独立 git 仓库、`README.md` **必须入库**）。
#: ⚠️ 本脚本**不调** ``_setup(..., "weekly")``，拿不到 profile 改写后的 cfg，
#: 所以 ``WEEKLY_SRC`` 是**全库第二处落点定义**——改一处就要改另一处，
#: 不一致时**不报错、只是静默地读旧位置**。
WEEKLY_DST = r"D:\大学编程学习\arXiv周报\交付"


def _sync_dir(src_dir: Path, dst_dir: Path) -> dict:
    """把 ``<src>/concern/*.md`` 与 ``<src>/report.md`` 同步到交付目录。

    **逐字节 copy + 逐字节校验**，和 ``papers.jsonl`` 同理：交付包里的卡片
    必须与源库**完全一致**，否则"导出的口径"就成了第二个真相源。

    ⚠️ 校验要拿**源**去比（读回来算 sha），不能只信 ``copyfile`` 没抛异常——
    "没报错"和"内容对"是两件事，本库栽在这上面的次数不必再数。

    ⚠️ 本函数**只写不删**（与整个脚本一致），但**必须把目标侧多出来的报回去**：
    源库里掉出范围的卡片不会被删掉，于是交付包里会留下一片**指向不存在论文的卡片**
    —— 而"多出来的旧文件"和"本来就该有"在目录列表里长得一模一样。
    """
    mismatched: list[str] = []
    stale: list[str] = []

    src_cards = src_dir / CARD_DIR
    src_files = sorted(src_cards.glob("*.md")) if src_cards.is_dir() else []
    dst_cards = dst_dir / CARD_DIR
    if src_files:
        dst_cards.mkdir(parents=True, exist_ok=True)

    verified = 0
    for f in src_files:
        dst = dst_cards / f.name
        shutil.copyfile(f, dst)
        # ⚠️ 数**校验通过**的片数，不是"循环跑了几轮"。否则
        # ``cards_read == cards_written`` 恒真——一个永远绿的检查比没有检查更糟
        # （CLAUDE.md「不变量检查」）。这里 sha 不一致时那个数会真的掉下来。
        if hashlib.sha256(f.read_bytes()).hexdigest() == hashlib.sha256(dst.read_bytes()).hexdigest():
            verified += 1
        else:
            mismatched.append(f.name)

    src_names = {f.name for f in src_files}
    if dst_cards.is_dir():
        stale = sorted(p.name for p in dst_cards.glob("*.md") if p.name not in src_names)

    index_src = src_dir / INDEX_NAME
    index_ok = True
    if index_src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / INDEX_NAME
        shutil.copyfile(index_src, dst)
        index_ok = (hashlib.sha256(index_src.read_bytes()).hexdigest()
                    == hashlib.sha256(dst.read_bytes()).hexdigest())

    # ⚠️ ``cards`` 与 ``index_written`` **必须分开数**。先前把两者都塞进一个
    # ``copied`` 里，于是 ``cards_read==cards_written`` 这条恒等式报 73≠74 ——
    # **字段名说"卡片"，数的是"卡片 + 周索引"**。恒等式本身是对的，
    # 它抓的正是这种"名字和数不是一回事"（本库栽过好几回）。
    return {"cards": len(src_files), "cards_verified": verified,
            "index_written": 1 if index_src.exists() else 0,
            "mismatched": mismatched, "stale_cards": stale, "index_ok": index_ok}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not ln.strip():
            continue
        try:
            rows.append(json.loads(ln))
        except ValueError:
            raise SystemExit(f"{path} 第 {i} 行不是合法 JSON —— 拒绝在损坏的源上导出") from None
    return rows


def _accumulate_announce(week_dir: Path) -> tuple[list[dict], int]:
    """``<周>/<日期>.announce.jsonl`` × N → 一条一周的列表（加 ``date`` 列）。

    ⚠️ 日期取自**文件名**而不是文件内容——旁路文件的格式里没有日期列，
    而文件名是 ``2026-09-17.announce.jsonl``，那是这个信息的唯一载体。
    取法写死成"前 10 个字符"，因为我们自己就是那样命名的；对不上就**炸**，
    不要回退成"随便取"（那会让一批行带错日期且不报错）。
    """
    out: list[dict] = []
    for path in sorted(week_dir.glob("*.announce.jsonl")):
        day = path.name[:10]
        if len(day) != 10 or day[4] != "-" or day[7] != "-":
            raise SystemExit(f"{path.name} 不符合 <YYYY-MM-DD>.announce.jsonl 的命名，拒绝猜日期")
        for row in _read_jsonl(path):
            out.append({"key": row["key"], "announce_type": row["announce_type"], "date": day})
    return out, len(out)


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser(description="把周报库导出成交付包（只输出聚合摘要）。")
    ap.add_argument("--out", default=WEEKLY_DST,
                    help=f"交付目录（默认 {WEEKLY_DST}）")
    args = ap.parse_args()

    # ⚠️ **这里与 `rllib/cli.py::_WEEKLY_PATHS` 是第二处落点定义。**
    # 本脚本从不调 `_setup(..., "weekly")`，所以拿不到 profile 改写后的 cfg，
    # 只能自己拼 —— 2026-09-20 之前这里写的是 `ROOT / "data" / "weekly"`。
    # 而 `cli.py::_apply_profile` 的 docstring 曾宣称「落点只有这一处定义」，
    # 那句话是假的，本文件就是反例。两处都指向同一个常量，别再各写各的。
    src = WEEKLY_SRC
    out = Path(args.out).resolve()

    papers_src = src / PAPERS_NAME
    if not papers_src.exists():
        raise SystemExit(f"{papers_src} 不存在 —— 周报库还没有数据，没有可交付的东西")

    # 语料：**逐字节 copy**，不重解析。重解析会把"导出的口径"变成第二个真相源。
    out.mkdir(parents=True, exist_ok=True)
    papers_dst = out / PAPERS_NAME
    shutil.copyfile(papers_src, papers_dst)

    # ⚠️ 恒等式必须拿**源**去比，不能拿目标跟它自己比——后者恒真，
    # 是一个"永远绿"的检查（本库栽过不止一次）。
    src_sha = hashlib.sha256(papers_src.read_bytes()).hexdigest()
    dst_sha = hashlib.sha256(papers_dst.read_bytes()).hexdigest()

    papers = _read_jsonl(papers_dst)
    paper_keys = {p.get("arxiv_id") or p.get("title_norm") for p in papers}

    weeks = 0
    announce_read = 0
    announce_written = 0
    cards_read = 0
    cards_verified = 0
    index_written = 0
    unmatched = 0
    mismatched: list[str] = []
    stale: list[str] = []
    index_bad: list[str] = []
    types: Counter[str] = Counter()

    for week_dir in sorted(p for p in src.glob("*") if p.is_dir() and not p.name.startswith(".")):
        rows, n_read = _accumulate_announce(week_dir)
        # ⚠️ 公告为空**不能 `continue`**：卡片与公告是同一周目录下的两样东西，
        # 一个没有不代表另一个没有。早先这里是 `if not rows: continue`，
        # 那会让"某一周只有卡片没有公告"时**静默地一片卡片都不导**。
        if rows:
            weeks += 1
            announce_read += n_read
            # 按 key 排序落盘：同一份输入永远产出同一份输出，diff 才有意义。
            rows.sort(key=lambda r: (r["date"], r["key"]))
            write_atomic(
                out / week_dir.name / ANNOUNCE_NAME,
                "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                        for r in rows).encode("utf-8"),
            )
            announce_written += len(rows)
            types.update(r["announce_type"] for r in rows)
            unmatched += sum(1 for r in rows if r["key"] not in paper_keys)

        sync = _sync_dir(week_dir, out / week_dir.name)
        cards_read += sync["cards"]
        cards_verified += sync["cards_verified"]
        index_written += sync["index_written"]
        if not sync["index_ok"]:
            index_bad.append(week_dir.name)
        mismatched += [f"{week_dir.name}/{n}" for n in sync["mismatched"]]
        stale += [f"{week_dir.name}/{n}" for n in sync["stale_cards"]]

    print(json.dumps({
        "out": str(out),
        "papers_read": len(papers),
        "weeks": weeks,
        "announce_read": announce_read,
        "announce_written": announce_written,
        "announce_by_type": dict(sorted(types.items())),
        # ⚠️ 这个数**必须是可视的**：>0 说明有公告行接不到任何论文上。
        "unmatched_keys": unmatched,
        "cards_read": cards_read,
        "cards_verified": cards_verified,
        "index_written": index_written,
        # ⚠️ 三个都是「沉默的缺口」的不同形状，必须各自可见：
        #   mismatched —— 拷过去了但字节对不上（copyfile 不会因此报错）
        #   stale      —— 交付侧多出来的卡片（源库里那篇已经掉出范围，但只写不删）
        #   index_bad  —— 周索引拷过去后字节不一致
        "cards_mismatched": mismatched,
        "cards_stale": stale,
        "index_mismatched": index_bad,
        "identity_papers_ok": src_sha == dst_sha,
        "identity_announce_ok": announce_read == announce_written,
        "identity_cards_ok": cards_read == cards_verified,
    }, ensure_ascii=False))

    # 对不上就非零退出——交付包里带着对不上的行或对不上的卡片，比不交更糟。
    return 1 if (unmatched or mismatched or index_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
