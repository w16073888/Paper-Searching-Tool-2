"""arXiv 日更：把三个分类的当日公告吸进**周报库**。

为什么是日更抓、而不是周更抓
----------------------------

``https://arxiv.org/rss/<分类>`` **实质上是"当天那一批公告"**，不是一条往回看很多天的
深窗口。实测 **2026-09-18**：feed 去重 **548** 篇，与前一天（09-17）入库的 **537** 篇
**只重叠 16 篇**（≈**3%**）——**日与日几乎完全换一批**。

⚠️ **本条 2026-09-18 更正过措辞。** 原文写的是"**滚动窗口**、在往前滚"，依据是
09-15 的 cs.LG **541** 条 → 09-17 的 **325** 条。那两个数是实测的、没错，
但"窗口在往前滚"是**从两个大小不同的样本推出来的形状**；而实测重复率给出的形状是
"**每天一批、几乎不重叠**"。⇒ **结论没变，而且更强了**：按周抓一次，一周里六天的
公告**全部**看不到；而且它不报任何错——表现为"这周只有三百来篇"，与"这周真的
只有三百来篇"分不开。所以**抓取必须日更；周更只负责发布**。

⚠️ **代价也随之明确**：每个**工作日**约 **550 篇新入库**（09-17 = 537、09-18 = 548），
而许可只能逐条问 OAI-PMH ⇒ **许可段 ≈ 550 篇 × 4 秒 ≈ 36 分钟/工作日**。
⚠️ **这是这条线在**有公告的那些天**都要付的账，不是"首次运行才这么慢"**——首跑那天
会误以为"以后就快了"，因为首跑的目标集恰好也是整个存量库、看起来是同一件事。
⚠️ **别把它读成"每天"**：feed 自带 ``<skipDays>Sunday, Saturday</skipDays>``，
**一周只有约 5 天出料**（2026-09-19 周六实测：三个 feed 各 **0** 条）。
按**日历天**摊是 ≈380 篇/天、≈25 分钟/天——**按 7 天外推会高估 40%**。
数字见 `MEASUREMENTS.md` **#28**（口径已订正）与 **#29**。

落点（完全隔离于 2025 主库，**绝不并进 ``data/papers.jsonl``**）
----------------------------------------------------------------

.. code-block:: text

   <周报库根>/
     papers.jsonl                 唯一真相源（append-only）
     <ISO周>/                     按周分文件夹（用户 2026-09-17 定）
       <日期>.jsonl               当天解析出的条目（对账 + 可重放）
       <日期>.announce.jsonl      当天每篇的公告类型（旁路文件，见下）
       <日期>.done                这一跑**走到头了**的标记（内容 = 整行汇总 JSON）
       raw/<日期>-<分类>.xml      当天原始 RSS（**重解析不重抓**）
       raw-archive/<真实运行日>-<HHMM>-superseded/
                                  被同日重跑覆盖掉的那份 raw 的**自动**副本

**为什么公告类型要单落一份旁路文件**（用户 2026-09-17 选 A）：``announce_type``
（``new`` / ``cross`` / ``replace`` / ``replace-cross``）回答"这一篇是不是本周新工作"。
它是 RSS 的扩展字段，而 ``CanonicalPaper`` 是**冻结契约**（铁律 5、``arxiv_rss.py``
开头明写"不碰 announce_type 之外的扩展字段"），所以它**没有**进 jsonl——只在
``raw/*.xml`` 里。把 raw 当唯一来源意味着每次渲染卡片都要重解析一遍 XML；落一份
``key -> announce_type`` 的旁路则让卡片渲染零解析。
**代价为零**（解析时本来就有这个值），**改动面为零**（不碰契约、不碰主库）。
实测动因：2026-09-17 那 537 条里混着 ``year=2023`` 的老预印本（被 replace/cross 推上来的），
没有它就没法在卡片上区分"本周新论文"与"三年前的老论文又被公告了一次"。

请求预算（两段，**打的是两台不同的主机，限流分开算**）
------------------------------------------------------

* **公告段**：三个分类各 1 请求 = **3 请求/天**打 ``arxiv.org``
  （``HOST_DELAYS["arxiv.org"] = 4.0`` → 约 12 秒）。
  ⚠️ **这 3 个请求每次运行都要真发**，不吃 `cache/` 的 URL 缓存——
  理由（以及 2026-09-18 那次静默事故）写在 `main` 里 `refresh=True` 那一段注释。
* **许可段**：RSS 里**没有**许可，只能从 OAI-PMH 逐条问（``GetRecord``，一篇一请求，
  打 ``oaipmh.arxiv.org``）。所以本脚本现在**不是 12 秒就结束**，而是
  "12 秒 + 4 秒 × 需要补许可的篇数"。稳态下这个篇数约等于当天新入库的量；
  **首次运行或长时间没跑之后，它会等于整个存量库**——那正是
  ``scripts/backfill_license.py`` 存在的意义，两处调的是**同一个函数**
  （:func:`rllib.weekly_license.fill_missing`）。
  实测（2026-09-18）一篇一请求的响应是 2,790 字节，网络耗时主要是那 4 秒间隔。
  想要快的冒烟测试用 ``--license-limit``。

``Fetcher`` 自带限速、熔断与磁盘余量刹车，所以退出码与主库长任务一致：
**3 = 被限流/封锁，4 = 磁盘余量不足**——都是有意的中止，不是崩溃。
许可段被打断时已取到的许可不在盘上但**在 ``cache/`` 里**，重跑是零网络命中。

用法
----

.. code-block:: text

   python scripts/daily_arxiv_fetch.py                 # 日更（正式用法）
   python scripts/daily_arxiv_fetch.py --limit 5       # 开发期冒烟（铁律 4.3）
   python scripts/daily_arxiv_fetch.py --date 2026-09-17
   python scripts/daily_arxiv_fetch.py --no-filter      # 只入库，不重跑 RL 打标

**Token 纪律**：终端只打一行聚合 JSON，绝不回显论文标题或摘要。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllib import ROOT  # noqa: E402
from rllib import feedwatch  # noqa: E402
from rllib import weekly_license  # noqa: E402
from rllib.adapters import arxiv_rss  # noqa: E402
from rllib.atomic import write_atomic  # noqa: E402
from rllib.cli import _setup  # noqa: E402
from rllib.console import use_utf8_console  # noqa: E402
from rllib.diskguard import DiskLow  # noqa: E402
from rllib.http import HostBlocked  # noqa: E402
from rllib.schema import CanonicalPaper  # noqa: E402
from rllib.store import PaperStore  # noqa: E402

#: 本脚本在**合并落盘**时声明的字段所有权（铁律 3）。
#:
#: 刻意只认领这几个，而不是把整条记录都认领掉——**多认领会静默抹掉别的任务的成果**：
#:
#: * ``topics`` —— ``store._merge`` 对它是**并集**，而并集只存在于内存里；
#:   不认领，第二天在 cs.AI 里又看见同一篇时新加的分类就被丢掉了。
#: * ``provenance`` —— 同理，且它走的是只增不减的并集合并。
#: * ``last_updated`` —— ``_merge`` 判定"有变化"时在内存里盖当天日期，不认领就写不进去。
#:
#: **``abstract`` / ``authors`` / ``arxiv_id`` / ``url`` / ``pdf_url`` 一个都不认领**：
#: ``_merge`` 对它们是"空值才填"（``_FILL_IF_EMPTY``），而本脚本看见的每一篇
#: 要么是新的（走 ``records[pid] = mine`` 这条整条落盘的路径），要么磁盘上早就有值，
#: 所以不认领不会丢东西。认领它们反而会让本脚本有权覆盖未来某个
#: ``weekly 全文/摘要`` 长任务的成果——那正是 ``owned_fields`` 要防的事。
#:
#: ⚠️ **``license`` 是唯一的例外，必须认领**，而理由正是上面那条的反面：
#: 许可**是**在"论文已经在库里"之后才补上去的（一篇论文入库那天可能还没问到，
#: 下一轮再看见它时才补到）。这种"补进了内存、磁盘上还是空"的值，不认领就会在
#: 合并落盘时被磁盘上的空值顶掉——**填了等于没填，而且不报错**。
#: 认领它是安全的：``store._would_downgrade`` 挡住了反方向（不许把已知抹成未知），
#: 而许可**不可撤销**（见 ``MEASUREMENTS.md`` §四c #24），所以不存在"旧值该让位"的情形。
OWNED_FIELDS = ("topics", "provenance", "last_updated", "license")


def _is_cross_view(t: str) -> bool:
    """这个公告类型是不是**交叉列出视角**（而非论文主分类的视角）。

    arXiv 的 ``announce_type`` 是**按分类各算一次**的：一篇论文只在它的主分类里被
    宣布一次（``new`` / ``replace``），出现在其他分类 feed 里的是**交叉列出**，
    同一篇那一条会标成 ``cross`` / ``replace-cross``。所以同一篇在两个 feed 里
    类型不同是**正常的**，且可判：两者去掉交叉视角后必然相等。

    实测（2026-09-17，三 feed 全量 537 篇）：105 处"冲突"**全部**是
    ``{cross,new}`` 或 ``{replace,replace-cross}`` 同侧同族配对，**没有一例**
    形如 ``{new,replace}``。⚠️ 这条实测是本函数存在的理由——最初这里写的是
    「类型不同 = 只可能是键撞了」，那句话是错的，而且据此用"先到先得"取值，
    会把主分类是 cs.AI、却先被 cs.LG 看见的论文**静默地**灌成 ``cross``。
    """
    return t == "cross" or t.endswith("-cross")


def _week_key(d: date) -> str:
    """``2026-09-17`` → ``2026-W38``。用 ISO 周（周一为首日），跨年不会重号。"""
    iso = d.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


#: 基线最多往回找几份 raw。**只界定"多陈的基线还算数"**，不是性能参数：
#: 回退**在第一个有内容的文件就停**，而只有周末/节假日的空 feed 才算"没内容"
#: （2026-09-19 实测 892 B / 0 条）。实测现实最坏（周一：两个周末 + 一个内容）
#: 三分类合计 **0.43 s**；7 份 1.37 MB 的 cs.LG 全解析也才 3.72 s，相对许可段的
#: 分钟级预算可忽略。
#: ⚠️ 不要"优化"成正则扫字节——实测 1.06 s/次，比 parse_feed 的 0.53 s **更慢**。
_BASELINE_LOOKBACK = 7


def _recent_raws(weekly_dir: Path, day: str, category: str,
                 *, limit: int | None = None) -> list[tuple[str, Path]]:
    """同一分类、**日期严格早于 ``day``** 的 raw，按日期**倒序**。

    ⚠️ **刻意不取"目录里上一个文件"**，而是按文件名里的日期筛。同一天重跑时今天的
    文件会被整体重写，把它算进来判据就变成"跟自己比"——**恒真，而恒真的检查比没有
    更糟**（2026-09-18 §12 那次 `identity_papers_ok` 写成"跟它自己比行数"就是同一个形状）。
    同一天重跑在本库是**常规操作**（09-22 跑三次、09-21 跑两次），所以 `d < day` 不能省。

    ⚠️ **跨周枚举**（``*/raw/*``），不是只在本周目录里找。2026-09-28（周一、ISO
    ``2026-W40``）的 `raw/` 目录**还不存在**，只找本周必然返回 ``None`` ⇒
    ``raw_unchanged == []`` ⇒ ``raw_stale = False``：**任何周一都结构性地看不见
    逐字节重发**。跨周后基线回到 ``W39/raw/2026-09-25-*.xml``（周五），检查复活。
    ⚠️ 这**不止影响周一**，而是"**任何本周目录里没有更早 raw 的时刻**"，
    也包括回填本周最早的日期。

    ⚠️ **模式里的 ``raw`` 那一段不能省。** 若有人"简化"成 ``**/*-<分类>.xml``，
    ``raw-archive/2026-09-22-1201-killed/`` 里那份**停摆的** feed 就会变成基线候选。
    """
    suffix = f"-{category}.xml"
    found: list[tuple[str, Path]] = []
    for p in weekly_dir.glob(f"*/raw/*{suffix}"):
        d = p.name[: -len(suffix)]
        if len(d) == 10 and d < day:
            found.append((d, p))
    found.sort(key=lambda t: t[0], reverse=True)
    return found[:limit] if limit else found


def _prev_raw_path(weekly_dir: Path, day: str, category: str) -> Path | None:
    """**日期严格早于 ``day``** 的最近那份 raw；没有就返回 ``None``（首跑）。

    薄封装：与 :func:`_load_baseline` **共用** :func:`_recent_raws`，
    免得两份"找上一份 raw"的近似实现各自漂移。
    """
    recent = _recent_raws(weekly_dir, day, category, limit=1)
    return recent[0][1] if recent else None


def _load_baseline(weekly_dir: Path, day: str,
                   category: str) -> feedwatch.Baseline | None:
    """找一份**有内容**的 raw 当基线；一本都没有就返回 ``None``。

    沿 :func:`_recent_raws` 往回走，取**第一个有非空 ID 集**的。必须**跨过**周末/
    节假日的空 feed（2026-09-19 实测 892 B / 0 条），而且必须**枚举候选文件**、
    不能按算术"昨天、前天"走——**2026-09-20（周日）连 raw 文件都没有**。

    读回用 ``read_bytes().decode("utf-8")``，是 ``main()`` 里 ``payload = xml.encode("utf-8")``
    的**精确逆**。
    ⚠️ **不要用不带编码的 ``read_text()``**——那会落到本机 locale（实测 **cp936**）；
    今天"能用"只是因为 raw 是**纯 ASCII**（三份大文件 ``nonascii=0``）。将来一个重音
    作者名或 CJK 标题就会抛错或**静默 mojibake**，而 mojibake 是**隐形**的：
    arXiv ID 是 ASCII，最大值照样能从一个垃圾文本里算出来。
    （``ET.fromstring`` 吃带编码声明的 ``str`` 没问题——stdlib 不是 lxml。）

    基线自带 ``build_date`` 与"它是不是真等于文件名那天"，见
    :func:`rllib.feedwatch.build_date_matches`——因为**文件名在这个库里撒过谎**：
    2026-09-21 的 09:02 那趟把 ``feed_build: Sun, 20 Sep 2026`` 写进了名为
    ``2026-09-21-*.xml`` 的文件。
    """
    for d, path in _recent_raws(weekly_dir, day, category, limit=_BASELINE_LOOKBACK):
        try:
            res = arxiv_rss.parse_feed(path.read_bytes().decode("utf-8"), category=category)
        except (OSError, ET.ParseError):
            continue          # 读不动/解析不了就换上一份：基线是佐证，不配中止当天的抓取
        sample = feedwatch.FeedSample.of(res.papers, res.announce_by_key)
        if not sample.ids:
            continue          # 周末那份空 feed，继续往回走
        return feedwatch.Baseline(
            day=d, name=path.name, sample=sample, build_date=res.build_date,
            build_date_matches_name=feedwatch.build_date_matches(res.build_date, d),
        )
    return None


def _archive_superseded(dst: Path, payload: bytes, week_dir: Path,
                        run_date: str, stamp: str) -> Path | None:
    """``dst`` 要被覆盖前，把**旧字节**留一份档；没覆盖或字节一样则返回 ``None``。

    为什么需要它：``raw/<日期>-<分类>.xml`` **同日同路径整体覆盖**，而它是
    ``announce_type``（新论文 / 修订版）的**唯一底档**，也是"重解析不重抓"的底气。
    2026-09-21 实测损失：13:43 的补跑把 09:02 抓到的**周日那份构建**覆盖掉了，
    那批字节是「arXiv 周日也重建 feed」这条结论的唯一证据，现在**没有了**。

    ⚠️ **它在 2026-09-22 那次事故上不会触发**，别引错案子：那天三跑的 raw
    **逐字节相同**（cs.AI 三跑都是 ``f6639c03…``），``old == payload`` ⇒ 正确地
    **不**归档。它真正防的是 **09-21 那种"两次抓到的字节不同"**。

    目录名用**真实运行日**（``run_date``）而不是 ``--date``，与手工的
    ``2026-09-22-1201-killed/`` 同一 ``<日期>-<HHMM>-<场合>`` 约定；否则一次回填
    会产出"09-21 的目录、09-25 建的"这种自相矛盾的名字。归档文件**保留原名**
    （``2026-09-21-cs.LG.xml``）——它自己已经说清是哪天的 raw。

    清单**增量写**（读现有 → 追加一行 → 整份原子回写），不攒到循环结束一次写：
    本库最激励这件事的两次跑**都是被杀死的**，攒在内存里意味着被杀后留下一个
    **装满了文件、却没有清单**的"证据"目录。用**裸文件名**形（``<sha256> *名字``）
    ⇒ ``sha256sum -c SHA256SUMS.txt`` 要**在这个目录里**跑。现有两份手工档案格式
    不一致（09-21 那份是全路径、killed 那份是裸名），这份选**较新且自包含**的那种。
    """
    try:
        old = dst.read_bytes()
    except FileNotFoundError:
        return None
    if old == payload:
        # 比字节而不是比 sha256：调用方手里那个 digest 是**新** payload 的，
        # 要比摘要仍然得把旧文件读进来再哈希一遍（1.37 MB ≈ 5 ms），白花。
        return None
    arch_dir = week_dir / "raw-archive" / f"{run_date}-{stamp}-superseded"
    out = arch_dir / dst.name
    write_atomic(out, old)
    sums = arch_dir / "SHA256SUMS.txt"
    try:
        prev = sums.read_bytes().decode("utf-8")
    except FileNotFoundError:
        prev = ""
    line = f"{hashlib.sha256(old).hexdigest()} *{out.name}\n"
    write_atomic(sums, (prev + line).encode("utf-8"))
    return out


def _run_filter() -> dict[str, object]:
    """入库之后重跑 RL 打标。**零网络开销**，但必须跑。

    ``rl.is_rl`` 是筛选面（facet）不是闸门（CLAUDE.md 检索工具 §1），可它得**存在**
    才谈得上当筛选面——不打标的话周报库每一篇的 ``is_rl`` 都是默认的 False，
    而那是"没算过"和"算过为假"共用的一个值。这正是本库反复栽的那类静默失败。
    """
    argv = [sys.executable, "-m", "rllib", "--store", "weekly", "filter", "--rl"]
    proc = subprocess.run(
        argv, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
    )
    out: dict[str, object] = {"filter_rc": proc.returncode}
    if proc.returncode == 0:
        # `filter` 打的是 **Python repr 不是 JSON**（单引号），所以 `json.loads` 必然失败。
        # 这里两种都试：repr 用 `literal_eval`（它是字面量，不执行代码）。
        # ⚠️ 曾经的降级分支只留**末 300 字符**，而那正好把 `total`/`rl_before`/`rl_after`
        # 那一串截掉、只剩 subfields——**汇总行看起来还在，最重要的数字却没了**，
        # 正是本库反复栽的那类"不报错的静默失真"。降级一律改成**头尾都留**。
        text = (proc.stdout or "").strip()
        lines = [ln for ln in text.splitlines() if ln.strip()]
        parsed: object | None = None
        for parse in (json.loads, ast.literal_eval):
            try:
                parsed = parse(lines[-1])
                break
            except (ValueError, SyntaxError, IndexError):
                continue
        if isinstance(parsed, dict):
            out["rl"] = parsed
        else:
            out["filter_out"] = text if len(text) <= 400 else f"{text[:200]} … {text[-200:]}"
    else:
        merged = ((proc.stderr or "") + (proc.stdout or "")).strip()
        out["filter_err"] = merged if len(merged) <= 400 else f"{merged[:200]} … {merged[-200:]}"
    return out


def main() -> int:
    use_utf8_console()
    t0 = time.monotonic()
    #: **真实运行时刻**，与 ``--date`` 是两回事：那个说"这批算哪天的账"，
    #: 这个说"这一跑是什么时候真的跑的"。留档目录名要用后者——否则一次回填
    #: 会产出"09-21 的目录、09-25 建的"这种自相矛盾的名字。
    run_at = datetime.now()

    ap = argparse.ArgumentParser(
        description="arXiv 每日公告 → 周报库。只输出聚合摘要。",
    )
    ap.add_argument("--date", default=None,
                    help="日期（默认今天）。只影响归档目录与快照文件名，不改变抓取内容")
    ap.add_argument("--limit", type=int, default=0,
                    help="每个分类最多入库 N 条（开发期冒烟用；0 = 全量）。"
                         "注意 raw/*.xml 始终是完整的，公告类型统计也始终是全订阅量的")
    ap.add_argument("--refresh", action="store_true",
                    help="绕过 HTTP 缓存重新拉取（默认命中 cache/ 就不重抓）")
    ap.add_argument("--license-limit", type=int, default=0,
                    help="许可段这一轮最多问几篇（0 = 全量）。⚠️ 与 --limit 是两回事："
                         "那个管入库篇数，这个管补许可的篇数。首次运行或久未运行时"
                         "存量库会整个进目标集，冒烟测试用这个把它压住")
    ap.add_argument("--no-filter", action="store_true", help="跳过后面的 RL 打标")
    args = ap.parse_args()

    day = date.fromisoformat(args.date) if args.date else date.today()
    today = day.isoformat()
    week = _week_key(day)
    stamp = run_at.strftime("%H%M")
    run_date = run_at.strftime("%Y-%m-%d")

    # 本脚本走 `cli._apply_profile`，落点跟着它走。⚠️ **但落点定义不止那一处**：
    # `scripts/export_weekly.py` 也自己拼了一份（`WEEKLY_SRC`）——原注释写「只有一处定义，
    # 脚本与 CLI 因此不可能各说各话」，**那句是假的**，2026-09-20 已把两处都订正。
    # ⇒ 改周报落点时 **cli.py::_WEEKLY_PATHS 与 export_weekly.py::WEEKLY_SRC 要一起改**。
    cfg, fetcher = _setup(False, "weekly")
    weekly_dir = ROOT / cfg["paths"]["data"]          # 周报库搬出本库后是绝对路径，见 cli.py
    week_dir = weekly_dir / week
    raw_dir = week_dir / "raw"

    per_category: dict[str, int] = {}
    announce: Counter[str] = Counter()
    #: 逐条的公告类型（``key -> new/cross/replace/replace-cross``）。与上面的 ``announce``
    #: 是两个用途：那个进汇总行回答"今天各占多少"，这个落旁路文件回答"**这一篇**是不是
    #: 本周新工作"。实测 2026-09-17 那 537 条里混着 ``year=2023`` 的老预印本，没它区分不了。
    announce_seen: dict[str, str] = {}
    announce_upgraded = 0      # 先撞上交叉视角、后被主分类视角纠正的条数（正常）
    announce_conflicts = 0     # 同侧视角互相矛盾 → 键撞了或源变了（**必须为 0**）
    announce_missing = 0       # feed 本身没给 announce_type 的条数
    failed: list[str] = []
    #: 解析出 0 条的 feed。**它是信息不是报警**（周末就是空的），但一定要看得见。
    empty_feeds: list[str] = []
    #: 分类 → ``<lastBuildDate>`` 原样字符串。"这份字节是哪一刻的"。
    feed_build: dict[str, str | None] = {}
    #: 分类 → 刚落盘那份 raw 的 sha256。用于与前一天比，见下面 ``raw_unchanged``。
    raw_digest: dict[str, str] = {}
    #: 分类 → 这个 feed 当天给了什么（ID 集 / 首次公告集 / 最大 ID）。
    #: ⚠️ 一律从 ``res.papers`` **全量**取，不吃 ``--limit``：feed 的顺序不是 ID 顺序，
    #: 截断后算出的最大值是一个**错的数**，而不是一个"缺的数"——前者更坏。
    #: 抓失败的 feed 会以 ``None`` 补进来（见循环之后），因为"缺席"与"没停摆"
    #: 在汇总行里长得一模一样，正是本库那个签名病。
    feed_samples: dict[str, feedwatch.FeedSample | None] = {}
    #: 这一跑自动归档掉的旧 raw（周报库根相对路径）。**动手的变成脚本了，不再是手工步骤。**
    raw_superseded: list[str] = []
    #: 归档失败的那些。**不静默**：留档是佐证不是料，写不出去不中止抓取，但必须看得见。
    raw_superseded_error: list[str] = []
    no_id = malformed = 0
    dupes = 0
    seen: dict[str, CanonicalPaper] = {}

    for category in arxiv_rss.CATEGORIES:
        url = arxiv_rss.RSS_URL.format(category=category)
        # ⚠️ **必须 `refresh=True`，不能吃 URL 缓存。** `Fetcher` 的缓存键是 `sha1(url)`，
        # 而 RSS 这条源**同一个 URL 每天的内容都不同**（滚动窗口）——那个缓存没有日期、
        # 没有 TTL，于是从第二天起，"日更"读的是**第一天那份字节**。
        #
        # 2026-09-18 实测事故：日更**退出码 0**、`accounting_ok: true`、`failed_feeds: []`、
        # `upsert: {"unchanged": 537}`——而 `http: {"hit": 3, "fetched": 0}`，
        # 三个 `raw/2026-09-18-*.xml` 与 09-17 的**逐字节相同**（sha256 相等），
        # `2026-09-18.announce.jsonl` 同理。换一个空缓存目录真取一次才发现：
        # **三个 feed 全部不同**，`lastBuildDate` 就是当天 04:00 UTC。
        # ⇒ 那次"成功"的日更**一篇都没抓到**，而它留下的产物**看起来完全正常**——
        # 正是本库那个签名病（「没读到」长得和「没有」一模一样）长在了链条最上游，
        # 而它恰好会毁掉"必须日更"的唯一理由（窗口一滚，那天就永远没有了）。
        #
        # 代价是每天固定 **3 请求**——那正是本脚本开头「请求预算」里写明的设计值；
        # 而 `raw/<日期>-<分类>.xml` 本来就是那份**可重放的持久副本**。
        # ⇒ URL 缓存对这条线**不提供任何东西，只制造静默失败**。
        #
        # ⚠️ **许可段不要跟着改**：OAI 的 `GetRecord` URL 按 id 永久稳定、是真的幂等，
        # 那里的缓存是**对的**（所以那里仍然只认 `args.refresh`）。
        xml = fetcher.get_text(url, refresh=True)
        if not xml:
            failed.append(category)
            continue

        # 先原样落盘再解析：重解析不重抓。RSS 是滚动窗口，抓过的这一份
        # 事后**没有任何办法**再拿到，所以它必须比解析结果先落。
        payload = xml.encode("utf-8")
        raw_path = raw_dir / f"{today}-{category}.xml"
        # 落点是**同日同路径整体覆盖**，所以先把马上要被换掉的旧字节留一份档。
        # 它防住了哪一次、又**没有**防住哪一次，见 _archive_superseded 的 docstring。
        try:
            archived = _archive_superseded(raw_path, payload, week_dir, run_date, stamp)
            if archived is not None:
                raw_superseded.append(str(archived))
        except OSError as exc:
            # 留档是佐证不是料：**不因为一个副本写不出去就中止当天的抓取**。
            # （`atomic.replace` 自己对 PermissionError 已经退避重试约 6 秒。）
            raw_superseded_error.append(f"{category}: {exc}")
        write_atomic(raw_path, payload)
        raw_digest[category] = hashlib.sha256(payload).hexdigest()

        res = arxiv_rss.parse_feed(xml, category=category)
        feed_build[category] = res.build_date
        feed_samples[category] = feedwatch.FeedSample.of(res.papers, res.announce_by_key)
        if not res.papers:
            # ⚠️ **零条不是错误。** feed 自己带 ``<skipDays>Sunday, Saturday</skipDays>``，
            # 周末本来就空的（2026-09-19 实测：三个 feed 各 0 条 / 892·892·900 字节，
            # 而 ``lastBuildDate`` 是**当天** 04:00）。但它必须是**看得见**的信息——
            # 本库 2026-09-18 那次事故里"零篇"长得和这个一模一样，而那次是**没读到**。
            empty_feeds.append(category)

        no_id += res.no_id
        malformed += res.malformed
        announce.update(res.announce_types)

        # 逐条收集。**遍历 res.papers 全量，刻意不吃 --limit**：raw/*.xml 与上面的聚合
        # 计数都是全订阅量的口径，旁路文件跟它们对齐。否则开发期 `--limit 5` 会悄悄落一份
        # 只有 5 条的旁路文件，而那份文件事后**看不出**是"源就只有 5 条"还是"我们只记了 5 条"。
        for p in res.papers:
            pkey = p.arxiv_id or p.title_norm
            atype = res.announce_by_key.get(pkey)
            if not atype:
                announce_missing += 1
                continue
            prev = announce_seen.get(pkey)
            if prev is None:
                announce_seen[pkey] = atype
            elif prev == atype:
                continue
            elif _is_cross_view(prev) and not _is_cross_view(atype):
                # 正常路径：先撞上的是它的交叉列出视角，后撞上的是主分类视角。**取后者**——
                # 卡片要回答"这篇是不是本周新工作"，而 `cross` 只是"它还被列在别的分类里"。
                announce_seen[pkey] = atype
                announce_upgraded += 1
            elif not _is_cross_view(prev) and _is_cross_view(atype):
                continue          # 手里已经是主分类视角，交叉视角不覆盖它
            else:
                # 两个**同侧**视角互相矛盾（``new`` vs ``replace``，或 ``cross`` vs
                # ``replace-cross``）——只可能是**键撞了**（两篇不同论文共用一个
                # ``title_norm``）或源变了。这一条今天实测为 0，非零就必须查：
                # 它不能静默按先到的那个算，那样卡片上那一栏看起来完全正常。
                announce_conflicts += 1

        papers = res.papers[: args.limit] if args.limit else res.papers
        per_category[category] = len(papers)

        for paper in papers:
            key = paper.arxiv_id or paper.title_norm
            first = seen.get(key)
            if first is None:
                seen[key] = paper
                continue
            # 同一篇被第二个 feed 看见。当场并掉分类，而不是留给 store 的 _merge：
            # 那样同一天的快照里会出现两条一模一样的记录，upsert 会记成
            # "1 new + 1 updated"，把"跨分类重复"伪装成"后来又更新过"。
            dupes += 1
            merged = list(dict.fromkeys([*first.topics, *paper.topics]))
            if merged != first.topics:
                first.topics = merged

    papers = sorted(seen.values(), key=lambda p: (p.arxiv_id or "", p.title_norm))

    # ---- 这条 feed 到底有没有**往前滚**？----
    # 判据是**重发条数**（今天标 new/cross、而上一份 raw 里已经有的那些），
    # 不是最大 ID、更不是字节。为什么、以及实测数字（正常日 0/903、09-22 cs.AI
    # 125/125），全部写在 `rllib/feedwatch.py` 的模块 docstring 里。
    #
    # 抓失败的 feed 也**补进样本表**（值为 None）：让它以"键不存在"的形式退场，
    # 它在汇总行里就只剩缺席，而缺席与"没停摆"长得一模一样。见 feed_report 的 docstring。
    for category in failed:
        feed_samples.setdefault(category, None)
    # ⚠️ 基线**只能是前一天的 raw**，不能拿 store 当基线：`CATEGORIES` 按
    # (cs.LG, stat.ML, cs.AI) 之序处理，轮到 cs.AI 时它当天的 cross 条目**已经被
    # cs.LG 入库了**——每一个当天的交叉列表都会看着像"已知"。
    baselines: dict[str, feedwatch.Baseline] = {}
    for category in feed_samples:
        baseline = _load_baseline(weekly_dir, today, category)
        if baseline is not None:
            baselines[category] = baseline
    feed_advance = feedwatch.feed_report(feed_samples, baselines)

    # ---- 这一份字节是不是**新的**？----
    # 2026-09-18 那次静默事故的**硬证据**就是"今天的 raw 与前一天逐字节相同"，
    # 而当时它是靠人**事后手工比 sha256** 才发现的——这一条把它写进代码。
    # 同一个 URL 昨天与今天的字节相同，只可能是两件事之一：
    #   ① 源没重建（周末 / 还没到公告时刻）——正常；
    #   ② **缓存又把旧字节喂回来了**——就是那次事故的真身。
    # 两者的产物完全一样（`raw/<今天>` 装着前一天的字节），所以**只能报警让人判**。
    # ⚠️ **2026-09-22 起，这里的基线是跨周找的**（`_prev_raw_path` 改成枚举
    # `*/raw/*`）。原来只在本周目录里 glob，于是**每个周一**（新 ISO 周、本周
    # `raw/` 还不存在）都返回 None ⇒ `raw_unchanged` 恒为空 ⇒ `raw_stale` 恒为假：
    # **周一逐字节重发是结构性看不见的**。这不是"只影响周一"，而是"任何本周目录里
    # 没有更早 raw 的时刻"，也包括回填本周最早的日期。
    # 唯一的跨周实例（09-18 周五 → 09-21 周一）是**真阳性不是误报**：三分类 sha
    # 全不同、最大 ID 全部前进（cs.AI 2609.20822 → 2609.22086）。
    # ⚠️ **但这一条抓不到 09-22 那次事故**：那次的重发**不是逐字节相同**
    # （`pubDate` 被重戳，首个差异字节在偏移 632 的 `<lastBuildDate>` 里）。
    # 它和上面的 `feed_advance` 打的是**两个不同的失效类**，别让谁冒领谁的功劳。
    raw_unchanged = [
        cat for cat, d in raw_digest.items()
        if (prev := _prev_raw_path(weekly_dir, today, cat)) is not None
        and hashlib.sha256(prev.read_bytes()).hexdigest() == d
    ]
    # ⚠️ **周末不走退出码。** arXiv 的 feed 带 ``<skipDays>Sunday, Saturday</skipDays>``，
    # 周末本来就没有新一批公告，"与前一天相同"未必是故障——而**一个会在正常日子变红的
    # 检查会训练人忽略红色**（CLAUDE.md）。判错的代价也**不对称**：周末误报的代价是
    # 白看一眼，工作日的漏报是一**整天**的公告**永远拿不回来**（窗口一滚就没了）。
    #
    # ⚠️ **"周末容忍"这条的依据只有两条，都还薄**：feed 自己那句 ``<skipDays>``，
    # 加 2026-09-19（周六）**一个样本**。⚠️ **arXiv 周日到底重建不重建 feed 没验证过**
    # ——若周日不重建，那天的字节会与周六逐字节相同，本条会**静默容忍**它（进
    # ``raw_unchanged`` 但不改退出码）；周一再结合 `upsert` 看是不是真拿到了新一批。
    # 见 ``MEASUREMENTS.md`` #29。
    raw_stale = bool(raw_unchanged) and day.weekday() < 5

    # ---- 这一份字节是不是"今天"的构建？----（待办 #13，2026-09-24 落地）
    # 判据：feed 自己写的 ``<lastBuildDate>`` 的**日期** == 本地今天。不为则这份是
    # **前一天（或更早）的构建**。它盖住两类失效：
    #   ① **跑早了**——本地 12:00（= UTC 04:00）之前跑，拿到的是昨天那份构建，
    #      而它退出码 0、汇总行全绿（09-21 的 09:02 那一跑就是：`feed_build` 写着
    #      `Sun, 20 Sep` 而那天是周一）。``raw_unchanged`` 盖不住它（重建了、但日期是昨天）。
    #   ② **源没重建**——构建停在了更早。
    # 比 ``raw_unchanged``（只比对**逐字节**）强一档：那份分不开"日期错了一天"与
    # "正常的新字节"；这一条直接量"这份字节是哪一天的"。
    #
    # ⚠️ **只报告、不进退出码**（与 ``feed_stalled`` 同级）：美国节假日 arXiv 会**顺延**、
    # 当天 feed 可能不重建或重建稍晚，本地午后跑时 ``<lastBuildDate>`` 仍是昨天——进
    # ``rc`` 会在这些"正常"日子变红，而**一个会在正常日子变红的检查会训练人忽略红色**
    # （本库红线，见库根 §2.2 待办 #13 的口子警告）。先攒够绿工作日证据再考虑提级。
    # ⚠️ 基准用**本地 ``date.today()``**，不用 ``day``：``--date`` 回填的过去日期会误报。
    feed_build_stale = [
        cat for cat, bd in feed_build.items()
        if bd and not feedwatch.build_date_matches(bd, date.today().isoformat())
    ]

    # RSS 那一段的网络计数。**必须在许可段之前取**：``fetcher.stats`` 是全局累加的，
    # 把 OAI 的算进来就会让 `http` 那个数把两段混成一个——而两段的健康值**恰好相反**
    # （公告段每次运行都该真取、`fetched: 3`；许可段则该尽量命中 `cache/`），
    # 混起来两个都看不出来。两段分开报，见 ``http_oai``；健康值的定义在下面 `http` 处。
    # > ⚠️ 此处曾引着一句"重跑同一天应该是 3 次全 hit、0 次 fetched"当判据——
    # > **那是 2026-09-18 改 `refresh=True` 之前的旧口径，早就指反了**。同族的教训
    # > 见「注释不是判据」：一句没跟着改动走的注释，会替错的实现背书。
    http_rss = dict(fetcher.stats)

    store = PaperStore(weekly_dir / "papers.jsonl").load()
    results: Counter[str] = Counter()
    for paper in papers:
        results[store.upsert(paper)] += 1

    # ---- 许可（OAI-PMH，见 rllib/weekly_license.py）----
    # 必须在 upsert **之后**：判据是"库里有 arXiv ID 但没有许可的那些"，
    # 当天刚解析出来的这批要先入库才算数。⚠️ 首次运行时库还是空的，
    # 这一步就只会补到存量——而存量与日更是**同一个函数**，所以
    # scripts/backfill_license.py 不需要另写一套逻辑。
    lic = weekly_license.fill_missing(
        store, fetcher, refresh=args.refresh, limit=args.license_limit or None,
    )

    # 把许可抄回**当天快照**要写的那批对象上。理由：快照是"当天解析出了什么"的凭据，
    # 而许可**不可能**从 RSS 或 raw/*.xml 重解析出来（只有 OAI-PMH 有），
    # 漏掉它，快照对账时就有一个说不清的洞。
    # 新入库的论文 `_insert` 存的就是 `papers` 里那个对象本身，identity 相同、
    # 不需要抄；已经在库里的那些才需要从库对象上取回来。
    for paper in papers:
        stored = store.find(paper)
        if stored is not None and stored is not paper:
            paper.license = stored.license

    snap_path = week_dir / f"{today}.jsonl"
    write_atomic(
        snap_path,
        "".join(
            json.dumps(p.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")) + "\n"
            for p in papers
        ).encode("utf-8"),
    )

    # 旁路文件（用户 2026-09-17 选 A）：公告类型**不进 ``CanonicalPaper``**（冻结契约，
    # 铁律 5），单独落一份 ``key -> announce_type``，与上面那份快照**同生命周期**——
    # 同一天重跑即整体重写，所以重跑不会累积重复行。
    # 它填的是卡片「来源」栏里唯一**不能**从 jsonl 得到的那一项。
    announce_path = week_dir / f"{today}.announce.jsonl"
    write_atomic(
        announce_path,
        "".join(
            json.dumps({"key": k, "announce_type": v},
                       ensure_ascii=False, separators=(",", ":")) + "\n"
            for k, v in sorted(announce_seen.items())
        ).encode("utf-8"),
    )

    store.save(owned_fields=OWNED_FIELDS)

    # 恒等式（铁律 7 的精神）：每个 feed 看见的条数之和，必须等于去重后的条数
    # 加上去重时丢掉的重复数。不成立就说明有 feed 的条数没被记进来——
    # 那是"某个分类悄悄没抓"，除了这一行之外没有任何地方会发现。
    identity_ok = sum(per_category.values()) == len(papers) + dupes

    summary: dict[str, object] = {
        "date": today,
        "week": week,
        "per_category": per_category,
        # **全订阅量**的公告类型分布（在 --limit 之前统计）。不可事后补：窗口一滚就没了。
        # 实测（2026-09-17）取值有四个：new / cross / replace / replace-cross。
        # 打整张表而不是固定几个键，新取值会自己冒出来（见 adapters/arxiv_rss.py）。
        "announce_types_feed": dict(announce),
        # —— 旁路文件的四个数 ——
        # `sidecar` 是落进 `<日期>.announce.jsonl` 的行数（全订阅量口径）。
        # `upgraded` = 被交叉视角先污染、又被主分类视角纠正的条数，**正常**（实测 105）；
        #   它掉到 0 也该看一眼：那意味着"同一篇被多个 feed 看见"这件事没发生。
        # `conflicts` 非零 = 键撞了或源变了（两篇不同论文共用一个 title_norm），**必须查**。
        # `missing` 是 feed 本身没给 announce_type 的条数——它不是错，但**要看得见**：
        # 否则"旁路文件比快照少几行"永远分不清是"源没提供"还是"我们漏记了"，
        # 那正是本库反复栽的那个病（分不清「没有」与「没读到」）。
        "announce_sidecar": len(announce_seen),
        "announce_upgraded": announce_upgraded,
        "announce_conflicts": announce_conflicts,
        "announce_missing": announce_missing,
        "unique": len(papers),
        "dupes_across_feeds": dupes,
        "upsert": dict(results),
        "no_id": no_id,
        "malformed": malformed,
        "failed_feeds": failed,
        # **"零篇"必须看得见，而且必须带上"这份字节是哪一刻的"。**
        # 周末三个 feed 各 0 条是**正常**的（feed 自带 `<skipDays>`）；而 2026-09-18
        # 那次故障留下的产物**长成同一个样子**——`upsert {"unchanged": N}`。
        # 分开它们的是这两行：`feed_build` 是当天还是停在前一天、`http.fetched`
        # 是 3 还是 0。`raw_unchanged_feeds` 是同一件事的**硬形式**（逐字节比对）。
        # ⚠️ `empty_feeds` **不进退出码**：拿它当错误会在每个周末误报。
        "empty_feeds": empty_feeds,
        "feed_build": feed_build,
        # —— 待办 #13（2026-09-24 落地）：这份字节是不是"今天"的构建 ——
        # `feed_build` 里那些 `<lastBuildDate>` 的日期 != 本地今天的分类。**只报告、不进退出码**。
        #   · 非空 = 跑早了（本地 12:00 / UTC 04:00 之前跑，拿到昨天那份构建，09-21 的 09:02 就是）
        #     或源没重建。**要人看一眼**，不是自动失败——美国节假日 arXiv 会顺延，进 rc 会误报。
        #   · 空 = 每个成功抓到的 feed 都确是当天构建。
        "feed_build_stale": feed_build_stale,
        "raw_unchanged_feeds": raw_unchanged,
        # —— feed 有没有往前滚（2026-09-22 起的新判据，见 rllib/feedwatch.py）——
        # 展开 `feed_advance` 那四个键：
        #   `feed_advance`            分类 → 读数（基线出处 / 最大 ID + 带符号 delta /
        #                             new_cross / reissued）。**人眼读这个。**
        #   `feed_stalled`            `reissued > 0` 的分类。**这是判据。**
        #   `feed_advance_unchecked`  没比成的分类（含抓失败的）。
        #   `feed_baseline_mismatch`  基线自己说它那天不是它那天（文件名撒过谎）。
        # ⚠️ **`feed_stalled` 有意不进退出码（先只报告）**：判据本身在 9 个正常日
        # 上实测 0/903、在事故上 125/125，但"跨月/跨年会不会误报"还没量过。
        # 按本库那条「一个会误报的检查比没有检查更糟」，攒够约 5 个绿工作日再提级。
        # ⚠️ **`feed_stalled: []` 不等于"都查过、都没停"**——没查成的那些在
        # `feed_advance_unchecked` 里，每一条还带 `unchecked_reason`。
        **feed_advance,
        # 这一跑自动留档的旧 raw。**动手的变成脚本了**：09-21 那批字节是靠人手工
        # 补的档（`raw-archive/2026-09-21/`），下一次没人记得。空列表是正常的
        # （字节没变就不归档，见 `_archive_superseded`）。
        "raw_superseded": raw_superseded,
        "raw_superseded_error": raw_superseded_error,
        # **RSS 那一段**的网络计数（许可段的在 `license.http_oai`，两段分开）。
        # ⚠️ **健康值是 `fetched: 3 / hit: 0`**——RSS 段**每天都该真去取**（理由见上面
        # `refresh=True` 那一段注释）。**`fetched` 为 0 是故障信号，不是"省了请求"**：
        # 2026-09-18 那次事故里它正是 0，而**汇总行其余每一个数都绿着**。
        # > 此处曾写着"重跑同一天应该是 3 次全 hit、0 次 fetched"，把 hit 当健康值——
        # > **那个判据指反了**，而且它分不开"同一天重跑"（对）与"第二天首次跑"（错），
        # > 两者都是 `hit: 3 / fetched: 0`。一个分不开这两种情形的检查，
        # > 在出错那天不会变红。
        "http": http_rss,
        # 许可段（OAI-PMH）。⚠️ 里面有**三个数不是一回事**，别把它们读成一个：
        # `license_filled` 取到了 / `license_none` 源明说这条记录没有许可 /
        # `license_failed` 我们没读到。后两者在库里都是 null，但一个关于源、一个关于我们。
        "license": lic,
        "weekly_total": len(store),
        "accounting_ok": identity_ok,
    }
    if not args.no_filter and papers:
        summary.update(_run_filter())

    # 有 feed 失败就不能算成功——否则 cron 与人都以为今天抓全了。
    # `raw_stale` 同理，而且更隐蔽：拿回来的若是**前一天那份字节**，今天这一批
    # 就**永远拿不回来了**（窗口一滚，arXiv 那边也没有了）。
    # ⚠️ `feed_stalled` **有意不进这里**（先只报告），理由见上面那段注释。
    rc = 1 if failed or raw_stale else 0

    # ---- 收尾：让"这一跑跑完了没有"在**盘上**看得见 ----
    # 动因（2026-09-22 实测）：前两跑被内存回收杀掉，盘上留下「`raw` 有、快照没有」
    # 的半截形状，那个状态躺了 **1 小时 40 分钟**——而从 `ls` 上看，它与"跑完了、
    # 只是还没生成快照"**完全一样**。落盘顺序是 raw → 快照 → 旁路 → papers.jsonl，
    # **raw 最先落**，所以中断必留这个形状，而当时**没有任何东西分辨得出来**。
    #
    # 位置有三条讲究：
    #   1. 在 `store.save()`（真正让当天数据 durable 的那次写）**之后**；
    #   2. 在 `_run_filter()` **之后**——否则会永久记下一趟"最重要的数字却没了"的跑；
    #   3. 在 `print` **之前**——stdout 是关掉的管道（`| head`）时 `print` 抛
    #      `BrokenPipeError`，会让一趟**成功**的跑反而**不留**标记。
    # 标记内容 = **同一行**汇总 JSON（与 `print` 共用同一次 `json.dumps`，
    # 免得"同一份内容的两个定义"老化成分叉）。
    #
    # ⚠️ **标记缺席 = 异常结束**：`HostBlocked`(3) / `DiskLow`(4) 是在 `main()`
    # **之外**捕获的，被外部杀掉同理——那两种情形**刻意不写标记**。
    # ⚠️ 发现它要**按精确文件名**（`<日期>.done`），不要 glob `*done*`：
    # `clean_tmp` 的残留模式是 `<名字>.<pid>.tmp`，被杀的一跑会在同一个目录里
    # 留下一个**截断的** `2026-09-22.done.<pid>.tmp`。
    # ⚠️ `--limit` 冒烟跑与"全 feed 都失败"也会走到这里、也会写出标记，所以标记里
    # **必须**有 `args` 与 `exit_code`——否则它会把一趟有上限的、或失败的跑
    # 认证成"当天抓取完成"（全失败时 `write_atomic` 还会顺手造出周目录 + 0 字节快照）。
    summary.update({
        "schema": 1,
        "ended": datetime.now().isoformat(timespec="seconds"),
        "duration_s": round(time.monotonic() - t0, 1),
        "exit_code": rc,
        "args": {
            "date": args.date,
            "limit": args.limit,
            "license_limit": args.license_limit,
            "refresh": args.refresh,
            "no_filter": args.no_filter,
        },
    })
    line = json.dumps(summary, ensure_ascii=False)
    write_atomic(week_dir / f"{today}.done", (line + "\n").encode("utf-8"))
    try:
        print(line)
    except OSError:
        pass      # stdout 被关了，不该让一趟跑成功的运行连标记都没留下

    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HostBlocked as exc:                      # 与 `python -m rllib` 同一套退出码
        print(json.dumps({"error": "host_blocked", "detail": str(exc)}, ensure_ascii=False),
              file=sys.stderr)
        raise SystemExit(3) from None
    except DiskLow as exc:
        print(json.dumps({"error": "disk_low", "detail": str(exc)}, ensure_ascii=False),
              file=sys.stderr)
        raise SystemExit(4) from None
