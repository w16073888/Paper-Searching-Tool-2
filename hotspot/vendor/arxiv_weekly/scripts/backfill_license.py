"""给周报库补 arXiv 再分发许可（一次性 / 可反复跑）。

与日更脚本的关系
----------------

``daily_arxiv_fetch.py`` 里有一段补许可，本脚本调的是**同一个函数**
（:func:`rllib.weekly_license.fill_missing`）。区别只有一个：日更顺手补当天入库的
那些，本脚本**不抓 RSS**、专门把存量库补完。所以：

* **首次运行**（2026-09-18 之前入库的论文全都没有许可）用它。
* 之后正常情况下**不需要**它——日更每天都在补。
* 日更跑到一半被限流或磁盘刹车打断时，也可以用它单独续。

判据是「库里有 arXiv ID 但没有许可的那些」，所以**重复跑是安全的**：
已经采到的不会再被问一次（许可**不可撤销**，见 ``MEASUREMENTS.md`` §四c #24），
已经抓过的靠 ``cache/`` 零网络命中。

为什么是逐条问、不是按窗口整日拉
--------------------------------

见 :mod:`rllib.adapters.arxiv_oai` 开头那三条（带宽 7 MB/天、缓存是一次性的、
datestamp 窗口会把"没罩住"读成"没有"）。一句话：``GetRecord`` 一篇一请求、
响应实测 2,790 字节、URL 按 id 永久稳定 ⇒ **一篇一辈子只抓一次**。

代价是墙钟时间：``HOST_DELAYS["oaipmh.arxiv.org"] = 4.0`` ⇒ 约 4 秒/篇。
537 篇的存量库约 **36 分钟**。这不是卡住，别把它当故障；想看进度就翻 ``cache/``
的文件数增长（脚本本身按 token 纪律只打一行聚合 JSON）。

用法
----

.. code-block:: text

   python scripts/backfill_license.py                # 补完所有缺的
   python scripts/backfill_license.py --limit 20     # 开发期冒烟（铁律 4.3）
   python scripts/backfill_license.py --refresh      # 绕过 cache/ 重抓

退出码与主库长任务一致：**3 = 被限流/封锁，4 = 磁盘余量不足**——有意的中止，
不是崩溃；已抓到的都在 ``cache/`` 里，重跑零网络接上。

⚠️ 本脚本**只认领 ``license`` 一个字段**（铁律 3），所以它能和主库那边的长任务
并行；但**不要和 ``daily_arxiv_fetch.py`` 同时跑**——两边都在补许可，值不会写坏
（``store._would_downgrade`` 挡着），但请求量翻倍，而 arXiv 的间隔是我们自己
定出来保护自己的。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllib import ROOT, weekly_license  # noqa: E402
from rllib.cli import _setup  # noqa: E402
from rllib.console import use_utf8_console  # noqa: E402
from rllib.diskguard import DiskLow  # noqa: E402
from rllib.http import HostBlocked  # noqa: E402
from rllib.store import PaperStore  # noqa: E402

#: 只认领许可。**刻意不认领 ``last_updated``**：本脚本一次能跑半小时以上，
#: 认领它会把全库"最后更新"刷成今天，而真正更新的是 ``enrich`` / ``extract``
#: 那些长任务——那样一来"这篇多久没动过"这个判断就废了。
#: ⚠️ 这**与 ``daily_arxiv_fetch.py`` 不同**，那边认领它是因为同时还在 upsert
#: 新论文（那确实是一次实质更新）。这里的差异是有意的，不是抄漏了。
OWNED_FIELDS = ("license",)


def main() -> int:
    use_utf8_console()

    ap = argparse.ArgumentParser(
        description="给周报库 papers.jsonl 补 arXiv 再分发许可（只输出聚合摘要）。",
    )
    ap.add_argument("--limit", type=int, default=0,
                    help="本轮最多问几篇（0 = 全量）。没问到的下一轮还在目标集里")
    ap.add_argument("--refresh", action="store_true",
                    help="绕过 HTTP 缓存重新拉取（默认命中 cache/ 就不重抓）")
    args = ap.parse_args()

    cfg, fetcher = _setup(False, "weekly")
    path = ROOT / cfg["paths"]["data"] / "papers.jsonl"
    store = PaperStore(path).load()

    report = weekly_license.fill_missing(
        store, fetcher, refresh=args.refresh, limit=args.limit or None,
    )
    # ⚠️ 落盘只在**跑完之后**这一处。中途被 HostBlocked / DiskLow 打断时，
    # 内存里已取到的许可不会写进去——但它们**在 ``cache/`` 里**，
    # 重跑是零网络命中。用"一次落盘"换掉"半截状态入库"，是这里的有意取舍。
    store.save(owned_fields=OWNED_FIELDS)

    summary: dict[str, object] = {
        "store": str(path.relative_to(ROOT)).replace("\\", "/"),
        "weekly_total": len(store),
        "owned_fields": list(OWNED_FIELDS),
        "at": datetime.now().isoformat(timespec="seconds"),
    }
    summary.update(report)

    # 恒等式（铁律 7 的精神）：目标集 = 这一轮补上的 + 源明说没有的 + 没读到的
    #                                + 这一轮没问的（受 --limit 限制）
    skipped = report["license_targets"] - report["license_attempted"]
    accounted = (report["license_filled"] + report["license_none"]
                 + report["license_failed"] + report["license_id_mismatch"] + skipped)
    summary["accounting_ok"] = accounted == report["license_targets"]

    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["accounting_ok"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HostBlocked as exc:
        print(json.dumps({"error": "host_blocked", "detail": str(exc)}, ensure_ascii=False),
              file=sys.stderr)
        raise SystemExit(3) from None
    except DiskLow as exc:
        print(json.dumps({"error": "disk_low", "detail": str(exc)}, ensure_ascii=False),
              file=sys.stderr)
        raise SystemExit(4) from None
