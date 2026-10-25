"""周报 concern 人眼验收页（产线）：把这一周的抽取结果排成一份**给人读**的清单。

⚠️ **2026-09-19 用户拍：这一遍验收取消，本脚本休眠。**
它验的是 ``concern`` 引文的质量（「缺口**或**展望合格，纯总结不合格」），
而 ``concern`` 现在**不发给任何人**（交付层只取摘要）⇒ **没有交付意义**。
靠山 ``weekly_sections.py`` 同批休眠，它读的 ``sections.jsonl`` 也就停在原处。
✅ **脚本保留、可随时重跑**（纯离线、读磁盘、幂等）——哪天恢复 ``concern`` 这条线，
它接管的仍然是同一件事。**它从来不是"不重要"，是"现在没有受众"。**

与 `weekly_sections.py` 的关系
------------------------------

**它接管的正是 ``sections_review.md``。** 那一份由 ``weekly_sections.py`` 从
**内存里的本次新增**写出——只覆盖最近一次跑的那些篇、无排序、无优先级
（09-17 那一轮的 47 篇就永远不在里面）。这一份从**磁盘**上重算本周全部、
按可疑度排。**一个文件一个属主** ⇒ 那一份已经删掉，别再把它加回来。

与 `weekly_render.py` 的关系
----------------------------

渲染器把每一篇落成**一片卡片**（``<周>/concern/<arxiv_id>.md``，机器区/人工区
隔离，**批注写在那里**）；本脚本把这些片**排成一份清单**，回答"先看哪片"。
两者都离线、都幂等，谁先谁后无所谓。

**产物不进交付**：``export_weekly.py`` 的落点清单是显式的，没有这一页。

**Token 纪律**：终端只打一行聚合 JSON，绝不回显标题或正文。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllib import ROOT  # noqa: E402
from rllib.cli import _setup  # noqa: E402
from rllib.console import use_utf8_console  # noqa: E402
from rllib.diskguard import DiskLow  # noqa: E402
from rllib.http import HostBlocked  # noqa: E402
from rllib.render import weekly_review  # noqa: E402


def main() -> int:
    use_utf8_console()
    # **没有开关**，这是有意的：这一页没有"部分渲染"的合法语义——截一半的验收
    # 清单和"本周就这么少"在页面上长得一模一样（同 `weekly_render.py` 里周索引
    # 不跟着 `--limit` 截断的理由）。要少看，是人眼的事，不是渲染器的事。
    cfg, _fetcher = _setup(False, "weekly")       # 完全离线，fetcher 用不上
    weekly_dir = ROOT / cfg["paths"]["data"]      # 周报库落点：绝对路径，见 cli.py::_WEEKLY_PATHS

    result = weekly_review.render_review(weekly_dir)
    print(json.dumps(result, ensure_ascii=False))

    # 恒等式（铁律 7）：本次处理的每一篇都必须落进**已登记**的 outcome 桶里。
    # 不成立 = 有人加了新桶却没登记进 `gap.OUTCOMES`，那几篇会在第 4 组的表里
    # 带着一行 ⚠️ 混着——**不该只靠人眼在第 4 组里发现它**。
    #
    # `no_week` 非空 = 有篇 `first_seen` 是空的，它**不在任何一页上**。这不是
    # "跳过"，是"这份验收清单少了一片而没人知道"——必须变成非零退出码。
    return 1 if (result["unknown_outcomes"] or result["no_week"]) else 0


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
