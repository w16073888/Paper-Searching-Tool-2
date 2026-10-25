"""摘要合订本渲染（产线）：把周报库里**在范围内的**每一篇印成一份 md，供整份喂给模型。

⚠️ 2026-09-19 用户拍：**周报线收缩成「日更 → 合订本 → 发群」**。
本脚本**默认只出合订本**，卡片与周索引**默认不再渲**（`--with-cards` 可临时恢复）。

两条理由，都不要当成"暂时省点事"：

1. 交付层**只取摘要**（`concern` 不进任何对外产物）⇒ 卡片与周索引的正文**全是 concern**，
   现在没有读者。
2. 更要紧的是：**继续每天渲它们，只会往盘上写一批"还没抽"的半成品**，而那些页面上
   明写着「跟着跑一次 `weekly_sections.py` 就补上」——**那是一句永不兑现的承诺**。
   本库最防的就是"说得像结论"（铁律 8 那条的形状）。

⚠️ **卡片与抽取是一对**：开 `--with-cards` **必须同时恢复 `weekly_sections.py`**，
否则那句话立刻变成假话。`--limit` / `--force` 也因此**只对 `--with-cards` 有意义**
（合订本没有指纹、也不截断）。

与 `weekly_sections.py` 的关系
------------------------------

**合订本不依赖抽取。** 它只读 ``papers.jsonl``（`render_digest` 里把 sections 与
公告旁路都是下划线接住不用的）⇒ **离线、秒级、随便重跑**。

⚠️ 卡片与索引**曾经**依赖它，而且依赖得很紧：**先抽后渲，两步不能并**；卡片的增量指纹
**必须覆盖正文**（一篇今天 ``outcome=""``、明天变 ``ok`` 加一段正文，而 ``papers.jsonl``
那一行一个字都没变 ⇒ 指纹只看元数据的话它会**永远停在「还没抽」**且不报错，
铁律 6 的形状。已写进 `rllib/render/weekly_card.py` 的 ``card_digest()``）。
**这一段现在休眠**——但哪天恢复 ``--with-cards``，它就是必须一起恢复的理由。

落点
----

.. code-block:: text

   <周报库根>/
     <周>/digest.md                 摘要合订本（**喂模型的原料**，2026-09-19 新增）
     <周>/concern/<arxiv_id>.md     每篇一片（如 2026-W38/concern/2407.06868.md）—— 休眠
     <周>/report.md                 周索引（这一周的**账**：每篇落在哪个 outcome）—— 休眠
     render_state.json              卡片的内容指纹，决定哪片要重渲（休眠，只有卡片用）

⚠️ **三类产物曾是一个脚本出，但它们从来不是一件事**：卡片与索引给人读，
合订本**给模型读**。后者的版式与措辞按"大模型是读者"定（不做锚点、
口径必须写在正文里、必须写明"不能回答什么"），理由写在
`rllib/render/weekly_card.py` 的「摘要合订本」那一节，改它之前先读那一段。

⚠️ **合订本是最不容易坏的那一份**：它只读 `papers.jsonl`（`_sections` / `_announce`
在 `render_digest` 里都是下划线接住不用的）⇒ **抽取停了它照样出得来**。
反过来说，卡片与索引**必须**有 `sections.jsonl` 才有内容——这就是它们跟着一起停的原因。

⚠️ ``render_state.json`` 与主库的 ``data/render_state.json`` **同名但不同文件**，
按目录区分（各在各自的 ``paths.data`` 下）。别把两者当成一个。

产物是**机器区 + 人工区**的：``<!-- AUTO:END -->`` 之前是程序生成的，重渲会覆盖；
之后永远是你的（卡片与合订本模板新建时都会追加 ``MANUAL_SCAFFOLD`` 骨架）。
**读者写下的批注不会丢** —— 这是复用主库 ``upsert_note`` 换来的，
那个函数里的结束标记是**整行匹配**的，抄一份就会把「截断文件再拼接」那个坑重种一遍。

**Token 纪律**：终端只打一行聚合 JSON，绝不回显标题或正文。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllib import ROOT  # noqa: E402
from rllib.cli import _setup  # noqa: E402
from rllib.console import use_utf8_console  # noqa: E402
from rllib.diskguard import DiskLow  # noqa: E402
from rllib.http import HostBlocked  # noqa: E402
from rllib.render import weekly_card  # noqa: E402


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser(description="摘要合订本渲染（离线）。只输出聚合摘要。")
    ap.add_argument("--with-cards", action="store_true",
                    help="同时渲卡片与周索引（**已休眠**，默认不做；见模块 docstring——"
                         "恢复它必须同时恢复 `weekly_sections.py`，否则那两页上的"
                         "「跟着跑一次就补上」永不兑现）")
    ap.add_argument("--limit", type=int, default=0,
                    help="只渲前 N 篇（开发期冒烟；0 = 全量）。**只对 `--with-cards` 有意义**")
    ap.add_argument("--force", action="store_true",
                    help="无视指纹全部重渲。**只对 `--with-cards` 有意义**（合订本没有指纹）")
    args = ap.parse_args()

    cfg, _fetcher = _setup(False, "weekly")       # 渲染不发请求，fetcher 用不上
    weekly_dir = ROOT / cfg["paths"]["data"]      # 周报库落点：绝对路径，见 cli.py::_WEEKLY_PATHS

    result: dict = {}
    if args.with_cards:
        result.update(weekly_card.render(weekly_dir, cfg, force=args.force,
                                        limit=args.limit or None))
        # 周索引**不跟着 `--limit` 截断**：它是这一周的账，截一半会变成一份说少了的账，
        # 而"少"和"就是这样"在页面上长得一模一样。
        result["index"] = weekly_card.render_index(weekly_dir, cfg)
        # 恒等式（铁律 7）：每篇目标要么这次动过、要么指纹命中跳过，两者之和必须等于目标数。
        # ⚠️ `fresh` **已经在 `cards` 里**（是同一个 Counter 的一个键），不要再加一次——
        # 加了它恒等式会变成 `2×fresh + 其余 == targets`，只在 fresh=0 时碰巧成立，
        # 而这种"大部分时候是绿的"检查正是铁律 7 点名要防的。
        result["identity_all_accounted"] = (
            sum(result["cards"].values()) == result["targets_scope"])
    else:
        # 明写出来，而不是让这三个键"就是不出现"——**"不在"和"没跑"长得一样**
        # （本库的头号病）。下一个看汇总行的人应该一眼看出这是有意的。
        result["cards_skipped"] = "2026-09-19 起默认不渲；--with-cards 可恢复（见 docstring）"

    # 合订本**同样不跟着 `--limit` 截断**，理由与索引那一行完全相同。
    # ⚠️ 它和卡片走的是**两套受管键**（`DIGEST_KEYS` vs `CARD_KEYS`）——传错不会报错，
    # 只会让那些计数冻在第一次写的值上（`INDEX_KEYS` 那一课）。
    result["digest"] = weekly_card.render_digest(weekly_dir, cfg)
    print(json.dumps(result, ensure_ascii=False))

    # `digest_fm_mismatch` / `digest_identity_bad` / `digest_undated` 非空 =
    # 合订本**少印了东西**，而"少印"在页面上表现为"它就是不在"——目录里没有、
    # 正文里也没有，**看不出缺**。而这一份是要整份喂给模型的：少印的那几篇，
    # 模型不但读不到，还会把剩下的当成全部。
    #
    # `digest_undated`（这篇的 `first_seen` 是空的 ⇒ 不知道印在哪一天 ⇒ 干脆没印）
    # **是卡片那侧 `no_week` 的等价物**——所以合订本单独跑时，那条恒等式不会漏掉。
    # （`no_week` 原来的话：有篇 `first_seen` 是空的、**没有落盘**。那不是"跳过"，
    #  是"我们不知道该写进哪一周"。）
    # ⚠️ `index_fm_mismatch` 见 `weekly_card.INDEX_KEYS`：2026-09-18 那两个计数
    # 冻了整整一轮，正文一直是新的、frontmatter 一直是旧的，**退出码 0、没有一行变红**。
    # 它只在 `--with-cards` 时才存在。
    bad = (result["digest"]["digest_fm_mismatch"]
           or result["digest"]["digest_identity_bad"]
           or result["digest"]["digest_undated"])
    if args.with_cards:
        bad = bad or result["no_week"] or result["index"]["index_fm_mismatch"]
    return 1 if bad else 0


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
