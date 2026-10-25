"""周报 concern 抽取（产线）：对周报库里**在卡片范围内的**每一篇，抽作者自述的展望 / 缺口。

⚠️ **2026-09-19 用户拍：这一步休眠，别跑。**
交付层只取摘要（``concern`` 不进任何对外产物）⇒ 它的产物 ``sections.jsonl`` /
卡片 / ``review.md`` **现在都没有读者**。它是周报线里**最贵的一步**（逐篇下 ``/src/``
源码包：**47 篇 = 174.3 MB / 35 分钟**，09-17 实测；09-18 那轮 89 篇用 48 分钟、
≈32 秒/篇），所以"顺手跑一下"的代价是实打实的。
⚠️ **它和卡片是一对**：`weekly_render.py --with-cards` **必须**与它同时恢复，
否则那两页上明写的「跟着跑一次 `weekly_sections.py` 就补上」**永不兑现**。
✅ **停下来是可逆的**：``papers.jsonl`` 里存着每篇的 ``arxiv_id``，而 ``arxiv.org/src/``
**按 id 永久稳定**（不像 RSS 那样窗口一滚就没了）⇒ 将来要补抽，缺的只是一次批量。
（⚠️ 那是**当前**的事实、不是承诺：主库实测 **3,289 篇根本没有 ``/src/``**。）

范围为 ``is_rl`` ∪ 命中 RL 子方向专有词 = **73 篇/天**（用户 2026-09-17 拍「73 吧」）。
判定的**本体不在这里**，在 ``rllib/weekly_scope.py`` —— 卡片渲染器要读同一条，
两处各写一份正是 probe v2→v3 那轮修的根因。⚠️ 因此**别在脚本里内联这个条件**。

与 `probe_sections_v3.py` 的关系
--------------------------------

探针是**冻结的证据**（每次全跑、落库外、产物是原始 jsonl）；这里是**产线**
（增量、落库内、供下游渲染卡片）。抽取逻辑两边共用库内那一份
（`rllib.extract.gap`），**不抄第二份** —— 理由写在那个模块的开头。

与 `daily_arxiv_fetch.py` 的关系
--------------------------------

日更负责"**今天有什么**"（`papers.jsonl` + 公告类型旁路）；本脚本负责
"**其中 RL 命中的那些，作者自己说了什么没做**"。两步**刻意分开**，因为成本差
两个数量级：日更 3 请求 / 约 12 秒；本脚本 47 篇 **174 MB / 35 分钟**。

⚠️ 它是**打 `arxiv.org` 的长任务**：限速、熔断、磁盘余量刹车全走 `Fetcher`，
退出码与主库长任务一致（**3 = 被限流/封锁，4 = 磁盘余量不足**）。
**不要和 `extract` / `fetch-pdf` 之类打同一个站点的任务同时跑**（CLAUDE.md）。

增量
----

默认只处理**还没有记录**的篇。这样日更之后跟一次，成本就是"今天新增的那几十篇"。

=================  ==========================================================
`--retry-failed`   重试 `failed` 与 `no_source` 的那些
`--redo`           全部重抽（**改了 `latex.py` 的判据之后必须跑这个**）
`--refresh`        绕过 HTTP 缓存重抓源码包（与上两个独立）
=================  ==========================================================

**为什么不把 `no_text` / `no_section` / `degenerate` 也放进 `--retry-failed`**：
那三个是**读到了东西之后的结论**，重试一百次还是同一个结论，要让它们变只能改判据
⇒ 那是 `--redo` 的事。而 `failed`（网络没抓到）与 `no_source`（arXiv 可能事后补上
源码包）是"**我们没拿到东西**"，与结论无关。**混起来就会让"重试"这个词变成
"重跑"，而重跑一次是 174 MB。**

落点
----

.. code-block:: text

   <周报库根>/
     sections.jsonl        累积真相源（按 arxiv_id 键，一篇一行，含抽出的正文）

⚠️ **本脚本不再产出人眼件。** 原来这里还有一份 ``sections_review.md``，由
``_review_md()`` 从**内存里的本次新增**写出——只覆盖最近一次跑的那些篇、无排序。
而"这一周的抽取结果"本就不该由**某一次运行**决定：09-17 那一轮的 47 篇在那份
文件里永远是缺的，**且看不出来**。它已由 ``rllib/render/weekly_review.py`` +
``scripts/weekly_review.py`` 接管（**从磁盘重算本周全部**、按可疑度排）。
**一个文件一个属主** ⇒ 那份删掉了，别再把它加回来。

本脚本现在**只写真相源**，人眼面全在渲染侧。

**Token 纪律**：终端只打一行聚合 JSON，绝不回显标题或正文。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rllib import ROOT  # noqa: E402
from rllib.atomic import write_atomic  # noqa: E402
from rllib.cli import _setup  # noqa: E402
from rllib.console import use_utf8_console  # noqa: E402
from rllib.diskguard import DiskLow  # noqa: E402
from rllib.extract import gap  # noqa: E402
from rllib.http import HostBlocked  # noqa: E402
from rllib.weekly_scope import in_scope, scope_of  # noqa: E402

#: 重试的语义边界（见模块 docstring）。**不是"所有非 ok"**。
RETRYABLE = frozenset({"failed", "no_source"})

SECTIONS_NAME = "sections.jsonl"


def _key(paper: dict) -> str:
    return paper.get("arxiv_id") or paper.get("title_norm") or ""


def _load_targets(weekly_dir: Path) -> list[dict]:
    """周报库里**在卡片范围内**的那些篇 —— ``is_rl`` 或命中子方向专有词。

    ⚠️ 判定的本体在 ``rllib.weekly_scope``，**不在这里**（见模块 docstring）。
    早先这里写的是 ``(r.get("rl") or {}).get("is_rl")``，那是 **47 篇**，
    而用户拍的范围是 **73** —— 那 26 篇盲区会**静默地不出卡片**。
    """
    path = weekly_dir / "papers.jsonl"
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [r for r in rows if in_scope(r)]


def _load_done(path: Path) -> dict[str, dict]:
    """已有的抽取记录。文件不存在 = 第一次跑，**不是出错**。

    ⚠️ 但**损坏**要炸出来：一个截断的 jsonl 会让 ``get`` 全部落空、
    于是静默地把每一篇都当成"没抽过"重抽一遍（174 MB），而结果看起来完全正常。
    """
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not ln.strip():
            continue
        try:
            row = json.loads(ln)
        except ValueError:
            raise SystemExit(f"{path} 第 {i} 行不是合法 JSON —— 拒绝在损坏的真相源上续跑") from None
        out[row["arxiv_id"]] = row
    return out


def main() -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser(
        description="周报 concern 抽取（增量）。只输出聚合摘要。",
    )
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 篇（开发期冒烟；0 = 全量）")
    ap.add_argument("--retry-failed", action="store_true",
                    help="重试 %s（见模块 docstring 的语义边界）" % sorted(RETRYABLE))
    ap.add_argument("--redo", action="store_true", help="全部重抽（改了判据之后必须跑）")
    ap.add_argument("--refresh", action="store_true", help="绕过 HTTP 缓存重抓源码包")
    args = ap.parse_args()

    cfg, fetcher = _setup(False, "weekly")
    weekly_dir = ROOT / cfg["paths"]["data"]          # 周报库落点：绝对路径，见 cli.py::_WEEKLY_PATHS
    sections_path = weekly_dir / SECTIONS_NAME

    targets = _load_targets(weekly_dir)
    done = _load_done(sections_path)

    if args.redo:
        pending = targets
    else:
        retryable = RETRYABLE if args.retry_failed else frozenset()
        pending = [p for p in targets
                   if _key(p) not in done or done[_key(p)].get("outcome") in retryable]
    # ⚠️ `already_done` 必须在 `--limit` **之前**算。截断之后再算，
    # 那 44 读起来是"已经抽过 44 篇"，实际是"被 --limit 砍掉 44 篇"——
    # 一个说反了的诊断数比没有诊断更糟：它会让人以为增量生效了。
    already_done = len(targets) - len(pending)
    if args.limit:
        pending = pending[: args.limit]

    today = date.today().isoformat()
    new_rows: list[dict] = []
    for paper in pending:
        row = gap.extract_one(paper, fetcher)
        row["extracted"] = today
        new_rows.append(row)

    # 合并落盘：**以 arxiv_id 为键 upsert**，本次没碰的篇一行不动。
    merged = dict(done)
    for row in new_rows:
        merged[row["arxiv_id"]] = row
    # 只写真相源。人眼面（周验收页）改由 `scripts/weekly_review.py` 从**磁盘**
    # 重算——本脚本在内存里只有"本次新增"，写不出这一周的账（见模块 docstring）。
    write_atomic(
        sections_path,
        "".join(json.dumps(merged[k], ensure_ascii=False, separators=(",", ":")) + "\n"
                for k in sorted(merged)).encode("utf-8"),
    )

    outcomes: Counter[str] = Counter(r["outcome"] for r in new_rows)
    # 恒等式一（铁律 7）：本次处理的每一篇都必须落进**已登记**的桶里。
    # 不成立就说明有人加了个新桶却没登记进 `gap.OUTCOMES`（或写错了字）。
    unknown = sorted(set(outcomes) - set(gap.OUTCOMES))
    # 恒等式二：目标全集里还有多少篇**没有记录**。它**必须被报出来**——
    # 沉默的缺口看起来和"全都抽完了"一模一样。
    covered = sum(1 for p in targets if _key(p) in merged)
    # ⚠️ 名字从 `stale` 改成了 `retryable_*`：它就是 `RETRYABLE = {failed, no_source}`
    # **这两个桶的篇数**，**不是缺陷计数**。`no_source` 是**关于 arXiv 的事实**（那篇
    # 真没交源码包），跟"陈旧"没关系——而 `stale` 这个名字会被读成"这几篇有问题"，
    # 正是本库最不许混的那件事（「没有」与「没读到」）。
    n_retryable = sum(1 for p in targets
                      if _key(p) in merged and merged[_key(p)].get("outcome") in RETRYABLE)

    # ⚠️ 键名从 `targets_rl` / `covered_rl` 改成了不带 `_rl` 的：
    # 范围是 **73**（is_rl ∪ 专有词），不是 47。**留旧名字会让读的人以为是 47**
    # —— 一个说错的名字和一个说错的数一样糟（本库栽过好几回）。
    # `scope_mix` 是那条判定的**证据面**：它把"为什么收这篇"变成可见的数，
    # 而不是只有一个 73。三项之和必须等于 `targets_scope`。
    scope_mix: Counter[str] = Counter(scope_of(r) for r in targets)
    print(json.dumps({
        "targets_scope": len(targets),
        "scope_mix": dict(sorted(scope_mix.items())),   # both / is_rl / subfield
        "already_done": 0 if args.redo else already_done,
        "processed": len(new_rows),
        "outcomes": dict(outcomes),
        "unknown_outcomes": unknown,
        "identity_outcomes_ok": not unknown and sum(outcomes.values()) == len(new_rows),
        "covered": covered,
        "uncovered": len(targets) - covered,           # 还没抽的（含本次之后剩下的）
        # 抽过、但结论是"我们没拿到"（`failed` / `no_source`）——**不是缺陷计数**。
        "retryable_outcomes": n_retryable,
        "sections_total": len(merged),
        "http": dict(fetcher.stats),
        "data_dir": str(weekly_dir),
    }, ensure_ascii=False))

    # 有篇失败就不算成功——否则调用方以为这批都抽过了。
    return 1 if outcomes.get("failed") else 0


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
