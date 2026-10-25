"""下载时的磁盘余量刹车。

**为什么需要**：2026-09-14 实测，``extract`` 一路写到
``OSError: [Errno 28] No space left on device`` 才停。那一刻 D: 是 **0 字节可用**——
连一个 3.4 MB 的校验账本都写不进去，一次跑了 40 分钟的哈希校验带着成果一起崩掉。
**"跑到满"不是一种停止方式**，它把单点故障扩散成了所有在 D: 上写盘的东西的故障。

这里给出一条明确的红线：**低于阈值就主动中止批次**（``DiskLow``，退出码 4）。
中止是干净的——长任务都分段落盘，重跑即续。

判定为什么是 ``阈值 + WRITE_HEADROOM`` 而不是光看阈值
----------------------------------------------------

刹车必须在缓冲区被吃掉**之前**踩下去。只看"还剩不到 1 GB 就停"的话，
检查通过之后那一次写盘还会再吃掉最多 :data:`WRITE_HEADROOM` 字节，
结果就是"说好留 1 GB，实际留了 0.94 GB"。

所以判据是 ``free < min_free_gb + WRITE_HEADROOM`` 才停，
**这样实际余量永远不会低于配置的红线**。
"""

from __future__ import annotations

import shutil
from pathlib import Path

#: 单次写盘的量级上界，用来把刹车点提前到缓冲线之上。
#:
#: 实测最大的 ``.src`` 是 31.2 MB（``cache/fa55e522….src``），正文 ``.txt.gz``
#: 再加几 MB；PDF 则更大。取 64 MB 是留一倍余量——**宁可早停一轮，
#: 不要在下一次写盘时把缓冲区空掉**。早停的代价只是下一轮多搬一次。
WRITE_HEADROOM = 64 << 20

#: 配置缺失时的默认红线（GB）。真实值在 config.yaml 的 ``disk.min_free_gb``。
DEFAULT_MIN_FREE_GB = 1.0


class DiskLow(RuntimeError):
    """可用空间低于红线。

    继承 ``RuntimeError`` 的理由同 :class:`rllib.http.HostBlocked`：能穿过各层
    调用栈被 CLI 捕获，又**不会**被 ``except OSError`` 之类的窄捕获吞掉。
    被吞掉就等于没有刹车，而且不会报任何错。
    """


def free_bytes(path: Path) -> int | None:
    """``path`` 所在卷的可用字节数。

    取不到返回 ``None``——**不谎报 0**。盘掉线时 ``disk_usage`` 抛 OSError，
    而"盘不见了"和"盘满了"的处置完全不同（前者要换线换口，后者要搬数据）。
    把前者报成后者，会让人对着一个满盘找问题。
    """
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


def shortfall(path: Path, min_free_gb: float) -> str | None:
    """够用返回 ``None``；不够返回一句能直接打给人看的说明。"""
    free = free_bytes(path)
    if free is None:
        # 读不到就不断言。真正的 I/O 错误会自己从别处冒出来，
        # 这里硬报一个数字反而会把排查方向带偏。
        return None
    need = int(min_free_gb * (1 << 30)) + WRITE_HEADROOM
    if free >= need:
        return None
    return (
        f"{path} 所在卷只剩 {free / (1 << 30):.2f} GB，"
        f"低于红线 {min_free_gb:.2f} GB（另留 {WRITE_HEADROOM >> 20} MB 给单次写盘）。"
        f"已主动中止，避免写到 ENOSPC 把同一卷上所有写盘的东西一起拖死。"
    )


def guard(path: Path, min_free_gb: float = DEFAULT_MIN_FREE_GB) -> None:
    """低于红线就抛 :class:`DiskLow`。调用点应在**发起写盘之前**。"""
    msg = shortfall(path, min_free_gb)
    if msg:
        raise DiskLow(msg)
