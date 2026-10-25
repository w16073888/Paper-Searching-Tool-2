"""原子替换文件，并容忍 Windows 的文件占用。

问题
----

Windows 与 POSIX 在这里有个会咬人的差异：``os.replace`` 底层是
``MoveFileEx(MOVEFILE_REPLACE_EXISTING)``，**只要有任何进程打开着目标文件，
哪怕只是只读，就会失败并抛 ``PermissionError: [WinError 5]``**。
Python 的 ``open()`` 不带 ``FILE_SHARE_DELETE``，所以任何读者都足以挡住替换。

实测后果：``enrich`` 与 ``backfill-abstracts`` 并行跑时 ``papers.jsonl`` 每
20–40 秒被重写一次，只要撞上**不合作的读者**（杀毒软件扫描刚写出的文件、
Windows 搜索索引器、编辑器，或一个随手跑的 ``open(path).readlines()``），
整次落盘就炸——而且是在跑了两小时、写了几千条之后才炸。

注意这**不是** :class:`rllib.store._FileLock` 能解决的问题：锁只串行化了
**写入者**，挡不住一个只是打开文件读一读的外部进程。

做法
----

先写临时文件再替换（保证不会留下半个文件），替换失败就**退避重试**。
重试发生在文件锁内，所以竞争写入者此时正被挡住、不会捣乱；能撞上的只有
外部读者，而它们持有句柄的时间通常是毫秒级。

顺带修掉一个隐患：临时文件名**带 pid**。固定名字（``papers.jsonl.tmp``）下，
万一锁超时被"硬闯"，两个进程会写同一个临时文件，交错的内容被整体替换上去——
一条静默的损坏路径。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

#: 退避序列（秒）。总计约 6 秒——足够等一个杀毒扫描让开，又不至于在锁内卡太久。
_BACKOFF = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.0)


def tmp_for(dst: Path) -> Path:
    """给 ``dst`` 一个**本进程独有**的临时文件名。"""
    return dst.with_name(f"{dst.name}.{os.getpid()}.tmp")


def replace(tmp: Path, dst: Path, *, backoff: tuple[float, ...] = _BACKOFF) -> None:
    """``tmp`` → ``dst`` 的原子替换，被占用时退避重试。

    重试耗尽后**仍然抛出**——静默吞掉会让调用方以为已经落盘，那比崩溃更糟。
    """
    last: PermissionError | None = None
    for delay in (None, *backoff):
        if delay is not None:
            time.sleep(delay)
        try:
            os.replace(tmp, dst)
            return
        except PermissionError as exc:      # WinError 5 / ERROR_SHARING_VIOLATION
            last = exc
    assert last is not None
    raise last


def write_atomic(dst: Path, data: bytes, *, backoff: tuple[float, ...] = _BACKOFF) -> None:
    """一次性把 ``data`` 原子写到 ``dst``。"""
    tmp = tmp_for(dst)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(data)
    replace(tmp, dst, backoff=backoff)


def clean_tmp(dst: Path) -> None:
    """清掉本可能残留的临时文件（中断在写入与替换之间时会有）。"""
    for pattern in (f"{dst.name}.*.tmp", f"{dst.name}.tmp"):
        for stale in dst.parent.glob(pattern):
            stale.unlink(missing_ok=True)
