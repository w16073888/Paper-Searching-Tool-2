"""控制台输出编码。

单独成模块，是因为 CLI 和 ``scripts/`` 下的检查脚本都要用，
散在各处迟早会漏掉一个，然后表现为"某个命令莫名乱码"。

**这个函数只管终端显示。** 真正要保真的地方（``data/papers.jsonl``、
Obsidian 笔记）一律显式 ``encoding="utf-8"``，不依赖它。
"""

from __future__ import annotations

import sys


def use_utf8_console() -> None:
    """把标准输出/错误强制成 UTF-8，让中文和 emoji 都不出事。

    两个真实故障，都实测踩到过：

    1. **崩溃**：Windows 下 Python 的 stdout 默认跟随控制台代码页（这里是
       cp936/GBK）。论文标题里有 emoji（NeurIPS 某篇含 🎧）和大量数学符号，
       GBK 编不出来，``UnicodeEncodeError`` 直接把命令打断。
    2. **乱码**：只把 errors 改成 replace 而不改编码，在 Git Bash / Windows
       Terminal 这类 UTF-8 终端里，GBK 字节会被按 UTF-8 解码，中文全成问号。

    所以正解是**改编码为 UTF-8**，并用 ``errors="replace"`` 兜住任何意外字符：
    宁可某个生僻字退化成 ``?``，也不能让整条命令死在编码上。

    真控制台上再顺手把输出代码页切成 65001，免得 cmd.exe 里显示成乱码。
    对管道/重定向无害——那两种情况下这个调用是空操作。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass

    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:  # noqa: BLE001  切不动就算了，上面已经保证能编出来
            pass
