"""paper_id → 文件名的**唯一**净化规则。

为什么必须集中在一处：``paper_id`` 形如 ``orid:c1Ng0f8ivn`` 或
``doi:10.48550/arxiv.2501.12345``，其中的 ``:`` 和 ``/`` 在 Windows 上都是
非法文件名字符。如果笔记用一种替换规则、PDF 用另一种，同一篇论文在磁盘上就有
两个不同的"身份"，而 ``local_pdf`` 字段里记录的名字会和实际文件对不上——
这种错位会一直潜伏到某天有人真的去打开文件才暴露。

所以全库所有把 paper_id 变成文件名的地方都走这里。
"""

from __future__ import annotations

import re
from pathlib import Path

#: Windows 文件名禁用字符远多于 POSIX；``:`` 尤其致命（paper_id 里就有）。
#: 一律替换成 ``-``，保证在 Windows / macOS / 同步盘上都建得出来。
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_stem(paper_id: str) -> str:
    """把 paper_id 变成不含路径分隔符的裸文件名（无扩展名）。

    ``orid:c1Ng0f8ivn`` → ``orid-c1Ng0f8ivn``
    """
    return _UNSAFE.sub("-", paper_id).strip("-")


def safe_name(paper_id: str, ext: str) -> str:
    """带扩展名的文件名。``ext`` 需自带点，如 ``.pdf``。"""
    return f"{safe_stem(paper_id)}{ext}"


def pdf_path(pdfs_dir: Path, paper_id: str) -> Path:
    return Path(pdfs_dir) / safe_name(paper_id, ".pdf")


def text_path(text_dir: Path, paper_id: str) -> Path:
    return Path(text_dir) / safe_name(paper_id, ".txt.gz")
