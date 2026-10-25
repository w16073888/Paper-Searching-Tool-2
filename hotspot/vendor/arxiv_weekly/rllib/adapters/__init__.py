"""数据源适配器。

每个模块负责把一个源解析成 :class:`rllib.schema.CanonicalPaper` 列表。
适配器**不负责去重、不负责合并、不写盘**——那些归 :mod:`rllib.store`。

现有适配器：
- :mod:`rllib.adapters.pmlr`        ICML / CoRL，索引页 1 请求拿全量
- :mod:`rllib.adapters.mlanthology` NeurIPS / ICLR（阶段 3）
- :mod:`rllib.adapters.openalex`    富化，非发现源（阶段 2）
- ⚠️ **原有两个 arXiv 适配器 2026-09-20 已随周报线搬走，本库那份已删**：
  ``arxiv_rss``（arXiv 每日公告流，见 ROADMAP 6.4 W）与 ``arxiv_oai``
  （OAI-PMH，**只取再分发许可**——``<license>`` 不在搜索 API 的 schema 里，
  那是唯一的入口）。两者现住
  ``D:/大学编程学习/arXiv周报/程序/rllib/adapters/``，调用方
  ``rllib/weekly_license.py`` 与 ``scripts/daily_arxiv_fetch.py`` 也一并搬走了。
  ⇒ **本库从此没有 arXiv 适配器**；D 层要用（``STATUS.md`` / ``ROADMAP.md``
  里那条「D 层解禁后复用」的计划）得去上面那个新路径取。
  两者都**从未进过** :data:`REGISTRY`（那个注册表是死代码，分派在 ``cli.py``
  里写死），挂上去会让人以为 ``rllib sync`` 能同步 arXiv。

RLC 2025 未纳入：其论文集在 Reinforcement Learning Journal (rlj.cs.umass.edu)，
该站点在本网络实测 000 不可达，mlanthology 亦无收录。
"""

from __future__ import annotations

from typing import Callable

from ..http import Fetcher
from ..schema import CanonicalPaper

#: adapter 名 -> 取数函数
REGISTRY: dict[str, Callable[..., list[CanonicalPaper]]] = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def get(name: str):
    """按名取适配器。未实现的模块会被跳过而不是让整个包导入失败。"""
    import importlib

    for mod in ("pmlr", "mlanthology", "openalex"):
        try:
            importlib.import_module(f".{mod}", __name__)
        except ImportError:
            pass
    if name not in REGISTRY:
        raise KeyError(f"未知 adapter: {name}（可用: {sorted(REGISTRY)}）")
    return REGISTRY[name]


__all__ = ["REGISTRY", "Fetcher", "CanonicalPaper", "get", "register"]
