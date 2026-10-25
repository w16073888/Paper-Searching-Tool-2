"""本地 RL 论文库。

数据契约见 :mod:`rllib.schema`，项目约束见仓库根目录的 CLAUDE.md。
"""

__version__ = "0.1.0"

#: 仓库根目录（本包位于 <root>/rllib/）
ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
