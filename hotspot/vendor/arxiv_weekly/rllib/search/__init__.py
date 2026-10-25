"""本地检索工具 —— 课题组用的那一层。

库是原料，这里是产品。设计动机与取舍见 ``PLAN.md`` 阶段 8。
"""

from .index import INDEX_NAME, SCHEMA_VERSION, build, connect
from .query import Hit, search

__all__ = ["INDEX_NAME", "SCHEMA_VERSION", "build", "connect", "Hit", "search"]
