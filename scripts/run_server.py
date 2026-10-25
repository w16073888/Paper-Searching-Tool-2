"""Web 服务入口：`python scripts/run_server.py`（前端走 HTTP，不再用 Streamlit）。

示例：
    python scripts/run_server.py                       # http://127.0.0.1:8000
    python scripts/run_server.py --port 8080 --web web  # 同时托管 web/ 静态页
    python scripts/run_server.py --config 测试数据集/config.yaml   # 用测试数据集调试

接口只有两条：`GET /api/meta`、`POST /api/ask`（+ `GET /api/health`）。
契约见 `hotspot/api/contract.py`，字段表见 `规划/2026-10-18-接口契约-前端与算法.md`。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hotspot  # noqa: E402 —— 装闸① + 接线 dasc
from hotspot.api.engine import Engine  # noqa: E402
from hotspot.api.http import serve  # noqa: E402
from hotspot.config import load_config  # noqa: E402


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="移动云杯热点挖掘 · HTTP 服务")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（对外给 0.0.0.0）")
    ap.add_argument("--port", type=int, default=8000, help="端口（默认 8000）")
    ap.add_argument("--web", default="web", help="静态文件目录（默认 web/，不存在则只提供 API）")
    ap.add_argument("--config", default=None, help="配置文件（默认项目根 config.yaml）")
    args = ap.parse_args()

    cfg_path = Path(args.config) if args.config else None
    if cfg_path is not None and not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    engine = Engine(load_config(cfg_path)) if cfg_path else Engine()

    web = Path(args.web)
    if not web.is_absolute():
        web = root / web
    serve(args.host, args.port, engine=engine,
          static_dir=web if web.is_dir() else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
