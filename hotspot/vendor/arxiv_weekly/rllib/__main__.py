"""允许 ``python -m rllib ...`` 直接调用 CLI。"""

from .cli import app
from .diskguard import DiskLow
from .http import HostBlocked


def _abort(tag: str, code: int, exc: Exception) -> None:
    """干净的提示 + 专用退出码，而不是甩一段 traceback。

    熔断和余量不足都不是崩溃，是**有意的**中止：继续跑只会把剩下的请求
    / 磁盘全打水漂。已经落盘的部分是完整的（长任务都分段保存），重跑即续。
    """
    import json
    import sys

    print(json.dumps({"error": tag, "detail": str(exc)}, ensure_ascii=False),
          file=sys.stderr)
    raise SystemExit(code) from None


def main() -> None:
    # 退出码：3 = 被限流/封锁（等 20–30 分钟或换网络）
    #         4 = 磁盘余量不足（该把 cache 搬到移动硬盘了，见 STORAGE.md）
    try:
        app()
    except HostBlocked as exc:
        _abort("host_blocked", 3, exc)
    except DiskLow as exc:
        _abort("disk_low", 4, exc)


if __name__ == "__main__":
    main()
