"""带限速与落盘缓存的 HTTP 客户端。

设计约束（见 CLAUDE.md token 纪律）：

* 所有响应**落盘 cache/**，重复解析不重新抓取，也就不需要为了"看一眼数据长什么样"
  而把响应体回显到终端。
* 本模块**从不打印响应体**。调用方同样只应输出聚合摘要。
* 每个 host 独立限速 + 指数退避。抓挂 arXiv 或顶会站点是真实风险。
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from . import atomic
from .diskguard import DEFAULT_MIN_FREE_GB, guard

log = logging.getLogger(__name__)

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

#: 各 host 的请求间隔（秒）。arXiv 特别保守——它对无节制抓取会封 IP。
DEFAULT_HOST_DELAY = 1.0
HOST_DELAYS = {
    "api.datacite.org": 0.3,         # 免费无 key，但保持克制
    "proceedings.mlr.press": 0.8,
    "mlanthology.org": 0.8,
    "raw.githubusercontent.com": 0.3,
    # arXiv：**实测 2.0 秒会被限流**。2026-09-13 跑 fetch-pdf 时连续请求
    # /pdf/，被边缘节点 RST（WinError 10054，换 UA 无效），持续约 20 分钟，
    # 期间连 arxiv.org 首页都连不上。属于短时限流而非长效封禁，但足以
    # 让一批任务全废。4.0 秒 ≈ 900 次/小时，留足余量。
    "arxiv.org": 4.0,
    # OAI-PMH：**另一台主机**（`oaipmh.arxiv.org`），与 arxiv.org 的限流分开算
    # （2026-09-15 实测：当日对 arxiv.org 前序请求为 0 时，export.arxiv.org/api
    # 仍返回 429 ⇒ 不是同一个计数器）。取许可走 `GetRecord`，一篇一请求，
    # 所以间隔直接决定日更的墙钟时间：4.0 秒 ⇒ 约 900 篇/小时。
    # 不因为"反正另一台"就调小——arXiv 的 OAI 接口对高频访问也会 slowDown，
    # 而那是 HTTP 200 里的 <error>，看不到就会静默变成"今天没有许可"。
    "oaipmh.arxiv.org": 4.0,
    # OpenAlex：**限速不是瓶颈，配额才是**——每次请求固定 10 credits、
    # 日额度 1000，即 100 请求/天，与查询大小无关（单标题和 50 标题同价）。
    # 所以调用方必须批量，见 rllib/adapters/openalex.py 的 chunk_titles 与
    # rllib/citations.py；单篇查是纯浪费，会把 2.6 天的活拖成 131 天。
    "api.openalex.org": 1.0,
}

#: 连续多少次**网络层**失败就判定为被限流/封锁，中止整批。
#:
#: 为什么需要这个：没有它，一次封锁会表现成"每篇都失败"——13000 篇 ×
#: 每次重试 3 遍 × 6 秒 ≈ 21 小时，跑完还是一片 failed，而日志里看不出
#: 出了什么系统性问题。把"单篇取不到"和"整站不通"区分开是这里的关键。
#:
#: 只统计连接被拒/超时这一类；404/403 不算——那是单篇的正常结果
#: （论文确实没有，或该篇受访问限制），累计再多也不代表站点在挡我们。
BREAKER_THRESHOLD = 10

#: 各 host 的单次请求超时（秒）。不设则用 config.yaml 里的全局值。
#:
#: **这个表是有实际代价的，不是调优洁癖。** 超时会**乘以重试次数**：一个
#: API 请求若无节制地用 60 秒超时，撞上三次重试就是单篇 180 秒。实测
#: DataCite 全量解析因此从预期的 1.5 小时劣化到 15 小时——平均值被少数
#: 卡死的请求整个拖垮。
#:
#: 判据是"正常响应有多大"：小型 JSON API 接口 20 秒足够；PDF 和索引页
#: 才是真的需要长超时，所以它们不在此表内，沿用配置里的全局值。
HOST_TIMEOUTS = {
    "api.datacite.org": 20.0,
    "api.openalex.org": 20.0,
    "raw.githubusercontent.com": 30.0,
    # OAI-PMH 单条响应实测 2,790 字节，传输本身是毫秒级；这个超时量的是
    # **服务端把它拼出来要多久**。取 30 而不是全局 60，是为了挡住卡死：
    # 一篇一请求，全局值 × 3 次重试 = 单篇 180 秒，几百篇的批会从十几分钟
    # 劣化成一天（DataCite 那次 1.5 小时变 15 小时就是同一个形状，
    # 见上面 HOST_TIMEOUTS 开头那段）。
    "oaipmh.arxiv.org": 30.0,
}

#: ``httpx.Client`` 底层是一个**长命连接池**，而它会悄悄劣化。
#:
#: 2026-09-16 两次独立观察到同一形状：进程跑到 45 分钟（19:21 那次）与
#: 56 分钟（20:17 那次）时，下载速率从 ~1.3 MB/s 掉到 **0.008 MB/s**，
#: 而**同一时刻**新建一条连接、打同一个 URL，跑出 **2.72 MB/s**——
#: 差 340 倍。判别点就在"同时"两个字：**如果限流发生在 IP 上，新连接
#: 也该慢**。它没有。所以劣化落在这个 client 对象上，不在链路、不在站点
#: 限流、不在代理，也不在"梯子没用"。
#:
#: 熔断器（`HostBlocked`）**抓不到它**：连接没断，HTTP 仍然 200，
#: 只是一滴一滴地流。它不报错、不计失败、不触发任何告警，
#: 唯一的表现是"今天好像有点慢"。
#:
#: 所以按**年龄**轮换，而不是等它报错。20 分钟远短于实测的 45 分钟劣化点，
#: 一次换连接的代价是重开一条 TCP + TLS，与它买回来的东西不成比例。
DEFAULT_CLIENT_MAX_AGE = 1200.0        # 秒；0 或 None 表示不轮换


def _declared_too_big(resp: httpx.Response, max_bytes: int) -> bool:
    """Content-Length 明说超限。分块传输没有这个头，只能下了才知道。"""
    declared = resp.headers.get("content-length")
    return bool(declared and declared.isdigit() and int(declared) > max_bytes)


class HostBlocked(RuntimeError):
    """连续多次网络层失败，判定为被限流或封锁。

    继承 ``RuntimeError`` 而不是自定义的 BaseException，是为了让它能穿过
    各层调用栈被 CLI 捕获，同时**不**被 ``except httpx.HTTPError`` 之类的
    窄捕获吞掉。
    """


class _RateLimiter:
    """按 host 串行化 + 强制最小间隔。线程安全，为后续并发预留。"""

    def __init__(self) -> None:
        self._last: dict[str, float] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock_for(self, host: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(host, threading.Lock())

    def wait(self, host: str) -> None:
        delay = HOST_DELAYS.get(host, DEFAULT_HOST_DELAY)
        with self._lock_for(host):
            elapsed = time.monotonic() - self._last.get(host, 0.0)
            if elapsed < delay:
                time.sleep(delay - elapsed)
            self._last[host] = time.monotonic()


class Fetcher:
    """限速 + 缓存 + 重试。进程内复用一个实例。"""

    def __init__(
        self,
        cache_dir: Path,
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
        user_agent: str = DEFAULT_UA,
        min_free_gb: float = DEFAULT_MIN_FREE_GB,
        client_max_age: float | None = DEFAULT_CLIENT_MAX_AGE,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.timeout = timeout
        #: 每次写盘前要求的最低可用空间（GB）。见 :mod:`rllib.diskguard`。
        self.min_free_gb = min_free_gb
        #: 连接池最长存活秒数，到点换一条新的。见 :data:`DEFAULT_CLIENT_MAX_AGE`。
        self.client_max_age = client_max_age
        self._limiter = _RateLimiter()
        self._user_agent = user_agent
        self._client = self._make_client()
        self._client_born = time.monotonic()
        #: 已退休、但还没 close 的旧 client。见 :meth:`_get_client`。
        self._retired: list[httpx.Client] = []
        #: ``failed`` = **总失败数**（含网络层耗尽与终态 4xx），语义不变。
        #:
        #: ``not_found`` = 其中**终态 4xx**（403/404/410）的那一部分，也就是
        #: "这个 URL 确实没有"，不是"我们没读到"。**它是 failed 的子集**，
        #: 不是另一类——所以 ``failed`` 不扣减，两边都记。
        #:
        #: 为什么要分：返回值都是 ``None``，只有 DEBUG 日志分得开，于是
        #: "arXiv 没有源码包"和"网络没抓到"在上层看起来一模一样。本库反复
        #: 栽的那类病就是它——2026-09-17 实测 ``2605.06850`` 的
        #: ``/src/`` 是 **404**，却被记成了"下载失败"。
        self.stats = {"hit": 0, "fetched": 0, "failed": 0, "not_found": 0}
        self._net_failures = 0
        #: 换过几次连接。长任务报告用——它涨得快说明链路不稳。
        self.client_rotations = 0

    # ------------------------------------------------------------------ #

    def _make_client(self) -> httpx.Client:
        return httpx.Client(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": self._user_agent},
        )

    def _get_client(self) -> httpx.Client:
        """取当前 client；**年龄超过 ``client_max_age`` 就换一条新连接**。

        为什么要有这个方法（而不是直接用 ``self._client``）：见
        :data:`DEFAULT_CLIENT_MAX_AGE` —— 连接池会悄悄劣化到 1/340 的速率，
        而熔断器对"没断、只是慢"完全无感。

        换下来的旧 client **不当场 close**：:meth:`stream_to` 可能正握着它的
        一条流（``with ... as resp`` 还没退出），当场关掉会把那条流掐断。
        退休表里**只留最近退休的一个**，更早的在这里 close —— 它已退休满一个
        完整轮换周期，而任何真实请求都受 ``HOST_TIMEOUTS`` / ``timeout`` 约束，
        早在几十秒内就该结束了。
        """
        if not self.client_max_age:
            return self._client
        if time.monotonic() - self._client_born < self.client_max_age:
            return self._client

        self._retired.append(self._client)
        self._client = self._make_client()
        self._client_born = time.monotonic()
        self.client_rotations += 1
        log.info("连接已轮换（第 %d 次，上一个存活 %.0f 秒）",
                 self.client_rotations, self.client_max_age)
        while len(self._retired) > 1:
            self._retired.pop(0).close()
        return self._client

    # ------------------------------------------------------------------ #

    def _record_failure(self, host: str, *, network: bool) -> None:
        """记一次失败。``network=True`` 表示连接被拒/超时，参与熔断计数。

        单篇取不到（404、论文确实没有）不该把整批拖停，所以只有网络层失败
        才累加。真正的封锁长这样：每一篇都是 ConnectError，累计到阈值就说明
        不是论文的问题，是站点在挡我们。
        """
        if not network:
            return
        self._net_failures += 1
        if self._net_failures >= BREAKER_THRESHOLD:
            raise HostBlocked(
                f"{host}：连续 {self._net_failures} 次网络层失败，判定为被限流或封锁，已中止整批。"
                f"继续跑只会把剩下的请求全打水漂。等 20–30 分钟再试，"
                f"并确认 HOST_DELAYS 里 {host} 的间隔是否仍然过激进。"
            )

    def _record_success(self) -> None:
        self._net_failures = 0

    def _cache_path(self, url: str, suffix: str) -> Path:
        return self.cache_dir / f"{hashlib.sha1(url.encode()).hexdigest()}{suffix}"

    @staticmethod
    def _write_cache(path: Path, content: bytes) -> None:
        """先写 ``.tmp`` 再原子替换。

        **不能就地 ``write_bytes``。** 缓存是"重解析不重抓取"的唯一依据，
        断电/强杀时留下一个**截断的**缓存文件，后果比丢缓存严重得多：
        之后每次重试都读到同一个坏文件，永远不会自愈。若被截断的是 HTML 单篇页，
        摘要有可能被静默截半写进库——一份看起来正常、实际残缺的记录。

        跨进程同名 URL 的竞争是良性的：同一 URL 的内容必然相同，谁先替换都一样。
        """
        atomic.write_atomic(path, content)

    def get_bytes(
        self,
        url: str,
        *,
        suffix: str = ".bin",
        refresh: bool = False,
    ) -> bytes | None:
        """取原始字节。命中缓存则直接返回，不产生网络请求。

        返回 ``None`` 表示重试耗尽仍失败——**不要抛异常**，因为单篇论文
        取不到不应中断整个批量流程，缺口的可见性由质量报告负责。
        """
        path = self._cache_path(url, suffix)
        if path.exists() and not refresh:
            # 命中缓存**刻意不重置熔断计数**：这一步没有发生任何网络接触，
            # 证明不了站点是通的。有人会想"顺手调一下 _record_success"——
            # 那会让熔断器在断网时被缓存喂着永远不触发。
            # （反面也踩过：写测试时忘了 refresh=True，走缓存的"成功"没有
            # 重置计数，检查报失败而熔断器完全正常。见 scripts/check_breaker.py。）
            self.stats["hit"] += 1
            return path.read_bytes()

        host = urlparse(url).hostname or ""
        last_err: Exception | None = None

        timeout = HOST_TIMEOUTS.get(host, self.timeout)

        for attempt in range(self.max_retries):
            # 在**发请求之前**刹车，不是写盘之前：空间不够时下载本身就是浪费，
            # 而且写盘那一刻才发现，那一次请求已经打出去了。
            guard(self.cache_dir, self.min_free_gb)
            self._limiter.wait(host)
            try:
                resp = self._get_client().get(url, timeout=timeout)
                if resp.status_code == 200:
                    self._write_cache(path, resp.content)
                    self.stats["fetched"] += 1
                    self._record_success()
                    return resp.content
                if resp.status_code in (403, 404, 410):
                    # 换再多姿势也没用，快速失败，省得把 token 烧在重试上
                    self.stats["not_found"] += 1     # 终态：这个 URL 确实没有
                    log.debug("HTTP %s (终态，不重试): %s", resp.status_code, url)
                    break
                last_err = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
                if resp.status_code in (429, 503):
                    self._record_failure(host, network=True)   # 明确在限我们
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_err = exc
                self._record_failure(host, network=True)

            if attempt < self.max_retries - 1:
                time.sleep(2**attempt)

        log.debug("取不到 %s: %s", url, last_err)
        self.stats["failed"] += 1
        return None

    def get_text(self, url: str, *, refresh: bool = False) -> str | None:
        raw = self.get_bytes(url, suffix=".html", refresh=refresh)
        return raw.decode("utf-8", errors="replace") if raw else None

    def stream_to(
        self,
        url: str,
        dest: Path,
        *,
        magic: bytes | None = None,
        timeout: float | None = None,
        max_bytes: int | None = None,
    ) -> int | None:
        """流式下载到 ``dest``，**不进 cache**，返回字节数（失败返回 None）。

        为什么不复用 :meth:`get_bytes`：PDF 有 8GB 量级，落进 ``cache/`` 再拷到
        ``pdfs/`` 会白占一倍磁盘。而这里的产物本身就是最终文件，不需要二次缓存。

        限速与重试仍然复用同一套逻辑——arXiv 对无节制抓取会封 IP，
        绕过限速比多占点磁盘危险得多。

        ``magic`` 是要校验的文件头（如 ``b"%PDF-"``）：很多站点出错时返回的是
        一个 HTTP 200 的 HTML 错误页，不校验就会把一堆 HTML 存成 ``.pdf``，
        而且要到几个月后真正去读的时候才发现。

        先写 ``.part`` 再原子改名，所以中断不会留下半个看着像成功的文件。
        """
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        host = urlparse(url).hostname or ""
        use_timeout = timeout if timeout is not None else HOST_TIMEOUTS.get(host, self.timeout)
        last_err: Exception | None = None

        for attempt in range(self.max_retries):
            # 量的是**落盘目录**所在卷，不是 cache：PDF 可能被配到别的盘上。
            guard(dest.parent, self.min_free_gb)
            self._limiter.wait(host)
            try:
                with self._get_client().stream("GET", url, timeout=use_timeout) as resp:
                    if resp.status_code in (403, 404, 410):
                        # 终态，重试无意义
                        part.unlink(missing_ok=True)
                        self.stats["failed"] += 1
                        self.stats["not_found"] += 1     # 与 get_bytes 同一口径
                        log.debug("HTTP %s (终态): %s", resp.status_code, url)
                        return None
                    elif resp.status_code in (429, 503):
                        last_err = httpx.HTTPStatusError(
                            f"HTTP {resp.status_code}", request=resp.request, response=resp
                        )
                        self._record_failure(host, network=True)   # 明确在限我们
                    elif resp.status_code != 200:
                        last_err = httpx.HTTPStatusError(
                            f"HTTP {resp.status_code}", request=resp.request, response=resp
                        )
                    elif max_bytes is not None and _declared_too_big(resp, max_bytes):
                        # 先看 Content-Length 再决定要不要下：扫描版和附件包能到几百 MB，
                        # 下完再删等于白占带宽和几分钟
                        part.unlink(missing_ok=True)
                        self.stats["failed"] += 1
                        log.debug("文件过大（Content-Length 超限）: %s", url)
                        return None
                    else:
                        head = b""
                        size = 0
                        oversized = False
                        with part.open("wb") as fh:
                            for chunk in resp.iter_bytes(1 << 16):
                                if size == 0:
                                    head = chunk[:16]
                                fh.write(chunk)
                                size += len(chunk)
                                if max_bytes is not None and size > max_bytes:
                                    oversized = True   # 没给 Content-Length 的那种
                                    break
                        if oversized:
                            part.unlink(missing_ok=True)
                            self.stats["failed"] += 1
                            log.debug("文件过大（流式超限）: %s", url)
                            return None
                        if magic is not None and not head.startswith(magic):
                            # 拿到的是 HTML 错误页之类，不是真文件。删掉并记为失败，
                            # 否则它会一直"存在"，让后续的跳过逻辑误以为已经下好了。
                            part.unlink(missing_ok=True)
                            self.stats["failed"] += 1
                            log.debug("文件头不对（疑似错误页）: %s", url)
                            return None
                        atomic.replace(part, dest)
                        self.stats["fetched"] += 1
                        self._record_success()
                        return size
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_err = exc
                self._record_failure(host, network=True)

            part.unlink(missing_ok=True)
            if attempt < self.max_retries - 1:
                time.sleep(2**attempt)

        log.debug("下载失败 %s: %s", url, last_err)
        self.stats["failed"] += 1
        return None

    def close(self) -> None:
        self._client.close()
        for old in self._retired:
            old.close()
        self._retired.clear()

    def __enter__(self) -> Fetcher:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
