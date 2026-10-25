"""CanonicalPaper —— 全库数据契约。

冻结约定（见 CLAUDE.md），改动前务必读完：

1. ``paper_id`` 在**首次插入时确定，此后永不改变**。
   它是所有下游产物（Obsidian 双链、notes 文件名）的锚点，一旦变动就全断。
   因此 upsert 时必须用 :func:`identity_keys` 做多键匹配，而不是靠 paper_id 反查。

2. 扩展 schema 时**只新增可选字段**，不要重命名或删除已有字段。

3. 机器区（其余全部字段）与人工区（:class:`UserData`、``rl.manual_override``）
   物理隔离。程序只读人工区，永不写入。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Venue = Literal["ICML", "NeurIPS", "ICLR", "CoRL"]

#: 无外部标识时，用标题哈希兜底所需的截断长度
_HASH_LEN = 16


# --------------------------------------------------------------------------- #
# 归一化工具
# --------------------------------------------------------------------------- #

def normalize_title(title: str) -> str:
    """标题归一化，用于跨源匹配与标题哈希兜底。

    去大小写、去标点、压空白。注意**不做词干化**——那会引入误匹配，
    而误匹配是唯一会静默污染全库的错误（见 CLAUDE.md 已知风险）。

    **必须先剥掉 LaTeX 命令**：会议源的标题里常带 ``\\(\\varepsilon\\)-Optimally``
    这类写法，而 OpenAlex 存的是 ``ε-Optimally``。不处理的话前者会多出一个
    "varepsilon" 词，相似度掉到 0.88 左右，正好卡在 0.92 阈值下方——
    表现是"看起来没收录"，实则是我们自己把匹配判丢了。
    """
    t = re.sub(r"\\[a-zA-Z]+", " ", title or "")   # \varepsilon \pi \alpha
    t = re.sub(r"[$\\{}\^_~²³]", " ", t)           # 数学定界符与上下标
    t = unicodedata.normalize("NFKD", t).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_doi(doi: str | None) -> str | None:
    """剥掉 URL 前缀与大小写差异，只留裸 DOI。"""
    if not doi:
        return None
    d = doi.strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    return d or None


def sha1_16(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:_HASH_LEN]


# --------------------------------------------------------------------------- #
# 子模型
# --------------------------------------------------------------------------- #

class Provenance(BaseModel):
    """某个字段/整条记录来自哪个源、何时取到。多源合并时用于追溯。"""

    source: str                      # pmlr | mlanthology | openalex | arxiv
    source_id: str | None = None     # 该源内的标识，如 pmlr 的 "singh25e"
    fetched: str = Field(default_factory=lambda: date.today().isoformat())


class RLAnnotation(BaseModel):
    """RL 筛选结果。**打标而不删数据** —— 判定标准一定会变。

    存下 ``score`` 与 ``reasons`` 是为了让规则可调后可**重打分而不重爬**。
    """

    is_rl: bool = False
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    subfield: list[str] = Field(default_factory=list)   # 预留给后续人工/LLM 分类
    manual_override: bool | None = None                 # 人工改判，重跑不覆盖


class UserData(BaseModel):
    """人工区。**程序只读不写**，任何自动化流程都不得修改这些字段。"""

    model_config = ConfigDict(extra="allow")            # 允许你在 jsonl 里加自己的字段

    read: bool = False
    rating: int | None = None
    note: str | None = None


# --------------------------------------------------------------------------- #
# 主模型
# --------------------------------------------------------------------------- #

class CanonicalPaper(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # ---- 身份 ----
    paper_id: str = ""
    source_key: str | None = None        # 源内标识（PMLR: "singh25e"），跨源稳定
    title: str
    title_norm: str = ""
    authors: list[str] = Field(default_factory=list)

    # ---- 出处 ----
    venue: str
    year: int
    track: str | None = None             # poster | spotlight | oral，拿得到才填

    # ---- 外部标识（用于跨源去重与取全文）----
    doi: str | None = None
    arxiv_id: str | None = None
    openreview_id: str | None = None

    # ---- 链接与本地文件 ----
    url: str | None = None
    pdf_url: str | None = None
    local_pdf: str | None = None

    # ---- 再分发许可（只有预印本流会填）----
    #: arXiv 给的许可**原样值**，实测（2026-09-18）是一串 URL，如
    #: ``http://creativecommons.org/licenses/by/4.0/`` /
    #: ``http://arxiv.org/licenses/nonexclusive-distrib/1.0/``，
    #: **不是** "CC BY 4.0" 这种可读名。把 URL 译成名字、把名字判成"能不能发"，
    #: 都是**展示层**的事；存在这里的是事实本身。
    #:
    #: 存原样而不是"可再分发"布尔的理由，与 ``announce_type`` 不写死成三元组同源
    #: （见 ``D:/大学编程学习/arXiv周报/程序/rllib/adapters/arxiv_rss.py`` 开头第 2 条
    #: ——**那个适配器 2026-09-20 已随周报线搬出本库**）：arXiv 哪天加一种许可，
    #: 它会作为新值自己冒出来；而布尔会把新许可**静默**归进某一类。
    #:
    #: ⚠️ ``None`` 的语义是**「没读到」，不是「没有许可」**——本库反复栽的那个病
    #: （``MEASUREMENTS.md`` §四c 末尾那条）。取不到既**不等于**可以自由分发，
    #: 也**不等于**不能；它只是未知，卡片上必须如实说"未知"。
    #:
    #: 为什么它可以是**稳定字段**（不像 ``citations`` 要反复重查）：arXiv 官方规定
    #: 许可是**提交者选的、不可撤销**（同 #24），采到一次就不会再变。
    #:
    #: 唯一的取数入口是 OAI-PMH：搜索 API 的 schema 里没有这个字段（官方原文，同 #24）。
    #: 见 :mod:`rllib.adapters.arxiv_oai`（**2026-09-20 已随周报线搬出本库**，现住
    #: ``D:/大学编程学习/arXiv周报/程序/rllib/adapters/arxiv_oai.py``）。
    #: 会议论文集记录**一律不填**——它们没有这个源，
    #: 那里的 ``None`` 说的是"这条路根本不适用"，与预印本流的"没读到"是两件事。
    license: str | None = None

    # ---- 内容 ----
    abstract: str | None = None
    topics: list[str] = Field(default_factory=list)
    citations: int | None = None
    citations_updated: str | None = None
    #: 上次尝试补引用数的日期。"试过"而非"成功"——成功与否看 ``citations``。
    #:
    #: 与 ``citations_updated`` 分开是因为这两种情况该分开重试：补到了（哪怕是 0）
    #: 不该在 ``retry_after_days`` 内重查，补不到（标题匹配不上）才该过阵子再试。
    #: 合成一个字段的话，前者会把配额浪费在重查上，后者会永远不再尝试。
    #: 见 :mod:`rllib.citations`。
    citations_attempted: str | None = None
    #: 上次尝试富化的日期。用于避免每周把注定匹配不上的论文重查一遍。
    #: 注意它记录的是"试过"，不是"成功"——成功与否看 arxiv_id/abstract/citations。
    enrich_attempted: str | None = None

    # ---- 标注与溯源 ----
    rl: RLAnnotation = Field(default_factory=RLAnnotation)
    provenance: list[Provenance] = Field(default_factory=list)
    first_seen: str = Field(default_factory=lambda: date.today().isoformat())
    last_updated: str = Field(default_factory=lambda: date.today().isoformat())

    # ---- 人工区（程序不得写入）----
    user: UserData = Field(default_factory=UserData)

    # ------------------------------------------------------------------ #
    # 派生
    # ------------------------------------------------------------------ #

    def model_post_init(self, __context) -> None:  # noqa: D105
        if not self.title_norm:
            object.__setattr__(self, "title_norm", normalize_title(self.title))

    @property
    def is_rl_effective(self) -> bool:
        """最终生效的 RL 判定：人工改判优先于自动打分。

        ``filters/rl.py`` 重打分时也会把 override 写进 ``rl.is_rl``，但那只在
        跑过 filter 之后成立。若有人直接在 jsonl 里改 ``manual_override`` 而没重跑
        filter，读 ``rl.is_rl`` 就会读到过期结论。下游（PDF 抓取、全文抽取、
        质量报告）一律用这个属性，免得各处理解不一致。
        """
        if self.rl.manual_override is not None:
            return bool(self.rl.manual_override)
        return self.rl.is_rl

    @property
    def cite_key(self) -> str:
        """BibTeX 引用键，形如 ``smith2025policy``。"""
        first = (self.authors[0].split()[-1] if self.authors else "anon").lower()
        first = re.sub(r"[^a-z]", "", first) or "anon"
        word = next(
            (w for w in re.findall(r"[a-z]{3,}", self.title_norm) if w not in _STOPWORDS),
            "paper",
        )
        return f"{first}{self.year}{word}"

    def identity_keys(self) -> list[str]:
        """全部可用于跨源匹配的键。upsert 时只要有任意一个命中就视为同一篇。"""
        keys: list[str] = []
        if self.arxiv_id:
            keys.append(f"arxiv:{self.arxiv_id.lower()}")
        d = normalize_doi(self.doi)
        if d:
            keys.append(f"doi:{d}")
        if self.openreview_id:
            keys.append(f"orid:{self.openreview_id}")
        if self.source_key:
            keys.append(f"src:{self.venue.lower()}{self.year}:{self.source_key}")
        if self.title_norm:
            keys.append(f"title:{sha1_16(self.title_norm)}")
        return keys

    def content_hash(self) -> str:
        """渲染相关字段的哈希。**不含** paper_id / last_updated / user。

        用于 ``render --changed-only``：只有这个值变了才重写 Markdown，
        否则每次重跑都会产生上万条无意义的 git diff。
        """
        payload = "|".join([
            self.title,
            ",".join(self.authors),
            self.venue,
            str(self.year),
            self.track or "",
            self.arxiv_id or "",
            self.doi or "",
            self.openreview_id or "",
            self.url or "",
            self.pdf_url or "",
            (self.abstract or "")[:2000],
            ",".join(sorted(self.topics)),
            str(self.citations if self.citations is not None else ""),
            f"{self.rl.is_rl}:{self.rl.score:.3f}:{','.join(sorted(self.rl.reasons))}",
        ])
        return sha1_16(payload)


#: 生成 cite_key 时跳过的虚词
_STOPWORDS = frozenset(
    "the a an of for and to in on with via using towards toward from is are we our "
    "this that can via new towards learning".split()
)
