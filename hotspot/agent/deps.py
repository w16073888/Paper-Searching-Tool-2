"""Request-scoped dependencies for the research agent."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import load_config
from ..corpus import open_corpus
from ..index.lance_store import LancePaperStore
from ..index.topic_store import TopicStore


@dataclass
class AgentDeps:
    cfg: dict = field(default_factory=load_config)
    question: str = ""
    source: str = "meeting"
    years: list[int] = field(default_factory=list)
    venues: list[str] = field(default_factory=list)
    weeks: list[str] = field(default_factory=list)
    store: LancePaperStore | None = None
    topics: TopicStore | None = None
    corpus: object | None = None
    seen_ids: set[str] = field(default_factory=set)
    fetched: set[str] = field(default_factory=set)
    fetched_order: list[str] = field(default_factory=list)
    history_citation_ids: list[str] = field(default_factory=list)
    search_calls: int = 0
    fetch_calls: int = 0
    topic_calls: int = 0
    search_limit: int = 12
    fetch_limit: int = 6
    max_claims: int = 6
    request_limit: int = 10
    tool_limit: int = 10
    scope_timeout_s: float = 30.0
    agent_timeout_s: float = 150.0
    verify_timeout_s: float = 30.0
    events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        agent = self.cfg.get("agent") or {}
        self.search_limit = int(agent.get("search_limit") or self.search_limit)
        self.fetch_limit = int(agent.get("fetch_limit") or self.fetch_limit)
        self.max_claims = int(agent.get("max_claims") or self.max_claims)
        self.request_limit = int(agent.get("request_limit") or self.request_limit)
        self.tool_limit = int(agent.get("tool_limit") or self.tool_limit)
        self.scope_timeout_s = float(
            agent.get("scope_timeout_s") or self.scope_timeout_s
        )
        self.agent_timeout_s = float(
            agent.get("agent_timeout_s") or self.agent_timeout_s
        )
        self.verify_timeout_s = float(
            agent.get("verify_timeout_s") or self.verify_timeout_s
        )
        if self.store is None:
            self.store = LancePaperStore(self.cfg, source=self.source)
        if self.topics is None:
            self.topics = TopicStore(self.cfg)
        if self.corpus is None:
            self.corpus = open_corpus(self.cfg)
