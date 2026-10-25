"""Pydantic models for the v2 agent/tool boundary."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    years: list[int] = Field(default_factory=list)
    venues: list[str] = Field(default_factory=list)
    weeks: list[str] = Field(default_factory=list)


class SearchHit(BaseModel):
    paper_id: str
    title: str
    year: int | None = None
    venue: str = ""
    week_id: str = ""
    first_seen: str = ""
    score: float = 0.0
    snippet: str = ""


class SearchResult(BaseModel):
    query: str
    results: list[SearchHit] = Field(default_factory=list)
    n_candidates: int = 0
    truncated: bool = False


class PaperDetail(BaseModel):
    paper_id: str
    title: str
    abstract: str = ""
    year: int | None = None
    venue: str = ""
    week_id: str = ""
    first_seen: str = ""
    subfields: list[str] = Field(default_factory=list)


class AnswerClaim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class AgentAnswer(BaseModel):
    claims: list[AnswerClaim] = Field(default_factory=list)
    no_answer_reason: str = ""
    limitations: list[str] = Field(default_factory=list)


class ScopeDecision(BaseModel):
    in_scope: bool
    domain: str = ""
    reason: str = ""


class TopicHit(BaseModel):
    cluster_id: int
    label: str
    size: int
    spread_state: str
    venue_counts: dict[str, int] = Field(default_factory=dict)
    year_counts: dict[int, int] = Field(default_factory=dict)
    representative_id: str = ""
    representative_title: str = ""
    score: float = 0.0


class TopicOverview(BaseModel):
    query: str
    topics: list[TopicHit] = Field(default_factory=list)
