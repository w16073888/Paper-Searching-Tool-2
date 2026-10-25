"""Generic paper-search tool; no answer-mode or controller policy."""

from __future__ import annotations

from ..core.models import SearchFilters, SearchResult
from ..index.lance_store import LancePaperStore


def search_papers(
    store: LancePaperStore,
    query: str,
    *,
    keywords: list[str] | None = None,
    years: list[int] | None = None,
    venues: list[str] | None = None,
    weeks: list[str] | None = None,
    limit: int = 12,
) -> SearchResult:
    """Search papers by semantic and lexical similarity under hard metadata filters."""
    return store.search(
        query,
        keywords=keywords,
        filters=SearchFilters(
            years=years or [],
            venues=venues or [],
            weeks=weeks or [],
        ),
        limit=max(1, min(int(limit), 30)),
    )
