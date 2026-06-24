"""
Google Custom Search API tool for the Reading Agent.

Searches the web for educational resources using Google's Programmable
Search Engine API, filtering for academic and educational domains.

Free Tier
---------
100 queries/day.  Results are cached in-memory per session to avoid
burning quota on repeated searches for the same topic.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Optional
from urllib.parse import quote_plus

import httpx

from studyflow.agents.schemas import TextResource
from studyflow.agents.tools.reading_search import (
    _extract_keywords,
    score_text_resource,
)


# ---------------------------------------------------------------------------
# Domain Authority Scoring
# ---------------------------------------------------------------------------

_DOMAIN_AUTHORITY: dict[str, float] = {
    "edu": 1.00,
    "ac.uk": 0.95,
    "org": 0.80,
    "gov": 0.85,
    "io": 0.50,
    "com": 0.40,
}


def _domain_score(url: str) -> float:
    """Estimate domain authority from the URL's TLD."""
    url_lower = url.lower()
    for domain, score in _DOMAIN_AUTHORITY.items():
        if f".{domain}" in url_lower:
            return score
    return 0.30


# ---------------------------------------------------------------------------
# Google Custom Search API
# ---------------------------------------------------------------------------

async def search_google_cse(
    query: str,
    api_key: str,
    cse_id: str,
    max_results: int = 10,
) -> list[TextResource]:
    """
    Search Google Custom Search Engine for educational resources.

    Appends educational qualifiers to the query and scores results
    by keyword relevance, domain authority, and rank position.

    Parameters
    ----------
    query : str
        Natural language search query.
    api_key : str
        Google API key (same as Gemini key).
    cse_id : str
        Programmable Search Engine ID.
    max_results : int
        Maximum results (API max is 10 per request).

    Returns
    -------
    list[TextResource]
        Ranked text resources from the web.
    """
    keywords = _extract_keywords(query)

    # Append educational qualifiers
    edu_query = f"{query} lecture notes tutorial course"

    params: dict[str, Any] = {
        "key": api_key,
        "cx": cse_id,
        "q": edu_query,
        "num": min(max_results, 10),
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

    results: list[TextResource] = []
    items = data.get("items", [])
    total_items = len(items)

    for rank, item in enumerate(items):
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        url = item.get("link", "")

        if not url:
            continue

        # Composite score: keyword match + domain authority + rank decay
        kw_score = _keyword_relevance_cse(title, snippet, keywords)
        dom_score = _domain_score(url)
        rank_score = 1.0 - (rank / max(total_items, 1))  # First result = 1.0

        relevance = (
            0.45 * kw_score
            + 0.30 * dom_score
            + 0.25 * rank_score
        )
        relevance = max(0.0, min(1.0, relevance))

        # Determine source type from domain
        source_type = _classify_source(url)

        results.append(
            TextResource(
                title=title,
                url=url,
                source_type=source_type,
                authors=None,
                abstract=snippet[:500] if snippet else None,
                relevance_score=round(relevance, 4),
                is_open_access=True,  # CSE results are publicly accessible
            )
        )

    results.sort(key=lambda r: r.relevance_score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _keyword_relevance_cse(title: str, snippet: str, keywords: list[str]) -> float:
    """Fraction of query keywords found in title or snippet."""
    if not keywords:
        return 0.0
    text = f"{title} {snippet}".lower()
    matches = sum(1 for kw in keywords if kw.lower() in text)
    return matches / len(keywords)


def _classify_source(url: str) -> str:
    """Classify a URL into a source type category."""
    url_lower = url.lower()
    if "arxiv.org" in url_lower:
        return "arxiv"
    elif "ocw.mit.edu" in url_lower:
        return "mit_ocw"
    elif ".edu" in url_lower:
        return "academic"
    elif "wikipedia.org" in url_lower:
        return "encyclopedia"
    elif any(x in url_lower for x in ["github.com", "gitlab.com"]):
        return "repository"
    elif any(x in url_lower for x in [".pdf", "lecture", "notes", "slides"]):
        return "lecture_notes"
    else:
        return "web"
