"""
Reading Agent Tool: Academic resource search and retrieval.

Provides the Reading Agent with tools to search for open-access
academic and educational text resources, prioritising:

1. **arXiv** — open-access preprints via the arXiv API.
2. **MIT OpenCourseWare** — lecture notes and course materials.
3. **OpenAlex** — open scholarly metadata catalogue (fallback).

Each result is scored by a relevance function that considers keyword
overlap, source priority, and open-access status.

Source Priority Weights
=======================

+-------------+----------+
| Source       | Weight   |
+=============+==========+
| arXiv       | 1.00     |
| MIT OCW     | 0.95     |
| OpenAlex OA | 0.85     |
| Other       | 0.50     |
+-------------+----------+
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import quote_plus

import httpx

from studyflow.agents.schemas import TextResource


# ---------------------------------------------------------------------------
# Source Priority Map
# ---------------------------------------------------------------------------

_SOURCE_PRIORITY: dict[str, float] = {
    "arxiv": 1.00,
    "mit_ocw": 0.95,
    "openalex": 0.85,
    "other": 0.50,
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_text_resource(
    title: str,
    abstract: str,
    query_keywords: list[str],
    source_type: str,
    is_open_access: bool = True,
) -> float:
    """
    Compute a relevance score for a text resource.

    .. math::

        S = 0.6 \\cdot R_{\\text{keyword}} + 0.3 \\cdot P_{\\text{source}}
            + 0.1 \\cdot \\mathbb{1}_{\\text{OA}}

    Parameters
    ----------
    title : str
        Resource title.
    abstract : str
        Abstract or description.
    query_keywords : list[str]
        Keywords extracted from the user's query.
    source_type : str
        Source category (e.g., ``'arxiv'``).
    is_open_access : bool
        Whether the resource is freely available.

    Returns
    -------
    float
        Score in [0, 1].
    """
    r_kw = _keyword_relevance(title, abstract, query_keywords)
    p_src = _SOURCE_PRIORITY.get(source_type, _SOURCE_PRIORITY["other"])
    oa_bonus = 1.0 if is_open_access else 0.0

    raw = 0.6 * r_kw + 0.3 * p_src + 0.1 * oa_bonus
    return max(0.0, min(1.0, raw))


def _keyword_relevance(title: str, abstract: str, keywords: list[str]) -> float:
    """Fraction of query keywords found in the title or abstract."""
    if not keywords:
        return 0.0
    text = f"{title} {abstract}".lower()
    matches = sum(1 for kw in keywords if kw.lower() in text)
    return matches / len(keywords)


def _extract_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from a query string."""
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "for", "and", "nor", "but", "or", "yet", "so", "at", "by",
        "in", "of", "on", "to", "up", "it", "its", "with", "from",
        "as", "into", "about", "what", "how", "why", "when", "where",
        "which", "who", "whom", "this", "that", "these", "those",
        "i", "me", "my", "we", "our", "you", "your", "he", "she",
        "they", "them", "their",
    }
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())
    return [w for w in words if w not in stop_words and len(w) > 1]


# ---------------------------------------------------------------------------
# arXiv Search
# ---------------------------------------------------------------------------

async def search_arxiv(
    query: str,
    max_results: int = 10,
) -> list[TextResource]:
    """
    Search the arXiv API for open-access papers related to the query.

    Uses the arXiv Atom feed API (``export.arxiv.org/api/query``).

    Parameters
    ----------
    query : str
        Natural language search query.
    max_results : int
        Maximum results to return.

    Returns
    -------
    list[TextResource]
        Ranked text resources from arXiv.
    """
    keywords = _extract_keywords(query)
    search_query = "+AND+".join(f"all:{quote_plus(kw)}" for kw in keywords)
    if not search_query:
        search_query = quote_plus(query)

    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query={search_query}"
        f"&start=0&max_results={max_results}"
        f"&sortBy=relevance&sortOrder=descending"
    )

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    return _parse_arxiv_response(resp.text, keywords)


def _parse_arxiv_response(xml_text: str, keywords: list[str]) -> list[TextResource]:
    """Parse arXiv Atom XML response into TextResource list."""
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)

    results: list[TextResource] = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        title = (title_el.text or "").strip() if title_el is not None else ""
        abstract = (summary_el.text or "").strip() if summary_el is not None else ""

        # Get the abstract page link (prefer abs link)
        url = ""
        for link in entry.findall("atom:link", ns):
            if link.get("type") == "text/html":
                url = link.get("href", "")
                break
            if link.get("rel") == "alternate":
                url = link.get("href", "")

        # Authors
        authors: list[str] = []
        for author in entry.findall("atom:author", ns):
            name_el = author.find("atom:name", ns)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        relevance = score_text_resource(
            title=title,
            abstract=abstract,
            query_keywords=keywords,
            source_type="arxiv",
            is_open_access=True,
        )

        results.append(
            TextResource(
                title=title,
                url=url,
                source_type="arxiv",
                authors=authors if authors else None,
                abstract=abstract[:500] if abstract else None,
                relevance_score=round(relevance, 4),
                is_open_access=True,
            )
        )

    results.sort(key=lambda r: r.relevance_score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# MIT OpenCourseWare Search
# ---------------------------------------------------------------------------

async def search_mit_ocw(
    query: str,
    max_results: int = 10,
) -> list[TextResource]:
    """
    Search MIT OpenCourseWare for course materials.

    Uses the MIT OCW search API (contentType filter for lecture notes).

    Parameters
    ----------
    query : str
        Natural language search query.
    max_results : int
        Maximum results to return.

    Returns
    -------
    list[TextResource]
        Ranked text resources from MIT OCW.
    """
    keywords = _extract_keywords(query)

    url = "https://ocw.mit.edu/search/"
    params = {
        "q": query,
        "type": "resourcefile",
    }

    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()

    # MIT OCW returns HTML; extract structured data from JSON-LD or
    # fall back to generating resources from known course URL patterns.
    return _build_ocw_results(query, keywords, max_results)


def _build_ocw_results(
    query: str, keywords: list[str], max_results: int
) -> list[TextResource]:
    """
    Build MIT OCW resource stubs from known course URL patterns.

    In production, this would parse the OCW API/HTML response.
    For now, it generates well-structured search URLs that link
    directly to OCW's search results.
    """
    encoded = quote_plus(query)
    base_url = f"https://ocw.mit.edu/search/?q={encoded}&type=resourcefile"

    relevance = score_text_resource(
        title=f"MIT OCW: {query}",
        abstract=f"Lecture notes and course materials for {query}",
        query_keywords=keywords,
        source_type="mit_ocw",
        is_open_access=True,
    )

    return [
        TextResource(
            title=f"MIT OpenCourseWare – {query}",
            url=base_url,
            source_type="mit_ocw",
            authors=None,
            abstract=f"Search results for '{query}' on MIT OpenCourseWare. "
                     f"Includes lecture notes, problem sets, and exam solutions.",
            relevance_score=round(relevance, 4),
            is_open_access=True,
        )
    ]


# ---------------------------------------------------------------------------
# OpenAlex Search (fallback)
# ---------------------------------------------------------------------------

async def search_openalex(
    query: str,
    max_results: int = 10,
) -> list[TextResource]:
    """
    Search OpenAlex for scholarly works related to the query.

    OpenAlex is a free, open scholarly metadata catalogue that
    replaced Microsoft Academic Graph.

    Parameters
    ----------
    query : str
        Natural language search query.
    max_results : int
        Maximum results to return.

    Returns
    -------
    list[TextResource]
        Ranked text resources from OpenAlex.
    """
    keywords = _extract_keywords(query)

    params = {
        "search": query,
        "per_page": min(max_results, 50),
        "sort": "relevance_score:desc",
        "filter": "is_oa:true",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            "https://api.openalex.org/works",
            params=params,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

    results: list[TextResource] = []
    for work in data.get("results", []):
        title = work.get("title", "") or ""
        abstract_inv = work.get("abstract_inverted_index", {})
        abstract = _reconstruct_abstract(abstract_inv) if abstract_inv else ""

        # Get best open-access URL
        oa = work.get("open_access", {})
        url = oa.get("oa_url", "") or work.get("doi", "") or ""
        if url.startswith("https://doi.org/"):
            url = url  # DOI URL is fine

        # Authors
        authors: list[str] = []
        for authorship in work.get("authorships", [])[:5]:
            author = authorship.get("author", {})
            name = author.get("display_name")
            if name:
                authors.append(name)

        is_oa = oa.get("is_oa", False)

        relevance = score_text_resource(
            title=title,
            abstract=abstract,
            query_keywords=keywords,
            source_type="openalex",
            is_open_access=is_oa,
        )

        if url:  # Only include results with accessible URLs
            results.append(
                TextResource(
                    title=title,
                    url=url,
                    source_type="openalex",
                    authors=authors if authors else None,
                    abstract=abstract[:500] if abstract else None,
                    relevance_score=round(relevance, 4),
                    is_open_access=is_oa,
                )
            )

    results.sort(key=lambda r: r.relevance_score, reverse=True)
    return results


def _reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
    """
    Reconstruct an abstract from OpenAlex's inverted-index format.

    OpenAlex stores abstracts as ``{word: [position, ...]}`` to save
    space.  This function reconstructs the original text.
    """
    if not inverted_index:
        return ""

    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))

    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)


# ---------------------------------------------------------------------------
# Unified Search (combines all sources)
# ---------------------------------------------------------------------------

async def search_academic_resources(
    query: str,
    max_results_per_source: int = 5,
) -> list[TextResource]:
    """
    Search across all academic sources and return a merged, ranked list.

    Searches arXiv, MIT OCW, and OpenAlex concurrently, then merges
    and re-ranks by relevance score.

    Parameters
    ----------
    query : str
        Natural language search query.
    max_results_per_source : int
        Maximum results from each individual source.

    Returns
    -------
    list[TextResource]
        Merged and ranked text resources from all sources.
    """
    import asyncio

    arxiv_task = search_arxiv(query, max_results_per_source)
    ocw_task = search_mit_ocw(query, max_results_per_source)
    openalex_task = search_openalex(query, max_results_per_source)

    results = await asyncio.gather(
        arxiv_task, ocw_task, openalex_task,
        return_exceptions=True,
    )

    merged: list[TextResource] = []
    for result in results:
        if isinstance(result, list):
            merged.extend(result)
        # Silently skip failed sources (logged elsewhere)

    # Sort by relevance descending
    merged.sort(key=lambda r: r.relevance_score, reverse=True)
    return merged
