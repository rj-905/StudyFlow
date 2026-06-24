"""
Lecture Agent Tool: Video search and relevance scoring.

Provides the Lecture Agent with the ability to search for educational
video content and rank results using a composite scoring function
based on view count, duration, and keyword relevance.

Scoring Function
================

Each video is scored by a weighted composite:

.. math::

    S = w_k \\cdot R_{\\text{keyword}} + w_v \\cdot R_{\\text{views}} + w_d \\cdot R_{\\text{duration}}

where:

* :math:`R_{\\text{keyword}} \\in [0, 1]` — fraction of query keywords
  found in the title/description.
* :math:`R_{\\text{views}}` — log-normalised view count:
  :math:`\\frac{\\log(1 + v)}{\\log(1 + v_{\\max})}`.
* :math:`R_{\\text{duration}}` — Gaussian preference centred on the
  ideal duration (default 600s / 10 min):
  :math:`e^{-\\frac{(d - d_{\\text{ideal}})^2}{2\\sigma^2}}`.
* :math:`w_k, w_v, w_d` — tunable weights (default 0.5, 0.25, 0.25).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from studyflow.agents.schemas import VideoResource


# ---------------------------------------------------------------------------
# Scoring Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class VideoScoringConfig:
    """
    Tunable weights and parameters for the video relevance scorer.

    Attributes
    ----------
    weight_keyword : float
        Weight for keyword relevance (default 0.50).
    weight_views : float
        Weight for view-count signal (default 0.25).
    weight_duration : float
        Weight for duration preference (default 0.25).
    ideal_duration_s : int
        Ideal video duration in seconds (default 600 = 10 min).
    duration_sigma : float
        Gaussian sigma for duration scoring (default 300 = 5 min).
    """

    weight_keyword: float = 0.50
    weight_views: float = 0.25
    weight_duration: float = 0.25
    ideal_duration_s: int = 600
    duration_sigma: float = 300.0


# ---------------------------------------------------------------------------
# Scoring Engine
# ---------------------------------------------------------------------------

def score_video(
    title: str,
    description: str,
    view_count: int,
    duration_seconds: int,
    query_keywords: list[str],
    max_views: int,
    config: Optional[VideoScoringConfig] = None,
) -> float:
    """
    Compute a composite relevance score for a single video.

    .. math::

        S = w_k \\cdot R_{\\text{keyword}} + w_v \\cdot R_{\\text{views}}
            + w_d \\cdot R_{\\text{duration}}

    Parameters
    ----------
    title : str
        Video title.
    description : str
        Video description (may be truncated).
    view_count : int
        Number of views.
    duration_seconds : int
        Video length in seconds.
    query_keywords : list[str]
        Keywords extracted from the user's query.
    max_views : int
        Maximum view count in the result set (for normalisation).
    config : VideoScoringConfig | None
        Scoring weights.  Defaults to ``VideoScoringConfig()``.

    Returns
    -------
    float
        Score in [0, 1].
    """
    cfg = config or VideoScoringConfig()

    r_kw = _keyword_relevance(title, description, query_keywords)
    r_views = _view_score(view_count, max_views)
    r_dur = _duration_score(duration_seconds, cfg.ideal_duration_s, cfg.duration_sigma)

    raw = (
        cfg.weight_keyword * r_kw
        + cfg.weight_views * r_views
        + cfg.weight_duration * r_dur
    )
    return max(0.0, min(1.0, raw))


def _keyword_relevance(title: str, description: str, keywords: list[str]) -> float:
    """Fraction of query keywords found in the title or description."""
    if not keywords:
        return 0.0
    text = f"{title} {description}".lower()
    matches = sum(1 for kw in keywords if kw.lower() in text)
    return matches / len(keywords)


def _view_score(view_count: int, max_views: int) -> float:
    """Log-normalised view count relative to the batch maximum."""
    if max_views <= 0:
        return 0.0
    return math.log(1 + view_count) / math.log(1 + max_views)


def _duration_score(duration_s: int, ideal: int, sigma: float) -> float:
    """
    Gaussian preference around the ideal duration.

    .. math::

        R_d = e^{-\\frac{(d - d_{\\text{ideal}})^2}{2\\sigma^2}}
    """
    if sigma <= 0:
        return 1.0 if duration_s == ideal else 0.0
    return math.exp(-((duration_s - ideal) ** 2) / (2 * sigma ** 2))


# ---------------------------------------------------------------------------
# YouTube Data API v3 Search Tool
# ---------------------------------------------------------------------------

async def search_youtube(
    query: str,
    api_key: str,
    max_results: int = 10,
    scoring_config: Optional[VideoScoringConfig] = None,
) -> list[VideoResource]:
    """
    Search YouTube for educational videos and return scored results.

    Uses the YouTube Data API v3.  Requires a valid API key.

    Parameters
    ----------
    query : str
        The search query (e.g., "Amortized Analysis tutorial").
    api_key : str
        YouTube Data API v3 key.
    max_results : int
        Number of results to request (max 50).
    scoring_config : VideoScoringConfig | None
        Custom scoring weights.

    Returns
    -------
    list[VideoResource]
        Videos ranked by composite relevance score (descending).
    """
    cfg = scoring_config or VideoScoringConfig()
    keywords = _extract_keywords(query)

    # Step 1: search for videos
    search_items = await _youtube_search_api(query, api_key, max_results)
    if not search_items:
        return []

    # Step 2: get video details (duration, view count)
    video_ids = [item["id"]["videoId"] for item in search_items if "videoId" in item.get("id", {})]
    details_map = await _youtube_video_details(video_ids, api_key) if video_ids else {}

    # Step 3: score and build results
    max_views = max(
        (d.get("view_count", 0) for d in details_map.values()),
        default=1,
    )

    results: list[VideoResource] = []
    for item in search_items:
        vid_id = item.get("id", {}).get("videoId", "")
        if not vid_id:
            continue

        snippet = item.get("snippet", {})
        title = snippet.get("title", "")
        description = snippet.get("description", "")
        channel = snippet.get("channelTitle", "")

        detail = details_map.get(vid_id, {})
        duration_s = detail.get("duration_seconds", 0)
        views = detail.get("view_count", 0)

        relevance = score_video(
            title=title,
            description=description,
            view_count=views,
            duration_seconds=duration_s,
            query_keywords=keywords,
            max_views=max_views,
            config=cfg,
        )

        results.append(
            VideoResource(
                title=title,
                url=f"https://www.youtube.com/watch?v={vid_id}",
                channel=channel,
                duration_seconds=duration_s,
                view_count=views,
                relevance_score=round(relevance, 4),
                description=description[:300] if description else None,
            )
        )

    # Sort by relevance descending
    results.sort(key=lambda v: v.relevance_score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# YouTube API Helpers
# ---------------------------------------------------------------------------

async def _youtube_search_api(
    query: str, api_key: str, max_results: int
) -> list[dict[str, Any]]:
    """Call YouTube Data API v3 search.list endpoint."""
    params = {
        "part": "snippet",
        "q": f"{query} tutorial lecture explanation",
        "type": "video",
        "maxResults": min(max_results, 50),
        "order": "relevance",
        "videoCategoryId": "27",  # Education
        "key": api_key,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            "https://www.googleapis.com/youtube/v3/search",
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()
    return data.get("items", [])


async def _youtube_video_details(
    video_ids: list[str], api_key: str
) -> dict[str, dict[str, Any]]:
    """Fetch video details (duration, viewCount) for a batch of IDs."""
    params = {
        "part": "contentDetails,statistics",
        "id": ",".join(video_ids),
        "key": api_key,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

    result: dict[str, dict[str, Any]] = {}
    for item in data.get("items", []):
        vid_id = item["id"]
        duration_iso = item.get("contentDetails", {}).get("duration", "PT0S")
        stats = item.get("statistics", {})
        result[vid_id] = {
            "duration_seconds": _parse_iso8601_duration(duration_iso),
            "view_count": int(stats.get("viewCount", 0)),
        }
    return result


def _parse_iso8601_duration(iso_duration: str) -> int:
    """
    Convert an ISO 8601 duration (e.g., ``PT1H2M10S``) to seconds.

    Parameters
    ----------
    iso_duration : str
        ISO 8601 duration string from YouTube API.

    Returns
    -------
    int
        Total seconds.
    """
    pattern = re.compile(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", re.IGNORECASE
    )
    match = pattern.match(iso_duration)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def _extract_keywords(query: str) -> list[str]:
    """
    Extract meaningful keywords from a query string.

    Strips common stop-words to improve keyword-matching accuracy.
    """
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
