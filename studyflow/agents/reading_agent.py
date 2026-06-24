"""
Reading Agent: Academic resource discovery and LLM-powered re-ranking.

Pipeline
--------
1. Search arXiv API + Google Custom Search concurrently.
2. Merge and deduplicate results.
3. Re-rank with Gemini 3.5 Flash for academic quality.
4. Return ``ReadingResult`` with top-ranked ``TextResource`` items.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional
from uuid import UUID

from studyflow.agents.schemas import (
    AgentRole,
    ReadingResult,
    SubAgentTask,
    TaskStatus,
    TextResource,
)
from studyflow.agents.tools.reading_search import search_arxiv
from studyflow.agents.tools.web_search import search_google_cse
from studyflow.config.llm_provider import GeminiProvider, TaskWeight, get_provider


# ---------------------------------------------------------------------------
# System prompt for the Reading re-ranker (Gemini 3.5 Flash)
# ---------------------------------------------------------------------------

_RERANK_SYSTEM = """You are an academic resource curator specialising in 
computer science and mathematics education. Given a list of text resources 
found for a study topic, re-rank them by educational quality. Consider:

1. **Academic rigour**: Peer-reviewed papers and university materials rank higher.
2. **Accessibility**: Clear writing accessible to undergraduates is preferred.
3. **Relevance**: Does the resource directly address the topic?
4. **Completeness**: Prefer comprehensive treatments over brief mentions.
5. **Open access**: Freely available resources are preferred.

Return a JSON array of objects, each with:
- "index": the original 0-based index
- "score": a float from 0.0 to 1.0
- "reason": a one-sentence justification

Order by score descending. Return ONLY the JSON array."""


# ---------------------------------------------------------------------------
# Reading Agent
# ---------------------------------------------------------------------------

class ReadingAgent:
    """
    Discovers and ranks academic text resources for a topic.

    Parameters
    ----------
    provider : GeminiProvider | None
        LLM provider for re-ranking.
    google_api_key : str | None
        Google API key for Custom Search.
    google_cse_id : str | None
        Google Custom Search Engine ID.
    max_results_per_source : int
        Maximum results from each search source.
    """

    def __init__(
        self,
        provider: Optional[GeminiProvider] = None,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
        max_results_per_source: int = 5,
    ) -> None:
        self._provider = provider or get_provider()
        self._api_key = google_api_key
        self._cse_id = google_cse_id
        self._max_per_source = max_results_per_source

    async def run(self, task: SubAgentTask) -> ReadingResult:
        """
        Execute the Reading Agent pipeline.

        Parameters
        ----------
        task : SubAgentTask
            Must have ``agent == AgentRole.READING``.

        Returns
        -------
        ReadingResult
        """
        topic = task.topic

        try:
            # Resolve API keys
            if self._api_key is None:
                from studyflow.config.llm_provider import get_google_api_key
                api_key = get_google_api_key()
            else:
                api_key = self._api_key

            if self._cse_id is None:
                from studyflow.config.llm_provider import get_google_cse_id
                cse_id = get_google_cse_id()
            else:
                cse_id = self._cse_id

            # Step 1: Search arXiv + Google CSE concurrently
            arxiv_task = search_arxiv(topic, self._max_per_source)
            cse_task = search_google_cse(
                topic, api_key, cse_id, self._max_per_source
            )

            results = await asyncio.gather(
                arxiv_task, cse_task,
                return_exceptions=True,
            )

            # Merge results, skipping failed sources
            merged: list[TextResource] = []
            for result in results:
                if isinstance(result, list):
                    merged.extend(result)

            # Deduplicate by URL
            seen_urls: set[str] = set()
            deduped: list[TextResource] = []
            for r in merged:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    deduped.append(r)

            if not deduped:
                return ReadingResult(
                    task_id=task.task_id,
                    topic=topic,
                    status=TaskStatus.COMPLETED,
                    resources=[],
                )

            # Step 2: LLM re-ranking with Gemini 3.5 Flash
            reranked = await self._rerank_with_llm(deduped, topic)

            return ReadingResult(
                task_id=task.task_id,
                topic=topic,
                status=TaskStatus.COMPLETED,
                resources=reranked,
            )

        except Exception as e:
            return ReadingResult(
                task_id=task.task_id,
                topic=topic,
                status=TaskStatus.FAILED,
                error=str(e),
            )

    async def _rerank_with_llm(
        self,
        resources: list[TextResource],
        topic: str,
    ) -> list[TextResource]:
        """
        Use Gemini 3.5 Flash to re-rank resources by academic quality.

        Falls back to algorithmic ranking on LLM failure.
        """
        resource_descriptions = []
        for i, r in enumerate(resources):
            resource_descriptions.append(
                f"[{i}] Title: {r.title}\n"
                f"    Source: {r.source_type}\n"
                f"    URL: {r.url}\n"
                f"    Authors: {', '.join(r.authors) if r.authors else 'Unknown'}\n"
                f"    Abstract: {(r.abstract or '')[:200]}"
            )

        prompt = (
            f"Topic: {topic}\n\n"
            f"Resources:\n" + "\n".join(resource_descriptions)
        )

        try:
            rankings = self._provider.generate_json(
                prompt=prompt,
                weight=TaskWeight.LIGHT,
                system_instruction=_RERANK_SYSTEM,
                temperature=0.2,
            )

            if isinstance(rankings, list):
                index_to_score: dict[int, float] = {}
                for item in rankings:
                    idx = item.get("index", -1)
                    score = item.get("score", 0.0)
                    if 0 <= idx < len(resources):
                        index_to_score[idx] = float(score)

                # Blend: 40% algorithmic + 60% LLM
                reranked = []
                for i, r in enumerate(resources):
                    llm_score = index_to_score.get(i, r.relevance_score)
                    blended = 0.4 * r.relevance_score + 0.6 * llm_score
                    reranked.append(
                        r.model_copy(
                            update={"relevance_score": round(blended, 4)}
                        )
                    )

                reranked.sort(key=lambda x: x.relevance_score, reverse=True)
                return reranked

        except Exception:
            pass

        return resources
