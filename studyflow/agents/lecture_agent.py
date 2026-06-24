"""
Lecture Agent: Video resource discovery and LLM-powered re-ranking.

Pipeline
--------
1. Search YouTube via Data API v3 → raw scored list.
2. Re-rank with Gemini 3.5 Flash for educational quality.
3. Return ``LectureResult`` with top-ranked ``VideoResource`` items.
4. Persist top results to SQLite.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from uuid import UUID

from studyflow.agents.schemas import (
    AgentRole,
    LectureResult,
    SubAgentTask,
    TaskStatus,
    VideoResource,
)
from studyflow.agents.tools.lecture_search import (
    VideoScoringConfig,
    search_youtube,
)
from studyflow.config.llm_provider import GeminiProvider, TaskWeight, get_provider


# ---------------------------------------------------------------------------
# System prompt for the Lecture re-ranker (Gemini 3.5 Flash)
# ---------------------------------------------------------------------------

_RERANK_SYSTEM = """You are an expert educational content curator. 
Given a list of YouTube videos found for a study topic, re-rank them by 
educational value. Consider:

1. **Clarity**: Does the title/description suggest a clear explanation?
2. **Depth**: Does it cover the topic at an appropriate academic level?
3. **Credibility**: Is the channel reputable (university lectures, 
   well-known educators)?
4. **Format**: Prefer structured tutorials over tangential mentions.

Return a JSON array of objects, each with:
- "index": the original 0-based index of the video
- "score": a float from 0.0 to 1.0 (your educational quality assessment)
- "reason": a one-sentence justification

Order by score descending. Return ONLY the JSON array."""


# ---------------------------------------------------------------------------
# Lecture Agent
# ---------------------------------------------------------------------------

class LectureAgent:
    """
    Discovers and ranks educational video content for a topic.

    Parameters
    ----------
    provider : GeminiProvider | None
        LLM provider for re-ranking.  Defaults to the global singleton.
    youtube_api_key : str | None
        YouTube Data API v3 key.  If None, pulled from env.
    scoring_config : VideoScoringConfig | None
        Custom scoring weights for initial ranking.
    max_results : int
        Number of YouTube results to fetch.
    """

    def __init__(
        self,
        provider: Optional[GeminiProvider] = None,
        youtube_api_key: Optional[str] = None,
        scoring_config: Optional[VideoScoringConfig] = None,
        max_results: int = 10,
    ) -> None:
        self._provider = provider or get_provider()
        self._yt_key = youtube_api_key
        self._scoring_config = scoring_config
        self._max_results = max_results

    async def run(self, task: SubAgentTask) -> LectureResult:
        """
        Execute the Lecture Agent pipeline.

        Parameters
        ----------
        task : SubAgentTask
            Must have ``agent == AgentRole.LECTURE``.

        Returns
        -------
        LectureResult
        """
        topic = task.topic

        try:
            # Resolve API key
            if self._yt_key is None:
                from studyflow.config.llm_provider import get_youtube_api_key
                yt_key = get_youtube_api_key()
            else:
                yt_key = self._yt_key

            # Step 1: YouTube search with algorithmic scoring
            raw_videos = await search_youtube(
                query=topic,
                api_key=yt_key,
                max_results=self._max_results,
                scoring_config=self._scoring_config,
            )

            if not raw_videos:
                return LectureResult(
                    task_id=task.task_id,
                    topic=topic,
                    status=TaskStatus.COMPLETED,
                    videos=[],
                )

            # Step 2: LLM re-ranking with Gemini 3.5 Flash
            reranked = await self._rerank_with_llm(raw_videos, topic)

            return LectureResult(
                task_id=task.task_id,
                topic=topic,
                status=TaskStatus.COMPLETED,
                videos=reranked,
            )

        except Exception as e:
            return LectureResult(
                task_id=task.task_id,
                topic=topic,
                status=TaskStatus.FAILED,
                error=str(e),
            )

    async def _rerank_with_llm(
        self,
        videos: list[VideoResource],
        topic: str,
    ) -> list[VideoResource]:
        """
        Use Gemini 3.5 Flash to re-rank videos by educational quality.

        Falls back to the original algorithmic ranking if the LLM
        call fails.
        """
        # Build the prompt
        video_descriptions = []
        for i, v in enumerate(videos):
            video_descriptions.append(
                f"[{i}] Title: {v.title}\n"
                f"    Channel: {v.channel or 'Unknown'}\n"
                f"    Duration: {v.duration_seconds or 0}s\n"
                f"    Views: {v.view_count or 0}\n"
                f"    Description: {(v.description or '')[:200]}"
            )

        prompt = (
            f"Topic: {topic}\n\n"
            f"Videos:\n" + "\n".join(video_descriptions)
        )

        try:
            rankings = self._provider.generate_json(
                prompt=prompt,
                weight=TaskWeight.LIGHT,
                system_instruction=_RERANK_SYSTEM,
                temperature=0.2,
            )

            # Apply LLM scores
            if isinstance(rankings, list):
                index_to_score: dict[int, float] = {}
                for item in rankings:
                    idx = item.get("index", -1)
                    score = item.get("score", 0.0)
                    if 0 <= idx < len(videos):
                        index_to_score[idx] = float(score)

                # Blend: 40% algorithmic + 60% LLM
                reranked = []
                for i, v in enumerate(videos):
                    llm_score = index_to_score.get(i, v.relevance_score)
                    blended = 0.4 * v.relevance_score + 0.6 * llm_score
                    reranked.append(
                        v.model_copy(
                            update={"relevance_score": round(blended, 4)}
                        )
                    )

                reranked.sort(key=lambda x: x.relevance_score, reverse=True)
                return reranked

        except Exception:
            pass  # Fall through to original ranking

        return videos
