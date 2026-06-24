"""
Notes Agent: RAG-enabled note synthesis using Gemini 2.5 Pro.

Pipeline
--------
1. Query ChromaDB for existing notes on the topic and related topics.
2. Build a context window from retrieved chunks + incoming resources.
3. Call Gemini 2.5 Pro to synthesize comprehensive, context-aware notes.
4. Chunk and embed the new notes back into ChromaDB.
5. Return ``NotesResult`` with the synthesized markdown.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from uuid import UUID

from studyflow.agents.schemas import (
    AgentRole,
    NotesResult,
    SubAgentTask,
    TaskStatus,
)
from studyflow.config.llm_provider import GeminiProvider, TaskWeight, get_provider
from studyflow.memory.vector_store import NotesVectorStore


# ---------------------------------------------------------------------------
# System prompt for Notes synthesis (Gemini 2.5 Pro)
# ---------------------------------------------------------------------------

_NOTES_SYSTEM = """You are an expert academic note-taker and knowledge 
synthesizer. Your task is to generate comprehensive, well-structured 
study notes on a given topic.

RULES:
1. Write in clear, concise Markdown with proper heading hierarchy.
2. If prior notes on this or related topics are provided, EXPLICITLY 
   weave that context in — reference connections, build on existing 
   knowledge, and avoid redundant repetition.
3. Include:
   - A clear definition/introduction
   - Key concepts and theorems (with intuitive explanations)
   - Worked examples where applicable
   - Common pitfalls and misconceptions
   - Connections to prerequisite/related topics
4. Use LaTeX for mathematical notation: $inline$ and $$display$$.
5. At the end, add a "## Key Takeaways" section with 3-5 bullet points.
6. Target length: 800-1500 words.

Return ONLY the Markdown notes, no preamble or meta-commentary."""


# ---------------------------------------------------------------------------
# Notes Agent
# ---------------------------------------------------------------------------

class NotesAgent:
    """
    RAG-enabled agent that synthesizes study notes by combining
    prior context from the vector store with new information.

    Parameters
    ----------
    provider : GeminiProvider | None
        LLM provider (uses Gemini 2.5 Pro for synthesis).
    vector_store : NotesVectorStore | None
        ChromaDB-backed notes store for RAG retrieval.
    max_context_chunks : int
        Maximum number of prior note chunks to include in context.
    """

    def __init__(
        self,
        provider: Optional[GeminiProvider] = None,
        vector_store: Optional[NotesVectorStore] = None,
        max_context_chunks: int = 8,
    ) -> None:
        self._provider = provider or get_provider()
        self._store = vector_store or NotesVectorStore()
        self._max_chunks = max_context_chunks

    async def run(self, task: SubAgentTask) -> NotesResult:
        """
        Execute the Notes Agent pipeline.

        Parameters
        ----------
        task : SubAgentTask
            Must have ``agent == AgentRole.NOTES``.
            ``task.context`` may contain:
            - ``"resource_summaries"`` (list[str]): summaries from
              Reading/Lecture agents to incorporate.
            - ``"related_topics"`` (list[str]): topics to pull
              prior context for.

        Returns
        -------
        NotesResult
        """
        topic = task.topic

        try:
            # Step 1: Retrieve prior context from ChromaDB
            prior_chunks, prior_ids = self._retrieve_context(
                topic=topic,
                related_topics=task.context.get("related_topics", []),
            )

            # Step 2: Build the synthesis prompt
            prompt = self._build_prompt(
                topic=topic,
                prior_context=prior_chunks,
                resource_summaries=task.context.get("resource_summaries", []),
                instructions=task.instructions,
            )

            # Step 3: Generate notes with Gemini 2.5 Pro
            synthesized = self._provider.generate(
                prompt=prompt,
                weight=TaskWeight.HEAVY,
                system_instruction=_NOTES_SYSTEM,
                temperature=0.5,
                max_output_tokens=8192,
            )

            # Step 4: Store notes in ChromaDB
            topic_id = str(task.context.get("topic_id", topic))
            chunk_ids = self._store.add_note(
                topic_id=topic_id,
                topic_title=topic,
                content=synthesized,
                source="agent_generated",
            )

            return NotesResult(
                task_id=task.task_id,
                topic=topic,
                status=TaskStatus.COMPLETED,
                synthesized_notes=synthesized,
                prior_context_used=prior_ids,
                chunk_ids_stored=chunk_ids,
            )

        except Exception as e:
            return NotesResult(
                task_id=task.task_id,
                topic=topic,
                status=TaskStatus.FAILED,
                error=str(e),
            )

    # ── Private helpers ───────────────────────────────────────────────────

    def _retrieve_context(
        self,
        topic: str,
        related_topics: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Retrieve prior note chunks from ChromaDB.

        Returns
        -------
        tuple[list[str], list[str]]
            (chunk_texts, chunk_ids)
        """
        all_chunks: list[str] = []
        all_ids: list[str] = []

        # Query by the main topic
        results = self._store.query_by_topic(
            topic_title=topic,
            n_results=self._max_chunks,
        )
        for r in results:
            if r.get("document"):
                all_chunks.append(r["document"])
                all_ids.append(r["id"])

        # Also pull from related topics
        remaining = self._max_chunks - len(all_chunks)
        for related in related_topics[:3]:  # Cap at 3 related topics
            if remaining <= 0:
                break
            related_results = self._store.query_by_topic(
                topic_title=related,
                n_results=min(remaining, 3),
            )
            for r in related_results:
                if r.get("document") and r["id"] not in all_ids:
                    all_chunks.append(r["document"])
                    all_ids.append(r["id"])
                    remaining -= 1

        return all_chunks, all_ids

    def _build_prompt(
        self,
        topic: str,
        prior_context: list[str],
        resource_summaries: list[str],
        instructions: str,
    ) -> str:
        """Build the full synthesis prompt for the LLM."""
        sections: list[str] = [f"# Generate Study Notes: {topic}"]

        # Prior context
        if prior_context:
            sections.append(
                "\n## Your Prior Notes on This/Related Topics\n"
                "Use this context — build on it, reference connections, "
                "avoid repeating what's already covered:\n\n"
                + "\n---\n".join(prior_context)
            )

        # External resources
        if resource_summaries:
            sections.append(
                "\n## Resource Summaries to Incorporate\n"
                "Weave insights from these sources into your notes:\n\n"
                + "\n---\n".join(resource_summaries)
            )

        # Task-specific instructions
        if instructions:
            sections.append(f"\n## Additional Instructions\n{instructions}")

        return "\n\n".join(sections)
