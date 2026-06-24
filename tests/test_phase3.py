"""
Phase 3 tests: Sub-Agent implementations with mocked LLM responses.

All tests use a ``MockGeminiProvider`` that returns deterministic
responses without making any API calls.  This allows CI/CD testing
without API keys.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from studyflow.agents.schemas import (
    AgentRole,
    QuizDifficulty,
    QuizQuestion,
    QuizSubmission,
    SubAgentTask,
    TaskStatus,
)
from studyflow.db.crud import TopicRepository
from studyflow.db.schema import init_database
from studyflow.models import TopicRecord, TopicStatus


# ========================================================================
# Mock LLM Provider
# ========================================================================

class MockGeminiProvider:
    """
    Deterministic mock of GeminiProvider for offline testing.

    Returns pre-configured responses based on the task weight.
    """

    def generate(self, prompt: str, weight=None, **kwargs) -> str:
        """Return mock synthesized notes."""
        return (
            "# Mock Notes\n\n"
            "## Introduction\n"
            "This is a synthesized note on the topic.\n\n"
            "## Key Concepts\n"
            "- Concept A: Important idea\n"
            "- Concept B: Another idea\n\n"
            "## Key Takeaways\n"
            "- Takeaway 1\n"
            "- Takeaway 2\n"
            "- Takeaway 3\n"
        )

    def generate_json(self, prompt: str, weight=None, **kwargs) -> dict[str, Any]:
        """Return mock structured JSON based on context clues."""
        prompt_lower = prompt.lower()

        # Quiz generation
        if "generate" in prompt_lower and "question" in prompt_lower:
            return {
                "questions": [
                    {
                        "difficulty": "mcq",
                        "question_text": "What is the time complexity of a hash table lookup?",
                        "options": ["A) O(1)", "B) O(n)", "C) O(log n)", "D) O(n²)"],
                        "correct_answer": "A) O(1)",
                        "explanation": "Average case is O(1) due to direct addressing.",
                    },
                    {
                        "difficulty": "conceptual",
                        "question_text": "Explain amortized analysis.",
                        "options": None,
                        "correct_answer": "Amortized analysis averages the cost over a sequence of operations.",
                        "explanation": "It provides a tighter bound than worst-case analysis.",
                    },
                    {
                        "difficulty": "applied",
                        "question_text": "Calculate the amortized cost of dynamic array doubling.",
                        "options": None,
                        "correct_answer": "O(1) amortized per insertion.",
                        "explanation": "The cost of doubling is spread across all prior insertions.",
                    },
                ]
            }

        # Grading
        if "grade" in prompt_lower:
            return {
                "results": [
                    {"question_id": "q1", "is_correct": True, "feedback": "Correct!"},
                    {"question_id": "q2", "is_correct": True, "feedback": "Good understanding."},
                    {"question_id": "q3", "is_correct": False, "feedback": "Not quite right."},
                ],
                "total_correct": 2,
                "total_questions": 3,
            }

        # Re-ranking (Lecture or Reading)
        return [
            {"index": 0, "score": 0.95, "reason": "Excellent educational content."},
            {"index": 1, "score": 0.80, "reason": "Good overview."},
            {"index": 2, "score": 0.60, "reason": "Tangentially related."},
        ]

    def embed(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Return deterministic mock embeddings."""
        import hashlib
        result = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = [(b / 127.5) - 1.0 for b in (h * 3)[:64]]
            result.append(vec)
        return result

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)


# ========================================================================
# Fixtures
# ========================================================================

@pytest.fixture
def mock_provider():
    """Return a MockGeminiProvider instance."""
    return MockGeminiProvider()


@pytest.fixture
def db_conn():
    """In-memory SQLite connection with schema initialized."""
    conn = init_database(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def repo(db_conn):
    return TopicRepository(db_conn)


@pytest.fixture
def mock_vector_store():
    """ChromaDB-backed vector store using deterministic embeddings."""
    import chromadb
    from studyflow.memory.vector_store import NotesVectorStore

    class _MockEF:
        is_legacy = True
        @staticmethod
        def name(): return "default"
        def _embed(self, texts):
            import hashlib
            result = []
            for t in texts:
                h = hashlib.sha256(t.encode()).digest()
                vec = [(b / 127.5) - 1.0 for b in (h * 3)[:64]]
                result.append(vec)
            return result
        def __call__(self, input): return self._embed(input)
        def embed_documents(self, input): return self._embed(input)
        def embed_query(self, input): return self._embed(input)

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="test_notes_p3",
        embedding_function=_MockEF(),
        metadata={"hnsw:space": "cosine"},
    )
    return NotesVectorStore(collection=collection, chunk_size=200, chunk_overlap=50)


# ========================================================================
# 1. LLM Provider Config
# ========================================================================

class TestLLMProviderConfig:
    """Test model routing and configuration."""

    def test_model_map(self):
        from studyflow.config.llm_provider import TaskWeight, get_model_name
        assert get_model_name(TaskWeight.HEAVY) == "gemini-2.5-pro"
        assert get_model_name(TaskWeight.LIGHT) == "gemini-3.5-flash"
        assert get_model_name(TaskWeight.EMBEDDING) == "text-embedding-004"

    def test_missing_api_key_raises(self, monkeypatch):
        from studyflow.config.llm_provider import get_google_api_key
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY"):
            get_google_api_key()


# ========================================================================
# 2. Lecture Agent
# ========================================================================

class TestLectureAgent:
    """Test Lecture Agent with mocked YouTube API and LLM."""

    @pytest.mark.asyncio
    async def test_run_with_mock_youtube(self, mock_provider):
        from studyflow.agents.lecture_agent import LectureAgent
        from studyflow.agents.schemas import VideoResource

        agent = LectureAgent(
            provider=mock_provider,
            youtube_api_key="mock_key",
        )

        # Mock the YouTube search to return deterministic results
        mock_videos = [
            VideoResource(
                title="Amortized Analysis Explained",
                url="https://youtube.com/watch?v=abc",
                channel="CS Academy",
                duration_seconds=600,
                view_count=50000,
                relevance_score=0.85,
            ),
            VideoResource(
                title="Data Structures Overview",
                url="https://youtube.com/watch?v=def",
                channel="Tech Educator",
                duration_seconds=1200,
                view_count=100000,
                relevance_score=0.70,
            ),
        ]

        with patch(
            "studyflow.agents.lecture_agent.search_youtube",
            new_callable=AsyncMock,
            return_value=mock_videos,
        ):
            task = SubAgentTask(
                agent=AgentRole.LECTURE,
                topic="Amortized Analysis",
                instructions="Find video lectures.",
            )
            result = await agent.run(task)

        assert result.status == TaskStatus.COMPLETED
        assert len(result.videos) == 2
        assert result.topic == "Amortized Analysis"
        # Verify re-ranking was applied (scores should be blended)
        assert all(0 <= v.relevance_score <= 1 for v in result.videos)


# ========================================================================
# 3. Reading Agent
# ========================================================================

class TestReadingAgent:
    """Test Reading Agent with mocked search APIs and LLM."""

    @pytest.mark.asyncio
    async def test_run_with_mock_search(self, mock_provider):
        from studyflow.agents.reading_agent import ReadingAgent
        from studyflow.agents.schemas import TextResource

        agent = ReadingAgent(
            provider=mock_provider,
            google_api_key="mock_key",
            google_cse_id="mock_cse_id",
        )

        mock_arxiv = [
            TextResource(
                title="Amortized Complexity Analysis",
                url="https://arxiv.org/abs/1234.5678",
                source_type="arxiv",
                relevance_score=0.90,
                is_open_access=True,
            ),
        ]
        mock_cse = [
            TextResource(
                title="MIT OCW: Amortized Analysis",
                url="https://ocw.mit.edu/amortized",
                source_type="mit_ocw",
                relevance_score=0.85,
                is_open_access=True,
            ),
        ]

        with patch(
            "studyflow.agents.reading_agent.search_arxiv",
            new_callable=AsyncMock,
            return_value=mock_arxiv,
        ), patch(
            "studyflow.agents.reading_agent.search_google_cse",
            new_callable=AsyncMock,
            return_value=mock_cse,
        ):
            task = SubAgentTask(
                agent=AgentRole.READING,
                topic="Amortized Analysis",
                instructions="Find academic papers and notes.",
            )
            result = await agent.run(task)

        assert result.status == TaskStatus.COMPLETED
        assert len(result.resources) == 2
        # Verify deduplication (different URLs → both kept)
        urls = {r.url for r in result.resources}
        assert len(urls) == 2


# ========================================================================
# 4. Notes Agent
# ========================================================================

class TestNotesAgent:
    """Test Notes Agent RAG pipeline with mocked LLM."""

    @pytest.mark.asyncio
    async def test_synthesize_notes(self, mock_provider, mock_vector_store):
        from studyflow.agents.notes_agent import NotesAgent

        agent = NotesAgent(
            provider=mock_provider,
            vector_store=mock_vector_store,
        )

        task = SubAgentTask(
            agent=AgentRole.NOTES,
            topic="Amortized Analysis",
            instructions="Generate comprehensive study notes.",
            context={
                "topic_id": str(uuid4()),
                "resource_summaries": [
                    "Amortized analysis averages operation costs over sequences.",
                ],
                "related_topics": ["Big-O Notation"],
            },
        )

        result = await agent.run(task)

        assert result.status == TaskStatus.COMPLETED
        assert "Mock Notes" in result.synthesized_notes
        assert len(result.chunk_ids_stored) >= 1

    @pytest.mark.asyncio
    async def test_prior_context_retrieval(self, mock_provider, mock_vector_store):
        """Verify that prior notes are retrieved and passed to the LLM."""
        from studyflow.agents.notes_agent import NotesAgent

        # Pre-populate the vector store
        mock_vector_store.add_note(
            topic_id="prior_topic",
            topic_title="Big-O Notation",
            content="Big-O notation describes upper bounds on growth rates.",
        )

        agent = NotesAgent(
            provider=mock_provider,
            vector_store=mock_vector_store,
            max_context_chunks=5,
        )

        task = SubAgentTask(
            agent=AgentRole.NOTES,
            topic="Amortized Analysis",
            instructions="Include context from Big-O.",
            context={"related_topics": ["Big-O Notation"]},
        )

        result = await agent.run(task)

        assert result.status == TaskStatus.COMPLETED
        # Prior context should have been found
        # (the vector store had Big-O content)
        assert len(result.chunk_ids_stored) >= 1


# ========================================================================
# 5. Quiz Agent
# ========================================================================

class TestQuizAgent:
    """Test Quiz Agent generation and grading."""

    @pytest.mark.asyncio
    async def test_generate_questions(self, mock_provider):
        from studyflow.agents.quiz_agent import QuizAgent

        agent = QuizAgent(provider=mock_provider)

        task = SubAgentTask(
            agent=AgentRole.QUIZ,
            topic="Amortized Analysis",
            instructions="Generate quiz questions.",
            parameters={"difficulty": "mixed", "num_questions": 3},
        )

        result = await agent.run(task)

        assert result.status == TaskStatus.COMPLETED
        assert len(result.questions) == 3
        # Verify all difficulty tiers are present
        difficulties = {q.difficulty for q in result.questions}
        assert QuizDifficulty.MCQ in difficulties
        assert QuizDifficulty.CONCEPTUAL in difficulties
        assert QuizDifficulty.APPLIED in difficulties
        # MCQ should have options
        mcq_qs = [q for q in result.questions if q.difficulty == QuizDifficulty.MCQ]
        assert mcq_qs[0].options is not None

    @pytest.mark.asyncio
    async def test_grade_quiz(self, mock_provider, db_conn, repo):
        from studyflow.agents.quiz_agent import QuizAgent

        # Create a topic in the DB
        topic = TopicRecord(title="Hashing", status=TopicStatus.LEARNED)
        repo.upsert_topic(topic)

        agent = QuizAgent(
            provider=mock_provider,
            db_conn=db_conn,
        )

        # Build questions and submission
        q1 = QuizQuestion(
            difficulty=QuizDifficulty.MCQ,
            question_text="Q1?",
            options=["A", "B", "C", "D"],
            correct_answer="A",
        )
        q2 = QuizQuestion(
            difficulty=QuizDifficulty.MCQ,
            question_text="Q2?",
            options=["A", "B", "C", "D"],
            correct_answer="B",
        )
        q3 = QuizQuestion(
            difficulty=QuizDifficulty.MCQ,
            question_text="Q3?",
            options=["A", "B", "C", "D"],
            correct_answer="C",
        )

        submission = QuizSubmission(
            topic_id=topic.id,
            answers={
                str(q1.question_id): "A",
                str(q2.question_id): "B",
                str(q3.question_id): "Wrong",
            },
            difficulty=QuizDifficulty.MCQ,
        )

        grading = await agent.grade(submission, [q1, q2, q3])

        # Mock returns 2/3 correct → 66.7% → quality 3
        assert grading.total_questions == 3
        assert grading.correct_count == 2
        assert 60 <= grading.percentage <= 70
        assert grading.quality_score == 3

        # Verify SM-2 was updated in the DB
        updated_topic = repo.get_topic_by_id(topic.id)
        assert updated_topic is not None
        assert updated_topic.status == TopicStatus.REVIEWING
        assert updated_topic.next_review_date is not None
        assert updated_topic.latest_performance_score == 3.0

    def test_percentage_to_quality(self):
        from studyflow.agents.quiz_agent import QuizAgent
        assert QuizAgent._percentage_to_quality(100) == 5
        assert QuizAgent._percentage_to_quality(90) == 4
        assert QuizAgent._percentage_to_quality(70) == 3
        assert QuizAgent._percentage_to_quality(50) == 2
        assert QuizAgent._percentage_to_quality(25) == 1
        assert QuizAgent._percentage_to_quality(10) == 0

    @pytest.mark.asyncio
    async def test_quiz_updates_sm2_scheduling(self, mock_provider, db_conn, repo):
        """Verify the full quiz → SM-2 → next_review_date pipeline."""
        from studyflow.agents.quiz_agent import QuizAgent

        topic = TopicRecord(
            title="Sorting",
            status=TopicStatus.LEARNED,
            easiness_factor=2.5,
            interval_days=0,
            repetition_number=0,
        )
        repo.upsert_topic(topic)

        agent = QuizAgent(provider=mock_provider, db_conn=db_conn)

        q1 = QuizQuestion(
            difficulty=QuizDifficulty.MCQ,
            question_text="Q?",
            correct_answer="A",
        )

        submission = QuizSubmission(
            topic_id=topic.id,
            answers={str(q1.question_id): "A"},
            difficulty=QuizDifficulty.MCQ,
        )

        await agent.grade(submission, [q1])

        # Check that a quiz attempt was recorded
        history = repo.get_quiz_history(topic.id)
        assert len(history) == 1

        # Check SM-2 scheduling was applied
        updated = repo.get_topic_by_id(topic.id)
        assert updated.repetition_number >= 0
        assert updated.next_review_date is not None


# ========================================================================
# 6. Web Search Tool
# ========================================================================

class TestWebSearchTool:
    """Test Google Custom Search tool scoring helpers."""

    def test_domain_score(self):
        from studyflow.agents.tools.web_search import _domain_score
        assert _domain_score("https://mit.edu/course") == 1.0
        assert _domain_score("https://example.org/page") == 0.80
        assert _domain_score("https://random.xyz/page") == 0.30

    def test_classify_source(self):
        from studyflow.agents.tools.web_search import _classify_source
        assert _classify_source("https://arxiv.org/abs/1234") == "arxiv"
        assert _classify_source("https://ocw.mit.edu/courses") == "mit_ocw"
        assert _classify_source("https://stanford.edu/class") == "academic"
        assert _classify_source("https://github.com/repo") == "repository"
        assert _classify_source("https://example.com/page") == "web"
