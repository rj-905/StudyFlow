"""
Phase 2 tests: A2A Schemas, Lecture Agent scoring, and Reading Agent scoring.

Tests the Pydantic communication models and the offline-testable parts
of the search tools (scoring functions, keyword extraction, XML parsing).
No actual API calls are made.
"""

from __future__ import annotations

import math
from uuid import uuid4

import pytest

from studyflow.agents.schemas import (
    AgentRole,
    LectureResult,
    NotesResult,
    OrchestratorResponse,
    PrerequisiteWarning,
    QuizDifficulty,
    QuizGradingResult,
    QuizQuestion,
    QuizResult,
    QuizSubmission,
    ReadingResult,
    StudyPlan,
    SubAgentTask,
    TaskStatus,
    TextResource,
    VideoResource,
)
from studyflow.agents.tools.lecture_search import (
    VideoScoringConfig,
    _duration_score,
    _extract_keywords,
    _keyword_relevance,
    _parse_iso8601_duration,
    _view_score,
    score_video,
)
from studyflow.agents.tools.reading_search import (
    _reconstruct_abstract,
    score_text_resource,
)


# ========================================================================
# 1. A2A Schema Validation
# ========================================================================

class TestStudyPlan:
    """Validate the Orchestrator's dispatch payload."""

    def test_create_study_plan(self):
        plan = StudyPlan(
            topic="Amortized Analysis",
            tasks=[
                SubAgentTask(
                    agent=AgentRole.LECTURE,
                    topic="Amortized Analysis",
                    instructions="Find video lectures on amortized analysis.",
                ),
                SubAgentTask(
                    agent=AgentRole.READING,
                    topic="Amortized Analysis",
                    instructions="Find academic papers and notes.",
                ),
            ],
        )
        assert plan.topic == "Amortized Analysis"
        assert len(plan.tasks) == 2
        assert plan.tasks[0].agent == AgentRole.LECTURE
        assert plan.prerequisite_warnings == []

    def test_study_plan_with_warnings(self):
        plan = StudyPlan(
            topic="AVL Trees",
            prerequisite_warnings=[
                PrerequisiteWarning(
                    topic="AVL Trees",
                    missing_prerequisites=["Binary Search Trees"],
                    message="You should learn Binary Search Trees first.",
                )
            ],
            tasks=[
                SubAgentTask(
                    agent=AgentRole.NOTES,
                    topic="AVL Trees",
                    instructions="Generate notes.",
                ),
            ],
        )
        assert len(plan.prerequisite_warnings) == 1
        assert "Binary Search Trees" in plan.prerequisite_warnings[0].missing_prerequisites

    def test_study_plan_requires_tasks(self):
        with pytest.raises(Exception):
            StudyPlan(topic="Test", tasks=[])

    def test_sub_agent_task_context(self):
        task = SubAgentTask(
            agent=AgentRole.NOTES,
            topic="Graphs",
            instructions="Synthesize notes",
            context={"prior_notes": ["chunk_1", "chunk_2"]},
            parameters={"max_length": 2000},
        )
        assert task.context["prior_notes"] == ["chunk_1", "chunk_2"]
        assert task.parameters["max_length"] == 2000


class TestResultSchemas:
    """Validate sub-agent result payloads."""

    def test_lecture_result(self):
        result = LectureResult(
            task_id=uuid4(),
            topic="Sorting Algorithms",
            videos=[
                VideoResource(
                    title="Merge Sort Explained",
                    url="https://youtube.com/watch?v=abc",
                    relevance_score=0.85,
                    view_count=100000,
                    duration_seconds=600,
                ),
            ],
        )
        assert result.status == TaskStatus.COMPLETED
        assert len(result.videos) == 1
        assert result.videos[0].relevance_score == 0.85

    def test_reading_result(self):
        result = ReadingResult(
            task_id=uuid4(),
            topic="Graph Theory",
            resources=[
                TextResource(
                    title="Introduction to Graph Theory",
                    url="https://arxiv.org/abs/1234.5678",
                    source_type="arxiv",
                    relevance_score=0.92,
                    is_open_access=True,
                ),
            ],
        )
        assert result.agent == AgentRole.READING
        assert result.resources[0].is_open_access is True

    def test_notes_result(self):
        result = NotesResult(
            task_id=uuid4(),
            topic="Dynamic Programming",
            synthesized_notes="# Dynamic Programming\n\nDP is an optimization technique...",
            prior_context_used=["chunk_abc", "chunk_def"],
            chunk_ids_stored=["new_chunk_1"],
        )
        assert "Dynamic Programming" in result.synthesized_notes
        assert len(result.prior_context_used) == 2

    def test_quiz_result(self):
        result = QuizResult(
            task_id=uuid4(),
            topic="Binary Trees",
            questions=[
                QuizQuestion(
                    difficulty=QuizDifficulty.MCQ,
                    question_text="What is the height of a balanced BST with n nodes?",
                    options=["O(n)", "O(log n)", "O(n log n)", "O(1)"],
                    correct_answer="O(log n)",
                    explanation="A balanced BST has height O(log n).",
                ),
                QuizQuestion(
                    difficulty=QuizDifficulty.CONCEPTUAL,
                    question_text="Explain the difference between BFS and DFS.",
                    correct_answer="BFS uses a queue, DFS uses a stack...",
                ),
            ],
        )
        assert len(result.questions) == 2
        assert result.questions[0].options is not None
        assert result.questions[1].options is None  # Open-ended

    def test_quiz_grading_result(self):
        grading = QuizGradingResult(
            topic_id=uuid4(),
            total_questions=5,
            correct_count=4,
            percentage=80.0,
            quality_score=4,
            difficulty=QuizDifficulty.MCQ,
            feedback=["Correct!", "Correct!", "Correct!", "Incorrect: ...", "Correct!"],
        )
        assert grading.quality_score == 4
        assert grading.percentage == 80.0

    def test_failed_result(self):
        result = LectureResult(
            task_id=uuid4(),
            topic="Test",
            status=TaskStatus.FAILED,
            error="YouTube API quota exceeded.",
        )
        assert result.status == TaskStatus.FAILED
        assert result.error is not None


class TestOrchestratorResponse:
    """Validate the aggregated response."""

    def test_aggregate_response(self):
        plan_id = uuid4()
        resp = OrchestratorResponse(
            plan_id=plan_id,
            topic="Hashing",
            lecture_result=LectureResult(task_id=uuid4(), topic="Hashing"),
            reading_result=ReadingResult(task_id=uuid4(), topic="Hashing"),
        )
        assert resp.plan_id == plan_id
        assert resp.lecture_result is not None
        assert resp.notes_result is None  # Not dispatched


# ========================================================================
# 2. Lecture Agent Scoring Functions
# ========================================================================

class TestVideoScoring:
    """Test the composite video relevance scorer."""

    def test_perfect_match(self):
        score = score_video(
            title="Amortized Analysis Tutorial",
            description="Learn about amortized analysis and data structures",
            view_count=1_000_000,
            duration_seconds=600,  # Ideal
            query_keywords=["amortized", "analysis"],
            max_views=1_000_000,
        )
        # Should be very high (close to 1.0)
        assert score > 0.8

    def test_no_keywords_match(self):
        score = score_video(
            title="Cooking Tutorial",
            description="How to make pasta",
            view_count=500,
            duration_seconds=300,
            query_keywords=["amortized", "analysis"],
            max_views=1000,
        )
        assert score < 0.5

    def test_keyword_relevance(self):
        assert _keyword_relevance("Big O Notation", "complexity analysis", ["big", "notation"]) == 1.0
        assert _keyword_relevance("Big O Notation", "", ["big", "xyz"]) == 0.5
        assert _keyword_relevance("Unrelated", "nothing", ["quantum", "physics"]) == 0.0
        assert _keyword_relevance("", "", []) == 0.0

    def test_view_score(self):
        assert _view_score(0, 1000) == pytest.approx(0.0)
        assert _view_score(1000, 1000) == pytest.approx(1.0)
        # Logarithmic scaling
        s1 = _view_score(100, 10000)
        s2 = _view_score(1000, 10000)
        assert 0 < s1 < s2 < 1.0

    def test_view_score_zero_max(self):
        assert _view_score(100, 0) == 0.0

    def test_duration_score(self):
        # Ideal duration should score 1.0
        assert _duration_score(600, 600, 300) == pytest.approx(1.0)
        # Very far from ideal should be low
        assert _duration_score(0, 600, 300) < 0.5
        # Symmetric around ideal
        s_short = _duration_score(300, 600, 300)
        s_long = _duration_score(900, 600, 300)
        assert s_short == pytest.approx(s_long)

    def test_custom_config(self):
        config = VideoScoringConfig(
            weight_keyword=0.8,
            weight_views=0.1,
            weight_duration=0.1,
        )
        score = score_video(
            title="Amortized Analysis",
            description="amortized analysis explained",
            view_count=100,
            duration_seconds=600,
            query_keywords=["amortized", "analysis"],
            max_views=1000,
            config=config,
        )
        # With high keyword weight, should still be high
        assert score > 0.7


class TestISO8601Duration:
    """Test YouTube ISO 8601 duration parsing."""

    def test_hours_minutes_seconds(self):
        assert _parse_iso8601_duration("PT1H2M10S") == 3730

    def test_minutes_only(self):
        assert _parse_iso8601_duration("PT15M") == 900

    def test_seconds_only(self):
        assert _parse_iso8601_duration("PT30S") == 30

    def test_zero(self):
        assert _parse_iso8601_duration("PT0S") == 0

    def test_invalid(self):
        assert _parse_iso8601_duration("invalid") == 0


class TestKeywordExtraction:
    """Test keyword extraction from queries."""

    def test_removes_stop_words(self):
        kw = _extract_keywords("What is the amortized analysis of dynamic arrays?")
        assert "amortized" in kw
        assert "analysis" in kw
        assert "dynamic" in kw
        assert "arrays" in kw
        assert "the" not in kw
        assert "is" not in kw

    def test_empty_query(self):
        assert _extract_keywords("") == []

    def test_single_letter_removed(self):
        kw = _extract_keywords("a b c data")
        assert kw == ["data"]


# ========================================================================
# 3. Reading Agent Scoring Functions
# ========================================================================

class TestTextScoring:
    """Test the text resource relevance scorer."""

    def test_arxiv_high_relevance(self):
        score = score_text_resource(
            title="Amortized Analysis of Data Structures",
            abstract="We present an amortized analysis framework...",
            query_keywords=["amortized", "analysis", "data", "structures"],
            source_type="arxiv",
            is_open_access=True,
        )
        # arXiv + all keywords + OA ≈ 0.6*1.0 + 0.3*1.0 + 0.1*1.0 = 1.0
        assert score == pytest.approx(1.0, abs=0.01)

    def test_non_oa_penalty(self):
        s_oa = score_text_resource("Title", "Abstract", ["title"], "arxiv", True)
        s_no = score_text_resource("Title", "Abstract", ["title"], "arxiv", False)
        assert s_oa > s_no

    def test_source_priority(self):
        s_arxiv = score_text_resource("X", "X", ["x"], "arxiv", True)
        s_other = score_text_resource("X", "X", ["x"], "other", True)
        assert s_arxiv > s_other

    def test_no_keyword_match(self):
        score = score_text_resource(
            title="Quantum Computing",
            abstract="Qubits and entanglement",
            query_keywords=["amortized", "analysis"],
            source_type="other",
            is_open_access=False,
        )
        # 0 keywords + low source + no OA
        assert score < 0.3


class TestAbstractReconstruction:
    """Test OpenAlex inverted-index abstract reconstruction."""

    def test_basic_reconstruction(self):
        inv_idx = {
            "Hello": [0],
            "world": [1],
            "from": [2],
            "OpenAlex": [3],
        }
        assert _reconstruct_abstract(inv_idx) == "Hello world from OpenAlex"

    def test_repeated_words(self):
        inv_idx = {
            "the": [0, 2],
            "cat": [1],
            "sat": [3],
        }
        assert _reconstruct_abstract(inv_idx) == "the cat the sat"

    def test_empty(self):
        assert _reconstruct_abstract({}) == ""
