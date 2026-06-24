"""
Agent-to-Agent (A2A) Communication Schemas for StudyFlow.

These Pydantic models define the **structured JSON contracts** that the
Orchestrator Agent uses to dispatch tasks to sub-agents and that
sub-agents use to return results.  All inter-agent communication passes
through these models, enforcing type safety at every boundary.

Protocol Flow
=============

1. User submits a topic string.
2. Orchestrator creates a :class:`StudyPlan` and dispatches
   :class:`SubAgentTask` items to each sub-agent.
3. Each sub-agent returns its typed result:
   - Lecture Agent  → :class:`LectureResult`
   - Reading Agent  → :class:`ReadingResult`
   - Notes Agent    → :class:`NotesResult`
   - Quiz Agent     → :class:`QuizResult`
4. The Orchestrator collects results into an :class:`OrchestratorResponse`.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AgentRole(str, enum.Enum):
    """Identifies which sub-agent a task is routed to."""

    ORCHESTRATOR = "orchestrator"
    LECTURE = "lecture"
    READING = "reading"
    NOTES = "notes"
    QUIZ = "quiz"


class TaskStatus(str, enum.Enum):
    """Lifecycle status of a dispatched sub-agent task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class QuizDifficulty(str, enum.Enum):
    """Difficulty tiers for the Quiz Agent's 'Difficulty Dial'."""

    MCQ = "mcq"
    CONCEPTUAL = "conceptual"
    APPLIED = "applied"


# ---------------------------------------------------------------------------
# Orchestrator → Sub-Agent  (Dispatch)
# ---------------------------------------------------------------------------

class SubAgentTask(BaseModel):
    """
    A single task dispatched from the Orchestrator to a sub-agent.

    The ``context`` field carries any upstream data (e.g., prior notes,
    prerequisite warnings) that the sub-agent should consider.
    """

    task_id: UUID = Field(default_factory=uuid4, description="Unique task identifier.")
    agent: AgentRole = Field(..., description="Target sub-agent for this task.")
    topic: str = Field(..., min_length=1, description="The learning topic.")
    instructions: str = Field(
        ...,
        description="Specific instructions for the sub-agent.",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Upstream context data (prior notes, prerequisites, etc.).",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific parameters (e.g., quiz difficulty, max results).",
    )

    model_config = ConfigDict(from_attributes=True)


class PrerequisiteWarning(BaseModel):
    """Warning issued when a topic has unmet prerequisite dependencies."""

    topic: str = Field(..., description="The requested topic.")
    missing_prerequisites: list[str] = Field(
        ...,
        description="Titles of prerequisite topics not yet learned.",
    )
    message: str = Field(
        ...,
        description="Human-readable warning message.",
    )


class StudyPlan(BaseModel):
    """
    The Orchestrator's execution plan dispatched to sub-agents.

    This is the top-level A2A payload: it contains the topic, any
    prerequisite warnings, and the list of sub-agent tasks.
    """

    plan_id: UUID = Field(default_factory=uuid4, description="Unique plan identifier.")
    topic: str = Field(..., min_length=1, description="The learning topic.")
    topic_id: Optional[UUID] = Field(
        default=None,
        description="FK → topics.id if the topic already exists in the DB.",
    )
    prerequisite_warnings: list[PrerequisiteWarning] = Field(
        default_factory=list,
        description="Any unmet dependency warnings.",
    )
    tasks: list[SubAgentTask] = Field(
        ...,
        min_length=1,
        description="Sub-agent tasks to execute.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Plan creation timestamp.",
    )

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Sub-Agent → Orchestrator  (Results)
# ---------------------------------------------------------------------------

class VideoResource(BaseModel):
    """A single video resource found by the Lecture Agent."""

    title: str = Field(..., description="Video title.")
    url: str = Field(..., description="Video URL.")
    channel: Optional[str] = Field(default=None, description="Channel/creator name.")
    duration_seconds: Optional[int] = Field(default=None, ge=0, description="Video duration in seconds.")
    view_count: Optional[int] = Field(default=None, ge=0, description="View count.")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Computed relevance score in [0, 1].",
    )
    description: Optional[str] = Field(default=None, description="Video description snippet.")


class LectureResult(BaseModel):
    """Result payload returned by the Lecture Agent."""

    task_id: UUID = Field(..., description="Matches the dispatched SubAgentTask.task_id.")
    agent: AgentRole = Field(default=AgentRole.LECTURE)
    status: TaskStatus = Field(default=TaskStatus.COMPLETED)
    topic: str = Field(..., description="The topic searched for.")
    videos: list[VideoResource] = Field(
        default_factory=list,
        description="Ranked list of video resources.",
    )
    error: Optional[str] = Field(default=None, description="Error message if failed.")

    model_config = ConfigDict(from_attributes=True)


class TextResource(BaseModel):
    """A single text resource found by the Reading Agent."""

    title: str = Field(..., description="Resource title.")
    url: str = Field(..., description="Resource URL.")
    source_type: str = Field(
        ...,
        description="Source category (e.g., 'arxiv', 'mit_ocw', 'textbook', 'blog').",
    )
    authors: Optional[list[str]] = Field(default=None, description="Author names.")
    abstract: Optional[str] = Field(default=None, description="Abstract or summary.")
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Computed relevance score in [0, 1].",
    )
    is_open_access: bool = Field(
        default=True,
        description="Whether the resource is freely accessible.",
    )


class ReadingResult(BaseModel):
    """Result payload returned by the Reading Agent."""

    task_id: UUID = Field(..., description="Matches the dispatched SubAgentTask.task_id.")
    agent: AgentRole = Field(default=AgentRole.READING)
    status: TaskStatus = Field(default=TaskStatus.COMPLETED)
    topic: str = Field(..., description="The topic searched for.")
    resources: list[TextResource] = Field(
        default_factory=list,
        description="Ranked list of text resources.",
    )
    error: Optional[str] = Field(default=None, description="Error message if failed.")

    model_config = ConfigDict(from_attributes=True)


class NotesResult(BaseModel):
    """Result payload returned by the Notes Agent."""

    task_id: UUID = Field(..., description="Matches the dispatched SubAgentTask.task_id.")
    agent: AgentRole = Field(default=AgentRole.NOTES)
    status: TaskStatus = Field(default=TaskStatus.COMPLETED)
    topic: str = Field(..., description="The topic notes were generated for.")
    synthesized_notes: str = Field(
        default="",
        description="The full synthesized markdown notes.",
    )
    prior_context_used: list[str] = Field(
        default_factory=list,
        description="IDs of prior note chunks retrieved from the vector store.",
    )
    chunk_ids_stored: list[str] = Field(
        default_factory=list,
        description="ChromaDB IDs of the newly stored note chunks.",
    )
    error: Optional[str] = Field(default=None, description="Error message if failed.")

    model_config = ConfigDict(from_attributes=True)


class QuizQuestion(BaseModel):
    """A single quiz question generated by the Quiz Agent."""

    question_id: UUID = Field(default_factory=uuid4)
    difficulty: QuizDifficulty = Field(..., description="Difficulty tier.")
    question_text: str = Field(..., description="The question body.")
    options: Optional[list[str]] = Field(
        default=None,
        description="MCQ answer options (None for open-ended questions).",
    )
    correct_answer: str = Field(..., description="The correct answer.")
    explanation: str = Field(
        default="",
        description="Explanation of the correct answer.",
    )


class QuizResult(BaseModel):
    """Result payload returned by the Quiz Agent."""

    task_id: UUID = Field(..., description="Matches the dispatched SubAgentTask.task_id.")
    agent: AgentRole = Field(default=AgentRole.QUIZ)
    status: TaskStatus = Field(default=TaskStatus.COMPLETED)
    topic: str = Field(..., description="The topic quizzed on.")
    questions: list[QuizQuestion] = Field(
        default_factory=list,
        description="Generated quiz questions.",
    )
    error: Optional[str] = Field(default=None, description="Error message if failed.")

    model_config = ConfigDict(from_attributes=True)


class QuizSubmission(BaseModel):
    """
    User's answers submitted back to the Quiz Agent for grading.

    After grading, the Quiz Agent reports performance to the Scheduler
    to update SM-2 parameters.
    """

    topic_id: UUID = Field(..., description="FK → topics.id")
    answers: dict[str, str] = Field(
        ...,
        description="Mapping of question_id → user_answer.",
    )
    difficulty: QuizDifficulty = Field(..., description="Difficulty tier taken.")


class QuizGradingResult(BaseModel):
    """
    Grading output returned after evaluating a QuizSubmission.

    ``quality_score`` is the SM-2 quality score (0–5) derived from
    the percentage of correct answers:
    - 100%  → 5
    - 80%+  → 4
    - 60%+  → 3
    - 40%+  → 2
    - 20%+  → 1
    - <20%  → 0
    """

    topic_id: UUID = Field(..., description="FK → topics.id")
    total_questions: int = Field(..., ge=0)
    correct_count: int = Field(..., ge=0)
    percentage: float = Field(..., ge=0.0, le=100.0)
    quality_score: int = Field(
        ...,
        ge=0,
        le=5,
        description="SM-2 quality score derived from percentage.",
    )
    difficulty: QuizDifficulty = Field(...)
    feedback: list[str] = Field(
        default_factory=list,
        description="Per-question feedback messages.",
    )


# ---------------------------------------------------------------------------
# Orchestrator Aggregate Response
# ---------------------------------------------------------------------------

class OrchestratorResponse(BaseModel):
    """
    The final aggregated response from the Orchestrator.

    Collects results from all sub-agents into a single payload
    returned to the user.
    """

    plan_id: UUID = Field(..., description="The plan this response fulfills.")
    topic: str = Field(..., description="The learning topic.")
    prerequisite_warnings: list[PrerequisiteWarning] = Field(default_factory=list)
    lecture_result: Optional[LectureResult] = Field(default=None)
    reading_result: Optional[ReadingResult] = Field(default=None)
    notes_result: Optional[NotesResult] = Field(default=None)
    quiz_result: Optional[QuizResult] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(from_attributes=True)
