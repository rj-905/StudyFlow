"""
Pydantic data models for StudyFlow's persistence layers.

These models serve as the canonical schema definitions for:
  • SQLite relational tables (topics, quiz attempts, prerequisites)
  • Vector DB metadata (note embeddings)
  • Inter-agent communication payloads (Phase 2 extension point)

Every model enforces strict typing and validation so that agents can
trust the shape of data flowing through the system.
"""

from __future__ import annotations

import enum
from datetime import UTC, date, datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TopicStatus(str, enum.Enum):
    """Lifecycle status of a learning topic."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    LEARNED = "learned"
    REVIEWING = "reviewing"


class DifficultyLevel(str, enum.Enum):
    """Difficulty tiers for the Quiz Agent's 'Difficulty Dial'."""

    MCQ = "mcq"
    CONCEPTUAL = "conceptual"
    APPLIED = "applied"


class ResourceType(str, enum.Enum):
    """Types of learning resources sourced by sub-agents."""

    VIDEO = "video"
    ARTICLE = "article"
    PAPER = "paper"
    TEXTBOOK = "textbook"
    NOTES = "notes"


# ---------------------------------------------------------------------------
# Topic Model  (maps to `topics` SQLite table)
# ---------------------------------------------------------------------------

class TopicRecord(BaseModel):
    """
    Represents a single learning topic tracked in the SQLite database.

    The SM-2 scheduling fields (``easiness_factor``, ``interval_days``,
    ``repetition_number``, ``next_review_date``) are updated after every
    quiz attempt via the :class:`SM2Engine`.

    SM-2 Parameters
    ---------------
    * ``easiness_factor`` — *EF* in the SM-2 algorithm.  Minimum clamped
      to 1.3 to avoid degenerate schedules.
    * ``interval_days`` — *I(n)* : the current inter-repetition interval
      in days.
    * ``repetition_number`` — *n* : how many **consecutive** correct
      recalls have occurred.
    * ``next_review_date`` — the calendar date on which this topic should
      next be surfaced by ``get_due_reviews()``.
    """

    id: UUID = Field(default_factory=uuid4, description="Primary key (UUID4).")
    title: str = Field(..., min_length=1, max_length=512, description="Canonical topic title.")
    description: Optional[str] = Field(default=None, description="Optional long-form description.")
    status: TopicStatus = Field(default=TopicStatus.NOT_STARTED, description="Current lifecycle status.")

    # -- SM-2 scheduling fields --
    easiness_factor: float = Field(
        default=2.5,
        ge=1.3,
        description="SM-2 easiness factor (EF). Clamped >= 1.3.",
    )
    interval_days: int = Field(
        default=0,
        ge=0,
        description="Current inter-repetition interval in days (I(n)).",
    )
    repetition_number: int = Field(
        default=0,
        ge=0,
        description="Consecutive correct-recall count (n).",
    )
    next_review_date: Optional[date] = Field(
        default=None,
        description="Next scheduled review date. None means not yet scheduled.",
    )

    # -- Timestamps --
    learned_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the topic was first marked 'learned'.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Row creation timestamp (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last-modified timestamp (UTC).",
    )

    # -- Performance --
    latest_performance_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=5.0,
        description="Most recent quiz quality score (0-5, SM-2 scale).",
    )

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Prerequisite Edge  (maps to `prerequisites` SQLite table)
# ---------------------------------------------------------------------------

class PrerequisiteEdge(BaseModel):
    """
    Directed dependency edge:  ``topic_id`` **requires** ``prerequisite_id``.

    The Orchestrator queries these edges before dispatching a study plan
    to ensure foundational topics have been covered.
    """

    id: UUID = Field(default_factory=uuid4, description="Edge primary key.")
    topic_id: UUID = Field(..., description="The topic that has a prerequisite.")
    prerequisite_id: UUID = Field(..., description="The prerequisite topic.")

    @field_validator("prerequisite_id")
    @classmethod
    def no_self_loop(cls, v: UUID, info) -> UUID:
        """Prevent a topic from being its own prerequisite."""
        if "topic_id" in info.data and v == info.data["topic_id"]:
            raise ValueError("A topic cannot be its own prerequisite.")
        return v

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Quiz Attempt  (maps to `quiz_attempts` SQLite table)
# ---------------------------------------------------------------------------

class QuizAttemptRecord(BaseModel):
    """
    Immutable record of a single quiz interaction.

    ``quality_score`` follows the SM-2 convention:

    +-------+---------------------------------------------+
    | Score | Meaning                                     |
    +=======+=============================================+
    |   0   | Complete blackout                            |
    |   1   | Incorrect; correct answer seemed familiar    |
    |   2   | Incorrect; correct answer was easy to recall |
    |   3   | Correct with serious difficulty              |
    |   4   | Correct after hesitation                     |
    |   5   | Perfect recall                               |
    +-------+---------------------------------------------+
    """

    id: UUID = Field(default_factory=uuid4, description="Attempt primary key.")
    topic_id: UUID = Field(..., description="FK → topics.id")
    difficulty: DifficultyLevel = Field(..., description="Quiz tier used.")
    quality_score: int = Field(
        ...,
        ge=0,
        le=5,
        description="SM-2 quality score (0–5).",
    )
    attempted_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the attempt occurred (UTC).",
    )

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Learning Resource  (maps to `resources` SQLite table)
# ---------------------------------------------------------------------------

class ResourceRecord(BaseModel):
    """
    Metadata for an external learning resource (video, paper, article, etc.)
    sourced by the Lecture or Reading agents.
    """

    id: UUID = Field(default_factory=uuid4, description="Resource primary key.")
    topic_id: UUID = Field(..., description="FK → topics.id")
    resource_type: ResourceType = Field(..., description="Kind of resource.")
    title: str = Field(..., min_length=1, max_length=1024)
    url: str = Field(..., min_length=1, description="Canonical URL.")
    relevance_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Agent-computed relevance score in [0, 1].",
    )
    metadata_json: Optional[str] = Field(
        default=None,
        description="Arbitrary JSON blob (view count, duration, etc.).",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Row creation timestamp (UTC).",
    )

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Note Embedding Metadata  (stored alongside ChromaDB vectors)
# ---------------------------------------------------------------------------

class NoteEmbeddingMeta(BaseModel):
    """
    Metadata attached to each vector in the ChromaDB notes collection.

    ChromaDB stores vectors + metadata; this model guarantees the
    metadata shape so the Notes Agent can filter and weave context
    reliably.
    """

    topic_id: str = Field(..., description="Stringified UUID of the parent topic.")
    topic_title: str = Field(..., description="Human-readable topic title.")
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Position of this chunk within the full note.",
    )
    source: str = Field(
        default="agent_generated",
        description="Origin of the note (e.g., 'agent_generated', 'user_uploaded').",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-8601 creation timestamp.",
    )

    def to_chroma_dict(self) -> dict[str, str | int]:
        """Serialize to a flat dict compatible with ChromaDB metadata."""
        return self.model_dump()
