"""
CRUD operations for the StudyFlow SQLite database.

Provides a ``TopicRepository`` class that encapsulates all data-access
logic for topics, prerequisites, quiz attempts, and resources.
Each method maps cleanly to the Pydantic models defined in
:mod:`studyflow.models`.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from typing import Optional, Sequence
from uuid import UUID

from studyflow.models import (
    DifficultyLevel,
    PrerequisiteEdge,
    QuizAttemptRecord,
    ResourceRecord,
    TopicRecord,
    TopicStatus,
)


class TopicRepository:
    """
    Data-access object for the ``topics``, ``prerequisites``,
    ``quiz_attempts``, and ``resources`` tables.

    Parameters
    ----------
    conn : sqlite3.Connection
        An initialised connection (schema must already exist).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ── Topics ────────────────────────────────────────────────────────────

    def upsert_topic(self, topic: TopicRecord) -> None:
        """Insert or update a topic row from a Pydantic model."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO topics (
                    id, title, description, status,
                    easiness_factor, interval_days, repetition_number,
                    next_review_date, latest_performance_score,
                    learned_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title                   = excluded.title,
                    description             = excluded.description,
                    status                  = excluded.status,
                    easiness_factor         = excluded.easiness_factor,
                    interval_days           = excluded.interval_days,
                    repetition_number       = excluded.repetition_number,
                    next_review_date        = excluded.next_review_date,
                    latest_performance_score = excluded.latest_performance_score,
                    learned_at              = excluded.learned_at,
                    updated_at              = excluded.updated_at
                """,
                (
                    str(topic.id),
                    topic.title,
                    topic.description,
                    topic.status.value,
                    topic.easiness_factor,
                    topic.interval_days,
                    topic.repetition_number,
                    topic.next_review_date.isoformat() if topic.next_review_date else None,
                    topic.latest_performance_score,
                    topic.learned_at.isoformat() if topic.learned_at else None,
                    topic.created_at.isoformat(),
                    topic.updated_at.isoformat(),
                ),
            )

    def get_topic_by_id(self, topic_id: UUID) -> Optional[TopicRecord]:
        """Fetch a single topic by primary key."""
        row = self._conn.execute(
            "SELECT * FROM topics WHERE id = ?", (str(topic_id),)
        ).fetchone()
        return self._row_to_topic(row) if row else None

    def get_topic_by_title(self, title: str) -> Optional[TopicRecord]:
        """Case-insensitive lookup by title."""
        row = self._conn.execute(
            "SELECT * FROM topics WHERE LOWER(title) = LOWER(?)", (title,)
        ).fetchone()
        return self._row_to_topic(row) if row else None

    def list_topics(
        self, status: Optional[TopicStatus] = None
    ) -> list[TopicRecord]:
        """Return all topics, optionally filtered by status."""
        if status:
            rows = self._conn.execute(
                "SELECT * FROM topics WHERE status = ? ORDER BY title",
                (status.value,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM topics ORDER BY title"
            ).fetchall()
        return [self._row_to_topic(r) for r in rows]

    def get_due_reviews(self, as_of: date) -> list[TopicRecord]:
        """
        Return topics whose ``next_review_date <= as_of``.

        This is the primary trigger for the Quiz Agent's spaced-repetition
        loop.

        Parameters
        ----------
        as_of : date
            The reference date (typically ``date.today()``).
        """
        rows = self._conn.execute(
            """
            SELECT * FROM topics
            WHERE next_review_date IS NOT NULL
              AND next_review_date <= ?
            ORDER BY next_review_date ASC
            """,
            (as_of.isoformat(),),
        ).fetchall()
        return [self._row_to_topic(r) for r in rows]

    # ── Prerequisites ─────────────────────────────────────────────────────

    def add_prerequisite(self, edge: PrerequisiteEdge) -> None:
        """Insert a prerequisite edge (idempotent via UNIQUE constraint)."""
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO prerequisites (id, topic_id, prerequisite_id)
                VALUES (?, ?, ?)
                """,
                (str(edge.id), str(edge.topic_id), str(edge.prerequisite_id)),
            )

    def get_prerequisites(self, topic_id: UUID) -> list[TopicRecord]:
        """Return the prerequisite topics for a given topic."""
        rows = self._conn.execute(
            """
            SELECT t.* FROM topics t
            JOIN prerequisites p ON t.id = p.prerequisite_id
            WHERE p.topic_id = ?
            ORDER BY t.title
            """,
            (str(topic_id),),
        ).fetchall()
        return [self._row_to_topic(r) for r in rows]

    def get_missing_prerequisites(self, topic_id: UUID) -> list[TopicRecord]:
        """
        Return prerequisites of ``topic_id`` that the student has **not**
        yet reached ``learned`` or ``reviewing`` status.

        The Orchestrator calls this before dispatching a study plan.
        """
        rows = self._conn.execute(
            """
            SELECT t.* FROM topics t
            JOIN prerequisites p ON t.id = p.prerequisite_id
            WHERE p.topic_id = ?
              AND t.status NOT IN ('learned', 'reviewing')
            ORDER BY t.title
            """,
            (str(topic_id),),
        ).fetchall()
        return [self._row_to_topic(r) for r in rows]

    # ── Quiz Attempts ─────────────────────────────────────────────────────

    def insert_quiz_attempt(self, attempt: QuizAttemptRecord) -> None:
        """Append an immutable quiz-attempt record."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO quiz_attempts (
                    id, topic_id, difficulty, quality_score, attempted_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(attempt.id),
                    str(attempt.topic_id),
                    attempt.difficulty.value,
                    attempt.quality_score,
                    attempt.attempted_at.isoformat(),
                ),
            )

    def get_quiz_history(
        self, topic_id: UUID, limit: int = 20
    ) -> list[QuizAttemptRecord]:
        """Most-recent-first quiz attempts for a topic."""
        rows = self._conn.execute(
            """
            SELECT * FROM quiz_attempts
            WHERE topic_id = ?
            ORDER BY attempted_at DESC
            LIMIT ?
            """,
            (str(topic_id), limit),
        ).fetchall()
        return [
            QuizAttemptRecord(
                id=UUID(r["id"]),
                topic_id=UUID(r["topic_id"]),
                difficulty=DifficultyLevel(r["difficulty"]),
                quality_score=r["quality_score"],
                attempted_at=datetime.fromisoformat(r["attempted_at"]),
            )
            for r in rows
        ]

    # ── Resources ─────────────────────────────────────────────────────────

    def insert_resource(self, resource: ResourceRecord) -> None:
        """Store a new learning resource linked to a topic."""
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO resources (
                    id, topic_id, resource_type, title, url,
                    relevance_score, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(resource.id),
                    str(resource.topic_id),
                    resource.resource_type.value,
                    resource.title,
                    resource.url,
                    resource.relevance_score,
                    resource.metadata_json,
                    resource.created_at.isoformat(),
                ),
            )

    def get_resources_for_topic(self, topic_id: UUID) -> list[ResourceRecord]:
        """Retrieve all resources associated with a topic."""
        rows = self._conn.execute(
            """
            SELECT * FROM resources
            WHERE topic_id = ?
            ORDER BY relevance_score DESC NULLS LAST
            """,
            (str(topic_id),),
        ).fetchall()
        return [
            ResourceRecord(
                id=UUID(r["id"]),
                topic_id=UUID(r["topic_id"]),
                resource_type=r["resource_type"],
                title=r["title"],
                url=r["url"],
                relevance_score=r["relevance_score"],
                metadata_json=r["metadata_json"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _row_to_topic(row: sqlite3.Row) -> TopicRecord:
        """Convert a raw ``sqlite3.Row`` into a ``TopicRecord``."""
        return TopicRecord(
            id=UUID(row["id"]),
            title=row["title"],
            description=row["description"],
            status=TopicStatus(row["status"]),
            easiness_factor=row["easiness_factor"],
            interval_days=row["interval_days"],
            repetition_number=row["repetition_number"],
            next_review_date=(
                date.fromisoformat(row["next_review_date"])
                if row["next_review_date"]
                else None
            ),
            latest_performance_score=row["latest_performance_score"],
            learned_at=(
                datetime.fromisoformat(row["learned_at"])
                if row["learned_at"]
                else None
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
