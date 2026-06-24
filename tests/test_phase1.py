"""
Phase 1 tests: Database layer, SM-2 engine, and Vector Store.

These tests use an in-memory SQLite database and a temporary ChromaDB
directory so they are fully isolated and repeatable.
"""

from __future__ import annotations

import math
import tempfile
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

import pytest

from studyflow.db.crud import TopicRepository
from studyflow.db.schema import init_database
from studyflow.models import (
    DifficultyLevel,
    NoteEmbeddingMeta,
    PrerequisiteEdge,
    QuizAttemptRecord,
    TopicRecord,
    TopicStatus,
)
from studyflow.scheduler.sm2_engine import SM2Engine, SM2Result, sm2_update


# ========================================================================
# Fixtures
# ========================================================================

@pytest.fixture
def db_conn():
    """In-memory SQLite connection with schema initialized."""
    conn = init_database(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def repo(db_conn):
    """TopicRepository backed by the in-memory DB."""
    return TopicRepository(db_conn)


@pytest.fixture
def engine():
    """Default SM2Engine instance."""
    return SM2Engine()


# ========================================================================
# 1. SQLite Schema & CRUD
# ========================================================================

class TestSchema:
    """Verify that all tables exist after init_database."""

    def test_tables_created(self, db_conn):
        tables = {
            row[0]
            for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "topics" in tables
        assert "prerequisites" in tables
        assert "quiz_attempts" in tables
        assert "resources" in tables

    def test_foreign_keys_enabled(self, db_conn):
        fk = db_conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1


class TestTopicCRUD:
    """Insert, update, and query topics."""

    def test_upsert_and_retrieve(self, repo):
        topic = TopicRecord(title="Big-O Notation")
        repo.upsert_topic(topic)

        fetched = repo.get_topic_by_id(topic.id)
        assert fetched is not None
        assert fetched.title == "Big-O Notation"
        assert fetched.status == TopicStatus.NOT_STARTED
        assert fetched.easiness_factor == 2.5

    def test_upsert_overwrites(self, repo):
        topic = TopicRecord(title="Hashing")
        repo.upsert_topic(topic)

        topic.status = TopicStatus.LEARNED
        topic.learned_at = datetime.now(UTC)
        repo.upsert_topic(topic)

        fetched = repo.get_topic_by_id(topic.id)
        assert fetched.status == TopicStatus.LEARNED
        assert fetched.learned_at is not None

    def test_list_topics_filter(self, repo):
        t1 = TopicRecord(title="A", status=TopicStatus.LEARNED)
        t2 = TopicRecord(title="B", status=TopicStatus.NOT_STARTED)
        repo.upsert_topic(t1)
        repo.upsert_topic(t2)

        learned = repo.list_topics(status=TopicStatus.LEARNED)
        assert len(learned) == 1
        assert learned[0].title == "A"

    def test_get_topic_by_title_case_insensitive(self, repo):
        repo.upsert_topic(TopicRecord(title="Amortized Analysis"))
        found = repo.get_topic_by_title("amortized analysis")
        assert found is not None
        assert found.title == "Amortized Analysis"


class TestPrerequisites:
    """Prerequisite edges and missing-prerequisite detection."""

    def test_add_and_query(self, repo):
        parent = TopicRecord(title="Binary Trees", status=TopicStatus.LEARNED)
        child = TopicRecord(title="AVL Trees")
        repo.upsert_topic(parent)
        repo.upsert_topic(child)

        edge = PrerequisiteEdge(topic_id=child.id, prerequisite_id=parent.id)
        repo.add_prerequisite(edge)

        prereqs = repo.get_prerequisites(child.id)
        assert len(prereqs) == 1
        assert prereqs[0].title == "Binary Trees"

    def test_missing_prerequisites(self, repo):
        prereq = TopicRecord(title="Calculus I", status=TopicStatus.NOT_STARTED)
        target = TopicRecord(title="Calculus II")
        repo.upsert_topic(prereq)
        repo.upsert_topic(target)

        edge = PrerequisiteEdge(topic_id=target.id, prerequisite_id=prereq.id)
        repo.add_prerequisite(edge)

        missing = repo.get_missing_prerequisites(target.id)
        assert len(missing) == 1
        assert missing[0].title == "Calculus I"

        # Now mark as learned → no longer missing
        prereq.status = TopicStatus.LEARNED
        prereq.learned_at = datetime.now(UTC)
        repo.upsert_topic(prereq)

        missing = repo.get_missing_prerequisites(target.id)
        assert len(missing) == 0


class TestDueReviews:
    """The get_due_reviews scheduler trigger."""

    def test_due_reviews_returns_overdue_topics(self, repo):
        t = TopicRecord(
            title="Graphs",
            status=TopicStatus.REVIEWING,
            next_review_date=date.today() - timedelta(days=1),
        )
        repo.upsert_topic(t)

        due = repo.get_due_reviews(date.today())
        assert len(due) == 1
        assert due[0].title == "Graphs"

    def test_future_reviews_not_returned(self, repo):
        t = TopicRecord(
            title="DP",
            status=TopicStatus.REVIEWING,
            next_review_date=date.today() + timedelta(days=5),
        )
        repo.upsert_topic(t)

        due = repo.get_due_reviews(date.today())
        assert len(due) == 0


class TestQuizAttempts:
    """Quiz attempt insertion and history retrieval."""

    def test_insert_and_retrieve(self, repo):
        topic = TopicRecord(title="Sorting")
        repo.upsert_topic(topic)

        attempt = QuizAttemptRecord(
            topic_id=topic.id,
            difficulty=DifficultyLevel.MCQ,
            quality_score=4,
        )
        repo.insert_quiz_attempt(attempt)

        history = repo.get_quiz_history(topic.id)
        assert len(history) == 1
        assert history[0].quality_score == 4
        assert history[0].difficulty == DifficultyLevel.MCQ


# ========================================================================
# 2. SM-2 Engine
# ========================================================================

class TestSM2Engine:
    """Core SM-2 algorithm correctness."""

    def test_first_review_perfect(self, engine):
        """First perfect recall → interval = 1 day."""
        result = engine.update(
            quality_score=5,
            easiness_factor=2.5,
            interval_days=0,
            repetition_number=0,
            review_date=date(2026, 6, 23),
        )
        assert result.interval_days == 1
        assert result.repetition_number == 1
        assert result.next_review_date == date(2026, 6, 24)

    def test_second_review_perfect(self, engine):
        """Second perfect recall → interval = 6 days."""
        result = engine.update(
            quality_score=5,
            easiness_factor=2.6,
            interval_days=1,
            repetition_number=1,
            review_date=date(2026, 6, 24),
        )
        assert result.interval_days == 6
        assert result.repetition_number == 2
        assert result.next_review_date == date(2026, 6, 30)

    def test_third_review_uses_ef(self, engine):
        """Third review → interval = round(prev_interval * EF)."""
        result = engine.update(
            quality_score=5,
            easiness_factor=2.6,
            interval_days=6,
            repetition_number=2,
            review_date=date(2026, 6, 30),
        )
        # EF updates: 2.6 + 0.1 = 2.7
        # interval = round(6 * 2.7) = round(16.2) = 16
        assert result.interval_days == 16
        assert result.repetition_number == 3
        assert result.next_review_date == date(2026, 7, 16)

    def test_failed_recall_resets(self, engine):
        """Quality < 3 → repetition resets to 0, interval = 1."""
        result = engine.update(
            quality_score=2,
            easiness_factor=2.5,
            interval_days=10,
            repetition_number=4,
            review_date=date(2026, 7, 1),
        )
        assert result.repetition_number == 0
        assert result.interval_days == 1
        assert result.next_review_date == date(2026, 7, 2)

    def test_ef_floor_clamped(self, engine):
        """Repeated failures should clamp EF to 1.3."""
        ef = 2.5
        for _ in range(20):
            result = engine.update(
                quality_score=0,
                easiness_factor=ef,
                interval_days=1,
                repetition_number=0,
            )
            ef = result.easiness_factor
        assert ef == 1.3

    def test_ef_increases_on_perfect(self, engine):
        """Quality=5 should increase EF."""
        result = engine.update(
            quality_score=5,
            easiness_factor=2.5,
            interval_days=0,
            repetition_number=0,
        )
        assert result.easiness_factor > 2.5

    def test_quality_score_validation(self, engine):
        with pytest.raises(ValueError, match="0–5"):
            engine.update(quality_score=6, easiness_factor=2.5,
                          interval_days=0, repetition_number=0)

    def test_module_level_shortcut(self):
        result = sm2_update(quality_score=4)
        assert isinstance(result, SM2Result)
        assert result.interval_days >= 1


class TestRetrievability:
    """Ebbinghaus forgetting-curve estimation."""

    def test_immediate_recall(self, engine):
        """Right after review, R = 1.0."""
        r = engine.retrievability(days_elapsed=0, stability=10.0)
        assert r == pytest.approx(1.0)

    def test_decay_over_time(self, engine):
        """R should decay monotonically."""
        r1 = engine.retrievability(days_elapsed=1, stability=10.0)
        r5 = engine.retrievability(days_elapsed=5, stability=10.0)
        assert 0 < r5 < r1 < 1.0

    def test_known_value(self, engine):
        """R(t=S) = e^{-1} ≈ 0.3679."""
        r = engine.retrievability(days_elapsed=10.0, stability=10.0)
        assert r == pytest.approx(math.exp(-1), rel=1e-9)

    def test_stability_from_interval(self, engine):
        """
        Verify S = -I / ln(R_target) with R_target = 0.9.

        For I=10:  S = -10 / ln(0.9) ≈ 94.91.
        """
        s = engine.stability_from_interval(10)
        expected = -10 / math.log(0.9)
        assert s == pytest.approx(expected, rel=1e-6)

    def test_stability_degenerate(self, engine):
        """Interval ≤ 0 → stability defaults to 1.0."""
        assert engine.stability_from_interval(0) == 1.0

    def test_invalid_stability(self, engine):
        with pytest.raises(ValueError, match="stability"):
            engine.retrievability(days_elapsed=1, stability=0)

    def test_negative_days_elapsed(self, engine):
        with pytest.raises(ValueError, match="days_elapsed"):
            engine.retrievability(days_elapsed=-1, stability=10)


# ========================================================================
# 3. Pydantic Model Validation
# ========================================================================

class TestModels:
    """Pydantic model constraints and edge cases."""

    def test_topic_ef_floor(self):
        with pytest.raises(Exception):
            TopicRecord(title="Test", easiness_factor=1.0)

    def test_self_loop_prerequisite(self):
        tid = uuid4()
        with pytest.raises(ValueError, match="own prerequisite"):
            PrerequisiteEdge(topic_id=tid, prerequisite_id=tid)

    def test_quiz_quality_bounds(self):
        with pytest.raises(Exception):
            QuizAttemptRecord(
                topic_id=uuid4(),
                difficulty=DifficultyLevel.MCQ,
                quality_score=6,
            )

    def test_note_embedding_meta_serialization(self):
        meta = NoteEmbeddingMeta(
            topic_id="abc-123",
            topic_title="Test Topic",
            chunk_index=0,
        )
        d = meta.to_chroma_dict()
        assert d["topic_id"] == "abc-123"
        assert d["chunk_index"] == 0
        assert "created_at" in d


# ========================================================================
# 4. ChromaDB Vector Store  (uses a temp directory)
# ========================================================================

class _DeterministicEmbeddingFunction:
    """
    A simple, offline embedding function for tests.

    Produces a fixed-length vector from a hash of the input text,
    avoiding any network calls to download ONNX/sentence-transformer
    models.  Implements the full ChromaDB EmbeddingFunction protocol.
    """

    _DIM: int = 64
    is_legacy: bool = True

    @staticmethod
    def name() -> str:
        """Return function name for ChromaDB's validation protocol."""
        return "default"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Core embedding logic: deterministic hash → float vector."""
        import hashlib
        embeddings = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = [(b / 127.5) - 1.0 for b in (h * (self._DIM // len(h) + 1))[:self._DIM]]
            embeddings.append(vec)
        return embeddings

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """ChromaDB calls this for document upserts."""
        return self._embed(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """ChromaDB calls this for query operations."""
        return self._embed(input)


class TestNotesVectorStore:
    """Integration tests for the ChromaDB-backed notes store."""

    @pytest.fixture
    def store(self, tmp_path):
        import chromadb
        from studyflow.memory.vector_store import NotesVectorStore

        # Use EphemeralClient with a deterministic embedding function
        # to avoid network calls to download the default ONNX model.
        client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(
            name="test_notes",
            embedding_function=_DeterministicEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
        return NotesVectorStore(collection=collection, chunk_size=50, chunk_overlap=10)

    def test_add_and_count(self, store):
        ids = store.add_note(
            topic_id="t1",
            topic_title="Linear Algebra",
            content="Vectors are elements of a vector space. "
                    "Matrices transform vectors linearly.",
        )
        assert len(ids) >= 1
        assert store.count() >= 1

    def test_query_by_topic(self, store):
        store.add_note(
            topic_id="t2",
            topic_title="Probability",
            content="Bayes theorem relates conditional probabilities.",
        )
        results = store.query_by_topic("Probability", n_results=3)
        assert len(results) >= 1
        assert "Bayes" in results[0]["document"]

    def test_query_by_text(self, store):
        store.add_note(
            topic_id="t3",
            topic_title="Graph Theory",
            content="A graph consists of vertices and edges.",
        )
        results = store.query_by_text("vertices edges", n_results=3)
        assert len(results) >= 1

    def test_delete_topic_notes(self, store):
        store.add_note(
            topic_id="t4",
            topic_title="Temp",
            content="Temporary note to delete.",
        )
        assert store.count() >= 1

        store.delete_topic_notes("t4")
        remaining = store.get_all_for_topic("t4")
        assert len(remaining) == 0

    def test_idempotent_upsert(self, store):
        """Adding the same note twice should not duplicate chunks."""
        store.add_note(topic_id="t5", topic_title="X", content="Hello world")
        count_after_first = store.count()

        store.add_note(topic_id="t5", topic_title="X", content="Hello world")
        count_after_second = store.count()

        assert count_after_first == count_after_second
