"""
SQLite schema definition and connection helpers for StudyFlow.

Tables
------
* ``topics``          — core learning topics with SM-2 scheduling metadata.
* ``prerequisites``   — directed dependency edges between topics.
* ``quiz_attempts``   — immutable log of every quiz interaction.
* ``resources``       — external learning resources linked to topics.

All DDL is idempotent (``IF NOT EXISTS``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

# Default database file lives alongside the package.
_DEFAULT_DB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "studyflow.db"


# ---------------------------------------------------------------------------
# DDL Statements
# ---------------------------------------------------------------------------

_TOPICS_DDL: str = """
CREATE TABLE IF NOT EXISTS topics (
    id                      TEXT PRIMARY KEY,
    title                   TEXT NOT NULL,
    description             TEXT,
    status                  TEXT NOT NULL DEFAULT 'not_started'
                                CHECK (status IN (
                                    'not_started', 'in_progress',
                                    'learned', 'reviewing'
                                )),

    -- SM-2 scheduling fields
    easiness_factor         REAL NOT NULL DEFAULT 2.5
                                CHECK (easiness_factor >= 1.3),
    interval_days           INTEGER NOT NULL DEFAULT 0
                                CHECK (interval_days >= 0),
    repetition_number       INTEGER NOT NULL DEFAULT 0
                                CHECK (repetition_number >= 0),
    next_review_date        TEXT,                           -- ISO-8601 date

    -- Performance
    latest_performance_score REAL
                                CHECK (latest_performance_score IS NULL
                                       OR (latest_performance_score >= 0
                                           AND latest_performance_score <= 5)),

    -- Timestamps (ISO-8601)
    learned_at              TEXT,
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at              TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_PREREQUISITES_DDL: str = """
CREATE TABLE IF NOT EXISTS prerequisites (
    id                TEXT PRIMARY KEY,
    topic_id          TEXT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    prerequisite_id   TEXT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,

    CHECK (topic_id != prerequisite_id),
    UNIQUE (topic_id, prerequisite_id)
);
"""

_QUIZ_ATTEMPTS_DDL: str = """
CREATE TABLE IF NOT EXISTS quiz_attempts (
    id              TEXT PRIMARY KEY,
    topic_id        TEXT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    difficulty      TEXT NOT NULL
                        CHECK (difficulty IN ('mcq', 'conceptual', 'applied')),
    quality_score   INTEGER NOT NULL
                        CHECK (quality_score >= 0 AND quality_score <= 5),
    attempted_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_RESOURCES_DDL: str = """
CREATE TABLE IF NOT EXISTS resources (
    id              TEXT PRIMARY KEY,
    topic_id        TEXT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    resource_type   TEXT NOT NULL
                        CHECK (resource_type IN (
                            'video', 'article', 'paper',
                            'textbook', 'notes'
                        )),
    title           TEXT NOT NULL,
    url             TEXT NOT NULL,
    relevance_score REAL
                        CHECK (relevance_score IS NULL
                               OR (relevance_score >= 0
                                   AND relevance_score <= 1)),
    metadata_json   TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# Indices for hot query paths
_INDICES_DDL: str = """
CREATE INDEX IF NOT EXISTS idx_topics_next_review
    ON topics(next_review_date);

CREATE INDEX IF NOT EXISTS idx_topics_status
    ON topics(status);

CREATE INDEX IF NOT EXISTS idx_quiz_attempts_topic
    ON quiz_attempts(topic_id, attempted_at DESC);

CREATE INDEX IF NOT EXISTS idx_prerequisites_topic
    ON prerequisites(topic_id);

CREATE INDEX IF NOT EXISTS idx_resources_topic
    ON resources(topic_id);
"""

_ALL_DDL: tuple[str, ...] = (
    _TOPICS_DDL,
    _PREREQUISITES_DDL,
    _QUIZ_ATTEMPTS_DDL,
    _RESOURCES_DDL,
    _INDICES_DDL,
)


# ---------------------------------------------------------------------------
# Connection Factory
# ---------------------------------------------------------------------------

def get_connection(db_path: Optional[str | Path] = None) -> sqlite3.Connection:
    """
    Return an SQLite connection with WAL mode and FK enforcement enabled.

    Parameters
    ----------
    db_path : str | Path | None
        Path to the ``.db`` file.  Defaults to ``data/studyflow.db``
        relative to the project root.  Pass ``":memory:"`` for tests.

    Returns
    -------
    sqlite3.Connection
        A connection with ``row_factory = sqlite3.Row`` for dict-like
        access to query results.
    """
    if db_path is None:
        db_path = _DEFAULT_DB_PATH

    path = Path(db_path) if not isinstance(db_path, Path) else db_path

    # Ensure parent directory exists (no-op for :memory:)
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ---------------------------------------------------------------------------
# Schema Initialization
# ---------------------------------------------------------------------------

def init_schema(conn: sqlite3.Connection) -> None:
    """
    Create all StudyFlow tables and indices idempotently.

    Parameters
    ----------
    conn : sqlite3.Connection
        An open database connection (as returned by :func:`get_connection`).
    """
    with conn:
        for ddl in _ALL_DDL:
            conn.executescript(ddl)


def init_database(db_path: Optional[str | Path] = None) -> sqlite3.Connection:
    """
    Convenience wrapper: open a connection **and** ensure the schema exists.

    Parameters
    ----------
    db_path : str | Path | None
        Forwarded to :func:`get_connection`.

    Returns
    -------
    sqlite3.Connection
    """
    conn = get_connection(db_path)
    init_schema(conn)
    return conn
