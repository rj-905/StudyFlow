"""
ChromaDB Vector Store for the Notes Agent's RAG pipeline.

This module manages a single ChromaDB collection (``studyflow_notes``)
that stores chunked, embedded notes.  The Notes Agent writes to this
store after generating synthesised summaries, and reads from it to
weave prior context into new notes.

Embedding Function
------------------
By default we use ChromaDB's built-in ``default`` embedding function
(all-MiniLM-L6-v2 via sentence-transformers).  This can be swapped for
an OpenAI embedding function in production via ``get_openai_ef()``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from studyflow.models import NoteEmbeddingMeta

# Default persist directory lives alongside the SQLite DB
_DEFAULT_PERSIST_DIR: Path = (
    Path(__file__).resolve().parent.parent / "data" / "chroma_db"
)

_COLLECTION_NAME: str = "studyflow_notes"


# ---------------------------------------------------------------------------
# ChromaDB Client Factory
# ---------------------------------------------------------------------------

def get_chroma_client(
    persist_directory: Optional[str | Path] = None,
) -> chromadb.ClientAPI:
    """
    Return a persistent ChromaDB client.

    Parameters
    ----------
    persist_directory : str | Path | None
        Where ChromaDB stores its data on disk.
        Defaults to ``data/chroma_db`` relative to the project root.

    Returns
    -------
    chromadb.ClientAPI
    """
    persist_dir = str(persist_directory or _DEFAULT_PERSIST_DIR)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_notes_collection(
    client: Optional[chromadb.ClientAPI] = None,
    persist_directory: Optional[str | Path] = None,
) -> chromadb.Collection:
    """
    Return (or create) the ``studyflow_notes`` collection.

    Parameters
    ----------
    client : chromadb.ClientAPI | None
        An existing client.  If ``None``, one is created via
        :func:`get_chroma_client`.
    persist_directory : str | Path | None
        Forwarded to :func:`get_chroma_client` when ``client`` is
        ``None``.

    Returns
    -------
    chromadb.Collection
    """
    if client is None:
        client = get_chroma_client(persist_directory)

    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Notes Store  (high-level CRUD over ChromaDB)
# ---------------------------------------------------------------------------

class NotesVectorStore:
    """
    High-level wrapper around a ChromaDB collection for the Notes Agent.

    Responsibilities
    ----------------
    * **add_note** — chunk a note, embed it, and store with metadata.
    * **query_by_topic** — retrieve chunks most relevant to a topic.
    * **query_by_text** — free-text semantic search across all notes.

    Parameters
    ----------
    collection : chromadb.Collection
        The ChromaDB collection to operate on.
    chunk_size : int
        Maximum character length per chunk when splitting notes.
    chunk_overlap : int
        Overlap in characters between consecutive chunks.
    """

    def __init__(
        self,
        collection: Optional[chromadb.Collection] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self._collection = collection or get_notes_collection()
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    # ── Write ─────────────────────────────────────────────────────────────

    def add_note(
        self,
        topic_id: str,
        topic_title: str,
        content: str,
        source: str = "agent_generated",
    ) -> list[str]:
        """
        Chunk, embed, and persist a note into the vector store.

        Parameters
        ----------
        topic_id : str
            Stringified UUID of the parent topic.
        topic_title : str
            Human-readable topic name (stored as metadata).
        content : str
            The full note text to embed.
        source : str
            Origin label (``'agent_generated'`` or ``'user_uploaded'``).

        Returns
        -------
        list[str]
            The ChromaDB document IDs for each stored chunk.
        """
        chunks = self._split_into_chunks(content)
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for idx, chunk in enumerate(chunks):
            doc_id = self._deterministic_id(topic_id, idx, chunk)
            meta = NoteEmbeddingMeta(
                topic_id=topic_id,
                topic_title=topic_title,
                chunk_index=idx,
                source=source,
            )
            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append(meta.to_chroma_dict())

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        return ids

    # ── Read ──────────────────────────────────────────────────────────────

    def query_by_topic(
        self,
        topic_title: str,
        n_results: int = 5,
    ) -> list[dict]:
        """
        Retrieve the most relevant note chunks for a given topic.

        Uses semantic similarity against the topic title.

        Parameters
        ----------
        topic_title : str
            The topic to search for.
        n_results : int
            Maximum number of chunks to return.

        Returns
        -------
        list[dict]
            Each dict has keys ``id``, ``document``, ``metadata``,
            ``distance``.
        """
        results = self._collection.query(
            query_texts=[topic_title],
            n_results=n_results,
        )
        return self._flatten_results(results)

    def query_by_text(
        self,
        query: str,
        n_results: int = 5,
        topic_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Free-text semantic search across all stored notes.

        Parameters
        ----------
        query : str
            The natural-language search query.
        n_results : int
            Maximum results.
        topic_filter : str | None
            If provided, restrict results to chunks from this topic_id.

        Returns
        -------
        list[dict]
        """
        where_filter = (
            {"topic_id": topic_filter} if topic_filter else None
        )
        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
        )
        return self._flatten_results(results)

    def get_all_for_topic(self, topic_id: str) -> list[dict]:
        """
        Return **all** chunks stored for a specific ``topic_id``.

        Unlike the query methods, this is an exact-match filter
        (no embedding similarity involved).
        """
        results = self._collection.get(
            where={"topic_id": topic_id},
        )
        return self._flatten_get_results(results)

    def delete_topic_notes(self, topic_id: str) -> None:
        """Remove all note chunks for a given topic."""
        existing = self.get_all_for_topic(topic_id)
        if existing:
            self._collection.delete(ids=[e["id"] for e in existing])

    def count(self) -> int:
        """Return the total number of vectors in the collection."""
        return self._collection.count()

    # ── Private helpers ───────────────────────────────────────────────────

    def _split_into_chunks(self, text: str) -> list[str]:
        """
        Naïve character-level chunker with overlap.

        Production systems should use a recursive text splitter that
        respects sentence boundaries; this is adequate for Phase 1.
        """
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            chunks.append(text[start:end])
            start += self._chunk_size - self._chunk_overlap
        return chunks

    @staticmethod
    def _deterministic_id(topic_id: str, chunk_index: int, content: str) -> str:
        """
        Produce a stable document ID so that re-adding the same note
        is idempotent (upsert semantics).
        """
        raw = f"{topic_id}::{chunk_index}::{content[:128]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    @staticmethod
    def _flatten_results(results: dict) -> list[dict]:
        """Convert ChromaDB's nested query result into a flat list."""
        flat: list[dict] = []
        if not results or not results.get("ids"):
            return flat
        for i, doc_id in enumerate(results["ids"][0]):
            flat.append(
                {
                    "id": doc_id,
                    "document": results["documents"][0][i] if results.get("documents") else None,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else None,
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                }
            )
        return flat

    @staticmethod
    def _flatten_get_results(results: dict) -> list[dict]:
        """Convert ChromaDB's ``get()`` result into a flat list."""
        flat: list[dict] = []
        if not results or not results.get("ids"):
            return flat
        for i, doc_id in enumerate(results["ids"]):
            flat.append(
                {
                    "id": doc_id,
                    "document": results["documents"][i] if results.get("documents") else None,
                    "metadata": results["metadatas"][i] if results.get("metadatas") else None,
                }
            )
        return flat
