"""
Central LLM Provider for StudyFlow.

Routes tasks to the appropriate Gemini model based on complexity:

* **Heavy reasoning** (Orchestrator, Notes synthesis, Quiz generation)
  → ``gemini-2.5-pro``
* **Lightweight tasks** (re-ranking, grading, summarisation)
  → ``gemini-3.5-flash``
* **Embeddings** (ChromaDB vector store)
  → ``text-embedding-004``

All calls enforce structured JSON output where applicable.
"""

from __future__ import annotations

import enum
import json
import os
from pathlib import Path
from typing import Any, Optional, TypeVar

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class TaskWeight(str, enum.Enum):
    """Categorises a task by computational weight for model routing."""

    HEAVY = "heavy"       # Complex reasoning: planning, synthesis, generation
    LIGHT = "light"       # Lightweight: ranking, grading, classification
    EMBEDDING = "embedding"


# Model name mapping
_MODEL_MAP: dict[TaskWeight, str] = {
    TaskWeight.HEAVY: "gemini-2.5-pro",
    TaskWeight.LIGHT: "gemini-3.5-flash",
    TaskWeight.EMBEDDING: "text-embedding-004",
}


def get_model_name(weight: TaskWeight) -> str:
    """Return the Gemini model name for a given task weight."""
    return _MODEL_MAP[weight]


# ---------------------------------------------------------------------------
# API Key Helpers
# ---------------------------------------------------------------------------

def get_google_api_key() -> str:
    """Return the Google API key from environment, raising if missing."""
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Get one at https://aistudio.google.com/apikey "
            "and add it to your .env file."
        )
    return key


def get_youtube_api_key() -> str:
    """Return the YouTube Data API key from environment."""
    key = os.environ.get("YOUTUBE_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "YOUTUBE_API_KEY is not set. "
            "Enable YouTube Data API v3 in Google Cloud Console."
        )
    return key


def get_google_cse_id() -> str:
    """Return the Google Custom Search Engine ID from environment."""
    cse_id = os.environ.get("GOOGLE_CSE_ID", "")
    if not cse_id:
        raise EnvironmentError(
            "GOOGLE_CSE_ID is not set. "
            "Create a Programmable Search Engine at "
            "https://programmablesearchengine.google.com"
        )
    return cse_id


# ---------------------------------------------------------------------------
# Gemini Client Wrapper
# ---------------------------------------------------------------------------

class GeminiProvider:
    """
    Unified interface to Google Gemini generative models.

    Wraps ``google.genai`` and provides:

    * **generate()** — free-form text generation.
    * **generate_json()** — structured JSON output (parsed into a dict).
    * **embed()** — text embedding via ``text-embedding-004``.

    Parameters
    ----------
    api_key : str | None
        Google API key.  Defaults to ``GOOGLE_API_KEY`` env var.

    Examples
    --------
    >>> provider = GeminiProvider()
    >>> result = provider.generate("Explain amortized analysis.", weight=TaskWeight.HEAVY)
    >>> print(result)
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        from google import genai

        self._api_key = api_key or get_google_api_key()
        self._client = genai.Client(api_key=self._api_key)

    # ── Text Generation ───────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        weight: TaskWeight = TaskWeight.HEAVY,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 8192,
    ) -> str:
        """
        Generate a free-form text response.

        Parameters
        ----------
        prompt : str
            The user/task prompt.
        weight : TaskWeight
            Determines which model to use.
        system_instruction : str | None
            System-level instruction prepended to the conversation.
        temperature : float
            Sampling temperature (0.0 = deterministic, 1.0 = creative).
        max_output_tokens : int
            Maximum response length.

        Returns
        -------
        str
            The model's text response.
        """
        from google.genai import types
        model_name = get_model_name(weight)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction
        )

        response = self._client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        return response.text

    def generate_json(
        self,
        prompt: str,
        weight: TaskWeight = TaskWeight.HEAVY,
        system_instruction: Optional[str] = None,
        temperature: float = 0.3,
        max_output_tokens: int = 8192,
    ) -> dict[str, Any]:
        """
        Generate a structured JSON response.

        Uses ``response_mime_type="application/json"`` to enforce
        valid JSON output from the model.

        Parameters
        ----------
        prompt : str
            The prompt (should describe the desired JSON structure).
        weight : TaskWeight
            Determines which model to use.
        system_instruction : str | None
            System-level instruction.
        temperature : float
            Lower temperature for more deterministic JSON.
        max_output_tokens : int
            Maximum response length.

        Returns
        -------
        dict[str, Any]
            Parsed JSON response.

        Raises
        ------
        ValueError
            If the model output is not valid JSON.
        """
        from google.genai import types
        model_name = get_model_name(weight)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            system_instruction=system_instruction
        )

        response = self._client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        try:
            return json.loads(response.text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Model did not return valid JSON. Raw output: {response.text[:500]}"
            ) from e

    # ── Embeddings ────────────────────────────────────────────────────────

    def embed(
        self,
        texts: list[str],
        task_type: str = "retrieval_document",
    ) -> list[list[float]]:
        """
        Generate embeddings using ``text-embedding-004``.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.
        task_type : str
            One of ``"retrieval_document"``, ``"retrieval_query"``,
            ``"semantic_similarity"``, ``"classification"``,
            ``"clustering"``.

        Returns
        -------
        list[list[float]]
            Embedding vectors (768-dimensional).
        """
        from google.genai import types
        model_name = get_model_name(TaskWeight.EMBEDDING)
        
        task_type_upper = task_type.upper()
        config = types.EmbedContentConfig(task_type=task_type_upper)

        result = self._client.models.embed_content(
            model=model_name,
            contents=texts,
            config=config,
        )
        
        embeddings = [item.values for item in result.embeddings]
        
        if texts and isinstance(embeddings[0], float):
            return [embeddings]
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text (optimised for retrieval)."""
        return self.embed([text], task_type="retrieval_query")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents (optimised for storage)."""
        return self.embed(texts, task_type="retrieval_document")


# ---------------------------------------------------------------------------
# ChromaDB-compatible Embedding Function
# ---------------------------------------------------------------------------

class GoogleEmbeddingFunction:
    """
    ChromaDB-compatible embedding function using Google's
    ``text-embedding-004`` model.

    Implements the full ChromaDB EmbeddingFunction protocol:
    ``__call__``, ``embed_documents``, ``embed_query``, ``name``.
    """

    is_legacy: bool = True

    def __init__(self, provider: Optional[GeminiProvider] = None) -> None:
        self._provider = provider or GeminiProvider()

    @staticmethod
    def name() -> str:
        """ChromaDB protocol: return embedding function name."""
        return "google-text-embedding-004"

    def __call__(self, input: list[str]) -> list[list[float]]:
        """ChromaDB calls this for batch embedding."""
        return self._provider.embed_documents(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """ChromaDB calls this for document upserts."""
        return self._provider.embed_documents(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """ChromaDB calls this for query operations."""
        # ChromaDB passes a list even for single queries
        return self._provider.embed(input, task_type="retrieval_query")


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_default_provider: Optional[GeminiProvider] = None


def get_provider() -> GeminiProvider:
    """Return the module-level default GeminiProvider (lazy init)."""
    global _default_provider
    if _default_provider is None:
        _default_provider = GeminiProvider()
    return _default_provider
