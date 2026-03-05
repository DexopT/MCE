"""
MCE — Embedding Model Manager
Lazy-loads a lightweight sentence-transformer model for CPU-friendly semantic search.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from utils.logger import get_logger

_log = get_logger("Embeddings")


# ──────────────────────────────────────────────
# Singleton Model Holder (keyed by model name)
# ──────────────────────────────────────────────

class EmbeddingModel:
    """
    Lazily loads the configured sentence-transformer model (default: all-MiniLM-L6-v2).
    Uses per-model-name singleton pattern so models are loaded once and reused.
    Raises ValueError if a different model name is requested after initialization.
    """

    _instances: dict[str, "EmbeddingModel"] = {}

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        if model_name in cls._instances:
            return cls._instances[model_name]

        instance = super().__new__(cls)
        instance._model_name = model_name
        instance._model = None
        cls._instances[model_name] = instance
        return instance

    def _load(self) -> None:
        if self._model is None:
            _log.info(f"Loading embedding model: [mce.token]{self._model_name}[/mce.token]")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            _log.info("[mce.success]Embedding model loaded![/mce.success]")

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts into dense vectors.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        self._load()
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text, returns 1-D array."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        self._load()
        return self._model.get_sentence_embedding_dimension()

    @classmethod
    def reset(cls) -> None:
        """Reset all singletons (for testing)."""
        cls._instances.clear()
