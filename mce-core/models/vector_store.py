"""
MCE — In-Memory Vector Store
Lightweight numpy-based vector store with cosine similarity search.
No heavy dependencies — no FAISS, no Postgres.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single search result with its similarity score."""
    index: int
    score: float
    document: str


# ──────────────────────────────────────────────
# Vector Store
# ──────────────────────────────────────────────

@dataclass
class VectorStore:
    """
    In-memory vector store using numpy cosine similarity.

    Usage:
        store = VectorStore()
        store.add(embeddings, documents)
        results = store.query(query_embedding, top_k=3)
    """

    _embeddings: np.ndarray | None = field(default=None, init=False, repr=False)
    _documents: list[str] = field(default_factory=list, init=False)

    @property
    def size(self) -> int:
        return len(self._documents)

    def add(self, embeddings: np.ndarray, documents: list[str]) -> None:
        """
        Add embeddings and their associated documents.

        Args:
            embeddings: np.ndarray of shape (n, dim)
            documents: list of n text documents
        """
        if len(embeddings) != len(documents):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings vs {len(documents)} documents"
            )

        if self._embeddings is None:
            self._embeddings = embeddings.copy()
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])

        self._documents.extend(documents)

    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        """
        Find the top-k most similar documents to *query_embedding*.

        Args:
            query_embedding: 1-D array of shape (dim,)
            top_k: number of results to return

        Returns:
            Sorted list of SearchResult (highest similarity first).
        """
        if self._embeddings is None or self.size == 0:
            return []

        # Cosine similarity = dot(a, b) / (||a|| * ||b||)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
        normed = self._embeddings / norms

        similarities = normed @ query_norm

        top_k = min(top_k, self.size)
        indices = np.argsort(similarities)[::-1][:top_k]

        return [
            SearchResult(
                index=int(idx),
                score=float(similarities[idx]),
                document=self._documents[idx],
            )
            for idx in indices
        ]

    def clear(self) -> None:
        """Remove all stored data."""
        self._embeddings = None
        self._documents.clear()
