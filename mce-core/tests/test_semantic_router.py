"""
MCE — Semantic Router Tests
Tests the Layer 2 semantic routing with mocked embeddings.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.vector_store import VectorStore, SearchResult


class TestVectorStore:
    """Test the in-memory vector store independently (no model needed)."""

    def test_add_and_query(self):
        store = VectorStore()
        # 3 documents with 4-dim embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        docs = ["doc about auth", "doc about database", "doc about testing"]
        store.add(embeddings, docs)

        # Query should find doc closest to [1, 0, 0, 0]
        query = np.array([0.9, 0.1, 0.0, 0.0])
        results = store.query(query, top_k=1)
        assert len(results) == 1
        assert results[0].document == "doc about auth"

    def test_top_k_ordering(self):
        store = VectorStore()
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.7, 0.3, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        docs = ["closest", "second", "farthest"]
        store.add(embeddings, docs)

        query = np.array([1.0, 0.0, 0.0, 0.0])
        results = store.query(query, top_k=2)
        assert len(results) == 2
        assert results[0].document == "closest"
        assert results[1].document == "second"

    def test_empty_store(self):
        store = VectorStore()
        query = np.array([1.0, 0.0, 0.0])
        results = store.query(query, top_k=5)
        assert results == []

    def test_clear(self):
        store = VectorStore()
        embeddings = np.array([[1.0, 0.0]])
        store.add(embeddings, ["doc"])
        assert store.size == 1
        store.clear()
        assert store.size == 0

    def test_mismatch_raises(self):
        store = VectorStore()
        embeddings = np.array([[1.0, 0.0]])
        with pytest.raises(ValueError):
            store.add(embeddings, ["doc1", "doc2"])

    def test_search_result_scores(self):
        store = VectorStore()
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        store.add(embeddings, ["a", "b"])

        results = store.query(np.array([1.0, 0.0]), top_k=2)
        # First result should have higher score
        assert results[0].score > results[1].score


class TestChunker:
    """Test text chunking utilities."""

    def test_small_text_no_split(self):
        from utils.chunker import chunk_text
        chunks = chunk_text("Hello world", max_tokens=100)
        assert len(chunks) == 1

    def test_large_text_splits(self):
        from utils.chunker import chunk_text
        text = "\n\n".join([f"Paragraph {i} with some words." for i in range(50)])
        chunks = chunk_text(text, max_tokens=50)
        assert len(chunks) > 1

    def test_token_counting(self):
        from utils.chunker import count_tokens
        count = count_tokens("hello world")
        assert isinstance(count, int)
        assert count > 0
