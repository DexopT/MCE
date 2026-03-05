"""
MCE — Text Chunking Utilities
Splits large payloads into token-aware chunks for the Semantic Router.
"""

from __future__ import annotations

import re
from typing import Optional

import tiktoken


# ──────────────────────────────────────────────
# Tokenizer (shared singleton)
# ──────────────────────────────────────────────

_ENC: Optional[tiktoken.Encoding] = None


def _encoder() -> tiktoken.Encoding:
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


def count_tokens(text: str) -> int:
    """Return the token count for *text* using cl100k_base."""
    return len(_encoder().encode(text))


# ──────────────────────────────────────────────
# Chunking Strategies
# ──────────────────────────────────────────────

def chunk_text(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[str]:
    """
    Split *text* into chunks of at most *max_tokens* tokens.

    Strategy:
    1. Split by double-newlines (paragraphs).
    2. If a paragraph itself exceeds max_tokens, split by sentences.
    3. Accumulate paragraphs/sentences into chunks respecting the budget.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_token_count = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # Big paragraph → break into sentences
        if para_tokens > max_tokens:
            # Flush current buffer first
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts, current_token_count = _overlap(current_parts, overlap_tokens)

            sentence_chunks = _split_sentences(para, max_tokens, overlap_tokens)
            chunks.extend(sentence_chunks)
            continue

        # Would exceed budget → flush
        if current_token_count + para_tokens > max_tokens:
            chunks.append("\n\n".join(current_parts))
            current_parts, current_token_count = _overlap(current_parts, overlap_tokens)

        current_parts.append(para)
        current_token_count += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return [c.strip() for c in chunks if c.strip()]


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _split_sentences(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    sentences = _SENTENCE_RE.split(text)
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for sent in sentences:
        st = count_tokens(sent)
        if buf_tokens + st > max_tokens and buf:
            chunks.append(" ".join(buf))
            buf, buf_tokens = _overlap_flat(buf, overlap_tokens)
        buf.append(sent)
        buf_tokens += st

    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _overlap(parts: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Keep trailing parts whose combined tokens fit within *overlap_tokens*."""
    kept: list[str] = []
    total = 0
    for p in reversed(parts):
        t = count_tokens(p)
        if total + t > overlap_tokens:
            break
        kept.insert(0, p)
        total += t
    return kept, total


def _overlap_flat(parts: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    return _overlap(parts, overlap_tokens)
