"""
MCE — Semantic Cache
Hash-based cache keyed on tool_name + sorted(arguments).
Returns cached pruned responses instantly for repeated requests.
Thread-safe via asyncio.Lock for concurrent request handling.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from utils.logger import get_logger, log_cache_hit

_log = get_logger("Cache")


# ──────────────────────────────────────────────
# Cache Entry
# ──────────────────────────────────────────────

@dataclass
class CacheEntry:
    """A single cached tool response."""
    key: str
    payload: Any
    token_count: int
    created_at: float = field(default_factory=time.time)


# ──────────────────────────────────────────────
# Semantic Cache
# ──────────────────────────────────────────────

class SemanticCache:
    """
    In-memory LRU cache for tool responses.

    Keyed on `hash(tool_name + canonical(arguments))`.
    Supports TTL-based expiry and configurable max size.
    Uses asyncio.Lock for safe concurrent access.
    """

    def __init__(self, max_entries: int = 512, ttl_seconds: int = 600):
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = asyncio.Lock()

    # ── Public API ────────────────────────────

    @staticmethod
    def make_key(tool_name: str, arguments: Optional[dict[str, Any]] = None) -> str:
        """Deterministic hash key for a tool call."""
        canonical = json.dumps(arguments or {}, sort_keys=True, default=str)
        raw = f"{tool_name}:{canonical}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, tool_name: str, arguments: Optional[dict[str, Any]] = None) -> Optional[CacheEntry]:
        """Look up a cached response.  Returns None on miss or expiry."""
        key = self.make_key(tool_name, arguments)
        entry = self._store.get(key)

        if entry is None:
            self._misses += 1
            return None

        # TTL check
        if time.time() - entry.created_at > self._ttl:
            del self._store[key]
            self._misses += 1
            return None

        # LRU: move to end
        self._store.move_to_end(key)
        self._hits += 1
        log_cache_hit(tool_name)
        return entry

    def put(
        self,
        tool_name: str,
        arguments: Optional[dict[str, Any]],
        payload: Any,
        token_count: int = 0,
    ) -> None:
        """Store a processed tool response."""
        key = self.make_key(tool_name, arguments)

        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = CacheEntry(
                key=key, payload=payload, token_count=token_count
            )
        else:
            # Evict oldest if full
            while len(self._store) >= self._max_entries:
                evicted_key, _ = self._store.popitem(last=False)
                _log.debug(f"Cache evicted: {evicted_key}")

            self._store[key] = CacheEntry(
                key=key, payload=payload, token_count=token_count
            )

    def invalidate(self, tool_name: str, arguments: Optional[dict[str, Any]] = None) -> bool:
        """Remove a specific entry.  Returns True if it existed."""
        key = self.make_key(tool_name, arguments)
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        """Flush the entire cache."""
        self._store.clear()
        _log.info("Cache cleared")

    # ── Stats ─────────────────────────────────

    @property
    def hit_count(self) -> int:
        return self._hits

    @property
    def miss_count(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def entry_count(self) -> int:
        return len(self._store)
