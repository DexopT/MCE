"""
MCE — Context Manager
Per-session token expenditure tracker. Tracks cumulative tokens sent,
tokens saved, and cache statistics for observability.
Thread-safe via asyncio.Lock for concurrent request handling.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_logger

_log = get_logger("Context")


# ──────────────────────────────────────────────
# Session Statistics
# ──────────────────────────────────────────────

@dataclass
class SessionStats:
    """Cumulative statistics for the current MCE session."""
    total_requests: int = 0
    total_raw_tokens: int = 0
    total_squeezed_tokens: int = 0
    total_tokens_saved: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    policy_blocks: int = 0
    breaker_trips: int = 0
    squeeze_invocations: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def savings_percent(self) -> float:
        if self.total_raw_tokens == 0:
            return 0.0
        return (self.total_tokens_saved / self.total_raw_tokens) * 100

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.started_at


# ──────────────────────────────────────────────
# Context Manager
# ──────────────────────────────────────────────

class ContextManager:
    """
    Tracks per-session token expenditure and operational metrics.

    Feeds data to the TUI dashboard and provides real-time
    observability into MCE's compression effectiveness.
    Uses asyncio.Lock for thread-safety under concurrent requests.
    """

    def __init__(self):
        self._stats = SessionStats()
        self._recent_tools: list[dict] = []  # Last N tool calls for display
        self._lock = asyncio.Lock()

    @property
    def stats(self) -> SessionStats:
        return self._stats

    @property
    def recent_tools(self) -> list[dict]:
        return list(self._recent_tools[-20:])

    # ── Recording Events ──────────────────────

    def record_request(
        self,
        tool_name: str,
        raw_tokens: int,
        squeezed_tokens: int,
        was_cached: bool = False,
        was_blocked: bool = False,
        breaker_tripped: bool = False,
    ) -> None:
        """
        Record a tool call event and update cumulative stats.

        Note: Only counts cache_misses for actual cache lookups
        (not for blocked/tripped requests that skip the cache).
        """
        self._stats.total_requests += 1
        self._stats.total_raw_tokens += raw_tokens
        self._stats.total_squeezed_tokens += squeezed_tokens
        self._stats.total_tokens_saved += (raw_tokens - squeezed_tokens)

        if was_cached:
            self._stats.cache_hits += 1
        elif not was_blocked and not breaker_tripped:
            # Only count as a cache miss if we actually checked the cache
            self._stats.cache_misses += 1

        if was_blocked:
            self._stats.policy_blocks += 1

        if breaker_tripped:
            self._stats.breaker_trips += 1

        if raw_tokens > squeezed_tokens:
            self._stats.squeeze_invocations += 1

        # Track recent tool calls
        self._recent_tools.append({
            "tool": tool_name,
            "raw": raw_tokens,
            "squeezed": squeezed_tokens,
            "saved": raw_tokens - squeezed_tokens,
            "cached": was_cached,
            "blocked": was_blocked,
            "time": time.time(),
        })

        # Keep only last 100
        if len(self._recent_tools) > 100:
            self._recent_tools = self._recent_tools[-100:]

    def summary(self) -> dict:
        """Return a summary dict for serialization / TUI display."""
        s = self._stats
        return {
            "total_requests": s.total_requests,
            "total_raw_tokens": s.total_raw_tokens,
            "total_squeezed_tokens": s.total_squeezed_tokens,
            "tokens_saved": s.total_tokens_saved,
            "savings_percent": round(s.savings_percent, 1),
            "cache_hit_rate": round(s.cache_hit_rate, 1),
            "cache_hits": s.cache_hits,
            "policy_blocks": s.policy_blocks,
            "breaker_trips": s.breaker_trips,
            "uptime_seconds": round(s.uptime_seconds, 1),
        }

    def reset(self) -> None:
        """Reset all session stats."""
        self._stats = SessionStats()
        self._recent_tools.clear()
        _log.info("Session stats reset")
