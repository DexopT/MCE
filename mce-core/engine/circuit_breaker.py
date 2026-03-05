"""
MCE — Circuit Breaker
Detects infinite loops: if the agent calls the same tool with the same
failing arguments 3 times, MCE trips the breaker and forces a context shift.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from schemas.mce_config import CircuitBreakerConfig
from utils.logger import get_logger

_log = get_logger("CircuitBreaker")


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class ToolCallRecord:
    """A single tool invocation record for the sliding window."""
    tool_name: str
    arguments_hash: str
    is_error: bool
    fingerprint: str  # hash(tool + args + error_flag)


@dataclass
class BreakerState:
    """Current state of the circuit breaker."""
    is_tripped: bool = False
    consecutive_failures: int = 0
    alert_message: str = ""


# ──────────────────────────────────────────────
# Circuit Breaker
# ──────────────────────────────────────────────

class CircuitBreaker:
    """
    Sliding-window loop detector.

    Maintains the last N tool calls. If the same tool+args combination
    fails >= threshold times, trips the breaker.
    """

    ALERT_TEMPLATE = (
        "[MCE Alert: You are stuck in a loop. "
        "Previous {count} attempts failed identically. "
        "Pause execution and formulate a completely different approach, "
        "or ask the user for help.]"
    )

    def __init__(self, config: CircuitBreakerConfig | None = None):
        cfg = config or CircuitBreakerConfig()
        self._window_size = cfg.window_size
        self._threshold = cfg.failure_threshold
        self._window: deque[ToolCallRecord] = deque(maxlen=self._window_size)
        self._lock = asyncio.Lock()

    # ── Public API ────────────────────────────

    def record(
        self,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
        is_error: bool = False,
    ) -> BreakerState:
        """
        Record a tool call and check for loops.

        Returns BreakerState — check .is_tripped to see if the
        breaker has been activated.
        """
        args_hash = self._hash_args(arguments)
        fingerprint = hashlib.sha256(
            f"{tool_name}:{args_hash}:{is_error}".encode()
        ).hexdigest()[:16]

        record = ToolCallRecord(
            tool_name=tool_name,
            arguments_hash=args_hash,
            is_error=is_error,
            fingerprint=fingerprint,
        )
        self._window.append(record)

        # Count identical failing fingerprints in current window
        if is_error:
            count = sum(1 for r in self._window if r.fingerprint == fingerprint)
            if count >= self._threshold:
                alert = self.ALERT_TEMPLATE.format(count=count)
                _log.warning(
                    f"[mce.error]BREAKER TRIPPED[/mce.error]: "
                    f"{tool_name} failed {count}× identically"
                )
                return BreakerState(
                    is_tripped=True,
                    consecutive_failures=count,
                    alert_message=alert,
                )

        return BreakerState(is_tripped=False)

    def reset(self) -> None:
        """Clear the sliding window."""
        self._window.clear()

    @property
    def window(self) -> list[ToolCallRecord]:
        """Current window contents (for observability)."""
        return list(self._window)

    # ── Internal ──────────────────────────────

    @staticmethod
    def _hash_args(arguments: Optional[dict[str, Any]]) -> str:
        canonical = json.dumps(arguments or {}, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
