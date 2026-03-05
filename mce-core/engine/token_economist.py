"""
MCE — Token Economist
Pre-flight token estimation and budget guardrails using tiktoken.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import tiktoken

from schemas.mce_config import TokenLimitsConfig
from utils.logger import get_logger

_log = get_logger("Economist")


# ──────────────────────────────────────────────
# Shared Encoder
# ──────────────────────────────────────────────

_ENC: tiktoken.Encoding | None = None


def _encoder() -> tiktoken.Encoding:
    global _ENC
    if _ENC is None:
        _ENC = tiktoken.get_encoding("cl100k_base")
    return _ENC


# ──────────────────────────────────────────────
# Report Types
# ──────────────────────────────────────────────

class Action(str, Enum):
    PASS_THROUGH = "pass_through"
    SQUEEZE = "squeeze"
    REJECT = "reject"


@dataclass
class TokenReport:
    """Result of evaluating a payload's token cost."""
    token_count: int
    is_over_budget: bool
    recommended_action: Action
    safe_limit: int
    squeeze_trigger: int
    absolute_max: int = 0


# ──────────────────────────────────────────────
# Token Economist
# ──────────────────────────────────────────────

class TokenEconomist:
    """
    Evaluates tool response payloads against configurable token budgets.

    - Under safe_limit    → PASS_THROUGH
    - safe_limit to max   → SQUEEZE
    - Over absolute_max   → REJECT (hard cap)
    """

    def __init__(self, config: TokenLimitsConfig | None = None):
        cfg = config or TokenLimitsConfig()
        self._safe = cfg.safe_limit
        self._trigger = cfg.squeeze_trigger
        self._max = cfg.absolute_max

    def count_tokens(self, text: str) -> int:
        """Count cl100k_base tokens in *text*."""
        return len(_encoder().encode(text))

    def count_any(self, payload: Any) -> int:
        """Count tokens in any payload (converts to JSON string if needed)."""
        if isinstance(payload, str):
            return self.count_tokens(payload)
        try:
            text = json.dumps(payload, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(payload)
        return self.count_tokens(text)

    def evaluate(self, payload: Any) -> TokenReport:
        """
        Evaluate a raw tool response payload.

        Returns a TokenReport with the recommended action:
        - PASS_THROUGH if under safe_limit
        - SQUEEZE if between safe_limit and absolute_max
        - REJECT if over absolute_max (hard cap, should still squeeze but flag it)
        """
        tokens = self.count_any(payload)

        if tokens <= self._safe:
            action = Action.PASS_THROUGH
            over_budget = False
        elif tokens <= self._max:
            action = Action.SQUEEZE
            over_budget = True
        else:
            # Over absolute max — still squeeze, but flag severity
            action = Action.SQUEEZE
            over_budget = True

        report = TokenReport(
            token_count=tokens,
            is_over_budget=over_budget,
            recommended_action=action,
            safe_limit=self._safe,
            squeeze_trigger=self._trigger,
            absolute_max=self._max,
        )

        if over_budget:
            severity = "CRITICAL" if tokens > self._max else "Budget exceeded"
            _log.info(
                f"[mce.warning]{severity}[/mce.warning]: "
                f"{tokens:,} tokens (safe={self._safe:,}, max={self._max:,}) → {action.value}"
            )
        else:
            _log.debug(f"Payload OK: {tokens:,} tokens ≤ {self._safe:,}")

        return report

    def serialize(self, payload: Any) -> str:
        """Convert payload to string representation."""
        if isinstance(payload, str):
            return payload
        try:
            return json.dumps(payload, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(payload)
