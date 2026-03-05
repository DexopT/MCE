"""
MCE — SessionLedger (CostWatch)
Persistent cost tracking built on MCE's existing token counting.
Hooks into token_economist.py to receive counts without duplication.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from models.cost_store import CostEvent, CostStore, CostSummary, BudgetAlert
from schemas.mce_config import CostWatchConfig
from utils.logger import get_logger

_log = get_logger("SessionLedger")


class SessionLedger:
    """
    Persistent cost tracking with budget alerts.

    Receives token counts from the proxy pipeline on every tool call,
    persists them asynchronously to SQLite, and checks budget thresholds.
    """

    def __init__(self, config: CostWatchConfig, session_id: str, store: CostStore):
        self._config = config
        self._session_id = session_id
        self._store = store

        # In-memory accumulators for fast access (no DB round-trips)
        self._session_tokens_in = 0
        self._session_tokens_out = 0
        self._session_tokens_saved = 0
        self._session_cost_usd = 0.0
        self._event_count = 0

        # Rate tracking (tokens per minute)
        self._recent_tokens: list[tuple[float, int]] = []  # (timestamp, token_count)

    # ── Recording ─────────────────────────────

    async def record_exchange(
        self,
        tool_name: str,
        tokens_in: int,
        tokens_out: int,
        tokens_saved: int = 0,
    ) -> list[BudgetAlert]:
        """
        Record a token exchange and check for budget alerts.

        This is designed to be called fire-and-forget from the proxy.
        Returns any triggered budget alerts.
        """
        # Estimate cost (using default sonnet pricing if no model specified)
        estimated_cost = self._estimate_cost(tokens_in, tokens_out)

        # Update in-memory accumulators
        self._session_tokens_in += tokens_in
        self._session_tokens_out += tokens_out
        self._session_tokens_saved += tokens_saved
        self._session_cost_usd += estimated_cost
        self._event_count += 1

        # Track rate
        now = datetime.now(timezone.utc).timestamp()
        self._recent_tokens.append((now, tokens_in + tokens_out))
        # Prune entries older than 60 seconds
        cutoff = now - 60
        self._recent_tokens = [(t, c) for t, c in self._recent_tokens if t > cutoff]

        # Persist to SQLite (fire-and-forget)
        event = CostEvent(
            session_id=self._session_id,
            tool_name=tool_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_saved=tokens_saved,
            estimated_cost_usd=estimated_cost,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        try:
            await self._store.record_exchange(event)
        except Exception as exc:
            _log.warning(f"Failed to persist cost event: {exc}")

        # Check budgets
        return await self.check_budgets()

    def _estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        """
        Estimate cost in USD based on configured model rates.

        Uses claude-sonnet-4 rates as the default estimate. The actual
        model used is unknown to MCE (we're a proxy), but this gives a
        reasonable baseline for budgeting.
        """
        costs = self._config.model_costs
        # Use sonnet as the default model for estimation
        model_key = "claude-sonnet-4"
        if model_key not in costs:
            # Fallback to first available model
            model_key = next(iter(costs), None)
            if model_key is None:
                return 0.0

        model = costs[model_key]
        # Cost per 1M tokens
        cost_in = (tokens_in / 1_000_000) * model.input
        cost_out = (tokens_out / 1_000_000) * model.output
        return cost_in + cost_out

    # ── Budget Checking ───────────────────────

    async def check_budgets(self) -> list[BudgetAlert]:
        """Check all budget thresholds and return any triggered alerts."""
        alerts: list[BudgetAlert] = []

        # Session budget check
        if self._session_cost_usd > self._config.session_budget_usd:
            alerts.append(BudgetAlert(
                alert_type="session_budget",
                threshold=self._config.session_budget_usd,
                actual_value=self._session_cost_usd,
                message=(
                    f"Session cost ${self._session_cost_usd:.2f} exceeds "
                    f"budget of ${self._config.session_budget_usd:.2f}"
                ),
            ))

        # Daily budget check (from DB for accuracy across sessions)
        try:
            daily = await self._store.get_daily_cost()
            if daily.total_cost_usd > self._config.daily_budget_usd:
                alerts.append(BudgetAlert(
                    alert_type="daily_budget",
                    threshold=self._config.daily_budget_usd,
                    actual_value=daily.total_cost_usd,
                    message=(
                        f"Daily cost ${daily.total_cost_usd:.2f} exceeds "
                        f"budget of ${self._config.daily_budget_usd:.2f}"
                    ),
                ))
        except Exception:
            pass  # Don't block on DB errors

        # Rate spike check
        total_recent = sum(c for _, c in self._recent_tokens)
        if total_recent > self._config.token_rate_alert_per_min:
            alerts.append(BudgetAlert(
                alert_type="rate_spike",
                threshold=float(self._config.token_rate_alert_per_min),
                actual_value=float(total_recent),
                message=(
                    f"Token rate {total_recent}/min exceeds "
                    f"threshold of {self._config.token_rate_alert_per_min}/min"
                ),
            ))

        # Log alerts
        for alert in alerts:
            _log.warning(f"[mce.warning]\\[CostWatch] {alert.message}[/mce.warning]")

        return alerts

    # ── Summaries ─────────────────────────────

    def get_session_summary(self) -> dict:
        """Get in-memory session cost summary (no DB hit)."""
        return {
            "session_tokens_in": self._session_tokens_in,
            "session_tokens_out": self._session_tokens_out,
            "session_tokens_saved": self._session_tokens_saved,
            "session_cost_usd": round(self._session_cost_usd, 4),
            "event_count": self._event_count,
        }

    async def get_summary(self, period: str = "today") -> CostSummary:
        """Get aggregated cost summary for a given period."""
        if period == "session":
            return await self._store.get_session_cost(self._session_id)
        elif period == "today":
            return await self._store.get_daily_cost()
        elif period == "month":
            return await self._store.get_monthly_cost()
        else:
            return await self._store.get_daily_cost()
