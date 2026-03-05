"""
MCE — Cost Store
Async SQLite wrapper for SessionLedger's persistent cost tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

from utils.logger import get_logger

_log = get_logger("CostStore")


# ──────────────────────────────────────────────
# Data Types
# ──────────────────────────────────────────────

@dataclass
class CostEvent:
    """A single token exchange event with cost estimation."""
    session_id: str
    tool_name: str
    tokens_in: int
    tokens_out: int
    tokens_saved: int
    estimated_cost_usd: float
    timestamp: str


@dataclass
class CostSummary:
    """Aggregated cost summary for a given period."""
    period: str          # "session" | "today" | "month"
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_tokens_saved: int = 0
    total_cost_usd: float = 0.0
    total_savings_usd: float = 0.0
    event_count: int = 0


@dataclass
class BudgetAlert:
    """A budget threshold alert."""
    alert_type: str      # "session_budget" | "daily_budget" | "rate_spike"
    threshold: float
    actual_value: float
    message: str


# ──────────────────────────────────────────────
# Cost Store
# ──────────────────────────────────────────────

class CostStore:
    """
    Async SQLite for persistent cost tracking.

    Stores cost events and supports aggregation queries
    for session, daily, and monthly cost summaries.
    """

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Open DB connection and initialize schema."""
        # Ensure the parent directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._init_schema()
        _log.debug(f"CostStore connected: {self._db_path}")

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        assert self._db is not None
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS cost_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                tool_name       TEXT NOT NULL,
                tokens_in       INTEGER DEFAULT 0,
                tokens_out      INTEGER DEFAULT 0,
                tokens_saved    INTEGER DEFAULT 0,
                estimated_cost  REAL DEFAULT 0.0,
                timestamp       TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_cost_session
                ON cost_events(session_id);
            CREATE INDEX IF NOT EXISTS idx_cost_timestamp
                ON cost_events(timestamp);
        """)
        await self._db.commit()

    # ── Recording ─────────────────────────────

    async def record_exchange(self, event: CostEvent) -> None:
        """Record a single cost event."""
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO cost_events
                (session_id, tool_name, tokens_in, tokens_out,
                 tokens_saved, estimated_cost, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.session_id, event.tool_name,
                event.tokens_in, event.tokens_out,
                event.tokens_saved, event.estimated_cost_usd,
                event.timestamp,
            ),
        )
        await self._db.commit()

    # ── Aggregation Queries ───────────────────

    async def get_session_cost(self, session_id: str) -> CostSummary:
        """Get aggregated cost for a specific session."""
        assert self._db is not None
        cursor = await self._db.execute(
            """
            SELECT
                COALESCE(SUM(tokens_in), 0)     AS total_in,
                COALESCE(SUM(tokens_out), 0)    AS total_out,
                COALESCE(SUM(tokens_saved), 0)  AS total_saved,
                COALESCE(SUM(estimated_cost), 0) AS total_cost,
                COUNT(*)                         AS cnt
            FROM cost_events WHERE session_id = ?
            """,
            (session_id,),
        )
        row = await cursor.fetchone()
        return CostSummary(
            period="session",
            total_tokens_in=row["total_in"],
            total_tokens_out=row["total_out"],
            total_tokens_saved=row["total_saved"],
            total_cost_usd=row["total_cost"],
            event_count=row["cnt"],
        )

    async def get_daily_cost(self, date: Optional[str] = None) -> CostSummary:
        """Get aggregated cost for a specific day (default: today)."""
        assert self._db is not None
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        cursor = await self._db.execute(
            """
            SELECT
                COALESCE(SUM(tokens_in), 0)     AS total_in,
                COALESCE(SUM(tokens_out), 0)    AS total_out,
                COALESCE(SUM(tokens_saved), 0)  AS total_saved,
                COALESCE(SUM(estimated_cost), 0) AS total_cost,
                COUNT(*)                         AS cnt
            FROM cost_events WHERE timestamp LIKE ?
            """,
            (f"{date}%",),
        )
        row = await cursor.fetchone()
        return CostSummary(
            period="today",
            total_tokens_in=row["total_in"],
            total_tokens_out=row["total_out"],
            total_tokens_saved=row["total_saved"],
            total_cost_usd=row["total_cost"],
            event_count=row["cnt"],
        )

    async def get_monthly_cost(self, month: Optional[str] = None) -> CostSummary:
        """Get aggregated cost for a month (default: current month)."""
        assert self._db is not None
        if month is None:
            month = datetime.now(timezone.utc).strftime("%Y-%m")

        cursor = await self._db.execute(
            """
            SELECT
                COALESCE(SUM(tokens_in), 0)     AS total_in,
                COALESCE(SUM(tokens_out), 0)    AS total_out,
                COALESCE(SUM(tokens_saved), 0)  AS total_saved,
                COALESCE(SUM(estimated_cost), 0) AS total_cost,
                COUNT(*)                         AS cnt
            FROM cost_events WHERE timestamp LIKE ?
            """,
            (f"{month}%",),
        )
        row = await cursor.fetchone()
        return CostSummary(
            period="month",
            total_tokens_in=row["total_in"],
            total_tokens_out=row["total_out"],
            total_tokens_saved=row["total_saved"],
            total_cost_usd=row["total_cost"],
            event_count=row["cnt"],
        )
