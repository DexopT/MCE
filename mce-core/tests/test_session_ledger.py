"""
MCE — SessionLedger Tests
Tests for the CostWatch component: cost recording, budget alerts.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.cost_store import CostStore, CostEvent
from engine.intelligence.session_ledger import SessionLedger
from schemas.mce_config import CostWatchConfig, ModelCost


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite database path."""
    return str(tmp_path / "test_cost.db")


@pytest.fixture
def cost_config():
    return CostWatchConfig(
        enabled=True,
        daily_budget_usd=5.00,
        session_budget_usd=1.00,
        token_rate_alert_per_min=500,
        model_costs={
            "claude-sonnet-4": ModelCost(input=3.0, output=15.0),
        },
    )


@pytest.mark.asyncio
class TestCostStore:
    async def test_connect_creates_tables(self, temp_db):
        store = CostStore(temp_db)
        await store.connect()
        assert Path(temp_db).exists()
        await store.close()

    async def test_record_and_get_session_cost(self, temp_db):
        store = CostStore(temp_db)
        await store.connect()

        event = CostEvent(
            session_id="sess-1",
            tool_name="read_file",
            tokens_in=100,
            tokens_out=80,
            tokens_saved=20,
            estimated_cost_usd=0.001,
            timestamp="2026-03-05T10:00:00Z",
        )
        await store.record_exchange(event)

        summary = await store.get_session_cost("sess-1")
        assert summary.total_tokens_in == 100
        assert summary.total_tokens_out == 80
        assert summary.total_tokens_saved == 20
        assert summary.event_count == 1

        await store.close()

    async def test_daily_cost_aggregation(self, temp_db):
        store = CostStore(temp_db)
        await store.connect()

        for i in range(3):
            event = CostEvent(
                session_id=f"sess-{i}",
                tool_name="tool",
                tokens_in=100,
                tokens_out=50,
                tokens_saved=50,
                estimated_cost_usd=0.01,
                timestamp="2026-03-05T10:00:00Z",
            )
            await store.record_exchange(event)

        summary = await store.get_daily_cost("2026-03-05")
        assert summary.total_tokens_in == 300
        assert summary.event_count == 3
        assert summary.total_cost_usd == pytest.approx(0.03)

        await store.close()

    async def test_empty_session_returns_zero(self, temp_db):
        store = CostStore(temp_db)
        await store.connect()

        summary = await store.get_session_cost("nonexistent")
        assert summary.total_tokens_in == 0
        assert summary.event_count == 0

        await store.close()


@pytest.mark.asyncio
class TestSessionLedger:
    async def test_record_exchange_updates_accumulators(self, temp_db, cost_config):
        store = CostStore(temp_db)
        await store.connect()

        ledger = SessionLedger(cost_config, "sess-1", store)

        await ledger.record_exchange(
            tool_name="read_file",
            tokens_in=500,
            tokens_out=400,
            tokens_saved=100,
        )

        summary = ledger.get_session_summary()
        assert summary["session_tokens_in"] == 500
        assert summary["session_tokens_out"] == 400
        assert summary["session_tokens_saved"] == 100
        assert summary["event_count"] == 1
        assert summary["session_cost_usd"] > 0

        await store.close()

    async def test_session_budget_alert(self, temp_db, cost_config):
        store = CostStore(temp_db)
        await store.connect()

        # Very low session budget to trigger alert
        cost_config.session_budget_usd = 0.0001
        ledger = SessionLedger(cost_config, "sess-1", store)

        alerts = await ledger.record_exchange(
            tool_name="read_file",
            tokens_in=10000,
            tokens_out=5000,
        )

        session_alerts = [a for a in alerts if a.alert_type == "session_budget"]
        assert len(session_alerts) > 0

        await store.close()

    async def test_multiple_exchanges_accumulate(self, temp_db, cost_config):
        store = CostStore(temp_db)
        await store.connect()

        ledger = SessionLedger(cost_config, "sess-1", store)

        for i in range(5):
            await ledger.record_exchange(
                tool_name=f"tool_{i}",
                tokens_in=100,
                tokens_out=80,
                tokens_saved=20,
            )

        summary = ledger.get_session_summary()
        assert summary["session_tokens_in"] == 500
        assert summary["event_count"] == 5

        await store.close()

    async def test_cost_estimation(self, temp_db, cost_config):
        store = CostStore(temp_db)
        await store.connect()

        ledger = SessionLedger(cost_config, "sess-1", store)

        await ledger.record_exchange(
            tool_name="read_file",
            tokens_in=1_000_000,  # 1M tokens in
            tokens_out=0,
        )

        summary = ledger.get_session_summary()
        # 1M input tokens at $3/M = $3.00
        assert summary["session_cost_usd"] == pytest.approx(3.0, abs=0.01)

        await store.close()
