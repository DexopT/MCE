"""
MCE — MemVault Tests
Tests for persistent memory: heuristic extraction, storage, injection.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.memory_store import MemoryStore, Memory
from engine.intelligence.memvault import MemVault
from schemas.mce_config import MemVaultConfig


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite database path."""
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def memvault_config():
    return MemVaultConfig(
        enabled=True,
        storage_path="~/.mce/projects",
        extraction_mode="heuristic",
        injection_token_budget=500,
        auto_update_claude_md=False,
    )


# ──────────────────────────────────────────────
# MemoryStore Tests
# ──────────────────────────────────────────────

@pytest.mark.asyncio
class TestMemoryStore:
    async def test_connect_creates_tables(self, temp_db):
        store = MemoryStore(temp_db)
        await store.connect()
        assert Path(temp_db).exists()
        await store.close()

    async def test_save_and_retrieve_memory(self, temp_db):
        store = MemoryStore(temp_db)
        await store.connect()

        memory = Memory(
            id="mem-1",
            project_id="proj-1",
            type="decision",
            content="Decided to use RS256 for JWT tokens",
            source_tool="write_file",
            created_at="2026-03-05T10:00:00Z",
            last_seen="2026-03-05T10:00:00Z",
            confidence=0.9,
        )
        await store.save_memory(memory)

        memories = await store.get_memories("proj-1")
        assert len(memories) == 1
        assert memories[0].id == "mem-1"
        assert memories[0].content == "Decided to use RS256 for JWT tokens"

        await store.close()

    async def test_filter_by_type(self, temp_db):
        store = MemoryStore(temp_db)
        await store.connect()

        for i, mtype in enumerate(["decision", "dead_end", "constraint"]):
            await store.save_memory(Memory(
                id=f"mem-{i}",
                project_id="proj-1",
                type=mtype,
                content=f"Test {mtype}",
                created_at="2026-03-05T10:00:00Z",
                last_seen="2026-03-05T10:00:00Z",
            ))

        decisions = await store.get_memories("proj-1", memory_type="decision")
        assert len(decisions) == 1
        assert decisions[0].type == "decision"

        await store.close()

    async def test_count_memories(self, temp_db):
        store = MemoryStore(temp_db)
        await store.connect()

        for i in range(5):
            await store.save_memory(Memory(
                id=f"mem-{i}",
                project_id="proj-1",
                type="decision",
                content=f"Decision {i}",
                created_at="2026-03-05T10:00:00Z",
                last_seen="2026-03-05T10:00:00Z",
            ))

        count = await store.count_memories("proj-1")
        assert count == 5

        await store.close()

    async def test_delete_memories(self, temp_db):
        store = MemoryStore(temp_db)
        await store.connect()

        for i in range(3):
            await store.save_memory(Memory(
                id=f"mem-{i}",
                project_id="proj-1",
                type="decision",
                content=f"Decision {i}",
                created_at="2026-03-05T10:00:00Z",
                last_seen="2026-03-05T10:00:00Z",
            ))

        deleted = await store.delete_memories("proj-1")
        assert deleted == 3

        count = await store.count_memories("proj-1")
        assert count == 0

        await store.close()

    async def test_log_tool_call(self, temp_db):
        from models.memory_store import ToolCallLog

        store = MemoryStore(temp_db)
        await store.connect()

        log = ToolCallLog(
            id="call-1",
            session_id="sess-1",
            tool_name="read_file",
            request='{"path": "src/main.py"}',
            response='{"content": "print(hello)"}',
            tokens_in=50,
            tokens_out=30,
            timestamp="2026-03-05T10:00:00Z",
        )
        await store.log_tool_call(log)

        calls = await store.get_session_tool_calls("sess-1")
        assert len(calls) == 1
        assert calls[0].tool_name == "read_file"

        await store.close()


# ──────────────────────────────────────────────
# MemVault Tests
# ──────────────────────────────────────────────

@pytest.mark.asyncio
class TestMemVault:
    async def test_observe_logs_tool_call(self, temp_db, memvault_config):
        store = MemoryStore(temp_db)
        await store.connect()

        vault = MemVault(memvault_config, "proj-1", "sess-1", store)

        await vault.observe(
            tool_name="read_file",
            arguments={"path": "src/auth/middleware.py"},
            response={"content": "def authenticate(): ..."},
            tokens_in=100,
            tokens_out=80,
        )

        calls = await store.get_session_tool_calls("sess-1")
        assert len(calls) == 1
        assert calls[0].tool_name == "read_file"

        await store.close()

    async def test_heuristic_extraction_finds_decisions(self, temp_db, memvault_config):
        store = MemoryStore(temp_db)
        await store.connect()

        vault = MemVault(memvault_config, "proj-1", "sess-1", store)

        # Simulate a tool call that contains decision language
        await vault.observe(
            tool_name="write_file",
            arguments={"path": "src/config.py"},
            response="Decided to use RS256 for JWT authentication",
        )

        memories = await vault.extract_session_learnings()
        decision_memories = [m for m in memories if m.type == "decision"]
        assert len(decision_memories) > 0

        await store.close()

    async def test_heuristic_extraction_finds_dead_ends(self, temp_db, memvault_config):
        store = MemoryStore(temp_db)
        await store.connect()

        vault = MemVault(memvault_config, "proj-1", "sess-1", store)

        await vault.observe(
            tool_name="execute_command",
            arguments={"command": "npm test"},
            response="Error: Redis pub/sub didn't work for job queue - abandon this approach",
        )

        memories = await vault.extract_session_learnings()
        dead_ends = [m for m in memories if m.type == "dead_end"]
        assert len(dead_ends) > 0

        await store.close()

    async def test_heuristic_extraction_finds_constraints(self, temp_db, memvault_config):
        store = MemoryStore(temp_db)
        await store.connect()

        vault = MemVault(memvault_config, "proj-1", "sess-1", store)

        await vault.observe(
            tool_name="read_file",
            arguments={"path": "legacy_api.py"},
            response="Don't touch this file, it's a constraint from the team lead",
        )

        memories = await vault.extract_session_learnings()
        constraints = [m for m in memories if m.type == "constraint"]
        assert len(constraints) > 0

        await store.close()

    async def test_inject_context_empty_db(self, temp_db, memvault_config):
        store = MemoryStore(temp_db)
        await store.connect()

        vault = MemVault(memvault_config, "proj-1", "sess-1", store)

        context = await vault.inject_context()
        assert context == ""  # No memories yet

        await store.close()

    async def test_inject_context_with_memories(self, temp_db, memvault_config):
        store = MemoryStore(temp_db)
        await store.connect()

        # Save some memories directly
        await store.save_memory(Memory(
            id="mem-1",
            project_id="proj-1",
            type="decisions",
            content="Use RS256 for JWT tokens",
            created_at="2026-03-05T10:00:00Z",
            last_seen="2026-03-05T10:00:00Z",
        ))
        await store.save_memory(Memory(
            id="mem-2",
            project_id="proj-1",
            type="dead_ends",
            content="Redis pub/sub failed for job queue",
            created_at="2026-03-05T10:00:00Z",
            last_seen="2026-03-05T10:00:00Z",
        ))

        vault = MemVault(memvault_config, "proj-1", "sess-1", store)
        context = await vault.inject_context()

        assert "MCE Session Memory" in context

        await store.close()

    async def test_memory_count(self, temp_db, memvault_config):
        store = MemoryStore(temp_db)
        await store.connect()

        for i in range(3):
            await store.save_memory(Memory(
                id=f"mem-{i}",
                project_id="proj-1",
                type="decision",
                content=f"Decision {i}",
                created_at="2026-03-05T10:00:00Z",
                last_seen="2026-03-05T10:00:00Z",
            ))

        vault = MemVault(memvault_config, "proj-1", "sess-1", store)
        count = await vault.get_memory_count()
        assert count == 3

        await store.close()

    async def test_no_observations_returns_empty(self, temp_db, memvault_config):
        store = MemoryStore(temp_db)
        await store.connect()

        vault = MemVault(memvault_config, "proj-1", "sess-1", store)
        memories = await vault.extract_session_learnings()
        assert len(memories) == 0

        await store.close()
