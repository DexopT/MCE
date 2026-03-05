"""
MCE — Memory Store
Async SQLite wrapper for MemVault's persistent memory and tool call logging.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from utils.logger import get_logger

_log = get_logger("MemoryStore")


# ──────────────────────────────────────────────
# Data Types
# ──────────────────────────────────────────────

@dataclass
class Memory:
    """A single extracted memory from a session."""
    id: str
    project_id: str
    type: str          # "decision" | "dead_end" | "constraint" | "preference" | "file_pattern"
    content: str
    source_tool: str = ""
    embedding: Optional[bytes] = None
    created_at: str = ""
    last_seen: str = ""
    confidence: float = 1.0


@dataclass
class ToolCallLog:
    """A logged tool call from a session."""
    id: str
    session_id: str
    tool_name: str
    request: str       # JSON string
    response: str      # JSON string (compressed by squeeze engine)
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: int = 0
    timestamp: str = ""


# ──────────────────────────────────────────────
# Memory Store
# ──────────────────────────────────────────────

class MemoryStore:
    """
    Async SQLite wrapper for MemVault.

    Manages two tables:
    - `memories` — extracted learnings from sessions
    - `tool_call_log` — raw tool call history for extraction
    """

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Open the database connection and initialize schema."""
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._init_schema()
        _log.debug(f"MemoryStore connected: {self._db_path}")

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        assert self._db is not None
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id          TEXT PRIMARY KEY,
                project_id  TEXT NOT NULL,
                type        TEXT NOT NULL,
                content     TEXT NOT NULL,
                source_tool TEXT DEFAULT '',
                embedding   BLOB,
                created_at  TEXT NOT NULL,
                last_seen   TEXT NOT NULL,
                confidence  REAL DEFAULT 1.0
            );

            CREATE INDEX IF NOT EXISTS idx_memories_project
                ON memories(project_id);
            CREATE INDEX IF NOT EXISTS idx_memories_type
                ON memories(project_id, type);

            CREATE TABLE IF NOT EXISTS tool_call_log (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                tool_name   TEXT NOT NULL,
                request     TEXT NOT NULL,
                response    TEXT NOT NULL,
                tokens_in   INTEGER DEFAULT 0,
                tokens_out  INTEGER DEFAULT 0,
                duration_ms INTEGER DEFAULT 0,
                timestamp   TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_tool_log_session
                ON tool_call_log(session_id);
        """)
        await self._db.commit()

    # ── Memory CRUD ───────────────────────────

    async def save_memory(self, memory: Memory) -> None:
        """Insert or update a memory entry."""
        assert self._db is not None
        await self._db.execute(
            """
            INSERT OR REPLACE INTO memories
                (id, project_id, type, content, source_tool, embedding,
                 created_at, last_seen, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.id, memory.project_id, memory.type, memory.content,
                memory.source_tool, memory.embedding,
                memory.created_at, memory.last_seen, memory.confidence,
            ),
        )
        await self._db.commit()

    async def get_memories(
        self,
        project_id: str,
        memory_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Retrieve memories for a project, optionally filtered by type."""
        assert self._db is not None

        if memory_type:
            cursor = await self._db.execute(
                "SELECT * FROM memories WHERE project_id = ? AND type = ? "
                "ORDER BY last_seen DESC LIMIT ?",
                (project_id, memory_type, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM memories WHERE project_id = ? "
                "ORDER BY last_seen DESC LIMIT ?",
                (project_id, limit),
            )

        rows = await cursor.fetchall()
        return [
            Memory(
                id=r["id"],
                project_id=r["project_id"],
                type=r["type"],
                content=r["content"],
                source_tool=r["source_tool"],
                embedding=r["embedding"],
                created_at=r["created_at"],
                last_seen=r["last_seen"],
                confidence=r["confidence"],
            )
            for r in rows
        ]

    async def count_memories(self, project_id: str) -> int:
        """Count total memories for a project."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT COUNT(*) FROM memories WHERE project_id = ?",
            (project_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def delete_memories(
        self, project_id: str, memory_type: Optional[str] = None
    ) -> int:
        """Delete memories, optionally filtered by type. Returns count deleted."""
        assert self._db is not None
        if memory_type:
            cursor = await self._db.execute(
                "DELETE FROM memories WHERE project_id = ? AND type = ?",
                (project_id, memory_type),
            )
        else:
            cursor = await self._db.execute(
                "DELETE FROM memories WHERE project_id = ?",
                (project_id,),
            )
        await self._db.commit()
        return cursor.rowcount

    # ── Tool Call Log ─────────────────────────

    async def log_tool_call(self, log: ToolCallLog) -> None:
        """Insert a tool call record."""
        assert self._db is not None
        await self._db.execute(
            """
            INSERT INTO tool_call_log
                (id, session_id, tool_name, request, response,
                 tokens_in, tokens_out, duration_ms, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log.id, log.session_id, log.tool_name, log.request,
                log.response, log.tokens_in, log.tokens_out,
                log.duration_ms, log.timestamp,
            ),
        )
        await self._db.commit()

    async def get_session_tool_calls(
        self, session_id: str, limit: int = 500
    ) -> list[ToolCallLog]:
        """Retrieve tool calls for a session."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM tool_call_log WHERE session_id = ? "
            "ORDER BY timestamp ASC LIMIT ?",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            ToolCallLog(
                id=r["id"],
                session_id=r["session_id"],
                tool_name=r["tool_name"],
                request=r["request"],
                response=r["response"],
                tokens_in=r["tokens_in"],
                tokens_out=r["tokens_out"],
                duration_ms=r["duration_ms"],
                timestamp=r["timestamp"],
            )
            for r in rows
        ]
