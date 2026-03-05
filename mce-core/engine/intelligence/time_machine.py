"""
MCE — TimeMachine
Git-like checkpointing for AI agent sessions. Stores structured
tool-call state, not just conversation text.

Supports:
- Manual and automatic checkpoints
- Branching (fork from any checkpoint)
- Restoration (roll back to a checkpoint)
- File diff capture via git integration
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from schemas.mce_config import TimeMachineConfig
from utils.logger import get_logger

_log = get_logger("TimeMachine")


# ──────────────────────────────────────────────
# Data Types
# ──────────────────────────────────────────────

@dataclass
class Checkpoint:
    """A snapshot of the session state at a point in time."""
    id: str
    session_id: str
    branch: str              # "main" or branch name
    sequence: int            # checkpoint number within session
    label: str               # user-provided or auto-generated label
    tool_call_count: int     # number of tool calls up to this point
    token_count: int         # cumulative tokens at this point
    file_diff: str           # git diff or empty
    tool_summary: str        # JSON array of recent tool calls
    created_at: str
    auto_triggered: bool = False


@dataclass
class BranchInfo:
    """A branch forked from a checkpoint."""
    name: str
    parent_checkpoint_id: str
    created_at: str
    checkpoint_count: int = 0


# ──────────────────────────────────────────────
# TimeMachine
# ──────────────────────────────────────────────

class TimeMachine:
    """
    Session checkpointing and branching system.

    Creates structured snapshots of the full tool-call state. Supports
    auto-checkpoint triggers (interval, file write, destructive tool)
    and manual checkpoints.
    """

    def __init__(
        self,
        config: TimeMachineConfig,
        session_id: str,
        db_path: str | Path,
    ):
        self._config = config
        self._session_id = session_id
        self._db_path = str(db_path)
        self._db: Optional[aiosqlite.Connection] = None

        # State tracking
        self._current_branch = "main"
        self._sequence = 0
        self._tool_calls: list[dict] = []  # buffer of tool calls since last checkpoint
        self._cumulative_tokens = 0
        self._last_checkpoint_time = time.monotonic()

    # ── Lifecycle ─────────────────────────────

    async def connect(self) -> None:
        """Open DB and initialize schema."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._init_schema()

        # Resume sequence from existing checkpoints
        cursor = await self._db.execute(
            "SELECT MAX(sequence) FROM checkpoints WHERE session_id = ?",
            (self._session_id,),
        )
        row = await cursor.fetchone()
        if row and row[0] is not None:
            self._sequence = row[0]

        _log.info("[mce.success]\\[TimeMachine] Initialized[/mce.success]")

    async def close(self) -> None:
        """Close DB connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def _init_schema(self) -> None:
        """Create checkpoint tables."""
        assert self._db is not None
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id              TEXT PRIMARY KEY,
                session_id      TEXT NOT NULL,
                branch          TEXT NOT NULL DEFAULT 'main',
                sequence        INTEGER NOT NULL,
                label           TEXT NOT NULL,
                tool_call_count INTEGER DEFAULT 0,
                token_count     INTEGER DEFAULT 0,
                file_diff       TEXT DEFAULT '',
                tool_summary    TEXT DEFAULT '[]',
                created_at      TEXT NOT NULL,
                auto_triggered  INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_cp_session
                ON checkpoints(session_id, branch);

            CREATE TABLE IF NOT EXISTS branches (
                name                TEXT NOT NULL,
                session_id          TEXT NOT NULL,
                parent_checkpoint   TEXT NOT NULL,
                created_at          TEXT NOT NULL,
                PRIMARY KEY (session_id, name)
            );
        """)
        await self._db.commit()

    # ── Recording Tool Calls ──────────────────

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        tokens: int,
        is_file_write: bool = False,
        is_destructive: bool = False,
    ) -> None:
        """
        Record a tool call for checkpoint tracking.
        Returns silently — the proxy calls maybe_checkpoint() separately.
        """
        self._tool_calls.append({
            "tool": tool_name,
            "args_preview": json.dumps(arguments, default=str)[:200],
            "tokens": tokens,
            "is_file_write": is_file_write,
            "is_destructive": is_destructive,
            "time": datetime.now(timezone.utc).isoformat(),
        })
        self._cumulative_tokens += tokens

    # ── Auto-Checkpoint Logic ─────────────────

    async def maybe_checkpoint(
        self,
        tool_name: str,
        is_file_write: bool = False,
        is_destructive: bool = False,
    ) -> Optional[Checkpoint]:
        """
        Check if an auto-checkpoint should be triggered.

        Triggers on:
        1. File write (if checkpoint_on_file_write enabled)
        2. Destructive tool (if checkpoint_on_destructive_tool enabled)
        3. Time interval (if auto_checkpoint_interval_mins set)
        """
        if not self._config.enabled:
            return None

        # Check max checkpoints
        if self._sequence >= self._config.max_checkpoints_per_session:
            return None

        trigger_reason = None

        # Trigger: file write
        if is_file_write and self._config.checkpoint_on_file_write:
            trigger_reason = f"file_write:{tool_name}"

        # Trigger: destructive tool
        elif is_destructive and self._config.checkpoint_on_destructive_tool:
            trigger_reason = f"destructive:{tool_name}"

        # Trigger: time interval
        elif self._config.auto_checkpoint_interval_mins > 0:
            elapsed = (time.monotonic() - self._last_checkpoint_time) / 60
            if elapsed >= self._config.auto_checkpoint_interval_mins:
                trigger_reason = "interval"

        if trigger_reason is None:
            return None

        label = f"Auto: {trigger_reason}"
        return await self.checkpoint(label=label, auto_triggered=True)

    # ── Checkpoint ────────────────────────────

    async def checkpoint(
        self,
        label: Optional[str] = None,
        auto_triggered: bool = False,
    ) -> Checkpoint:
        """
        Create a checkpoint of the current session state.

        Captures:
        - Number of tool calls since session start
        - Cumulative token count
        - Summary of recent tool calls
        - File diff (if git available and enabled)
        """
        assert self._db is not None

        self._sequence += 1
        now = datetime.now(timezone.utc).isoformat()

        if label is None:
            label = f"Checkpoint #{self._sequence}"

        # Capture file diff
        file_diff = ""
        if self._config.capture_file_diffs:
            file_diff = self._capture_git_diff()

        # Build tool summary (last 10 calls since previous checkpoint)
        tool_summary = json.dumps(self._tool_calls[-10:], default=str)

        cp_id = hashlib.sha256(
            f"{self._session_id}:{self._current_branch}:{self._sequence}:{now}".encode()
        ).hexdigest()[:16]

        checkpoint = Checkpoint(
            id=cp_id,
            session_id=self._session_id,
            branch=self._current_branch,
            sequence=self._sequence,
            label=label,
            tool_call_count=len(self._tool_calls),
            token_count=self._cumulative_tokens,
            file_diff=file_diff,
            tool_summary=tool_summary,
            created_at=now,
            auto_triggered=auto_triggered,
        )

        # Persist
        await self._db.execute(
            """
            INSERT INTO checkpoints
                (id, session_id, branch, sequence, label, tool_call_count,
                 token_count, file_diff, tool_summary, created_at, auto_triggered)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                checkpoint.id, checkpoint.session_id, checkpoint.branch,
                checkpoint.sequence, checkpoint.label, checkpoint.tool_call_count,
                checkpoint.token_count, checkpoint.file_diff,
                checkpoint.tool_summary, checkpoint.created_at,
                1 if checkpoint.auto_triggered else 0,
            ),
        )
        await self._db.commit()

        # Reset timer
        self._last_checkpoint_time = time.monotonic()

        _log.info(
            f"[mce.success]\\[TimeMachine] Checkpoint #{self._sequence}: "
            f"{label}[/mce.success]"
        )

        return checkpoint

    # ── Branch ────────────────────────────────

    async def branch(self, branch_name: str, from_checkpoint_id: Optional[str] = None) -> BranchInfo:
        """
        Fork a new branch from a checkpoint.

        If no checkpoint_id specified, branches from the latest checkpoint.
        """
        assert self._db is not None

        now = datetime.now(timezone.utc).isoformat()

        # Find the source checkpoint
        if from_checkpoint_id:
            cursor = await self._db.execute(
                "SELECT * FROM checkpoints WHERE id = ?",
                (from_checkpoint_id,),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM checkpoints WHERE session_id = ? AND branch = ? "
                "ORDER BY sequence DESC LIMIT 1",
                (self._session_id, self._current_branch),
            )

        row = await cursor.fetchone()
        if row is None:
            # If no checkpoint exists, create one first
            cp = await self.checkpoint(label=f"Pre-branch: {branch_name}")
            parent_id = cp.id
        else:
            parent_id = row["id"]

        # Create branch record
        await self._db.execute(
            """
            INSERT OR REPLACE INTO branches
                (name, session_id, parent_checkpoint, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (branch_name, self._session_id, parent_id, now),
        )
        await self._db.commit()

        # Switch to the new branch
        self._current_branch = branch_name

        info = BranchInfo(
            name=branch_name,
            parent_checkpoint_id=parent_id,
            created_at=now,
        )

        _log.info(
            f"[mce.success]\\[TimeMachine] Branched '{branch_name}' "
            f"from checkpoint {parent_id[:8]}[/mce.success]"
        )

        return info

    # ── Restore ───────────────────────────────

    async def restore(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """
        Restore session state to a specific checkpoint.

        This doesn't actually replay tool calls — it provides the
        checkpoint data so the proxy can reconstruct state.
        """
        assert self._db is not None

        cursor = await self._db.execute(
            "SELECT * FROM checkpoints WHERE id = ?",
            (checkpoint_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            _log.warning(f"Checkpoint {checkpoint_id} not found")
            return None

        checkpoint = Checkpoint(
            id=row["id"],
            session_id=row["session_id"],
            branch=row["branch"],
            sequence=row["sequence"],
            label=row["label"],
            tool_call_count=row["tool_call_count"],
            token_count=row["token_count"],
            file_diff=row["file_diff"],
            tool_summary=row["tool_summary"],
            created_at=row["created_at"],
            auto_triggered=bool(row["auto_triggered"]),
        )

        # Update internal state
        self._current_branch = checkpoint.branch
        self._sequence = checkpoint.sequence
        self._cumulative_tokens = checkpoint.token_count

        _log.info(
            f"[mce.success]\\[TimeMachine] Restored to checkpoint "
            f"#{checkpoint.sequence}: {checkpoint.label}[/mce.success]"
        )

        return checkpoint

    # ── List / Query ──────────────────────────

    async def list_checkpoints(
        self,
        branch: Optional[str] = None,
        limit: int = 50,
    ) -> list[Checkpoint]:
        """List all checkpoints for the current session."""
        assert self._db is not None

        if branch:
            cursor = await self._db.execute(
                "SELECT * FROM checkpoints WHERE session_id = ? AND branch = ? "
                "ORDER BY sequence DESC LIMIT ?",
                (self._session_id, branch, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM checkpoints WHERE session_id = ? "
                "ORDER BY sequence DESC LIMIT ?",
                (self._session_id, limit),
            )

        rows = await cursor.fetchall()
        return [
            Checkpoint(
                id=r["id"],
                session_id=r["session_id"],
                branch=r["branch"],
                sequence=r["sequence"],
                label=r["label"],
                tool_call_count=r["tool_call_count"],
                token_count=r["token_count"],
                file_diff=r["file_diff"],
                tool_summary=r["tool_summary"],
                created_at=r["created_at"],
                auto_triggered=bool(r["auto_triggered"]),
            )
            for r in rows
        ]

    async def list_branches(self) -> list[BranchInfo]:
        """List all branches for the current session."""
        assert self._db is not None

        cursor = await self._db.execute(
            "SELECT b.*, COUNT(c.id) AS cp_count "
            "FROM branches b LEFT JOIN checkpoints c "
            "ON c.session_id = b.session_id AND c.branch = b.name "
            "WHERE b.session_id = ? GROUP BY b.name",
            (self._session_id,),
        )
        rows = await cursor.fetchall()
        return [
            BranchInfo(
                name=r["name"],
                parent_checkpoint_id=r["parent_checkpoint"],
                created_at=r["created_at"],
                checkpoint_count=r["cp_count"],
            )
            for r in rows
        ]

    @property
    def current_branch(self) -> str:
        return self._current_branch

    @property
    def checkpoint_count(self) -> int:
        return self._sequence

    # ── Git Diff Capture ──────────────────────

    def _capture_git_diff(self) -> str:
        """
        Capture the current git diff for file-level changes.
        Returns empty string if git is not available or not in a repo.
        """
        try:
            import subprocess
            result = subprocess.run(
                ["git", "diff", "--stat"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=Path.cwd(),
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()[:2000]  # Limit size
        except Exception:
            pass
        return ""

    # ── Summary for TUI ───────────────────────

    def get_timeline_summary(self) -> dict:
        """Return a summary dict for the TUI dashboard."""
        return {
            "checkpoints": self._sequence,
            "current_branch": self._current_branch,
            "tool_calls_since_cp": len(self._tool_calls),
            "cumulative_tokens": self._cumulative_tokens,
        }
