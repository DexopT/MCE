"""
MCE — MemVault
Cross-session persistent memory built from MCE's tool call history.

MCE already records every tool call via context_manager.py — MemVault
adds semantic extraction and persistent storage on top. After each session,
it extracts decisions, dead ends, constraints, and preferences from tool
call patterns. At session start, it injects the most relevant memories
into the agent's context.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from models.memory_store import Memory, MemoryStore, ToolCallLog
from schemas.mce_config import MemVaultConfig
from utils.logger import get_logger

_log = get_logger("MemVault")


# ──────────────────────────────────────────────
# Heuristic Extraction Patterns
# ──────────────────────────────────────────────

# Keywords that signal architectural decisions
DECISION_PATTERNS = [
    re.compile(r"\b(decided|chose|using|went with|switched to|prefer|adopting)\b", re.I),
    re.compile(r"\b(will use|should use|must use|better to use)\b", re.I),
    re.compile(r"\b(architecture|pattern|approach|strategy|design)\b", re.I),
]

# Keywords that signal dead ends
DEAD_END_PATTERNS = [
    re.compile(r"\b(didn'?t work|doesn'?t work|won'?t work|failed|broken)\b", re.I),
    re.compile(r"\b(dead end|dead-end|abandon|give up|scrap|rollback|revert)\b", re.I),
    re.compile(r"\b(error|exception|crash|bug|issue|problem)\b", re.I),
]

# Keywords that signal constraints
CONSTRAINT_PATTERNS = [
    re.compile(r"\b(don'?t touch|never|always|must not|should not|do not)\b", re.I),
    re.compile(r"\b(constraint|requirement|restriction|rule|policy)\b", re.I),
    re.compile(r"\b(off[ -]limits|forbidden|protected|locked)\b", re.I),
]

# Keywords that signal preferences
PREFERENCE_PATTERNS = [
    re.compile(r"\b(prefer|like|want|style|convention|standard)\b", re.I),
    re.compile(r"\b(format|naming|pattern|template)\b", re.I),
]


class MemVault:
    """
    Cross-session persistent memory for AI agent sessions.

    Observes tool calls in real-time, extracts learnings at session end,
    and injects relevant context at session start.
    """

    def __init__(
        self,
        config: MemVaultConfig,
        project_id: str,
        session_id: str,
        store: MemoryStore,
    ):
        self._config = config
        self._project_id = project_id
        self._session_id = session_id
        self._store = store

        # In-memory buffer for current session observations
        self._observation_buffer: list[dict[str, Any]] = []

    # ── Real-time Observation ─────────────────

    async def observe(
        self,
        tool_name: str,
        arguments: dict,
        response: Any,
        tokens_in: int = 0,
        tokens_out: int = 0,
        duration_ms: int = 0,
    ) -> None:
        """
        Called on every tool call through the proxy.
        Logs the call and buffers observations for end-of-session extraction.
        """
        now = datetime.now(timezone.utc).isoformat()
        call_id = hashlib.sha256(
            f"{self._session_id}:{tool_name}:{now}".encode()
        ).hexdigest()[:16]

        # Buffer observation for extraction
        self._observation_buffer.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "response_summary": self._summarize_response(response),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "timestamp": now,
        })

        # Log to SQLite
        try:
            log_entry = ToolCallLog(
                id=call_id,
                session_id=self._session_id,
                tool_name=tool_name,
                request=json.dumps(arguments, default=str)[:5000],
                response=self._summarize_response(response)[:5000],
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                duration_ms=duration_ms,
                timestamp=now,
            )
            await self._store.log_tool_call(log_entry)
        except Exception as exc:
            _log.debug(f"Failed to log tool call: {exc}")

    def _summarize_response(self, response: Any) -> str:
        """Create a compact summary of the response for logging."""
        if isinstance(response, str):
            return response[:2000]
        try:
            text = json.dumps(response, default=str)
            return text[:2000]
        except Exception:
            return str(response)[:2000]

    # ── Session End: Extract Learnings ────────

    async def extract_session_learnings(self) -> list[Memory]:
        """
        Run at session end. Extract decisions, constraints, dead ends,
        and preferences from the tool call log.

        Uses heuristic extraction by default (regex pattern matching).
        """
        if not self._observation_buffer:
            _log.debug("No observations to extract from")
            return []

        _log.info(
            f"[mce.badge]\\[MemVault][/mce.badge] Extracting learnings from "
            f"{len(self._observation_buffer)} tool calls"
        )

        memories: list[Memory] = []
        now = datetime.now(timezone.utc).isoformat()

        for obs in self._observation_buffer:
            # Combine arguments + response text for pattern matching
            text = f"{obs['tool_name']} {json.dumps(obs['arguments'], default=str)} {obs['response_summary']}"

            # Check each pattern category
            for mem_type, patterns in [
                ("decision", DECISION_PATTERNS),
                ("dead_end", DEAD_END_PATTERNS),
                ("constraint", CONSTRAINT_PATTERNS),
                ("preference", PREFERENCE_PATTERNS),
            ]:
                if any(p.search(text) for p in patterns):
                    memory_id = hashlib.sha256(
                        f"{self._project_id}:{mem_type}:{text[:200]}".encode()
                    ).hexdigest()[:16]

                    memory = Memory(
                        id=memory_id,
                        project_id=self._project_id,
                        type=mem_type,
                        content=self._format_memory(obs, mem_type),
                        source_tool=obs["tool_name"],
                        created_at=now,
                        last_seen=now,
                        confidence=0.8,
                    )
                    memories.append(memory)

            # Detect file patterns (files read/written frequently)
            if obs["tool_name"] in ("read_file", "write_file", "edit_file"):
                file_path = obs["arguments"].get("path", obs["arguments"].get("file", ""))
                if file_path:
                    memory_id = hashlib.sha256(
                        f"{self._project_id}:file_pattern:{file_path}".encode()
                    ).hexdigest()[:16]

                    memory = Memory(
                        id=memory_id,
                        project_id=self._project_id,
                        type="file_pattern",
                        content=f"File frequently accessed: {file_path}",
                        source_tool=obs["tool_name"],
                        created_at=now,
                        last_seen=now,
                        confidence=0.6,
                    )
                    memories.append(memory)

        # Deduplicate by ID and persist
        seen_ids = set()
        unique_memories = []
        for m in memories:
            if m.id not in seen_ids:
                seen_ids.add(m.id)
                unique_memories.append(m)

        for memory in unique_memories:
            try:
                await self._store.save_memory(memory)
            except Exception as exc:
                _log.debug(f"Failed to save memory: {exc}")

        _log.info(
            f"[mce.success]\\[MemVault] Extracted {len(unique_memories)} memories[/mce.success]"
        )
        return unique_memories

    def _format_memory(self, observation: dict, mem_type: str) -> str:
        """Format a memory entry from an observation."""
        tool = observation["tool_name"]
        args_summary = json.dumps(observation["arguments"], default=str)[:200]
        response_preview = observation["response_summary"][:200]
        ts = observation["timestamp"][:10]  # Date only

        if mem_type == "decision":
            return f"[{ts}] Decision via {tool}: {args_summary}"
        elif mem_type == "dead_end":
            return f"[{ts}] Dead end via {tool}: {args_summary} → {response_preview}"
        elif mem_type == "constraint":
            return f"[{ts}] Constraint via {tool}: {args_summary}"
        elif mem_type == "preference":
            return f"[{ts}] Preference via {tool}: {args_summary}"
        return f"[{ts}] {mem_type} via {tool}: {args_summary}"

    # ── Session Start: Inject Context ─────────

    async def inject_context(self, token_budget: Optional[int] = None) -> str:
        """
        Retrieve the most relevant memories and format them for injection
        into the agent's system prompt or CLAUDE.md.

        Respects the token budget — never injects more than allowed.
        """
        budget = token_budget or self._config.injection_token_budget

        # Get recent memories, prioritizing by type
        all_memories: list[Memory] = []
        for mem_type in self._config.memory_types:
            memories = await self._store.get_memories(
                self._project_id, memory_type=mem_type, limit=20
            )
            all_memories.extend(memories)

        if not all_memories:
            return ""

        # Build the injection block
        lines: list[str] = [
            "## MCE Session Memory [Auto-injected]",
        ]

        # Group by type
        by_type: dict[str, list[Memory]] = {}
        for m in all_memories:
            by_type.setdefault(m.type, []).append(m)

        type_headers = {
            "decision": "### Architectural Decisions",
            "dead_end": "### Dead Ends (Do Not Retry)",
            "constraint": "### Active Constraints",
            "preference": "### Preferences",
            "file_pattern": "### Key Files",
        }

        # Simple token budget: estimate ~4 chars per token
        char_budget = budget * 4
        current_chars = sum(len(l) for l in lines)

        for mem_type, header in type_headers.items():
            if mem_type not in by_type:
                continue
            lines.append(header)
            for m in by_type[mem_type][:10]:  # Max 10 per type
                line = f"- {m.content}"
                if current_chars + len(line) > char_budget:
                    break
                lines.append(line)
                current_chars += len(line)

        return "\n".join(lines)

    async def get_memory_count(self) -> int:
        """Return total memory count for the current project."""
        return await self._store.count_memories(self._project_id)

    # ── CLAUDE.md Integration ─────────────────

    async def update_claude_md(self, project_path: Path) -> None:
        """
        Auto-update CLAUDE.md with a fresh memory block.
        Only writes if auto_update_claude_md is enabled in config.
        """
        if not self._config.auto_update_claude_md:
            return

        context_block = await self.inject_context()
        if not context_block:
            return

        claude_md = project_path / "CLAUDE.md"
        marker_start = "<!-- MCE:MEMORY:START -->"
        marker_end = "<!-- MCE:MEMORY:END -->"

        memory_section = f"\n{marker_start}\n{context_block}\n{marker_end}\n"

        if claude_md.exists():
            content = claude_md.read_text(encoding="utf-8")
            if marker_start in content:
                # Replace existing block
                start = content.index(marker_start)
                end = content.index(marker_end) + len(marker_end)
                content = content[:start] + memory_section.strip() + content[end:]
            else:
                # Append
                content = content.rstrip() + "\n\n" + memory_section
        else:
            content = memory_section

        claude_md.write_text(content, encoding="utf-8")
        _log.info("[mce.success]\\[MemVault] Updated CLAUDE.md[/mce.success]")
