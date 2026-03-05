"""
MCE — DriftSentinel
Monitors live tool calls for constraint violations. Extracts
constraints from MemVault at session start and checks every
subsequent tool call against them.

Runs as middleware in the proxy pipeline — after the squeeze
engine but before the response is returned to the agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from schemas.mce_config import DriftSentinelConfig
from utils.logger import get_logger

_log = get_logger("DriftSentinel")


# ──────────────────────────────────────────────
# Data Types
# ──────────────────────────────────────────────

@dataclass
class Constraint:
    """An extracted constraint from MemVault or config."""
    id: str
    description: str
    pattern: str           # regex or keyword
    severity: str = "MEDIUM"  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    source: str = "memvault"   # "memvault" | "config" | "user"

    @property
    def compiled(self) -> re.Pattern:
        try:
            return re.compile(self.pattern, re.IGNORECASE)
        except re.error:
            return re.compile(re.escape(self.pattern), re.IGNORECASE)


@dataclass
class DriftViolation:
    """A detected constraint violation."""
    constraint: Constraint
    tool_name: str
    violation_detail: str
    severity: str
    timestamp: str = ""
    blocked: bool = False


# ──────────────────────────────────────────────
# DriftSentinel
# ──────────────────────────────────────────────

class DriftSentinel:
    """
    Constraint drift detection engine.

    Monitors tool calls for violations of extracted constraints.
    Supports both static constraints (from config) and dynamic
    constraints (from MemVault's extracted memories).
    """

    def __init__(self, config: DriftSentinelConfig):
        self._config = config
        self._constraints: list[Constraint] = []
        self._violations: list[DriftViolation] = []

    # ── Constraint Loading ────────────────────

    async def load_constraints_from_memvault(self, memvault) -> int:
        """
        Extract constraints from MemVault's stored memories.

        Looks for memories of type "constraint" and converts them
        into active monitoring rules.
        """
        if not self._config.load_constraints_from_memvault:
            return 0

        if memvault is None:
            return 0

        try:
            memories = await memvault._store.get_memories(
                memvault._project_id, memory_type="constraint", limit=50
            )

            for mem in memories:
                # Extract keywords from the constraint content
                constraint = Constraint(
                    id=mem.id,
                    description=mem.content,
                    pattern=self._extract_pattern(mem.content),
                    severity="HIGH",
                    source="memvault",
                )
                self._constraints.append(constraint)

            _log.info(
                f"[mce.badge]\\[DriftSentinel][/mce.badge] Loaded "
                f"{len(memories)} constraints from MemVault"
            )
            return len(memories)

        except Exception as exc:
            _log.debug(f"Failed to load constraints from MemVault: {exc}")
            return 0

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint manually (from config or user)."""
        self._constraints.append(constraint)

    def _extract_pattern(self, content: str) -> str:
        """
        Extract a searchable pattern from constraint text.

        Looks for file paths, tool names, and key phrases
        like "don't touch", "never modify", etc.
        """
        patterns = []

        # Extract file paths
        file_matches = re.findall(r'[\w./\\-]+\.\w+', content)
        patterns.extend(file_matches)

        # Extract key forbidden actions
        for keyword in ["don't touch", "never", "must not", "do not",
                        "off limits", "forbidden", "protected"]:
            if keyword.lower() in content.lower():
                # Extract the object of the constraint
                idx = content.lower().index(keyword.lower())
                context = content[idx:idx+80]
                patterns.append(re.escape(context[:40]))

        if patterns:
            return "|".join(patterns)
        # Fallback: use first significant words
        words = [w for w in content.split() if len(w) > 4][:5]
        return "|".join(re.escape(w) for w in words) if words else content[:30]

    # ── Live Monitoring ───────────────────────

    async def check_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        response: Any = None,
    ) -> Optional[DriftViolation]:
        """
        Check a tool call against all active constraints.

        Returns a DriftViolation if a constraint is violated, None otherwise.
        Called from the proxy pipeline after squeeze, before return.
        """
        if not self._constraints:
            return None

        # Build searchable text from tool call
        search_text = f"{tool_name} {self._serialize(arguments)}"
        if response is not None:
            search_text += f" {self._serialize(response)}"

        for constraint in self._constraints:
            compiled = constraint.compiled
            if compiled.search(search_text):
                violation = DriftViolation(
                    constraint=constraint,
                    tool_name=tool_name,
                    violation_detail=(
                        f"Tool '{tool_name}' may violate constraint: "
                        f"{constraint.description}"
                    ),
                    severity=constraint.severity,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    blocked=(
                        constraint.severity == "CRITICAL"
                        and self._config.block_on_critical_violation
                    ),
                )

                self._violations.append(violation)

                if self._config.alert_on_constraint_violation:
                    _log.warning(
                        f"[mce.warning]\\[DriftSentinel] VIOLATION: "
                        f"{violation.violation_detail}[/mce.warning]"
                    )

                return violation

        return None

    def _serialize(self, data: Any) -> str:
        """Convert data to searchable string."""
        if isinstance(data, str):
            return data
        try:
            import json
            return json.dumps(data, default=str)
        except Exception:
            return str(data)

    # ── Status ────────────────────────────────

    @property
    def constraint_count(self) -> int:
        return len(self._constraints)

    @property
    def violation_count(self) -> int:
        return len(self._violations)

    @property
    def recent_violations(self) -> list[DriftViolation]:
        return list(self._violations[-10:])

    def get_guardian_summary(self) -> dict:
        """Return summary for TUI dashboard."""
        return {
            "constraints": self.constraint_count,
            "violations": self.violation_count,
            "recent_violations": [
                {
                    "tool": v.tool_name,
                    "severity": v.severity,
                    "detail": v.violation_detail[:60],
                }
                for v in self._violations[-5:]
            ],
        }
