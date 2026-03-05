"""
MCE — Policy Engine (Agent Firewall & Sandboxing)
Inspects tool payloads for destructive commands and enforces security policies.
"""

from __future__ import annotations

import asyncio
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from schemas.mce_config import PolicyConfig
from utils.logger import get_logger, log_policy_block

_log = get_logger("Policy")


# ──────────────────────────────────────────────
# Decision Types
# ──────────────────────────────────────────────

class PolicyDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    HITL = "hitl"  # Human-in-the-Loop


@dataclass
class PolicyResult:
    """Result of a policy check."""
    decision: PolicyDecision
    reason: str = ""
    matched_rule: str = ""


# ──────────────────────────────────────────────
# Policy Engine
# ──────────────────────────────────────────────

class PolicyEngine:
    """
    Regex-based command scanner.

    Checks tool payloads against:
    - Blocked command patterns (rm -rf, mkfs, etc.)
    - Blocked network targets
    - Human-in-the-loop triggers (DROP, git push, etc.)
    """

    def __init__(self, config: PolicyConfig | None = None):
        cfg = config or PolicyConfig()
        self._blocked_cmds = [re.compile(re.escape(p), re.IGNORECASE) for p in cfg.blocked_commands]
        self._blocked_cmd_raw = cfg.blocked_commands
        self._blocked_net = cfg.blocked_network
        self._hitl_patterns = [re.compile(re.escape(p), re.IGNORECASE) for p in cfg.hitl_commands]
        self._hitl_raw = cfg.hitl_commands

    def check(self, tool_name: str, payload: Any) -> PolicyResult:
        """
        Evaluate a tool call against all policy rules.

        Args:
            tool_name: The MCP tool being called.
            payload: The arguments / payload to inspect.

        Returns:
            PolicyResult with decision (allow / block / hitl).
        """
        text = self._to_text(payload)

        # 1. Check blocked commands
        for idx, pattern in enumerate(self._blocked_cmds):
            if pattern.search(text):
                reason = (
                    f"[MCE Blocked: Destructive command not permitted in current policy] "
                    f"Matched: '{self._blocked_cmd_raw[idx]}'"
                )
                log_policy_block(tool_name, reason)
                return PolicyResult(
                    decision=PolicyDecision.BLOCK,
                    reason=reason,
                    matched_rule=self._blocked_cmd_raw[idx],
                )

        # 2. Check blocked network targets
        for target in self._blocked_net:
            if target in text:
                reason = (
                    f"[MCE Blocked: Unauthorized network target] "
                    f"Matched: '{target}'"
                )
                log_policy_block(tool_name, reason)
                return PolicyResult(
                    decision=PolicyDecision.BLOCK,
                    reason=reason,
                    matched_rule=target,
                )

        # 3. Check HitL triggers
        for idx, pattern in enumerate(self._hitl_patterns):
            if pattern.search(text):
                reason = (
                    f"[MCE HitL: High-risk command requires human approval] "
                    f"Matched: '{self._hitl_raw[idx]}'"
                )
                _log.warning(
                    f"[mce.warning]HitL triggered[/mce.warning]: "
                    f"{tool_name} — {self._hitl_raw[idx]}"
                )
                return PolicyResult(
                    decision=PolicyDecision.HITL,
                    reason=reason,
                    matched_rule=self._hitl_raw[idx],
                )

        return PolicyResult(decision=PolicyDecision.ALLOW)

    async def prompt_human(self, tool_name: str, result: PolicyResult) -> bool:
        """
        Prompt the operator for approval on HitL commands.

        Sends a Y/N prompt to the terminal. If no interactive terminal
        is available, defaults to DENY (safe default).
        """
        if not sys.stdin.isatty():
            _log.warning(
                "[mce.badge]\\[MCE HitL][/mce.badge] "
                f"Non-interactive terminal — auto-denying '{tool_name}'"
            )
            return False

        prompt_text = (
            f"\n[MCE HitL] Tool '{tool_name}' requires approval.\n"
            f"  Rule: {result.matched_rule}\n"
            f"  Approve? (Y/N): "
        )

        try:
            # Run blocking input() in a thread to avoid blocking the event loop
            response = await asyncio.to_thread(input, prompt_text)
            approved = response.strip().upper() in ("Y", "YES")
            if approved:
                _log.info(
                    f"[mce.success]HitL APPROVED[/mce.success]: {tool_name}"
                )
            else:
                _log.warning(
                    f"[mce.warning]HitL DENIED[/mce.warning]: {tool_name}"
                )
            return approved
        except (EOFError, KeyboardInterrupt):
            _log.warning("HitL prompt interrupted — denying")
            return False

    @staticmethod
    def _to_text(payload: Any) -> str:
        """Flatten payload into a searchable string."""
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            import json
            return json.dumps(payload, default=str)
        return str(payload)
