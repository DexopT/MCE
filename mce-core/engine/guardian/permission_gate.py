"""
MCE — PermissionGate
Profile-based permission system that replaces binary HitL prompts
with configurable profiles (exploration, focused_work, review).

Extends the existing PolicyEngine by providing a pre-filter:
before PolicyEngine evaluates rules, PermissionGate checks whether
the active profile auto-allows or auto-blocks the action category.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from schemas.mce_config import PermissionProfilesConfig, PermissionProfile
from utils.logger import get_logger

_log = get_logger("PermissionGate")


class GateDecision(str, Enum):
    """Result of a PermissionGate check."""
    AUTO_ALLOW = "auto"       # Proceed without prompting
    PROMPT = "prompt"         # Defer to PolicyEngine / HitL
    BLOCK = "block"           # Block outright


@dataclass
class GateResult:
    """Result from a PermissionGate evaluation."""
    decision: GateDecision
    category: str             # "file_read" | "file_write" | "shell_exec" | "destructive"
    profile_name: str
    reason: str = ""


# Tool-to-category mapping
TOOL_CATEGORIES: dict[str, str] = {
    # File read
    "read_file": "file_read",
    "view_file": "file_read",
    "list_directory": "file_read",
    "search_files": "file_read",
    "grep_search": "file_read",

    # File write
    "write_file": "file_write",
    "edit_file": "file_write",
    "create_file": "file_write",
    "replace_file_content": "file_write",
    "multi_replace_file_content": "file_write",
    "rename_file": "file_write",

    # Shell execution
    "execute_command": "shell_exec",
    "run_command": "shell_exec",
    "shell_exec": "shell_exec",

    # Destructive
    "delete_file": "destructive",
    "rm": "destructive",
    "rmdir": "destructive",
}


class PermissionGate:
    """
    Smart permission profile system.

    Evaluates tool calls against the active permission profile
    before they reach the PolicyEngine. This replaces the binary
    hitl_commands list with nuanced, switchable profiles.
    """

    def __init__(self, config: PermissionProfilesConfig):
        self._config = config
        self._active_profile_name = config.active
        self._profiles = config.profiles
        self._override_count = 0

    @property
    def active_profile_name(self) -> str:
        return self._active_profile_name

    @property
    def active_profile(self) -> PermissionProfile:
        """Get the active permission profile, falling back to focused_work."""
        return self._profiles.get(
            self._active_profile_name,
            PermissionProfile()
        )

    def switch_profile(self, profile_name: str) -> bool:
        """Switch to a different permission profile."""
        if profile_name not in self._profiles:
            _log.warning(f"Profile '{profile_name}' not found")
            return False

        old = self._active_profile_name
        self._active_profile_name = profile_name
        _log.info(
            f"[mce.success]\\[PermissionGate] Profile switched: "
            f"{old} → {profile_name}[/mce.success]"
        )
        return True

    def check(self, tool_name: str) -> GateResult:
        """
        Evaluate a tool call against the active permission profile.

        Returns a GateResult indicating whether the call should be
        auto-allowed, prompt the user, or be blocked.
        """
        category = self._categorize_tool(tool_name)
        profile = self.active_profile
        profile_name = self._active_profile_name

        # Get the permission setting for this category
        setting = getattr(profile, category, "prompt")

        if setting == "auto":
            return GateResult(
                decision=GateDecision.AUTO_ALLOW,
                category=category,
                profile_name=profile_name,
                reason=f"Auto-allowed by '{profile_name}' profile",
            )
        elif setting == "block":
            return GateResult(
                decision=GateDecision.BLOCK,
                category=category,
                profile_name=profile_name,
                reason=f"Blocked by '{profile_name}' profile ({category})",
            )
        else:  # "prompt"
            return GateResult(
                decision=GateDecision.PROMPT,
                category=category,
                profile_name=profile_name,
                reason=f"Requires approval under '{profile_name}' profile",
            )

    def _categorize_tool(self, tool_name: str) -> str:
        """Map a tool name to a permission category."""
        # Direct mapping
        if tool_name in TOOL_CATEGORIES:
            return TOOL_CATEGORIES[tool_name]

        # Heuristic categorization
        name_lower = tool_name.lower()
        if any(k in name_lower for k in ("read", "view", "list", "search", "get", "find")):
            return "file_read"
        if any(k in name_lower for k in ("write", "edit", "create", "update", "replace")):
            return "file_write"
        if any(k in name_lower for k in ("exec", "run", "shell", "command")):
            return "shell_exec"
        if any(k in name_lower for k in ("delete", "remove", "destroy", "drop")):
            return "destructive"

        # Default to prompt
        return "shell_exec"

    def list_profiles(self) -> dict[str, dict]:
        """List all available profiles."""
        return {
            name: {
                "file_read": p.file_read,
                "file_write": p.file_write,
                "shell_exec": p.shell_exec,
                "destructive": p.destructive,
                "active": name == self._active_profile_name,
            }
            for name, p in self._profiles.items()
        }

    def get_gate_summary(self) -> dict:
        """Return summary for TUI dashboard."""
        return {
            "active_profile": self._active_profile_name,
            "profile_count": len(self._profiles),
        }
