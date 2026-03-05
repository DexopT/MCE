"""
MCE — Skill Loader
Parses .skill.md and .skill.yaml files from the skills directory.
Matches skills to tool calls via keyword/tool_name triggers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import yaml

from engine.skills.skill_schema import Skill, SkillTriggers
from schemas.mce_config import SkillsConfig
from utils.logger import get_logger

_log = get_logger("SkillForge")


class SkillLoader:
    """
    Loads and manages SkillForge skill files.

    Scans the configured skills directory for .skill.md and .skill.yaml
    files, parses their frontmatter + content, and matches them against
    tool calls via keyword/tool_name triggers.
    """

    def __init__(self, config: SkillsConfig):
        self._config = config
        self._skills: dict[str, Skill] = {}  # name → Skill

    # ── Loading ───────────────────────────────

    def load_skills(self, project_path: Optional[str | Path] = None) -> int:
        """
        Scan the skills directory and load all skill files.
        Returns the number of skills loaded.
        """
        if project_path:
            skills_dir = Path(project_path) / self._config.path
        else:
            skills_dir = Path(self._config.path)

        if not skills_dir.exists():
            _log.debug(f"Skills directory not found: {skills_dir}")
            return 0

        count = 0
        # Look for .skill.md and .skill.yaml files
        for pattern in ["*.skill.md", "*.skill.yaml", "*.skill.yml"]:
            for skill_file in skills_dir.glob(pattern):
                try:
                    skill = self._parse_skill_file(skill_file)
                    self._skills[skill.name] = skill
                    count += 1
                    _log.debug(f"Loaded skill: {skill.name} (v{skill.version})")
                except Exception as exc:
                    _log.warning(f"Failed to parse skill file {skill_file}: {exc}")

        if count > 0:
            _log.info(
                f"[mce.success]\\[SkillForge] Loaded {count} skills[/mce.success]"
            )
        return count

    def _parse_skill_file(self, path: Path) -> Skill:
        """
        Parse a skill file with YAML frontmatter + markdown content.

        Format:
        ---
        name: skill-name
        version: 1.0.0
        triggers:
          tool_names: [...]
          keywords: [...]
        ---
        ## Constraints
        - ...
        ## Workflow
        1. ...
        """
        content = path.read_text(encoding="utf-8")
        frontmatter, body = self._split_frontmatter(content)

        if not frontmatter:
            raise ValueError(f"No YAML frontmatter found in {path}")

        meta = yaml.safe_load(frontmatter) or {}

        # Parse triggers
        triggers_raw = meta.get("triggers", {})
        triggers = SkillTriggers(
            tool_names=triggers_raw.get("tool_names", []),
            keywords=triggers_raw.get("keywords", []),
        )

        # Extract constraints and workflow from markdown body
        constraints = self._extract_section(body, "Constraints")
        workflow = self._extract_section(body, "Workflow")

        return Skill(
            name=meta.get("name", path.stem.replace(".skill", "")),
            version=meta.get("version", "1.0.0"),
            triggers=triggers,
            risk_level=meta.get("risk_level", "low"),
            requires_checkpoint=meta.get("requires_checkpoint", False),
            constraints=constraints,
            workflow=workflow,
            raw_content=body,
            source_file=str(path),
        )

    def _split_frontmatter(self, content: str) -> tuple[str, str]:
        """Split YAML frontmatter from markdown body."""
        match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return "", content

    def _extract_section(self, markdown: str, heading: str) -> list[str]:
        """Extract bullet points from a markdown section."""
        pattern = rf'##\s+{heading}.*?\n((?:[-*]\s+.*\n?|\d+\.\s+.*\n?)*)'
        match = re.search(pattern, markdown, re.IGNORECASE)
        if not match:
            return []

        items = []
        for line in match.group(1).strip().split("\n"):
            line = line.strip()
            if line.startswith(("-", "*")):
                items.append(line.lstrip("-* ").strip())
            elif re.match(r'\d+\.', line):
                items.append(re.sub(r'^\d+\.\s*', '', line).strip())
        return items

    # ── Matching ──────────────────────────────

    def match_tool_call(
        self,
        tool_name: str,
        arguments: dict,
    ) -> list[Skill]:
        """
        Match a tool call against all loaded skills.
        Returns skills that match via tool_name or keyword triggers.
        """
        if not self._config.auto_trigger:
            return []

        matched: list[Skill] = []
        args_text = str(arguments).lower()

        for skill in self._skills.values():
            # Match by tool name
            if tool_name in skill.triggers.tool_names:
                matched.append(skill)
                continue

            # Match by keyword
            for keyword in skill.triggers.keywords:
                if keyword.lower() in args_text or keyword.lower() in tool_name.lower():
                    matched.append(skill)
                    break

        return matched

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(name)

    def list_skills(self) -> list[Skill]:
        """List all loaded skills."""
        return list(self._skills.values())

    @property
    def skill_count(self) -> int:
        return len(self._skills)

    def get_skills_summary(self) -> dict:
        """Summary for TUI / status."""
        return {
            "skill_count": self.skill_count,
            "skills": [
                {"name": s.name, "version": s.version, "risk": s.risk_level}
                for s in self._skills.values()
            ],
        }
