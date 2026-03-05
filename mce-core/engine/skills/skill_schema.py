"""
MCE — Skill Schema
Pydantic models for SkillForge skill files (.skill.md / .skill.yaml).
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class SkillTriggers(BaseModel):
    """Conditions that activate a skill."""
    tool_names: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class Skill(BaseModel):
    """A loaded skill definition."""
    name: str
    version: str = "1.0.0"
    triggers: SkillTriggers = Field(default_factory=SkillTriggers)
    risk_level: str = "low"         # "low" | "medium" | "high"
    requires_checkpoint: bool = False
    constraints: list[str] = Field(default_factory=list)
    workflow: list[str] = Field(default_factory=list)
    raw_content: str = ""           # Full markdown content
    source_file: str = ""           # Path to the source file
