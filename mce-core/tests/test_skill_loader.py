"""
MCE — SkillForge Tests
Tests for skill file parsing, loading, and trigger matching.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.skills.skill_loader import SkillLoader
from engine.skills.skill_schema import Skill, SkillTriggers
from schemas.mce_config import SkillsConfig


@pytest.fixture
def skills_dir(tmp_path):
    """Create a temporary skills directory with test skill files."""
    skills = tmp_path / ".mce" / "skills"
    skills.mkdir(parents=True)

    # Write a test skill file
    (skills / "db-migration.skill.md").write_text(
        """---
name: db-migration
version: 1.2.0
triggers:
  tool_names: ["execute_sql", "run_migration"]
  keywords: ["migrate", "ALTER TABLE", "schema change"]
risk_level: high
requires_checkpoint: true
---

## Constraints
- Always check migrations/ numbering before creating new files
- Never run migrations without confirming backup exists
- Include rollback SQL in every migration file

## Workflow
1. Generate migration with correct version number
2. Validate rollback SQL
3. Confirm with user before applying
""",
        encoding="utf-8",
    )

    # Write another skill file
    (skills / "deploy.skill.md").write_text(
        """---
name: deploy
version: 1.0.0
triggers:
  tool_names: ["deploy", "push"]
  keywords: ["deploy", "production", "release"]
risk_level: medium
---

## Constraints
- Run tests before deploying
- Use staging first

## Workflow
1. Run test suite
2. Deploy to staging
3. Deploy to production
""",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def skill_config():
    return SkillsConfig(
        enabled=True,
        path=".mce/skills",
        auto_trigger=True,
    )


class TestSkillLoading:
    def test_load_skills(self, skills_dir, skill_config):
        loader = SkillLoader(skill_config)
        count = loader.load_skills(skills_dir)
        assert count == 2

    def test_skill_count(self, skills_dir, skill_config):
        loader = SkillLoader(skill_config)
        loader.load_skills(skills_dir)
        assert loader.skill_count == 2

    def test_get_skill_by_name(self, skills_dir, skill_config):
        loader = SkillLoader(skill_config)
        loader.load_skills(skills_dir)

        skill = loader.get_skill("db-migration")
        assert skill is not None
        assert skill.version == "1.2.0"
        assert skill.risk_level == "high"
        assert skill.requires_checkpoint is True

    def test_skill_triggers(self, skills_dir, skill_config):
        loader = SkillLoader(skill_config)
        loader.load_skills(skills_dir)

        skill = loader.get_skill("db-migration")
        assert "execute_sql" in skill.triggers.tool_names
        assert "migrate" in skill.triggers.keywords

    def test_skill_constraints(self, skills_dir, skill_config):
        loader = SkillLoader(skill_config)
        loader.load_skills(skills_dir)

        skill = loader.get_skill("db-migration")
        assert len(skill.constraints) == 3
        assert any("backup" in c for c in skill.constraints)

    def test_skill_workflow(self, skills_dir, skill_config):
        loader = SkillLoader(skill_config)
        loader.load_skills(skills_dir)

        skill = loader.get_skill("db-migration")
        assert len(skill.workflow) == 3
        assert any("rollback" in w for w in skill.workflow)

    def test_list_skills(self, skills_dir, skill_config):
        loader = SkillLoader(skill_config)
        loader.load_skills(skills_dir)

        skills = loader.list_skills()
        names = {s.name for s in skills}
        assert "db-migration" in names
        assert "deploy" in names

    def test_empty_directory(self, tmp_path, skill_config):
        loader = SkillLoader(skill_config)
        count = loader.load_skills(tmp_path)
        assert count == 0

    def test_nonexistent_directory(self, skill_config):
        loader = SkillLoader(skill_config)
        count = loader.load_skills("/nonexistent/path")
        assert count == 0


class TestSkillMatching:
    def test_match_by_tool_name(self, skills_dir, skill_config):
        loader = SkillLoader(skill_config)
        loader.load_skills(skills_dir)

        matches = loader.match_tool_call("execute_sql", {"query": "SELECT 1"})
        assert len(matches) == 1
        assert matches[0].name == "db-migration"

    def test_match_by_keyword(self, skills_dir, skill_config):
        loader = SkillLoader(skill_config)
        loader.load_skills(skills_dir)

        matches = loader.match_tool_call(
            "write_file",
            {"content": "ALTER TABLE users ADD COLUMN email TEXT"},
        )
        assert len(matches) == 1
        assert matches[0].name == "db-migration"

    def test_no_match(self, skills_dir, skill_config):
        loader = SkillLoader(skill_config)
        loader.load_skills(skills_dir)

        matches = loader.match_tool_call("read_file", {"path": "src/main.py"})
        assert len(matches) == 0

    def test_auto_trigger_disabled(self, skills_dir, skill_config):
        skill_config.auto_trigger = False
        loader = SkillLoader(skill_config)
        loader.load_skills(skills_dir)

        matches = loader.match_tool_call("execute_sql", {"query": "SELECT 1"})
        assert len(matches) == 0

    def test_skills_summary(self, skills_dir, skill_config):
        loader = SkillLoader(skill_config)
        loader.load_skills(skills_dir)

        summary = loader.get_skills_summary()
        assert summary["skill_count"] == 2
        assert len(summary["skills"]) == 2
