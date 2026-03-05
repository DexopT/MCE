"""
MCE — DriftSentinel Tests
Tests for constraint loading, violation detection, and severity handling.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.guardian.drift_sentinel import DriftSentinel, Constraint, DriftViolation
from schemas.mce_config import DriftSentinelConfig


@pytest.fixture
def ds_config():
    return DriftSentinelConfig(
        enabled=True,
        alert_on_constraint_violation=True,
        block_on_critical_violation=True,
        load_constraints_from_memvault=True,
    )


class TestDriftSentinelConstraints:
    def test_add_constraint(self, ds_config):
        ds = DriftSentinel(ds_config)
        ds.add_constraint(Constraint(
            id="c1",
            description="Don't touch the legacy API",
            pattern="legacy_api\\.py",
            severity="HIGH",
        ))
        assert ds.constraint_count == 1

    def test_multiple_constraints(self, ds_config):
        ds = DriftSentinel(ds_config)
        for i in range(5):
            ds.add_constraint(Constraint(
                id=f"c{i}",
                description=f"Constraint {i}",
                pattern=f"pattern_{i}",
            ))
        assert ds.constraint_count == 5


@pytest.mark.asyncio
class TestDriftSentinelDetection:
    async def test_detects_file_constraint_violation(self, ds_config):
        ds = DriftSentinel(ds_config)
        ds.add_constraint(Constraint(
            id="c1",
            description="Don't modify legacy_api.py",
            pattern="legacy_api\\.py",
            severity="HIGH",
        ))

        violation = await ds.check_tool_call(
            tool_name="write_file",
            arguments={"path": "src/legacy_api.py", "content": "new code"},
        )
        assert violation is not None
        assert violation.severity == "HIGH"
        assert "legacy_api" in violation.violation_detail

    async def test_no_violation_on_clean_call(self, ds_config):
        ds = DriftSentinel(ds_config)
        ds.add_constraint(Constraint(
            id="c1",
            description="Don't modify legacy_api.py",
            pattern="legacy_api\\.py",
            severity="HIGH",
        ))

        violation = await ds.check_tool_call(
            tool_name="read_file",
            arguments={"path": "src/new_module.py"},
        )
        assert violation is None

    async def test_critical_violation_sets_blocked(self, ds_config):
        ds = DriftSentinel(ds_config)
        ds.add_constraint(Constraint(
            id="c1",
            description="Never delete the database",
            pattern="database",
            severity="CRITICAL",
        ))

        violation = await ds.check_tool_call(
            tool_name="delete_file",
            arguments={"path": "database.sqlite"},
        )
        assert violation is not None
        assert violation.blocked is True

    async def test_non_critical_not_blocked(self, ds_config):
        ds = DriftSentinel(ds_config)
        ds.add_constraint(Constraint(
            id="c1",
            description="Prefer using TypeScript",
            pattern="typescript",
            severity="LOW",
        ))

        violation = await ds.check_tool_call(
            tool_name="write_file",
            arguments={"content": "some typescript code"},
        )
        assert violation is not None
        assert violation.blocked is False

    async def test_violation_tracking(self, ds_config):
        ds = DriftSentinel(ds_config)
        ds.add_constraint(Constraint(
            id="c1",
            description="Don't touch config",
            pattern="config",
        ))

        await ds.check_tool_call("read_file", {"path": "config.yaml"})
        await ds.check_tool_call("write_file", {"path": "config.yaml"})

        assert ds.violation_count == 2
        assert len(ds.recent_violations) == 2

    async def test_empty_constraints_returns_none(self, ds_config):
        ds = DriftSentinel(ds_config)
        violation = await ds.check_tool_call("read_file", {"path": "anything.py"})
        assert violation is None

    async def test_guardian_summary(self, ds_config):
        ds = DriftSentinel(ds_config)
        ds.add_constraint(Constraint(
            id="c1", description="Test", pattern="test"
        ))
        await ds.check_tool_call("write_file", {"content": "test data"})

        summary = ds.get_guardian_summary()
        assert summary["constraints"] == 1
        assert summary["violations"] == 1
