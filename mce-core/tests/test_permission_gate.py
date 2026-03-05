"""
MCE — PermissionGate Tests
Tests for profile-based permission system.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.guardian.permission_gate import PermissionGate, GateDecision
from schemas.mce_config import PermissionProfilesConfig, PermissionProfile


@pytest.fixture
def gate_config():
    return PermissionProfilesConfig(
        active="focused_work",
        profiles={
            "exploration": PermissionProfile(
                file_read="auto", file_write="prompt", shell_exec="prompt", destructive="block"
            ),
            "focused_work": PermissionProfile(
                file_read="auto", file_write="auto", shell_exec="prompt", destructive="prompt"
            ),
            "review": PermissionProfile(
                file_read="auto", file_write="block", shell_exec="block", destructive="block"
            ),
        },
    )


class TestPermissionGateProfiles:
    def test_default_profile(self, gate_config):
        gate = PermissionGate(gate_config)
        assert gate.active_profile_name == "focused_work"

    def test_switch_profile(self, gate_config):
        gate = PermissionGate(gate_config)
        assert gate.switch_profile("exploration")
        assert gate.active_profile_name == "exploration"

    def test_switch_invalid_profile(self, gate_config):
        gate = PermissionGate(gate_config)
        assert not gate.switch_profile("nonexistent")
        assert gate.active_profile_name == "focused_work"

    def test_list_profiles(self, gate_config):
        gate = PermissionGate(gate_config)
        profiles = gate.list_profiles()
        assert "exploration" in profiles
        assert "focused_work" in profiles
        assert "review" in profiles
        assert profiles["focused_work"]["active"] is True


class TestPermissionGateChecks:
    def test_focused_work_auto_allows_file_read(self, gate_config):
        gate = PermissionGate(gate_config)
        result = gate.check("read_file")
        assert result.decision == GateDecision.AUTO_ALLOW
        assert result.category == "file_read"

    def test_focused_work_auto_allows_file_write(self, gate_config):
        gate = PermissionGate(gate_config)
        result = gate.check("write_file")
        assert result.decision == GateDecision.AUTO_ALLOW
        assert result.category == "file_write"

    def test_focused_work_prompts_shell_exec(self, gate_config):
        gate = PermissionGate(gate_config)
        result = gate.check("execute_command")
        assert result.decision == GateDecision.PROMPT

    def test_focused_work_prompts_destructive(self, gate_config):
        gate = PermissionGate(gate_config)
        result = gate.check("delete_file")
        assert result.decision == GateDecision.PROMPT

    def test_exploration_blocks_destructive(self, gate_config):
        gate = PermissionGate(gate_config)
        gate.switch_profile("exploration")
        result = gate.check("delete_file")
        assert result.decision == GateDecision.BLOCK

    def test_review_blocks_writes(self, gate_config):
        gate = PermissionGate(gate_config)
        gate.switch_profile("review")
        result = gate.check("write_file")
        assert result.decision == GateDecision.BLOCK

    def test_review_allows_reads(self, gate_config):
        gate = PermissionGate(gate_config)
        gate.switch_profile("review")
        result = gate.check("read_file")
        assert result.decision == GateDecision.AUTO_ALLOW

    def test_review_blocks_shell(self, gate_config):
        gate = PermissionGate(gate_config)
        gate.switch_profile("review")
        result = gate.check("run_command")
        assert result.decision == GateDecision.BLOCK


class TestPermissionGateCategorization:
    def test_known_read_tools(self, gate_config):
        gate = PermissionGate(gate_config)
        for tool in ["read_file", "view_file", "list_directory", "search_files"]:
            result = gate.check(tool)
            assert result.category == "file_read", f"{tool} should be file_read"

    def test_known_write_tools(self, gate_config):
        gate = PermissionGate(gate_config)
        for tool in ["write_file", "edit_file", "create_file", "replace_file_content"]:
            result = gate.check(tool)
            assert result.category == "file_write", f"{tool} should be file_write"

    def test_known_exec_tools(self, gate_config):
        gate = PermissionGate(gate_config)
        for tool in ["execute_command", "run_command", "shell_exec"]:
            result = gate.check(tool)
            assert result.category == "shell_exec", f"{tool} should be shell_exec"

    def test_known_destructive_tools(self, gate_config):
        gate = PermissionGate(gate_config)
        for tool in ["delete_file", "rm", "rmdir"]:
            result = gate.check(tool)
            assert result.category == "destructive", f"{tool} should be destructive"

    def test_heuristic_categorization(self, gate_config):
        gate = PermissionGate(gate_config)
        assert gate.check("get_config").category == "file_read"
        assert gate.check("update_settings").category == "file_write"
        assert gate.check("remove_cache").category == "destructive"

    def test_gate_summary(self, gate_config):
        gate = PermissionGate(gate_config)
        summary = gate.get_gate_summary()
        assert summary["active_profile"] == "focused_work"
        assert summary["profile_count"] == 3
