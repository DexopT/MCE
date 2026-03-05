"""
MCE — TimeMachine Tests
Tests for session checkpointing, branching, and restoration.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.intelligence.time_machine import TimeMachine, Checkpoint
from schemas.mce_config import TimeMachineConfig


@pytest.fixture
def temp_db(tmp_path):
    return str(tmp_path / "test_timeline.db")


@pytest.fixture
def tm_config():
    return TimeMachineConfig(
        enabled=True,
        auto_checkpoint_interval_mins=0,  # disabled for tests
        checkpoint_on_file_write=True,
        checkpoint_on_destructive_tool=True,
        max_checkpoints_per_session=50,
        capture_file_diffs=False,
    )


@pytest.mark.asyncio
class TestTimeMachineCheckpoints:
    async def test_create_checkpoint(self, temp_db, tm_config):
        tm = TimeMachine(tm_config, "sess-1", temp_db)
        await tm.connect()

        cp = await tm.checkpoint(label="Test checkpoint")
        assert cp.sequence == 1
        assert cp.label == "Test checkpoint"
        assert cp.session_id == "sess-1"
        assert cp.branch == "main"

        await tm.close()

    async def test_multiple_checkpoints_increment_sequence(self, temp_db, tm_config):
        tm = TimeMachine(tm_config, "sess-1", temp_db)
        await tm.connect()

        cp1 = await tm.checkpoint(label="First")
        cp2 = await tm.checkpoint(label="Second")
        cp3 = await tm.checkpoint(label="Third")

        assert cp1.sequence == 1
        assert cp2.sequence == 2
        assert cp3.sequence == 3

        await tm.close()

    async def test_list_checkpoints(self, temp_db, tm_config):
        tm = TimeMachine(tm_config, "sess-1", temp_db)
        await tm.connect()

        for i in range(5):
            await tm.checkpoint(label=f"CP {i+1}")

        cps = await tm.list_checkpoints()
        assert len(cps) == 5
        # Listed in DESC order
        assert cps[0].sequence == 5
        assert cps[4].sequence == 1

        await tm.close()

    async def test_checkpoint_captures_tool_calls(self, temp_db, tm_config):
        tm = TimeMachine(tm_config, "sess-1", temp_db)
        await tm.connect()

        tm.record_tool_call("read_file", {"path": "src/main.py"}, 100)
        tm.record_tool_call("write_file", {"path": "src/main.py"}, 200)

        cp = await tm.checkpoint(label="After edits")
        assert cp.tool_call_count == 2
        assert cp.token_count == 300

        await tm.close()


@pytest.mark.asyncio
class TestTimeMachineBranching:
    async def test_create_branch(self, temp_db, tm_config):
        tm = TimeMachine(tm_config, "sess-1", temp_db)
        await tm.connect()

        await tm.checkpoint(label="Base")
        branch = await tm.branch("experiment")

        assert branch.name == "experiment"
        assert tm.current_branch == "experiment"

        await tm.close()

    async def test_branch_from_specific_checkpoint(self, temp_db, tm_config):
        tm = TimeMachine(tm_config, "sess-1", temp_db)
        await tm.connect()

        cp1 = await tm.checkpoint(label="Version 1")
        cp2 = await tm.checkpoint(label="Version 2")

        branch = await tm.branch("from-v1", from_checkpoint_id=cp1.id)
        assert branch.parent_checkpoint_id == cp1.id

        await tm.close()

    async def test_list_branches(self, temp_db, tm_config):
        tm = TimeMachine(tm_config, "sess-1", temp_db)
        await tm.connect()

        await tm.checkpoint(label="Base")
        await tm.branch("branch-a")
        await tm.branch("branch-b")

        branches = await tm.list_branches()
        names = {b.name for b in branches}
        assert "branch-a" in names
        assert "branch-b" in names

        await tm.close()


@pytest.mark.asyncio
class TestTimeMachineRestore:
    async def test_restore_checkpoint(self, temp_db, tm_config):
        tm = TimeMachine(tm_config, "sess-1", temp_db)
        await tm.connect()

        cp1 = await tm.checkpoint(label="Version 1")
        tm.record_tool_call("delete_file", {"path": "important.py"}, 50)
        await tm.checkpoint(label="Version 2 (mistake)")

        restored = await tm.restore(cp1.id)
        assert restored is not None
        assert restored.sequence == 1
        assert restored.label == "Version 1"
        assert tm.checkpoint_count == 1  # Sequence reset to checkpoint

        await tm.close()

    async def test_restore_nonexistent_returns_none(self, temp_db, tm_config):
        tm = TimeMachine(tm_config, "sess-1", temp_db)
        await tm.connect()

        result = await tm.restore("nonexistent-id")
        assert result is None

        await tm.close()


@pytest.mark.asyncio
class TestTimeMachineAutoTrigger:
    async def test_auto_trigger_on_file_write(self, temp_db, tm_config):
        tm = TimeMachine(tm_config, "sess-1", temp_db)
        await tm.connect()

        tm.record_tool_call("write_file", {"path": "foo.py"}, 100, is_file_write=True)
        cp = await tm.maybe_checkpoint("write_file", is_file_write=True)
        assert cp is not None
        assert "file_write" in cp.label

        await tm.close()

    async def test_auto_trigger_on_destructive_tool(self, temp_db, tm_config):
        tm = TimeMachine(tm_config, "sess-1", temp_db)
        await tm.connect()

        tm.record_tool_call("delete_file", {"path": "foo.py"}, 50, is_destructive=True)
        cp = await tm.maybe_checkpoint("delete_file", is_destructive=True)
        assert cp is not None
        assert "destructive" in cp.label

        await tm.close()

    async def test_no_trigger_on_read(self, temp_db, tm_config):
        tm = TimeMachine(tm_config, "sess-1", temp_db)
        await tm.connect()

        tm.record_tool_call("read_file", {"path": "foo.py"}, 100)
        cp = await tm.maybe_checkpoint("read_file")
        assert cp is None

        await tm.close()

    async def test_max_checkpoints_limit(self, temp_db, tm_config):
        tm_config.max_checkpoints_per_session = 3
        tm = TimeMachine(tm_config, "sess-1", temp_db)
        await tm.connect()

        for i in range(3):
            await tm.checkpoint(label=f"CP {i}")

        # 4th should not trigger (at max)
        tm.record_tool_call("write_file", {"path": "foo.py"}, 100, is_file_write=True)
        cp = await tm.maybe_checkpoint("write_file", is_file_write=True)
        assert cp is None

        await tm.close()

    async def test_timeline_summary(self, temp_db, tm_config):
        tm = TimeMachine(tm_config, "sess-1", temp_db)
        await tm.connect()

        tm.record_tool_call("read_file", {"path": "foo.py"}, 100)
        await tm.checkpoint(label="First")

        summary = tm.get_timeline_summary()
        assert summary["checkpoints"] == 1
        assert summary["current_branch"] == "main"

        await tm.close()
