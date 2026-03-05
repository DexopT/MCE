"""
MCE CLI — Checkpoint Commands
Manage session checkpoints: list, create, restore, branch.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from utils.project import get_project_id, get_session_id, get_storage_paths

app = typer.Typer()
console = Console()


def _get_time_machine():
    """Initialize TimeMachine for CLI usage."""
    from engine.intelligence.time_machine import TimeMachine
    from schemas.mce_config import TimeMachineConfig

    project_id = get_project_id(str(Path.cwd()))
    session_id = get_session_id()
    paths = get_storage_paths(project_id)

    config = TimeMachineConfig(enabled=True, capture_file_diffs=True)
    tm = TimeMachine(config, session_id, paths["session_dir"] / "timeline.db")
    return tm


@app.command("list")
def list_checkpoints(
    branch: str = typer.Option(None, help="Filter by branch name"),
    limit: int = typer.Option(20, help="Max checkpoints to show"),
):
    """List all checkpoints for the current session."""
    async def _run():
        tm = _get_time_machine()
        await tm.connect()
        cps = await tm.list_checkpoints(branch=branch, limit=limit)

        if not cps:
            console.print("[dim]No checkpoints found.[/dim]")
            await tm.close()
            return

        table = Table(title="Checkpoints")
        table.add_column("Seq", style="cyan", justify="right")
        table.add_column("ID", style="dim")
        table.add_column("Label", style="white")
        table.add_column("Branch", style="blue")
        table.add_column("Tools", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Time", style="dim")

        for cp in cps:
            table.add_row(
                str(cp.sequence),
                cp.id[:8],
                cp.label,
                cp.branch,
                str(cp.tool_call_count),
                f"{cp.token_count:,}",
                cp.created_at[:19],
            )
        console.print(table)
        await tm.close()

    asyncio.run(_run())


@app.command("create")
def create_checkpoint(
    label: str = typer.Argument("Manual checkpoint"),
):
    """Create a manual checkpoint."""
    async def _run():
        tm = _get_time_machine()
        await tm.connect()
        cp = await tm.checkpoint(label=label)
        console.print(f"[green]✓[/green] Checkpoint #{cp.sequence}: {cp.label} ({cp.id[:8]})")
        await tm.close()

    asyncio.run(_run())


@app.command()
def restore(
    checkpoint_id: str = typer.Argument(..., help="Checkpoint ID (first 8 chars ok)"),
):
    """Restore to a specific checkpoint."""
    async def _run():
        tm = _get_time_machine()
        await tm.connect()
        cp = await tm.restore(checkpoint_id)
        if cp:
            console.print(f"[green]✓[/green] Restored to #{cp.sequence}: {cp.label}")
        else:
            console.print(f"[red]✗[/red] Checkpoint '{checkpoint_id}' not found")
        await tm.close()

    asyncio.run(_run())


@app.command()
def branch(
    name: str = typer.Argument(..., help="New branch name"),
    from_checkpoint: str = typer.Option(None, "--from", help="Fork from checkpoint ID"),
):
    """Create a new branch from a checkpoint."""
    async def _run():
        tm = _get_time_machine()
        await tm.connect()
        info = await tm.branch(name, from_checkpoint_id=from_checkpoint)
        console.print(
            f"[green]✓[/green] Branch '{info.name}' created from {info.parent_checkpoint_id[:8]}"
        )
        await tm.close()

    asyncio.run(_run())
