"""
MCE CLI — Memory Commands
Manage persistent memory: list, search, clear.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from utils.project import get_project_id, get_storage_paths

app = typer.Typer()
console = Console()


def _get_memory_store():
    """Initialize MemoryStore for CLI usage."""
    from models.memory_store import MemoryStore

    project_id = get_project_id(str(Path.cwd()))
    paths = get_storage_paths(project_id)
    store = MemoryStore(str(paths["memory_db"]))
    return store, project_id


@app.command("list")
def list_memories(
    memory_type: str = typer.Option(None, "--type", help="Filter: decision, dead_end, constraint"),
    limit: int = typer.Option(20, help="Max memories to show"),
):
    """List stored memories for the current project."""
    async def _run():
        store, project_id = _get_memory_store()
        await store.connect()
        memories = await store.get_memories(project_id, memory_type=memory_type, limit=limit)

        if not memories:
            console.print("[dim]No memories found.[/dim]")
            await store.close()
            return

        table = Table(title=f"Memories (project: {project_id[:8]})")
        table.add_column("Type", style="cyan")
        table.add_column("Content", style="white", max_width=60)
        table.add_column("Confidence", justify="right")
        table.add_column("Created", style="dim")

        for mem in memories:
            table.add_row(
                mem.type,
                mem.content[:60],
                f"{mem.confidence:.1f}" if mem.confidence else "—",
                mem.created_at[:10],
            )
        console.print(table)
        await store.close()

    asyncio.run(_run())


@app.command()
def count():
    """Show memory count for the current project."""
    async def _run():
        store, project_id = _get_memory_store()
        await store.connect()
        total = await store.count_memories(project_id)
        console.print(f"Memories: [bold]{total}[/bold] (project: {project_id[:8]})")
        await store.close()

    asyncio.run(_run())


@app.command()
def clear(
    confirm: bool = typer.Option(False, "--yes", help="Skip confirmation"),
):
    """Clear all memories for the current project."""
    async def _run():
        store, project_id = _get_memory_store()
        await store.connect()

        if not confirm:
            count = await store.count_memories(project_id)
            if not typer.confirm(f"Delete {count} memories for project {project_id[:8]}?"):
                console.print("[dim]Cancelled.[/dim]")
                await store.close()
                return

        deleted = await store.delete_memories(project_id)
        console.print(f"[green]✓[/green] Deleted {deleted} memories")
        await store.close()

    asyncio.run(_run())
