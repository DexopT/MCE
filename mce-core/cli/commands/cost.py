"""
MCE CLI — Cost Commands
View cost tracking data: session, daily, summary.
"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from utils.project import get_project_id, get_storage_paths

app = typer.Typer()
console = Console()


def _get_cost_store():
    """Initialize CostStore for CLI usage."""
    from models.cost_store import CostStore

    project_id = get_project_id(str(Path.cwd()))
    paths = get_storage_paths(project_id)
    store = CostStore(str(paths["cost_db"]))
    return store


@app.command()
def session(
    session_id: str = typer.Argument(None, help="Session ID (uses latest if omitted)"),
):
    """Show cost summary for a session."""
    async def _run():
        store = _get_cost_store()
        await store.connect()

        sid = session_id or "latest"
        summary = await store.get_session_cost(sid)

        table = Table(title=f"Session Cost: {sid[:16]}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Cost", f"[green]${summary.total_cost_usd:.4f}[/green]")
        table.add_row("Tokens In", f"{summary.total_tokens_in:,}")
        table.add_row("Tokens Out", f"{summary.total_tokens_out:,}")
        table.add_row("Tokens Saved", f"[green]{summary.total_tokens_saved:,}[/green]")
        table.add_row("Exchanges", f"{summary.event_count}")

        console.print(table)
        await store.close()

    asyncio.run(_run())


@app.command()
def daily(
    day: str = typer.Argument(None, help="Date (YYYY-MM-DD), defaults to today"),
):
    """Show cost summary for a day."""
    async def _run():
        store = _get_cost_store()
        await store.connect()

        target_day = day or str(date.today())
        summary = await store.get_daily_cost(target_day)

        table = Table(title=f"Daily Cost: {target_day}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Cost", f"[green]${summary.total_cost_usd:.4f}[/green]")
        table.add_row("Tokens In", f"{summary.total_tokens_in:,}")
        table.add_row("Tokens Out", f"{summary.total_tokens_out:,}")
        table.add_row("Exchanges", f"{summary.event_count}")

        console.print(table)
        await store.close()

    asyncio.run(_run())
