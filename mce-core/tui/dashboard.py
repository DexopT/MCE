"""
MCE — Observability TUI Dashboard
Rich-powered live terminal display showing real-time interception logs,
token savings, cache statistics, and active squeeze operations.
"""

from __future__ import annotations

import time
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.context_manager import ContextManager
from utils.logger import console as mce_console


class Dashboard:
    """
    Live terminal dashboard using Rich.

    Displays:
    - Session token savings summary
    - Cache hit/miss ratio
    - Recent tool call log
    - Active component status
    """

    def __init__(self, context: ContextManager):
        self._context = context
        self._console = Console()
        self._live: Optional[Live] = None

    def _build_stats_panel(self) -> Panel:
        """Build the main statistics panel."""
        stats = self._context.stats
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="bold white")

        table.add_row("Total Requests", f"{stats.total_requests:,}")
        table.add_row("Raw Tokens", f"{stats.total_raw_tokens:,}")
        table.add_row("Squeezed Tokens", f"{stats.total_squeezed_tokens:,}")
        table.add_row(
            "Tokens Saved",
            f"[green]{stats.total_tokens_saved:,}[/green] "
            f"([green]{stats.savings_percent:.1f}%[/green])",
        )
        table.add_row("Squeeze Runs", f"{stats.squeeze_invocations:,}")
        table.add_row(
            "Cache Hit Rate",
            f"[yellow]{stats.cache_hit_rate:.1f}%[/yellow] "
            f"({stats.cache_hits}/{stats.cache_hits + stats.cache_misses})",
        )
        table.add_row("Policy Blocks", f"[red]{stats.policy_blocks}[/red]")
        table.add_row("Breaker Trips", f"[red]{stats.breaker_trips}[/red]")
        table.add_row("Uptime", f"{stats.uptime_seconds:.0f}s")

        return Panel(table, title="[bold cyan]📊 MCE Session Stats[/bold cyan]", border_style="cyan")

    def _build_recent_panel(self) -> Panel:
        """Build the recent tool calls panel."""
        recent = self._context.recent_tools[-10:]

        table = Table(show_header=True, box=None, padding=(0, 1))
        table.add_column("Tool", style="white", max_width=25)
        table.add_column("Raw", style="yellow", justify="right")
        table.add_column("Out", style="green", justify="right")
        table.add_column("Saved", style="magenta", justify="right")
        table.add_column("Cache", style="blue", justify="center")

        for entry in reversed(recent):
            cache_badge = "✓" if entry.get("cached") else "✗"
            blocked = entry.get("blocked")
            tool = entry.get("tool", "?")

            if blocked:
                table.add_row(
                    f"[red]{tool}[/red]",
                    "—",
                    "—",
                    "[red]BLOCKED[/red]",
                    "—",
                )
            else:
                table.add_row(
                    tool,
                    f"{entry.get('raw', 0):,}",
                    f"{entry.get('squeezed', 0):,}",
                    f"{entry.get('saved', 0):,}",
                    f"[green]{cache_badge}[/green]" if entry.get("cached") else f"[dim]{cache_badge}[/dim]",
                )

        return Panel(table, title="[bold yellow]📋 Recent Tool Calls[/bold yellow]", border_style="yellow")

    def render(self) -> Layout:
        """Build the full dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(
                Panel(
                    Text("MCE — Model Context Engine", style="bold white", justify="center"),
                    border_style="blue",
                ),
                name="header",
                size=3,
            ),
            Layout(name="body", ratio=1),
        )

        layout["body"].split_row(
            Layout(self._build_stats_panel(), name="stats", ratio=1),
            Layout(self._build_recent_panel(), name="recent", ratio=2),
        )

        return layout

    def start(self, refresh_rate: float = 1.0) -> None:
        """Start the live dashboard (blocking)."""
        with Live(
            self.render(),
            console=self._console,
            refresh_per_second=1 / refresh_rate,
            screen=True,
        ) as live:
            self._live = live
            try:
                while True:
                    live.update(self.render())
                    time.sleep(refresh_rate)
            except KeyboardInterrupt:
                pass

    def snapshot(self) -> str:
        """Return a static snapshot of the dashboard as a string."""
        with self._console.capture() as capture:
            self._console.print(self._build_stats_panel())
            self._console.print(self._build_recent_panel())
        return capture.get()
