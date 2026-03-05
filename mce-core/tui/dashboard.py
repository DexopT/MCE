"""
MCE — Observability TUI Dashboard
Rich-powered live terminal display showing real-time interception logs,
token savings, cache statistics, cost tracking, and memory status.
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
    - Cost Watch panel (session/daily cost + budget)
    - Memory panel (memory count, project)
    - Cache hit/miss ratio
    - Recent tool call log
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

        return Panel(table, title="[bold cyan]📊 SQUEEZE ENGINE[/bold cyan]", border_style="cyan")

    def _build_cost_panel(self) -> Panel:
        """Build the Cost Watch panel."""
        cost = self._context.cost_summary
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold green")
        table.add_column("Value", style="bold white")

        session_cost = cost.get("session_cost_usd", 0.0)
        tokens_in = cost.get("session_tokens_in", 0)
        tokens_out = cost.get("session_tokens_out", 0)
        tokens_saved = cost.get("session_tokens_saved", 0)
        events = cost.get("event_count", 0)

        table.add_row("Session Cost", f"[green]${session_cost:.2f}[/green]")
        table.add_row("Tokens In", f"{tokens_in:,}")
        table.add_row("Tokens Out", f"{tokens_out:,}")
        table.add_row("Tokens Saved", f"[green]{tokens_saved:,}[/green]")
        table.add_row("Exchanges", f"{events:,}")

        return Panel(table, title="[bold green]💰 COST WATCH[/bold green]", border_style="green")

    def _build_memory_panel(self) -> Panel:
        """Build the Memory panel."""
        memory = self._context.memory_summary
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold magenta")
        table.add_column("Value", style="bold white")

        mem_count = memory.get("memory_count", 0)
        project_id = memory.get("project_id", "—")

        table.add_row("Memories", f"[magenta]{mem_count}[/magenta]")
        table.add_row("Project", f"{project_id}")

        return Panel(table, title="[bold magenta]🧠 MEMORY[/bold magenta]", border_style="magenta")

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

        return Panel(table, title="[bold yellow]📋 LIVE TOOL CALLS[/bold yellow]", border_style="yellow")

    def _build_timeline_panel(self) -> Panel:
        """Build the Timeline panel."""
        timeline = self._context.timeline_summary
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold blue")
        table.add_column("Value", style="bold white")

        checkpoints = timeline.get("checkpoints", 0)
        branch = timeline.get("current_branch", "main")
        pending = timeline.get("tool_calls_since_cp", 0)
        tokens = timeline.get("cumulative_tokens", 0)

        table.add_row("Checkpoints", f"[blue]{checkpoints}[/blue]")
        table.add_row("Branch", f"{branch}")
        table.add_row("Pending Calls", f"{pending}")
        table.add_row("Total Tokens", f"{tokens:,}")

        return Panel(table, title="[bold blue]⏰ TIMELINE[/bold blue]", border_style="blue")

    def _build_guardian_panel(self) -> Panel:
        """Build the Guardian panel."""
        guardian = self._context.guardian_summary
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold red")
        table.add_column("Value", style="bold white")

        constraints = guardian.get("constraints", 0)
        violations = guardian.get("violations", 0)

        table.add_row("Constraints", f"{constraints}")
        table.add_row(
            "Violations",
            f"[red]{violations}[/red]" if violations > 0 else f"[green]{violations}[/green]",
        )

        return Panel(table, title="[bold red]🛡️ GUARDIAN[/bold red]", border_style="red")

    def render(self) -> Layout:
        """Build the full dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(
                Panel(
                    Text("MCE — Model Context Engine v1.0", style="bold white", justify="center"),
                    border_style="blue",
                ),
                name="header",
                size=3,
            ),
            Layout(name="top_row", size=14),
            Layout(name="mid_row", size=10),
            Layout(name="bottom_row", ratio=1),
        )

        # Top row: Stats + Cost
        layout["top_row"].split_row(
            Layout(self._build_stats_panel(), name="stats", ratio=2),
            Layout(self._build_cost_panel(), name="cost", ratio=1),
        )

        # Middle row: Memory + Timeline + Guardian
        layout["mid_row"].split_row(
            Layout(self._build_memory_panel(), name="memory", ratio=1),
            Layout(self._build_timeline_panel(), name="timeline", ratio=1),
            Layout(self._build_guardian_panel(), name="guardian", ratio=1),
        )

        # Bottom row: Recent tool calls (full width)
        layout["bottom_row"].split_row(
            Layout(self._build_recent_panel(), name="recent", ratio=1),
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
            self._console.print(self._build_cost_panel())
            self._console.print(self._build_memory_panel())
            self._console.print(self._build_recent_panel())
        return capture.get()
