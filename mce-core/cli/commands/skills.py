"""
MCE CLI — Skills Commands
Manage SkillForge skills: list, info.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from schemas.mce_config import SkillsConfig

app = typer.Typer()
console = Console()


@app.command("list")
def list_skills(
    path: str = typer.Option(".mce/skills", help="Skills directory path"),
):
    """List all loaded skills."""
    from engine.skills.skill_loader import SkillLoader

    config = SkillsConfig(enabled=True, path=path, auto_trigger=True)
    loader = SkillLoader(config)
    count = loader.load_skills(Path.cwd())

    if count == 0:
        console.print("[dim]No skills found.[/dim]")
        return

    table = Table(title="SkillForge Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="dim")
    table.add_column("Risk", justify="center")
    table.add_column("Triggers", style="white")
    table.add_column("Checkpoint", justify="center")

    for skill in loader.list_skills():
        triggers = ", ".join(skill.triggers.tool_names[:3]) or ", ".join(skill.triggers.keywords[:3])
        risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(skill.risk_level, "white")
        table.add_row(
            skill.name,
            skill.version,
            f"[{risk_color}]{skill.risk_level}[/{risk_color}]",
            triggers[:40],
            "✓" if skill.requires_checkpoint else "—",
        )
    console.print(table)


@app.command()
def info(
    name: str = typer.Argument(..., help="Skill name"),
    path: str = typer.Option(".mce/skills", help="Skills directory path"),
):
    """Show details of a specific skill."""
    from engine.skills.skill_loader import SkillLoader

    config = SkillsConfig(enabled=True, path=path, auto_trigger=True)
    loader = SkillLoader(config)
    loader.load_skills(Path.cwd())

    skill = loader.get_skill(name)
    if skill is None:
        console.print(f"[red]✗[/red] Skill '{name}' not found")
        return

    console.print(f"\n[bold cyan]{skill.name}[/bold cyan] v{skill.version}")
    console.print(f"Risk: [{{'low': 'green', 'medium': 'yellow', 'high': 'red'}[skill.risk_level]}]{skill.risk_level}[/{{'low': 'green', 'medium': 'yellow', 'high': 'red'}[skill.risk_level]}]")

    if skill.constraints:
        console.print("\n[bold]Constraints:[/bold]")
        for c in skill.constraints:
            console.print(f"  • {c}")

    if skill.workflow:
        console.print("\n[bold]Workflow:[/bold]")
        for i, w in enumerate(skill.workflow, 1):
            console.print(f"  {i}. {w}")
