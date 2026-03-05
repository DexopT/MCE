"""
MCE CLI — Profile Commands
Manage permission profiles: list, switch, show.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from schemas.mce_config import PermissionProfilesConfig, PermissionProfile

app = typer.Typer()
console = Console()


def _load_config() -> PermissionProfilesConfig:
    """Load permission profiles from config."""
    try:
        import yaml
        from pathlib import Path

        config_path = Path.cwd() / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f)
            pp_data = data.get("permission_profiles", {})
            profiles = {}
            for name, settings in pp_data.get("profiles", {}).items():
                profiles[name] = PermissionProfile(**settings)
            return PermissionProfilesConfig(
                active=pp_data.get("active", "focused_work"),
                profiles=profiles,
            )
    except Exception:
        pass
    return PermissionProfilesConfig()


@app.command("list")
def list_profiles():
    """List all permission profiles."""
    config = _load_config()

    from engine.guardian.permission_gate import PermissionGate
    gate = PermissionGate(config)

    table = Table(title="Permission Profiles")
    table.add_column("Profile", style="cyan")
    table.add_column("File Read", justify="center")
    table.add_column("File Write", justify="center")
    table.add_column("Shell Exec", justify="center")
    table.add_column("Destructive", justify="center")
    table.add_column("Active", justify="center")

    for name, info in gate.list_profiles().items():
        active_badge = "[green]●[/green]" if info["active"] else "[dim]○[/dim]"
        table.add_row(
            name,
            _format_perm(info["file_read"]),
            _format_perm(info["file_write"]),
            _format_perm(info["shell_exec"]),
            _format_perm(info["destructive"]),
            active_badge,
        )
    console.print(table)


@app.command()
def switch(
    profile_name: str = typer.Argument(..., help="Profile to switch to"),
):
    """Switch to a different permission profile."""
    config = _load_config()

    from engine.guardian.permission_gate import PermissionGate
    gate = PermissionGate(config)

    if gate.switch_profile(profile_name):
        console.print(f"[green]✓[/green] Switched to '{profile_name}' profile")
    else:
        console.print(f"[red]✗[/red] Profile '{profile_name}' not found")
        console.print("Available profiles:")
        for name in gate.list_profiles():
            console.print(f"  • {name}")


def _format_perm(value: str) -> str:
    """Format a permission value with color."""
    if value == "auto":
        return "[green]auto[/green]"
    elif value == "block":
        return "[red]block[/red]"
    return "[yellow]prompt[/yellow]"
