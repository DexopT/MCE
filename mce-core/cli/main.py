"""
MCE — CLI Entry Point
Typer-based CLI for MCE operations: checkpoints, memory, cost, profiles, skills.
"""

from __future__ import annotations

import typer

from cli.commands import checkpoint, memory, cost, profile, skills

app = typer.Typer(
    name="mce",
    help="MCE — Model Context Engine CLI",
    no_args_is_help=True,
    add_completion=False,
)

# Register sub-commands
app.add_typer(checkpoint.app, name="checkpoint", help="Manage session checkpoints")
app.add_typer(memory.app, name="memory", help="Manage persistent memory")
app.add_typer(cost.app, name="cost", help="View cost tracking data")
app.add_typer(profile.app, name="profile", help="Manage permission profiles")
app.add_typer(skills.app, name="skills", help="Manage SkillForge skills")


@app.command()
def status():
    """Show MCE system status."""
    typer.echo("MCE — Model Context Engine v1.0")
    typer.echo("Status: Active")


@app.command()
def version():
    """Show MCE version."""
    typer.echo("MCE v1.0.0 — Meridian Intelligence Layer")


def main():
    app()


if __name__ == "__main__":
    main()
