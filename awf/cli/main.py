"""
AI Workflow Fabric - Main CLI Application

This module defines the main Typer application and registers subcommands.
"""

from __future__ import annotations

import typer
from rich.console import Console

# Create main app
app = typer.Typer(
    name="awf",
    help="AI Workflow Fabric - Kubernetes for AI Agents",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Rich console for pretty output
console = Console()

# Import and register subcommands
from awf.cli.agents import agents_app
from awf.cli.tasks import tasks_app
from awf.cli.server import server_app

app.add_typer(agents_app, name="agents", help="Manage AI agents")
app.add_typer(tasks_app, name="tasks", help="Manage tasks")
app.add_typer(server_app, name="server", help="API server management")


@app.command()
def version():
    """Show AWF version."""
    try:
        from awf import __version__
    except ImportError:
        __version__ = "0.1.0"
    console.print(f"[bold cyan]AI Workflow Fabric[/bold cyan] v{__version__}")


@app.callback()
def callback():
    """
    AI Workflow Fabric - Kubernetes for AI Agents
    
    A middleware abstraction layer for orchestrating AI agents across
    frameworks like LangGraph, CrewAI, and AutoGen.
    """
    pass


if __name__ == "__main__":
    app()
