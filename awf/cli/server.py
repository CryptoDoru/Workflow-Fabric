"""
AI Workflow Fabric - Server CLI Commands

Commands for managing the AWF API server.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

server_app = typer.Typer(no_args_is_help=True)
console = Console()


@server_app.command("start")
def start_server(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload for development"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes"),
    log_level: str = typer.Option("info", "--log-level", "-l", help="Log level"),
):
    """Start the AWF API server."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed.[/red]")
        console.print("Install with: pip install ai-workflow-fabric[api]")
        raise typer.Exit(1)
    
    console.print(Panel(
        f"[bold green]Starting AWF API Server[/bold green]\n\n"
        f"Host: {host}\n"
        f"Port: {port}\n"
        f"Workers: {workers}\n"
        f"Reload: {reload}\n"
        f"Log Level: {log_level}\n\n"
        f"API Docs: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs",
        title="AI Workflow Fabric",
        expand=False,
    ))
    
    uvicorn.run(
        "awf.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level=log_level,
    )


@server_app.command("health")
def health_check(
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Server URL"),
):
    """Check if the AWF server is healthy."""
    try:
        import httpx
    except ImportError:
        console.print("[red]httpx not installed.[/red]")
        console.print("Install with: pip install httpx")
        raise typer.Exit(1)
    
    try:
        response = httpx.get(f"{url}/health", timeout=5.0)
        
        if response.status_code == 200:
            data = response.json()
            console.print(f"[green]Server is healthy![/green]")
            console.print(f"  Status: {data.get('status', 'unknown')}")
            console.print(f"  Version: {data.get('version', 'unknown')}")
        else:
            console.print(f"[red]Server returned status {response.status_code}[/red]")
            raise typer.Exit(1)
    except httpx.ConnectError:
        console.print(f"[red]Cannot connect to server at {url}[/red]")
        console.print("Is the server running? Start it with: awf server start")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@server_app.command("config")
def show_config():
    """Show server configuration."""
    console.print(Panel(
        "[bold]Default Configuration[/bold]\n\n"
        "Host: 0.0.0.0\n"
        "Port: 8000\n"
        "Workers: 1\n"
        "Registry: In-Memory\n"
        "Database: SQLite (optional)\n\n"
        "[dim]Environment Variables:[/dim]\n"
        "  AWF_HOST - Server host\n"
        "  AWF_PORT - Server port\n"
        "  AWF_DATABASE_URL - Database URL\n"
        "  AWF_LOG_LEVEL - Logging level",
        title="Server Configuration",
        expand=False,
    ))
