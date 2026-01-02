"""
AI Workflow Fabric - Agent CLI Commands

Commands for managing AI agents: register, list, get, delete.
"""

from __future__ import annotations

import json
import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON

from awf.core.types import AgentManifest, AgentStatus

agents_app = typer.Typer(no_args_is_help=True)
console = Console()


def get_registry():
    """Get the registry instance."""
    from awf.registry.memory import InMemoryRegistry
    # In a real app, this would be configured from settings
    return InMemoryRegistry()


@agents_app.command("list")
def list_agents(
    framework: Optional[str] = typer.Option(None, "--framework", "-f", help="Filter by framework"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    capabilities: Optional[str] = typer.Option(None, "--capabilities", "-c", help="Filter by capabilities (comma-separated)"),
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json"),
):
    """List all registered agents."""
    registry = get_registry()
    
    async def _list():
        caps = capabilities.split(",") if capabilities else None
        status_enum = AgentStatus(status) if status else None
        
        agents = await registry.search(
            framework=framework,
            status=status_enum,
            capabilities=caps,
        )
        return agents
    
    agents = asyncio.run(_list())
    
    if output == "json":
        data = [
            {
                "id": a.id,
                "name": a.name,
                "version": a.version,
                "framework": a.framework,
                "status": a.status.value,
            }
            for a in agents
        ]
        console.print(JSON(json.dumps(data, indent=2)))
        return
    
    if not agents:
        console.print("[yellow]No agents found.[/yellow]")
        return
    
    table = Table(title="Registered Agents")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Version")
    table.add_column("Framework", style="magenta")
    table.add_column("Status")
    table.add_column("Capabilities")
    
    for agent in agents:
        caps = ", ".join(c.name for c in (agent.capabilities or [])[:3])
        if agent.capabilities and len(agent.capabilities) > 3:
            caps += f" (+{len(agent.capabilities) - 3})"
        
        status_style = {
            AgentStatus.ACTIVE: "[green]active[/green]",
            AgentStatus.INACTIVE: "[yellow]inactive[/yellow]",
            AgentStatus.DEPRECATED: "[red]deprecated[/red]",
        }.get(agent.status, agent.status.value)
        
        table.add_row(
            agent.id,
            agent.name,
            agent.version,
            agent.framework,
            status_style,
            caps,
        )
    
    console.print(table)


@agents_app.command("get")
def get_agent(
    agent_id: str = typer.Argument(..., help="Agent ID"),
    output: str = typer.Option("panel", "--output", "-o", help="Output format: panel, json"),
):
    """Get details of a specific agent."""
    registry = get_registry()
    
    async def _get():
        return await registry.get(agent_id)
    
    agent = asyncio.run(_get())
    
    if not agent:
        console.print(f"[red]Agent '{agent_id}' not found.[/red]")
        raise typer.Exit(1)
    
    if output == "json":
        # Convert to dict for JSON output
        data = {
            "id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "version": agent.version,
            "framework": agent.framework,
            "status": agent.status.value,
            "capabilities": [
                {"name": c.name, "type": c.type.value, "description": c.description}
                for c in (agent.capabilities or [])
            ],
            "metadata": agent.metadata,
        }
        console.print(JSON(json.dumps(data, indent=2)))
        return
    
    # Rich panel output
    info = f"""[bold]ID:[/bold] {agent.id}
[bold]Name:[/bold] {agent.name}
[bold]Version:[/bold] {agent.version}
[bold]Framework:[/bold] {agent.framework}
[bold]Status:[/bold] {agent.status.value}
[bold]Description:[/bold] {agent.description or 'N/A'}
"""
    
    if agent.capabilities:
        info += "\n[bold]Capabilities:[/bold]\n"
        for cap in agent.capabilities:
            info += f"  - {cap.name} ({cap.type.value}): {cap.description or 'No description'}\n"
    
    console.print(Panel(info, title=f"Agent: {agent.name}", expand=False))


@agents_app.command("register")
def register_agent(
    manifest_file: Path = typer.Argument(..., help="Path to agent manifest JSON file"),
):
    """Register an agent from a manifest file."""
    if not manifest_file.exists():
        console.print(f"[red]File not found: {manifest_file}[/red]")
        raise typer.Exit(1)
    
    try:
        with open(manifest_file) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON: {e}[/red]")
        raise typer.Exit(1)
    
    try:
        # Convert dict to AgentManifest
        from awf.core.types import Capability, CapabilityType
        
        capabilities = []
        for cap in data.get("capabilities", []):
            capabilities.append(Capability(
                name=cap["name"],
                type=CapabilityType(cap.get("type", "tool")),
                description=cap.get("description"),
            ))
        
        agent = AgentManifest(
            id=data["id"],
            name=data["name"],
            version=data.get("version", "1.0.0"),
            framework=data.get("framework", "custom"),
            description=data.get("description"),
            capabilities=capabilities,
            status=AgentStatus(data.get("status", "active")),
            metadata=data.get("metadata"),
        )
    except (KeyError, ValueError) as e:
        console.print(f"[red]Invalid manifest: {e}[/red]")
        raise typer.Exit(1)
    
    registry = get_registry()
    
    async def _register():
        await registry.register(agent)
    
    asyncio.run(_register())
    
    console.print(f"[green]Successfully registered agent:[/green] {agent.id}")


@agents_app.command("delete")
def delete_agent(
    agent_id: str = typer.Argument(..., help="Agent ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete an agent from the registry."""
    if not force:
        confirm = typer.confirm(f"Are you sure you want to delete agent '{agent_id}'?")
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)
    
    registry = get_registry()
    
    async def _delete():
        # First check if agent exists
        agent = await registry.get(agent_id)
        if not agent:
            return False
        await registry.unregister(agent_id)
        return True
    
    deleted = asyncio.run(_delete())
    
    if deleted:
        console.print(f"[green]Deleted agent:[/green] {agent_id}")
    else:
        console.print(f"[red]Agent '{agent_id}' not found.[/red]")
        raise typer.Exit(1)


@agents_app.command("trust")
def get_trust_score(
    agent_id: str = typer.Argument(..., help="Agent ID"),
):
    """Get trust score for an agent."""
    registry = get_registry()
    
    async def _get_trust():
        from awf.security.trust import TrustScoringEngine
        
        agent = await registry.get(agent_id)
        if not agent:
            return None, None
        
        engine = TrustScoringEngine()
        score = await engine.compute_score(agent)
        return agent, score
    
    agent, score = asyncio.run(_get_trust())
    
    if not agent:
        console.print(f"[red]Agent '{agent_id}' not found.[/red]")
        raise typer.Exit(1)
    
    # Color based on score
    if score.score >= 0.9:
        score_color = "green"
    elif score.score >= 0.7:
        score_color = "yellow"
    elif score.score >= 0.4:
        score_color = "orange3"
    else:
        score_color = "red"
    
    info = f"""[bold]Agent:[/bold] {agent.name}
[bold]Trust Score:[/bold] [{score_color}]{score.score:.2f}[/{score_color}]
[bold]Sandbox Tier:[/bold] {score.sandbox_tier.value}

[bold]Factor Breakdown:[/bold]
  Publisher Trust:    {score.factors.get('publisher_trust', 0):.2f}
  Audit Status:       {score.factors.get('audit_status', 0):.2f}
  Community Trust:    {score.factors.get('community_trust', 0):.2f}
  Permission Risk:    {score.factors.get('permission_risk', 0):.2f}
  Historical Behavior:{score.factors.get('historical_behavior', 0):.2f}
"""
    
    console.print(Panel(info, title="Trust Assessment", expand=False))
