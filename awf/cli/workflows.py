"""
AI Workflow Fabric - Workflow CLI Commands

Commands for managing workflows: list, run, status, create.
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
from rich.progress import Progress, SpinnerColumn, TextColumn

workflows_app = typer.Typer(no_args_is_help=True)
console = Console()


@workflows_app.command("list")
def list_workflows(
    output: str = typer.Option("table", "--output", "-o", help="Output format: table, json"),
):
    """List all registered workflows."""
    # In a real implementation, this would query the API
    console.print("[yellow]Workflow listing requires API server connection.[/yellow]")
    console.print("Start the server with: awf server start")
    console.print()
    
    # Demo workflows
    demo_workflows = [
        {
            "id": "research-pipeline",
            "name": "Research Pipeline",
            "steps": 3,
            "status": "active",
        },
        {
            "id": "code-review",
            "name": "Code Review Pipeline",
            "steps": 4,
            "status": "active",
        },
        {
            "id": "customer-support",
            "name": "Customer Support Flow",
            "steps": 3,
            "status": "active",
        },
    ]
    
    if output == "json":
        console.print(JSON(json.dumps(demo_workflows, indent=2)))
        return
    
    table = Table(title="Workflows (Demo)")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Steps")
    table.add_column("Status")
    
    for wf in demo_workflows:
        status_style = "[green]active[/green]" if wf["status"] == "active" else "[yellow]inactive[/yellow]"
        table.add_row(wf["id"], wf["name"], str(wf["steps"]), status_style)
    
    console.print(table)


@workflows_app.command("run")
def run_workflow(
    workflow_file: Path = typer.Argument(..., help="Path to workflow YAML or JSON file"),
    input_data: Optional[str] = typer.Option(None, "--input", "-i", help="Input data as JSON string"),
    input_file: Optional[Path] = typer.Option(None, "--input-file", "-f", help="Input data from JSON file"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Validate without executing"),
    wait: bool = typer.Option(True, "--wait/--no-wait", "-w", help="Wait for completion"),
    timeout: int = typer.Option(300, "--timeout", "-t", help="Execution timeout in seconds"),
):
    """Run a workflow from a file."""
    import yaml
    
    if not workflow_file.exists():
        console.print(f"[red]File not found: {workflow_file}[/red]")
        raise typer.Exit(1)
    
    # Load workflow definition
    try:
        with open(workflow_file) as f:
            if workflow_file.suffix in [".yaml", ".yml"]:
                try:
                    workflow_def = yaml.safe_load(f)
                except ImportError:
                    console.print("[red]PyYAML not installed. Install with: pip install pyyaml[/red]")
                    raise typer.Exit(1)
            else:
                workflow_def = json.load(f)
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        console.print(f"[red]Invalid file format: {e}[/red]")
        raise typer.Exit(1)
    
    # Load input data
    workflow_input = {}
    if input_data:
        try:
            workflow_input = json.loads(input_data)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid input JSON: {e}[/red]")
            raise typer.Exit(1)
    elif input_file:
        if not input_file.exists():
            console.print(f"[red]Input file not found: {input_file}[/red]")
            raise typer.Exit(1)
        with open(input_file) as f:
            workflow_input = json.load(f)
    
    # Display workflow info
    console.print(Panel(
        f"[bold]Workflow:[/bold] {workflow_def.get('name', workflow_def.get('id', 'Unknown'))}\n"
        f"[bold]ID:[/bold] {workflow_def.get('id', 'N/A')}\n"
        f"[bold]Steps:[/bold] {len(workflow_def.get('steps', []))}\n"
        f"[bold]Input:[/bold] {json.dumps(workflow_input)[:100]}...",
        title="Workflow Execution",
        expand=False,
    ))
    
    if dry_run:
        console.print("[yellow]Dry run mode - validating workflow...[/yellow]")
        
        # Validate structure
        errors = []
        if "id" not in workflow_def:
            errors.append("Missing required field: id")
        if "steps" not in workflow_def:
            errors.append("Missing required field: steps")
        elif not isinstance(workflow_def["steps"], list):
            errors.append("Field 'steps' must be a list")
        
        if errors:
            console.print("[red]Validation failed:[/red]")
            for error in errors:
                console.print(f"  - {error}")
            raise typer.Exit(1)
        
        console.print("[green]Workflow validation passed![/green]")
        return
    
    # Execute workflow (simulated)
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Executing workflow...", total=None)
        
        async def simulate_execution():
            steps = workflow_def.get("steps", [])
            for i, step in enumerate(steps):
                step_name = step.get("name", step.get("id", f"Step {i+1}"))
                progress.update(task, description=f"Executing: {step_name}")
                await asyncio.sleep(0.5)  # Simulate execution
            return {"status": "completed", "steps_executed": len(steps)}
        
        result = asyncio.run(simulate_execution())
    
    console.print()
    console.print(Panel(
        f"[bold green]Workflow completed successfully![/bold green]\n\n"
        f"Status: {result['status']}\n"
        f"Steps executed: {result['steps_executed']}",
        title="Result",
        expand=False,
    ))


@workflows_app.command("status")
def workflow_status(
    execution_id: str = typer.Argument(..., help="Workflow execution ID"),
    output: str = typer.Option("panel", "--output", "-o", help="Output format: panel, json"),
):
    """Get status of a workflow execution."""
    console.print("[yellow]Status lookup requires API server connection.[/yellow]")
    console.print(f"Execution ID: {execution_id}")
    
    # Demo status
    demo_status = {
        "execution_id": execution_id,
        "workflow_id": "research-pipeline",
        "status": "completed",
        "started_at": "2024-01-01T10:00:00Z",
        "completed_at": "2024-01-01T10:02:30Z",
        "steps": [
            {"id": "search", "status": "completed", "duration_ms": 1200},
            {"id": "analyze", "status": "completed", "duration_ms": 800},
            {"id": "summarize", "status": "completed", "duration_ms": 500},
        ],
    }
    
    if output == "json":
        console.print(JSON(json.dumps(demo_status, indent=2)))
        return
    
    steps_info = "\n".join(
        f"  - {s['id']}: {s['status']} ({s['duration_ms']}ms)"
        for s in demo_status["steps"]
    )
    
    console.print(Panel(
        f"[bold]Execution ID:[/bold] {demo_status['execution_id']}\n"
        f"[bold]Workflow:[/bold] {demo_status['workflow_id']}\n"
        f"[bold]Status:[/bold] [green]{demo_status['status']}[/green]\n"
        f"[bold]Started:[/bold] {demo_status['started_at']}\n"
        f"[bold]Completed:[/bold] {demo_status['completed_at']}\n\n"
        f"[bold]Steps:[/bold]\n{steps_info}",
        title="Workflow Execution Status",
        expand=False,
    ))


@workflows_app.command("create")
def create_workflow(
    name: str = typer.Option(..., "--name", "-n", help="Workflow name"),
    output_file: Path = typer.Option(Path("workflow.yaml"), "--output", "-o", help="Output file path"),
    template: str = typer.Option("basic", "--template", "-t", help="Template: basic, research, review"),
):
    """Create a new workflow template."""
    import yaml
    
    templates = {
        "basic": {
            "id": name.lower().replace(" ", "-"),
            "name": name,
            "version": "1.0.0",
            "description": f"Workflow: {name}",
            "steps": [
                {
                    "id": "step-1",
                    "name": "First Step",
                    "agent_id": "your-agent-id",
                    "timeout_ms": 30000,
                },
                {
                    "id": "step-2",
                    "name": "Second Step",
                    "agent_id": "another-agent-id",
                    "depends_on": ["step-1"],
                    "timeout_ms": 30000,
                },
            ],
        },
        "research": {
            "id": name.lower().replace(" ", "-"),
            "name": name,
            "version": "1.0.0",
            "description": "Research pipeline workflow",
            "steps": [
                {
                    "id": "search",
                    "name": "Web Search",
                    "agent_id": "web-search-agent",
                    "timeout_ms": 30000,
                    "retry": {"max_attempts": 3, "backoff_multiplier": 2.0},
                },
                {
                    "id": "analyze",
                    "name": "Analyze Results",
                    "agent_id": "analysis-agent",
                    "depends_on": ["search"],
                    "timeout_ms": 60000,
                },
                {
                    "id": "summarize",
                    "name": "Generate Summary",
                    "agent_id": "summarizer-agent",
                    "depends_on": ["analyze"],
                    "timeout_ms": 45000,
                },
            ],
        },
        "review": {
            "id": name.lower().replace(" ", "-"),
            "name": name,
            "version": "1.0.0",
            "description": "Code review workflow",
            "steps": [
                {
                    "id": "parse",
                    "name": "Parse Code",
                    "agent_id": "code-parser-agent",
                    "timeout_ms": 30000,
                },
                {
                    "id": "security",
                    "name": "Security Scan",
                    "agent_id": "security-scanner-agent",
                    "depends_on": ["parse"],
                    "timeout_ms": 60000,
                },
                {
                    "id": "quality",
                    "name": "Quality Review",
                    "agent_id": "quality-reviewer-agent",
                    "depends_on": ["parse"],
                    "timeout_ms": 60000,
                },
                {
                    "id": "report",
                    "name": "Generate Report",
                    "agent_id": "report-generator-agent",
                    "depends_on": ["security", "quality"],
                    "timeout_ms": 30000,
                },
            ],
        },
    }
    
    if template not in templates:
        console.print(f"[red]Unknown template: {template}[/red]")
        console.print(f"Available templates: {', '.join(templates.keys())}")
        raise typer.Exit(1)
    
    workflow_def = templates[template]
    
    try:
        with open(output_file, "w") as f:
            if output_file.suffix in [".yaml", ".yml"]:
                yaml.dump(workflow_def, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(workflow_def, f, indent=2)
    except ImportError:
        # Fallback to JSON if yaml not available
        output_file = output_file.with_suffix(".json")
        with open(output_file, "w") as f:
            json.dump(workflow_def, f, indent=2)
    
    console.print(f"[green]Created workflow template:[/green] {output_file}")
    console.print()
    console.print("Next steps:")
    console.print(f"  1. Edit {output_file} to customize your workflow")
    console.print(f"  2. Run with: awf workflows run {output_file}")
