"""
AI Workflow Fabric - Command Line Interface

This package provides the CLI for managing AWF agents, tasks, and servers.

Usage:
    awf --help              # Show help
    awf agents list         # List registered agents
    awf agents register     # Register an agent from manifest
    awf tasks submit        # Submit a task
    awf server start        # Start the API server
"""

from awf.cli.main import app

__all__ = ["app"]
