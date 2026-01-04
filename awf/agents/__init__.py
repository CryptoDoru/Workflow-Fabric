"""
AI Workflow Fabric - Agents Package

This package contains built-in AWF agents including the Watcher Agent
for autonomous observability and remediation.
"""

from awf.agents.watcher import WatcherAgent, WatcherConfig

__all__ = ["WatcherAgent", "WatcherConfig"]
