"""
AI Workflow Fabric - Adapter Registry for Orchestration

This module provides the concrete AdapterRegistry implementation that connects
the orchestration engine to framework adapters (LangGraph, CrewAI, AutoGen).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Type

from awf.adapters.base import BaseAdapter
from awf.core.types import (
    AgentManifest,
    Event,
    Task,
    TaskResult,
)


# =============================================================================
# Adapter Registry
# =============================================================================


@dataclass
class AdapterEntry:
    """Entry in the adapter registry."""
    
    adapter: BaseAdapter
    framework: str
    priority: int = 0  # Higher priority adapters are preferred
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrchestrationAdapterRegistry:
    """
    Registry of framework adapters for the orchestration engine.
    
    This registry manages framework adapters and provides lookup functionality
    for the StepExecutor. It supports:
    
    - Registering adapters by framework name
    - Looking up adapters for specific agents
    - Health checking across adapters
    - Dynamic adapter enable/disable
    
    Example usage:
        ```python
        from awf.adapters.langgraph import LangGraphAdapter
        from awf.orchestration.registry import OrchestrationAdapterRegistry
        
        # Create registry
        registry = OrchestrationAdapterRegistry()
        
        # Register adapters
        langgraph_adapter = LangGraphAdapter()
        registry.register_adapter("langgraph", langgraph_adapter)
        
        # Use with orchestrator
        from awf.orchestration import Orchestrator
        orchestrator = Orchestrator(adapter_registry=registry, ...)
        ```
    """
    
    def __init__(self, agent_registry: Optional[Any] = None):
        """
        Initialize the adapter registry.
        
        Args:
            agent_registry: Optional agent registry for looking up agent metadata.
                           Should implement async get(agent_id) -> AgentManifest
        """
        self._adapters: Dict[str, AdapterEntry] = {}
        self._agent_registry = agent_registry
        self._agent_framework_cache: Dict[str, str] = {}
        self._lock = asyncio.Lock()
    
    # =========================================================================
    # Adapter Registration
    # =========================================================================
    
    def register_adapter(
        self,
        framework: str,
        adapter: BaseAdapter,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a framework adapter.
        
        Args:
            framework: The framework name (e.g., "langgraph", "crewai")
            adapter: The adapter instance
            priority: Priority for this adapter (higher = preferred)
            metadata: Optional metadata about the adapter
        
        Raises:
            ValueError: If framework is empty or adapter is None
        """
        if not framework:
            raise ValueError("Framework name cannot be empty")
        if adapter is None:
            raise ValueError("Adapter cannot be None")
        
        self._adapters[framework.lower()] = AdapterEntry(
            adapter=adapter,
            framework=framework.lower(),
            priority=priority,
            enabled=True,
            metadata=metadata or {},
        )
    
    def unregister_adapter(self, framework: str) -> bool:
        """
        Remove a framework adapter.
        
        Args:
            framework: The framework name to remove
        
        Returns:
            True if the adapter was removed, False if not found
        """
        framework_lower = framework.lower()
        if framework_lower in self._adapters:
            del self._adapters[framework_lower]
            return True
        return False
    
    def enable_adapter(self, framework: str) -> bool:
        """
        Enable a disabled adapter.
        
        Args:
            framework: The framework name
        
        Returns:
            True if the adapter was enabled, False if not found
        """
        entry = self._adapters.get(framework.lower())
        if entry:
            entry.enabled = True
            return True
        return False
    
    def disable_adapter(self, framework: str) -> bool:
        """
        Disable an adapter temporarily.
        
        Args:
            framework: The framework name
        
        Returns:
            True if the adapter was disabled, False if not found
        """
        entry = self._adapters.get(framework.lower())
        if entry:
            entry.enabled = False
            return True
        return False
    
    # =========================================================================
    # Adapter Protocol Implementation
    # =========================================================================
    
    def get_adapter(self, framework: str) -> Optional[BaseAdapter]:
        """
        Get an adapter by framework name.
        
        This method is called by StepExecutor to get the appropriate adapter
        for executing a step.
        
        Args:
            framework: The framework name (e.g., "langgraph")
        
        Returns:
            The adapter if found and enabled, None otherwise
        """
        entry = self._adapters.get(framework.lower())
        if entry and entry.enabled:
            return entry.adapter
        return None
    
    def get_agent_framework(self, agent_id: str) -> Optional[str]:
        """
        Get the framework for an agent.
        
        This looks up the agent in the cache or agent registry to determine
        which framework it belongs to.
        
        Args:
            agent_id: The agent ID
        
        Returns:
            The framework name if known, None otherwise
        """
        # Check cache first
        if agent_id in self._agent_framework_cache:
            return self._agent_framework_cache[agent_id]
        
        # Check each adapter for the agent
        for framework, entry in self._adapters.items():
            if entry.enabled:
                manifest = entry.adapter.get_manifest(agent_id)
                if manifest is not None:
                    self._agent_framework_cache[agent_id] = framework
                    return framework
        
        return None
    
    async def get_agent_framework_async(self, agent_id: str) -> Optional[str]:
        """
        Get the framework for an agent (async version).
        
        Uses the agent registry if available for lookup.
        
        Args:
            agent_id: The agent ID
        
        Returns:
            The framework name if known, None otherwise
        """
        # Check cache first
        if agent_id in self._agent_framework_cache:
            return self._agent_framework_cache[agent_id]
        
        # Try agent registry if available
        if self._agent_registry is not None:
            try:
                manifest = await self._agent_registry.get(agent_id)
                if manifest is not None:
                    framework = getattr(manifest, 'framework', None)
                    if framework:
                        self._agent_framework_cache[agent_id] = framework
                        return framework
            except Exception:
                pass
        
        # Fall back to sync lookup
        return self.get_agent_framework(agent_id)
    
    def cache_agent_framework(self, agent_id: str, framework: str) -> None:
        """
        Cache the framework for an agent.
        
        This can be used to pre-populate the cache for known agents.
        
        Args:
            agent_id: The agent ID
            framework: The framework name
        """
        self._agent_framework_cache[agent_id] = framework.lower()
    
    def clear_cache(self) -> None:
        """Clear the agent-framework cache."""
        self._agent_framework_cache.clear()
    
    # =========================================================================
    # Introspection
    # =========================================================================
    
    def list_adapters(self) -> List[Dict[str, Any]]:
        """
        List all registered adapters.
        
        Returns:
            List of adapter information dicts
        """
        result = []
        for framework, entry in self._adapters.items():
            result.append({
                "framework": framework,
                "enabled": entry.enabled,
                "priority": entry.priority,
                "framework_version": entry.adapter.framework_version,
                "metadata": entry.metadata,
            })
        return sorted(result, key=lambda x: (-x["priority"], x["framework"]))
    
    def list_frameworks(self) -> List[str]:
        """
        List all registered framework names.
        
        Returns:
            List of framework names
        """
        return list(self._adapters.keys())
    
    def list_enabled_frameworks(self) -> List[str]:
        """
        List all enabled framework names.
        
        Returns:
            List of enabled framework names
        """
        return [
            framework
            for framework, entry in self._adapters.items()
            if entry.enabled
        ]
    
    def get_adapter_info(self, framework: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an adapter.
        
        Args:
            framework: The framework name
        
        Returns:
            Adapter info dict if found, None otherwise
        """
        entry = self._adapters.get(framework.lower())
        if entry:
            return {
                "framework": entry.framework,
                "enabled": entry.enabled,
                "priority": entry.priority,
                "framework_version": entry.adapter.framework_version,
                "metadata": entry.metadata,
                "health": entry.adapter.health_check(),
            }
        return None
    
    # =========================================================================
    # Health and Diagnostics
    # =========================================================================
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health of all registered adapters.
        
        Returns:
            Health status dict with overall status and per-adapter status
        """
        adapters_health = {}
        all_healthy = True
        
        for framework, entry in self._adapters.items():
            if entry.enabled:
                try:
                    health = entry.adapter.health_check()
                    adapters_health[framework] = health
                    if not health.get("healthy", False):
                        all_healthy = False
                except Exception as e:
                    adapters_health[framework] = {
                        "healthy": False,
                        "error": str(e),
                    }
                    all_healthy = False
            else:
                adapters_health[framework] = {
                    "healthy": True,
                    "enabled": False,
                    "note": "Adapter is disabled",
                }
        
        return {
            "healthy": all_healthy,
            "total_adapters": len(self._adapters),
            "enabled_adapters": len(self.list_enabled_frameworks()),
            "cached_agents": len(self._agent_framework_cache),
            "adapters": adapters_health,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Statistics dict
        """
        total_agents = 0
        agents_by_framework = {}
        
        for framework, entry in self._adapters.items():
            if entry.enabled:
                agents = entry.adapter.list_agents()
                count = len(agents)
                agents_by_framework[framework] = count
                total_agents += count
        
        return {
            "total_adapters": len(self._adapters),
            "enabled_adapters": len(self.list_enabled_frameworks()),
            "total_agents": total_agents,
            "agents_by_framework": agents_by_framework,
            "cached_lookups": len(self._agent_framework_cache),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_adapter_registry(
    adapters: Optional[Dict[str, BaseAdapter]] = None,
    agent_registry: Optional[Any] = None,
) -> OrchestrationAdapterRegistry:
    """
    Create an adapter registry with the given adapters.
    
    Args:
        adapters: Dict mapping framework names to adapter instances
        agent_registry: Optional agent registry for lookups
    
    Returns:
        Configured OrchestrationAdapterRegistry
    
    Example:
        ```python
        from awf.adapters.langgraph import LangGraphAdapter
        from awf.adapters.crewai import CrewAIAdapter
        
        registry = create_adapter_registry({
            "langgraph": LangGraphAdapter(),
            "crewai": CrewAIAdapter(),
        })
        ```
    """
    registry = OrchestrationAdapterRegistry(agent_registry=agent_registry)
    
    if adapters:
        for framework, adapter in adapters.items():
            registry.register_adapter(framework, adapter)
    
    return registry


async def auto_discover_adapters(
    registry: OrchestrationAdapterRegistry,
) -> List[str]:
    """
    Auto-discover and register available framework adapters.
    
    This function attempts to import and register adapters for all
    supported frameworks.
    
    Args:
        registry: The adapter registry to populate
    
    Returns:
        List of successfully registered framework names
    """
    registered = []
    
    # Try LangGraph
    try:
        from awf.adapters.langgraph import LangGraphAdapter
        adapter = LangGraphAdapter()
        registry.register_adapter("langgraph", adapter)
        registered.append("langgraph")
    except ImportError:
        pass
    except Exception:
        pass
    
    # Try CrewAI
    try:
        from awf.adapters.crewai import CrewAIAdapter
        adapter = CrewAIAdapter()
        registry.register_adapter("crewai", adapter)
        registered.append("crewai")
    except ImportError:
        pass
    except Exception:
        pass
    
    # Try AutoGen
    try:
        from awf.adapters.autogen import AutoGenAdapter
        adapter = AutoGenAdapter()
        registry.register_adapter("autogen", adapter)
        registered.append("autogen")
    except ImportError:
        pass
    except Exception:
        pass
    
    return registered
