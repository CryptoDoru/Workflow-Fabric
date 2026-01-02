"""
AI Workflow Fabric - In-Memory Agent Registry

This module provides an in-memory implementation of the AgentRegistry
interface for development and testing purposes.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Dict, List, Optional

from awf.adapters.base import AgentRegistry
from awf.core.types import AgentManifest, AgentStatus


class InMemoryRegistry(AgentRegistry):
    """
    In-memory implementation of the AgentRegistry.
    
    This registry stores manifests in memory and provides fast lookup
    and search capabilities. Suitable for development, testing, and
    single-instance deployments.
    
    Example usage:
        ```python
        registry = InMemoryRegistry()
        
        # Register an agent
        await registry.register(manifest)
        
        # Search for agents
        agents = await registry.search(capabilities=["web_search"])
        
        # Get specific agent
        agent = await registry.get("my-agent-id")
        ```
    """
    
    def __init__(self):
        """Initialize the in-memory registry."""
        self._manifests: Dict[str, AgentManifest] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, manifest: AgentManifest) -> None:
        """
        Store an agent manifest.
        
        Args:
            manifest: The AgentManifest to store
        """
        async with self._lock:
            manifest.updated_at = datetime.utcnow()
            self._manifests[manifest.id] = manifest
    
    async def get(self, agent_id: str) -> Optional[AgentManifest]:
        """
        Retrieve an agent manifest by ID.
        
        Args:
            agent_id: The ID of the agent
        
        Returns:
            The AgentManifest if found, None otherwise
        """
        return self._manifests.get(agent_id)
    
    async def search(
        self,
        capabilities: Optional[List[str]] = None,
        framework: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_trust_score: Optional[float] = None,
    ) -> List[AgentManifest]:
        """
        Search for agents matching criteria.
        
        Args:
            capabilities: Required capabilities (agent must have ALL)
            framework: Filter by framework name
            tags: Required tags (agent must have ALL)
            min_trust_score: Minimum trust score
        
        Returns:
            List of matching AgentManifest objects
        """
        results: List[AgentManifest] = []
        
        for manifest in self._manifests.values():
            # Skip non-active agents
            if manifest.status != AgentStatus.ACTIVE:
                continue
            
            # Check framework filter
            if framework and manifest.framework != framework:
                continue
            
            # Check capabilities filter
            if capabilities:
                agent_caps = {cap.name for cap in manifest.capabilities}
                if not all(c in agent_caps for c in capabilities):
                    continue
            
            # Check tags filter
            if tags:
                agent_tags = set(manifest.tags)
                if not all(t in agent_tags for t in tags):
                    continue
            
            # Check trust score filter
            if min_trust_score is not None:
                if manifest.trust_score is None:
                    continue
                if manifest.trust_score < min_trust_score:
                    continue
            
            results.append(manifest)
        
        return results
    
    async def delete(self, agent_id: str) -> bool:
        """
        Remove an agent from the registry.
        
        Args:
            agent_id: The ID of the agent to remove
        
        Returns:
            True if removed, False if not found
        """
        async with self._lock:
            if agent_id in self._manifests:
                del self._manifests[agent_id]
                return True
            return False
    
    async def list_all(self) -> List[AgentManifest]:
        """
        List all registered agents.
        
        Returns:
            List of all AgentManifest objects
        """
        return list(self._manifests.values())
    
    async def update(self, manifest: AgentManifest) -> bool:
        """
        Update an existing agent manifest.
        
        Args:
            manifest: The updated manifest
        
        Returns:
            True if updated, False if not found
        """
        async with self._lock:
            if manifest.id in self._manifests:
                manifest.updated_at = datetime.utcnow()
                self._manifests[manifest.id] = manifest
                return True
            return False
    
    async def set_status(self, agent_id: str, status: AgentStatus) -> bool:
        """
        Update the status of an agent.
        
        Args:
            agent_id: The ID of the agent
            status: The new status
        
        Returns:
            True if updated, False if not found
        """
        async with self._lock:
            if agent_id in self._manifests:
                self._manifests[agent_id].status = status
                self._manifests[agent_id].updated_at = datetime.utcnow()
                return True
            return False
    
    async def count(self) -> int:
        """Get the number of registered agents."""
        return len(self._manifests)
    
    async def clear(self) -> None:
        """Remove all agents from the registry."""
        async with self._lock:
            self._manifests.clear()
    
    def __len__(self) -> int:
        """Return the number of registered agents."""
        return len(self._manifests)
