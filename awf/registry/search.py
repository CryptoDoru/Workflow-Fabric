"""
AI Workflow Fabric - Capability Search Engine

This module provides advanced search capabilities for finding agents
based on their capabilities, with support for fuzzy matching and
semantic similarity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from awf.core.types import AgentManifest, Capability, CapabilityType


@dataclass
class SearchResult:
    """A search result with relevance score."""
    
    manifest: AgentManifest
    score: float
    matched_capabilities: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "agent_id": self.manifest.id,
            "agent_name": self.manifest.name,
            "score": self.score,
            "matched_capabilities": self.matched_capabilities,
            "framework": self.manifest.framework,
            "trust_score": self.manifest.trust_score,
        }


class CapabilitySearchEngine:
    """
    Search engine for finding agents by capabilities.
    
    Supports:
    - Exact capability matching
    - Fuzzy/partial matching
    - Capability type filtering
    - Relevance scoring
    
    Example usage:
        ```python
        engine = CapabilitySearchEngine()
        
        # Index agents
        engine.index(manifest1)
        engine.index(manifest2)
        
        # Search
        results = engine.search("web search")
        
        # Advanced search
        results = engine.search(
            query="search",
            capability_types=[CapabilityType.TOOL],
            min_score=0.5,
        )
        ```
    """
    
    def __init__(self):
        """Initialize the search engine."""
        self._manifests: Dict[str, AgentManifest] = {}
        self._capability_index: Dict[str, Set[str]] = {}  # capability -> agent_ids
        self._type_index: Dict[CapabilityType, Set[str]] = {}  # type -> agent_ids
    
    def index(self, manifest: AgentManifest) -> None:
        """
        Index an agent manifest for searching.
        
        Args:
            manifest: The manifest to index
        """
        agent_id = manifest.id
        self._manifests[agent_id] = manifest
        
        # Index each capability
        for cap in manifest.capabilities:
            # Exact name index
            cap_name = cap.name.lower()
            if cap_name not in self._capability_index:
                self._capability_index[cap_name] = set()
            self._capability_index[cap_name].add(agent_id)
            
            # Type index
            if cap.type not in self._type_index:
                self._type_index[cap.type] = set()
            self._type_index[cap.type].add(agent_id)
            
            # Index individual words for fuzzy matching
            words = self._tokenize(cap_name)
            for word in words:
                if word not in self._capability_index:
                    self._capability_index[word] = set()
                self._capability_index[word].add(agent_id)
    
    def remove(self, agent_id: str) -> None:
        """
        Remove an agent from the index.
        
        Args:
            agent_id: The ID of the agent to remove
        """
        if agent_id not in self._manifests:
            return
        
        manifest = self._manifests[agent_id]
        
        # Remove from capability index
        for cap in manifest.capabilities:
            cap_name = cap.name.lower()
            if cap_name in self._capability_index:
                self._capability_index[cap_name].discard(agent_id)
            
            # Remove from word index
            words = self._tokenize(cap_name)
            for word in words:
                if word in self._capability_index:
                    self._capability_index[word].discard(agent_id)
            
            # Remove from type index
            if cap.type in self._type_index:
                self._type_index[cap.type].discard(agent_id)
        
        del self._manifests[agent_id]
    
    def search(
        self,
        query: str,
        capability_types: Optional[List[CapabilityType]] = None,
        min_score: float = 0.0,
        limit: int = 10,
    ) -> List[SearchResult]:
        """
        Search for agents by capability query.
        
        Args:
            query: The search query (capability name or description)
            capability_types: Filter by capability types
            min_score: Minimum relevance score (0-1)
            limit: Maximum number of results
        
        Returns:
            List of SearchResult objects, sorted by relevance
        """
        if not query:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query.lower())
        
        # Find candidate agents
        candidates: Dict[str, float] = {}
        matched_caps: Dict[str, List[str]] = {}
        
        for token in query_tokens:
            # Exact match
            if token in self._capability_index:
                for agent_id in self._capability_index[token]:
                    candidates[agent_id] = candidates.get(agent_id, 0) + 1.0
                    if agent_id not in matched_caps:
                        matched_caps[agent_id] = []
                    matched_caps[agent_id].append(token)
            
            # Prefix match
            for cap_name, agent_ids in self._capability_index.items():
                if cap_name.startswith(token) and cap_name != token:
                    for agent_id in agent_ids:
                        candidates[agent_id] = candidates.get(agent_id, 0) + 0.5
                        if agent_id not in matched_caps:
                            matched_caps[agent_id] = []
                        if cap_name not in matched_caps[agent_id]:
                            matched_caps[agent_id].append(cap_name)
        
        # Filter by capability type
        if capability_types:
            type_agents: Set[str] = set()
            for cap_type in capability_types:
                if cap_type in self._type_index:
                    type_agents.update(self._type_index[cap_type])
            candidates = {
                k: v for k, v in candidates.items() if k in type_agents
            }
        
        # Normalize scores and build results
        max_score = max(candidates.values()) if candidates else 1.0
        results: List[SearchResult] = []
        
        for agent_id, raw_score in candidates.items():
            score = raw_score / max_score
            
            if score >= min_score:
                manifest = self._manifests[agent_id]
                results.append(SearchResult(
                    manifest=manifest,
                    score=score,
                    matched_capabilities=matched_caps.get(agent_id, []),
                ))
        
        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        
        return results[:limit]
    
    def find_by_capability(
        self, capability_name: str, exact: bool = True
    ) -> List[AgentManifest]:
        """
        Find agents with a specific capability.
        
        Args:
            capability_name: The capability name to search for
            exact: If True, require exact match; if False, allow partial match
        
        Returns:
            List of matching AgentManifest objects
        """
        cap_name = capability_name.lower()
        
        if exact:
            agent_ids = self._capability_index.get(cap_name, set())
        else:
            agent_ids: Set[str] = set()
            for indexed_cap, ids in self._capability_index.items():
                if cap_name in indexed_cap or indexed_cap in cap_name:
                    agent_ids.update(ids)
        
        return [self._manifests[aid] for aid in agent_ids if aid in self._manifests]
    
    def find_by_type(self, capability_type: CapabilityType) -> List[AgentManifest]:
        """
        Find agents with capabilities of a specific type.
        
        Args:
            capability_type: The capability type to search for
        
        Returns:
            List of matching AgentManifest objects
        """
        agent_ids = self._type_index.get(capability_type, set())
        return [self._manifests[aid] for aid in agent_ids if aid in self._manifests]
    
    def get_all_capabilities(self) -> List[str]:
        """
        Get all indexed capability names.
        
        Returns:
            List of unique capability names
        """
        return list(self._capability_index.keys())
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into searchable words."""
        # Split on non-alphanumeric characters
        words = re.split(r'[^a-z0-9]+', text.lower())
        # Filter empty strings and very short words
        return [w for w in words if len(w) >= 2]
    
    def clear(self) -> None:
        """Clear all indexed data."""
        self._manifests.clear()
        self._capability_index.clear()
        self._type_index.clear()
    
    def __len__(self) -> int:
        """Return the number of indexed agents."""
        return len(self._manifests)
