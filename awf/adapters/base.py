"""
AI Workflow Fabric - Base Adapter

This module defines the abstract base class for framework adapters.
All framework-specific adapters (LangGraph, CrewAI, etc.) must inherit from this class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional, Type, TypeVar

from awf.core.types import (
    AgentManifest,
    Capability,
    Event,
    Task,
    TaskResult,
    TrustScore,
)


# Type variable for framework-specific agent types
T = TypeVar("T")


class BaseAdapter(ABC):
    """
    Abstract base class for framework adapters.
    
    Framework adapters translate between the native framework's API and
    the Agent State Protocol (ASP). Each adapter must implement:
    
    1. Registration: Convert native agents to ASP manifests
    2. Execution: Translate ASP tasks to native invocations
    3. Events: Bridge native events to ASP event stream
    """
    
    # Framework identifier (e.g., "langgraph", "crewai")
    framework_name: str = "unknown"
    framework_version: Optional[str] = None
    
    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        trust_scorer: Optional[TrustScorer] = None,
    ):
        """
        Initialize the adapter.
        
        Args:
            registry: Optional agent registry for storing manifests
            trust_scorer: Optional trust scorer for computing trust scores
        """
        self.registry = registry
        self.trust_scorer = trust_scorer
        self._registered_agents: Dict[str, Any] = {}  # id -> native agent
    
    # =========================================================================
    # Registration Interface
    # =========================================================================
    
    @abstractmethod
    def register(
        self,
        agent: T,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentManifest:
        """
        Register a native agent with AWF.
        
        This method should:
        1. Extract metadata from the native agent
        2. Generate an ASP-compliant manifest
        3. Store the agent for later execution
        4. Optionally compute trust score
        
        Args:
            agent: The native agent object (e.g., LangGraph StateGraph)
            agent_id: Optional custom agent ID (auto-generated if not provided)
            metadata: Optional additional metadata to include in manifest
        
        Returns:
            The generated AgentManifest
        
        Raises:
            ValueError: If the agent is invalid or cannot be registered
        """
        pass
    
    @abstractmethod
    def unregister(self, agent_id: str) -> bool:
        """
        Remove an agent from the registry.
        
        Args:
            agent_id: The ID of the agent to remove
        
        Returns:
            True if the agent was removed, False if it wasn't found
        """
        pass
    
    @abstractmethod
    def get_manifest(self, agent_id: str) -> Optional[AgentManifest]:
        """
        Retrieve the manifest for a registered agent.
        
        Args:
            agent_id: The ID of the agent
        
        Returns:
            The AgentManifest if found, None otherwise
        """
        pass
    
    @abstractmethod
    def list_agents(self) -> List[AgentManifest]:
        """
        List all registered agents for this adapter.
        
        Returns:
            List of AgentManifest objects
        """
        pass
    
    # =========================================================================
    # Capability Extraction
    # =========================================================================
    
    @abstractmethod
    def extract_capabilities(self, agent: T) -> List[Capability]:
        """
        Extract capabilities from a native agent.
        
        This method should introspect the native agent to determine
        what capabilities it has (tools, memory, etc.).
        
        Args:
            agent: The native agent object
        
        Returns:
            List of Capability objects
        """
        pass
    
    @abstractmethod
    def infer_input_schema(self, agent: T) -> Optional[Dict[str, Any]]:
        """
        Infer the input schema from a native agent.
        
        Args:
            agent: The native agent object
        
        Returns:
            JSON Schema dict for the agent's expected input, or None
        """
        pass
    
    @abstractmethod
    def infer_output_schema(self, agent: T) -> Optional[Dict[str, Any]]:
        """
        Infer the output schema from a native agent.
        
        Args:
            agent: The native agent object
        
        Returns:
            JSON Schema dict for the agent's expected output, or None
        """
        pass
    
    # =========================================================================
    # Execution Interface
    # =========================================================================
    
    @abstractmethod
    async def execute(self, task: Task) -> TaskResult:
        """
        Execute a task on a registered agent.
        
        This method should:
        1. Look up the native agent by task.agent_id
        2. Translate the ASP task to native format
        3. Execute the agent
        4. Translate the result back to ASP format
        
        Args:
            task: The ASP Task to execute
        
        Returns:
            The TaskResult
        
        Raises:
            AgentNotFoundError: If the agent is not registered
            ExecutionError: If execution fails
        """
        pass
    
    @abstractmethod
    async def execute_streaming(
        self, task: Task
    ) -> AsyncIterator[Event]:
        """
        Execute a task with streaming events.
        
        Yields events as they occur during execution, including
        state changes, intermediate outputs, and the final result.
        
        Args:
            task: The ASP Task to execute
        
        Yields:
            Event objects as they occur
        
        Raises:
            AgentNotFoundError: If the agent is not registered
            ExecutionError: If execution fails
        """
        pass
    
    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            task_id: The ID of the task to cancel
        
        Returns:
            True if the task was cancelled, False if it wasn't found or already complete
        """
        pass
    
    # =========================================================================
    # Status and Health
    # =========================================================================
    
    def get_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of an agent.
        
        Args:
            agent_id: The ID of the agent
        
        Returns:
            Status dict with at least 'status' key, or None if not found
        """
        manifest = self.get_manifest(agent_id)
        if manifest:
            return {
                "status": manifest.status.value,
                "trust_score": manifest.trust_score,
            }
        return None
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of this adapter.
        
        Returns:
            Dict with 'healthy' boolean and optional details
        """
        return {
            "healthy": True,
            "framework": self.framework_name,
            "framework_version": self.framework_version,
            "registered_agents": len(self._registered_agents),
        }
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _generate_agent_id(self, agent: T) -> str:
        """
        Generate a deterministic agent ID based on agent structure.
        
        Override this method for framework-specific ID generation.
        
        Args:
            agent: The native agent object
        
        Returns:
            A unique string identifier
        """
        import hashlib
        # Default: hash the string representation
        content = str(agent)
        return f"{self.framework_name}-{hashlib.sha256(content.encode()).hexdigest()[:12]}"
    
    def _validate_agent(self, agent: T) -> None:
        """
        Validate that an agent can be registered.
        
        Override this method for framework-specific validation.
        
        Args:
            agent: The native agent object
        
        Raises:
            ValueError: If the agent is invalid
        """
        if agent is None:
            raise ValueError("Agent cannot be None")


# =============================================================================
# Supporting Interfaces
# =============================================================================


class AgentRegistry(ABC):
    """
    Abstract interface for agent registries.
    
    A registry stores and retrieves agent manifests, supporting
    discovery and search operations.
    """
    
    @abstractmethod
    async def register(self, manifest: AgentManifest) -> None:
        """Store an agent manifest."""
        pass
    
    @abstractmethod
    async def get(self, agent_id: str) -> Optional[AgentManifest]:
        """Retrieve an agent manifest by ID."""
        pass
    
    @abstractmethod
    async def search(
        self,
        capabilities: Optional[List[str]] = None,
        framework: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_trust_score: Optional[float] = None,
    ) -> List[AgentManifest]:
        """Search for agents matching criteria."""
        pass
    
    @abstractmethod
    async def delete(self, agent_id: str) -> bool:
        """Remove an agent from the registry."""
        pass
    
    @abstractmethod
    async def list_all(self) -> List[AgentManifest]:
        """List all registered agents."""
        pass


class TrustScorer(ABC):
    """
    Abstract interface for trust scoring.
    
    Computes trust scores for agents based on various factors.
    """
    
    @abstractmethod
    async def compute_score(self, manifest: AgentManifest) -> TrustScore:
        """Compute trust score for an agent."""
        pass
    
    @abstractmethod
    async def update_score(
        self, agent_id: str, event: Event
    ) -> Optional[TrustScore]:
        """Update trust score based on an event (e.g., abuse report)."""
        pass


# =============================================================================
# Exceptions
# =============================================================================


class AdapterError(Exception):
    """Base exception for adapter errors."""
    pass


class AgentNotFoundError(AdapterError):
    """Raised when an agent is not found in the registry."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        super().__init__(f"Agent not found: {agent_id}")


class RegistrationError(AdapterError):
    """Raised when agent registration fails."""
    pass


class ExecutionError(AdapterError):
    """Raised when task execution fails."""
    
    def __init__(self, message: str, cause: Optional[Exception] = None):
        self.cause = cause
        super().__init__(message)


class ValidationError(AdapterError):
    """Raised when validation fails."""
    pass
