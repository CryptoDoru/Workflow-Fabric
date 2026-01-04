"""
AI Workflow Fabric - CrewAI Adapter

This module provides the adapter for integrating CrewAI agents with AWF.
It handles registration, execution, and event bridging for CrewAI Agent
and Crew objects.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from awf.adapters.base import (
    AgentNotFoundError,
    AgentRegistry,
    BaseAdapter,
    ExecutionError,
    RegistrationError,
    TrustScorer,
    ValidationError,
)
from awf.core.types import (
    AgentManifest,
    AgentStatus,
    Capability,
    CapabilityType,
    Event,
    EventType,
    Schema,
    SchemaProperty,
    Task,
    TaskError,
    TaskMetrics,
    TaskResult,
    TaskStatus,
)

# Type hints for CrewAI (optional import)
try:
    from crewai import Agent, Crew, Task as CrewAITask
    
    CREWAI_AVAILABLE = True
except ImportError:
    Agent = Any
    Crew = Any
    CrewAITask = Any
    CREWAI_AVAILABLE = False


# Type alias for CrewAI types
CrewAIAgent = Union[Agent, Crew]


class CrewAIAdapter(BaseAdapter):
    """
    Adapter for CrewAI agents and crews.
    
    This adapter translates between CrewAI's Agent/Crew API and the
    Agent State Protocol (ASP).
    
    Example usage:
        ```python
        from crewai import Agent, Crew, Task
        from awf.adapters.crewai import CrewAIAdapter
        
        # Create your CrewAI agent
        researcher = Agent(
            role="Senior Researcher",
            goal="Research AI topics thoroughly",
            backstory="An expert AI researcher with years of experience",
        )
        
        # Register with AWF
        adapter = CrewAIAdapter()
        manifest = adapter.register(researcher, agent_id="my-researcher")
        
        # Or register an entire crew
        crew = Crew(agents=[researcher], tasks=[...])
        crew_manifest = adapter.register(crew, agent_id="my-research-crew")
        
        # Execute via ASP
        task = Task(agent_id="my-researcher", input={"topic": "AI Safety"})
        result = await adapter.execute(task)
        ```
    """
    
    framework_name = "crewai"
    framework_version: Optional[str] = None
    
    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        trust_scorer: Optional[TrustScorer] = None,
    ):
        """
        Initialize the CrewAI adapter.
        
        Args:
            registry: Optional agent registry for storing manifests
            trust_scorer: Optional trust scorer for computing trust scores
        """
        super().__init__(registry, trust_scorer)
        
        if not CREWAI_AVAILABLE:
            raise ImportError(
                "CrewAI is not installed. "
                "Install it with: pip install crewai"
            )
        
        # Try to get CrewAI version
        try:
            import crewai
            self.framework_version = getattr(crewai, "__version__", None)
        except Exception:
            pass
        
        # Store manifests and agents/crews
        self._manifests: Dict[str, AgentManifest] = {}
        self._agents: Dict[str, CrewAIAgent] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
    
    # =========================================================================
    # Registration Interface
    # =========================================================================
    
    def register(
        self,
        agent: CrewAIAgent,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentManifest:
        """
        Register a CrewAI agent or crew with AWF.
        
        Args:
            agent: A CrewAI Agent or Crew object
            agent_id: Optional custom agent ID
            metadata: Optional additional metadata
        
        Returns:
            The generated AgentManifest
        
        Raises:
            ValidationError: If the agent is invalid
            RegistrationError: If registration fails
        """
        # Validate the agent
        self._validate_agent(agent)
        
        # Determine if this is an Agent or Crew
        is_crew = self._is_crew(agent)
        
        # Generate or validate agent ID
        if agent_id is None:
            agent_id = self._generate_agent_id(agent)
        
        # Check for existing registration
        if agent_id in self._manifests:
            raise RegistrationError(
                f"Agent already registered with ID: {agent_id}. "
                "Use unregister() first or provide a different ID."
            )
        
        # Extract capabilities
        capabilities = self.extract_capabilities(agent)
        
        # Build manifest
        manifest = AgentManifest(
            id=agent_id,
            name=self._get_agent_name(agent, metadata),
            version=metadata.get("version", "1.0.0") if metadata else "1.0.0",
            framework=self.framework_name,
            framework_version=self.framework_version,
            capabilities=capabilities,
            description=self._get_agent_description(agent),
            tags=metadata.get("tags", []) if metadata else [],
            publisher=metadata.get("publisher") if metadata else None,
            status=AgentStatus.ACTIVE,
            metadata={
                "type": "crew" if is_crew else "agent",
                "role": self._get_role(agent) if not is_crew else None,
                "goal": self._get_goal(agent) if not is_crew else None,
                "agents": self._get_crew_agents(agent) if is_crew else None,
                "process": self._get_crew_process(agent) if is_crew else None,
                **(metadata.get("extra", {}) if metadata else {}),
            },
        )
        
        # Store the agent and manifest
        self._agents[agent_id] = agent
        self._manifests[agent_id] = manifest
        self._registered_agents[agent_id] = agent
        
        return manifest
    
    def unregister(self, agent_id: str) -> bool:
        """Remove an agent from the registry."""
        if agent_id in self._manifests:
            del self._manifests[agent_id]
            del self._agents[agent_id]
            del self._registered_agents[agent_id]
            return True
        return False
    
    def get_manifest(self, agent_id: str) -> Optional[AgentManifest]:
        """Retrieve the manifest for a registered agent."""
        return self._manifests.get(agent_id)
    
    def list_agents(self) -> List[AgentManifest]:
        """List all registered agents for this adapter."""
        return list(self._manifests.values())
    
    # =========================================================================
    # Capability Extraction
    # =========================================================================
    
    def extract_capabilities(self, agent: CrewAIAgent) -> List[Capability]:
        """
        Extract capabilities from a CrewAI agent or crew.
        
        For agents: extracts tools and role-based capabilities
        For crews: extracts combined capabilities from all agents
        """
        capabilities: List[Capability] = []
        
        if self._is_crew(agent):
            # Extract from all agents in the crew
            crew_agents = getattr(agent, "agents", [])
            for crew_agent in crew_agents:
                agent_caps = self._extract_agent_capabilities(crew_agent)
                capabilities.extend(agent_caps)
            
            # Add crew-level orchestration capability
            capabilities.append(Capability(
                name="crew_orchestration",
                type=CapabilityType.REASONING,
                description="Orchestrates multiple agents to complete tasks",
                metadata={"agent_count": len(crew_agents)},
            ))
        else:
            capabilities = self._extract_agent_capabilities(agent)
        
        return capabilities
    
    def _extract_agent_capabilities(self, agent: Agent) -> List[Capability]:
        """Extract capabilities from a single CrewAI agent."""
        capabilities: List[Capability] = []
        
        # Add role-based capability
        role = getattr(agent, "role", None)
        if role:
            capabilities.append(Capability(
                name=self._normalize_capability_name(role),
                type=CapabilityType.REASONING,
                description=f"Agent role: {role}",
                metadata={"source": "role"},
            ))
        
        # Extract tools
        tools = getattr(agent, "tools", []) or []
        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))
            capabilities.append(Capability(
                name=tool_name,
                type=CapabilityType.TOOL,
                description=getattr(tool, "description", None),
                permissions=self._infer_tool_permissions(tool),
                metadata={"tool_type": type(tool).__name__},
            ))
        
        # Check for memory capability
        if getattr(agent, "memory", False):
            capabilities.append(Capability(
                name="agent_memory",
                type=CapabilityType.MEMORY,
                description="Agent has memory enabled for context retention",
            ))
        
        return capabilities
    
    def infer_input_schema(self, agent: CrewAIAgent) -> Optional[Dict[str, Any]]:
        """
        Infer input schema for a CrewAI agent/crew.
        
        CrewAI uses task descriptions, so schema is flexible.
        """
        return {
            "type": "object",
            "properties": {
                "task_description": {
                    "type": "string",
                    "description": "Description of the task to perform",
                },
                "context": {
                    "type": "object",
                    "description": "Additional context for the task",
                },
            },
            "required": ["task_description"],
        }
    
    def infer_output_schema(self, agent: CrewAIAgent) -> Optional[Dict[str, Any]]:
        """
        Infer output schema for a CrewAI agent/crew.
        """
        return {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "description": "The result of the task execution",
                },
                "raw_output": {
                    "type": "string",
                    "description": "Raw output from the agent",
                },
            },
        }
    
    # =========================================================================
    # Execution Interface
    # =========================================================================
    
    async def execute(self, task: Task) -> TaskResult:
        """
        Execute a task on a registered CrewAI agent/crew.
        
        Args:
            task: The ASP Task to execute
        
        Returns:
            The TaskResult
        
        Raises:
            AgentNotFoundError: If the agent is not registered
            ExecutionError: If execution fails
        """
        # Look up the agent
        agent = self._agents.get(task.agent_id)
        if agent is None:
            raise AgentNotFoundError(task.agent_id)
        
        started_at = datetime.now(timezone.utc)
        start_time = time.perf_counter()
        
        try:
            # Execute based on type
            loop = asyncio.get_event_loop()
            
            if self._is_crew(agent):
                output = await self._execute_crew(agent, task, loop)
            else:
                output = await self._execute_agent(agent, task, loop)
            
            execution_time = int((time.perf_counter() - start_time) * 1000)
            
            return TaskResult(
                task_id=task.id,
                agent_id=task.agent_id,
                status=TaskStatus.COMPLETED,
                output=self._normalize_output(output),
                metrics=TaskMetrics(execution_time_ms=execution_time),
                trace_id=task.trace_id,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
        
        except asyncio.TimeoutError:
            execution_time = int((time.perf_counter() - start_time) * 1000)
            return TaskResult(
                task_id=task.id,
                agent_id=task.agent_id,
                status=TaskStatus.TIMEOUT,
                error=TaskError(
                    code="TIMEOUT",
                    message=f"Task timed out after {task.timeout_ms}ms",
                    retryable=True,
                ),
                metrics=TaskMetrics(execution_time_ms=execution_time),
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
        
        except Exception as e:
            execution_time = int((time.perf_counter() - start_time) * 1000)
            return TaskResult(
                task_id=task.id,
                agent_id=task.agent_id,
                status=TaskStatus.FAILED,
                error=TaskError(
                    code="EXECUTION_ERROR",
                    message=str(e),
                    stack_trace=self._format_exception(e),
                    retryable=self._is_retryable(e),
                ),
                metrics=TaskMetrics(execution_time_ms=execution_time),
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
    
    async def _execute_crew(
        self, crew: Crew, task: Task, loop: asyncio.AbstractEventLoop
    ) -> Any:
        """Execute a task on a CrewAI Crew."""
        # Create CrewAI task from ASP task
        task_description = task.input.get("task_description", str(task.input))
        
        # Run crew.kickoff() in thread pool
        if task.timeout_ms:
            timeout = task.timeout_ms / 1000.0
            output = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: crew.kickoff(inputs=task.input)
                ),
                timeout=timeout,
            )
        else:
            output = await loop.run_in_executor(
                None, lambda: crew.kickoff(inputs=task.input)
            )
        
        return output
    
    async def _execute_agent(
        self, agent: Agent, task: Task, loop: asyncio.AbstractEventLoop
    ) -> Any:
        """Execute a task on a single CrewAI Agent."""
        # For single agent execution, we create a minimal crew
        task_description = task.input.get("task_description", str(task.input))
        
        # Create a CrewAI Task
        crewai_task = CrewAITask(
            description=task_description,
            agent=agent,
            expected_output="Completed task result",
        )
        
        # Create minimal crew with single agent
        crew = Crew(
            agents=[agent],
            tasks=[crewai_task],
            verbose=False,
        )
        
        # Execute
        if task.timeout_ms:
            timeout = task.timeout_ms / 1000.0
            output = await asyncio.wait_for(
                loop.run_in_executor(
                    None, lambda: crew.kickoff(inputs=task.input)
                ),
                timeout=timeout,
            )
        else:
            output = await loop.run_in_executor(
                None, lambda: crew.kickoff(inputs=task.input)
            )
        
        return output
    
    async def execute_streaming(self, task: Task) -> AsyncIterator[Event]:
        """
        Execute a task with streaming events.
        
        Note: CrewAI has limited streaming support, so this implementation
        emits events at key points in execution.
        """
        agent = self._agents.get(task.agent_id)
        if agent is None:
            raise AgentNotFoundError(task.agent_id)
        
        started_at = datetime.now(timezone.utc)
        start_time = time.perf_counter()
        
        # Emit task started event
        yield Event(
            type=EventType.TASK_STARTED,
            source=task.agent_id,
            correlation_id=task.id,
            trace_id=task.trace_id,
            data={"input": task.input},
        )
        
        try:
            # Execute the task
            loop = asyncio.get_event_loop()
            
            if self._is_crew(agent):
                # Emit agent activation events for each agent in crew
                crew_agents = getattr(agent, "agents", [])
                for i, crew_agent in enumerate(crew_agents):
                    yield Event(
                        type=EventType.STATE_CHANGED,
                        source=task.agent_id,
                        correlation_id=task.id,
                        trace_id=task.trace_id,
                        data={
                            "state": "agent_activated",
                            "agent_index": i,
                            "agent_role": getattr(crew_agent, "role", "unknown"),
                        },
                    )
                
                output = await self._execute_crew(agent, task, loop)
            else:
                output = await self._execute_agent(agent, task, loop)
            
            execution_time = int((time.perf_counter() - start_time) * 1000)
            
            # Emit completion event
            yield Event(
                type=EventType.TASK_COMPLETED,
                source=task.agent_id,
                correlation_id=task.id,
                trace_id=task.trace_id,
                data={
                    "output": self._normalize_output(output),
                    "metrics": {"executionTimeMs": execution_time},
                },
            )
        
        except Exception as e:
            execution_time = int((time.perf_counter() - start_time) * 1000)
            yield Event(
                type=EventType.TASK_FAILED,
                source=task.agent_id,
                correlation_id=task.id,
                trace_id=task.trace_id,
                data={
                    "error": {
                        "code": "EXECUTION_ERROR",
                        "message": str(e),
                    },
                    "metrics": {"executionTimeMs": execution_time},
                },
            )
    
    async def cancel(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
            del self._running_tasks[task_id]
            return True
        return False
    
    # =========================================================================
    # Private Helper Methods
    # =========================================================================
    
    def _validate_agent(self, agent: CrewAIAgent) -> None:
        """Validate that the agent is a valid CrewAI agent or crew."""
        super()._validate_agent(agent)
        
        if not CREWAI_AVAILABLE:
            raise ValidationError("CrewAI is not installed")
        
        # Check if it's an Agent or Crew
        is_agent = hasattr(agent, "role") and hasattr(agent, "goal")
        is_crew = hasattr(agent, "agents") and hasattr(agent, "kickoff")
        
        if not (is_agent or is_crew):
            raise ValidationError(
                f"Expected CrewAI Agent or Crew, got {type(agent).__name__}"
            )
    
    def _is_crew(self, agent: CrewAIAgent) -> bool:
        """Check if the object is a Crew."""
        return hasattr(agent, "agents") and hasattr(agent, "kickoff")
    
    def _generate_agent_id(self, agent: CrewAIAgent) -> str:
        """Generate a deterministic agent ID."""
        if self._is_crew(agent):
            # Hash based on crew agents
            crew_agents = getattr(agent, "agents", [])
            roles = sorted([getattr(a, "role", "") for a in crew_agents])
            content = f"crew:{roles}"
        else:
            # Hash based on role and goal
            role = getattr(agent, "role", "")
            goal = getattr(agent, "goal", "")
            content = f"agent:{role}:{goal}"
        
        hash_value = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"crewai-{hash_value}"
    
    def _get_agent_name(
        self, agent: CrewAIAgent, metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Get a name for the agent."""
        if metadata and "name" in metadata:
            return metadata["name"]
        
        if self._is_crew(agent):
            return "CrewAI Crew"
        
        role = getattr(agent, "role", None)
        return role if role else "CrewAI Agent"
    
    def _get_agent_description(self, agent: CrewAIAgent) -> Optional[str]:
        """Get description for the agent."""
        if self._is_crew(agent):
            agents = getattr(agent, "agents", [])
            roles = [getattr(a, "role", "agent") for a in agents]
            return f"A crew of {len(agents)} agents: {', '.join(roles)}"
        
        backstory = getattr(agent, "backstory", None)
        goal = getattr(agent, "goal", None)
        
        if backstory:
            return backstory
        elif goal:
            return f"Goal: {goal}"
        return None
    
    def _get_role(self, agent: Agent) -> Optional[str]:
        """Get the role from an agent."""
        return getattr(agent, "role", None)
    
    def _get_goal(self, agent: Agent) -> Optional[str]:
        """Get the goal from an agent."""
        return getattr(agent, "goal", None)
    
    def _get_crew_agents(self, crew: Crew) -> List[Dict[str, Any]]:
        """Get agent info from a crew."""
        agents = getattr(crew, "agents", [])
        return [
            {
                "role": getattr(a, "role", None),
                "goal": getattr(a, "goal", None),
            }
            for a in agents
        ]
    
    def _get_crew_process(self, crew: Crew) -> Optional[str]:
        """Get the process type from a crew."""
        process = getattr(crew, "process", None)
        if process:
            return str(process)
        return None
    
    def _normalize_capability_name(self, name: str) -> str:
        """Normalize a capability name to lowercase with underscores."""
        return name.lower().replace(" ", "_").replace("-", "_")
    
    def _infer_tool_permissions(self, tool: Any) -> List[str]:
        """Infer permissions required by a tool."""
        permissions: List[str] = []
        
        try:
            name = getattr(tool, "name", str(tool)).lower()
            
            if any(kw in name for kw in ["web", "http", "api", "search", "scrape"]):
                permissions.append("network:external")
            if any(kw in name for kw in ["file", "read", "write", "disk"]):
                permissions.append("filesystem:read")
            if any(kw in name for kw in ["exec", "shell", "command"]):
                permissions.append("process:execute")
            if any(kw in name for kw in ["sql", "database", "db"]):
                permissions.append("database:query")
        except Exception:
            pass
        
        return permissions
    
    def _normalize_output(self, output: Any) -> Dict[str, Any]:
        """Normalize crew/agent output to a dictionary."""
        if isinstance(output, dict):
            return output
        elif hasattr(output, "raw"):
            return {"result": str(output), "raw_output": output.raw}
        else:
            return {"result": str(output)}
    
    def _format_exception(self, e: Exception) -> str:
        """Format exception with traceback."""
        import traceback
        return "".join(traceback.format_exception(type(e), e, e.__traceback__))
    
    def _is_retryable(self, e: Exception) -> bool:
        """Determine if an exception is retryable."""
        retryable_types = (
            TimeoutError,
            ConnectionError,
            asyncio.TimeoutError,
        )
        return isinstance(e, retryable_types)
