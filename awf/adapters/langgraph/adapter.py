"""
AI Workflow Fabric - LangGraph Adapter

This module provides the adapter for integrating LangGraph agents with AWF.
It handles registration, execution, and event bridging for LangGraph StateGraph
and CompiledGraph objects.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Type, Union

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

# Type hints for LangGraph (optional import)
try:
    from langgraph.graph import StateGraph
    from langgraph.graph.graph import CompiledGraph
    
    LANGGRAPH_AVAILABLE = True
except ImportError:
    StateGraph = Any
    CompiledGraph = Any
    LANGGRAPH_AVAILABLE = False


# Type alias for LangGraph graph types
LangGraphAgent = Union[StateGraph, CompiledGraph]


class LangGraphAdapter(BaseAdapter):
    """
    Adapter for LangGraph agents.
    
    This adapter translates between LangGraph's StateGraph/CompiledGraph API
    and the Agent State Protocol (ASP).
    
    Example usage:
        ```python
        from langgraph.graph import StateGraph
        from awf.adapters.langgraph import LangGraphAdapter
        
        # Create your LangGraph agent
        graph = StateGraph(MyState)
        graph.add_node("researcher", researcher_node)
        graph.add_node("writer", writer_node)
        graph.add_edge("researcher", "writer")
        compiled = graph.compile()
        
        # Register with AWF
        adapter = LangGraphAdapter()
        manifest = adapter.register(compiled, agent_id="my-research-agent")
        
        # Execute via ASP
        task = Task(agent_id="my-research-agent", input={"topic": "AI Safety"})
        result = await adapter.execute(task)
        ```
    """
    
    framework_name = "langgraph"
    framework_version: Optional[str] = None
    
    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        trust_scorer: Optional[TrustScorer] = None,
    ):
        """
        Initialize the LangGraph adapter.
        
        Args:
            registry: Optional agent registry for storing manifests
            trust_scorer: Optional trust scorer for computing trust scores
        """
        super().__init__(registry, trust_scorer)
        
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is not installed. "
                "Install it with: pip install langgraph"
            )
        
        # Try to get LangGraph version
        try:
            import langgraph
            self.framework_version = getattr(langgraph, "__version__", None)
        except Exception:
            pass
        
        # Store manifests and compiled graphs
        self._manifests: Dict[str, AgentManifest] = {}
        self._graphs: Dict[str, CompiledGraph] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
    
    # =========================================================================
    # Registration Interface
    # =========================================================================
    
    def register(
        self,
        agent: LangGraphAgent,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentManifest:
        """
        Register a LangGraph agent with AWF.
        
        Args:
            agent: A LangGraph StateGraph or CompiledGraph
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
        
        # Compile if necessary
        compiled = self._ensure_compiled(agent)
        
        # Generate or validate agent ID
        if agent_id is None:
            agent_id = self._generate_agent_id(compiled)
        
        # Check for existing registration
        if agent_id in self._manifests:
            raise RegistrationError(
                f"Agent already registered with ID: {agent_id}. "
                "Use unregister() first or provide a different ID."
            )
        
        # Extract capabilities
        capabilities = self.extract_capabilities(compiled)
        
        # Infer schemas
        input_schema = self.infer_input_schema(compiled)
        output_schema = self.infer_output_schema(compiled)
        
        # Build manifest
        manifest = AgentManifest(
            id=agent_id,
            name=metadata.get("name", agent_id) if metadata else agent_id,
            version=metadata.get("version", "1.0.0") if metadata else "1.0.0",
            framework=self.framework_name,
            framework_version=self.framework_version,
            capabilities=capabilities,
            input_schema=self._dict_to_schema(input_schema) if input_schema else None,
            output_schema=self._dict_to_schema(output_schema) if output_schema else None,
            description=metadata.get("description") if metadata else None,
            tags=metadata.get("tags", []) if metadata else [],
            publisher=metadata.get("publisher") if metadata else None,
            status=AgentStatus.ACTIVE,
            metadata={
                "nodes": self._extract_node_names(compiled),
                "edges": self._extract_edges(compiled),
                **(metadata.get("extra", {}) if metadata else {}),
            },
        )
        
        # Store the graph and manifest
        self._graphs[agent_id] = compiled
        self._manifests[agent_id] = manifest
        self._registered_agents[agent_id] = compiled
        
        return manifest
    
    def unregister(self, agent_id: str) -> bool:
        """Remove an agent from the registry."""
        if agent_id in self._manifests:
            del self._manifests[agent_id]
            del self._graphs[agent_id]
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
    
    def extract_capabilities(self, agent: LangGraphAgent) -> List[Capability]:
        """
        Extract capabilities from a LangGraph agent.
        
        Analyzes the graph structure to determine capabilities:
        - Each node becomes a capability
        - Tools used by nodes are captured
        - Memory/state patterns are detected
        """
        compiled = self._ensure_compiled(agent)
        capabilities: List[Capability] = []
        
        # Extract node-based capabilities
        nodes = self._extract_node_names(compiled)
        for node_name in nodes:
            cap = Capability(
                name=node_name,
                type=CapabilityType.REASONING,
                description=f"Graph node: {node_name}",
                metadata={"node_type": "langgraph_node"},
            )
            capabilities.append(cap)
        
        # Try to extract tool capabilities
        tools = self._extract_tools(compiled)
        for tool in tools:
            cap = Capability(
                name=tool.get("name", "unknown_tool"),
                type=CapabilityType.TOOL,
                description=tool.get("description"),
                permissions=self._infer_tool_permissions(tool),
                metadata={"tool_type": tool.get("type", "unknown")},
            )
            capabilities.append(cap)
        
        return capabilities
    
    def infer_input_schema(self, agent: LangGraphAgent) -> Optional[Dict[str, Any]]:
        """
        Infer input schema from the LangGraph state type.
        
        LangGraph uses TypedDict or Pydantic models for state.
        We inspect these to generate JSON schema.
        """
        compiled = self._ensure_compiled(agent)
        
        try:
            # Get the state schema from the graph
            state_schema = getattr(compiled, "schema", None)
            if state_schema is None:
                # Try to get from builder
                builder = getattr(compiled, "builder", None)
                if builder:
                    state_schema = getattr(builder, "schema", None)
            
            if state_schema:
                return self._type_to_json_schema(state_schema)
        except Exception:
            pass
        
        return None
    
    def infer_output_schema(self, agent: LangGraphAgent) -> Optional[Dict[str, Any]]:
        """
        Infer output schema from the LangGraph state type.
        
        In LangGraph, input and output typically share the same state schema.
        """
        # For LangGraph, output schema is typically the same as input
        return self.infer_input_schema(agent)
    
    # =========================================================================
    # Execution Interface
    # =========================================================================
    
    async def execute(self, task: Task) -> TaskResult:
        """
        Execute a task on a registered LangGraph agent.
        
        Args:
            task: The ASP Task to execute
        
        Returns:
            The TaskResult
        
        Raises:
            AgentNotFoundError: If the agent is not registered
            ExecutionError: If execution fails
        """
        # Look up the graph
        graph = self._graphs.get(task.agent_id)
        if graph is None:
            raise AgentNotFoundError(task.agent_id)
        
        started_at = datetime.utcnow()
        start_time = time.perf_counter()
        
        try:
            # Execute the graph
            # LangGraph's invoke() is synchronous but we run in thread pool
            loop = asyncio.get_event_loop()
            
            # Prepare config with task metadata
            config = {
                "configurable": {
                    "task_id": task.id,
                    "trace_id": task.trace_id,
                    "awf_context": task.context,
                }
            }
            
            # Handle timeout
            if task.timeout_ms:
                timeout = task.timeout_ms / 1000.0
                output = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, lambda: graph.invoke(task.input, config=config)
                    ),
                    timeout=timeout,
                )
            else:
                output = await loop.run_in_executor(
                    None, lambda: graph.invoke(task.input, config=config)
                )
            
            execution_time = int((time.perf_counter() - start_time) * 1000)
            
            return TaskResult(
                task_id=task.id,
                agent_id=task.agent_id,
                status=TaskStatus.COMPLETED,
                output=output if isinstance(output, dict) else {"result": output},
                metrics=TaskMetrics(execution_time_ms=execution_time),
                trace_id=task.trace_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
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
                trace_id=task.trace_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
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
                trace_id=task.trace_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )
    
    async def execute_streaming(self, task: Task) -> AsyncIterator[Event]:
        """
        Execute a task with streaming events.
        
        Uses LangGraph's stream() method to yield events as they occur.
        """
        graph = self._graphs.get(task.agent_id)
        if graph is None:
            raise AgentNotFoundError(task.agent_id)
        
        started_at = datetime.utcnow()
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
            # Prepare config
            config = {
                "configurable": {
                    "task_id": task.id,
                    "trace_id": task.trace_id,
                }
            }
            
            # Stream execution
            loop = asyncio.get_event_loop()
            
            # Get stream iterator
            def get_stream():
                return list(graph.stream(task.input, config=config))
            
            stream_results = await loop.run_in_executor(None, get_stream)
            
            # Yield state change events for each chunk
            for chunk in stream_results:
                yield Event(
                    type=EventType.STATE_CHANGED,
                    source=task.agent_id,
                    correlation_id=task.id,
                    trace_id=task.trace_id,
                    data={"chunk": chunk},
                )
            
            # Get final output (last chunk contains final state)
            final_output = stream_results[-1] if stream_results else {}
            execution_time = int((time.perf_counter() - start_time) * 1000)
            
            # Emit completion event
            yield Event(
                type=EventType.TASK_COMPLETED,
                source=task.agent_id,
                correlation_id=task.id,
                trace_id=task.trace_id,
                data={
                    "output": final_output,
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
    
    def _validate_agent(self, agent: LangGraphAgent) -> None:
        """Validate that the agent is a valid LangGraph graph."""
        super()._validate_agent(agent)
        
        if not LANGGRAPH_AVAILABLE:
            raise ValidationError("LangGraph is not installed")
        
        # Check if it's a StateGraph or CompiledGraph
        valid_types = (StateGraph, CompiledGraph) if LANGGRAPH_AVAILABLE else ()
        if not isinstance(agent, valid_types):
            raise ValidationError(
                f"Expected StateGraph or CompiledGraph, got {type(agent).__name__}"
            )
    
    def _ensure_compiled(self, agent: LangGraphAgent) -> CompiledGraph:
        """Ensure we have a compiled graph."""
        if isinstance(agent, CompiledGraph):
            return agent
        
        # Compile the StateGraph
        return agent.compile()
    
    def _generate_agent_id(self, agent: LangGraphAgent) -> str:
        """Generate a deterministic agent ID based on graph structure."""
        compiled = self._ensure_compiled(agent)
        
        # Create hash from nodes and edges
        nodes = sorted(self._extract_node_names(compiled))
        edges = sorted([str(e) for e in self._extract_edges(compiled)])
        content = f"nodes:{nodes}|edges:{edges}"
        
        hash_value = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"langgraph-{hash_value}"
    
    def _extract_node_names(self, compiled: CompiledGraph) -> List[str]:
        """Extract node names from a compiled graph."""
        try:
            # Access the graph's nodes
            nodes = getattr(compiled, "nodes", {})
            return list(nodes.keys())
        except Exception:
            return []
    
    def _extract_edges(self, compiled: CompiledGraph) -> List[Dict[str, Any]]:
        """Extract edges from a compiled graph."""
        try:
            edges = getattr(compiled, "edges", [])
            return [
                {"source": e[0], "target": e[1]} if isinstance(e, tuple) else e
                for e in edges
            ]
        except Exception:
            return []
    
    def _extract_tools(self, compiled: CompiledGraph) -> List[Dict[str, Any]]:
        """Extract tools used by the graph."""
        tools: List[Dict[str, Any]] = []
        
        try:
            # Iterate through nodes to find tool calls
            nodes = getattr(compiled, "nodes", {})
            for node_name, node_func in nodes.items():
                # Check if node uses tools
                node_tools = getattr(node_func, "tools", None)
                if node_tools:
                    for tool in node_tools:
                        tools.append({
                            "name": getattr(tool, "name", str(tool)),
                            "description": getattr(tool, "description", None),
                            "type": "langgraph_tool",
                        })
        except Exception:
            pass
        
        return tools
    
    def _infer_tool_permissions(self, tool: Dict[str, Any]) -> List[str]:
        """Infer required permissions for a tool."""
        permissions: List[str] = []
        name = tool.get("name", "").lower()
        
        # Heuristic permission inference based on tool name
        if any(kw in name for kw in ["web", "http", "api", "fetch", "search"]):
            permissions.append("network:external")
        if any(kw in name for kw in ["file", "read", "write", "disk"]):
            permissions.append("filesystem:read")
        if any(kw in name for kw in ["exec", "shell", "command", "subprocess"]):
            permissions.append("process:execute")
        if any(kw in name for kw in ["sql", "database", "db", "query"]):
            permissions.append("database:query")
        
        return permissions
    
    def _type_to_json_schema(self, type_hint: Type) -> Dict[str, Any]:
        """Convert a Python type hint to JSON schema."""
        try:
            # Handle Pydantic models
            if hasattr(type_hint, "model_json_schema"):
                return type_hint.model_json_schema()
            
            # Handle TypedDict
            if hasattr(type_hint, "__annotations__"):
                properties = {}
                required = []
                
                for field_name, field_type in type_hint.__annotations__.items():
                    properties[field_name] = self._python_type_to_json(field_type)
                    # Check if field is required (not Optional)
                    if not self._is_optional(field_type):
                        required.append(field_name)
                
                return {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
        except Exception:
            pass
        
        return {"type": "object"}
    
    def _python_type_to_json(self, python_type: Type) -> Dict[str, Any]:
        """Convert a Python type to JSON schema type."""
        type_map = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"},
        }
        
        # Handle basic types
        if python_type in type_map:
            return type_map[python_type]
        
        # Handle Optional
        origin = getattr(python_type, "__origin__", None)
        if origin is Union:
            args = getattr(python_type, "__args__", ())
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return self._python_type_to_json(non_none[0])
        
        # Handle List[T]
        if origin is list:
            args = getattr(python_type, "__args__", ())
            if args:
                return {
                    "type": "array",
                    "items": self._python_type_to_json(args[0]),
                }
            return {"type": "array"}
        
        # Handle Dict[K, V]
        if origin is dict:
            return {"type": "object"}
        
        # Default
        return {"type": "string"}
    
    def _is_optional(self, type_hint: Type) -> bool:
        """Check if a type hint is Optional."""
        origin = getattr(type_hint, "__origin__", None)
        if origin is Union:
            args = getattr(type_hint, "__args__", ())
            return type(None) in args
        return False
    
    def _dict_to_schema(self, schema_dict: Dict[str, Any]) -> Schema:
        """Convert a JSON schema dict to a Schema object."""
        properties = {}
        required = schema_dict.get("required", [])
        
        for prop_name, prop_schema in schema_dict.get("properties", {}).items():
            properties[prop_name] = SchemaProperty(
                name=prop_name,
                type=prop_schema.get("type", "string"),
                description=prop_schema.get("description"),
                required=prop_name in required,
                default=prop_schema.get("default"),
                enum=prop_schema.get("enum"),
                items=prop_schema.get("items"),
            )
        
        return Schema(
            type=schema_dict.get("type", "object"),
            properties=properties,
            required=required,
            description=schema_dict.get("description"),
        )
    
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


# Import Union for type checking
from typing import Union
