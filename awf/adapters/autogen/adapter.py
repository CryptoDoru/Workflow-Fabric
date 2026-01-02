"""
AI Workflow Fabric - AutoGen Adapter

This module provides the adapter for integrating Microsoft AutoGen agents with AWF.
It handles registration, execution, and event bridging for AutoGen ConversableAgent,
AssistantAgent, UserProxyAgent, and GroupChat objects.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime
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
    Task,
    TaskError,
    TaskMetrics,
    TaskResult,
    TaskStatus,
)

# Type hints for AutoGen (optional import)
try:
    from autogen import (
        ConversableAgent,
        AssistantAgent,
        UserProxyAgent,
        GroupChat,
        GroupChatManager,
    )
    
    AUTOGEN_AVAILABLE = True
except ImportError:
    # Fallback for when autogen is not installed
    ConversableAgent = Any
    AssistantAgent = Any
    UserProxyAgent = Any
    GroupChat = Any
    GroupChatManager = Any
    AUTOGEN_AVAILABLE = False


# Type alias for AutoGen agent types
AutoGenAgent = Union[ConversableAgent, GroupChat, GroupChatManager]


class AutoGenAdapter(BaseAdapter):
    """
    Adapter for Microsoft AutoGen agents and group chats.
    
    This adapter translates between AutoGen's conversation-based API and
    the Agent State Protocol (ASP).
    
    AutoGen uses a conversation-centric model where agents communicate through
    messages. This adapter supports:
    
    - Single agents (AssistantAgent, UserProxyAgent, ConversableAgent)
    - Group chats (GroupChat with GroupChatManager)
    - Function calling and code execution
    
    Example usage:
        ```python
        from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
        from awf.adapters.autogen import AutoGenAdapter
        
        # Create AutoGen agents
        assistant = AssistantAgent(
            name="assistant",
            llm_config={"model": "gpt-4"},
        )
        user_proxy = UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            code_execution_config={"work_dir": "coding"},
        )
        
        # Register single agent with AWF
        adapter = AutoGenAdapter()
        manifest = adapter.register(assistant, agent_id="my-assistant")
        
        # Or register a group chat
        group_chat = GroupChat(
            agents=[assistant, user_proxy],
            messages=[],
            max_round=10,
        )
        manager = GroupChatManager(groupchat=group_chat)
        group_manifest = adapter.register(manager, agent_id="my-group-chat")
        
        # Execute via ASP
        task = Task(agent_id="my-assistant", input={"message": "Write a Python function"})
        result = await adapter.execute(task)
        ```
    """
    
    framework_name = "autogen"
    framework_version: Optional[str] = None
    
    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        trust_scorer: Optional[TrustScorer] = None,
    ):
        """
        Initialize the AutoGen adapter.
        
        Args:
            registry: Optional agent registry for storing manifests
            trust_scorer: Optional trust scorer for computing trust scores
        """
        super().__init__(registry, trust_scorer)
        
        if not AUTOGEN_AVAILABLE:
            raise ImportError(
                "AutoGen is not installed. "
                "Install it with: pip install pyautogen"
            )
        
        # Try to get AutoGen version
        try:
            import autogen
            self.framework_version = getattr(autogen, "__version__", None)
        except Exception:
            pass
        
        # Store manifests and agents
        self._manifests: Dict[str, AgentManifest] = {}
        self._agents: Dict[str, AutoGenAgent] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._conversation_history: Dict[str, List[Dict[str, Any]]] = {}
    
    # =========================================================================
    # Registration Interface
    # =========================================================================
    
    def register(
        self,
        agent: AutoGenAgent,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentManifest:
        """
        Register an AutoGen agent or group chat with AWF.
        
        Args:
            agent: An AutoGen ConversableAgent, GroupChat, or GroupChatManager
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
        
        # Determine agent type
        agent_type = self._get_agent_type(agent)
        
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
                "type": agent_type,
                "name": self._get_native_name(agent),
                "system_message": self._get_system_message(agent),
                "llm_config": self._get_llm_config(agent),
                "agents": self._get_group_agents(agent) if agent_type in ("group_chat", "group_chat_manager") else None,
                "max_round": self._get_max_round(agent) if agent_type in ("group_chat", "group_chat_manager") else None,
                "human_input_mode": self._get_human_input_mode(agent),
                "code_execution_enabled": self._has_code_execution(agent),
                **(metadata.get("extra", {}) if metadata else {}),
            },
        )
        
        # Store the agent and manifest
        self._agents[agent_id] = agent
        self._manifests[agent_id] = manifest
        self._registered_agents[agent_id] = agent
        self._conversation_history[agent_id] = []
        
        return manifest
    
    def unregister(self, agent_id: str) -> bool:
        """Remove an agent from the registry."""
        if agent_id in self._manifests:
            del self._manifests[agent_id]
            del self._agents[agent_id]
            del self._registered_agents[agent_id]
            if agent_id in self._conversation_history:
                del self._conversation_history[agent_id]
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
    
    def extract_capabilities(self, agent: AutoGenAgent) -> List[Capability]:
        """
        Extract capabilities from an AutoGen agent or group chat.
        
        For single agents: extracts LLM and function capabilities
        For group chats: extracts combined capabilities from all agents
        """
        capabilities: List[Capability] = []
        agent_type = self._get_agent_type(agent)
        
        if agent_type == "group_chat_manager":
            # Extract from all agents in the group chat
            group_chat = self._get_group_chat(agent)
            if group_chat:
                group_agents = getattr(group_chat, "agents", [])
                for group_agent in group_agents:
                    agent_caps = self._extract_single_agent_capabilities(group_agent)
                    capabilities.extend(agent_caps)
                
                # Add orchestration capability
                capabilities.append(Capability(
                    name="group_orchestration",
                    type=CapabilityType.REASONING,
                    description="Orchestrates multi-agent conversations",
                    metadata={
                        "agent_count": len(group_agents),
                        "max_round": getattr(group_chat, "max_round", None),
                    },
                ))
        elif agent_type == "group_chat":
            # Extract from all agents in the group chat
            group_agents = getattr(agent, "agents", [])
            for group_agent in group_agents:
                agent_caps = self._extract_single_agent_capabilities(group_agent)
                capabilities.extend(agent_caps)
        else:
            capabilities = self._extract_single_agent_capabilities(agent)
        
        return capabilities
    
    def _extract_single_agent_capabilities(self, agent: ConversableAgent) -> List[Capability]:
        """Extract capabilities from a single AutoGen agent."""
        capabilities: List[Capability] = []
        
        # Add LLM capability if configured
        llm_config = self._get_llm_config(agent)
        if llm_config:
            model = llm_config.get("model", "unknown")
            capabilities.append(Capability(
                name="llm_reasoning",
                type=CapabilityType.REASONING,
                description=f"LLM-powered reasoning using {model}",
                metadata={
                    "model": model,
                    "config": llm_config,
                },
            ))
        
        # Extract function/tool capabilities
        functions = self._get_functions(agent)
        for func in functions:
            func_name = func.get("name", "unknown_function")
            capabilities.append(Capability(
                name=func_name,
                type=CapabilityType.TOOL,
                description=func.get("description"),
                permissions=self._infer_function_permissions(func),
                metadata={
                    "parameters": func.get("parameters"),
                    "source": "autogen_function",
                },
            ))
        
        # Check for code execution capability
        if self._has_code_execution(agent):
            capabilities.append(Capability(
                name="code_execution",
                type=CapabilityType.TOOL,
                description="Execute Python code in a sandboxed environment",
                permissions=["process:execute", "filesystem:write"],
                metadata={
                    "work_dir": self._get_code_execution_config(agent).get("work_dir"),
                },
            ))
        
        # Check for human input capability
        human_input_mode = self._get_human_input_mode(agent)
        if human_input_mode and human_input_mode != "NEVER":
            capabilities.append(Capability(
                name="human_input",
                type=CapabilityType.REASONING,
                description=f"Can request human input (mode: {human_input_mode})",
                metadata={"mode": human_input_mode},
            ))
        
        return capabilities
    
    def infer_input_schema(self, agent: AutoGenAgent) -> Optional[Dict[str, Any]]:
        """
        Infer input schema for an AutoGen agent.
        
        AutoGen uses message-based communication, so the schema is flexible.
        """
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to send to the agent",
                },
                "context": {
                    "type": "object",
                    "description": "Additional context for the conversation",
                },
                "clear_history": {
                    "type": "boolean",
                    "description": "Whether to clear conversation history before this message",
                    "default": False,
                },
            },
            "required": ["message"],
        }
    
    def infer_output_schema(self, agent: AutoGenAgent) -> Optional[Dict[str, Any]]:
        """
        Infer output schema for an AutoGen agent.
        """
        return {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "The agent's response message",
                },
                "conversation_history": {
                    "type": "array",
                    "description": "The full conversation history",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "content": {"type": "string"},
                            "name": {"type": "string"},
                        },
                    },
                },
                "cost": {
                    "type": "number",
                    "description": "Total cost of the conversation",
                },
            },
        }
    
    # =========================================================================
    # Execution Interface
    # =========================================================================
    
    async def execute(self, task: Task) -> TaskResult:
        """
        Execute a task on a registered AutoGen agent.
        
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
        
        started_at = datetime.utcnow()
        start_time = time.perf_counter()
        
        try:
            # Execute based on agent type
            loop = asyncio.get_event_loop()
            agent_type = self._get_agent_type(agent)
            
            if agent_type == "group_chat_manager":
                output = await self._execute_group_chat(agent, task, loop)
            else:
                output = await self._execute_single_agent(agent, task, loop)
            
            execution_time = int((time.perf_counter() - start_time) * 1000)
            
            return TaskResult(
                task_id=task.id,
                agent_id=task.agent_id,
                status=TaskStatus.COMPLETED,
                output=output,
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
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )
    
    async def _execute_single_agent(
        self, agent: ConversableAgent, task: Task, loop: asyncio.AbstractEventLoop
    ) -> Dict[str, Any]:
        """Execute a task on a single AutoGen agent."""
        # Extract message from input
        message = task.input.get("message", str(task.input))
        clear_history = task.input.get("clear_history", False)
        
        # Clear history if requested
        if clear_history and task.agent_id in self._conversation_history:
            self._conversation_history[task.agent_id] = []
        
        # Create a temporary UserProxyAgent for initiating conversation if needed
        # AutoGen requires two agents for conversation
        def run_conversation():
            # For single agent, we create a simple reply
            # Using generate_reply which is the core method
            if hasattr(agent, "generate_reply"):
                messages = [{"role": "user", "content": message}]
                reply = agent.generate_reply(messages=messages)
                return reply
            else:
                # Fallback for agents without generate_reply
                return {"response": "Agent does not support direct replies"}
        
        # Handle timeout
        if task.timeout_ms:
            timeout = task.timeout_ms / 1000.0
            result = await asyncio.wait_for(
                loop.run_in_executor(None, run_conversation),
                timeout=timeout,
            )
        else:
            result = await loop.run_in_executor(None, run_conversation)
        
        # Store conversation history
        self._conversation_history.setdefault(task.agent_id, []).append({
            "role": "user",
            "content": message,
        })
        self._conversation_history[task.agent_id].append({
            "role": "assistant",
            "content": str(result) if result else "",
            "name": self._get_native_name(agent),
        })
        
        return {
            "response": str(result) if result else "",
            "conversation_history": self._conversation_history.get(task.agent_id, []),
        }
    
    async def _execute_group_chat(
        self, manager: GroupChatManager, task: Task, loop: asyncio.AbstractEventLoop
    ) -> Dict[str, Any]:
        """Execute a task on a GroupChatManager."""
        message = task.input.get("message", str(task.input))
        clear_history = task.input.get("clear_history", False)
        
        # Get group chat
        group_chat = self._get_group_chat(manager)
        if not group_chat:
            raise ExecutionError("GroupChatManager has no associated GroupChat")
        
        # Clear history if requested
        if clear_history:
            group_chat.messages = []
        
        def run_group_chat():
            # Get the first agent to initiate the conversation
            agents = getattr(group_chat, "agents", [])
            if not agents:
                raise ExecutionError("GroupChat has no agents")
            
            # Use the first agent as initiator
            initiator = agents[0]
            
            # Initiate chat
            result = initiator.initiate_chat(
                manager,
                message=message,
                clear_history=clear_history,
            )
            
            return result
        
        # Handle timeout
        if task.timeout_ms:
            timeout = task.timeout_ms / 1000.0
            result = await asyncio.wait_for(
                loop.run_in_executor(None, run_group_chat),
                timeout=timeout,
            )
        else:
            result = await loop.run_in_executor(None, run_group_chat)
        
        # Get conversation history from group chat
        messages = getattr(group_chat, "messages", [])
        
        # Get the last response
        last_response = messages[-1].get("content", "") if messages else ""
        
        return {
            "response": last_response,
            "conversation_history": messages,
            "chat_result": self._serialize_chat_result(result),
        }
    
    async def execute_streaming(self, task: Task) -> AsyncIterator[Event]:
        """
        Execute a task with streaming events.
        
        Note: AutoGen has limited streaming support, so this implementation
        emits events at key conversation points.
        """
        agent = self._agents.get(task.agent_id)
        if agent is None:
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
            loop = asyncio.get_event_loop()
            agent_type = self._get_agent_type(agent)
            
            if agent_type == "group_chat_manager":
                # For group chat, emit events for each agent turn
                group_chat = self._get_group_chat(agent)
                if group_chat:
                    agents = getattr(group_chat, "agents", [])
                    for i, group_agent in enumerate(agents):
                        yield Event(
                            type=EventType.STATE_CHANGED,
                            source=task.agent_id,
                            correlation_id=task.id,
                            trace_id=task.trace_id,
                            data={
                                "state": "agent_turn",
                                "agent_index": i,
                                "agent_name": self._get_native_name(group_agent),
                            },
                        )
                
                output = await self._execute_group_chat(agent, task, loop)
            else:
                output = await self._execute_single_agent(agent, task, loop)
            
            execution_time = int((time.perf_counter() - start_time) * 1000)
            
            # Emit completion event
            yield Event(
                type=EventType.TASK_COMPLETED,
                source=task.agent_id,
                correlation_id=task.id,
                trace_id=task.trace_id,
                data={
                    "output": output,
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
    
    def _validate_agent(self, agent: AutoGenAgent) -> None:
        """Validate that the agent is a valid AutoGen agent."""
        super()._validate_agent(agent)
        
        if not AUTOGEN_AVAILABLE:
            raise ValidationError("AutoGen is not installed")
        
        # Check if it's a valid AutoGen type
        valid_types = (
            ConversableAgent,
            GroupChat,
            GroupChatManager,
        )
        
        if not isinstance(agent, valid_types):
            raise ValidationError(
                f"Expected AutoGen ConversableAgent, GroupChat, or GroupChatManager, "
                f"got {type(agent).__name__}"
            )
    
    def _get_agent_type(self, agent: AutoGenAgent) -> str:
        """Determine the type of AutoGen agent."""
        if isinstance(agent, GroupChatManager):
            return "group_chat_manager"
        elif isinstance(agent, GroupChat):
            return "group_chat"
        elif isinstance(agent, UserProxyAgent):
            return "user_proxy"
        elif isinstance(agent, AssistantAgent):
            return "assistant"
        elif isinstance(agent, ConversableAgent):
            return "conversable"
        return "unknown"
    
    def _generate_agent_id(self, agent: AutoGenAgent) -> str:
        """Generate a deterministic agent ID."""
        agent_type = self._get_agent_type(agent)
        
        if agent_type == "group_chat_manager":
            group_chat = self._get_group_chat(agent)
            if group_chat:
                agents = getattr(group_chat, "agents", [])
                names = sorted([self._get_native_name(a) for a in agents])
                content = f"group:{names}"
            else:
                content = f"group:{id(agent)}"
        elif agent_type == "group_chat":
            agents = getattr(agent, "agents", [])
            names = sorted([self._get_native_name(a) for a in agents])
            content = f"group:{names}"
        else:
            name = self._get_native_name(agent)
            system_message = self._get_system_message(agent) or ""
            content = f"{agent_type}:{name}:{system_message[:100]}"
        
        hash_value = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"autogen-{hash_value}"
    
    def _get_agent_name(
        self, agent: AutoGenAgent, metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Get a display name for the agent."""
        if metadata and "name" in metadata:
            return metadata["name"]
        
        native_name = self._get_native_name(agent)
        if native_name:
            return native_name
        
        agent_type = self._get_agent_type(agent)
        return f"AutoGen {agent_type.replace('_', ' ').title()}"
    
    def _get_native_name(self, agent: AutoGenAgent) -> Optional[str]:
        """Get the native name from an AutoGen agent."""
        return getattr(agent, "name", None)
    
    def _get_agent_description(self, agent: AutoGenAgent) -> Optional[str]:
        """Get description for the agent."""
        agent_type = self._get_agent_type(agent)
        
        if agent_type == "group_chat_manager":
            group_chat = self._get_group_chat(agent)
            if group_chat:
                agents = getattr(group_chat, "agents", [])
                names = [self._get_native_name(a) or "agent" for a in agents]
                return f"Group chat manager orchestrating: {', '.join(names)}"
        elif agent_type == "group_chat":
            agents = getattr(agent, "agents", [])
            names = [self._get_native_name(a) or "agent" for a in agents]
            return f"Group chat with: {', '.join(names)}"
        
        system_message = self._get_system_message(agent)
        if system_message:
            # Return first 200 chars of system message as description
            return system_message[:200] + ("..." if len(system_message) > 200 else "")
        
        return None
    
    def _get_system_message(self, agent: AutoGenAgent) -> Optional[str]:
        """Get the system message from an agent."""
        return getattr(agent, "system_message", None)
    
    def _get_llm_config(self, agent: AutoGenAgent) -> Optional[Dict[str, Any]]:
        """Get the LLM configuration from an agent."""
        llm_config = getattr(agent, "llm_config", None)
        if isinstance(llm_config, dict):
            # Return a safe copy without sensitive data
            safe_config = {
                k: v for k, v in llm_config.items()
                if k not in ("api_key", "api_secret", "api_base")
            }
            return safe_config
        return None
    
    def _get_functions(self, agent: ConversableAgent) -> List[Dict[str, Any]]:
        """Get registered functions from an agent."""
        functions: List[Dict[str, Any]] = []
        
        # Check function_map attribute
        function_map = getattr(agent, "_function_map", {}) or getattr(agent, "function_map", {})
        for func_name, func in function_map.items():
            func_info = {
                "name": func_name,
                "description": getattr(func, "__doc__", None),
            }
            functions.append(func_info)
        
        # Check llm_config for function definitions
        llm_config = self._get_llm_config(agent)
        if llm_config and "functions" in llm_config:
            for func_def in llm_config["functions"]:
                if isinstance(func_def, dict):
                    functions.append(func_def)
        
        return functions
    
    def _has_code_execution(self, agent: AutoGenAgent) -> bool:
        """Check if agent has code execution enabled."""
        code_config = self._get_code_execution_config(agent)
        return bool(code_config)
    
    def _get_code_execution_config(self, agent: AutoGenAgent) -> Dict[str, Any]:
        """Get code execution configuration."""
        config = getattr(agent, "code_execution_config", None)
        if isinstance(config, dict):
            return config
        return {}
    
    def _get_human_input_mode(self, agent: AutoGenAgent) -> Optional[str]:
        """Get human input mode from an agent."""
        return getattr(agent, "human_input_mode", None)
    
    def _get_group_chat(self, agent: AutoGenAgent) -> Optional[GroupChat]:
        """Get GroupChat from a GroupChatManager."""
        if isinstance(agent, GroupChatManager):
            return getattr(agent, "groupchat", None)
        return None
    
    def _get_group_agents(self, agent: AutoGenAgent) -> Optional[List[Dict[str, Any]]]:
        """Get agent info from a group chat."""
        group_chat = None
        
        if isinstance(agent, GroupChatManager):
            group_chat = self._get_group_chat(agent)
        elif isinstance(agent, GroupChat):
            group_chat = agent
        
        if not group_chat:
            return None
        
        agents = getattr(group_chat, "agents", [])
        return [
            {
                "name": self._get_native_name(a),
                "type": self._get_agent_type(a),
                "system_message": (self._get_system_message(a) or "")[:100],
            }
            for a in agents
        ]
    
    def _get_max_round(self, agent: AutoGenAgent) -> Optional[int]:
        """Get max_round from a group chat."""
        group_chat = None
        
        if isinstance(agent, GroupChatManager):
            group_chat = self._get_group_chat(agent)
        elif isinstance(agent, GroupChat):
            group_chat = agent
        
        if group_chat:
            return getattr(group_chat, "max_round", None)
        return None
    
    def _infer_function_permissions(self, func: Dict[str, Any]) -> List[str]:
        """Infer permissions required by a function."""
        permissions: List[str] = []
        
        name = func.get("name", "").lower()
        description = (func.get("description") or "").lower()
        combined = f"{name} {description}"
        
        if any(kw in combined for kw in ["web", "http", "api", "fetch", "request", "url"]):
            permissions.append("network:external")
        if any(kw in combined for kw in ["file", "read", "write", "disk", "path", "save"]):
            permissions.append("filesystem:read")
        if any(kw in combined for kw in ["exec", "shell", "command", "run", "subprocess"]):
            permissions.append("process:execute")
        if any(kw in combined for kw in ["sql", "database", "db", "query", "table"]):
            permissions.append("database:query")
        
        return permissions
    
    def _serialize_chat_result(self, result: Any) -> Optional[Dict[str, Any]]:
        """Serialize a chat result to a dictionary."""
        if result is None:
            return None
        
        if isinstance(result, dict):
            return result
        
        # Try to extract common attributes
        serialized = {}
        
        if hasattr(result, "chat_history"):
            serialized["chat_history"] = result.chat_history
        if hasattr(result, "summary"):
            serialized["summary"] = result.summary
        if hasattr(result, "cost"):
            serialized["cost"] = result.cost
        
        return serialized if serialized else {"raw": str(result)}
    
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
    
    # =========================================================================
    # Conversation Management
    # =========================================================================
    
    def get_conversation_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get the conversation history for an agent.
        
        Args:
            agent_id: The ID of the agent
        
        Returns:
            List of conversation messages
        """
        return self._conversation_history.get(agent_id, [])
    
    def clear_conversation_history(self, agent_id: str) -> bool:
        """
        Clear the conversation history for an agent.
        
        Args:
            agent_id: The ID of the agent
        
        Returns:
            True if history was cleared, False if agent not found
        """
        if agent_id in self._conversation_history:
            self._conversation_history[agent_id] = []
            return True
        return False
