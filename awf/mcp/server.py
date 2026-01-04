"""
AI Workflow Fabric - MCP Server Implementation

This module implements an MCP (Model Context Protocol) server that allows
AI assistants to interact with AWF for agent orchestration.

Usage:
    # Start standalone MCP server
    python -m awf.mcp.server

    # Or import and use programmatically
    from awf.mcp.server import AWFMCPServer
    server = AWFMCPServer()
    server.run()
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence
from uuid import uuid4

# AWF imports
from awf.core.types import (
    AgentManifest,
    AgentStatus,
    Capability,
    CapabilityType,
)
from awf.registry.memory import InMemoryRegistry
from awf.security.trust import TrustScoringEngine
from awf.security.policy import PolicyEngine
from awf.orchestration.types import (
    WorkflowDefinition,
    StepDefinition,
    RetryPolicy,
    WorkflowResult,
    ExecutionStatus,
)
from awf.orchestration.orchestrator import Orchestrator, OrchestratorConfig
from awf.orchestration.registry import OrchestrationAdapterRegistry


# =============================================================================
# MCP Protocol Types
# =============================================================================


@dataclass
class MCPTool:
    """Definition of an MCP tool."""
    
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Any] = field(repr=False)


@dataclass
class MCPResource:
    """Definition of an MCP resource."""
    
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"


@dataclass
class MCPPrompt:
    """Definition of an MCP prompt template."""
    
    name: str
    description: str
    arguments: List[Dict[str, Any]]


# =============================================================================
# AWF MCP Server
# =============================================================================


class AWFMCPServer:
    """
    MCP Server for AI Workflow Fabric.
    
    Provides tools, resources, and prompts for AI assistants to:
    - Register and manage agents
    - Create and execute workflows
    - Query trust scores and policies
    - Monitor workflow executions
    
    Example usage with Claude/OpenCode:
    ```
    # In your MCP configuration
    {
        "mcpServers": {
            "awf": {
                "command": "python",
                "args": ["-m", "awf.mcp.server"]
            }
        }
    }
    ```
    """
    
    def __init__(
        self,
        awf_api_url: str = "http://localhost:8000",
        use_local: bool = True,
    ):
        """
        Initialize the MCP server.
        
        Args:
            awf_api_url: URL of the AWF API server (used when use_local=False)
            use_local: If True, use in-memory components instead of API calls
        """
        self.awf_api_url = awf_api_url
        self.use_local = use_local
        
        # Local components (when use_local=True)
        if use_local:
            self.registry = InMemoryRegistry()
            self.trust_engine = TrustScoringEngine()
            self.policy_engine = PolicyEngine()
            self.adapter_registry = OrchestrationAdapterRegistry()
            self.orchestrator = Orchestrator(
                adapter_registry=self.adapter_registry,
                agent_registry=self.registry,
                config=OrchestratorConfig(
                    default_timeout_ms=300000,
                    max_parallel_steps=10,
                    emit_events=True,
                ),
            )
            self.workflows: Dict[str, WorkflowDefinition] = {}
            self.executions: Dict[str, WorkflowResult] = {}
        
        # Register tools
        self._tools: Dict[str, MCPTool] = {}
        self._register_tools()
        
        # Register resources
        self._resources: Dict[str, MCPResource] = {}
        self._register_resources()
        
        # Register prompts
        self._prompts: Dict[str, MCPPrompt] = {}
        self._register_prompts()
    
    # -------------------------------------------------------------------------
    # Tool Registration
    # -------------------------------------------------------------------------
    
    def _register_tools(self) -> None:
        """Register all MCP tools."""
        
        # Agent Management Tools
        self._register_tool(MCPTool(
            name="awf_register_agent",
            description=(
                "Register a new AI agent with AWF. The agent will be assigned "
                "a trust score and can then be used in workflows."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique identifier for the agent"
                    },
                    "name": {
                        "type": "string",
                        "description": "Human-readable name for the agent"
                    },
                    "version": {
                        "type": "string",
                        "description": "Version string (e.g., '1.0.0')",
                        "default": "1.0.0"
                    },
                    "framework": {
                        "type": "string",
                        "description": "Framework used (langgraph, crewai, autogen, openai, custom)",
                        "enum": ["langgraph", "crewai", "autogen", "openai", "custom"]
                    },
                    "capabilities": {
                        "type": "array",
                        "description": "List of agent capabilities",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {
                                    "type": "string",
                                    "enum": ["tool", "reasoning", "memory", "planning", "coding", "communication"]
                                },
                                "description": {"type": "string"}
                            },
                            "required": ["name", "type"]
                        }
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of what the agent does"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization"
                    }
                },
                "required": ["id", "name", "framework", "capabilities"]
            },
            handler=self._handle_register_agent,
        ))
        
        self._register_tool(MCPTool(
            name="awf_list_agents",
            description=(
                "List all registered agents, optionally filtered by capability, "
                "framework, or minimum trust score."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "capability": {
                        "type": "string",
                        "description": "Filter by capability name"
                    },
                    "framework": {
                        "type": "string",
                        "description": "Filter by framework"
                    },
                    "min_trust_score": {
                        "type": "number",
                        "description": "Minimum trust score (0.0-1.0)",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                }
            },
            handler=self._handle_list_agents,
        ))
        
        self._register_tool(MCPTool(
            name="awf_get_agent",
            description="Get details of a specific agent by ID.",
            input_schema={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The agent ID to retrieve"
                    }
                },
                "required": ["agent_id"]
            },
            handler=self._handle_get_agent,
        ))
        
        self._register_tool(MCPTool(
            name="awf_get_trust_score",
            description=(
                "Get the trust score for an agent. Trust scores determine "
                "sandbox restrictions and execution permissions."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The agent ID to check"
                    }
                },
                "required": ["agent_id"]
            },
            handler=self._handle_get_trust_score,
        ))
        
        self._register_tool(MCPTool(
            name="awf_delete_agent",
            description="Remove an agent from the registry.",
            input_schema={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The agent ID to delete"
                    }
                },
                "required": ["agent_id"]
            },
            handler=self._handle_delete_agent,
        ))
        
        # Workflow Management Tools
        self._register_tool(MCPTool(
            name="awf_create_workflow",
            description=(
                "Create a new workflow definition. Workflows chain multiple "
                "agents together with dependencies, retries, and fallbacks."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Unique workflow identifier"
                    },
                    "name": {
                        "type": "string",
                        "description": "Human-readable workflow name"
                    },
                    "description": {
                        "type": "string",
                        "description": "What the workflow does"
                    },
                    "steps": {
                        "type": "array",
                        "description": "Workflow steps",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Step ID"},
                                "agent_id": {"type": "string", "description": "Agent to execute"},
                                "depends_on": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Step IDs this step depends on"
                                },
                                "timeout_ms": {
                                    "type": "integer",
                                    "description": "Timeout in milliseconds"
                                },
                                "retry_max_attempts": {
                                    "type": "integer",
                                    "description": "Max retry attempts"
                                }
                            },
                            "required": ["id", "agent_id"]
                        }
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Overall workflow timeout"
                    }
                },
                "required": ["id", "name", "steps"]
            },
            handler=self._handle_create_workflow,
        ))
        
        self._register_tool(MCPTool(
            name="awf_list_workflows",
            description="List all workflow definitions.",
            input_schema={
                "type": "object",
                "properties": {}
            },
            handler=self._handle_list_workflows,
        ))
        
        self._register_tool(MCPTool(
            name="awf_get_workflow",
            description="Get details of a specific workflow.",
            input_schema={
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "description": "The workflow ID"
                    }
                },
                "required": ["workflow_id"]
            },
            handler=self._handle_get_workflow,
        ))
        
        self._register_tool(MCPTool(
            name="awf_execute_workflow",
            description=(
                "Execute a workflow with the given input data. Returns "
                "execution results including step outputs and any errors."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "description": "The workflow to execute"
                    },
                    "input": {
                        "type": "object",
                        "description": "Input data for the workflow"
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context variables"
                    }
                },
                "required": ["workflow_id", "input"]
            },
            handler=self._handle_execute_workflow,
        ))
        
        self._register_tool(MCPTool(
            name="awf_get_execution",
            description="Get the status and results of a workflow execution.",
            input_schema={
                "type": "object",
                "properties": {
                    "execution_id": {
                        "type": "string",
                        "description": "The execution ID"
                    }
                },
                "required": ["execution_id"]
            },
            handler=self._handle_get_execution,
        ))
        
        self._register_tool(MCPTool(
            name="awf_delete_workflow",
            description="Delete a workflow definition.",
            input_schema={
                "type": "object",
                "properties": {
                    "workflow_id": {
                        "type": "string",
                        "description": "The workflow ID to delete"
                    }
                },
                "required": ["workflow_id"]
            },
            handler=self._handle_delete_workflow,
        ))
    
    def _register_tool(self, tool: MCPTool) -> None:
        """Register a single tool."""
        self._tools[tool.name] = tool
    
    # -------------------------------------------------------------------------
    # Resource Registration
    # -------------------------------------------------------------------------
    
    def _register_resources(self) -> None:
        """Register all MCP resources."""
        
        self._resources["awf://agents"] = MCPResource(
            uri="awf://agents",
            name="Agent Registry",
            description="List of all registered agents with their capabilities and trust scores",
        )
        
        self._resources["awf://workflows"] = MCPResource(
            uri="awf://workflows",
            name="Workflow Definitions",
            description="List of all workflow definitions",
        )
        
        self._resources["awf://executions"] = MCPResource(
            uri="awf://executions",
            name="Recent Executions",
            description="Recent workflow executions and their status",
        )
        
        self._resources["awf://health"] = MCPResource(
            uri="awf://health",
            name="System Health",
            description="AWF system health and status information",
        )
    
    # -------------------------------------------------------------------------
    # Prompt Registration
    # -------------------------------------------------------------------------
    
    def _register_prompts(self) -> None:
        """Register all MCP prompts."""
        
        self._prompts["create_research_workflow"] = MCPPrompt(
            name="create_research_workflow",
            description="Create a multi-agent research workflow",
            arguments=[
                {
                    "name": "topic",
                    "description": "The research topic",
                    "required": True,
                },
                {
                    "name": "depth",
                    "description": "Research depth (quick, standard, deep)",
                    "required": False,
                }
            ],
        )
        
        self._prompts["debug_workflow"] = MCPPrompt(
            name="debug_workflow",
            description="Debug a failed workflow execution",
            arguments=[
                {
                    "name": "execution_id",
                    "description": "The failed execution ID",
                    "required": True,
                }
            ],
        )
    
    # -------------------------------------------------------------------------
    # Tool Handlers
    # -------------------------------------------------------------------------
    
    async def _handle_register_agent(self, **kwargs) -> Dict[str, Any]:
        """Handle agent registration."""
        capabilities = [
            Capability(
                name=cap["name"],
                type=CapabilityType(cap["type"]),
                description=cap.get("description", ""),
            )
            for cap in kwargs.get("capabilities", [])
        ]
        
        manifest = AgentManifest(
            id=kwargs["id"],
            name=kwargs["name"],
            version=kwargs.get("version", "1.0.0"),
            framework=kwargs["framework"],
            capabilities=capabilities,
            description=kwargs.get("description"),
            tags=kwargs.get("tags", []),
            status=AgentStatus.ACTIVE,
        )
        
        # Compute trust score
        trust = await self.trust_engine.compute_score(manifest)
        manifest.trust_score = trust.score
        
        # Register
        await self.registry.register(manifest)
        
        return {
            "success": True,
            "agent_id": manifest.id,
            "trust_score": trust.score,
            "sandbox_tier": trust.sandbox_tier.value,
            "message": f"Agent '{manifest.name}' registered successfully",
        }
    
    async def _handle_list_agents(self, **kwargs) -> Dict[str, Any]:
        """Handle listing agents."""
        capabilities = [kwargs["capability"]] if kwargs.get("capability") else None
        
        agents = await self.registry.search(
            capabilities=capabilities,
            framework=kwargs.get("framework"),
            min_trust_score=kwargs.get("min_trust_score"),
        )
        
        return {
            "agents": [
                {
                    "id": a.id,
                    "name": a.name,
                    "framework": a.framework,
                    "trust_score": a.trust_score,
                    "status": a.status.value,
                    "capabilities": [c.name for c in a.capabilities],
                }
                for a in agents
            ],
            "total": len(agents),
        }
    
    async def _handle_get_agent(self, **kwargs) -> Dict[str, Any]:
        """Handle getting agent details."""
        agent = await self.registry.get(kwargs["agent_id"])
        
        if agent is None:
            return {"error": f"Agent '{kwargs['agent_id']}' not found"}
        
        return {
            "id": agent.id,
            "name": agent.name,
            "version": agent.version,
            "framework": agent.framework,
            "description": agent.description,
            "trust_score": agent.trust_score,
            "status": agent.status.value,
            "capabilities": [
                {"name": c.name, "type": c.type.value, "description": c.description}
                for c in agent.capabilities
            ],
            "tags": agent.tags,
            "registered_at": agent.registered_at.isoformat() if agent.registered_at else None,
        }
    
    async def _handle_get_trust_score(self, **kwargs) -> Dict[str, Any]:
        """Handle getting trust score."""
        agent = await self.registry.get(kwargs["agent_id"])
        
        if agent is None:
            return {"error": f"Agent '{kwargs['agent_id']}' not found"}
        
        trust = await self.trust_engine.compute_score(agent)
        
        return {
            "agent_id": agent.id,
            "score": trust.score,
            "sandbox_tier": trust.sandbox_tier.value,
            "factors": {
                "publisher_trust": trust.factors.publisher_trust,
                "audit_status": trust.factors.audit_status,
                "community_trust": trust.factors.community_trust,
                "permission_analysis": trust.factors.permission_analysis,
                "historical_behavior": trust.factors.historical_behavior,
            },
            "computed_at": trust.computed_at.isoformat(),
        }
    
    async def _handle_delete_agent(self, **kwargs) -> Dict[str, Any]:
        """Handle agent deletion."""
        deleted = await self.registry.delete(kwargs["agent_id"])
        
        if not deleted:
            return {"error": f"Agent '{kwargs['agent_id']}' not found"}
        
        return {
            "success": True,
            "message": f"Agent '{kwargs['agent_id']}' deleted",
        }
    
    async def _handle_create_workflow(self, **kwargs) -> Dict[str, Any]:
        """Handle workflow creation."""
        steps = []
        for step_data in kwargs["steps"]:
            retry = None
            if step_data.get("retry_max_attempts"):
                retry = RetryPolicy(max_attempts=step_data["retry_max_attempts"])
            
            steps.append(StepDefinition(
                id=step_data["id"],
                agent_id=step_data["agent_id"],
                depends_on=step_data.get("depends_on", []),
                timeout_ms=step_data.get("timeout_ms"),
                retry=retry,
            ))
        
        workflow = WorkflowDefinition(
            id=kwargs["id"],
            name=kwargs["name"],
            description=kwargs.get("description"),
            steps=steps,
            timeout_ms=kwargs.get("timeout_ms"),
        )
        
        self.workflows[workflow.id] = workflow
        
        return {
            "success": True,
            "workflow_id": workflow.id,
            "step_count": len(steps),
            "message": f"Workflow '{workflow.name}' created",
        }
    
    async def _handle_list_workflows(self, **kwargs) -> Dict[str, Any]:
        """Handle listing workflows."""
        return {
            "workflows": [
                {
                    "id": w.id,
                    "name": w.name,
                    "description": w.description,
                    "step_count": len(w.steps),
                    "created_at": w.created_at.isoformat(),
                }
                for w in self.workflows.values()
            ],
            "total": len(self.workflows),
        }
    
    async def _handle_get_workflow(self, **kwargs) -> Dict[str, Any]:
        """Handle getting workflow details."""
        workflow = self.workflows.get(kwargs["workflow_id"])
        
        if workflow is None:
            return {"error": f"Workflow '{kwargs['workflow_id']}' not found"}
        
        return workflow.to_dict()
    
    async def _handle_execute_workflow(self, **kwargs) -> Dict[str, Any]:
        """Handle workflow execution."""
        workflow = self.workflows.get(kwargs["workflow_id"])
        
        if workflow is None:
            return {"error": f"Workflow '{kwargs['workflow_id']}' not found"}
        
        try:
            result = await self.orchestrator.execute(
                workflow=workflow,
                input_data=kwargs.get("input", {}),
                context=kwargs.get("context", {}),
            )
            
            self.executions[result.execution_id] = result
            
            return {
                "execution_id": result.execution_id,
                "status": result.status.value,
                "output": result.output,
                "error": result.error,
                "step_results": {
                    step_id: {
                        "status": sr.status.value,
                        "output": sr.output,
                        "error": sr.error,
                    }
                    for step_id, sr in result.step_results.items()
                },
                "total_execution_time_ms": result.total_execution_time_ms,
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _handle_get_execution(self, **kwargs) -> Dict[str, Any]:
        """Handle getting execution details."""
        result = self.executions.get(kwargs["execution_id"])
        
        if result is None:
            return {"error": f"Execution '{kwargs['execution_id']}' not found"}
        
        return result.to_dict()
    
    async def _handle_delete_workflow(self, **kwargs) -> Dict[str, Any]:
        """Handle workflow deletion."""
        if kwargs["workflow_id"] not in self.workflows:
            return {"error": f"Workflow '{kwargs['workflow_id']}' not found"}
        
        del self.workflows[kwargs["workflow_id"]]
        
        return {
            "success": True,
            "message": f"Workflow '{kwargs['workflow_id']}' deleted",
        }
    
    # -------------------------------------------------------------------------
    # MCP Protocol Implementation
    # -------------------------------------------------------------------------
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an incoming MCP request.
        
        Args:
            request: The MCP request object
            
        Returns:
            MCP response object
        """
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "tools/list":
                result = await self._handle_tools_list()
            elif method == "tools/call":
                result = await self._handle_tools_call(params)
            elif method == "resources/list":
                result = await self._handle_resources_list()
            elif method == "resources/read":
                result = await self._handle_resources_read(params)
            elif method == "prompts/list":
                result = await self._handle_prompts_list()
            elif method == "prompts/get":
                result = await self._handle_prompts_get(params)
            else:
                return self._error_response(request_id, -32601, f"Method not found: {method}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            }
            
        except Exception as e:
            return self._error_response(request_id, -32603, str(e))
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": {
                "name": "awf-mcp-server",
                "version": "1.0.0",
            },
        }
    
    async def _handle_tools_list(self) -> Dict[str, Any]:
        """Handle tools/list."""
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
                for tool in self._tools.values()
            ]
        }
    
    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool = self._tools[tool_name]
        result = await tool.handler(**arguments)
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, indent=2, default=str),
                }
            ],
            "isError": "error" in result,
        }
    
    async def _handle_resources_list(self) -> Dict[str, Any]:
        """Handle resources/list."""
        return {
            "resources": [
                {
                    "uri": res.uri,
                    "name": res.name,
                    "description": res.description,
                    "mimeType": res.mime_type,
                }
                for res in self._resources.values()
            ]
        }
    
    async def _handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read."""
        uri = params.get("uri")
        
        if uri == "awf://agents":
            agents = await self.registry.search()
            content = json.dumps([
                {"id": a.id, "name": a.name, "trust_score": a.trust_score}
                for a in agents
            ], indent=2)
        elif uri == "awf://workflows":
            content = json.dumps([
                {"id": w.id, "name": w.name, "steps": len(w.steps)}
                for w in self.workflows.values()
            ], indent=2)
        elif uri == "awf://executions":
            content = json.dumps([
                {"id": e.execution_id, "status": e.status.value}
                for e in list(self.executions.values())[-10:]
            ], indent=2)
        elif uri == "awf://health":
            count = await self.registry.count()
            content = json.dumps({
                "status": "healthy",
                "agents": count,
                "workflows": len(self.workflows),
                "executions": len(self.executions),
            }, indent=2)
        else:
            raise ValueError(f"Unknown resource: {uri}")
        
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": content,
                }
            ]
        }
    
    async def _handle_prompts_list(self) -> Dict[str, Any]:
        """Handle prompts/list."""
        return {
            "prompts": [
                {
                    "name": p.name,
                    "description": p.description,
                    "arguments": p.arguments,
                }
                for p in self._prompts.values()
            ]
        }
    
    async def _handle_prompts_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/get."""
        name = params.get("name")
        arguments = params.get("arguments", {})
        
        if name == "create_research_workflow":
            topic = arguments.get("topic", "AI")
            depth = arguments.get("depth", "standard")
            
            return {
                "description": f"Create a research workflow for: {topic}",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"""Create a {depth} research workflow for the topic: "{topic}"

The workflow should include these steps:
1. Search - Find relevant sources
2. Analyze - Extract key information  
3. Synthesize - Combine findings
4. Report - Generate final report

Use the awf_create_workflow tool to create this workflow.""",
                        },
                    }
                ],
            }
        elif name == "debug_workflow":
            execution_id = arguments.get("execution_id")
            
            return {
                "description": f"Debug execution: {execution_id}",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"""Debug the failed workflow execution: {execution_id}

1. Use awf_get_execution to get the execution details
2. Identify which step failed and why
3. Check the agent's trust score
4. Suggest fixes for the failure""",
                        },
                    }
                ],
            }
        
        raise ValueError(f"Unknown prompt: {name}")
    
    def _error_response(
        self,
        request_id: Any,
        code: int,
        message: str,
    ) -> Dict[str, Any]:
        """Create an error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        }
    
    # -------------------------------------------------------------------------
    # Server Runner
    # -------------------------------------------------------------------------
    
    async def run_stdio(self) -> None:
        """Run the MCP server using stdio transport."""
        import sys
        
        while True:
            try:
                # Read line from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    break
                
                # Parse JSON-RPC request
                try:
                    request = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Handle request
                response = await self.handle_request(request)
                
                # Write response to stdout
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                
            except Exception as e:
                sys.stderr.write(f"Error: {e}\n")
                sys.stderr.flush()
    
    def run(self) -> None:
        """Run the MCP server."""
        asyncio.run(self.run_stdio())


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """Main entry point for the MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AI Workflow Fabric MCP Server"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="AWF API URL (when not using local mode)",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Use remote AWF API instead of local components",
    )
    
    args = parser.parse_args()
    
    server = AWFMCPServer(
        awf_api_url=args.api_url,
        use_local=not args.remote,
    )
    server.run()


if __name__ == "__main__":
    main()
