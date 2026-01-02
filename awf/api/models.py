"""
AI Workflow Fabric - Pydantic Models for REST API

This module defines request/response models for the AWF REST API.
All models use Pydantic for validation and serialization.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field, ConfigDict
except ImportError:
    raise ImportError(
        "Pydantic is required for the AWF API. "
        "Install with: pip install awf[api]"
    )


# =============================================================================
# Enums (mirror core types)
# =============================================================================


class AgentStatusEnum(str, Enum):
    """Agent status values."""
    REGISTERED = "registered"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEPRECATED = "deprecated"


class TaskStatusEnum(str, Enum):
    """Task status values."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class CapabilityTypeEnum(str, Enum):
    """Capability type values."""
    TOOL = "tool"
    REASONING = "reasoning"
    MEMORY = "memory"
    COMMUNICATION = "communication"
    CUSTOM = "custom"


class SandboxTierEnum(str, Enum):
    """Sandbox tier values."""
    WASM = "wasm"
    GVISOR = "gvisor"
    GVISOR_STRICT = "gvisor_strict"
    BLOCKED = "blocked"


# =============================================================================
# Capability Models
# =============================================================================


class CapabilityCreate(BaseModel):
    """Request model for creating a capability."""
    
    name: str = Field(..., description="Unique name of the capability")
    type: CapabilityTypeEnum = Field(
        default=CapabilityTypeEnum.CUSTOM,
        description="Type of capability"
    )
    description: Optional[str] = Field(None, description="Human-readable description")
    permissions: List[str] = Field(default_factory=list, description="Required permissions")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for input")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for output")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(use_enum_values=True)


class CapabilityResponse(BaseModel):
    """Response model for a capability."""
    
    name: str
    type: str
    description: Optional[str] = None
    permissions: List[str] = []
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = {}


# =============================================================================
# Agent Models
# =============================================================================


class AgentCreate(BaseModel):
    """Request model for registering an agent."""
    
    id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Human-readable name")
    version: str = Field(..., description="Semantic version string")
    framework: str = Field(..., description="Source framework (e.g., langgraph, crewai)")
    framework_version: Optional[str] = Field(None, description="Framework version")
    description: Optional[str] = Field(None, description="Human-readable description")
    capabilities: List[CapabilityCreate] = Field(
        default_factory=list,
        description="List of agent capabilities"
    )
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for input")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for output")
    publisher: Optional[str] = Field(None, description="Publisher identifier")
    documentation_url: Optional[str] = Field(None, description="URL to documentation")
    source_url: Optional[str] = Field(None, description="URL to source code")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(
        use_enum_values=True,
        json_schema_extra={
            "example": {
                "id": "web-search-agent",
                "name": "Web Search Agent",
                "version": "1.0.0",
                "framework": "langgraph",
                "description": "An agent that searches the web for information",
                "capabilities": [
                    {
                        "name": "web_search",
                        "type": "tool",
                        "description": "Search the web for information"
                    }
                ],
                "tags": ["search", "web", "research"]
            }
        }
    )


class AgentUpdate(BaseModel):
    """Request model for updating an agent."""
    
    name: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None
    capabilities: Optional[List[CapabilityCreate]] = None
    tags: Optional[List[str]] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    status: Optional[AgentStatusEnum] = None
    metadata: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(use_enum_values=True)


class AgentResponse(BaseModel):
    """Response model for an agent."""
    
    id: str
    name: str
    version: str
    framework: str
    framework_version: Optional[str] = None
    status: str
    trust_score: Optional[float] = None
    description: Optional[str] = None
    capabilities: List[CapabilityResponse] = []
    tags: List[str] = []
    publisher: Optional[str] = None
    documentation_url: Optional[str] = None
    source_url: Optional[str] = None
    registered_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = {}
    
    model_config = ConfigDict(from_attributes=True)


class AgentListResponse(BaseModel):
    """Response model for listing agents."""
    
    agents: List[AgentResponse]
    total: int
    page: int = 1
    page_size: int = 50


class AgentSearchQuery(BaseModel):
    """Query parameters for searching agents."""
    
    capabilities: Optional[List[str]] = Field(
        None,
        description="Required capabilities (agent must have ALL)"
    )
    framework: Optional[str] = Field(None, description="Filter by framework")
    tags: Optional[List[str]] = Field(
        None,
        description="Required tags (agent must have ALL)"
    )
    min_trust_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum trust score"
    )
    status: Optional[AgentStatusEnum] = Field(None, description="Filter by status")
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(50, ge=1, le=100, description="Results per page")
    
    model_config = ConfigDict(use_enum_values=True)


# =============================================================================
# Task Models
# =============================================================================


class TaskCreate(BaseModel):
    """Request model for creating a task."""
    
    agent_id: str = Field(..., description="ID of the agent to execute the task")
    input: Dict[str, Any] = Field(..., description="Input data for the agent")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracing")
    timeout_ms: Optional[int] = Field(
        None,
        ge=100,
        le=3600000,
        description="Timeout in milliseconds (max 1 hour)"
    )
    priority: int = Field(0, ge=-10, le=10, description="Task priority (-10 to 10)")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "agent_id": "web-search-agent",
                "input": {"query": "What is AI Workflow Fabric?"},
                "timeout_ms": 30000,
                "priority": 0
            }
        }
    )


class TaskErrorResponse(BaseModel):
    """Response model for task errors."""
    
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    retryable: bool = False


class TaskMetricsResponse(BaseModel):
    """Response model for task execution metrics."""
    
    execution_time_ms: int
    token_usage: Optional[Dict[str, int]] = None
    memory_usage_bytes: Optional[int] = None
    sandbox_tier: Optional[str] = None
    sandbox_overhead_ms: Optional[int] = None


class TaskResponse(BaseModel):
    """Response model for a task result."""
    
    task_id: str
    agent_id: str
    status: str
    output: Optional[Dict[str, Any]] = None
    partial_output: Optional[Dict[str, Any]] = None
    error: Optional[TaskErrorResponse] = None
    metrics: Optional[TaskMetricsResponse] = None
    trace_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = {}
    
    model_config = ConfigDict(from_attributes=True)


class TaskSubmitResponse(BaseModel):
    """Response model for task submission."""
    
    task_id: str
    status: str = "pending"
    message: str = "Task submitted successfully"


# =============================================================================
# Trust Models
# =============================================================================


class TrustFactorsResponse(BaseModel):
    """Response model for trust score breakdown."""
    
    publisher_trust: float = Field(..., ge=0.0, le=1.0)
    audit_status: float = Field(..., ge=0.0, le=1.0)
    community_trust: float = Field(..., ge=0.0, le=1.0)
    permission_analysis: float = Field(..., ge=0.0, le=1.0)
    historical_behavior: float = Field(..., ge=0.0, le=1.0)
    computed_score: float = Field(..., ge=0.0, le=1.0)


class TrustScoreResponse(BaseModel):
    """Response model for trust score."""
    
    agent_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    sandbox_tier: str
    factors: TrustFactorsResponse
    computed_at: datetime
    expires_at: Optional[datetime] = None


# =============================================================================
# Policy Models
# =============================================================================


class PolicyCreate(BaseModel):
    """Request model for creating a policy."""
    
    id: str = Field(..., description="Unique policy identifier")
    name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(None, description="Policy description")
    environments: List[str] = Field(
        default_factory=list,
        description="Environments where policy applies"
    )
    agent_ids: Optional[List[str]] = Field(
        None,
        description="Specific agents (None = all)"
    )
    min_trust_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_trust_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    require_capabilities: List[str] = Field(default_factory=list)
    deny_capabilities: List[str] = Field(default_factory=list)
    max_execution_time_ms: Optional[int] = Field(None, ge=100)
    max_memory_bytes: Optional[int] = Field(None, ge=1024)
    allow_network: bool = True
    allow_filesystem: bool = False
    enabled: bool = True


class PolicyResponse(BaseModel):
    """Response model for a policy."""
    
    id: str
    name: str
    description: Optional[str] = None
    environments: List[str] = []
    agent_ids: Optional[List[str]] = None
    min_trust_score: Optional[float] = None
    max_trust_score: Optional[float] = None
    require_capabilities: List[str] = []
    deny_capabilities: List[str] = []
    max_execution_time_ms: Optional[int] = None
    max_memory_bytes: Optional[int] = None
    allow_network: bool = True
    allow_filesystem: bool = False
    enabled: bool = True
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Health & Status Models
# =============================================================================


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = "healthy"
    version: str
    uptime_seconds: float
    registry_count: int
    components: Dict[str, str] = {}
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "uptime_seconds": 3600.5,
                "registry_count": 42,
                "components": {
                    "registry": "healthy",
                    "executor": "healthy",
                    "trust_engine": "healthy"
                }
            }
        }
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "not_found",
                "message": "Agent with ID 'xyz' not found",
                "request_id": "req_abc123"
            }
        }
    )


# =============================================================================
# Workflow Models
# =============================================================================


class WorkflowStepCreate(BaseModel):
    """Request model for a workflow step."""
    
    id: str = Field(..., description="Step identifier")
    agent_id: str = Field(..., description="Agent to execute")
    input_map: Dict[str, str] = Field(
        default_factory=dict,
        description="JSONPath mappings for input"
    )
    condition: Optional[str] = Field(None, description="Conditional execution expression")
    timeout_ms: Optional[int] = Field(None, ge=100)
    on_error: Optional[str] = Field(
        None,
        pattern="^(fail|continue|fallback)$",
        description="Error handling strategy"
    )


class WorkflowCreate(BaseModel):
    """Request model for creating a workflow."""
    
    id: str = Field(..., description="Unique workflow identifier")
    name: str = Field(..., description="Human-readable name")
    description: Optional[str] = None
    steps: List[WorkflowStepCreate] = Field(..., min_length=1)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    timeout_ms: Optional[int] = Field(None, ge=100)
    max_retries: int = Field(0, ge=0, le=10)


class WorkflowResponse(BaseModel):
    """Response model for a workflow."""
    
    id: str
    name: str
    description: Optional[str] = None
    version: str
    steps: List[Dict[str, Any]]
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    timeout_ms: Optional[int] = None
    max_retries: int = 0
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class WorkflowExecutionResponse(BaseModel):
    """Response model for workflow execution status."""
    
    execution_id: str
    workflow_id: str
    status: str
    current_step: Optional[str] = None
    step_results: Dict[str, TaskResponse] = {}
    output: Optional[Dict[str, Any]] = None
    error: Optional[TaskErrorResponse] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
