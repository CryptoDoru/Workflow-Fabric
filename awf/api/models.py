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


class ExecutionStatusEnum(str, Enum):
    """Workflow execution status values."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class StepStatusEnum(str, Enum):
    """Workflow step status values."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class WorkflowEventTypeEnum(str, Enum):
    """Workflow event type values."""
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_CANCELLED = "workflow.cancelled"
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    STEP_RETRYING = "step.retrying"
    STEP_SKIPPED = "step.skipped"
    STEP_TIMEOUT = "step.timeout"
    STEP_FALLBACK = "step.fallback"


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


class RetryPolicyCreate(BaseModel):
    """Request model for retry policy configuration."""
    
    max_attempts: int = Field(3, ge=1, le=10, description="Maximum retry attempts")
    backoff_ms: int = Field(1000, ge=100, le=60000, description="Initial backoff delay in ms")
    backoff_multiplier: float = Field(2.0, ge=1.0, le=5.0, description="Backoff multiplier")
    max_backoff_ms: int = Field(30000, ge=1000, le=300000, description="Maximum backoff delay")


class FallbackPolicyCreate(BaseModel):
    """Request model for fallback policy configuration."""
    
    skip: bool = Field(False, description="Skip step on failure")
    static_value: Optional[Dict[str, Any]] = Field(None, description="Static fallback value")
    agent_id: Optional[str] = Field(None, description="Fallback agent ID")


class WorkflowStepCreate(BaseModel):
    """Request model for a workflow step."""
    
    id: str = Field(..., description="Step identifier")
    agent_id: str = Field(..., description="Agent to execute")
    input_map: Dict[str, str] = Field(
        default_factory=dict,
        description="JSONPath mappings for input (e.g., {'query': '$.input.topic'})"
    )
    output_map: Optional[Dict[str, str]] = Field(
        None,
        description="JSONPath mappings for output"
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="Step IDs this step depends on"
    )
    condition: Optional[str] = Field(
        None, 
        description="Conditional execution expression (e.g., '$.input.flag == True')"
    )
    timeout_ms: Optional[int] = Field(None, ge=100, le=3600000, description="Step timeout")
    retry: Optional[RetryPolicyCreate] = Field(None, description="Retry policy")
    fallback: Optional[FallbackPolicyCreate] = Field(None, description="Fallback policy")
    metadata: Dict[str, Any] = Field(default_factory=dict)


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


class StepResultResponse(BaseModel):
    """Response model for a workflow step result."""
    
    step_id: str = Field(..., description="Step identifier")
    status: StepStatusEnum = Field(..., description="Step execution status")
    output: Optional[Dict[str, Any]] = Field(None, description="Step output data")
    error: Optional[Dict[str, Any]] = Field(None, description="Error details if failed")
    error_category: Optional[str] = Field(None, description="Error category")
    started_at: Optional[datetime] = Field(None, description="Step start time")
    completed_at: Optional[datetime] = Field(None, description="Step completion time")
    execution_time_ms: Optional[int] = Field(None, description="Execution duration in ms")
    retry_count: int = Field(0, ge=0, description="Number of retries")
    used_fallback: bool = Field(False, description="Whether fallback was used")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(use_enum_values=True)


class WorkflowExecuteRequest(BaseModel):
    """Request model for executing a workflow."""
    
    input: Dict[str, Any] = Field(..., description="Input data for the workflow")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context data"
    )
    timeout_ms: Optional[int] = Field(
        None,
        ge=1000,
        le=3600000,
        description="Execution timeout in ms (max 1 hour)"
    )
    trace_id: Optional[str] = Field(None, description="Trace ID for observability")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for request tracking")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "input": {"topic": "AI safety", "max_length": 1000},
                "context": {"user_id": "user-123"},
                "timeout_ms": 60000
            }
        }
    )


class WorkflowEventResponse(BaseModel):
    """Response model for a workflow event (SSE)."""
    
    type: WorkflowEventTypeEnum = Field(..., description="Event type")
    execution_id: str = Field(..., description="Execution ID")
    workflow_id: str = Field(..., description="Workflow ID")
    timestamp: datetime = Field(..., description="Event timestamp")
    step_id: Optional[str] = Field(None, description="Step ID if step event")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    
    model_config = ConfigDict(use_enum_values=True)


class WorkflowExecutionResponse(BaseModel):
    """Response model for workflow execution status."""
    
    execution_id: str = Field(..., description="Unique execution identifier")
    workflow_id: str = Field(..., description="Workflow definition ID")
    status: ExecutionStatusEnum = Field(..., description="Execution status")
    input: Dict[str, Any] = Field(default_factory=dict, description="Workflow input")
    output: Optional[Dict[str, Any]] = Field(None, description="Workflow output")
    error: Optional[Dict[str, Any]] = Field(None, description="Error details if failed")
    step_results: Dict[str, StepResultResponse] = Field(
        default_factory=dict,
        description="Results for each step"
    )
    started_at: Optional[datetime] = Field(None, description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")
    total_execution_time_ms: Optional[int] = Field(None, description="Total duration in ms")
    total_retry_count: int = Field(0, ge=0, description="Total retries across all steps")
    total_fallback_count: int = Field(0, ge=0, description="Total fallbacks used")
    trace_id: Optional[str] = Field(None, description="Trace ID for observability")
    
    model_config = ConfigDict(use_enum_values=True, from_attributes=True)


# =============================================================================
# Watcher Agent Models
# =============================================================================


class RemediationRiskEnum(str, Enum):
    """Risk level for remediation actions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GrafanaAlertWebhook(BaseModel):
    """Incoming alert webhook payload from Grafana."""
    
    alertId: Optional[str] = Field(None, alias="alertId", description="Alert ID")
    title: str = Field(..., description="Alert title")
    message: Optional[str] = Field(None, description="Alert message")
    state: str = Field(..., description="Alert state (alerting, ok, pending, no_data)")
    labels: Dict[str, str] = Field(default_factory=dict, description="Alert labels")
    annotations: Dict[str, str] = Field(default_factory=dict, description="Alert annotations")
    startsAt: Optional[str] = Field(None, description="Alert start time (ISO 8601)")
    endsAt: Optional[str] = Field(None, description="Alert end time (ISO 8601)")
    generatorURL: Optional[str] = Field(None, description="URL to alert generator")
    fingerprint: Optional[str] = Field(None, description="Alert fingerprint")
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "alertId": "alert-123",
                "title": "High Error Rate",
                "message": "Agent error rate exceeded 5%",
                "state": "alerting",
                "labels": {
                    "agent_id": "my-agent",
                    "workflow_id": "my-workflow",
                    "severity": "warning"
                },
                "annotations": {
                    "summary": "Error rate is above threshold"
                },
                "startsAt": "2024-01-01T00:00:00Z"
            }
        }
    )


class GrafanaAlertResponse(BaseModel):
    """Response after processing a Grafana alert."""
    
    status: str = Field(..., description="Processing status")
    alert_id: Optional[str] = Field(None, description="Alert ID that was processed")
    reason: Optional[str] = Field(None, description="Reason for status")
    approval_id: Optional[str] = Field(None, description="Approval ID if pending")
    action: Optional[str] = Field(None, description="Action taken or recommended")
    result: Optional[Dict[str, Any]] = Field(None, description="Remediation result")
    investigation: Optional[Dict[str, Any]] = Field(None, description="Investigation summary")


class RemediationActionResponse(BaseModel):
    """Response model for a remediation action."""
    
    script_id: str = Field(..., description="Script identifier")
    name: str = Field(..., description="Action name")
    description: str = Field(..., description="Action description")
    risk_level: RemediationRiskEnum = Field(..., description="Risk level")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    
    model_config = ConfigDict(use_enum_values=True)


class InvestigationResponse(BaseModel):
    """Response model for an investigation result."""
    
    alert_id: str = Field(..., description="Alert ID")
    started_at: datetime = Field(..., description="Investigation start time")
    completed_at: Optional[datetime] = Field(None, description="Investigation completion time")
    root_cause: Optional[str] = Field(None, description="Identified root cause")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in root cause")
    affected_components: List[str] = Field(default_factory=list, description="Affected components")
    recommended_action: Optional[str] = Field(None, description="Recommended action")
    action_parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")


class ApprovalRequestResponse(BaseModel):
    """Response model for an approval request."""
    
    id: str = Field(..., description="Approval request ID")
    action: RemediationActionResponse = Field(..., description="Action pending approval")
    investigation: InvestigationResponse = Field(..., description="Investigation that led to action")
    requested_at: datetime = Field(..., description="Request creation time")
    expires_at: datetime = Field(..., description="Request expiration time")
    status: str = Field(..., description="Request status (pending, approved, rejected, expired)")
    approved_by: Optional[str] = Field(None, description="Approver username")
    rejected_by: Optional[str] = Field(None, description="Rejector username")
    rejection_reason: Optional[str] = Field(None, description="Rejection reason")


class ApprovalActionRequest(BaseModel):
    """Request model for approving or rejecting an action."""
    
    user: str = Field(..., description="Username performing the action")
    reason: Optional[str] = Field(None, description="Reason (required for rejection)")


class ApprovalActionResponse(BaseModel):
    """Response after approving or rejecting an action."""
    
    status: str = Field(..., description="Result status")
    approval_id: str = Field(..., description="Approval ID")
    approved_by: Optional[str] = Field(None, description="Approver if approved")
    rejected_by: Optional[str] = Field(None, description="Rejector if rejected")
    reason: Optional[str] = Field(None, description="Rejection reason")
    result: Optional[Dict[str, Any]] = Field(None, description="Execution result if approved")
    error: Optional[str] = Field(None, description="Error message if any")
