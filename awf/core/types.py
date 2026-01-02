"""
AI Workflow Fabric - Core ASP Types

This module defines the core data types for the Agent State Protocol (ASP).
All types are implemented as Python dataclasses with full type hints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4


# =============================================================================
# Enumerations
# =============================================================================


class AgentStatus(str, Enum):
    """Possible states of an agent in the registry."""
    
    REGISTERED = "registered"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEPRECATED = "deprecated"


class TaskStatus(str, Enum):
    """Possible states of a task execution."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class CapabilityType(str, Enum):
    """Categories of agent capabilities."""
    
    TOOL = "tool"
    REASONING = "reasoning"
    MEMORY = "memory"
    COMMUNICATION = "communication"
    CUSTOM = "custom"


class SandboxTier(str, Enum):
    """Isolation levels for agent execution."""
    
    WASM = "wasm"           # High trust (0.90+): ~10ms overhead
    GVISOR = "gvisor"       # Medium trust (0.70-0.89): ~100ms overhead
    GVISOR_STRICT = "gvisor_strict"  # Low trust (0.40-0.69): ~150ms overhead
    BLOCKED = "blocked"     # Very low trust (<0.40): execution denied


class EventType(str, Enum):
    """Types of events emitted during execution."""
    
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    STATE_CHANGED = "state.changed"
    AGENT_REGISTERED = "agent.registered"
    AGENT_UPDATED = "agent.updated"
    POLICY_VIOLATION = "policy.violation"


# =============================================================================
# Schema Types
# =============================================================================


@dataclass
class SchemaProperty:
    """A single property in a JSON schema."""
    
    name: str
    type: str
    description: Optional[str] = None
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    items: Optional[Dict[str, Any]] = None  # For array types


@dataclass
class Schema:
    """JSON Schema representation for input/output validation."""
    
    type: str = "object"
    properties: Dict[str, SchemaProperty] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    additional_properties: bool = False
    description: Optional[str] = None
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to standard JSON Schema format."""
        schema: Dict[str, Any] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": self.type,
            "additionalProperties": self.additional_properties,
        }
        
        if self.description:
            schema["description"] = self.description
        
        if self.properties:
            schema["properties"] = {}
            for name, prop in self.properties.items():
                prop_schema: Dict[str, Any] = {"type": prop.type}
                if prop.description:
                    prop_schema["description"] = prop.description
                if prop.default is not None:
                    prop_schema["default"] = prop.default
                if prop.enum:
                    prop_schema["enum"] = prop.enum
                if prop.items:
                    prop_schema["items"] = prop.items
                schema["properties"][name] = prop_schema
        
        if self.required:
            schema["required"] = self.required
        
        return schema


# =============================================================================
# Capability Types
# =============================================================================


@dataclass
class Capability:
    """A declared capability of an agent."""
    
    name: str
    type: CapabilityType
    description: Optional[str] = None
    input_schema: Optional[Schema] = None
    output_schema: Optional[Schema] = None
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "name": self.name,
            "type": self.type.value,
        }
        
        if self.description:
            result["description"] = self.description
        if self.input_schema:
            result["inputSchema"] = self.input_schema.to_json_schema()
        if self.output_schema:
            result["outputSchema"] = self.output_schema.to_json_schema()
        if self.permissions:
            result["permissions"] = self.permissions
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result


# =============================================================================
# Agent Manifest
# =============================================================================


@dataclass
class AgentManifest:
    """
    ASP Agent Manifest - describes an agent's identity and capabilities.
    
    This is the core registration artifact for any agent in the AWF ecosystem.
    """
    
    # Required fields
    id: str
    name: str
    version: str
    
    # Framework information
    framework: str
    framework_version: Optional[str] = None
    
    # Capabilities
    capabilities: List[Capability] = field(default_factory=list)
    
    # Schemas
    input_schema: Optional[Schema] = None
    output_schema: Optional[Schema] = None
    
    # Trust and security
    trust_score: Optional[float] = None
    publisher: Optional[str] = None
    audit_status: Optional[str] = None
    
    # Metadata
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    documentation_url: Optional[str] = None
    source_url: Optional[str] = None
    
    # Registration metadata
    status: AgentStatus = AgentStatus.REGISTERED
    registered_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default timestamps if not provided."""
        now = datetime.utcnow()
        if self.registered_at is None:
            self.registered_at = now
        if self.updated_at is None:
            self.updated_at = now
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation (ASP JSON format)."""
        result: Dict[str, Any] = {
            "asp_version": "1.0",
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "framework": self.framework,
            "status": self.status.value,
        }
        
        if self.framework_version:
            result["frameworkVersion"] = self.framework_version
        
        if self.capabilities:
            result["capabilities"] = [cap.to_dict() for cap in self.capabilities]
        
        if self.input_schema:
            result["inputSchema"] = self.input_schema.to_json_schema()
        
        if self.output_schema:
            result["outputSchema"] = self.output_schema.to_json_schema()
        
        if self.trust_score is not None:
            result["trustScore"] = self.trust_score
        
        if self.publisher:
            result["publisher"] = self.publisher
        
        if self.description:
            result["description"] = self.description
        
        if self.tags:
            result["tags"] = self.tags
        
        if self.documentation_url:
            result["documentationUrl"] = self.documentation_url
        
        if self.source_url:
            result["sourceUrl"] = self.source_url
        
        if self.registered_at:
            result["registeredAt"] = self.registered_at.isoformat()
        
        if self.updated_at:
            result["updatedAt"] = self.updated_at.isoformat()
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result


# =============================================================================
# Task Types
# =============================================================================


@dataclass
class Task:
    """
    ASP Task - represents a unit of work to be executed by an agent.
    """
    
    # Required fields
    agent_id: str
    input: Dict[str, Any]
    
    # Optional identification
    id: str = field(default_factory=lambda: str(uuid4()))
    correlation_id: Optional[str] = None
    
    # Execution parameters
    timeout_ms: Optional[int] = None
    priority: int = 0
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "id": self.id,
            "agentId": self.agent_id,
            "input": self.input,
            "priority": self.priority,
            "createdAt": self.created_at.isoformat(),
        }
        
        if self.correlation_id:
            result["correlationId"] = self.correlation_id
        
        if self.timeout_ms:
            result["timeoutMs"] = self.timeout_ms
        
        if self.context:
            result["context"] = self.context
        
        if self.trace_id:
            result["traceId"] = self.trace_id
        
        if self.parent_span_id:
            result["parentSpanId"] = self.parent_span_id
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result


@dataclass
class TaskMetrics:
    """Execution metrics for a completed task."""
    
    execution_time_ms: int
    token_usage: Optional[Dict[str, int]] = None  # input_tokens, output_tokens
    memory_usage_bytes: Optional[int] = None
    cpu_time_ms: Optional[int] = None
    network_calls: Optional[int] = None
    sandbox_tier: Optional[SandboxTier] = None
    sandbox_overhead_ms: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "executionTimeMs": self.execution_time_ms,
        }
        
        if self.token_usage:
            result["tokenUsage"] = self.token_usage
        
        if self.memory_usage_bytes is not None:
            result["memoryUsageBytes"] = self.memory_usage_bytes
        
        if self.cpu_time_ms is not None:
            result["cpuTimeMs"] = self.cpu_time_ms
        
        if self.network_calls is not None:
            result["networkCalls"] = self.network_calls
        
        if self.sandbox_tier:
            result["sandboxTier"] = self.sandbox_tier.value
        
        if self.sandbox_overhead_ms is not None:
            result["sandboxOverheadMs"] = self.sandbox_overhead_ms
        
        return result


@dataclass
class TaskError:
    """Structured error information for failed tasks."""
    
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    retryable: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        
        if self.details:
            result["details"] = self.details
        
        if self.stack_trace:
            result["stackTrace"] = self.stack_trace
        
        return result


@dataclass
class TaskResult:
    """
    ASP Task Result - the outcome of task execution.
    """
    
    # Required fields
    task_id: str
    agent_id: str
    status: TaskStatus
    
    # Output
    output: Optional[Dict[str, Any]] = None
    partial_output: Optional[Dict[str, Any]] = None
    
    # Error information (if failed)
    error: Optional[TaskError] = None
    
    # Metrics
    metrics: Optional[TaskMetrics] = None
    
    # Tracing
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "taskId": self.task_id,
            "agentId": self.agent_id,
            "status": self.status.value,
        }
        
        if self.output:
            result["output"] = self.output
        
        if self.partial_output:
            result["partialOutput"] = self.partial_output
        
        if self.error:
            result["error"] = self.error.to_dict()
        
        if self.metrics:
            result["metrics"] = self.metrics.to_dict()
        
        if self.trace_id:
            result["traceId"] = self.trace_id
        
        if self.span_id:
            result["spanId"] = self.span_id
        
        if self.started_at:
            result["startedAt"] = self.started_at.isoformat()
        
        if self.completed_at:
            result["completedAt"] = self.completed_at.isoformat()
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result


# =============================================================================
# Event Types
# =============================================================================


@dataclass
class Event:
    """
    ASP Event - represents a significant occurrence during execution.
    """
    
    # Required fields
    type: EventType
    source: str  # Agent ID or system component
    
    # Optional identification
    id: str = field(default_factory=lambda: str(uuid4()))
    correlation_id: Optional[str] = None
    
    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Tracing
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "id": self.id,
            "type": self.type.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }
        
        if self.correlation_id:
            result["correlationId"] = self.correlation_id
        
        if self.data:
            result["data"] = self.data
        
        if self.trace_id:
            result["traceId"] = self.trace_id
        
        if self.span_id:
            result["spanId"] = self.span_id
        
        return result


# =============================================================================
# Trust Types
# =============================================================================


@dataclass
class TrustFactors:
    """Breakdown of trust score components."""
    
    publisher_trust: float = 0.0      # Weight: 0.25
    audit_status: float = 0.0         # Weight: 0.25
    community_trust: float = 0.0      # Weight: 0.20
    permission_analysis: float = 0.0  # Weight: 0.15
    historical_behavior: float = 0.0  # Weight: 0.15
    
    def compute_score(self) -> float:
        """Compute weighted trust score."""
        return (
            self.publisher_trust * 0.25 +
            self.audit_status * 0.25 +
            self.community_trust * 0.20 +
            self.permission_analysis * 0.15 +
            self.historical_behavior * 0.15
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "publisherTrust": self.publisher_trust,
            "auditStatus": self.audit_status,
            "communityTrust": self.community_trust,
            "permissionAnalysis": self.permission_analysis,
            "historicalBehavior": self.historical_behavior,
            "computedScore": self.compute_score(),
        }


@dataclass
class TrustScore:
    """Complete trust assessment for an agent."""
    
    score: float
    factors: TrustFactors
    sandbox_tier: SandboxTier
    computed_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    @classmethod
    def compute(cls, factors: TrustFactors) -> TrustScore:
        """Compute trust score and determine sandbox tier."""
        score = factors.compute_score()
        
        # Determine sandbox tier based on score
        if score >= 0.90:
            tier = SandboxTier.WASM
        elif score >= 0.70:
            tier = SandboxTier.GVISOR
        elif score >= 0.40:
            tier = SandboxTier.GVISOR_STRICT
        else:
            tier = SandboxTier.BLOCKED
        
        return cls(
            score=score,
            factors=factors,
            sandbox_tier=tier,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "score": self.score,
            "factors": self.factors.to_dict(),
            "sandboxTier": self.sandbox_tier.value,
            "computedAt": self.computed_at.isoformat(),
        }
        
        if self.expires_at:
            result["expiresAt"] = self.expires_at.isoformat()
        
        return result


# =============================================================================
# Workflow Types
# =============================================================================


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    
    id: str
    agent_id: str
    input_map: Dict[str, str] = field(default_factory=dict)  # JSONPath mappings
    condition: Optional[str] = None  # Conditional execution expression
    timeout_ms: Optional[int] = None
    retry_policy: Optional[Dict[str, Any]] = None
    on_error: Optional[str] = None  # "fail", "continue", "fallback"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "id": self.id,
            "agentId": self.agent_id,
        }
        
        if self.input_map:
            result["inputMap"] = self.input_map
        
        if self.condition:
            result["condition"] = self.condition
        
        if self.timeout_ms:
            result["timeoutMs"] = self.timeout_ms
        
        if self.retry_policy:
            result["retryPolicy"] = self.retry_policy
        
        if self.on_error:
            result["onError"] = self.on_error
        
        return result


@dataclass
class Workflow:
    """
    ASP Workflow - a composition of multiple agent invocations.
    """
    
    id: str
    name: str
    steps: List[WorkflowStep]
    
    # Schema
    input_schema: Optional[Schema] = None
    output_schema: Optional[Schema] = None
    
    # Execution configuration
    timeout_ms: Optional[int] = None
    max_retries: int = 0
    
    # Metadata
    description: Optional[str] = None
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "steps": [step.to_dict() for step in self.steps],
            "createdAt": self.created_at.isoformat(),
        }
        
        if self.input_schema:
            result["inputSchema"] = self.input_schema.to_json_schema()
        
        if self.output_schema:
            result["outputSchema"] = self.output_schema.to_json_schema()
        
        if self.timeout_ms:
            result["timeoutMs"] = self.timeout_ms
        
        if self.max_retries:
            result["maxRetries"] = self.max_retries
        
        if self.description:
            result["description"] = self.description
        
        return result


@dataclass
class WorkflowExecution:
    """State of a workflow execution."""
    
    id: str
    workflow_id: str
    status: TaskStatus
    input: Dict[str, Any]
    
    # Step results
    step_results: Dict[str, TaskResult] = field(default_factory=dict)
    current_step: Optional[str] = None
    
    # Final output
    output: Optional[Dict[str, Any]] = None
    error: Optional[TaskError] = None
    
    # Tracing
    trace_id: Optional[str] = None
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "id": self.id,
            "workflowId": self.workflow_id,
            "status": self.status.value,
            "input": self.input,
        }
        
        if self.step_results:
            result["stepResults"] = {
                k: v.to_dict() for k, v in self.step_results.items()
            }
        
        if self.current_step:
            result["currentStep"] = self.current_step
        
        if self.output:
            result["output"] = self.output
        
        if self.error:
            result["error"] = self.error.to_dict()
        
        if self.trace_id:
            result["traceId"] = self.trace_id
        
        if self.started_at:
            result["startedAt"] = self.started_at.isoformat()
        
        if self.completed_at:
            result["completedAt"] = self.completed_at.isoformat()
        
        return result


# =============================================================================
# Policy Types
# =============================================================================


@dataclass
class Policy:
    """
    Security/governance policy for agent execution.
    """
    
    id: str
    name: str
    
    # Scope
    environments: List[str] = field(default_factory=list)  # e.g., ["production", "staging"]
    agent_ids: Optional[List[str]] = None  # If None, applies to all
    
    # Rules
    min_trust_score: Optional[float] = None
    max_trust_score: Optional[float] = None
    require_capabilities: List[str] = field(default_factory=list)
    deny_capabilities: List[str] = field(default_factory=list)
    
    # Execution limits
    max_execution_time_ms: Optional[int] = None
    max_memory_bytes: Optional[int] = None
    allow_network: bool = True
    allow_filesystem: bool = False
    
    # Metadata
    description: Optional[str] = None
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "enabled": self.enabled,
            "createdAt": self.created_at.isoformat(),
        }
        
        if self.environments:
            result["environments"] = self.environments
        
        if self.agent_ids:
            result["agentIds"] = self.agent_ids
        
        if self.min_trust_score is not None:
            result["minTrustScore"] = self.min_trust_score
        
        if self.max_trust_score is not None:
            result["maxTrustScore"] = self.max_trust_score
        
        if self.require_capabilities:
            result["requireCapabilities"] = self.require_capabilities
        
        if self.deny_capabilities:
            result["denyCapabilities"] = self.deny_capabilities
        
        if self.max_execution_time_ms:
            result["maxExecutionTimeMs"] = self.max_execution_time_ms
        
        if self.max_memory_bytes:
            result["maxMemoryBytes"] = self.max_memory_bytes
        
        result["allowNetwork"] = self.allow_network
        result["allowFilesystem"] = self.allow_filesystem
        
        if self.description:
            result["description"] = self.description
        
        return result


@dataclass
class PolicyViolation:
    """Record of a policy violation."""
    
    policy_id: str
    policy_name: str
    agent_id: str
    violation_type: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "policyId": self.policy_id,
            "policyName": self.policy_name,
            "agentId": self.agent_id,
            "violationType": self.violation_type,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }
