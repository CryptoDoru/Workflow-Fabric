"""
AI Workflow Fabric - Orchestration Engine Types

This module defines the data types for workflow orchestration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4


# =============================================================================
# Enumerations
# =============================================================================


class ExecutionStatus(str, Enum):
    """Status of a workflow execution."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class StepStatus(str, Enum):
    """Status of a workflow step."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class ErrorCategory(str, Enum):
    """Categories of execution errors."""
    
    INVALID_INPUT = "invalid_input"
    AGENT_NOT_FOUND = "agent_not_found"
    AGENT_INACTIVE = "agent_inactive"
    TIMEOUT = "timeout"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    INTERNAL_ERROR = "internal_error"
    PERMISSION_DENIED = "permission_denied"
    CANCELLED = "cancelled"


class WorkflowEventType(str, Enum):
    """Types of workflow events."""
    
    # Workflow lifecycle
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_CANCELLED = "workflow.cancelled"
    
    # Step lifecycle
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    STEP_RETRYING = "step.retrying"
    STEP_SKIPPED = "step.skipped"
    STEP_TIMEOUT = "step.timeout"
    STEP_FALLBACK = "step.fallback"


# =============================================================================
# Retry Policy
# =============================================================================


@dataclass
class RetryPolicy:
    """Configuration for step retry behavior."""
    
    max_attempts: int = 3
    backoff_ms: int = 1000
    backoff_multiplier: float = 2.0
    max_backoff_ms: int = 30000
    jitter_factor: float = 0.25
    retry_on: List[ErrorCategory] = field(default_factory=lambda: [
        ErrorCategory.TIMEOUT,
        ErrorCategory.EXTERNAL_SERVICE_ERROR,
    ])
    no_retry_on: List[ErrorCategory] = field(default_factory=lambda: [
        ErrorCategory.INVALID_INPUT,
        ErrorCategory.PERMISSION_DENIED,
        ErrorCategory.AGENT_NOT_FOUND,
    ])
    
    def should_retry(self, error_category: ErrorCategory, attempt: int) -> bool:
        """Determine if a retry should be attempted."""
        if attempt >= self.max_attempts:
            return False
        if error_category in self.no_retry_on:
            return False
        if self.retry_on and error_category not in self.retry_on:
            return False
        return True
    
    def get_delay_ms(self, attempt: int) -> int:
        """Calculate delay for a retry attempt with jitter."""
        import random
        base_delay = min(
            self.backoff_ms * (self.backoff_multiplier ** attempt),
            self.max_backoff_ms
        )
        jitter = random.uniform(0, self.jitter_factor * base_delay)
        return int(base_delay + jitter)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "maxAttempts": self.max_attempts,
            "backoffMs": self.backoff_ms,
            "backoffMultiplier": self.backoff_multiplier,
            "maxBackoffMs": self.max_backoff_ms,
            "jitterFactor": self.jitter_factor,
            "retryOn": [e.value for e in self.retry_on],
            "noRetryOn": [e.value for e in self.no_retry_on],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RetryPolicy:
        """Create from dictionary representation."""
        return cls(
            max_attempts=data.get("maxAttempts", 3),
            backoff_ms=data.get("backoffMs", 1000),
            backoff_multiplier=data.get("backoffMultiplier", 2.0),
            max_backoff_ms=data.get("maxBackoffMs", 30000),
            jitter_factor=data.get("jitterFactor", 0.25),
            retry_on=[ErrorCategory(e) for e in data.get("retryOn", [])],
            no_retry_on=[ErrorCategory(e) for e in data.get("noRetryOn", [])],
        )


# =============================================================================
# Fallback Policy
# =============================================================================


@dataclass
class FallbackPolicy:
    """Configuration for step fallback behavior."""
    
    agent_id: Optional[str] = None
    static_value: Optional[Dict[str, Any]] = None
    skip: bool = False
    
    def __post_init__(self):
        """Validate fallback configuration."""
        options_set = sum([
            self.agent_id is not None,
            self.static_value is not None,
            self.skip,
        ])
        if options_set > 1:
            raise ValueError(
                "FallbackPolicy can only have one of: agent_id, static_value, or skip"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {}
        if self.agent_id:
            result["agentId"] = self.agent_id
        if self.static_value:
            result["staticValue"] = self.static_value
        if self.skip:
            result["skip"] = True
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FallbackPolicy:
        """Create from dictionary representation."""
        return cls(
            agent_id=data.get("agentId"),
            static_value=data.get("staticValue"),
            skip=data.get("skip", False),
        )


# =============================================================================
# Step Definition
# =============================================================================


@dataclass
class StepDefinition:
    """Definition of a single workflow step."""
    
    id: str
    agent_id: str
    input_map: Dict[str, str] = field(default_factory=dict)
    timeout_ms: Optional[int] = None
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    retry: Optional[RetryPolicy] = None
    fallback: Optional[FallbackPolicy] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate step definition."""
        if not self.id:
            raise ValueError("Step id is required")
        if not self.agent_id:
            raise ValueError("Step agent_id is required")
        # Validate input_map keys are valid identifiers
        for key in self.input_map:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
                raise ValueError(f"Invalid input key: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "id": self.id,
            "agentId": self.agent_id,
        }
        if self.input_map:
            result["inputMap"] = self.input_map
        if self.timeout_ms:
            result["timeoutMs"] = self.timeout_ms
        if self.depends_on:
            result["dependsOn"] = self.depends_on
        if self.condition:
            result["condition"] = self.condition
        if self.retry:
            result["retry"] = self.retry.to_dict()
        if self.fallback:
            result["fallback"] = self.fallback.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StepDefinition:
        """Create from dictionary representation."""
        return cls(
            id=data["id"],
            agent_id=data["agentId"],
            input_map=data.get("inputMap", {}),
            timeout_ms=data.get("timeoutMs"),
            depends_on=data.get("dependsOn", []),
            condition=data.get("condition"),
            retry=RetryPolicy.from_dict(data["retry"]) if data.get("retry") else None,
            fallback=FallbackPolicy.from_dict(data["fallback"]) if data.get("fallback") else None,
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Workflow Definition
# =============================================================================


@dataclass
class WorkflowDefinition:
    """Definition of a workflow."""
    
    id: str
    name: str
    steps: List[StepDefinition]
    version: str = "1.0.0"
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_map: Optional[Dict[str, str]] = None
    timeout_ms: Optional[int] = None
    default_retry: Optional[RetryPolicy] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Validate workflow definition."""
        if not self.id:
            raise ValueError("Workflow id is required")
        if not self.name:
            raise ValueError("Workflow name is required")
        if not self.steps:
            raise ValueError("Workflow must have at least one step")
        
        # Validate unique step IDs
        step_ids = [s.id for s in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Step IDs must be unique within a workflow")
        
        # Validate depends_on references
        step_id_set = set(step_ids)
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_id_set:
                    raise ValueError(
                        f"Step '{step.id}' depends on unknown step '{dep}'"
                    )
    
    def get_step(self, step_id: str) -> Optional[StepDefinition]:
        """Get a step by ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "steps": [s.to_dict() for s in self.steps],
            "createdAt": self.created_at.isoformat(),
        }
        if self.description:
            result["description"] = self.description
        if self.input_schema:
            result["inputSchema"] = self.input_schema
        if self.output_map:
            result["outputMap"] = self.output_map
        if self.timeout_ms:
            result["timeoutMs"] = self.timeout_ms
        if self.default_retry:
            result["defaultRetry"] = self.default_retry.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkflowDefinition:
        """Create from dictionary representation."""
        return cls(
            id=data["id"],
            name=data["name"],
            version=data.get("version", "1.0.0"),
            steps=[StepDefinition.from_dict(s) for s in data["steps"]],
            description=data.get("description"),
            input_schema=data.get("inputSchema"),
            output_map=data.get("outputMap"),
            timeout_ms=data.get("timeoutMs"),
            default_retry=RetryPolicy.from_dict(data["defaultRetry"]) if data.get("defaultRetry") else None,
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Step Result
# =============================================================================


@dataclass
class StepResult:
    """Result of executing a workflow step."""
    
    step_id: str
    status: StepStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    error_category: Optional[ErrorCategory] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[int] = None
    retry_count: int = 0
    used_fallback: bool = False
    token_usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "stepId": self.step_id,
            "status": self.status.value,
        }
        if self.output is not None:
            result["output"] = self.output
        if self.error:
            result["error"] = self.error
        if self.error_category:
            result["errorCategory"] = self.error_category.value
        if self.started_at:
            result["startedAt"] = self.started_at.isoformat()
        if self.completed_at:
            result["completedAt"] = self.completed_at.isoformat()
        if self.execution_time_ms is not None:
            result["executionTimeMs"] = self.execution_time_ms
        if self.retry_count:
            result["retryCount"] = self.retry_count
        if self.used_fallback:
            result["usedFallback"] = True
        if self.token_usage:
            result["tokenUsage"] = self.token_usage
        if self.metadata:
            result["metadata"] = self.metadata
        return result


# =============================================================================
# Workflow Result
# =============================================================================


@dataclass
class WorkflowResult:
    """Result of executing a workflow."""
    
    execution_id: str
    workflow_id: str
    status: ExecutionStatus
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_execution_time_ms: Optional[int] = None
    total_token_usage: Optional[Dict[str, int]] = None
    total_retry_count: int = 0
    total_fallback_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "executionId": self.execution_id,
            "workflowId": self.workflow_id,
            "status": self.status.value,
            "input": self.input,
        }
        if self.output is not None:
            result["output"] = self.output
        if self.error:
            result["error"] = self.error
        if self.step_results:
            result["stepResults"] = {
                k: v.to_dict() for k, v in self.step_results.items()
            }
        if self.started_at:
            result["startedAt"] = self.started_at.isoformat()
        if self.completed_at:
            result["completedAt"] = self.completed_at.isoformat()
        if self.total_execution_time_ms is not None:
            result["totalExecutionTimeMs"] = self.total_execution_time_ms
        if self.total_token_usage:
            result["totalTokenUsage"] = self.total_token_usage
        if self.total_retry_count:
            result["totalRetryCount"] = self.total_retry_count
        if self.total_fallback_count:
            result["totalFallbackCount"] = self.total_fallback_count
        if self.metadata:
            result["metadata"] = self.metadata
        return result


# =============================================================================
# Workflow Event
# =============================================================================


@dataclass
class WorkflowEvent:
    """An event emitted during workflow execution."""
    
    type: WorkflowEventType
    execution_id: str
    workflow_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    step_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "type": self.type.value,
            "executionId": self.execution_id,
            "workflowId": self.workflow_id,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.step_id:
            result["stepId"] = self.step_id
        if self.data:
            result["data"] = self.data
        return result


# =============================================================================
# Execution Context
# =============================================================================


@dataclass
class ExecutionContext:
    """Context for a workflow execution."""
    
    execution_id: str
    workflow: WorkflowDefinition
    input: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    status: ExecutionStatus = ExecutionStatus.PENDING
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    current_step: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    trace_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    @classmethod
    def create(
        cls,
        workflow: WorkflowDefinition,
        input: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> ExecutionContext:
        """Create a new execution context."""
        return cls(
            execution_id=str(uuid4()),
            workflow=workflow,
            input=input,
            context=context or {},
            trace_id=trace_id or str(uuid4()),
            correlation_id=correlation_id,
        )
    
    def get_step_result(self, step_id: str) -> Optional[StepResult]:
        """Get the result of a step."""
        return self.step_results.get(step_id)
    
    def set_step_result(self, result: StepResult) -> None:
        """Store a step result."""
        self.step_results[result.step_id] = result
    
    def is_step_completed(self, step_id: str) -> bool:
        """Check if a step has completed (successfully or not)."""
        result = self.step_results.get(step_id)
        if result is None:
            return False
        return result.status in (
            StepStatus.COMPLETED,
            StepStatus.FAILED,
            StepStatus.SKIPPED,
            StepStatus.TIMEOUT,
        )
    
    def are_dependencies_met(self, step: StepDefinition) -> bool:
        """Check if all dependencies for a step are completed."""
        for dep_id in step.depends_on:
            dep_result = self.step_results.get(dep_id)
            if dep_result is None:
                return False
            if dep_result.status != StepStatus.COMPLETED:
                return False
        return True
