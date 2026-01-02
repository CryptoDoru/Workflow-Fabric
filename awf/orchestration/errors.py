"""
AI Workflow Fabric - Orchestration Errors

This module defines the exception hierarchy for workflow orchestration.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from awf.orchestration.types import ErrorCategory


# =============================================================================
# Base Exception
# =============================================================================


class OrchestrationError(Exception):
    """Base exception for all orchestration errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL_ERROR,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "details": self.details,
        }


# =============================================================================
# Workflow Definition Errors
# =============================================================================


class WorkflowDefinitionError(OrchestrationError):
    """Error in workflow definition."""
    
    def __init__(
        self,
        message: str,
        workflow_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            category=ErrorCategory.INVALID_INPUT,
            details=details,
        )
        self.workflow_id = workflow_id


class CyclicDependencyError(WorkflowDefinitionError):
    """Workflow contains cyclic dependencies."""
    
    def __init__(
        self,
        cycle: list[str],
        workflow_id: Optional[str] = None,
    ):
        super().__init__(
            f"Cyclic dependency detected: {' -> '.join(cycle)}",
            workflow_id=workflow_id,
            details={"cycle": cycle},
        )
        self.cycle = cycle


class InvalidStepReferenceError(WorkflowDefinitionError):
    """Step references an unknown step."""
    
    def __init__(
        self,
        step_id: str,
        referenced_step_id: str,
        workflow_id: Optional[str] = None,
    ):
        super().__init__(
            f"Step '{step_id}' references unknown step '{referenced_step_id}'",
            workflow_id=workflow_id,
            details={
                "step_id": step_id,
                "referenced_step_id": referenced_step_id,
            },
        )
        self.step_id = step_id
        self.referenced_step_id = referenced_step_id


class DuplicateStepIdError(WorkflowDefinitionError):
    """Duplicate step ID in workflow."""
    
    def __init__(
        self,
        step_id: str,
        workflow_id: Optional[str] = None,
    ):
        super().__init__(
            f"Duplicate step ID: '{step_id}'",
            workflow_id=workflow_id,
            details={"step_id": step_id},
        )
        self.step_id = step_id


class InvalidInputMapError(WorkflowDefinitionError):
    """Invalid input mapping expression."""
    
    def __init__(
        self,
        step_id: str,
        input_key: str,
        expression: str,
        reason: str,
        workflow_id: Optional[str] = None,
    ):
        super().__init__(
            f"Invalid input mapping in step '{step_id}': {reason}",
            workflow_id=workflow_id,
            details={
                "step_id": step_id,
                "input_key": input_key,
                "expression": expression,
                "reason": reason,
            },
        )
        self.step_id = step_id
        self.input_key = input_key
        self.expression = expression
        self.reason = reason


# =============================================================================
# Execution Errors
# =============================================================================


class ExecutionError(OrchestrationError):
    """Error during workflow execution."""
    
    def __init__(
        self,
        message: str,
        execution_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        category: ErrorCategory = ErrorCategory.INTERNAL_ERROR,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, category=category, details=details)
        self.execution_id = execution_id
        self.workflow_id = workflow_id


class StepExecutionError(ExecutionError):
    """Error during step execution."""
    
    def __init__(
        self,
        message: str,
        step_id: str,
        execution_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        category: ErrorCategory = ErrorCategory.INTERNAL_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            execution_id=execution_id,
            workflow_id=workflow_id,
            category=category,
            details=details,
        )
        self.step_id = step_id
        self.cause = cause


class AgentNotFoundError(StepExecutionError):
    """Agent not found in registry."""
    
    def __init__(
        self,
        agent_id: str,
        step_id: str,
        execution_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ):
        super().__init__(
            f"Agent '{agent_id}' not found in registry",
            step_id=step_id,
            execution_id=execution_id,
            workflow_id=workflow_id,
            category=ErrorCategory.AGENT_NOT_FOUND,
            details={"agent_id": agent_id},
        )
        self.agent_id = agent_id


class AgentInactiveError(StepExecutionError):
    """Agent is not active."""
    
    def __init__(
        self,
        agent_id: str,
        step_id: str,
        execution_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ):
        super().__init__(
            f"Agent '{agent_id}' is not active",
            step_id=step_id,
            execution_id=execution_id,
            workflow_id=workflow_id,
            category=ErrorCategory.AGENT_INACTIVE,
            details={"agent_id": agent_id},
        )
        self.agent_id = agent_id


class StepTimeoutError(StepExecutionError):
    """Step execution timed out."""
    
    def __init__(
        self,
        step_id: str,
        timeout_ms: int,
        execution_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ):
        super().__init__(
            f"Step '{step_id}' timed out after {timeout_ms}ms",
            step_id=step_id,
            execution_id=execution_id,
            workflow_id=workflow_id,
            category=ErrorCategory.TIMEOUT,
            details={"timeout_ms": timeout_ms},
        )
        self.timeout_ms = timeout_ms


class WorkflowTimeoutError(ExecutionError):
    """Workflow execution timed out."""
    
    def __init__(
        self,
        timeout_ms: int,
        execution_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        current_step: Optional[str] = None,
    ):
        super().__init__(
            f"Workflow timed out after {timeout_ms}ms",
            execution_id=execution_id,
            workflow_id=workflow_id,
            category=ErrorCategory.TIMEOUT,
            details={
                "timeout_ms": timeout_ms,
                "current_step": current_step,
            },
        )
        self.timeout_ms = timeout_ms
        self.current_step = current_step


class WorkflowCancelledError(ExecutionError):
    """Workflow was cancelled."""
    
    def __init__(
        self,
        execution_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        reason: Optional[str] = None,
    ):
        super().__init__(
            f"Workflow cancelled{': ' + reason if reason else ''}",
            execution_id=execution_id,
            workflow_id=workflow_id,
            category=ErrorCategory.CANCELLED,
            details={"reason": reason} if reason else None,
        )
        self.reason = reason


class ConditionEvaluationError(StepExecutionError):
    """Error evaluating step condition."""
    
    def __init__(
        self,
        step_id: str,
        condition: str,
        reason: str,
        execution_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ):
        super().__init__(
            f"Failed to evaluate condition for step '{step_id}': {reason}",
            step_id=step_id,
            execution_id=execution_id,
            workflow_id=workflow_id,
            category=ErrorCategory.INVALID_INPUT,
            details={
                "condition": condition,
                "reason": reason,
            },
        )
        self.condition = condition
        self.reason = reason


class InputMappingError(StepExecutionError):
    """Error mapping step inputs."""
    
    def __init__(
        self,
        step_id: str,
        expression: str,
        reason: str,
        execution_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ):
        super().__init__(
            f"Failed to map input for step '{step_id}': {reason}",
            step_id=step_id,
            execution_id=execution_id,
            workflow_id=workflow_id,
            category=ErrorCategory.INVALID_INPUT,
            details={
                "expression": expression,
                "reason": reason,
            },
        )
        self.expression = expression
        self.reason = reason


class DependencyFailedError(StepExecutionError):
    """A dependency step failed."""
    
    def __init__(
        self,
        step_id: str,
        failed_dependency: str,
        execution_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ):
        super().__init__(
            f"Step '{step_id}' skipped because dependency '{failed_dependency}' failed",
            step_id=step_id,
            execution_id=execution_id,
            workflow_id=workflow_id,
            category=ErrorCategory.INTERNAL_ERROR,
            details={"failed_dependency": failed_dependency},
        )
        self.failed_dependency = failed_dependency


# =============================================================================
# Fallback Errors
# =============================================================================


class FallbackError(StepExecutionError):
    """Error during fallback execution."""
    
    def __init__(
        self,
        step_id: str,
        original_error: str,
        fallback_error: str,
        execution_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ):
        super().__init__(
            f"Fallback failed for step '{step_id}': {fallback_error}",
            step_id=step_id,
            execution_id=execution_id,
            workflow_id=workflow_id,
            category=ErrorCategory.INTERNAL_ERROR,
            details={
                "original_error": original_error,
                "fallback_error": fallback_error,
            },
        )
        self.original_error = original_error
        self.fallback_error = fallback_error


class NoFallbackAvailableError(StepExecutionError):
    """No fallback configured for failed step."""
    
    def __init__(
        self,
        step_id: str,
        original_error: str,
        execution_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ):
        super().__init__(
            f"Step '{step_id}' failed with no fallback available",
            step_id=step_id,
            execution_id=execution_id,
            workflow_id=workflow_id,
            category=ErrorCategory.INTERNAL_ERROR,
            details={"original_error": original_error},
        )
        self.original_error = original_error


# =============================================================================
# Registry Errors
# =============================================================================


class RegistryError(OrchestrationError):
    """Error interacting with agent registry."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            category=ErrorCategory.EXTERNAL_SERVICE_ERROR,
            details=details,
        )


class AdapterNotFoundError(RegistryError):
    """Adapter not found for agent framework."""
    
    def __init__(
        self,
        framework: str,
        agent_id: str,
    ):
        super().__init__(
            f"No adapter registered for framework '{framework}'",
            details={
                "framework": framework,
                "agent_id": agent_id,
            },
        )
        self.framework = framework
        self.agent_id = agent_id


# =============================================================================
# Utility Functions
# =============================================================================


def categorize_exception(exc: Exception) -> ErrorCategory:
    """Categorize an exception into an ErrorCategory."""
    if isinstance(exc, OrchestrationError):
        return exc.category
    
    # Map common exception types
    exception_name = type(exc).__name__.lower()
    
    if "timeout" in exception_name:
        return ErrorCategory.TIMEOUT
    elif "permission" in exception_name or "forbidden" in exception_name:
        return ErrorCategory.PERMISSION_DENIED
    elif "notfound" in exception_name or "missing" in exception_name:
        return ErrorCategory.AGENT_NOT_FOUND
    elif "validation" in exception_name or "invalid" in exception_name:
        return ErrorCategory.INVALID_INPUT
    elif "connection" in exception_name or "network" in exception_name:
        return ErrorCategory.EXTERNAL_SERVICE_ERROR
    
    return ErrorCategory.INTERNAL_ERROR
