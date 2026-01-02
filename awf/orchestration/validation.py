"""
AI Workflow Fabric - Workflow Validation

This module provides comprehensive validation for workflow definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from awf.orchestration.types import (
    StepDefinition,
    WorkflowDefinition,
    RetryPolicy,
    FallbackPolicy,
)
from awf.orchestration.mapping import ExpressionValidator
from awf.orchestration.dag import build_dag, WorkflowDAG
from awf.orchestration.errors import (
    WorkflowDefinitionError,
    CyclicDependencyError,
    InvalidStepReferenceError,
    DuplicateStepIdError,
    InvalidInputMapError,
)


# =============================================================================
# Validation Result
# =============================================================================


@dataclass
class ValidationError:
    """A single validation error."""
    
    message: str
    step_id: Optional[str] = None
    field: Optional[str] = None
    severity: str = "error"  # error, warning, info
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result: Dict[str, Any] = {
            "message": self.message,
            "severity": self.severity,
        }
        if self.step_id:
            result["stepId"] = self.step_id
        if self.field:
            result["field"] = self.field
        return result


@dataclass
class ValidationResult:
    """Result of workflow validation."""
    
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    dag: Optional[WorkflowDAG] = None
    
    @property
    def error_count(self) -> int:
        return len(self.errors)
    
    @property
    def warning_count(self) -> int:
        return len(self.warnings)
    
    def add_error(
        self,
        message: str,
        step_id: Optional[str] = None,
        field: Optional[str] = None,
    ) -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(
            message=message,
            step_id=step_id,
            field=field,
            severity="error",
        ))
        self.valid = False
    
    def add_warning(
        self,
        message: str,
        step_id: Optional[str] = None,
        field: Optional[str] = None,
    ) -> None:
        """Add a validation warning."""
        self.warnings.append(ValidationError(
            message=message,
            step_id=step_id,
            field=field,
            severity="warning",
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errorCount": self.error_count,
            "warningCount": self.warning_count,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
        }


# =============================================================================
# Workflow Validator
# =============================================================================


class WorkflowValidator:
    """
    Validates workflow definitions for correctness and best practices.
    
    Validation includes:
    - Structural validation (required fields, valid IDs)
    - DAG validation (no cycles, valid references)
    - Input mapping validation (valid expressions, available steps)
    - Policy validation (retry, fallback)
    - Best practice warnings
    """
    
    # Reserved step IDs that cannot be used
    RESERVED_STEP_IDS = {"input", "output", "context", "steps"}
    
    # Maximum limits
    MAX_STEPS = 100
    MAX_STEP_ID_LENGTH = 64
    MAX_WORKFLOW_ID_LENGTH = 128
    MAX_DEPENDS_ON = 20
    MAX_INPUT_MAP_SIZE = 50
    
    def __init__(
        self,
        strict: bool = False,
        check_agents: bool = False,
        available_agents: Optional[Set[str]] = None,
    ):
        """
        Initialize the validator.
        
        Args:
            strict: If True, treat warnings as errors
            check_agents: If True, validate that agent IDs exist
            available_agents: Set of available agent IDs (for check_agents)
        """
        self.strict = strict
        self.check_agents = check_agents
        self.available_agents = available_agents or set()
    
    def validate(self, workflow: WorkflowDefinition) -> ValidationResult:
        """
        Validate a workflow definition.
        
        Args:
            workflow: The workflow to validate
        
        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult(valid=True)
        
        # 1. Validate workflow-level fields
        self._validate_workflow_fields(workflow, result)
        
        # 2. Validate steps
        self._validate_steps(workflow, result)
        
        # 3. Validate DAG structure
        self._validate_dag(workflow, result)
        
        # 4. Validate input mappings
        self._validate_input_mappings(workflow, result)
        
        # 5. Validate output mapping
        self._validate_output_mapping(workflow, result)
        
        # 6. Best practice warnings
        self._check_best_practices(workflow, result)
        
        # If strict mode, promote warnings to errors
        if self.strict and result.warnings:
            for warning in result.warnings:
                warning.severity = "error"
                result.errors.append(warning)
            result.warnings = []
            result.valid = False
        
        return result
    
    def _validate_workflow_fields(
        self,
        workflow: WorkflowDefinition,
        result: ValidationResult,
    ) -> None:
        """Validate workflow-level fields."""
        # ID
        if not workflow.id:
            result.add_error("Workflow ID is required", field="id")
        elif len(workflow.id) > self.MAX_WORKFLOW_ID_LENGTH:
            result.add_error(
                f"Workflow ID exceeds maximum length of {self.MAX_WORKFLOW_ID_LENGTH}",
                field="id",
            )
        elif not self._is_valid_identifier(workflow.id):
            result.add_error(
                "Workflow ID must be alphanumeric with dashes/underscores",
                field="id",
            )
        
        # Name
        if not workflow.name:
            result.add_error("Workflow name is required", field="name")
        
        # Steps
        if not workflow.steps:
            result.add_error(
                "Workflow must have at least one step",
                field="steps",
            )
        elif len(workflow.steps) > self.MAX_STEPS:
            result.add_error(
                f"Workflow exceeds maximum of {self.MAX_STEPS} steps",
                field="steps",
            )
        
        # Timeout
        if workflow.timeout_ms is not None and workflow.timeout_ms <= 0:
            result.add_error(
                "Workflow timeout must be positive",
                field="timeoutMs",
            )
        
        # Default retry policy
        if workflow.default_retry:
            self._validate_retry_policy(
                workflow.default_retry,
                result,
                field="defaultRetry",
            )
    
    def _validate_steps(
        self,
        workflow: WorkflowDefinition,
        result: ValidationResult,
    ) -> None:
        """Validate individual steps."""
        seen_ids: Set[str] = set()
        
        for step in workflow.steps:
            # Check for duplicate IDs
            if step.id in seen_ids:
                result.add_error(
                    f"Duplicate step ID: '{step.id}'",
                    step_id=step.id,
                    field="id",
                )
            seen_ids.add(step.id)
            
            # Validate step fields
            self._validate_step(step, workflow, result)
    
    def _validate_step(
        self,
        step: StepDefinition,
        workflow: WorkflowDefinition,
        result: ValidationResult,
    ) -> None:
        """Validate a single step."""
        # ID
        if not step.id:
            result.add_error("Step ID is required", step_id="(unknown)", field="id")
        elif len(step.id) > self.MAX_STEP_ID_LENGTH:
            result.add_error(
                f"Step ID exceeds maximum length of {self.MAX_STEP_ID_LENGTH}",
                step_id=step.id,
                field="id",
            )
        elif step.id in self.RESERVED_STEP_IDS:
            result.add_error(
                f"Step ID '{step.id}' is reserved",
                step_id=step.id,
                field="id",
            )
        elif not self._is_valid_identifier(step.id):
            result.add_error(
                "Step ID must be alphanumeric with dashes/underscores",
                step_id=step.id,
                field="id",
            )
        
        # Agent ID
        if not step.agent_id:
            result.add_error(
                "Step agent_id is required",
                step_id=step.id,
                field="agentId",
            )
        elif self.check_agents and step.agent_id not in self.available_agents:
            result.add_error(
                f"Agent '{step.agent_id}' not found in registry",
                step_id=step.id,
                field="agentId",
            )
        
        # Depends on
        if len(step.depends_on) > self.MAX_DEPENDS_ON:
            result.add_error(
                f"Step has too many dependencies (max {self.MAX_DEPENDS_ON})",
                step_id=step.id,
                field="dependsOn",
            )
        
        # Timeout
        if step.timeout_ms is not None and step.timeout_ms <= 0:
            result.add_error(
                "Step timeout must be positive",
                step_id=step.id,
                field="timeoutMs",
            )
        
        # Input map size
        if len(step.input_map) > self.MAX_INPUT_MAP_SIZE:
            result.add_error(
                f"Input map has too many entries (max {self.MAX_INPUT_MAP_SIZE})",
                step_id=step.id,
                field="inputMap",
            )
        
        # Retry policy
        if step.retry:
            self._validate_retry_policy(
                step.retry,
                result,
                step_id=step.id,
                field="retry",
            )
        
        # Fallback policy
        if step.fallback:
            self._validate_fallback_policy(
                step.fallback,
                result,
                step_id=step.id,
                workflow=workflow,
            )
        
        # Condition
        if step.condition:
            self._validate_condition(step.condition, result, step_id=step.id)
    
    def _validate_retry_policy(
        self,
        policy: RetryPolicy,
        result: ValidationResult,
        step_id: Optional[str] = None,
        field: str = "retry",
    ) -> None:
        """Validate a retry policy."""
        if policy.max_attempts < 1:
            result.add_error(
                "Retry max_attempts must be at least 1",
                step_id=step_id,
                field=field,
            )
        
        if policy.backoff_ms < 0:
            result.add_error(
                "Retry backoff_ms must be non-negative",
                step_id=step_id,
                field=field,
            )
        
        if policy.backoff_multiplier < 1.0:
            result.add_error(
                "Retry backoff_multiplier must be at least 1.0",
                step_id=step_id,
                field=field,
            )
        
        if policy.max_backoff_ms < policy.backoff_ms:
            result.add_warning(
                "Retry max_backoff_ms is less than backoff_ms",
                step_id=step_id,
                field=field,
            )
        
        if policy.jitter_factor < 0 or policy.jitter_factor > 1:
            result.add_error(
                "Retry jitter_factor must be between 0 and 1",
                step_id=step_id,
                field=field,
            )
    
    def _validate_fallback_policy(
        self,
        policy: FallbackPolicy,
        result: ValidationResult,
        step_id: str,
        workflow: WorkflowDefinition,
    ) -> None:
        """Validate a fallback policy."""
        options_set = sum([
            policy.agent_id is not None,
            policy.static_value is not None,
            policy.skip,
        ])
        
        if options_set == 0:
            result.add_error(
                "Fallback must specify agent_id, static_value, or skip",
                step_id=step_id,
                field="fallback",
            )
        elif options_set > 1:
            result.add_error(
                "Fallback can only have one of: agent_id, static_value, or skip",
                step_id=step_id,
                field="fallback",
            )
        
        # Check fallback agent exists
        if policy.agent_id:
            if self.check_agents and policy.agent_id not in self.available_agents:
                result.add_error(
                    f"Fallback agent '{policy.agent_id}' not found in registry",
                    step_id=step_id,
                    field="fallback.agentId",
                )
            
            # Check not referencing self
            step = workflow.get_step(step_id)
            if step and policy.agent_id == step.agent_id:
                result.add_warning(
                    "Fallback agent is the same as primary agent",
                    step_id=step_id,
                    field="fallback.agentId",
                )
    
    def _validate_condition(
        self,
        condition: str,
        result: ValidationResult,
        step_id: str,
    ) -> None:
        """Validate a step condition expression."""
        # Basic validation - ensure it looks like a valid expression
        if not condition.strip():
            result.add_error(
                "Condition cannot be empty",
                step_id=step_id,
                field="condition",
            )
            return
        
        # Check for valid JSONPath references
        # Conditions should reference $.steps.X.output.Y or $.input.X
        valid_prefixes = ["$.input.", "$.steps.", "$.context."]
        
        # Find all $. references in the condition
        import re
        refs = re.findall(r'\$\.[a-zA-Z_][a-zA-Z0-9_.\[\]]*', condition)
        
        for ref in refs:
            is_valid, error = ExpressionValidator.validate(ref)
            if not is_valid:
                result.add_error(
                    f"Invalid expression in condition: {error}",
                    step_id=step_id,
                    field="condition",
                )
    
    def _validate_dag(
        self,
        workflow: WorkflowDefinition,
        result: ValidationResult,
    ) -> None:
        """Validate the workflow DAG structure."""
        try:
            dag = build_dag(workflow)
            dag.get_topological_order()  # This will detect cycles
            result.dag = dag
        except CyclicDependencyError as e:
            result.add_error(
                f"Cyclic dependency detected: {' -> '.join(e.cycle)}",
                field="steps",
            )
        except InvalidStepReferenceError as e:
            result.add_error(
                f"Step '{e.step_id}' references unknown step '{e.referenced_step_id}'",
                step_id=e.step_id,
                field="dependsOn",
            )
    
    def _validate_input_mappings(
        self,
        workflow: WorkflowDefinition,
        result: ValidationResult,
    ) -> None:
        """Validate input mappings for all steps."""
        # Build set of steps that come before each step
        step_order: Dict[str, Set[str]] = {}
        step_ids = {s.id for s in workflow.steps}
        
        # For each step, find all steps that must complete before it
        for step in workflow.steps:
            # A step's available data includes all its ancestors
            available = set()
            to_visit = list(step.depends_on)
            
            while to_visit:
                current = to_visit.pop()
                if current in step_ids and current not in available:
                    available.add(current)
                    # Find current step's dependencies
                    current_step = workflow.get_step(current)
                    if current_step:
                        to_visit.extend(current_step.depends_on)
            
            step_order[step.id] = available
        
        # Validate each step's input map
        for step in workflow.steps:
            available_steps = list(step_order[step.id])
            
            errors = ExpressionValidator.validate_input_map(
                step.input_map,
                available_steps,
            )
            
            for error in errors:
                result.add_error(
                    error,
                    step_id=step.id,
                    field="inputMap",
                )
    
    def _validate_output_mapping(
        self,
        workflow: WorkflowDefinition,
        result: ValidationResult,
    ) -> None:
        """Validate the workflow output mapping."""
        if not workflow.output_map:
            return
        
        # All steps are available for output mapping
        available_steps = [s.id for s in workflow.steps]
        
        errors = ExpressionValidator.validate_input_map(
            workflow.output_map,
            available_steps,
        )
        
        for error in errors:
            result.add_error(
                error,
                field="outputMap",
            )
    
    def _check_best_practices(
        self,
        workflow: WorkflowDefinition,
        result: ValidationResult,
    ) -> None:
        """Check for best practice violations (warnings)."""
        # Check for steps without retry policy
        for step in workflow.steps:
            if not step.retry and not workflow.default_retry:
                result.add_warning(
                    "Step has no retry policy configured",
                    step_id=step.id,
                )
        
        # Check for very long chains
        if result.dag:
            parallel_groups = result.dag.get_parallel_groups()
            if len(parallel_groups) > 20:
                result.add_warning(
                    f"Workflow has {len(parallel_groups)} sequential levels, "
                    "consider parallelizing more steps",
                )
        
        # Check for single-step workflows (may indicate over-engineering)
        if len(workflow.steps) == 1:
            result.add_warning(
                "Workflow has only one step - consider using direct agent execution",
            )
        
        # Check for very wide parallel groups
        if result.dag:
            for i, group in enumerate(result.dag.get_parallel_groups()):
                if len(group) > 10:
                    result.add_warning(
                        f"Level {i+1} has {len(group)} parallel steps - "
                        "ensure adequate resources",
                    )
    
    def _is_valid_identifier(self, value: str) -> bool:
        """Check if a value is a valid identifier."""
        import re
        return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_-]*$', value))


# =============================================================================
# Convenience Functions
# =============================================================================


def validate_workflow(
    workflow: WorkflowDefinition,
    strict: bool = False,
) -> ValidationResult:
    """
    Validate a workflow definition.
    
    Args:
        workflow: The workflow to validate
        strict: If True, treat warnings as errors
    
    Returns:
        ValidationResult
    """
    validator = WorkflowValidator(strict=strict)
    return validator.validate(workflow)


def is_valid_workflow(workflow: WorkflowDefinition) -> bool:
    """
    Check if a workflow definition is valid.
    
    Args:
        workflow: The workflow to check
    
    Returns:
        True if valid, False otherwise
    """
    result = validate_workflow(workflow)
    return result.valid
