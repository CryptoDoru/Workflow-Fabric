"""
Tests for AWF Orchestration Types

Tests for types.py - core workflow types and data structures.
"""

import pytest
from datetime import datetime, timezone

from awf.orchestration.types import (
    ExecutionStatus,
    StepStatus,
    ErrorCategory,
    WorkflowEventType,
    RetryPolicy,
    FallbackPolicy,
    StepDefinition,
    WorkflowDefinition,
    StepResult,
    WorkflowResult,
    WorkflowEvent,
    ExecutionContext,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestExecutionStatus:
    """Tests for ExecutionStatus enum."""
    
    def test_all_values_present(self):
        """Test all expected status values exist."""
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.CANCELLED.value == "cancelled"
        assert ExecutionStatus.TIMEOUT.value == "timeout"


class TestStepStatus:
    """Tests for StepStatus enum."""
    
    def test_all_values_present(self):
        """Test all expected status values exist."""
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.RUNNING.value == "running"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.SKIPPED.value == "skipped"
        assert StepStatus.TIMEOUT.value == "timeout"


class TestErrorCategory:
    """Tests for ErrorCategory enum."""
    
    def test_all_values_present(self):
        """Test all expected error categories exist."""
        assert ErrorCategory.INVALID_INPUT.value == "invalid_input"
        assert ErrorCategory.AGENT_NOT_FOUND.value == "agent_not_found"
        assert ErrorCategory.TIMEOUT.value == "timeout"
        assert ErrorCategory.INTERNAL_ERROR.value == "internal_error"


class TestWorkflowEventType:
    """Tests for WorkflowEventType enum."""
    
    def test_workflow_events(self):
        """Test workflow lifecycle events."""
        assert WorkflowEventType.WORKFLOW_STARTED.value == "workflow.started"
        assert WorkflowEventType.WORKFLOW_COMPLETED.value == "workflow.completed"
        assert WorkflowEventType.WORKFLOW_FAILED.value == "workflow.failed"
    
    def test_step_events(self):
        """Test step lifecycle events."""
        assert WorkflowEventType.STEP_STARTED.value == "step.started"
        assert WorkflowEventType.STEP_COMPLETED.value == "step.completed"
        assert WorkflowEventType.STEP_RETRYING.value == "step.retrying"


# =============================================================================
# RetryPolicy Tests
# =============================================================================


class TestRetryPolicy:
    """Tests for RetryPolicy dataclass."""
    
    def test_default_values(self):
        """Test default retry policy values."""
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.backoff_ms == 1000
        assert policy.backoff_multiplier == 2.0
        assert policy.max_backoff_ms == 30000
        assert policy.jitter_factor == 0.25
    
    def test_custom_values(self):
        """Test custom retry policy values."""
        policy = RetryPolicy(
            max_attempts=5,
            backoff_ms=500,
            backoff_multiplier=1.5,
        )
        assert policy.max_attempts == 5
        assert policy.backoff_ms == 500
        assert policy.backoff_multiplier == 1.5
    
    def test_should_retry_within_attempts(self):
        """Test should_retry returns True within max attempts."""
        policy = RetryPolicy(max_attempts=3)
        assert policy.should_retry(ErrorCategory.TIMEOUT, 1) is True
        assert policy.should_retry(ErrorCategory.TIMEOUT, 2) is True
    
    def test_should_retry_exceeds_attempts(self):
        """Test should_retry returns False when attempts exceeded."""
        policy = RetryPolicy(max_attempts=3)
        assert policy.should_retry(ErrorCategory.TIMEOUT, 3) is False
        assert policy.should_retry(ErrorCategory.TIMEOUT, 4) is False
    
    def test_should_retry_respects_no_retry_on(self):
        """Test should_retry respects no_retry_on list."""
        policy = RetryPolicy()
        assert policy.should_retry(ErrorCategory.INVALID_INPUT, 1) is False
        assert policy.should_retry(ErrorCategory.PERMISSION_DENIED, 1) is False
    
    def test_get_delay_ms_exponential_backoff(self):
        """Test delay increases exponentially."""
        policy = RetryPolicy(backoff_ms=1000, backoff_multiplier=2.0, jitter_factor=0)
        # With no jitter, should be exactly exponential
        delay_1 = policy.get_delay_ms(1)
        delay_2 = policy.get_delay_ms(2)
        # Allow for floating point precision
        assert delay_1 == 2000  # 1000 * 2^1
        assert delay_2 == 4000  # 1000 * 2^2
    
    def test_get_delay_ms_respects_max(self):
        """Test delay respects max_backoff_ms."""
        policy = RetryPolicy(
            backoff_ms=10000,
            backoff_multiplier=2.0,
            max_backoff_ms=15000,
            jitter_factor=0,
        )
        delay = policy.get_delay_ms(5)  # Would be 320000 without max
        assert delay == 15000
    
    def test_to_dict(self):
        """Test serialization to dict."""
        policy = RetryPolicy(max_attempts=5)
        data = policy.to_dict()
        assert data["maxAttempts"] == 5
        assert "backoffMs" in data
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {"maxAttempts": 5, "backoffMs": 2000}
        policy = RetryPolicy.from_dict(data)
        assert policy.max_attempts == 5
        assert policy.backoff_ms == 2000


# =============================================================================
# FallbackPolicy Tests
# =============================================================================


class TestFallbackPolicy:
    """Tests for FallbackPolicy dataclass."""
    
    def test_agent_fallback(self):
        """Test fallback with agent ID."""
        policy = FallbackPolicy(agent_id="fallback-agent")
        assert policy.agent_id == "fallback-agent"
        assert policy.static_value is None
        assert policy.skip is False
    
    def test_static_value_fallback(self):
        """Test fallback with static value."""
        policy = FallbackPolicy(static_value={"default": "value"})
        assert policy.agent_id is None
        assert policy.static_value == {"default": "value"}
        assert policy.skip is False
    
    def test_skip_fallback(self):
        """Test fallback with skip."""
        policy = FallbackPolicy(skip=True)
        assert policy.agent_id is None
        assert policy.static_value is None
        assert policy.skip is True
    
    def test_multiple_options_raises_error(self):
        """Test that setting multiple options raises error."""
        with pytest.raises(ValueError):
            FallbackPolicy(agent_id="agent", static_value={"foo": "bar"})
        
        with pytest.raises(ValueError):
            FallbackPolicy(agent_id="agent", skip=True)
    
    def test_to_dict(self):
        """Test serialization to dict."""
        policy = FallbackPolicy(agent_id="fallback-agent")
        data = policy.to_dict()
        assert data["agentId"] == "fallback-agent"
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {"agentId": "fallback-agent"}
        policy = FallbackPolicy.from_dict(data)
        assert policy.agent_id == "fallback-agent"


# =============================================================================
# StepDefinition Tests
# =============================================================================


class TestStepDefinition:
    """Tests for StepDefinition dataclass."""
    
    def test_minimal_step(self):
        """Test step with minimal required fields."""
        step = StepDefinition(id="step1", agent_id="agent1")
        assert step.id == "step1"
        assert step.agent_id == "agent1"
        assert step.input_map == {}
        assert step.depends_on == []
    
    def test_step_with_input_map(self):
        """Test step with input mapping."""
        step = StepDefinition(
            id="step1",
            agent_id="agent1",
            input_map={"query": "$.input.topic"},
        )
        assert step.input_map["query"] == "$.input.topic"
    
    def test_step_with_dependencies(self):
        """Test step with dependencies."""
        step = StepDefinition(
            id="step2",
            agent_id="agent2",
            depends_on=["step1"],
        )
        assert "step1" in step.depends_on
    
    def test_step_with_retry_policy(self):
        """Test step with retry policy."""
        retry = RetryPolicy(max_attempts=5)
        step = StepDefinition(
            id="step1",
            agent_id="agent1",
            retry=retry,
        )
        assert step.retry.max_attempts == 5
    
    def test_empty_id_raises_error(self):
        """Test that empty ID raises error."""
        with pytest.raises(ValueError):
            StepDefinition(id="", agent_id="agent1")
    
    def test_empty_agent_id_raises_error(self):
        """Test that empty agent_id raises error."""
        with pytest.raises(ValueError):
            StepDefinition(id="step1", agent_id="")
    
    def test_invalid_input_key_raises_error(self):
        """Test that invalid input key raises error."""
        with pytest.raises(ValueError):
            StepDefinition(
                id="step1",
                agent_id="agent1",
                input_map={"123invalid": "$.input.x"},
            )
    
    def test_to_dict(self):
        """Test serialization to dict."""
        step = StepDefinition(
            id="step1",
            agent_id="agent1",
            input_map={"query": "$.input.topic"},
            depends_on=["step0"],
        )
        data = step.to_dict()
        assert data["id"] == "step1"
        assert data["agentId"] == "agent1"
        assert data["inputMap"]["query"] == "$.input.topic"
        assert "step0" in data["dependsOn"]
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "id": "step1",
            "agentId": "agent1",
            "inputMap": {"query": "$.input.topic"},
        }
        step = StepDefinition.from_dict(data)
        assert step.id == "step1"
        assert step.agent_id == "agent1"


# =============================================================================
# WorkflowDefinition Tests
# =============================================================================


class TestWorkflowDefinition:
    """Tests for WorkflowDefinition dataclass."""
    
    def test_minimal_workflow(self):
        """Test workflow with minimal required fields."""
        workflow = WorkflowDefinition(
            id="wf1",
            name="Test Workflow",
            steps=[StepDefinition(id="step1", agent_id="agent1")],
        )
        assert workflow.id == "wf1"
        assert workflow.name == "Test Workflow"
        assert len(workflow.steps) == 1
    
    def test_workflow_with_multiple_steps(self):
        """Test workflow with multiple steps."""
        workflow = WorkflowDefinition(
            id="wf1",
            name="Multi-step Workflow",
            steps=[
                StepDefinition(id="step1", agent_id="agent1"),
                StepDefinition(id="step2", agent_id="agent2", depends_on=["step1"]),
            ],
        )
        assert len(workflow.steps) == 2
    
    def test_empty_id_raises_error(self):
        """Test that empty ID raises error."""
        with pytest.raises(ValueError):
            WorkflowDefinition(
                id="",
                name="Test",
                steps=[StepDefinition(id="s1", agent_id="a1")],
            )
    
    def test_empty_name_raises_error(self):
        """Test that empty name raises error."""
        with pytest.raises(ValueError):
            WorkflowDefinition(
                id="wf1",
                name="",
                steps=[StepDefinition(id="s1", agent_id="a1")],
            )
    
    def test_empty_steps_raises_error(self):
        """Test that empty steps raises error."""
        with pytest.raises(ValueError):
            WorkflowDefinition(id="wf1", name="Test", steps=[])
    
    def test_duplicate_step_ids_raises_error(self):
        """Test that duplicate step IDs raise error."""
        with pytest.raises(ValueError):
            WorkflowDefinition(
                id="wf1",
                name="Test",
                steps=[
                    StepDefinition(id="step1", agent_id="agent1"),
                    StepDefinition(id="step1", agent_id="agent2"),  # Duplicate!
                ],
            )
    
    def test_invalid_dependency_raises_error(self):
        """Test that invalid dependency raises error."""
        with pytest.raises(ValueError):
            WorkflowDefinition(
                id="wf1",
                name="Test",
                steps=[
                    StepDefinition(
                        id="step1",
                        agent_id="agent1",
                        depends_on=["nonexistent"],
                    ),
                ],
            )
    
    def test_get_step(self):
        """Test get_step method."""
        workflow = WorkflowDefinition(
            id="wf1",
            name="Test",
            steps=[
                StepDefinition(id="step1", agent_id="agent1"),
                StepDefinition(id="step2", agent_id="agent2"),
            ],
        )
        assert workflow.get_step("step1") is not None
        assert workflow.get_step("step1").agent_id == "agent1"
        assert workflow.get_step("nonexistent") is None
    
    def test_to_dict(self):
        """Test serialization to dict."""
        workflow = WorkflowDefinition(
            id="wf1",
            name="Test",
            steps=[StepDefinition(id="step1", agent_id="agent1")],
        )
        data = workflow.to_dict()
        assert data["id"] == "wf1"
        assert data["name"] == "Test"
        assert len(data["steps"]) == 1
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "id": "wf1",
            "name": "Test",
            "steps": [{"id": "step1", "agentId": "agent1"}],
        }
        workflow = WorkflowDefinition.from_dict(data)
        assert workflow.id == "wf1"
        assert len(workflow.steps) == 1


# =============================================================================
# StepResult Tests
# =============================================================================


class TestStepResult:
    """Tests for StepResult dataclass."""
    
    def test_successful_result(self):
        """Test successful step result."""
        result = StepResult(
            step_id="step1",
            status=StepStatus.COMPLETED,
            output={"data": "value"},
        )
        assert result.step_id == "step1"
        assert result.status == StepStatus.COMPLETED
        assert result.output == {"data": "value"}
    
    def test_failed_result(self):
        """Test failed step result."""
        result = StepResult(
            step_id="step1",
            status=StepStatus.FAILED,
            error={"message": "Something went wrong"},
            error_category=ErrorCategory.INTERNAL_ERROR,
        )
        assert result.status == StepStatus.FAILED
        assert result.error["message"] == "Something went wrong"
    
    def test_result_with_metrics(self):
        """Test result with execution metrics."""
        result = StepResult(
            step_id="step1",
            status=StepStatus.COMPLETED,
            execution_time_ms=1500,
            retry_count=2,
        )
        assert result.execution_time_ms == 1500
        assert result.retry_count == 2
    
    def test_to_dict(self):
        """Test serialization to dict."""
        result = StepResult(
            step_id="step1",
            status=StepStatus.COMPLETED,
            output={"data": "value"},
        )
        data = result.to_dict()
        assert data["stepId"] == "step1"
        assert data["status"] == "completed"


# =============================================================================
# WorkflowResult Tests
# =============================================================================


class TestWorkflowResult:
    """Tests for WorkflowResult dataclass."""
    
    def test_successful_result(self):
        """Test successful workflow result."""
        result = WorkflowResult(
            execution_id="exec-1",
            workflow_id="wf-1",
            status=ExecutionStatus.COMPLETED,
            input={"topic": "AI"},
            output={"result": "done"},
        )
        assert result.execution_id == "exec-1"
        assert result.status == ExecutionStatus.COMPLETED
    
    def test_result_with_step_results(self):
        """Test workflow result with step results."""
        step_result = StepResult(
            step_id="step1",
            status=StepStatus.COMPLETED,
        )
        result = WorkflowResult(
            execution_id="exec-1",
            workflow_id="wf-1",
            status=ExecutionStatus.COMPLETED,
            input={},
            step_results={"step1": step_result},
        )
        assert "step1" in result.step_results
    
    def test_to_dict(self):
        """Test serialization to dict."""
        result = WorkflowResult(
            execution_id="exec-1",
            workflow_id="wf-1",
            status=ExecutionStatus.COMPLETED,
            input={"topic": "AI"},
        )
        data = result.to_dict()
        assert data["executionId"] == "exec-1"
        assert data["workflowId"] == "wf-1"


# =============================================================================
# WorkflowEvent Tests
# =============================================================================


class TestWorkflowEvent:
    """Tests for WorkflowEvent dataclass."""
    
    def test_workflow_started_event(self):
        """Test workflow started event."""
        event = WorkflowEvent(
            type=WorkflowEventType.WORKFLOW_STARTED,
            execution_id="exec-1",
            workflow_id="wf-1",
        )
        assert event.type == WorkflowEventType.WORKFLOW_STARTED
        assert event.execution_id == "exec-1"
    
    def test_step_event(self):
        """Test step-level event."""
        event = WorkflowEvent(
            type=WorkflowEventType.STEP_COMPLETED,
            execution_id="exec-1",
            workflow_id="wf-1",
            step_id="step1",
            data={"output": {"result": "done"}},
        )
        assert event.step_id == "step1"
        assert "output" in event.data
    
    def test_to_dict(self):
        """Test serialization to dict."""
        event = WorkflowEvent(
            type=WorkflowEventType.STEP_STARTED,
            execution_id="exec-1",
            workflow_id="wf-1",
            step_id="step1",
        )
        data = event.to_dict()
        assert data["type"] == "step.started"
        assert data["stepId"] == "step1"


# =============================================================================
# ExecutionContext Tests
# =============================================================================


class TestExecutionContext:
    """Tests for ExecutionContext dataclass."""
    
    def test_create_context(self):
        """Test creating execution context."""
        workflow = WorkflowDefinition(
            id="wf-1",
            name="Test",
            steps=[StepDefinition(id="s1", agent_id="a1")],
        )
        context = ExecutionContext.create(
            workflow=workflow,
            input={"topic": "AI"},
        )
        assert context.workflow == workflow
        assert context.input == {"topic": "AI"}
        assert context.execution_id is not None
        assert context.status == ExecutionStatus.PENDING
    
    def test_create_with_trace_id(self):
        """Test creating context with trace ID."""
        workflow = WorkflowDefinition(
            id="wf-1",
            name="Test",
            steps=[StepDefinition(id="s1", agent_id="a1")],
        )
        context = ExecutionContext.create(
            workflow=workflow,
            input={},
            trace_id="trace-123",
        )
        assert context.trace_id == "trace-123"
    
    def test_set_and_get_step_result(self):
        """Test storing and retrieving step results."""
        workflow = WorkflowDefinition(
            id="wf-1",
            name="Test",
            steps=[StepDefinition(id="s1", agent_id="a1")],
        )
        context = ExecutionContext.create(workflow=workflow, input={})
        
        result = StepResult(step_id="s1", status=StepStatus.COMPLETED)
        context.set_step_result(result)
        
        retrieved = context.get_step_result("s1")
        assert retrieved is not None
        assert retrieved.status == StepStatus.COMPLETED
    
    def test_is_step_completed(self):
        """Test checking if step is completed."""
        workflow = WorkflowDefinition(
            id="wf-1",
            name="Test",
            steps=[StepDefinition(id="s1", agent_id="a1")],
        )
        context = ExecutionContext.create(workflow=workflow, input={})
        
        assert context.is_step_completed("s1") is False
        
        context.set_step_result(StepResult(
            step_id="s1",
            status=StepStatus.COMPLETED,
        ))
        assert context.is_step_completed("s1") is True
    
    def test_are_dependencies_met(self):
        """Test checking if dependencies are met."""
        workflow = WorkflowDefinition(
            id="wf-1",
            name="Test",
            steps=[
                StepDefinition(id="s1", agent_id="a1"),
                StepDefinition(id="s2", agent_id="a2", depends_on=["s1"]),
            ],
        )
        context = ExecutionContext.create(workflow=workflow, input={})
        
        step2 = workflow.get_step("s2")
        assert context.are_dependencies_met(step2) is False
        
        context.set_step_result(StepResult(
            step_id="s1",
            status=StepStatus.COMPLETED,
        ))
        assert context.are_dependencies_met(step2) is True
