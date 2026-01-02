"""
Tests for AWF Orchestration Validation and Mapping Modules

Comprehensive tests for workflow validation and JSONPath input/output mapping.
"""

import pytest
from datetime import datetime, timezone

from awf.orchestration.types import (
    StepDefinition,
    WorkflowDefinition,
    StepResult,
    StepStatus,
    ExecutionContext,
    RetryPolicy,
    FallbackPolicy,
)
from awf.orchestration.validation import (
    ValidationError,
    ValidationResult,
    WorkflowValidator,
    validate_workflow,
    is_valid_workflow,
)
from awf.orchestration.mapping import (
    JSONPathError,
    InputMapper,
    OutputMapper,
    ExpressionValidator,
)


# =============================================================================
# Helper Functions
# =============================================================================


def make_workflow(steps_config: list[tuple[str, str, list[str]]]) -> WorkflowDefinition:
    """Create a workflow from simple config."""
    steps = [
        StepDefinition(id=step_id, agent_id=agent_id, depends_on=depends_on)
        for step_id, agent_id, depends_on in steps_config
    ]
    return WorkflowDefinition(id="test-workflow", name="Test Workflow", steps=steps)


def make_context(
    workflow: WorkflowDefinition,
    input_data: dict,
    step_results: dict[str, StepResult] = None,
) -> ExecutionContext:
    """Create an execution context for testing."""
    context = ExecutionContext.create(workflow=workflow, input=input_data)
    if step_results:
        for result in step_results.values():
            context.set_step_result(result)
    return context


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""
    
    def test_valid_result(self):
        """Test creating a valid result."""
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.error_count == 0
        assert result.warning_count == 0
    
    def test_add_error_invalidates(self):
        """Test adding an error invalidates the result."""
        result = ValidationResult(valid=True)
        result.add_error("Something wrong")
        
        assert result.valid is False
        assert result.error_count == 1
    
    def test_add_warning_keeps_valid(self):
        """Test adding a warning keeps result valid."""
        result = ValidationResult(valid=True)
        result.add_warning("Minor issue")
        
        assert result.valid is True
        assert result.warning_count == 1
    
    def test_error_with_step_id(self):
        """Test error with step context."""
        result = ValidationResult(valid=True)
        result.add_error("Step error", step_id="step1", field="agentId")
        
        assert result.errors[0].step_id == "step1"
        assert result.errors[0].field == "agentId"
    
    def test_to_dict(self):
        """Test serialization."""
        result = ValidationResult(valid=True)
        result.add_error("Error 1")
        result.add_warning("Warning 1")
        
        data = result.to_dict()
        assert data["valid"] is False
        assert data["errorCount"] == 1
        assert data["warningCount"] == 1


# =============================================================================
# WorkflowValidator Tests
# =============================================================================


class TestWorkflowValidator:
    """Tests for WorkflowValidator."""
    
    def test_validate_minimal_valid_workflow(self):
        """Test validation of minimal valid workflow."""
        workflow = make_workflow([("step1", "agent1", [])])
        result = validate_workflow(workflow)
        
        assert result.valid is True
        assert result.error_count == 0
    
    def test_validate_complex_valid_workflow(self):
        """Test validation of complex valid workflow."""
        workflow = WorkflowDefinition(
            id="complex-workflow",
            name="Complex Workflow",
            steps=[
                StepDefinition(
                    id="step1",
                    agent_id="agent1",
                    input_map={"query": "$.input.topic"},
                    retry=RetryPolicy(max_attempts=3),
                ),
                StepDefinition(
                    id="step2",
                    agent_id="agent2",
                    input_map={"data": "$.steps.step1.output.result"},
                    depends_on=["step1"],
                    fallback=FallbackPolicy(skip=True),
                ),
            ],
            timeout_ms=60000,
            default_retry=RetryPolicy(max_attempts=2),
        )
        result = validate_workflow(workflow)
        
        assert result.valid is True


class TestValidatorWorkflowFields:
    """Tests for workflow-level field validation."""
    
    def test_empty_workflow_id(self):
        """Test that empty workflow ID is caught."""
        # Can't test directly because WorkflowDefinition validates
        # But we can test the validator directly
        validator = WorkflowValidator()
        
        # Create a mock workflow with empty ID by bypassing __post_init__
        step = StepDefinition(id="s1", agent_id="a1")
        workflow = object.__new__(WorkflowDefinition)
        workflow.id = ""
        workflow.name = "Test"
        workflow.steps = [step]
        workflow.version = "1.0.0"
        workflow.description = None
        workflow.input_schema = None
        workflow.output_map = None
        workflow.timeout_ms = None
        workflow.default_retry = None
        workflow.metadata = {}
        workflow.created_at = datetime.now(timezone.utc)
        
        result = validator.validate(workflow)
        assert result.valid is False
        assert any("ID" in e.message for e in result.errors)
    
    def test_negative_timeout(self):
        """Test that negative timeout is caught."""
        step = StepDefinition(id="s1", agent_id="a1")
        workflow = object.__new__(WorkflowDefinition)
        workflow.id = "test"
        workflow.name = "Test"
        workflow.steps = [step]
        workflow.version = "1.0.0"
        workflow.description = None
        workflow.input_schema = None
        workflow.output_map = None
        workflow.timeout_ms = -1000
        workflow.default_retry = None
        workflow.metadata = {}
        workflow.created_at = datetime.now(timezone.utc)
        
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        
        assert any("timeout" in e.message.lower() for e in result.errors)


class TestValidatorStepFields:
    """Tests for step-level field validation."""
    
    def test_reserved_step_id(self):
        """Test that reserved step IDs are rejected."""
        # "input", "output", "context", "steps" are reserved
        for reserved_id in ["input", "output", "context", "steps"]:
            step = StepDefinition(id=reserved_id, agent_id="agent")
            workflow = object.__new__(WorkflowDefinition)
            workflow.id = "test"
            workflow.name = "Test"
            workflow.steps = [step]
            workflow.version = "1.0.0"
            workflow.description = None
            workflow.input_schema = None
            workflow.output_map = None
            workflow.timeout_ms = None
            workflow.default_retry = None
            workflow.metadata = {}
            workflow.created_at = datetime.now(timezone.utc)
            
            validator = WorkflowValidator()
            result = validator.validate(workflow)
            
            assert any("reserved" in e.message.lower() for e in result.errors)
    
    def test_step_negative_timeout(self):
        """Test that negative step timeout is caught."""
        step = StepDefinition(id="s1", agent_id="a1", timeout_ms=-100)
        workflow = object.__new__(WorkflowDefinition)
        workflow.id = "test"
        workflow.name = "Test"
        workflow.steps = [step]
        workflow.version = "1.0.0"
        workflow.description = None
        workflow.input_schema = None
        workflow.output_map = None
        workflow.timeout_ms = None
        workflow.default_retry = None
        workflow.metadata = {}
        workflow.created_at = datetime.now(timezone.utc)
        
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        
        assert any("timeout" in e.message.lower() for e in result.errors)


class TestValidatorRetryPolicy:
    """Tests for retry policy validation."""
    
    def test_invalid_max_attempts(self):
        """Test that max_attempts < 1 is caught."""
        retry = RetryPolicy(max_attempts=0)
        step = StepDefinition(id="s1", agent_id="a1", retry=retry)
        workflow = object.__new__(WorkflowDefinition)
        workflow.id = "test"
        workflow.name = "Test"
        workflow.steps = [step]
        workflow.version = "1.0.0"
        workflow.description = None
        workflow.input_schema = None
        workflow.output_map = None
        workflow.timeout_ms = None
        workflow.default_retry = None
        workflow.metadata = {}
        workflow.created_at = datetime.now(timezone.utc)
        
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        
        assert any("max_attempts" in e.message for e in result.errors)
    
    def test_invalid_jitter_factor(self):
        """Test that jitter_factor outside 0-1 is caught."""
        retry = RetryPolicy(jitter_factor=1.5)
        step = StepDefinition(id="s1", agent_id="a1", retry=retry)
        workflow = object.__new__(WorkflowDefinition)
        workflow.id = "test"
        workflow.name = "Test"
        workflow.steps = [step]
        workflow.version = "1.0.0"
        workflow.description = None
        workflow.input_schema = None
        workflow.output_map = None
        workflow.timeout_ms = None
        workflow.default_retry = None
        workflow.metadata = {}
        workflow.created_at = datetime.now(timezone.utc)
        
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        
        assert any("jitter" in e.message for e in result.errors)


class TestValidatorInputMappings:
    """Tests for input mapping validation."""
    
    def test_valid_input_mapping(self):
        """Test valid input mapping expressions."""
        workflow = WorkflowDefinition(
            id="test",
            name="Test",
            steps=[
                StepDefinition(
                    id="step1",
                    agent_id="agent1",
                    input_map={"query": "$.input.topic"},
                ),
                StepDefinition(
                    id="step2",
                    agent_id="agent2",
                    input_map={"data": "$.steps.step1.output.result"},
                    depends_on=["step1"],
                ),
            ],
        )
        result = validate_workflow(workflow)
        assert result.valid is True
    
    def test_reference_to_unavailable_step(self):
        """Test that referencing a step that hasn't run yet is caught."""
        workflow = WorkflowDefinition(
            id="test",
            name="Test",
            steps=[
                StepDefinition(
                    id="step1",
                    agent_id="agent1",
                    # Referencing step2 which runs after step1!
                    input_map={"data": "$.steps.step2.output.result"},
                ),
                StepDefinition(
                    id="step2",
                    agent_id="agent2",
                    depends_on=["step1"],
                ),
            ],
        )
        result = validate_workflow(workflow)
        
        # Should have error about referencing step2 from step1
        assert any("step2" in e.message and "step1" in str(e.step_id) 
                   for e in result.errors)


class TestValidatorBestPractices:
    """Tests for best practice warnings."""
    
    def test_no_retry_policy_warning(self):
        """Test warning when no retry policy configured."""
        workflow = make_workflow([("step1", "agent1", [])])
        result = validate_workflow(workflow)
        
        # Should have warning about missing retry
        assert any("retry" in w.message.lower() for w in result.warnings)
    
    def test_single_step_warning(self):
        """Test warning for single-step workflows."""
        workflow = make_workflow([("step1", "agent1", [])])
        result = validate_workflow(workflow)
        
        # Should have warning about single step
        assert any("only one step" in w.message.lower() for w in result.warnings)


class TestValidatorStrictMode:
    """Tests for strict validation mode."""
    
    def test_strict_promotes_warnings(self):
        """Test that strict mode promotes warnings to errors."""
        workflow = make_workflow([("step1", "agent1", [])])
        
        # Non-strict: valid with warnings
        result = validate_workflow(workflow, strict=False)
        assert result.valid is True
        assert result.warning_count > 0
        
        # Strict: invalid because warnings become errors
        result = validate_workflow(workflow, strict=True)
        assert result.valid is False


class TestIsValidWorkflow:
    """Tests for is_valid_workflow convenience function."""
    
    def test_valid_workflow(self):
        """Test returns True for valid workflow."""
        workflow = make_workflow([("step1", "agent1", [])])
        assert is_valid_workflow(workflow) is True
    
    def test_invalid_workflow(self):
        """Test returns False for invalid workflow."""
        # Create workflow with issue
        step = StepDefinition(id="input", agent_id="agent")  # Reserved ID
        workflow = object.__new__(WorkflowDefinition)
        workflow.id = "test"
        workflow.name = "Test"
        workflow.steps = [step]
        workflow.version = "1.0.0"
        workflow.description = None
        workflow.input_schema = None
        workflow.output_map = None
        workflow.timeout_ms = None
        workflow.default_retry = None
        workflow.metadata = {}
        workflow.created_at = datetime.now(timezone.utc)
        
        assert is_valid_workflow(workflow) is False


# =============================================================================
# JSONPath Expression Validator Tests
# =============================================================================


class TestExpressionValidator:
    """Tests for JSONPath expression validation."""
    
    def test_valid_input_expression(self):
        """Test valid $.input.X expressions."""
        valid, error = ExpressionValidator.validate("$.input.topic")
        assert valid is True
        assert error is None
    
    def test_valid_steps_expression(self):
        """Test valid $.steps.X.output.Y expressions."""
        valid, error = ExpressionValidator.validate("$.steps.step1.output.result")
        assert valid is True
    
    def test_valid_context_expression(self):
        """Test valid $.context.X expressions."""
        valid, error = ExpressionValidator.validate("$.context.userId")
        assert valid is True
    
    def test_valid_nested_expression(self):
        """Test valid nested path expressions."""
        valid, error = ExpressionValidator.validate("$.steps.step1.output.data.items[0].name")
        assert valid is True
    
    def test_literal_value_valid(self):
        """Test that non-$ values are treated as literals."""
        valid, error = ExpressionValidator.validate("literal value")
        assert valid is True
    
    def test_empty_expression_invalid(self):
        """Test empty expression is invalid."""
        valid, error = ExpressionValidator.validate("")
        assert valid is False
        assert "empty" in error.lower()
    
    def test_invalid_root(self):
        """Test invalid root path."""
        valid, error = ExpressionValidator.validate("$.invalid.path")
        assert valid is False
    
    def test_validate_input_map(self):
        """Test validating entire input map."""
        input_map = {
            "query": "$.input.topic",
            "data": "$.steps.step1.output.result",
        }
        errors = ExpressionValidator.validate_input_map(input_map, ["step1"])
        assert errors == []
    
    def test_validate_input_map_unknown_step(self):
        """Test input map referencing unknown step."""
        input_map = {
            "data": "$.steps.unknown.output.result",
        }
        errors = ExpressionValidator.validate_input_map(input_map, ["step1"])
        assert len(errors) > 0
        assert any("unknown" in e for e in errors)


# =============================================================================
# InputMapper Tests
# =============================================================================


class TestInputMapper:
    """Tests for InputMapper."""
    
    def test_map_input_expression(self):
        """Test mapping $.input.X expression."""
        workflow = make_workflow([("step1", "agent1", [])])
        context = make_context(workflow, {"topic": "AI safety"})
        
        mapper = InputMapper(context)
        result = mapper.map_inputs({"query": "$.input.topic"})
        
        assert result["query"] == "AI safety"
    
    def test_map_step_output_expression(self):
        """Test mapping $.steps.X.output.Y expression."""
        workflow = make_workflow([
            ("step1", "agent1", []),
            ("step2", "agent2", ["step1"]),
        ])
        step1_result = StepResult(
            step_id="step1",
            status=StepStatus.COMPLETED,
            output={"result": "research data"},
        )
        context = make_context(workflow, {}, {"step1": step1_result})
        
        mapper = InputMapper(context)
        result = mapper.map_inputs({"data": "$.steps.step1.output.result"})
        
        assert result["data"] == "research data"
    
    def test_map_nested_output(self):
        """Test mapping nested output paths."""
        workflow = make_workflow([("step1", "agent1", [])])
        step1_result = StepResult(
            step_id="step1",
            status=StepStatus.COMPLETED,
            output={
                "data": {
                    "items": [
                        {"name": "first"},
                        {"name": "second"},
                    ]
                }
            },
        )
        context = make_context(workflow, {}, {"step1": step1_result})
        
        mapper = InputMapper(context)
        result = mapper.map_inputs({"first_name": "$.steps.step1.output.data.items[0].name"})
        
        assert result["first_name"] == "first"
    
    def test_map_context_expression(self):
        """Test mapping $.context.X expression."""
        workflow = make_workflow([("step1", "agent1", [])])
        context = ExecutionContext.create(
            workflow=workflow,
            input={},
            context={"userId": "user123"},
        )
        
        mapper = InputMapper(context)
        result = mapper.map_inputs({"user": "$.context.userId"})
        
        assert result["user"] == "user123"
    
    def test_map_literal_value(self):
        """Test that non-$ values are returned as literals."""
        workflow = make_workflow([("step1", "agent1", [])])
        context = make_context(workflow, {})
        
        mapper = InputMapper(context)
        result = mapper.map_inputs({"fixed": "constant value"})
        
        assert result["fixed"] == "constant value"
    
    def test_map_missing_path_non_strict(self):
        """Test missing path returns None in non-strict mode."""
        workflow = make_workflow([("step1", "agent1", [])])
        context = make_context(workflow, {})
        
        mapper = InputMapper(context)
        result = mapper.map_inputs({"missing": "$.input.nonexistent"}, strict=False)
        
        assert result["missing"] is None
    
    def test_map_missing_path_strict_raises(self):
        """Test missing path raises in strict mode."""
        workflow = make_workflow([("step1", "agent1", [])])
        context = make_context(workflow, {})
        
        mapper = InputMapper(context)
        with pytest.raises(JSONPathError):
            mapper.map_inputs({"missing": "$.input.nonexistent"}, strict=True)
    
    def test_map_step_status(self):
        """Test mapping step status."""
        workflow = make_workflow([("step1", "agent1", [])])
        step1_result = StepResult(
            step_id="step1",
            status=StepStatus.COMPLETED,
            output={"data": "value"},
        )
        context = make_context(workflow, {}, {"step1": step1_result})
        
        mapper = InputMapper(context)
        result = mapper.map_inputs({"status": "$.steps.step1.status"})
        
        assert result["status"] == "completed"
    
    def test_map_array_index_out_of_range(self):
        """Test array index out of range raises error."""
        workflow = make_workflow([("step1", "agent1", [])])
        step1_result = StepResult(
            step_id="step1",
            status=StepStatus.COMPLETED,
            output={"items": ["a", "b"]},
        )
        context = make_context(workflow, {}, {"step1": step1_result})
        
        mapper = InputMapper(context)
        with pytest.raises(JSONPathError) as exc:
            mapper.map_inputs({"item": "$.steps.step1.output.items[10]"}, strict=True)
        
        assert "out of range" in str(exc.value).lower()


# =============================================================================
# OutputMapper Tests
# =============================================================================


class TestOutputMapper:
    """Tests for OutputMapper."""
    
    def test_map_explicit_output(self):
        """Test mapping with explicit output_map."""
        workflow = WorkflowDefinition(
            id="test",
            name="Test",
            steps=[
                StepDefinition(id="step1", agent_id="agent1"),
                StepDefinition(id="step2", agent_id="agent2", depends_on=["step1"]),
            ],
            output_map={
                "summary": "$.steps.step2.output.summary",
                "count": "$.steps.step1.output.count",
            },
        )
        
        step1_result = StepResult(
            step_id="step1",
            status=StepStatus.COMPLETED,
            output={"count": 42},
        )
        step2_result = StepResult(
            step_id="step2",
            status=StepStatus.COMPLETED,
            output={"summary": "All done"},
        )
        context = make_context(
            workflow, {},
            {"step1": step1_result, "step2": step2_result},
        )
        
        mapper = OutputMapper(context)
        output = mapper.map_output(workflow.output_map)
        
        assert output["summary"] == "All done"
        assert output["count"] == 42
    
    def test_map_default_output_single_leaf(self):
        """Test default output returns last step's output."""
        workflow = make_workflow([
            ("step1", "agent1", []),
            ("step2", "agent2", ["step1"]),
        ])
        
        step1_result = StepResult(
            step_id="step1",
            status=StepStatus.COMPLETED,
            output={"intermediate": "data"},
        )
        step2_result = StepResult(
            step_id="step2",
            status=StepStatus.COMPLETED,
            output={"final": "result"},
        )
        context = make_context(
            workflow, {},
            {"step1": step1_result, "step2": step2_result},
        )
        
        mapper = OutputMapper(context)
        output = mapper.map_output(None)  # No explicit output_map
        
        # Should return step2's output (the leaf)
        assert output == {"final": "result"}
    
    def test_map_default_output_multiple_leaves(self):
        """Test default output with multiple leaves returns last completed."""
        workflow = make_workflow([
            ("step1", "agent1", []),
            ("step2", "agent2", ["step1"]),
            ("step3", "agent3", ["step1"]),  # Another branch
        ])
        
        step1_result = StepResult(
            step_id="step1",
            status=StepStatus.COMPLETED,
            output={"data": "1"},
        )
        step2_result = StepResult(
            step_id="step2",
            status=StepStatus.COMPLETED,
            output={"data": "2"},
        )
        step3_result = StepResult(
            step_id="step3",
            status=StepStatus.COMPLETED,
            output={"data": "3"},
        )
        context = make_context(
            workflow, {},
            {"step1": step1_result, "step2": step2_result, "step3": step3_result},
        )
        
        mapper = OutputMapper(context)
        output = mapper.map_output(None)
        
        # Should return one of the leaf outputs
        assert output in [{"data": "2"}, {"data": "3"}]


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestMappingEdgeCases:
    """Edge cases for mapping functionality."""
    
    def test_empty_input_map(self):
        """Test empty input map returns empty dict."""
        workflow = make_workflow([("step1", "agent1", [])])
        context = make_context(workflow, {"topic": "test"})
        
        mapper = InputMapper(context)
        result = mapper.map_inputs({})
        
        assert result == {}
    
    def test_deeply_nested_path(self):
        """Test deeply nested paths work correctly."""
        workflow = make_workflow([("step1", "agent1", [])])
        step1_result = StepResult(
            step_id="step1",
            status=StepStatus.COMPLETED,
            output={
                "level1": {
                    "level2": {
                        "level3": {
                            "level4": {
                                "value": "deep"
                            }
                        }
                    }
                }
            },
        )
        context = make_context(workflow, {}, {"step1": step1_result})
        
        mapper = InputMapper(context)
        result = mapper.map_inputs({
            "deep": "$.steps.step1.output.level1.level2.level3.level4.value"
        })
        
        assert result["deep"] == "deep"
    
    def test_special_characters_in_values(self):
        """Test values with special characters."""
        workflow = make_workflow([("step1", "agent1", [])])
        context = make_context(workflow, {
            "query": "What is $.input.topic?",  # Contains $. but as literal
        })
        
        mapper = InputMapper(context)
        result = mapper.map_inputs({"q": "$.input.query"})
        
        assert result["q"] == "What is $.input.topic?"
    
    def test_null_output_from_step(self):
        """Test handling of null/None output from step."""
        workflow = make_workflow([("step1", "agent1", [])])
        step1_result = StepResult(
            step_id="step1",
            status=StepStatus.COMPLETED,
            output=None,
        )
        context = make_context(workflow, {}, {"step1": step1_result})
        
        mapper = InputMapper(context)
        with pytest.raises(JSONPathError):
            mapper.map_inputs({"data": "$.steps.step1.output.something"}, strict=True)
