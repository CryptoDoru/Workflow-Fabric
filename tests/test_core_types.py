"""
Tests for AWF Core Types

Tests all ASP dataclasses and their serialization methods.
"""

from datetime import datetime
from uuid import UUID

import pytest

from awf.core.types import (
    AgentManifest,
    AgentStatus,
    Capability,
    CapabilityType,
    Event,
    EventType,
    Policy,
    PolicyViolation,
    SandboxTier,
    Schema,
    SchemaProperty,
    Task,
    TaskError,
    TaskMetrics,
    TaskResult,
    TaskStatus,
    TrustFactors,
    TrustScore,
    Workflow,
    WorkflowExecution,
    WorkflowStep,
)


class TestSchemaProperty:
    """Tests for SchemaProperty dataclass."""

    def test_create_required_property(self):
        prop = SchemaProperty(
            name="query",
            type="string",
            required=True,
        )
        assert prop.name == "query"
        assert prop.type == "string"
        assert prop.required is True

    def test_create_optional_property_with_default(self):
        prop = SchemaProperty(
            name="limit",
            type="integer",
            required=False,
            default=10,
        )
        assert prop.default == 10
        assert prop.required is False

    def test_create_enum_property(self):
        prop = SchemaProperty(
            name="status",
            type="string",
            enum=["active", "inactive", "pending"],
        )
        assert prop.enum == ["active", "inactive", "pending"]


class TestSchema:
    """Tests for Schema dataclass."""

    def test_create_empty_schema(self):
        schema = Schema()
        assert schema.type == "object"
        assert schema.properties == {}
        assert schema.required == []

    def test_create_schema_with_properties(self, sample_schema):
        assert "query" in sample_schema.properties
        assert "max_results" in sample_schema.properties
        assert sample_schema.required == ["query"]

    def test_to_json_schema(self, sample_schema):
        json_schema = sample_schema.to_json_schema()
        
        assert json_schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert json_schema["type"] == "object"
        assert "properties" in json_schema
        assert "query" in json_schema["properties"]
        assert json_schema["required"] == ["query"]

    def test_to_json_schema_property_types(self, sample_schema):
        json_schema = sample_schema.to_json_schema()
        
        assert json_schema["properties"]["query"]["type"] == "string"
        assert json_schema["properties"]["max_results"]["type"] == "integer"
        assert json_schema["properties"]["max_results"]["default"] == 10


class TestCapability:
    """Tests for Capability dataclass."""

    def test_create_tool_capability(self, sample_capability):
        assert sample_capability.name == "web_search"
        assert sample_capability.type == CapabilityType.TOOL
        assert "network:external" in sample_capability.permissions

    def test_to_dict(self, sample_capability):
        data = sample_capability.to_dict()
        
        assert data["name"] == "web_search"
        assert data["type"] == "tool"
        assert "description" in data
        assert data["permissions"] == ["network:external"]

    def test_capability_types(self):
        for cap_type in CapabilityType:
            cap = Capability(name="test", type=cap_type)
            assert cap.type == cap_type


class TestAgentManifest:
    """Tests for AgentManifest dataclass."""

    def test_create_manifest(self, sample_manifest):
        assert sample_manifest.id == "test-agent-001"
        assert sample_manifest.name == "Test Research Agent"
        assert sample_manifest.framework == "langgraph"
        assert sample_manifest.status == AgentStatus.ACTIVE

    def test_manifest_auto_timestamps(self):
        manifest = AgentManifest(
            id="test",
            name="Test",
            version="1.0.0",
            framework="test",
        )
        assert manifest.registered_at is not None
        assert manifest.updated_at is not None
        assert isinstance(manifest.registered_at, datetime)

    def test_to_dict(self, sample_manifest):
        data = sample_manifest.to_dict()
        
        assert data["asp_version"] == "1.0"
        assert data["id"] == "test-agent-001"
        assert data["framework"] == "langgraph"
        assert "capabilities" in data
        assert len(data["capabilities"]) == 3
        assert "registeredAt" in data

    def test_agent_statuses(self):
        for status in AgentStatus:
            manifest = AgentManifest(
                id="test",
                name="Test",
                version="1.0.0",
                framework="test",
                status=status,
            )
            assert manifest.status == status


class TestTask:
    """Tests for Task dataclass."""

    def test_create_task(self, sample_task):
        assert sample_task.agent_id == "test-agent-001"
        assert sample_task.input["query"] == "AI safety research"
        assert sample_task.timeout_ms == 30000

    def test_task_auto_id(self):
        task = Task(agent_id="test", input={})
        assert task.id is not None
        assert len(task.id) > 0

    def test_task_custom_id(self):
        task = Task(id="custom-id", agent_id="test", input={})
        assert task.id == "custom-id"

    def test_to_dict(self, sample_task):
        data = sample_task.to_dict()
        
        assert data["agentId"] == "test-agent-001"
        assert data["input"] == {"query": "AI safety research", "max_results": 5}
        assert data["timeoutMs"] == 30000
        assert data["traceId"] == "trace-abc-123"
        assert "createdAt" in data


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_create_completed_result(self, sample_task_result):
        assert sample_task_result.status == TaskStatus.COMPLETED
        assert sample_task_result.output is not None
        assert sample_task_result.error is None

    def test_create_failed_result(self, sample_failed_result):
        assert sample_failed_result.status == TaskStatus.FAILED
        assert sample_failed_result.error is not None
        assert sample_failed_result.error.code == "EXECUTION_ERROR"

    def test_to_dict_completed(self, sample_task_result):
        data = sample_task_result.to_dict()
        
        assert data["status"] == "completed"
        assert "output" in data
        assert "metrics" in data
        assert data["metrics"]["executionTimeMs"] == 1500

    def test_to_dict_failed(self, sample_failed_result):
        data = sample_failed_result.to_dict()
        
        assert data["status"] == "failed"
        assert "error" in data
        assert data["error"]["code"] == "EXECUTION_ERROR"

    def test_task_statuses(self):
        for status in TaskStatus:
            result = TaskResult(
                task_id="test",
                agent_id="test",
                status=status,
            )
            assert result.status == status


class TestTaskMetrics:
    """Tests for TaskMetrics dataclass."""

    def test_create_metrics(self):
        metrics = TaskMetrics(
            execution_time_ms=1500,
            token_usage={"input_tokens": 100, "output_tokens": 200},
        )
        assert metrics.execution_time_ms == 1500
        assert metrics.token_usage["input_tokens"] == 100

    def test_to_dict(self):
        metrics = TaskMetrics(
            execution_time_ms=1500,
            sandbox_tier=SandboxTier.WASM,
            sandbox_overhead_ms=10,
        )
        data = metrics.to_dict()
        
        assert data["executionTimeMs"] == 1500
        assert data["sandboxTier"] == "wasm"
        assert data["sandboxOverheadMs"] == 10


class TestTaskError:
    """Tests for TaskError dataclass."""

    def test_create_error(self):
        error = TaskError(
            code="TIMEOUT",
            message="Task timed out",
            retryable=True,
        )
        assert error.code == "TIMEOUT"
        assert error.retryable is True

    def test_to_dict(self):
        error = TaskError(
            code="EXECUTION_ERROR",
            message="Failed to execute",
            details={"step": "researcher"},
            stack_trace="Traceback...",
            retryable=False,
        )
        data = error.to_dict()
        
        assert data["code"] == "EXECUTION_ERROR"
        assert data["message"] == "Failed to execute"
        assert data["details"] == {"step": "researcher"}
        assert "stackTrace" in data
        assert data["retryable"] is False


class TestEvent:
    """Tests for Event dataclass."""

    def test_create_event(self, sample_event):
        assert sample_event.type == EventType.TASK_COMPLETED
        assert sample_event.source == "test-agent-001"

    def test_event_auto_id(self):
        event = Event(type=EventType.TASK_STARTED, source="test")
        assert event.id is not None

    def test_to_dict(self, sample_event):
        data = sample_event.to_dict()
        
        assert data["type"] == "task.completed"
        assert data["source"] == "test-agent-001"
        assert "timestamp" in data
        assert "id" in data

    def test_event_types(self):
        for event_type in EventType:
            event = Event(type=event_type, source="test")
            assert event.type == event_type


class TestTrustFactors:
    """Tests for TrustFactors dataclass."""

    def test_create_trust_factors(self, sample_trust_factors):
        assert sample_trust_factors.publisher_trust == 0.9
        assert sample_trust_factors.audit_status == 0.8

    def test_compute_score(self, sample_trust_factors):
        score = sample_trust_factors.compute_score()
        
        # Expected: 0.9*0.25 + 0.8*0.25 + 0.7*0.20 + 0.85*0.15 + 0.95*0.15
        # = 0.225 + 0.2 + 0.14 + 0.1275 + 0.1425 = 0.835
        assert 0.83 <= score <= 0.84

    def test_to_dict(self, sample_trust_factors):
        data = sample_trust_factors.to_dict()
        
        assert "publisherTrust" in data
        assert "computedScore" in data
        assert data["publisherTrust"] == 0.9


class TestTrustScore:
    """Tests for TrustScore dataclass."""

    def test_compute_from_factors(self, sample_trust_factors):
        trust_score = TrustScore.compute(sample_trust_factors)
        
        assert 0.83 <= trust_score.score <= 0.84
        assert trust_score.sandbox_tier == SandboxTier.GVISOR  # 0.70-0.89

    def test_sandbox_tier_wasm(self):
        factors = TrustFactors(
            publisher_trust=1.0,
            audit_status=1.0,
            community_trust=1.0,
            permission_analysis=1.0,
            historical_behavior=1.0,
        )
        trust_score = TrustScore.compute(factors)
        assert trust_score.sandbox_tier == SandboxTier.WASM

    def test_sandbox_tier_blocked(self):
        factors = TrustFactors(
            publisher_trust=0.1,
            audit_status=0.1,
            community_trust=0.1,
            permission_analysis=0.1,
            historical_behavior=0.1,
        )
        trust_score = TrustScore.compute(factors)
        assert trust_score.sandbox_tier == SandboxTier.BLOCKED

    def test_to_dict(self, sample_trust_score):
        data = sample_trust_score.to_dict()
        
        assert "score" in data
        assert "factors" in data
        assert "sandboxTier" in data
        assert "computedAt" in data


class TestWorkflowStep:
    """Tests for WorkflowStep dataclass."""

    def test_create_step(self, sample_workflow_step):
        assert sample_workflow_step.id == "step-1"
        assert sample_workflow_step.agent_id == "test-agent-001"
        assert "$.input.topic" in sample_workflow_step.input_map.values()

    def test_to_dict(self, sample_workflow_step):
        data = sample_workflow_step.to_dict()
        
        assert data["id"] == "step-1"
        assert data["agentId"] == "test-agent-001"
        assert "inputMap" in data


class TestWorkflow:
    """Tests for Workflow dataclass."""

    def test_create_workflow(self, sample_workflow):
        assert sample_workflow.id == "test-workflow-001"
        assert len(sample_workflow.steps) == 2

    def test_to_dict(self, sample_workflow):
        data = sample_workflow.to_dict()
        
        assert data["id"] == "test-workflow-001"
        assert data["name"] == "Test Research Workflow"
        assert len(data["steps"]) == 2
        assert "createdAt" in data


class TestPolicy:
    """Tests for Policy dataclass."""

    def test_create_policy(self, sample_policy):
        assert sample_policy.id == "policy-001"
        assert sample_policy.min_trust_score == 0.8
        assert "production" in sample_policy.environments

    def test_to_dict(self, sample_policy):
        data = sample_policy.to_dict()
        
        assert data["id"] == "policy-001"
        assert data["name"] == "Production Minimum Trust"
        assert data["minTrustScore"] == 0.8
        assert data["denyCapabilities"] == ["process:execute"]
        assert data["allowNetwork"] is True
        assert data["allowFilesystem"] is False


class TestPolicyViolation:
    """Tests for PolicyViolation dataclass."""

    def test_create_violation(self):
        violation = PolicyViolation(
            policy_id="policy-001",
            policy_name="Production Minimum Trust",
            agent_id="untrusted-agent",
            violation_type="trust_score_below_minimum",
            details={"required": 0.8, "actual": 0.5},
        )
        
        assert violation.policy_id == "policy-001"
        assert violation.violation_type == "trust_score_below_minimum"

    def test_to_dict(self):
        violation = PolicyViolation(
            policy_id="policy-001",
            policy_name="Test Policy",
            agent_id="test-agent",
            violation_type="capability_denied",
            details={"capability": "process:execute"},
        )
        data = violation.to_dict()
        
        assert data["policyId"] == "policy-001"
        assert data["agentId"] == "test-agent"
        assert data["violationType"] == "capability_denied"
        assert "timestamp" in data


class TestEnumerations:
    """Tests for all enumeration types."""

    def test_agent_status_values(self):
        assert AgentStatus.REGISTERED.value == "registered"
        assert AgentStatus.ACTIVE.value == "active"
        assert AgentStatus.SUSPENDED.value == "suspended"
        assert AgentStatus.DEPRECATED.value == "deprecated"

    def test_task_status_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"
        assert TaskStatus.TIMEOUT.value == "timeout"

    def test_capability_type_values(self):
        assert CapabilityType.TOOL.value == "tool"
        assert CapabilityType.REASONING.value == "reasoning"
        assert CapabilityType.MEMORY.value == "memory"
        assert CapabilityType.COMMUNICATION.value == "communication"
        assert CapabilityType.CUSTOM.value == "custom"

    def test_sandbox_tier_values(self):
        assert SandboxTier.WASM.value == "wasm"
        assert SandboxTier.GVISOR.value == "gvisor"
        assert SandboxTier.GVISOR_STRICT.value == "gvisor_strict"
        assert SandboxTier.BLOCKED.value == "blocked"

    def test_event_type_values(self):
        assert EventType.TASK_STARTED.value == "task.started"
        assert EventType.TASK_COMPLETED.value == "task.completed"
        assert EventType.TASK_FAILED.value == "task.failed"
        assert EventType.STATE_CHANGED.value == "state.changed"
