"""
Tests for the Orchestrator - Full workflow execution integration tests.

These tests verify end-to-end workflow execution including:
- Sequential and parallel step execution
- Condition evaluation
- DAG ordering
- Event emission
- Cancellation
- Error handling
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from awf.core.types import (
    AgentManifest,
    AgentStatus,
    Capability,
    CapabilityType,
    Event,
    Task,
    TaskResult,
    TaskStatus,
)
from awf.orchestration import (
    ExecutionContext,
    ExecutionStatus,
    FallbackPolicy,
    Orchestrator,
    OrchestratorConfig,
    OrchestrationAdapterRegistry,
    RetryPolicy,
    StepDefinition,
    StepStatus,
    WorkflowDefinition,
    WorkflowEvent,
    WorkflowEventType,
    WorkflowResult,
)
from awf.orchestration.errors import (
    WorkflowCancelledError,
    WorkflowDefinitionError,
    WorkflowTimeoutError,
)


# =============================================================================
# Mock Implementations
# =============================================================================


class MockAdapter:
    """Mock adapter for testing workflow execution."""
    
    framework_name = "mock"
    framework_version = "1.0.0"
    
    def __init__(self):
        self.execute_calls: List[Task] = []
        self._responses: Dict[str, TaskResult] = {}
        self._delays: Dict[str, float] = {}
        self._errors: Dict[str, Exception] = {}
        self._manifests: Dict[str, AgentManifest] = {}
    
    def set_response(
        self,
        agent_id: str,
        output: Dict[str, Any],
        delay_ms: float = 0,
    ) -> None:
        """Configure the response for an agent."""
        self._responses[agent_id] = TaskResult(
            task_id="test",
            agent_id=agent_id,
            status=TaskStatus.COMPLETED,
            output=output,
        )
        if delay_ms > 0:
            self._delays[agent_id] = delay_ms
    
    def set_error(self, agent_id: str, error: Exception) -> None:
        """Configure an error for an agent."""
        self._errors[agent_id] = error
    
    def register_agent(self, manifest: AgentManifest) -> None:
        """Register an agent."""
        self._manifests[manifest.id] = manifest
    
    async def execute(self, task: Task) -> TaskResult:
        """Execute a task."""
        self.execute_calls.append(task)
        
        # Check for delay
        if task.agent_id in self._delays:
            await asyncio.sleep(self._delays[task.agent_id] / 1000.0)
        
        # Check for error
        if task.agent_id in self._errors:
            raise self._errors[task.agent_id]
        
        # Return configured response or default
        if task.agent_id in self._responses:
            resp = self._responses[task.agent_id]
            return TaskResult(
                task_id=task.id,
                agent_id=task.agent_id,
                status=resp.status,
                output=resp.output,
            )
        
        return TaskResult(
            task_id=task.id,
            agent_id=task.agent_id,
            status=TaskStatus.COMPLETED,
            output={"result": f"output from {task.agent_id}"},
        )
    
    async def execute_streaming(self, task: Task) -> AsyncIterator[Event]:
        """Execute with streaming."""
        result = await self.execute(task)
        yield Event(
            type="state.changed",
            agent_id=task.agent_id,
            task_id=task.id,
            data={"output": result.output},
            source="mock",
        )
    
    async def cancel(self, task_id: str) -> bool:
        """Cancel a task."""
        return True
    
    def get_manifest(self, agent_id: str) -> Optional[AgentManifest]:
        """Get agent manifest."""
        return self._manifests.get(agent_id)
    
    def list_agents(self) -> List[AgentManifest]:
        """List all agents."""
        return list(self._manifests.values())
    
    def health_check(self) -> Dict[str, Any]:
        """Return health status."""
        return {"healthy": True, "framework": "mock"}


class MockAgentRegistry:
    """Mock agent registry for testing."""
    
    def __init__(self):
        self._agents: Dict[str, AgentManifest] = {}
    
    def register(self, manifest: AgentManifest) -> None:
        """Register an agent."""
        self._agents[manifest.id] = manifest
    
    async def get(self, agent_id: str) -> Optional[AgentManifest]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)


class MockAdapterRegistry:
    """Mock adapter registry for testing."""
    
    def __init__(self, adapter: MockAdapter):
        self._adapter = adapter
    
    def get_adapter(self, framework: str) -> Optional[MockAdapter]:
        """Get the mock adapter."""
        return self._adapter
    
    def get_agent_framework(self, agent_id: str) -> Optional[str]:
        """All agents use mock framework."""
        return "mock"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_adapter() -> MockAdapter:
    """Create a mock adapter."""
    return MockAdapter()


@pytest.fixture
def mock_agent_registry(mock_adapter: MockAdapter) -> MockAgentRegistry:
    """Create a mock agent registry with sample agents."""
    registry = MockAgentRegistry()
    
    agents = ["research-agent", "writer-agent", "reviewer-agent", "summarizer-agent"]
    for agent_id in agents:
        manifest = AgentManifest(
            id=agent_id,
            name=agent_id.replace("-", " ").title(),
            version="1.0.0",
            framework="mock",
            status=AgentStatus.ACTIVE,
            capabilities=[
                Capability(
                    name="test",
                    type=CapabilityType.TOOL,
                    description="Test capability",
                )
            ],
        )
        registry.register(manifest)
        mock_adapter.register_agent(manifest)
    
    return registry


@pytest.fixture
def mock_adapter_registry(mock_adapter: MockAdapter) -> MockAdapterRegistry:
    """Create a mock adapter registry."""
    return MockAdapterRegistry(mock_adapter)


@pytest.fixture
def orchestrator(
    mock_adapter_registry: MockAdapterRegistry,
    mock_agent_registry: MockAgentRegistry,
) -> Orchestrator:
    """Create an orchestrator."""
    return Orchestrator(
        adapter_registry=mock_adapter_registry,
        agent_registry=mock_agent_registry,
        config=OrchestratorConfig(
            default_timeout_ms=30000,
            validate_on_execute=True,
            emit_events=False,
        ),
    )


# =============================================================================
# Basic Execution Tests
# =============================================================================


class TestOrchestratorBasicExecution:
    """Tests for basic workflow execution."""
    
    @pytest.mark.asyncio
    async def test_execute_single_step_workflow(
        self,
        orchestrator: Orchestrator,
        mock_adapter: MockAdapter,
    ):
        """Test executing a single-step workflow."""
        mock_adapter.set_response("research-agent", {"research": "AI safety findings"})
        
        workflow = WorkflowDefinition(
            id="simple-workflow",
            name="Simple Workflow",
            steps=[
                StepDefinition(
                    id="research",
                    agent_id="research-agent",
                    input_map={"query": "$.input.topic"},
                ),
            ],
            output_map={"result": "$.steps.research.output.research"},
        )
        
        result = await orchestrator.execute(
            workflow=workflow,
            input_data={"topic": "AI safety"},
        )
        
        assert result.status == ExecutionStatus.COMPLETED
        assert result.output == {"result": "AI safety findings"}
        assert "research" in result.step_results
        assert result.step_results["research"].status == StepStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_sequential_workflow(
        self,
        orchestrator: Orchestrator,
        mock_adapter: MockAdapter,
    ):
        """Test executing a sequential multi-step workflow."""
        mock_adapter.set_response("research-agent", {"findings": "research data"})
        mock_adapter.set_response("writer-agent", {"article": "written content"})
        
        workflow = WorkflowDefinition(
            id="sequential-workflow",
            name="Sequential Workflow",
            steps=[
                StepDefinition(
                    id="research",
                    agent_id="research-agent",
                    input_map={"query": "$.input.topic"},
                ),
                StepDefinition(
                    id="write",
                    agent_id="writer-agent",
                    input_map={"research": "$.steps.research.output.findings"},
                    depends_on=["research"],
                ),
            ],
        )
        
        result = await orchestrator.execute(
            workflow=workflow,
            input_data={"topic": "AI safety"},
        )
        
        assert result.status == ExecutionStatus.COMPLETED
        assert "research" in result.step_results
        assert "write" in result.step_results
        
        # Verify execution order
        research_call = next(c for c in mock_adapter.execute_calls if c.agent_id == "research-agent")
        write_call = next(c for c in mock_adapter.execute_calls if c.agent_id == "writer-agent")
        
        research_idx = mock_adapter.execute_calls.index(research_call)
        write_idx = mock_adapter.execute_calls.index(write_call)
        assert research_idx < write_idx, "Research should execute before write"
    
    @pytest.mark.asyncio
    async def test_execute_parallel_workflow(
        self,
        orchestrator: Orchestrator,
        mock_adapter: MockAdapter,
    ):
        """Test executing a workflow with parallel steps."""
        mock_adapter.set_response("research-agent", {"data": "research"}, delay_ms=50)
        mock_adapter.set_response("reviewer-agent", {"review": "review data"}, delay_ms=50)
        mock_adapter.set_response("summarizer-agent", {"summary": "combined"})
        
        workflow = WorkflowDefinition(
            id="parallel-workflow",
            name="Parallel Workflow",
            steps=[
                StepDefinition(
                    id="research",
                    agent_id="research-agent",
                    input_map={},
                ),
                StepDefinition(
                    id="review",
                    agent_id="reviewer-agent",
                    input_map={},
                ),
                StepDefinition(
                    id="summarize",
                    agent_id="summarizer-agent",
                    input_map={
                        "research": "$.steps.research.output",
                        "review": "$.steps.review.output",
                    },
                    depends_on=["research", "review"],
                ),
            ],
        )
        
        start = datetime.now(timezone.utc)
        result = await orchestrator.execute(
            workflow=workflow,
            input_data={},
        )
        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.step_results) == 3
        
        # Parallel execution should take less than 2x the individual step time
        # (research and review can run in parallel)
        # Allow for some overhead
        assert elapsed_ms < 200, f"Expected parallel execution, took {elapsed_ms}ms"
    
    @pytest.mark.asyncio
    async def test_workflow_with_context(
        self,
        orchestrator: Orchestrator,
        mock_adapter: MockAdapter,
    ):
        """Test workflow execution with context data."""
        mock_adapter.set_response("research-agent", {"result": "success"})
        
        workflow = WorkflowDefinition(
            id="context-workflow",
            name="Context Workflow",
            steps=[
                StepDefinition(
                    id="step1",
                    agent_id="research-agent",
                    input_map={"context_val": "$.context.api_key"},
                ),
            ],
        )
        
        result = await orchestrator.execute(
            workflow=workflow,
            input_data={},
            context={"api_key": "secret-key", "user_id": "user-123"},
        )
        
        assert result.status == ExecutionStatus.COMPLETED


# =============================================================================
# Condition Tests
# =============================================================================


class TestOrchestratorConditions:
    """Tests for conditional step execution."""
    
    @pytest.mark.asyncio
    async def test_condition_skip_step(
        self,
        orchestrator: Orchestrator,
        mock_adapter: MockAdapter,
    ):
        """Test that a step is skipped when condition is false."""
        mock_adapter.set_response("research-agent", {"data": "research"})
        mock_adapter.set_response("reviewer-agent", {"review": "review"})
        
        workflow = WorkflowDefinition(
            id="conditional-workflow",
            name="Conditional Workflow",
            steps=[
                StepDefinition(
                    id="research",
                    agent_id="research-agent",
                    input_map={},
                ),
                StepDefinition(
                    id="review",
                    agent_id="reviewer-agent",
                    input_map={},
                    depends_on=["research"],
                    # Only run if input includes review_needed=true
                    condition='$.input.review_needed == True',
                ),
            ],
        )
        
        result = await orchestrator.execute(
            workflow=workflow,
            input_data={"review_needed": False},
        )
        
        assert result.status == ExecutionStatus.COMPLETED
        assert "research" in result.step_results
        assert result.step_results["research"].status == StepStatus.COMPLETED
        assert "review" in result.step_results
        assert result.step_results["review"].status == StepStatus.SKIPPED
    
    @pytest.mark.asyncio
    async def test_condition_run_step(
        self,
        orchestrator: Orchestrator,
        mock_adapter: MockAdapter,
    ):
        """Test that a step runs when condition is true."""
        mock_adapter.set_response("research-agent", {"data": "research"})
        mock_adapter.set_response("reviewer-agent", {"review": "review"})
        
        workflow = WorkflowDefinition(
            id="conditional-workflow",
            name="Conditional Workflow",
            steps=[
                StepDefinition(
                    id="research",
                    agent_id="research-agent",
                    input_map={},
                ),
                StepDefinition(
                    id="review",
                    agent_id="reviewer-agent",
                    input_map={},
                    depends_on=["research"],
                    condition='$.input.review_needed == True',
                ),
            ],
        )
        
        result = await orchestrator.execute(
            workflow=workflow,
            input_data={"review_needed": True},
        )
        
        assert result.status == ExecutionStatus.COMPLETED
        assert result.step_results["review"].status == StepStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_condition_based_on_step_output(
        self,
        orchestrator: Orchestrator,
        mock_adapter: MockAdapter,
    ):
        """Test condition based on previous step output."""
        mock_adapter.set_response("research-agent", {"quality": "high"})
        mock_adapter.set_response("reviewer-agent", {"review": "reviewed"})
        
        workflow = WorkflowDefinition(
            id="output-conditional-workflow",
            name="Output Conditional",
            steps=[
                StepDefinition(
                    id="research",
                    agent_id="research-agent",
                    input_map={},
                ),
                StepDefinition(
                    id="review",
                    agent_id="reviewer-agent",
                    input_map={},
                    depends_on=["research"],
                    # Only review if research quality is not high
                    condition='$.steps.research.output.quality != "high"',
                ),
            ],
        )
        
        result = await orchestrator.execute(
            workflow=workflow,
            input_data={},
        )
        
        # Research returned high quality, so review should be skipped
        assert result.status == ExecutionStatus.COMPLETED
        assert result.step_results["review"].status == StepStatus.SKIPPED


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestOrchestratorErrorHandling:
    """Tests for error handling."""
    
    @pytest.mark.asyncio
    async def test_step_failure_without_fallback(
        self,
        orchestrator: Orchestrator,
        mock_adapter: MockAdapter,
    ):
        """Test workflow failure when step fails without fallback."""
        mock_adapter.set_error("research-agent", RuntimeError("Agent failed"))
        
        workflow = WorkflowDefinition(
            id="failing-workflow",
            name="Failing Workflow",
            steps=[
                StepDefinition(
                    id="research",
                    agent_id="research-agent",
                    input_map={},
                ),
            ],
        )
        
        result = await orchestrator.execute(
            workflow=workflow,
            input_data={},
        )
        
        assert result.status == ExecutionStatus.FAILED
        assert result.step_results["research"].status == StepStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_step_failure_with_fallback_skip(
        self,
        orchestrator: Orchestrator,
        mock_adapter: MockAdapter,
    ):
        """Test step failure with skip fallback."""
        mock_adapter.set_error("research-agent", RuntimeError("Agent failed"))
        mock_adapter.set_response("writer-agent", {"article": "written"})
        
        workflow = WorkflowDefinition(
            id="fallback-workflow",
            name="Fallback Workflow",
            steps=[
                StepDefinition(
                    id="research",
                    agent_id="research-agent",
                    input_map={},
                    fallback=FallbackPolicy(skip=True),
                ),
                StepDefinition(
                    id="write",
                    agent_id="writer-agent",
                    input_map={},
                    depends_on=["research"],
                ),
            ],
        )
        
        result = await orchestrator.execute(
            workflow=workflow,
            input_data={},
        )
        
        assert result.status == ExecutionStatus.COMPLETED
        assert result.step_results["research"].status == StepStatus.SKIPPED
        assert result.step_results["write"].status == StepStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_invalid_workflow_raises_error(
        self,
        orchestrator: Orchestrator,
    ):
        """Test that invalid workflow raises ValueError at construction."""
        # WorkflowDefinition validates at construction time, not execution
        with pytest.raises(ValueError, match="depends on unknown step"):
            WorkflowDefinition(
                id="invalid-workflow",
                name="Invalid",
                steps=[
                    StepDefinition(
                        id="step1",
                        agent_id="research-agent",
                        depends_on=["nonexistent"],  # Invalid dependency
                    ),
                ],
            )
    
    @pytest.mark.asyncio
    async def test_dependency_failure_affects_dependent_steps(
        self,
        orchestrator: Orchestrator,
        mock_adapter: MockAdapter,
    ):
        """Test behavior when a dependency fails."""
        mock_adapter.set_error("research-agent", RuntimeError("Research failed"))
        mock_adapter.set_response("writer-agent", {"article": "written"})
        
        workflow = WorkflowDefinition(
            id="dep-failure-workflow",
            name="Dependency Failure",
            steps=[
                StepDefinition(
                    id="research",
                    agent_id="research-agent",
                    input_map={},
                ),
                StepDefinition(
                    id="write",
                    agent_id="writer-agent",
                    input_map={},
                    depends_on=["research"],
                ),
            ],
        )
        
        result = await orchestrator.execute(
            workflow=workflow,
            input_data={},
        )
        
        # Overall workflow should fail
        assert result.status == ExecutionStatus.FAILED
        assert result.step_results["research"].status == StepStatus.FAILED
        # Write depends on failed research - check that it's handled
        # (behavior may vary: could be SKIPPED, FAILED, or NOT_RUN)
        assert "write" in result.step_results


# =============================================================================
# Timeout Tests
# =============================================================================


class TestOrchestratorTimeouts:
    """Tests for timeout handling."""
    
    @pytest.mark.asyncio
    async def test_workflow_timeout(
        self,
        mock_adapter_registry: MockAdapterRegistry,
        mock_agent_registry: MockAgentRegistry,
        mock_adapter: MockAdapter,
    ):
        """Test workflow timeout."""
        orchestrator = Orchestrator(
            adapter_registry=mock_adapter_registry,
            agent_registry=mock_agent_registry,
            config=OrchestratorConfig(
                default_timeout_ms=100,  # 100ms timeout
                emit_events=False,
            ),
        )
        
        mock_adapter.set_response("research-agent", {"data": "result"}, delay_ms=500)
        
        workflow = WorkflowDefinition(
            id="slow-workflow",
            name="Slow Workflow",
            steps=[
                StepDefinition(
                    id="research",
                    agent_id="research-agent",
                    input_map={},
                ),
            ],
        )
        
        result = await orchestrator.execute(
            workflow=workflow,
            input_data={},
        )
        
        assert result.status == ExecutionStatus.TIMEOUT
    
    @pytest.mark.asyncio
    async def test_custom_workflow_timeout(
        self,
        orchestrator: Orchestrator,
        mock_adapter: MockAdapter,
    ):
        """Test workflow with custom timeout."""
        mock_adapter.set_response("research-agent", {"data": "result"}, delay_ms=200)
        
        workflow = WorkflowDefinition(
            id="custom-timeout-workflow",
            name="Custom Timeout",
            steps=[
                StepDefinition(
                    id="research",
                    agent_id="research-agent",
                    input_map={},
                ),
            ],
        )
        
        result = await orchestrator.execute(
            workflow=workflow,
            input_data={},
            timeout_ms=50,  # 50ms timeout, step takes 200ms
        )
        
        assert result.status == ExecutionStatus.TIMEOUT


# =============================================================================
# Event Tests
# =============================================================================


class TestOrchestratorEvents:
    """Tests for event emission."""
    
    @pytest.mark.asyncio
    async def test_events_emitted_during_execution(
        self,
        mock_adapter_registry: MockAdapterRegistry,
        mock_agent_registry: MockAgentRegistry,
        mock_adapter: MockAdapter,
    ):
        """Test that events are emitted during workflow execution."""
        events: List[WorkflowEvent] = []
        
        async def capture_event(event: WorkflowEvent) -> None:
            events.append(event)
        
        orchestrator = Orchestrator(
            adapter_registry=mock_adapter_registry,
            agent_registry=mock_agent_registry,
            config=OrchestratorConfig(emit_events=True),
            event_callback=capture_event,
        )
        
        mock_adapter.set_response("research-agent", {"data": "result"})
        
        workflow = WorkflowDefinition(
            id="events-workflow",
            name="Events Workflow",
            steps=[
                StepDefinition(
                    id="research",
                    agent_id="research-agent",
                    input_map={},
                ),
            ],
        )
        
        await orchestrator.execute(
            workflow=workflow,
            input_data={},
        )
        
        # Should have workflow and step events
        event_types = [e.type for e in events]
        
        assert WorkflowEventType.WORKFLOW_STARTED in event_types
        assert WorkflowEventType.STEP_STARTED in event_types
        assert WorkflowEventType.STEP_COMPLETED in event_types
        assert WorkflowEventType.WORKFLOW_COMPLETED in event_types


# =============================================================================
# Cancellation Tests
# =============================================================================


class TestOrchestratorCancellation:
    """Tests for workflow cancellation."""
    
    @pytest.mark.asyncio
    async def test_cancel_workflow(
        self,
        mock_adapter_registry: MockAdapterRegistry,
        mock_agent_registry: MockAgentRegistry,
        mock_adapter: MockAdapter,
    ):
        """Test cancelling a running workflow."""
        orchestrator = Orchestrator(
            adapter_registry=mock_adapter_registry,
            agent_registry=mock_agent_registry,
            config=OrchestratorConfig(emit_events=False),
        )
        
        mock_adapter.set_response("research-agent", {"data": "result"}, delay_ms=1000)
        
        workflow = WorkflowDefinition(
            id="cancel-workflow",
            name="Cancel Workflow",
            steps=[
                StepDefinition(
                    id="research",
                    agent_id="research-agent",
                    input_map={},
                ),
            ],
        )
        
        # Start execution in background
        async def execute_and_track():
            return await orchestrator.execute(
                workflow=workflow,
                input_data={},
            )
        
        task = asyncio.create_task(execute_and_track())
        
        # Wait a bit then cancel
        await asyncio.sleep(0.05)
        
        # Get execution ID and cancel
        if orchestrator._active_executions:
            exec_id = list(orchestrator._active_executions.keys())[0]
            await orchestrator.cancel(exec_id)
        
        result = await task
        
        assert result.status == ExecutionStatus.CANCELLED


# =============================================================================
# Output Mapping Tests
# =============================================================================


class TestOrchestratorOutputMapping:
    """Tests for workflow output mapping."""
    
    @pytest.mark.asyncio
    async def test_output_mapping(
        self,
        orchestrator: Orchestrator,
        mock_adapter: MockAdapter,
    ):
        """Test workflow output mapping."""
        mock_adapter.set_response("research-agent", {"findings": "AI research"})
        mock_adapter.set_response("writer-agent", {"article": "Written article"})
        
        workflow = WorkflowDefinition(
            id="output-mapping-workflow",
            name="Output Mapping",
            steps=[
                StepDefinition(
                    id="research",
                    agent_id="research-agent",
                    input_map={},
                ),
                StepDefinition(
                    id="write",
                    agent_id="writer-agent",
                    input_map={},
                    depends_on=["research"],
                ),
            ],
            output_map={
                "research_findings": "$.steps.research.output.findings",
                "final_article": "$.steps.write.output.article",
            },
        )
        
        result = await orchestrator.execute(
            workflow=workflow,
            input_data={},
        )
        
        assert result.status == ExecutionStatus.COMPLETED
        assert result.output == {
            "research_findings": "AI research",
            "final_article": "Written article",
        }


# =============================================================================
# Complex Workflow Tests
# =============================================================================


class TestOrchestratorComplexWorkflows:
    """Tests for complex workflow patterns."""
    
    @pytest.mark.asyncio
    async def test_diamond_dependency_workflow(
        self,
        orchestrator: Orchestrator,
        mock_adapter: MockAdapter,
    ):
        """Test diamond-shaped dependency graph (A -> B,C -> D)."""
        mock_adapter.set_response("research-agent", {"data": "research"})
        mock_adapter.set_response("reviewer-agent", {"review": "review"})
        mock_adapter.set_response("writer-agent", {"article": "article"})
        mock_adapter.set_response("summarizer-agent", {"summary": "summary"})
        
        workflow = WorkflowDefinition(
            id="diamond-workflow",
            name="Diamond Workflow",
            steps=[
                StepDefinition(
                    id="start",
                    agent_id="research-agent",
                    input_map={},
                ),
                StepDefinition(
                    id="path-a",
                    agent_id="reviewer-agent",
                    input_map={},
                    depends_on=["start"],
                ),
                StepDefinition(
                    id="path-b",
                    agent_id="writer-agent",
                    input_map={},
                    depends_on=["start"],
                ),
                StepDefinition(
                    id="merge",
                    agent_id="summarizer-agent",
                    input_map={},
                    depends_on=["path-a", "path-b"],
                ),
            ],
        )
        
        result = await orchestrator.execute(
            workflow=workflow,
            input_data={},
        )
        
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.step_results) == 4
        
        # All steps should complete
        for step_id, step_result in result.step_results.items():
            assert step_result.status == StepStatus.COMPLETED, f"Step {step_id} should complete"
    
    @pytest.mark.asyncio
    async def test_single_step_workflow_as_minimal(
        self,
        orchestrator: Orchestrator,
        mock_adapter: MockAdapter,
    ):
        """Test that a single-step workflow is the minimum valid workflow."""
        # Empty workflows are not allowed - verify single step works
        mock_adapter.set_response("research-agent", {"data": "result"})
        
        workflow = WorkflowDefinition(
            id="minimal-workflow",
            name="Minimal Workflow",
            steps=[
                StepDefinition(
                    id="single",
                    agent_id="research-agent",
                    input_map={},
                ),
            ],
        )
        
        result = await orchestrator.execute(
            workflow=workflow,
            input_data={},
        )
        
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.step_results) == 1
