"""
Tests for the StepExecutor and AdapterRegistry.

These tests verify retry, fallback, timeout, and event emission behavior
using mock adapters.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from awf.core.types import (
    AgentManifest,
    AgentStatus,
    Capability,
    CapabilityType,
    Event,
    EventType,
    Task,
    TaskResult,
    TaskStatus,
)
from awf.orchestration import (
    AdapterProtocol,
    AdapterRegistry,
    AgentRegistryProtocol,
    AsyncEventCallback,
    ErrorCategory,
    ExecutionContext,
    FallbackPolicy,
    OrchestrationAdapterRegistry,
    RetryPolicy,
    StepDefinition,
    StepExecutor,
    StepExecutorConfig,
    StepStatus,
    WorkflowDefinition,
    WorkflowEvent,
    WorkflowEventType,
    create_adapter_registry,
)
from awf.orchestration.errors import (
    AdapterNotFoundError,
    AgentNotFoundError,
    FallbackError,
    InputMappingError,
    NoFallbackAvailableError,
    StepExecutionError,
    StepTimeoutError,
)


# =============================================================================
# Mock Implementations
# =============================================================================


class MockAdapter:
    """Mock adapter for testing."""
    
    framework_name = "mock"
    framework_version = "1.0.0"
    
    def __init__(
        self,
        execute_result: Optional[TaskResult] = None,
        execute_error: Optional[Exception] = None,
        execute_delay_ms: int = 0,
    ):
        self.execute_result = execute_result or TaskResult(
            task_id="test",
            agent_id="test-agent",
            status=TaskStatus.COMPLETED,
            output={"result": "success"},
        )
        self.execute_error = execute_error
        self.execute_delay_ms = execute_delay_ms
        self.execute_calls: List[Task] = []
        self.cancel_calls: List[str] = []
        self._registered_agents: Dict[str, AgentManifest] = {}
        self._manifests: Dict[str, AgentManifest] = {}
    
    def register_agent(self, manifest: AgentManifest) -> None:
        """Register an agent for testing."""
        self._registered_agents[manifest.id] = manifest
        self._manifests[manifest.id] = manifest
    
    async def execute(self, task: Task) -> TaskResult:
        """Execute a task."""
        self.execute_calls.append(task)
        
        if self.execute_delay_ms > 0:
            await asyncio.sleep(self.execute_delay_ms / 1000.0)
        
        if self.execute_error:
            raise self.execute_error
        
        return TaskResult(
            task_id=task.id,
            agent_id=task.agent_id,
            status=self.execute_result.status,
            output=self.execute_result.output,
            error=self.execute_result.error,
        )
    
    async def execute_streaming(self, task: Task) -> AsyncIterator[Event]:
        """Execute with streaming."""
        self.execute_calls.append(task)
        
        if self.execute_delay_ms > 0:
            await asyncio.sleep(self.execute_delay_ms / 1000.0)
        
        if self.execute_error:
            raise self.execute_error
        
        yield Event(
            type=EventType.STATE_CHANGED,
            agent_id=task.agent_id,
            task_id=task.id,
            data={"output": self.execute_result.output},
        )
    
    async def cancel(self, task_id: str) -> bool:
        """Cancel a task."""
        self.cancel_calls.append(task_id)
        return True
    
    def get_manifest(self, agent_id: str) -> Optional[AgentManifest]:
        """Get agent manifest."""
        return self._manifests.get(agent_id)
    
    def list_agents(self) -> List[AgentManifest]:
        """List all agents."""
        return list(self._manifests.values())
    
    def health_check(self) -> Dict[str, Any]:
        """Return health status."""
        return {
            "healthy": True,
            "framework": self.framework_name,
            "framework_version": self.framework_version,
            "registered_agents": len(self._registered_agents),
        }


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
    
    def __init__(self):
        self._adapters: Dict[str, MockAdapter] = {}
        self._agent_framework: Dict[str, str] = {}
    
    def register(self, framework: str, adapter: MockAdapter) -> None:
        """Register an adapter."""
        self._adapters[framework] = adapter
    
    def set_agent_framework(self, agent_id: str, framework: str) -> None:
        """Set the framework for an agent."""
        self._agent_framework[agent_id] = framework
    
    def get_adapter(self, framework: str) -> Optional[MockAdapter]:
        """Get an adapter by framework."""
        return self._adapters.get(framework)
    
    def get_agent_framework(self, agent_id: str) -> Optional[str]:
        """Get the framework for an agent."""
        return self._agent_framework.get(agent_id)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agent() -> AgentManifest:
    """Create a mock agent manifest."""
    return AgentManifest(
        id="test-agent",
        name="Test Agent",
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


@pytest.fixture
def mock_adapter() -> MockAdapter:
    """Create a mock adapter."""
    return MockAdapter()


@pytest.fixture
def mock_adapter_registry(mock_adapter: MockAdapter) -> MockAdapterRegistry:
    """Create a mock adapter registry."""
    registry = MockAdapterRegistry()
    registry.register("mock", mock_adapter)
    return registry


@pytest.fixture
def mock_agent_registry(mock_agent: AgentManifest) -> MockAgentRegistry:
    """Create a mock agent registry."""
    registry = MockAgentRegistry()
    registry.register(mock_agent)
    return registry


@pytest.fixture
def sample_workflow() -> WorkflowDefinition:
    """Create a sample workflow."""
    return WorkflowDefinition(
        id="test-workflow",
        name="Test Workflow",
        steps=[
            StepDefinition(
                id="step1",
                agent_id="test-agent",
                input_map={"query": "$.input.topic"},
            ),
        ],
    )


@pytest.fixture
def execution_context(sample_workflow: WorkflowDefinition) -> ExecutionContext:
    """Create an execution context."""
    return ExecutionContext(
        execution_id="exec-001",
        workflow=sample_workflow,
        input={"topic": "AI safety"},
        correlation_id="corr-001",
    )


@pytest.fixture
def step_executor(
    mock_adapter_registry: MockAdapterRegistry,
    mock_agent_registry: MockAgentRegistry,
) -> StepExecutor:
    """Create a step executor."""
    # Set up the agent framework mapping
    mock_adapter_registry.set_agent_framework("test-agent", "mock")
    
    return StepExecutor(
        adapter_registry=mock_adapter_registry,
        agent_registry=mock_agent_registry,
        config=StepExecutorConfig(
            default_timeout_ms=5000,
            emit_events=True,
        ),
    )


# =============================================================================
# AdapterRegistry Tests
# =============================================================================


class TestOrchestrationAdapterRegistry:
    """Tests for the OrchestrationAdapterRegistry."""
    
    def test_register_adapter(self):
        """Test registering an adapter."""
        registry = OrchestrationAdapterRegistry()
        adapter = MockAdapter()
        
        registry.register_adapter("mock", adapter)
        
        assert registry.get_adapter("mock") is adapter
        assert registry.get_adapter("MOCK") is adapter  # Case insensitive
    
    def test_register_adapter_validation(self):
        """Test adapter registration validation."""
        registry = OrchestrationAdapterRegistry()
        adapter = MockAdapter()
        
        with pytest.raises(ValueError, match="Framework name cannot be empty"):
            registry.register_adapter("", adapter)
        
        with pytest.raises(ValueError, match="Adapter cannot be None"):
            registry.register_adapter("mock", None)
    
    def test_unregister_adapter(self):
        """Test unregistering an adapter."""
        registry = OrchestrationAdapterRegistry()
        adapter = MockAdapter()
        
        registry.register_adapter("mock", adapter)
        assert registry.unregister_adapter("mock") is True
        assert registry.get_adapter("mock") is None
        assert registry.unregister_adapter("mock") is False
    
    def test_enable_disable_adapter(self):
        """Test enabling and disabling adapters."""
        registry = OrchestrationAdapterRegistry()
        adapter = MockAdapter()
        
        registry.register_adapter("mock", adapter)
        
        assert registry.disable_adapter("mock") is True
        assert registry.get_adapter("mock") is None  # Disabled
        
        assert registry.enable_adapter("mock") is True
        assert registry.get_adapter("mock") is adapter  # Re-enabled
        
        assert registry.disable_adapter("nonexistent") is False
        assert registry.enable_adapter("nonexistent") is False
    
    def test_list_adapters(self):
        """Test listing adapters."""
        registry = OrchestrationAdapterRegistry()
        adapter1 = MockAdapter()
        adapter2 = MockAdapter()
        adapter2.framework_name = "other"
        
        registry.register_adapter("mock", adapter1, priority=10)
        registry.register_adapter("other", adapter2, priority=5)
        
        adapters = registry.list_adapters()
        assert len(adapters) == 2
        # Higher priority first
        assert adapters[0]["framework"] == "mock"
        assert adapters[0]["priority"] == 10
    
    def test_list_frameworks(self):
        """Test listing frameworks."""
        registry = OrchestrationAdapterRegistry()
        registry.register_adapter("mock", MockAdapter())
        registry.register_adapter("other", MockAdapter())
        
        frameworks = registry.list_frameworks()
        assert "mock" in frameworks
        assert "other" in frameworks
    
    def test_list_enabled_frameworks(self):
        """Test listing only enabled frameworks."""
        registry = OrchestrationAdapterRegistry()
        registry.register_adapter("mock", MockAdapter())
        registry.register_adapter("other", MockAdapter())
        
        registry.disable_adapter("other")
        
        enabled = registry.list_enabled_frameworks()
        assert "mock" in enabled
        assert "other" not in enabled
    
    def test_get_adapter_info(self):
        """Test getting adapter info."""
        registry = OrchestrationAdapterRegistry()
        adapter = MockAdapter()
        registry.register_adapter("mock", adapter, priority=10, metadata={"key": "value"})
        
        info = registry.get_adapter_info("mock")
        assert info is not None
        assert info["framework"] == "mock"
        assert info["enabled"] is True
        assert info["priority"] == 10
        assert info["metadata"] == {"key": "value"}
        assert "health" in info
        
        assert registry.get_adapter_info("nonexistent") is None
    
    def test_cache_agent_framework(self):
        """Test caching agent-framework mappings."""
        registry = OrchestrationAdapterRegistry()
        adapter = MockAdapter()
        registry.register_adapter("mock", adapter)
        
        registry.cache_agent_framework("agent-1", "mock")
        assert registry.get_agent_framework("agent-1") == "mock"
        
        registry.clear_cache()
        # Without cache, it should check adapters
        assert registry.get_agent_framework("agent-1") is None
    
    def test_get_agent_framework_from_adapter(self):
        """Test getting framework from adapter's registered agents."""
        registry = OrchestrationAdapterRegistry()
        adapter = MockAdapter()
        
        manifest = AgentManifest(
            id="test-agent",
            name="Test",
            version="1.0.0",
            framework="mock",
            status=AgentStatus.ACTIVE,
        )
        adapter.register_agent(manifest)
        
        registry.register_adapter("mock", adapter)
        
        framework = registry.get_agent_framework("test-agent")
        assert framework == "mock"
    
    def test_health_check(self):
        """Test health check across adapters."""
        registry = OrchestrationAdapterRegistry()
        registry.register_adapter("mock", MockAdapter())
        registry.register_adapter("other", MockAdapter())
        
        health = registry.health_check()
        assert health["healthy"] is True
        assert health["total_adapters"] == 2
        assert health["enabled_adapters"] == 2
        assert "mock" in health["adapters"]
        assert "other" in health["adapters"]
    
    def test_get_statistics(self):
        """Test getting registry statistics."""
        registry = OrchestrationAdapterRegistry()
        adapter = MockAdapter()
        adapter.register_agent(AgentManifest(
            id="agent1", name="Agent 1", version="1.0.0",
            framework="mock", status=AgentStatus.ACTIVE,
        ))
        registry.register_adapter("mock", adapter)
        
        stats = registry.get_statistics()
        assert stats["total_adapters"] == 1
        assert stats["enabled_adapters"] == 1
        assert stats["total_agents"] == 1
        assert stats["agents_by_framework"]["mock"] == 1


class TestCreateAdapterRegistry:
    """Tests for the create_adapter_registry helper."""
    
    def test_create_empty_registry(self):
        """Test creating an empty registry."""
        registry = create_adapter_registry()
        assert len(registry.list_frameworks()) == 0
    
    def test_create_with_adapters(self):
        """Test creating a registry with adapters."""
        adapters = {
            "mock": MockAdapter(),
            "other": MockAdapter(),
        }
        registry = create_adapter_registry(adapters=adapters)
        
        assert len(registry.list_frameworks()) == 2
        assert registry.get_adapter("mock") is not None
        assert registry.get_adapter("other") is not None


# =============================================================================
# StepExecutor Basic Tests
# =============================================================================


class TestStepExecutorBasic:
    """Basic tests for the StepExecutor."""
    
    @pytest.mark.asyncio
    async def test_execute_step_success(
        self,
        step_executor: StepExecutor,
        execution_context: ExecutionContext,
        mock_adapter: MockAdapter,
    ):
        """Test successful step execution."""
        step = StepDefinition(
            id="step1",
            agent_id="test-agent",
            input_map={"query": "$.input.topic"},
        )
        
        result = await step_executor.execute_step(step, execution_context)
        
        assert result.status == StepStatus.COMPLETED
        assert result.output == {"result": "success"}
        assert result.step_id == "step1"
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.execution_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_execute_step_agent_not_found(
        self,
        step_executor: StepExecutor,
        execution_context: ExecutionContext,
    ):
        """Test step execution with missing agent."""
        step = StepDefinition(
            id="step1",
            agent_id="nonexistent-agent",
            input_map={},
        )
        
        result = await step_executor.execute_step(step, execution_context)
        
        assert result.status == StepStatus.FAILED
        assert "AgentNotFoundError" in result.error.get("type", "")
    
    @pytest.mark.asyncio
    async def test_execute_step_cancelled(
        self,
        step_executor: StepExecutor,
        execution_context: ExecutionContext,
    ):
        """Test cancelling a step."""
        step = StepDefinition(
            id="step1",
            agent_id="test-agent",
            input_map={},
        )
        
        # Cancel before execution
        await step_executor.cancel_step("step1")
        
        result = await step_executor.execute_step(step, execution_context)
        
        assert result.status == StepStatus.SKIPPED
        assert result.metadata.get("reason") == "cancelled"


# =============================================================================
# StepExecutor Retry Tests
# =============================================================================


class TestStepExecutorRetry:
    """Tests for retry behavior."""
    
    @pytest.mark.asyncio
    async def test_retry_on_transient_error(
        self,
        mock_adapter_registry: MockAdapterRegistry,
        mock_agent_registry: MockAgentRegistry,
        execution_context: ExecutionContext,
    ):
        """Test retry on transient errors."""
        # Create adapter that fails twice then succeeds
        call_count = 0
        
        class FailingAdapter(MockAdapter):
            async def execute(self, task: Task) -> TaskResult:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("Transient network error")
                return await super().execute(task)
        
        adapter = FailingAdapter()
        mock_adapter_registry._adapters["mock"] = adapter
        mock_adapter_registry.set_agent_framework("test-agent", "mock")
        
        executor = StepExecutor(
            adapter_registry=mock_adapter_registry,
            agent_registry=mock_agent_registry,
            config=StepExecutorConfig(emit_events=False),
        )
        
        step = StepDefinition(
            id="step1",
            agent_id="test-agent",
            input_map={},
            retry=RetryPolicy(
                max_attempts=5,
                backoff_ms=10,
                backoff_multiplier=1.0,
            ),
        )
        
        result = await executor.execute_step(step, execution_context)
        
        assert result.status == StepStatus.COMPLETED
        assert result.retry_count == 2  # Two retries needed
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_exhausted(
        self,
        mock_adapter_registry: MockAdapterRegistry,
        mock_agent_registry: MockAgentRegistry,
        execution_context: ExecutionContext,
    ):
        """Test when all retries are exhausted."""
        adapter = MockAdapter(execute_error=ConnectionError("Always fails"))
        mock_adapter_registry._adapters["mock"] = adapter
        mock_adapter_registry.set_agent_framework("test-agent", "mock")
        
        executor = StepExecutor(
            adapter_registry=mock_adapter_registry,
            agent_registry=mock_agent_registry,
            config=StepExecutorConfig(emit_events=False),
        )
        
        step = StepDefinition(
            id="step1",
            agent_id="test-agent",
            input_map={},
            retry=RetryPolicy(
                max_attempts=3,
                backoff_ms=10,
            ),
        )
        
        result = await executor.execute_step(step, execution_context)
        
        assert result.status == StepStatus.FAILED
        assert "Always fails" in result.error.get("message", "")
    
    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(
        self,
        mock_adapter_registry: MockAdapterRegistry,
        mock_agent_registry: MockAgentRegistry,
        execution_context: ExecutionContext,
    ):
        """Test that non-retryable errors don't retry."""
        call_count = 0
        
        class FailingAdapter(MockAdapter):
            async def execute(self, task: Task) -> TaskResult:
                nonlocal call_count
                call_count += 1
                raise ValueError("Validation error - not retryable")
        
        adapter = FailingAdapter()
        mock_adapter_registry._adapters["mock"] = adapter
        mock_adapter_registry.set_agent_framework("test-agent", "mock")
        
        executor = StepExecutor(
            adapter_registry=mock_adapter_registry,
            agent_registry=mock_agent_registry,
            config=StepExecutorConfig(emit_events=False),
        )
        
        step = StepDefinition(
            id="step1",
            agent_id="test-agent",
            input_map={},
            retry=RetryPolicy(
                max_attempts=5,
                backoff_ms=10,
                # Only retry timeout/external errors (not INVALID_INPUT which ValueError maps to)
                retry_on=[ErrorCategory.TIMEOUT, ErrorCategory.EXTERNAL_SERVICE_ERROR],
            ),
        )
        
        result = await executor.execute_step(step, execution_context)
        
        assert result.status == StepStatus.FAILED
        assert call_count == 1  # No retries attempted


# =============================================================================
# StepExecutor Fallback Tests
# =============================================================================


class TestStepExecutorFallback:
    """Tests for fallback behavior."""
    
    @pytest.mark.asyncio
    async def test_fallback_skip(
        self,
        mock_adapter_registry: MockAdapterRegistry,
        mock_agent_registry: MockAgentRegistry,
        execution_context: ExecutionContext,
    ):
        """Test fallback with skip=True."""
        adapter = MockAdapter(execute_error=RuntimeError("Primary failed"))
        mock_adapter_registry._adapters["mock"] = adapter
        mock_adapter_registry.set_agent_framework("test-agent", "mock")
        
        executor = StepExecutor(
            adapter_registry=mock_adapter_registry,
            agent_registry=mock_agent_registry,
            config=StepExecutorConfig(emit_events=False),
        )
        
        step = StepDefinition(
            id="step1",
            agent_id="test-agent",
            input_map={},
            fallback=FallbackPolicy(skip=True),
        )
        
        result = await executor.execute_step(step, execution_context)
        
        assert result.status == StepStatus.SKIPPED
        assert result.used_fallback is True
        assert "Primary failed" in result.metadata.get("original_error", "")
    
    @pytest.mark.asyncio
    async def test_fallback_static_value(
        self,
        mock_adapter_registry: MockAdapterRegistry,
        mock_agent_registry: MockAgentRegistry,
        execution_context: ExecutionContext,
    ):
        """Test fallback with static value."""
        adapter = MockAdapter(execute_error=RuntimeError("Primary failed"))
        mock_adapter_registry._adapters["mock"] = adapter
        mock_adapter_registry.set_agent_framework("test-agent", "mock")
        
        executor = StepExecutor(
            adapter_registry=mock_adapter_registry,
            agent_registry=mock_agent_registry,
            config=StepExecutorConfig(emit_events=False),
        )
        
        fallback_value = {"default": "value", "status": "fallback"}
        step = StepDefinition(
            id="step1",
            agent_id="test-agent",
            input_map={},
            fallback=FallbackPolicy(static_value=fallback_value),
        )
        
        result = await executor.execute_step(step, execution_context)
        
        assert result.status == StepStatus.COMPLETED
        assert result.output == fallback_value
        assert result.used_fallback is True
    
    @pytest.mark.asyncio
    async def test_fallback_agent(
        self,
        mock_adapter_registry: MockAdapterRegistry,
        mock_agent_registry: MockAgentRegistry,
        execution_context: ExecutionContext,
    ):
        """Test fallback with alternative agent."""
        # Primary adapter fails
        primary_adapter = MockAdapter(execute_error=RuntimeError("Primary failed"))
        mock_adapter_registry._adapters["mock"] = primary_adapter
        mock_adapter_registry.set_agent_framework("test-agent", "mock")
        
        # Fallback adapter succeeds
        fallback_result = TaskResult(
            task_id="test",
            agent_id="fallback-agent",
            status=TaskStatus.COMPLETED,
            output={"result": "from fallback"},
        )
        fallback_adapter = MockAdapter(execute_result=fallback_result)
        mock_adapter_registry._adapters["fallback"] = fallback_adapter
        mock_adapter_registry.set_agent_framework("fallback-agent", "fallback")
        
        # Register fallback agent
        fallback_manifest = AgentManifest(
            id="fallback-agent",
            name="Fallback Agent",
            version="1.0.0",
            framework="fallback",
            status=AgentStatus.ACTIVE,
        )
        mock_agent_registry.register(fallback_manifest)
        
        executor = StepExecutor(
            adapter_registry=mock_adapter_registry,
            agent_registry=mock_agent_registry,
            config=StepExecutorConfig(emit_events=False),
        )
        
        step = StepDefinition(
            id="step1",
            agent_id="test-agent",
            input_map={},
            fallback=FallbackPolicy(agent_id="fallback-agent"),
        )
        
        result = await executor.execute_step(step, execution_context)
        
        assert result.status == StepStatus.COMPLETED
        assert result.output == {"result": "from fallback"}
        assert result.used_fallback is True
        assert result.metadata.get("fallback_agent") == "fallback-agent"
    
    @pytest.mark.asyncio
    async def test_fallback_agent_also_fails(
        self,
        mock_adapter_registry: MockAdapterRegistry,
        mock_agent_registry: MockAgentRegistry,
        execution_context: ExecutionContext,
    ):
        """Test when both primary and fallback agents fail."""
        # Primary adapter fails
        primary_adapter = MockAdapter(execute_error=RuntimeError("Primary failed"))
        mock_adapter_registry._adapters["mock"] = primary_adapter
        mock_adapter_registry.set_agent_framework("test-agent", "mock")
        
        # Fallback adapter also fails
        fallback_adapter = MockAdapter(execute_error=RuntimeError("Fallback failed"))
        mock_adapter_registry._adapters["fallback"] = fallback_adapter
        mock_adapter_registry.set_agent_framework("fallback-agent", "fallback")
        
        # Register fallback agent
        fallback_manifest = AgentManifest(
            id="fallback-agent",
            name="Fallback Agent",
            version="1.0.0",
            framework="fallback",
            status=AgentStatus.ACTIVE,
        )
        mock_agent_registry.register(fallback_manifest)
        
        executor = StepExecutor(
            adapter_registry=mock_adapter_registry,
            agent_registry=mock_agent_registry,
            config=StepExecutorConfig(emit_events=False),
        )
        
        step = StepDefinition(
            id="step1",
            agent_id="test-agent",
            input_map={},
            fallback=FallbackPolicy(agent_id="fallback-agent"),
        )
        
        # When both primary and fallback fail, the FallbackError is raised
        with pytest.raises(FallbackError) as exc_info:
            await executor.execute_step(step, execution_context)
        
        assert "step1" in str(exc_info.value)
        assert "Fallback failed" in str(exc_info.value)


# =============================================================================
# StepExecutor Timeout Tests
# =============================================================================


class TestStepExecutorTimeout:
    """Tests for timeout behavior."""
    
    @pytest.mark.asyncio
    async def test_step_timeout(
        self,
        mock_adapter_registry: MockAdapterRegistry,
        mock_agent_registry: MockAgentRegistry,
        execution_context: ExecutionContext,
    ):
        """Test step timeout."""
        # Create adapter that takes too long
        adapter = MockAdapter(execute_delay_ms=1000)  # 1 second
        mock_adapter_registry._adapters["mock"] = adapter
        mock_adapter_registry.set_agent_framework("test-agent", "mock")
        
        executor = StepExecutor(
            adapter_registry=mock_adapter_registry,
            agent_registry=mock_agent_registry,
            config=StepExecutorConfig(
                default_timeout_ms=100,  # 100ms timeout
                emit_events=False,
            ),
        )
        
        step = StepDefinition(
            id="step1",
            agent_id="test-agent",
            input_map={},
        )
        
        result = await executor.execute_step(step, execution_context)
        
        assert result.status == StepStatus.FAILED
        assert result.error_category == ErrorCategory.TIMEOUT
    
    @pytest.mark.asyncio
    async def test_step_custom_timeout(
        self,
        mock_adapter_registry: MockAdapterRegistry,
        mock_agent_registry: MockAgentRegistry,
        execution_context: ExecutionContext,
    ):
        """Test step with custom timeout."""
        # Create adapter that takes a bit
        adapter = MockAdapter(execute_delay_ms=100)  # 100ms
        mock_adapter_registry._adapters["mock"] = adapter
        mock_adapter_registry.set_agent_framework("test-agent", "mock")
        
        executor = StepExecutor(
            adapter_registry=mock_adapter_registry,
            agent_registry=mock_agent_registry,
            config=StepExecutorConfig(
                default_timeout_ms=50,  # Short default timeout
                emit_events=False,
            ),
        )
        
        step = StepDefinition(
            id="step1",
            agent_id="test-agent",
            input_map={},
            timeout_ms=500,  # Long enough custom timeout
        )
        
        result = await executor.execute_step(step, execution_context)
        
        assert result.status == StepStatus.COMPLETED


# =============================================================================
# StepExecutor Input Mapping Tests
# =============================================================================


class TestStepExecutorInputMapping:
    """Tests for input mapping."""
    
    @pytest.mark.asyncio
    async def test_input_mapping_success(
        self,
        step_executor: StepExecutor,
        mock_adapter: MockAdapter,
    ):
        """Test successful input mapping."""
        workflow = WorkflowDefinition(
            id="test-workflow",
            name="Test",
            steps=[
                StepDefinition(
                    id="step1",
                    agent_id="test-agent",
                    input_map={
                        "query": "$.input.topic",
                        "count": "$.input.count",
                    },
                ),
            ],
        )
        
        context = ExecutionContext(
            execution_id="exec-001",
            workflow=workflow,
            input={"topic": "AI safety", "count": 10},
            correlation_id="corr-001",
        )
        
        step = workflow.steps[0]
        result = await step_executor.execute_step(step, context)
        
        assert result.status == StepStatus.COMPLETED
        # Check that the adapter received mapped inputs
        assert len(mock_adapter.execute_calls) == 1
        task = mock_adapter.execute_calls[0]
        assert task.input == {"query": "AI safety", "count": 10}
    
    @pytest.mark.asyncio
    async def test_input_mapping_missing_path(
        self,
        step_executor: StepExecutor,
    ):
        """Test input mapping with missing path."""
        workflow = WorkflowDefinition(
            id="test-workflow",
            name="Test",
            steps=[
                StepDefinition(
                    id="step1",
                    agent_id="test-agent",
                    input_map={"query": "$.input.nonexistent"},
                ),
            ],
        )
        
        context = ExecutionContext(
            execution_id="exec-001",
            workflow=workflow,
            input={"topic": "AI safety"},
            correlation_id="corr-001",
        )
        
        step = workflow.steps[0]
        result = await step_executor.execute_step(step, context)
        
        assert result.status == StepStatus.FAILED
        assert result.error_category == ErrorCategory.INVALID_INPUT


# =============================================================================
# StepExecutor Event Emission Tests
# =============================================================================


class TestStepExecutorEvents:
    """Tests for event emission."""
    
    @pytest.mark.asyncio
    async def test_events_emitted_on_success(
        self,
        mock_adapter_registry: MockAdapterRegistry,
        mock_agent_registry: MockAgentRegistry,
        execution_context: ExecutionContext,
    ):
        """Test events are emitted on successful execution."""
        events: List[WorkflowEvent] = []
        
        async def capture_event(event: WorkflowEvent) -> None:
            events.append(event)
        
        adapter = MockAdapter()
        mock_adapter_registry._adapters["mock"] = adapter
        mock_adapter_registry.set_agent_framework("test-agent", "mock")
        
        executor = StepExecutor(
            adapter_registry=mock_adapter_registry,
            agent_registry=mock_agent_registry,
            config=StepExecutorConfig(emit_events=True),
            event_callback=capture_event,
        )
        
        step = StepDefinition(
            id="step1",
            agent_id="test-agent",
            input_map={},
        )
        
        await executor.execute_step(step, execution_context)
        
        # Should have STEP_STARTED and STEP_COMPLETED
        event_types = [e.type for e in events]
        assert WorkflowEventType.STEP_STARTED in event_types
        assert WorkflowEventType.STEP_COMPLETED in event_types
    
    @pytest.mark.asyncio
    async def test_events_emitted_on_failure(
        self,
        mock_adapter_registry: MockAdapterRegistry,
        mock_agent_registry: MockAgentRegistry,
        execution_context: ExecutionContext,
    ):
        """Test events are emitted on failure."""
        events: List[WorkflowEvent] = []
        
        async def capture_event(event: WorkflowEvent) -> None:
            events.append(event)
        
        adapter = MockAdapter(execute_error=RuntimeError("Test error"))
        mock_adapter_registry._adapters["mock"] = adapter
        mock_adapter_registry.set_agent_framework("test-agent", "mock")
        
        executor = StepExecutor(
            adapter_registry=mock_adapter_registry,
            agent_registry=mock_agent_registry,
            config=StepExecutorConfig(emit_events=True),
            event_callback=capture_event,
        )
        
        step = StepDefinition(
            id="step1",
            agent_id="test-agent",
            input_map={},
        )
        
        await executor.execute_step(step, execution_context)
        
        # Should have STEP_STARTED and STEP_FAILED
        event_types = [e.type for e in events]
        assert WorkflowEventType.STEP_STARTED in event_types
        assert WorkflowEventType.STEP_FAILED in event_types
    
    @pytest.mark.asyncio
    async def test_events_emitted_on_retry(
        self,
        mock_adapter_registry: MockAdapterRegistry,
        mock_agent_registry: MockAgentRegistry,
        execution_context: ExecutionContext,
    ):
        """Test events are emitted on retry."""
        events: List[WorkflowEvent] = []
        call_count = 0
        
        async def capture_event(event: WorkflowEvent) -> None:
            events.append(event)
        
        class FailOnceAdapter(MockAdapter):
            async def execute(self, task: Task) -> TaskResult:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ConnectionError("Transient error")
                return await super().execute(task)
        
        adapter = FailOnceAdapter()
        mock_adapter_registry._adapters["mock"] = adapter
        mock_adapter_registry.set_agent_framework("test-agent", "mock")
        
        executor = StepExecutor(
            adapter_registry=mock_adapter_registry,
            agent_registry=mock_agent_registry,
            config=StepExecutorConfig(emit_events=True),
            event_callback=capture_event,
        )
        
        step = StepDefinition(
            id="step1",
            agent_id="test-agent",
            input_map={},
            retry=RetryPolicy(max_attempts=3, backoff_ms=10),
        )
        
        await executor.execute_step(step, execution_context)
        
        # Should have STEP_STARTED, STEP_RETRYING, and STEP_COMPLETED
        event_types = [e.type for e in events]
        assert WorkflowEventType.STEP_STARTED in event_types
        assert WorkflowEventType.STEP_RETRYING in event_types
        assert WorkflowEventType.STEP_COMPLETED in event_types
    
    @pytest.mark.asyncio
    async def test_events_disabled(
        self,
        mock_adapter_registry: MockAdapterRegistry,
        mock_agent_registry: MockAgentRegistry,
        execution_context: ExecutionContext,
    ):
        """Test events can be disabled."""
        events: List[WorkflowEvent] = []
        
        async def capture_event(event: WorkflowEvent) -> None:
            events.append(event)
        
        adapter = MockAdapter()
        mock_adapter_registry._adapters["mock"] = adapter
        mock_adapter_registry.set_agent_framework("test-agent", "mock")
        
        executor = StepExecutor(
            adapter_registry=mock_adapter_registry,
            agent_registry=mock_agent_registry,
            config=StepExecutorConfig(emit_events=False),  # Events disabled
            event_callback=capture_event,
        )
        
        step = StepDefinition(
            id="step1",
            agent_id="test-agent",
            input_map={},
        )
        
        await executor.execute_step(step, execution_context)
        
        # No events should be captured
        assert len(events) == 0


# =============================================================================
# StepExecutor Error Category Tests
# =============================================================================


class TestStepExecutorErrorCategories:
    """Tests for error categorization."""
    
    @pytest.mark.asyncio
    async def test_timeout_error_category(
        self,
        mock_adapter_registry: MockAdapterRegistry,
        mock_agent_registry: MockAgentRegistry,
        execution_context: ExecutionContext,
    ):
        """Test timeout errors are categorized correctly."""
        adapter = MockAdapter(execute_delay_ms=1000)
        mock_adapter_registry._adapters["mock"] = adapter
        mock_adapter_registry.set_agent_framework("test-agent", "mock")
        
        executor = StepExecutor(
            adapter_registry=mock_adapter_registry,
            agent_registry=mock_agent_registry,
            config=StepExecutorConfig(default_timeout_ms=50, emit_events=False),
        )
        
        step = StepDefinition(id="step1", agent_id="test-agent", input_map={})
        result = await executor.execute_step(step, execution_context)
        
        assert result.error_category == ErrorCategory.TIMEOUT
    
    @pytest.mark.asyncio
    async def test_transient_error_category(
        self,
        mock_adapter_registry: MockAdapterRegistry,
        mock_agent_registry: MockAgentRegistry,
        execution_context: ExecutionContext,
    ):
        """Test transient/external service errors are categorized correctly."""
        adapter = MockAdapter(execute_error=ConnectionError("Network failed"))
        mock_adapter_registry._adapters["mock"] = adapter
        mock_adapter_registry.set_agent_framework("test-agent", "mock")
        
        executor = StepExecutor(
            adapter_registry=mock_adapter_registry,
            agent_registry=mock_agent_registry,
            config=StepExecutorConfig(emit_events=False),
        )
        
        step = StepDefinition(id="step1", agent_id="test-agent", input_map={})
        result = await executor.execute_step(step, execution_context)
        
        # ConnectionError maps to EXTERNAL_SERVICE_ERROR (transient/network issues)
        assert result.error_category == ErrorCategory.EXTERNAL_SERVICE_ERROR
    
    @pytest.mark.asyncio
    async def test_input_error_category(
        self,
        step_executor: StepExecutor,
    ):
        """Test input errors are categorized correctly."""
        workflow = WorkflowDefinition(
            id="test-workflow",
            name="Test",
            steps=[
                StepDefinition(
                    id="step1",
                    agent_id="test-agent",
                    input_map={"query": "$.input.missing"},
                ),
            ],
        )
        
        context = ExecutionContext(
            execution_id="exec-001",
            workflow=workflow,
            input={},
            correlation_id="corr-001",
        )
        
        result = await step_executor.execute_step(workflow.steps[0], context)
        
        assert result.error_category == ErrorCategory.INVALID_INPUT
