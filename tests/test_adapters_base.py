"""
Tests for AWF Adapters Base Classes

Tests the abstract base adapter and supporting interfaces.
"""

from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from awf.adapters.base import (
    AdapterError,
    AgentNotFoundError,
    AgentRegistry,
    BaseAdapter,
    ExecutionError,
    RegistrationError,
    TrustScorer,
    ValidationError,
)
from awf.core.types import (
    AgentManifest,
    AgentStatus,
    Capability,
    CapabilityType,
    Event,
    Task,
    TaskResult,
    TaskStatus,
    TrustScore,
)


class ConcreteAdapter(BaseAdapter):
    """Concrete implementation of BaseAdapter for testing."""

    framework_name = "test_framework"
    framework_version = "1.0.0"

    def __init__(self):
        super().__init__()
        self._manifests: Dict[str, AgentManifest] = {}

    def register(
        self,
        agent: Any,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentManifest:
        self._validate_agent(agent)
        
        if agent_id is None:
            agent_id = self._generate_agent_id(agent)
        
        manifest = AgentManifest(
            id=agent_id,
            name=metadata.get("name", agent_id) if metadata else agent_id,
            version="1.0.0",
            framework=self.framework_name,
            capabilities=self.extract_capabilities(agent),
            status=AgentStatus.ACTIVE,
        )
        
        self._manifests[agent_id] = manifest
        self._registered_agents[agent_id] = agent
        return manifest

    def unregister(self, agent_id: str) -> bool:
        if agent_id in self._manifests:
            del self._manifests[agent_id]
            del self._registered_agents[agent_id]
            return True
        return False

    def get_manifest(self, agent_id: str) -> Optional[AgentManifest]:
        return self._manifests.get(agent_id)

    def list_agents(self) -> List[AgentManifest]:
        return list(self._manifests.values())

    def extract_capabilities(self, agent: Any) -> List[Capability]:
        return [
            Capability(name="test_capability", type=CapabilityType.REASONING)
        ]

    def infer_input_schema(self, agent: Any) -> Optional[Dict[str, Any]]:
        return {"type": "object", "properties": {"input": {"type": "string"}}}

    def infer_output_schema(self, agent: Any) -> Optional[Dict[str, Any]]:
        return {"type": "object", "properties": {"output": {"type": "string"}}}

    async def execute(self, task: Task) -> TaskResult:
        agent = self._registered_agents.get(task.agent_id)
        if agent is None:
            raise AgentNotFoundError(task.agent_id)
        
        return TaskResult(
            task_id=task.id,
            agent_id=task.agent_id,
            status=TaskStatus.COMPLETED,
            output={"result": "test output"},
        )

    async def execute_streaming(self, task: Task) -> AsyncIterator[Event]:
        agent = self._registered_agents.get(task.agent_id)
        if agent is None:
            raise AgentNotFoundError(task.agent_id)
        
        # Yield a single event for testing
        yield Event(
            type="task.completed",
            source=task.agent_id,
            data={"result": "test output"},
        )

    async def cancel(self, task_id: str) -> bool:
        return True


class TestBaseAdapterRegistration:
    """Tests for BaseAdapter registration methods."""

    def test_register_agent(self):
        adapter = ConcreteAdapter()
        agent = MagicMock()
        
        manifest = adapter.register(agent, agent_id="test-agent")
        
        assert manifest.id == "test-agent"
        assert manifest.framework == "test_framework"
        assert manifest.status == AgentStatus.ACTIVE

    def test_register_agent_auto_id(self):
        adapter = ConcreteAdapter()
        agent = MagicMock()
        
        manifest = adapter.register(agent)
        
        assert manifest.id is not None
        assert manifest.id.startswith("test_framework-")

    def test_register_agent_with_metadata(self):
        adapter = ConcreteAdapter()
        agent = MagicMock()
        
        manifest = adapter.register(
            agent,
            agent_id="test-agent",
            metadata={"name": "Custom Name"},
        )
        
        assert manifest.name == "Custom Name"

    def test_unregister_existing_agent(self):
        adapter = ConcreteAdapter()
        agent = MagicMock()
        adapter.register(agent, agent_id="test-agent")
        
        result = adapter.unregister("test-agent")
        
        assert result is True
        assert adapter.get_manifest("test-agent") is None

    def test_unregister_nonexistent_agent(self):
        adapter = ConcreteAdapter()
        
        result = adapter.unregister("nonexistent")
        
        assert result is False

    def test_get_manifest(self):
        adapter = ConcreteAdapter()
        agent = MagicMock()
        adapter.register(agent, agent_id="test-agent")
        
        manifest = adapter.get_manifest("test-agent")
        
        assert manifest is not None
        assert manifest.id == "test-agent"

    def test_get_manifest_nonexistent(self):
        adapter = ConcreteAdapter()
        
        manifest = adapter.get_manifest("nonexistent")
        
        assert manifest is None

    def test_list_agents(self):
        adapter = ConcreteAdapter()
        adapter.register(MagicMock(), agent_id="agent-1")
        adapter.register(MagicMock(), agent_id="agent-2")
        
        agents = adapter.list_agents()
        
        assert len(agents) == 2
        assert any(a.id == "agent-1" for a in agents)
        assert any(a.id == "agent-2" for a in agents)


class TestBaseAdapterValidation:
    """Tests for BaseAdapter validation methods."""

    def test_validate_agent_none(self):
        adapter = ConcreteAdapter()
        
        with pytest.raises(ValueError, match="cannot be None"):
            adapter._validate_agent(None)

    def test_validate_agent_valid(self):
        adapter = ConcreteAdapter()
        agent = MagicMock()
        
        # Should not raise
        adapter._validate_agent(agent)


class TestBaseAdapterExecution:
    """Tests for BaseAdapter execution methods."""

    @pytest.mark.asyncio
    async def test_execute_registered_agent(self):
        adapter = ConcreteAdapter()
        agent = MagicMock()
        adapter.register(agent, agent_id="test-agent")
        
        task = Task(agent_id="test-agent", input={"query": "test"})
        result = await adapter.execute(task)
        
        assert result.status == TaskStatus.COMPLETED
        assert result.output == {"result": "test output"}

    @pytest.mark.asyncio
    async def test_execute_unregistered_agent(self):
        adapter = ConcreteAdapter()
        
        task = Task(agent_id="nonexistent", input={})
        
        with pytest.raises(AgentNotFoundError):
            await adapter.execute(task)

    @pytest.mark.asyncio
    async def test_execute_streaming(self):
        adapter = ConcreteAdapter()
        agent = MagicMock()
        adapter.register(agent, agent_id="test-agent")
        
        task = Task(agent_id="test-agent", input={})
        events = []
        
        async for event in adapter.execute_streaming(task):
            events.append(event)
        
        assert len(events) == 1
        assert events[0].source == "test-agent"

    @pytest.mark.asyncio
    async def test_cancel_task(self):
        adapter = ConcreteAdapter()
        
        result = await adapter.cancel("task-123")
        
        assert result is True


class TestBaseAdapterStatus:
    """Tests for BaseAdapter status methods."""

    def test_get_status(self):
        adapter = ConcreteAdapter()
        agent = MagicMock()
        adapter.register(agent, agent_id="test-agent")
        
        status = adapter.get_status("test-agent")
        
        assert status is not None
        assert status["status"] == "active"

    def test_get_status_nonexistent(self):
        adapter = ConcreteAdapter()
        
        status = adapter.get_status("nonexistent")
        
        assert status is None

    def test_health_check(self):
        adapter = ConcreteAdapter()
        adapter.register(MagicMock(), agent_id="agent-1")
        
        health = adapter.health_check()
        
        assert health["healthy"] is True
        assert health["framework"] == "test_framework"
        assert health["registered_agents"] == 1


class TestAdapterExceptions:
    """Tests for adapter exception classes."""

    def test_adapter_error(self):
        error = AdapterError("Test error")
        assert str(error) == "Test error"

    def test_agent_not_found_error(self):
        error = AgentNotFoundError("missing-agent")
        assert error.agent_id == "missing-agent"
        assert "missing-agent" in str(error)

    def test_registration_error(self):
        error = RegistrationError("Registration failed")
        assert str(error) == "Registration failed"

    def test_execution_error(self):
        cause = ValueError("Original error")
        error = ExecutionError("Execution failed", cause=cause)
        assert error.cause == cause
        assert str(error) == "Execution failed"

    def test_validation_error(self):
        error = ValidationError("Invalid input")
        assert str(error) == "Invalid input"


class TestAgentRegistryInterface:
    """Tests for AgentRegistry abstract interface."""

    def test_interface_methods(self):
        # Verify interface has required methods
        assert hasattr(AgentRegistry, "register")
        assert hasattr(AgentRegistry, "get")
        assert hasattr(AgentRegistry, "search")
        assert hasattr(AgentRegistry, "delete")
        assert hasattr(AgentRegistry, "list_all")


class TestTrustScorerInterface:
    """Tests for TrustScorer abstract interface."""

    def test_interface_methods(self):
        # Verify interface has required methods
        assert hasattr(TrustScorer, "compute_score")
        assert hasattr(TrustScorer, "update_score")
