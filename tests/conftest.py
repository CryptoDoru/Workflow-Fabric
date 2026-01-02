"""
AI Workflow Fabric - Test Configuration and Fixtures

This module provides shared fixtures and configuration for all tests.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from awf.core.types import (
    AgentManifest,
    AgentStatus,
    Capability,
    CapabilityType,
    Event,
    EventType,
    Policy,
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
    WorkflowStep,
)


# =============================================================================
# Event Loop Configuration
# =============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_schema() -> Schema:
    """Create a sample schema for testing."""
    return Schema(
        type="object",
        properties={
            "query": SchemaProperty(
                name="query",
                type="string",
                description="The search query",
                required=True,
            ),
            "max_results": SchemaProperty(
                name="max_results",
                type="integer",
                description="Maximum number of results",
                required=False,
                default=10,
            ),
        },
        required=["query"],
        description="Input schema for search agent",
    )


@pytest.fixture
def sample_capability() -> Capability:
    """Create a sample capability for testing."""
    return Capability(
        name="web_search",
        type=CapabilityType.TOOL,
        description="Search the web for information",
        permissions=["network:external"],
        metadata={"provider": "tavily"},
    )


@pytest.fixture
def sample_capabilities() -> List[Capability]:
    """Create a list of sample capabilities."""
    return [
        Capability(
            name="web_search",
            type=CapabilityType.TOOL,
            description="Search the web",
            permissions=["network:external"],
        ),
        Capability(
            name="summarize",
            type=CapabilityType.REASONING,
            description="Summarize text content",
        ),
        Capability(
            name="memory",
            type=CapabilityType.MEMORY,
            description="Store and retrieve information",
        ),
    ]


@pytest.fixture
def sample_manifest(sample_capabilities: List[Capability]) -> AgentManifest:
    """Create a sample agent manifest for testing."""
    return AgentManifest(
        id="test-agent-001",
        name="Test Research Agent",
        version="1.0.0",
        framework="langgraph",
        framework_version="0.2.0",
        capabilities=sample_capabilities,
        description="A test agent for research tasks",
        tags=["research", "search", "test"],
        publisher="test-publisher",
        status=AgentStatus.ACTIVE,
        trust_score=0.85,
        metadata={
            "nodes": ["researcher", "writer"],
            "edges": [{"source": "researcher", "target": "writer"}],
        },
    )


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task for testing."""
    return Task(
        agent_id="test-agent-001",
        input={"query": "AI safety research", "max_results": 5},
        timeout_ms=30000,
        context={"user_id": "user-123"},
        trace_id="trace-abc-123",
    )


@pytest.fixture
def sample_task_result(sample_task: Task) -> TaskResult:
    """Create a sample task result for testing."""
    return TaskResult(
        task_id=sample_task.id,
        agent_id=sample_task.agent_id,
        status=TaskStatus.COMPLETED,
        output={"results": ["result1", "result2"], "summary": "Test summary"},
        metrics=TaskMetrics(
            execution_time_ms=1500,
            token_usage={"input_tokens": 100, "output_tokens": 250},
        ),
        trace_id=sample_task.trace_id,
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_failed_result(sample_task: Task) -> TaskResult:
    """Create a sample failed task result for testing."""
    return TaskResult(
        task_id=sample_task.id,
        agent_id=sample_task.agent_id,
        status=TaskStatus.FAILED,
        error=TaskError(
            code="EXECUTION_ERROR",
            message="Agent execution failed",
            retryable=True,
        ),
        metrics=TaskMetrics(execution_time_ms=500),
        trace_id=sample_task.trace_id,
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
    )


@pytest.fixture
def sample_event() -> Event:
    """Create a sample event for testing."""
    return Event(
        type=EventType.TASK_COMPLETED,
        source="test-agent-001",
        correlation_id="task-123",
        trace_id="trace-abc-123",
        data={"output": {"result": "test"}},
    )


@pytest.fixture
def sample_trust_factors() -> TrustFactors:
    """Create sample trust factors for testing."""
    return TrustFactors(
        publisher_trust=0.9,
        audit_status=0.8,
        community_trust=0.7,
        permission_analysis=0.85,
        historical_behavior=0.95,
    )


@pytest.fixture
def sample_trust_score(sample_trust_factors: TrustFactors) -> TrustScore:
    """Create a sample trust score for testing."""
    return TrustScore.compute(sample_trust_factors)


@pytest.fixture
def sample_workflow_step() -> WorkflowStep:
    """Create a sample workflow step for testing."""
    return WorkflowStep(
        id="step-1",
        agent_id="test-agent-001",
        input_map={"query": "$.input.topic"},
        timeout_ms=30000,
    )


@pytest.fixture
def sample_workflow(sample_workflow_step: WorkflowStep) -> Workflow:
    """Create a sample workflow for testing."""
    return Workflow(
        id="test-workflow-001",
        name="Test Research Workflow",
        steps=[
            sample_workflow_step,
            WorkflowStep(
                id="step-2",
                agent_id="writer-agent",
                input_map={"content": "$.steps[0].output.results"},
            ),
        ],
        description="A test workflow for research and writing",
    )


@pytest.fixture
def sample_policy() -> Policy:
    """Create a sample policy for testing."""
    return Policy(
        id="policy-001",
        name="Production Minimum Trust",
        environments=["production"],
        min_trust_score=0.8,
        deny_capabilities=["process:execute"],
        max_execution_time_ms=60000,
        allow_network=True,
        allow_filesystem=False,
        description="Minimum requirements for production agents",
    )


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_langgraph_graph():
    """Create a mock LangGraph compiled graph for testing."""
    mock_graph = MagicMock()
    mock_graph.nodes = {
        "researcher": MagicMock(__name__="researcher"),
        "writer": MagicMock(__name__="writer"),
    }
    mock_graph.edges = [("researcher", "writer")]
    mock_graph.invoke = MagicMock(return_value={"result": "test output"})
    mock_graph.stream = MagicMock(return_value=[
        {"researcher": {"data": "research results"}},
        {"writer": {"data": "written content"}},
    ])
    return mock_graph


@pytest.fixture
def mock_crewai_agent():
    """Create a mock CrewAI agent for testing."""
    mock_agent = MagicMock()
    mock_agent.role = "Researcher"
    mock_agent.goal = "Research AI topics"
    mock_agent.backstory = "An expert AI researcher"
    mock_agent.tools = []
    mock_agent.memory = True
    return mock_agent


@pytest.fixture
def mock_crewai_crew(mock_crewai_agent):
    """Create a mock CrewAI crew for testing."""
    mock_crew = MagicMock()
    mock_crew.agents = [mock_crewai_agent]
    mock_crew.tasks = []
    mock_crew.process = "sequential"
    mock_crew.kickoff = MagicMock(return_value={"result": "crew output"})
    return mock_crew


# =============================================================================
# Async Helpers
# =============================================================================


@pytest.fixture
def async_mock():
    """Helper to create async mock functions."""
    def _async_mock(return_value=None):
        async def mock_func(*args, **kwargs):
            return return_value
        return mock_func
    return _async_mock


# =============================================================================
# Test Data Generators
# =============================================================================


@pytest.fixture
def manifest_factory():
    """Factory for creating test manifests with custom attributes."""
    def _create_manifest(
        agent_id: str = "test-agent",
        name: str = "Test Agent",
        framework: str = "langgraph",
        trust_score: Optional[float] = None,
        **kwargs
    ) -> AgentManifest:
        return AgentManifest(
            id=agent_id,
            name=name,
            version="1.0.0",
            framework=framework,
            trust_score=trust_score,
            status=AgentStatus.ACTIVE,
            **kwargs
        )
    return _create_manifest


@pytest.fixture
def task_factory():
    """Factory for creating test tasks with custom attributes."""
    def _create_task(
        agent_id: str = "test-agent",
        input_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Task:
        return Task(
            agent_id=agent_id,
            input=input_data or {"query": "test"},
            **kwargs
        )
    return _create_task


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Add any cleanup logic here if needed


# =============================================================================
# Markers
# =============================================================================


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "langgraph: marks tests that require LangGraph"
    )
    config.addinivalue_line(
        "markers", "crewai: marks tests that require CrewAI"
    )
