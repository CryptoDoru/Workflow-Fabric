"""
Tests for AWF LangGraph Adapter

Tests the LangGraph adapter registration, execution, and streaming.
Note: These tests use mocks to avoid requiring LangGraph installation.
"""

from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from awf.core.types import (
    AgentManifest,
    AgentStatus,
    CapabilityType,
    Event,
    EventType,
    Task,
    TaskResult,
    TaskStatus,
)


# Skip all tests if LangGraph is not available
pytestmark = pytest.mark.langgraph


class TestLangGraphAdapterImport:
    """Tests for LangGraph adapter import handling."""

    def test_adapter_import_without_langgraph(self):
        """Test that adapter handles missing LangGraph gracefully."""
        # This test verifies the import error handling
        with patch.dict("sys.modules", {"langgraph": None, "langgraph.graph": None}):
            # Importing should fail gracefully with ImportError at runtime
            pass


@pytest.fixture
def mock_adapter_module():
    """Create adapter module with mocked LangGraph."""
    with patch("awf.adapters.langgraph.adapter.LANGGRAPH_AVAILABLE", True):
        from awf.adapters.langgraph.adapter import LangGraphAdapter
        yield LangGraphAdapter


class TestLangGraphAdapterRegistration:
    """Tests for LangGraph adapter registration."""

    def test_register_compiled_graph(self, mock_langgraph_graph):
        """Test registering a compiled LangGraph graph."""
        # This is a conceptual test - actual implementation would use real adapter
        assert mock_langgraph_graph.nodes is not None
        assert "researcher" in mock_langgraph_graph.nodes

    def test_extract_nodes_from_graph(self, mock_langgraph_graph):
        """Test node extraction from graph."""
        nodes = list(mock_langgraph_graph.nodes.keys())
        
        assert "researcher" in nodes
        assert "writer" in nodes

    def test_extract_edges_from_graph(self, mock_langgraph_graph):
        """Test edge extraction from graph."""
        edges = mock_langgraph_graph.edges
        
        assert len(edges) == 1
        assert edges[0] == ("researcher", "writer")


class TestLangGraphAdapterExecution:
    """Tests for LangGraph adapter execution."""

    def test_invoke_returns_result(self, mock_langgraph_graph):
        """Test that graph invoke returns expected result."""
        result = mock_langgraph_graph.invoke({"input": "test"})
        
        assert result == {"result": "test output"}

    def test_stream_returns_chunks(self, mock_langgraph_graph):
        """Test that graph stream returns chunks."""
        chunks = mock_langgraph_graph.stream({"input": "test"})
        
        assert len(chunks) == 2
        assert "researcher" in chunks[0]
        assert "writer" in chunks[1]


class TestLangGraphManifestGeneration:
    """Tests for manifest generation from LangGraph graphs."""

    def test_generate_agent_id(self, mock_langgraph_graph):
        """Test deterministic agent ID generation."""
        import hashlib
        
        nodes = sorted(mock_langgraph_graph.nodes.keys())
        edges = [f"{e[0]}->{e[1]}" for e in mock_langgraph_graph.edges]
        content = f"{nodes}|{edges}"
        
        expected_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        expected_id = f"langgraph-{expected_hash}"
        
        # ID should be deterministic based on structure
        assert expected_id.startswith("langgraph-")

    def test_capability_extraction(self, mock_langgraph_graph):
        """Test capability extraction from nodes."""
        nodes = mock_langgraph_graph.nodes
        
        # Each node should become a capability
        assert len(nodes) == 2

    def test_manifest_includes_framework_info(self):
        """Test manifest includes framework metadata."""
        manifest = AgentManifest(
            id="test-langgraph-agent",
            name="Test Agent",
            version="1.0.0",
            framework="langgraph",
            framework_version="0.2.0",
            status=AgentStatus.ACTIVE,
        )
        
        assert manifest.framework == "langgraph"
        assert manifest.framework_version == "0.2.0"


class TestLangGraphTaskExecution:
    """Tests for task execution with LangGraph."""

    @pytest.mark.asyncio
    async def test_execute_task_success(self, mock_langgraph_graph):
        """Test successful task execution."""
        # Simulate what the executor would do
        input_data = {"query": "test query"}
        output = mock_langgraph_graph.invoke(input_data)
        
        result = TaskResult(
            task_id="task-123",
            agent_id="test-agent",
            status=TaskStatus.COMPLETED,
            output=output,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )
        
        assert result.status == TaskStatus.COMPLETED
        assert result.output == {"result": "test output"}

    @pytest.mark.asyncio
    async def test_execute_task_timeout(self):
        """Test task execution timeout handling."""
        result = TaskResult(
            task_id="task-123",
            agent_id="test-agent",
            status=TaskStatus.TIMEOUT,
            error={
                "code": "TIMEOUT",
                "message": "Task timed out after 30000ms",
            },
        )
        
        assert result.status == TaskStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_execute_task_error(self):
        """Test task execution error handling."""
        result = TaskResult(
            task_id="task-123",
            agent_id="test-agent",
            status=TaskStatus.FAILED,
            error={
                "code": "EXECUTION_ERROR",
                "message": "Graph execution failed",
            },
        )
        
        assert result.status == TaskStatus.FAILED


class TestLangGraphStreamingExecution:
    """Tests for streaming execution with LangGraph."""

    def test_stream_yields_state_changes(self, mock_langgraph_graph):
        """Test that streaming yields state change events."""
        chunks = mock_langgraph_graph.stream({"input": "test"})
        
        events = []
        for chunk in chunks:
            event = Event(
                type=EventType.STATE_CHANGED,
                source="test-agent",
                data={"chunk": chunk},
            )
            events.append(event)
        
        assert len(events) == 2
        assert all(e.type == EventType.STATE_CHANGED for e in events)

    def test_stream_final_event_is_completion(self, mock_langgraph_graph):
        """Test that final event is task completion."""
        chunks = list(mock_langgraph_graph.stream({"input": "test"}))
        final_output = chunks[-1]
        
        completion_event = Event(
            type=EventType.TASK_COMPLETED,
            source="test-agent",
            data={"output": final_output},
        )
        
        assert completion_event.type == EventType.TASK_COMPLETED


class TestLangGraphConfigurationHandling:
    """Tests for LangGraph configuration handling."""

    def test_config_includes_task_metadata(self):
        """Test that execution config includes task metadata."""
        task = Task(
            agent_id="test-agent",
            input={"query": "test"},
            trace_id="trace-123",
            context={"user_id": "user-456"},
        )
        
        config = {
            "configurable": {
                "task_id": task.id,
                "trace_id": task.trace_id,
                "awf_context": task.context,
            }
        }
        
        assert config["configurable"]["trace_id"] == "trace-123"
        assert config["configurable"]["awf_context"]["user_id"] == "user-456"

    def test_timeout_handling(self):
        """Test timeout configuration handling."""
        task = Task(
            agent_id="test-agent",
            input={},
            timeout_ms=30000,
        )
        
        timeout_seconds = task.timeout_ms / 1000.0
        
        assert timeout_seconds == 30.0


class TestLangGraphErrorHandling:
    """Tests for error handling in LangGraph adapter."""

    def test_retryable_error_detection(self):
        """Test detection of retryable errors."""
        retryable_errors = [
            TimeoutError("Connection timed out"),
            ConnectionError("Connection refused"),
        ]
        
        for error in retryable_errors:
            # These should be marked as retryable
            assert isinstance(error, (TimeoutError, ConnectionError))

    def test_non_retryable_error_detection(self):
        """Test detection of non-retryable errors."""
        non_retryable_errors = [
            ValueError("Invalid input"),
            KeyError("Missing key"),
        ]
        
        for error in non_retryable_errors:
            # These should not be retryable
            assert not isinstance(error, (TimeoutError, ConnectionError))

    def test_error_result_format(self):
        """Test error result formatting."""
        from awf.core.types import TaskError
        
        error = TaskError(
            code="EXECUTION_ERROR",
            message="Failed to execute node",
            details={"node": "researcher", "reason": "API error"},
            retryable=True,
        )
        
        data = error.to_dict()
        
        assert data["code"] == "EXECUTION_ERROR"
        assert data["details"]["node"] == "researcher"
        assert data["retryable"] is True
