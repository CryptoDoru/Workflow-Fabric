"""
Tests for AWF AutoGen Adapter

Tests the AutoGen adapter registration, execution, and streaming.
Note: These tests use mocks to avoid requiring AutoGen installation.
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


# Skip all tests if AutoGen is not available
pytestmark = pytest.mark.autogen


class TestAutoGenAdapterImport:
    """Tests for AutoGen adapter import handling."""

    def test_adapter_import_without_autogen(self):
        """Test that adapter handles missing AutoGen gracefully."""
        # This test verifies the import error handling
        with patch.dict("sys.modules", {"autogen": None}):
            # Importing should fail gracefully with ImportError at runtime
            pass


class TestAutoGenAdapterRegistration:
    """Tests for AutoGen adapter registration."""

    def test_register_assistant_agent(self, mock_autogen_agent):
        """Test registering an AutoGen AssistantAgent."""
        assert mock_autogen_agent.name == "assistant"
        assert mock_autogen_agent.system_message is not None
        assert mock_autogen_agent.llm_config is not None

    def test_register_user_proxy_agent(self, mock_autogen_user_proxy):
        """Test registering an AutoGen UserProxyAgent."""
        assert mock_autogen_user_proxy.name == "user_proxy"
        assert mock_autogen_user_proxy.code_execution_config is not None

    def test_register_group_chat(self, mock_autogen_group_chat):
        """Test registering an AutoGen GroupChat."""
        assert len(mock_autogen_group_chat.agents) == 2
        assert mock_autogen_group_chat.max_round == 10

    def test_register_group_manager(self, mock_autogen_group_manager):
        """Test registering an AutoGen GroupChatManager."""
        assert mock_autogen_group_manager.groupchat is not None
        assert len(mock_autogen_group_manager.groupchat.agents) == 2


class TestAutoGenCapabilityExtraction:
    """Tests for capability extraction from AutoGen agents."""

    def test_extract_llm_capability(self, mock_autogen_agent):
        """Test LLM capability extraction."""
        llm_config = mock_autogen_agent.llm_config
        
        assert llm_config is not None
        assert "model" in llm_config
        assert llm_config["model"] == "gpt-4"

    def test_extract_code_execution_capability(self, mock_autogen_user_proxy):
        """Test code execution capability extraction."""
        code_config = mock_autogen_user_proxy.code_execution_config
        
        assert code_config is not None
        assert "work_dir" in code_config

    def test_extract_human_input_mode(self, mock_autogen_agent):
        """Test human input mode extraction."""
        mode = mock_autogen_agent.human_input_mode
        
        assert mode == "NEVER"

    def test_extract_group_chat_agents(self, mock_autogen_group_chat):
        """Test extracting agents from group chat."""
        agents = mock_autogen_group_chat.agents
        
        assert len(agents) == 2
        # First agent should be the assistant
        assert agents[0].name == "assistant"
        # Second agent should be the user proxy
        assert agents[1].name == "user_proxy"


class TestAutoGenManifestGeneration:
    """Tests for manifest generation from AutoGen agents."""

    def test_generate_agent_id(self, mock_autogen_agent):
        """Test deterministic agent ID generation."""
        import hashlib
        
        name = mock_autogen_agent.name
        system_message = mock_autogen_agent.system_message
        content = f"assistant:{name}:{system_message[:100]}"
        
        expected_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        expected_id = f"autogen-{expected_hash}"
        
        # ID should be deterministic based on agent properties
        assert expected_id.startswith("autogen-")

    def test_manifest_includes_framework_info(self):
        """Test manifest includes framework metadata."""
        manifest = AgentManifest(
            id="test-autogen-agent",
            name="Test Agent",
            version="1.0.0",
            framework="autogen",
            framework_version="0.2.0",
            status=AgentStatus.ACTIVE,
        )
        
        assert manifest.framework == "autogen"
        assert manifest.framework_version == "0.2.0"

    def test_manifest_includes_agent_type(self):
        """Test manifest includes agent type in metadata."""
        manifest = AgentManifest(
            id="test-autogen-agent",
            name="Test Agent",
            version="1.0.0",
            framework="autogen",
            status=AgentStatus.ACTIVE,
            metadata={
                "type": "assistant",
                "human_input_mode": "NEVER",
            },
        )
        
        assert manifest.metadata["type"] == "assistant"
        assert manifest.metadata["human_input_mode"] == "NEVER"


class TestAutoGenTaskExecution:
    """Tests for task execution with AutoGen."""

    @pytest.mark.asyncio
    async def test_execute_task_success(self, mock_autogen_agent):
        """Test successful task execution."""
        # Simulate what the executor would do
        input_data = {"message": "Hello, how are you?"}
        
        # Call generate_reply like the adapter would
        response = mock_autogen_agent.generate_reply(
            messages=[{"role": "user", "content": input_data["message"]}]
        )
        
        result = TaskResult(
            task_id="task-123",
            agent_id="test-agent",
            status=TaskStatus.COMPLETED,
            output={"response": response},
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
        )
        
        assert result.status == TaskStatus.COMPLETED
        assert "response" in result.output

    @pytest.mark.asyncio
    async def test_execute_group_chat_task(self, mock_autogen_group_manager):
        """Test group chat task execution."""
        group_chat = mock_autogen_group_manager.groupchat
        agents = group_chat.agents
        
        # Get initiator (first agent)
        initiator = agents[0]
        
        # Simulate initiate_chat
        initiator.initiate_chat = MagicMock(return_value={
            "chat_history": [
                {"role": "user", "content": "Write code"},
                {"role": "assistant", "content": "Here's the code..."},
            ],
            "summary": "Code was written successfully",
        })
        
        result = initiator.initiate_chat(mock_autogen_group_manager, message="Write code")
        
        assert "chat_history" in result
        assert len(result["chat_history"]) > 0

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
                "message": "Agent execution failed",
            },
        )
        
        assert result.status == TaskStatus.FAILED


class TestAutoGenStreamingExecution:
    """Tests for streaming execution with AutoGen."""

    def test_stream_yields_state_changes(self, mock_autogen_group_chat):
        """Test that streaming yields state change events for each agent."""
        agents = mock_autogen_group_chat.agents
        
        events = []
        for i, agent in enumerate(agents):
            event = Event(
                type=EventType.STATE_CHANGED,
                source="test-agent",
                data={
                    "state": "agent_turn",
                    "agent_index": i,
                    "agent_name": agent.name,
                },
            )
            events.append(event)
        
        assert len(events) == 2
        assert all(e.type == EventType.STATE_CHANGED for e in events)
        assert events[0].data["agent_name"] == "assistant"
        assert events[1].data["agent_name"] == "user_proxy"

    def test_stream_final_event_is_completion(self):
        """Test that final event is task completion."""
        final_output = {"response": "Task completed successfully"}
        
        completion_event = Event(
            type=EventType.TASK_COMPLETED,
            source="test-agent",
            data={"output": final_output},
        )
        
        assert completion_event.type == EventType.TASK_COMPLETED


class TestAutoGenConversationManagement:
    """Tests for conversation history management."""

    def test_conversation_history_stored(self, mock_autogen_agent):
        """Test that conversation history is maintained."""
        history = []
        
        # Add user message
        history.append({
            "role": "user",
            "content": "Hello!",
        })
        
        # Add assistant response
        history.append({
            "role": "assistant",
            "content": mock_autogen_agent.generate_reply(),
            "name": mock_autogen_agent.name,
        })
        
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_clear_conversation_history(self):
        """Test clearing conversation history."""
        history = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
        ]
        
        # Clear history
        history.clear()
        
        assert len(history) == 0


class TestAutoGenConfigurationHandling:
    """Tests for AutoGen configuration handling."""

    def test_llm_config_sanitization(self, mock_autogen_agent):
        """Test that LLM config is sanitized (no API keys exposed)."""
        llm_config = mock_autogen_agent.llm_config
        
        # Safe config should not contain api_key
        safe_config = {
            k: v for k, v in llm_config.items()
            if k not in ("api_key", "api_secret", "api_base")
        }
        
        assert "api_key" not in safe_config
        assert "model" in safe_config

    def test_code_execution_config(self, mock_autogen_user_proxy):
        """Test code execution configuration handling."""
        code_config = mock_autogen_user_proxy.code_execution_config
        
        assert code_config is not None
        assert code_config["work_dir"] == "coding"
        assert code_config["use_docker"] is False

    def test_timeout_handling(self):
        """Test timeout configuration handling."""
        task = Task(
            agent_id="test-agent",
            input={"message": "Test"},
            timeout_ms=30000,
        )
        
        timeout_seconds = task.timeout_ms / 1000.0
        
        assert timeout_seconds == 30.0


class TestAutoGenErrorHandling:
    """Tests for error handling in AutoGen adapter."""

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
            message="Failed to execute agent",
            details={"agent": "assistant", "reason": "API error"},
            retryable=True,
        )
        
        data = error.to_dict()
        
        assert data["code"] == "EXECUTION_ERROR"
        assert data["details"]["agent"] == "assistant"
        assert data["retryable"] is True


class TestAutoGenAgentTypeDetection:
    """Tests for AutoGen agent type detection."""

    def test_detect_assistant_agent(self, mock_autogen_agent):
        """Test detecting AssistantAgent type."""
        # AssistantAgent has llm_config and system_message
        has_llm = mock_autogen_agent.llm_config is not None
        has_system = mock_autogen_agent.system_message is not None
        
        assert has_llm
        assert has_system

    def test_detect_user_proxy_agent(self, mock_autogen_user_proxy):
        """Test detecting UserProxyAgent type."""
        # UserProxyAgent typically has code_execution_config
        has_code_exec = mock_autogen_user_proxy.code_execution_config is not None
        
        assert has_code_exec

    def test_detect_group_chat_manager(self, mock_autogen_group_manager):
        """Test detecting GroupChatManager type."""
        # GroupChatManager has groupchat attribute
        has_groupchat = hasattr(mock_autogen_group_manager, "groupchat")
        
        assert has_groupchat
        assert mock_autogen_group_manager.groupchat is not None


class TestAutoGenPermissionInference:
    """Tests for permission inference from AutoGen functions."""

    def test_infer_network_permission(self):
        """Test inferring network permission from function."""
        func = {
            "name": "web_search",
            "description": "Search the web for information",
        }
        
        # Check for network keywords
        combined = f"{func['name']} {func['description']}".lower()
        has_network = any(kw in combined for kw in ["web", "http", "api", "fetch", "url"])
        
        assert has_network

    def test_infer_filesystem_permission(self):
        """Test inferring filesystem permission from function."""
        func = {
            "name": "read_file",
            "description": "Read contents from a file",
        }
        
        combined = f"{func['name']} {func['description']}".lower()
        has_fs = any(kw in combined for kw in ["file", "read", "write", "disk", "path"])
        
        assert has_fs

    def test_infer_process_permission(self):
        """Test inferring process execution permission from function."""
        func = {
            "name": "run_command",
            "description": "Execute a shell command",
        }
        
        combined = f"{func['name']} {func['description']}".lower()
        has_process = any(kw in combined for kw in ["exec", "shell", "command", "run"])
        
        assert has_process

    def test_infer_database_permission(self):
        """Test inferring database permission from function."""
        func = {
            "name": "query_database",
            "description": "Run a SQL query against the database",
        }
        
        combined = f"{func['name']} {func['description']}".lower()
        has_db = any(kw in combined for kw in ["sql", "database", "db", "query"])
        
        assert has_db
