"""
AI Workflow Fabric - Watcher Agent Tests

Comprehensive tests for the Watcher Agent autonomous observability system.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from awf.agents.watcher import (
    WatcherAgent,
    WatcherConfig,
    GrafanaAlert,
    Investigation,
    RemediationAction,
    RemediationResult,
    RemediationScript,
    RemediationRisk,
    ApprovalRequest,
    GrafanaMCPClient,
    get_watcher_manifest,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def watcher_config():
    """Create a test watcher configuration."""
    return WatcherConfig(
        mcp_endpoint="http://localhost:8686",
        auto_remediation_enabled=True,
        dry_run_mode=True,  # Safe for testing
        approval_timeout_seconds=3600,
    )


@pytest.fixture
def watcher(watcher_config):
    """Create a Watcher Agent instance."""
    return WatcherAgent(watcher_config)


@pytest.fixture
def sample_alert():
    """Create a sample Grafana alert."""
    return GrafanaAlert(
        id="alert-001",
        title="High Error Rate Detected",
        message="Agent error rate exceeded 5% threshold",
        state="alerting",
        labels={
            "agent_id": "test-agent",
            "workflow_id": "test-workflow",
            "execution_id": "exec-123",
            "severity": "warning",
        },
        annotations={
            "summary": "Error rate is above threshold",
            "description": "The agent test-agent has an error rate of 7.5%",
        },
        starts_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def timeout_alert():
    """Create a timeout-related alert."""
    return GrafanaAlert(
        id="alert-002",
        title="Step Timeout",
        message="Step execution timed out",
        state="alerting",
        labels={
            "step_id": "step-1",
            "workflow_id": "test-workflow",
            "execution_id": "exec-456",
        },
        annotations={
            "summary": "Step exceeded timeout threshold",
        },
        starts_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def resolved_alert():
    """Create a resolved alert."""
    return GrafanaAlert(
        id="alert-003",
        title="High Error Rate Detected",
        message="Alert resolved",
        state="ok",
        labels={"agent_id": "test-agent"},
        annotations={},
        starts_at=datetime.now(timezone.utc) - timedelta(hours=1),
        ends_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_webhook_payload():
    """Sample Grafana webhook payload."""
    return {
        "alertId": "alert-123",
        "title": "High Error Rate",
        "message": "Error rate exceeded threshold",
        "state": "alerting",
        "labels": {
            "agent_id": "my-agent",
            "workflow_id": "my-workflow",
            "severity": "warning",
        },
        "annotations": {
            "summary": "Error rate is above threshold",
        },
        "startsAt": "2024-01-01T00:00:00Z",
    }


# =============================================================================
# Manifest Tests
# =============================================================================


class TestWatcherManifest:
    """Tests for the Watcher Agent manifest."""

    def test_manifest_has_required_fields(self):
        """Test that manifest has all required ASP fields."""
        manifest = get_watcher_manifest()
        
        assert manifest.id == "awf-watcher"
        assert manifest.name == "AWF Watcher Agent"
        assert manifest.version == "1.0.0"
        assert manifest.framework == "native"
    
    def test_manifest_has_capabilities(self):
        """Test that manifest defines expected capabilities."""
        manifest = get_watcher_manifest()
        
        capability_names = [cap.name for cap in manifest.capabilities]
        assert "grafana_query" in capability_names
        assert "alert_investigation" in capability_names
        assert "remediation_execution" in capability_names
        assert "hitl_approval" in capability_names
    
    def test_manifest_has_tags(self):
        """Test that manifest has appropriate tags."""
        manifest = get_watcher_manifest()
        
        assert "observability" in manifest.tags
        assert "monitoring" in manifest.tags
        assert "remediation" in manifest.tags


# =============================================================================
# GrafanaAlert Tests
# =============================================================================


class TestGrafanaAlert:
    """Tests for GrafanaAlert data class."""

    def test_from_webhook_basic(self, sample_webhook_payload):
        """Test creating alert from webhook payload."""
        alert = GrafanaAlert.from_webhook(sample_webhook_payload)
        
        assert alert.id == "alert-123"
        assert alert.title == "High Error Rate"
        assert alert.state == "alerting"
        assert alert.labels["agent_id"] == "my-agent"
    
    def test_from_webhook_minimal(self):
        """Test creating alert from minimal payload."""
        payload = {
            "title": "Test Alert",
            "state": "alerting",
        }
        alert = GrafanaAlert.from_webhook(payload)
        
        assert alert.title == "Test Alert"
        assert alert.state == "alerting"
        assert alert.id is not None  # Should generate UUID
        assert alert.labels == {}
    
    def test_from_webhook_with_datetime(self):
        """Test creating alert with datetime parsing."""
        payload = {
            "title": "Test",
            "state": "alerting",
            "startsAt": "2024-01-15T12:00:00+00:00",
            "endsAt": "2024-01-15T13:00:00+00:00",
        }
        alert = GrafanaAlert.from_webhook(payload)
        
        assert alert.starts_at.year == 2024
        assert alert.starts_at.month == 1
        assert alert.ends_at is not None


# =============================================================================
# Investigation Tests
# =============================================================================


class TestInvestigation:
    """Tests for Investigation data class."""

    def test_investigation_to_dict(self, sample_alert):
        """Test investigation serialization."""
        investigation = Investigation(
            alert=sample_alert,
            started_at=datetime.now(timezone.utc),
            root_cause="High error rate due to API failures",
            confidence=0.85,
            affected_components=["agent:test-agent", "workflow:test-workflow"],
        )
        investigation.completed_at = datetime.now(timezone.utc)
        
        result = investigation.to_dict()
        
        assert result["alert_id"] == "alert-001"
        assert result["root_cause"] == "High error rate due to API failures"
        assert result["confidence"] == 0.85
        assert len(result["affected_components"]) == 2
    
    def test_investigation_default_values(self, sample_alert):
        """Test investigation with default values."""
        investigation = Investigation(
            alert=sample_alert,
            started_at=datetime.now(timezone.utc),
        )
        
        assert investigation.root_cause is None
        assert investigation.confidence == 0.0
        assert investigation.affected_components == []
        assert investigation.logs == []
        assert investigation.traces == []


# =============================================================================
# RemediationAction Tests
# =============================================================================


class TestRemediationAction:
    """Tests for RemediationAction data class."""

    def test_action_requires_approval_high_risk(self, watcher_config):
        """Test that high-risk actions require approval."""
        action = RemediationAction(
            id="action-1",
            script_id="disable_agent",
            name="Disable Agent",
            description="Disable problematic agent",
            risk_level=RemediationRisk.HIGH,
        )
        
        assert action.requires_approval(watcher_config) is True
    
    def test_action_requires_approval_hitl_list(self, watcher_config):
        """Test that actions in HITL list require approval."""
        action = RemediationAction(
            id="action-2",
            script_id="agent.disable",  # In default HITL list
            name="Disable Agent",
            description="Disable agent",
            risk_level=RemediationRisk.LOW,
        )
        
        assert action.requires_approval(watcher_config) is True
    
    def test_action_no_approval_low_risk(self, watcher_config):
        """Test that low-risk actions don't require approval."""
        action = RemediationAction(
            id="action-3",
            script_id="notify_oncall",
            name="Notify On-Call",
            description="Send notification",
            risk_level=RemediationRisk.LOW,
        )
        
        assert action.requires_approval(watcher_config) is False


# =============================================================================
# RemediationScript Tests
# =============================================================================


class TestRemediationScript:
    """Tests for RemediationScript execution."""

    @pytest.mark.asyncio
    async def test_execute_dry_run(self):
        """Test dry run execution."""
        script = RemediationScript(
            id="test_script",
            name="Test Script",
            description="A test script",
            risk_level=RemediationRisk.LOW,
            python_callable=AsyncMock(return_value="executed"),
        )
        
        result = await script.execute(
            parameters={"param1": "value1"},
            timeout_ms=5000,
            dry_run=True,
        )
        
        assert result.success is True
        assert result.dry_run is True
        assert "[DRY RUN]" in result.output
        assert result.execution_time_ms == 0
    
    @pytest.mark.asyncio
    async def test_execute_python_callable_async(self):
        """Test executing async Python callable."""
        async def my_handler(param1: str) -> str:
            return f"Executed with {param1}"
        
        script = RemediationScript(
            id="test_script",
            name="Test Script",
            description="A test script",
            risk_level=RemediationRisk.LOW,
            python_callable=my_handler,
        )
        
        result = await script.execute(
            parameters={"param1": "test"},
            timeout_ms=5000,
            dry_run=False,
        )
        
        assert result.success is True
        assert "Executed with test" in result.output
    
    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test execution timeout handling."""
        async def slow_handler() -> str:
            await asyncio.sleep(10)
            return "done"
        
        script = RemediationScript(
            id="slow_script",
            name="Slow Script",
            description="A slow script",
            risk_level=RemediationRisk.LOW,
            python_callable=slow_handler,
        )
        
        result = await script.execute(
            parameters={},
            timeout_ms=100,  # Very short timeout
            dry_run=False,
        )
        
        assert result.success is False
        assert "timed out" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_execute_no_method(self):
        """Test execution with no method defined."""
        script = RemediationScript(
            id="empty_script",
            name="Empty Script",
            description="No execution method",
            risk_level=RemediationRisk.LOW,
        )
        
        result = await script.execute(
            parameters={},
            timeout_ms=5000,
            dry_run=False,
        )
        
        assert result.success is False
        assert "No execution method" in result.error


# =============================================================================
# ApprovalRequest Tests
# =============================================================================


class TestApprovalRequest:
    """Tests for ApprovalRequest data class."""

    def test_is_expired_not_expired(self, sample_alert):
        """Test that non-expired request returns False."""
        investigation = Investigation(
            alert=sample_alert,
            started_at=datetime.now(timezone.utc),
        )
        action = RemediationAction(
            id="action-1",
            script_id="test",
            name="Test",
            description="Test action",
            risk_level=RemediationRisk.HIGH,
        )
        
        request = ApprovalRequest(
            id="approval-1",
            action=action,
            investigation=investigation,
            requested_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        
        assert request.is_expired() is False
    
    def test_is_expired_expired(self, sample_alert):
        """Test that expired request returns True."""
        investigation = Investigation(
            alert=sample_alert,
            started_at=datetime.now(timezone.utc),
        )
        action = RemediationAction(
            id="action-1",
            script_id="test",
            name="Test",
            description="Test action",
            risk_level=RemediationRisk.HIGH,
        )
        
        request = ApprovalRequest(
            id="approval-1",
            action=action,
            investigation=investigation,
            requested_at=datetime.now(timezone.utc) - timedelta(hours=2),
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        
        assert request.is_expired() is True
    
    def test_to_dict(self, sample_alert):
        """Test serialization to dictionary."""
        investigation = Investigation(
            alert=sample_alert,
            started_at=datetime.now(timezone.utc),
            root_cause="Test cause",
        )
        action = RemediationAction(
            id="action-1",
            script_id="test",
            name="Test",
            description="Test action",
            risk_level=RemediationRisk.HIGH,
            parameters={"key": "value"},
        )
        
        request = ApprovalRequest(
            id="approval-1",
            action=action,
            investigation=investigation,
            requested_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        
        result = request.to_dict()
        
        assert result["id"] == "approval-1"
        assert result["status"] == "pending"
        assert result["action"]["script_id"] == "test"
        assert result["action"]["risk_level"] == "high"


# =============================================================================
# GrafanaMCPClient Tests
# =============================================================================


class TestGrafanaMCPClient:
    """Tests for MCP client."""

    def test_init_default_endpoint(self):
        """Test default endpoint configuration."""
        client = GrafanaMCPClient()
        assert client.endpoint == "http://localhost:8686"
    
    def test_init_custom_endpoint(self):
        """Test custom endpoint configuration."""
        client = GrafanaMCPClient("http://grafana:9999")
        assert client.endpoint == "http://grafana:9999"
    
    def test_mock_response(self):
        """Test mock response when MCP is unavailable."""
        client = GrafanaMCPClient()
        result = client._mock_response("test_tool", {"arg": "value"})
        
        assert result["status"] == "mocked"
        assert result["tool"] == "test_tool"
        assert result["arguments"]["arg"] == "value"
    
    @pytest.mark.asyncio
    async def test_query_prometheus_mocked(self):
        """Test Prometheus query with mocked MCP."""
        client = GrafanaMCPClient()
        
        # Without aiohttp, should return mock response
        result = await client.query_prometheus("up{job='awf'}")
        
        assert result["status"] == "mocked"
        assert result["tool"] == "query_prometheus"
    
    @pytest.mark.asyncio
    async def test_query_loki_mocked(self):
        """Test Loki query with mocked MCP."""
        client = GrafanaMCPClient()
        
        result = await client.query_loki('{service_namespace="awf"}')
        
        assert result["status"] == "mocked"
        assert result["tool"] == "query_loki"
    
    @pytest.mark.asyncio
    async def test_query_tempo_mocked(self):
        """Test Tempo query with mocked MCP."""
        client = GrafanaMCPClient()
        
        result = await client.query_tempo("abc123")
        
        assert result["status"] == "mocked"
        assert result["tool"] == "query_tempo"


# =============================================================================
# WatcherAgent Tests
# =============================================================================


class TestWatcherAgent:
    """Tests for the Watcher Agent."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        watcher = WatcherAgent()
        
        assert watcher.config is not None
        assert watcher._started is False
        assert len(watcher._scripts) > 0  # Built-in scripts
    
    def test_init_custom_config(self, watcher_config):
        """Test initialization with custom config."""
        watcher = WatcherAgent(watcher_config)
        
        assert watcher.config.mcp_endpoint == "http://localhost:8686"
        assert watcher.config.dry_run_mode is True
    
    def test_builtin_scripts_registered(self, watcher):
        """Test that built-in scripts are registered."""
        assert "restart_workflow" in watcher._scripts
        assert "retry_step" in watcher._scripts
        assert "increase_timeout" in watcher._scripts
        assert "notify_oncall" in watcher._scripts
        assert "disable_agent" in watcher._scripts
    
    def test_register_custom_script(self, watcher):
        """Test registering a custom script."""
        custom_script = RemediationScript(
            id="custom_action",
            name="Custom Action",
            description="A custom remediation",
            risk_level=RemediationRisk.MEDIUM,
            python_callable=AsyncMock(return_value="done"),
        )
        
        watcher.register_script(custom_script)
        
        assert "custom_action" in watcher._scripts
        assert watcher._scripts["custom_action"].name == "Custom Action"
    
    @pytest.mark.asyncio
    async def test_start_stop(self, watcher):
        """Test starting and stopping the agent."""
        assert watcher._started is False
        
        await watcher.start()
        assert watcher._started is True
        
        await watcher.stop()
        assert watcher._started is False
    
    @pytest.mark.asyncio
    async def test_handle_resolved_alert(self, watcher, resolved_alert):
        """Test that resolved alerts are ignored."""
        result = await watcher.handle_alert(resolved_alert)
        
        assert result["status"] == "ignored"
        assert result["reason"] == "alert_resolved"
    
    @pytest.mark.asyncio
    async def test_handle_alert_timeout(self, watcher, timeout_alert):
        """Test handling a timeout alert."""
        # Mock MCP client to return mock data
        watcher.mcp = MagicMock()
        watcher.mcp.query_prometheus = AsyncMock(return_value={"status": "mocked"})
        watcher.mcp.query_loki = AsyncMock(return_value=[])
        watcher.mcp.query_tempo = AsyncMock(return_value={})
        watcher.mcp.create_annotation = AsyncMock(return_value={})
        
        result = await watcher.handle_alert(timeout_alert)
        
        # Should recommend increasing timeout
        assert result["status"] in ["remediated", "pending_approval"]
    
    @pytest.mark.asyncio
    async def test_handle_alert_error_rate(self, watcher, sample_alert):
        """Test handling an error rate alert."""
        watcher.mcp = MagicMock()
        watcher.mcp.query_prometheus = AsyncMock(return_value={"status": "mocked"})
        watcher.mcp.query_loki = AsyncMock(return_value=[])
        watcher.mcp.query_tempo = AsyncMock(return_value={})
        watcher.mcp.create_annotation = AsyncMock(return_value={})
        
        result = await watcher.handle_alert(sample_alert)
        
        # Error rate alerts should be handled
        assert result["status"] in ["remediated", "pending_approval", "no_action"]
    
    @pytest.mark.asyncio
    async def test_analyze_root_cause_timeout(self, watcher, timeout_alert):
        """Test root cause analysis for timeout."""
        investigation = Investigation(
            alert=timeout_alert,
            started_at=datetime.now(timezone.utc),
        )
        
        root_cause = watcher._analyze_root_cause(investigation)
        
        assert "timeout" in root_cause.lower()
    
    @pytest.mark.asyncio
    async def test_analyze_root_cause_error_rate(self, watcher, sample_alert):
        """Test root cause analysis for error rate."""
        investigation = Investigation(
            alert=sample_alert,
            started_at=datetime.now(timezone.utc),
        )
        
        root_cause = watcher._analyze_root_cause(investigation)
        
        assert "error" in root_cause.lower() or "Alert triggered" in root_cause
    
    def test_calculate_confidence(self, watcher, sample_alert):
        """Test confidence calculation."""
        investigation = Investigation(
            alert=sample_alert,
            started_at=datetime.now(timezone.utc),
            metrics={"some_metric": 1.0},
            logs=[{"message": "error"}],
            traces=[{"trace_id": "123"}],
        )
        
        confidence = watcher._calculate_confidence(investigation)
        
        # Base 0.5 + 0.2 (metrics) + 0.15 (logs) + 0.15 (traces) = 1.0
        assert confidence == 1.0
    
    def test_calculate_confidence_no_data(self, watcher, sample_alert):
        """Test confidence with no data."""
        investigation = Investigation(
            alert=sample_alert,
            started_at=datetime.now(timezone.utc),
        )
        
        confidence = watcher._calculate_confidence(investigation)
        
        assert confidence == 0.5  # Base confidence only
    
    def test_identify_affected_components(self, watcher, sample_alert):
        """Test affected component identification."""
        investigation = Investigation(
            alert=sample_alert,
            started_at=datetime.now(timezone.utc),
        )
        
        components = watcher._identify_affected_components(investigation)
        
        assert "agent:test-agent" in components
        assert "workflow:test-workflow" in components
    
    @pytest.mark.asyncio
    async def test_determine_action_timeout(self, watcher, timeout_alert):
        """Test action determination for timeout."""
        investigation = Investigation(
            alert=timeout_alert,
            started_at=datetime.now(timezone.utc),
            root_cause="Step execution exceeded configured timeout threshold",
            confidence=0.8,
        )
        
        action = await watcher._determine_action(investigation)
        
        assert action is not None
        assert action.script_id == "increase_timeout"
    
    @pytest.mark.asyncio
    async def test_determine_action_default_notify(self, watcher, sample_alert):
        """Test default action is notify."""
        investigation = Investigation(
            alert=sample_alert,
            started_at=datetime.now(timezone.utc),
            root_cause="Unknown issue",
            confidence=0.3,  # Low confidence
        )
        
        action = await watcher._determine_action(investigation)
        
        assert action is not None
        assert action.script_id == "notify_oncall"
    
    @pytest.mark.asyncio
    async def test_execute_remediation_dry_run(self, watcher):
        """Test remediation execution in dry run mode."""
        action = RemediationAction(
            id="action-1",
            script_id="notify_oncall",
            name="Notify",
            description="Test notification",
            risk_level=RemediationRisk.LOW,
            parameters={"message": "Test", "severity": "info"},
        )
        
        # Mock annotation creation
        watcher.mcp.create_annotation = AsyncMock(return_value={})
        
        result = await watcher._execute_remediation(action)
        
        assert result.success is True
        assert result.dry_run is True
    
    @pytest.mark.asyncio
    async def test_execute_remediation_unknown_script(self, watcher):
        """Test execution with unknown script."""
        action = RemediationAction(
            id="action-1",
            script_id="unknown_script",
            name="Unknown",
            description="Unknown action",
            risk_level=RemediationRisk.LOW,
        )
        
        result = await watcher._execute_remediation(action)
        
        assert result.success is False
        assert "Unknown remediation script" in result.error


# =============================================================================
# HITL Approval Tests
# =============================================================================


class TestHITLApproval:
    """Tests for Human-in-the-Loop approval workflow."""

    @pytest.mark.asyncio
    async def test_request_hitl_approval(self, watcher, sample_alert):
        """Test creating an approval request."""
        investigation = Investigation(
            alert=sample_alert,
            started_at=datetime.now(timezone.utc),
            root_cause="High error rate",
        )
        action = RemediationAction(
            id="action-1",
            script_id="disable_agent",
            name="Disable Agent",
            description="Disable the problematic agent",
            risk_level=RemediationRisk.HIGH,
            parameters={"agent_id": "test-agent", "reason": "High error rate"},
            investigation=investigation,
        )
        
        request = await watcher._request_hitl_approval(action)
        
        assert request.id is not None
        assert request.status == "pending"
        assert request.action == action
        assert request.id in watcher._pending_approvals
    
    @pytest.mark.asyncio
    async def test_approve_action_success(self, watcher, sample_alert):
        """Test approving an action."""
        # Create pending approval
        investigation = Investigation(
            alert=sample_alert,
            started_at=datetime.now(timezone.utc),
        )
        action = RemediationAction(
            id="action-1",
            script_id="notify_oncall",
            name="Notify",
            description="Send notification",
            risk_level=RemediationRisk.HIGH,
            parameters={"message": "Test", "severity": "info"},
            investigation=investigation,
        )
        
        request = await watcher._request_hitl_approval(action)
        
        # Mock annotation creation
        watcher.mcp.create_annotation = AsyncMock(return_value={})
        
        # Approve it
        result = await watcher.approve_action(request.id, "admin-user")
        
        assert result["status"] == "executed"
        assert result["approved_by"] == "admin-user"
        assert watcher._pending_approvals[request.id].status == "approved"
    
    @pytest.mark.asyncio
    async def test_approve_action_not_found(self, watcher):
        """Test approving non-existent request."""
        result = await watcher.approve_action("nonexistent-id", "admin")
        
        assert "error" in result
        assert result["error"] == "Approval request not found"
    
    @pytest.mark.asyncio
    async def test_approve_action_expired(self, watcher, sample_alert):
        """Test approving expired request."""
        investigation = Investigation(
            alert=sample_alert,
            started_at=datetime.now(timezone.utc),
        )
        action = RemediationAction(
            id="action-1",
            script_id="notify_oncall",
            name="Notify",
            description="Test",
            risk_level=RemediationRisk.HIGH,
            parameters={"message": "Test", "severity": "info"},
            investigation=investigation,
        )
        
        # Create request but manually expire it
        request = await watcher._request_hitl_approval(action)
        request.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        
        result = await watcher.approve_action(request.id, "admin")
        
        assert "error" in result
        assert "expired" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_reject_action_success(self, watcher, sample_alert):
        """Test rejecting an action."""
        investigation = Investigation(
            alert=sample_alert,
            started_at=datetime.now(timezone.utc),
        )
        action = RemediationAction(
            id="action-1",
            script_id="disable_agent",
            name="Disable",
            description="Test",
            risk_level=RemediationRisk.HIGH,
            parameters={"agent_id": "test", "reason": "test"},
            investigation=investigation,
        )
        
        request = await watcher._request_hitl_approval(action)
        
        result = await watcher.reject_action(
            request.id,
            "admin-user",
            "Not a real issue",
        )
        
        assert result["status"] == "rejected"
        assert result["rejected_by"] == "admin-user"
        assert result["reason"] == "Not a real issue"
        assert watcher._pending_approvals[request.id].status == "rejected"
    
    @pytest.mark.asyncio
    async def test_reject_action_not_found(self, watcher):
        """Test rejecting non-existent request."""
        result = await watcher.reject_action("nonexistent", "admin", "reason")
        
        assert "error" in result
    
    def test_get_pending_approvals(self, watcher, sample_alert):
        """Test getting pending approvals list."""
        investigation = Investigation(
            alert=sample_alert,
            started_at=datetime.now(timezone.utc),
        )
        action = RemediationAction(
            id="action-1",
            script_id="test",
            name="Test",
            description="Test",
            risk_level=RemediationRisk.HIGH,
            investigation=investigation,
        )
        
        # Create request manually (sync version)
        now = datetime.now(timezone.utc)
        request = ApprovalRequest(
            id=str(uuid.uuid4()),
            action=action,
            investigation=investigation,
            requested_at=now,
            expires_at=now + timedelta(hours=1),
        )
        watcher._pending_approvals[request.id] = request
        
        pending = watcher.get_pending_approvals()
        
        assert len(pending) == 1
        assert pending[0]["status"] == "pending"
    
    def test_get_pending_approvals_cleans_expired(self, watcher, sample_alert):
        """Test that getting pending approvals marks expired ones."""
        investigation = Investigation(
            alert=sample_alert,
            started_at=datetime.now(timezone.utc),
        )
        action = RemediationAction(
            id="action-1",
            script_id="test",
            name="Test",
            description="Test",
            risk_level=RemediationRisk.HIGH,
            investigation=investigation,
        )
        
        # Create expired request
        now = datetime.now(timezone.utc)
        request = ApprovalRequest(
            id=str(uuid.uuid4()),
            action=action,
            investigation=investigation,
            requested_at=now - timedelta(hours=2),
            expires_at=now - timedelta(hours=1),
        )
        watcher._pending_approvals[request.id] = request
        
        pending = watcher.get_pending_approvals()
        
        # Expired requests should be marked and excluded
        assert len(pending) == 0
        assert request.status == "expired"


# =============================================================================
# Built-in Remediation Handler Tests
# =============================================================================


class TestBuiltinRemediations:
    """Tests for built-in remediation handlers."""

    @pytest.mark.asyncio
    async def test_restart_workflow(self, watcher):
        """Test restart workflow handler."""
        result = await watcher._remediate_restart_workflow("exec-123")
        
        assert "restart" in result.lower()
        assert "exec-123" in result
    
    @pytest.mark.asyncio
    async def test_retry_step(self, watcher):
        """Test retry step handler."""
        result = await watcher._remediate_retry_step("exec-123", "step-1")
        
        assert "retry" in result.lower()
        assert "step-1" in result
    
    @pytest.mark.asyncio
    async def test_increase_timeout(self, watcher):
        """Test increase timeout handler."""
        result = await watcher._remediate_increase_timeout("step-1", 60000)
        
        assert "timeout" in result.lower()
        assert "60000" in result
    
    @pytest.mark.asyncio
    async def test_notify_oncall(self, watcher):
        """Test notify on-call handler."""
        result = await watcher._remediate_notify_oncall("Test alert", "warning")
        
        assert "notified" in result.lower()
    
    @pytest.mark.asyncio
    async def test_disable_agent(self, watcher):
        """Test disable agent handler."""
        result = await watcher._remediate_disable_agent("test-agent", "High error rate")
        
        assert "disabled" in result.lower()
        assert "test-agent" in result


# =============================================================================
# API Endpoint Tests
# =============================================================================


class TestWatcherAPIEndpoints:
    """Tests for Watcher Agent API endpoints."""

    @pytest.fixture
    def app(self):
        """Create a fresh test application."""
        try:
            from awf.api.app import create_app
            return create_app(debug=True)
        except ImportError:
            pytest.skip("FastAPI not installed")
    
    @pytest.fixture
    def client(self, app):
        """Create a test client."""
        try:
            from fastapi.testclient import TestClient
            with TestClient(app) as test_client:
                yield test_client
        except ImportError:
            pytest.skip("FastAPI not installed")
    
    def test_watcher_health_endpoint(self, client):
        """Test watcher health endpoint."""
        response = client.get("/watcher/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "pending_approvals" in data
        assert "registered_scripts" in data
    
    def test_receive_grafana_alert(self, client):
        """Test receiving Grafana alert webhook."""
        payload = {
            "title": "Test Alert",
            "state": "alerting",
            "message": "Test message",
            "labels": {"severity": "warning"},
            "annotations": {},
        }
        
        response = client.post("/webhooks/grafana-alerts", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_receive_grafana_alert_resolved(self, client):
        """Test that resolved alerts are ignored."""
        payload = {
            "title": "Test Alert",
            "state": "ok",
            "message": "Resolved",
            "labels": {},
            "annotations": {},
        }
        
        response = client.post("/webhooks/grafana-alerts", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ignored"
        assert data["reason"] == "alert_resolved"
    
    def test_list_pending_approvals_empty(self, client):
        """Test listing approvals when empty."""
        response = client.get("/watcher/approvals")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_approval_not_found(self, client):
        """Test getting non-existent approval."""
        response = client.get("/watcher/approvals/nonexistent-id")
        
        assert response.status_code == 404
    
    def test_approve_action_not_found(self, client):
        """Test approving non-existent request."""
        response = client.post(
            "/watcher/approvals/nonexistent/approve",
            json={"user": "admin"},
        )
        
        assert response.status_code == 404
    
    def test_reject_action_missing_reason(self, client):
        """Test rejecting without reason."""
        response = client.post(
            "/watcher/approvals/nonexistent/reject",
            json={"user": "admin"},  # Missing reason
        )
        
        assert response.status_code == 400
