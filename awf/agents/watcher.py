"""
AI Workflow Fabric - Watcher Agent

The Watcher Agent is AWF's autonomous observability agent that:
- Receives alerts from Grafana
- Investigates issues by querying metrics, logs, and traces via MCP
- Executes pre-approved remediation scripts
- Requests human approval for high-risk actions

This is AWF's key differentiator - providing "agentic observability" where
an AI agent actively monitors and remediates issues in production.

Usage:
    from awf.agents.watcher import WatcherAgent, WatcherConfig
    
    config = WatcherConfig(
        mcp_endpoint="http://localhost:8686",
        hitl_required_actions=["agent.disable", "workflow.cancel"],
    )
    
    watcher = WatcherAgent(config)
    await watcher.start()
    
    # The agent now listens for alerts and handles them autonomously
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import subprocess
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from awf.core.types import (
    AgentManifest,
    AgentStatus,
    Capability,
    CapabilityType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class RemediationRisk(str, Enum):
    """Risk level for remediation actions."""
    
    LOW = "low"       # Auto-execute without approval
    MEDIUM = "medium" # Auto-execute with notification
    HIGH = "high"     # Requires HITL approval


@dataclass
class WatcherConfig:
    """Configuration for the Watcher Agent."""
    
    # MCP Server endpoint (Grafana MCP)
    mcp_endpoint: str = "http://localhost:8686"
    
    # Actions that require human approval
    hitl_required_actions: List[str] = field(default_factory=lambda: [
        "agent.disable",
        "workflow.cancel",
        "config.change",
        "scale.down",
    ])
    
    # Path to remediation scripts
    remediation_scripts_dir: str = "./remediations"
    
    # Approval timeout
    approval_timeout_seconds: int = 3600  # 1 hour
    
    # Alert processing settings
    max_concurrent_investigations: int = 5
    investigation_timeout_seconds: int = 60
    
    # Notification settings
    notification_webhook_url: Optional[str] = None
    slack_webhook_url: Optional[str] = None
    
    # LLM settings for investigation
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    
    # Auto-remediation settings
    auto_remediation_enabled: bool = True
    dry_run_mode: bool = False


# =============================================================================
# Agent Manifest
# =============================================================================


def get_watcher_manifest() -> AgentManifest:
    """Get the Watcher Agent's ASP manifest."""
    return AgentManifest(
        id="awf-watcher",
        name="AWF Watcher Agent",
        version="1.0.0",
        description="Autonomous monitoring and remediation agent for AWF workflows. "
                    "Uses Grafana MCP Server to query observability data and execute "
                    "pre-approved remediation actions.",
        framework="native",
        capabilities=[
            Capability(
                name="grafana_query",
                type=CapabilityType.TOOL,
                description="Query Grafana datasources via MCP server",
                permissions=["grafana:read", "grafana:query"],
            ),
            Capability(
                name="alert_investigation",
                type=CapabilityType.REASONING,
                description="Investigate alerts by correlating metrics, logs, and traces",
            ),
            Capability(
                name="remediation_execution",
                type=CapabilityType.TOOL,
                description="Execute pre-approved remediation scripts",
                permissions=["awf:workflow:restart", "awf:agent:reconfigure"],
            ),
            Capability(
                name="hitl_approval",
                type=CapabilityType.COMMUNICATION,
                description="Request and track human-in-the-loop approvals",
            ),
        ],
        tags=["observability", "monitoring", "remediation", "autonomous"],
        status=AgentStatus.ACTIVE,
    )


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class GrafanaAlert:
    """Alert received from Grafana Alerting."""
    
    id: str
    title: str
    message: str
    state: str  # "alerting", "ok", "pending", "no_data"
    labels: Dict[str, str]
    annotations: Dict[str, str]
    starts_at: datetime
    ends_at: Optional[datetime] = None
    generator_url: Optional[str] = None
    fingerprint: Optional[str] = None
    
    @classmethod
    def from_webhook(cls, data: Dict[str, Any]) -> GrafanaAlert:
        """Create from Grafana webhook payload."""
        # Handle startsAt - can be None, string, or datetime
        starts_at_raw = data.get("startsAt")
        if starts_at_raw is None:
            starts_at = datetime.now(timezone.utc)
        elif isinstance(starts_at_raw, str):
            starts_at = datetime.fromisoformat(starts_at_raw.replace("Z", "+00:00"))
        else:
            starts_at = starts_at_raw
        
        # Handle endsAt
        ends_at_raw = data.get("endsAt")
        ends_at = None
        if ends_at_raw:
            if isinstance(ends_at_raw, str):
                ends_at = datetime.fromisoformat(ends_at_raw.replace("Z", "+00:00"))
            else:
                ends_at = ends_at_raw
        
        return cls(
            id=data.get("alertId") or str(uuid.uuid4()),
            title=data.get("title", "Unknown Alert"),
            message=data.get("message", ""),
            state=data.get("state", "alerting"),
            labels=data.get("labels", {}),
            annotations=data.get("annotations", {}),
            starts_at=starts_at,
            ends_at=ends_at,
            generator_url=data.get("generatorURL"),
            fingerprint=data.get("fingerprint"),
        )


@dataclass
class Investigation:
    """Result of investigating an alert."""
    
    alert: GrafanaAlert
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Data collected
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    traces: List[Dict[str, Any]] = field(default_factory=list)
    
    # Analysis
    root_cause: Optional[str] = None
    confidence: float = 0.0
    affected_components: List[str] = field(default_factory=list)
    related_alerts: List[str] = field(default_factory=list)
    
    # Recommendation
    recommended_action: Optional[str] = None
    action_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert.id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "affected_components": self.affected_components,
            "recommended_action": self.recommended_action,
            "action_parameters": self.action_parameters,
        }


@dataclass
class RemediationAction:
    """A remediation action to execute."""
    
    id: str
    script_id: str
    name: str
    description: str
    risk_level: RemediationRisk
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 30000
    
    # Context
    investigation: Optional[Investigation] = None
    triggered_by: str = "watcher"
    
    def requires_approval(self, config: WatcherConfig) -> bool:
        """Check if this action requires HITL approval."""
        if self.risk_level == RemediationRisk.HIGH:
            return True
        if self.script_id in config.hitl_required_actions:
            return True
        return False


@dataclass
class RemediationResult:
    """Result of executing a remediation."""
    
    action_id: str
    success: bool
    output: str
    error: Optional[str] = None
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time_ms: int = 0
    dry_run: bool = False


@dataclass
class ApprovalRequest:
    """A pending HITL approval request."""
    
    id: str
    action: RemediationAction
    investigation: Investigation
    requested_at: datetime
    expires_at: datetime
    status: str = "pending"  # "pending", "approved", "rejected", "expired"
    approved_by: Optional[str] = None
    rejected_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if approval request has expired."""
        return datetime.now(timezone.utc) > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "action": {
                "script_id": self.action.script_id,
                "name": self.action.name,
                "description": self.action.description,
                "risk_level": self.action.risk_level.value,
                "parameters": self.action.parameters,
            },
            "investigation": self.investigation.to_dict(),
            "requested_at": self.requested_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "status": self.status,
            "approved_by": self.approved_by,
            "rejected_by": self.rejected_by,
            "rejection_reason": self.rejection_reason,
        }


# =============================================================================
# Remediation Scripts
# =============================================================================


@dataclass
class RemediationScript:
    """A remediation script definition."""
    
    id: str
    name: str
    description: str
    risk_level: RemediationRisk
    script_path: Optional[str] = None
    python_callable: Optional[Callable] = None
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        timeout_ms: int = 30000,
        dry_run: bool = False,
    ) -> RemediationResult:
        """Execute the remediation script."""
        start_time = datetime.now(timezone.utc)
        action_id = str(uuid.uuid4())
        
        try:
            if dry_run:
                return RemediationResult(
                    action_id=action_id,
                    success=True,
                    output=f"[DRY RUN] Would execute {self.name} with params: {parameters}",
                    dry_run=True,
                    execution_time_ms=0,
                )
            
            if self.python_callable:
                # Execute Python callable
                result = await self._execute_callable(parameters, timeout_ms)
            elif self.script_path:
                # Execute external script
                result = await self._execute_script(parameters, timeout_ms)
            else:
                return RemediationResult(
                    action_id=action_id,
                    success=False,
                    output="",
                    error="No execution method defined for this script",
                )
            
            execution_time = int(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
            
            return RemediationResult(
                action_id=action_id,
                success=True,
                output=result,
                execution_time_ms=execution_time,
            )
            
        except asyncio.TimeoutError:
            return RemediationResult(
                action_id=action_id,
                success=False,
                output="",
                error=f"Execution timed out after {timeout_ms}ms",
            )
        except Exception as e:
            return RemediationResult(
                action_id=action_id,
                success=False,
                output="",
                error=str(e),
            )
    
    async def _execute_callable(
        self,
        parameters: Dict[str, Any],
        timeout_ms: int,
    ) -> str:
        """Execute Python callable."""
        if inspect.iscoroutinefunction(self.python_callable):
            result = await asyncio.wait_for(
                self.python_callable(**parameters),
                timeout=timeout_ms / 1000,
            )
        else:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.python_callable(**parameters),
                ),
                timeout=timeout_ms / 1000,
            )
        return str(result)
    
    async def _execute_script(
        self,
        parameters: Dict[str, Any],
        timeout_ms: int,
    ) -> str:
        """Execute external script."""
        # Convert parameters to environment variables
        env = {f"PARAM_{k.upper()}": str(v) for k, v in parameters.items()}
        
        process = await asyncio.create_subprocess_exec(
            self.script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_ms / 1000,
        )
        
        if process.returncode != 0:
            raise RuntimeError(f"Script failed: {stderr.decode()}")
        
        return stdout.decode()


# =============================================================================
# MCP Client for Grafana
# =============================================================================


class GrafanaMCPClient:
    """
    Client for Grafana MCP Server.
    
    The Grafana MCP Server (github.com/grafana/mcp-grafana) provides 52 tools
    for AI integration with Grafana, including:
    - Prometheus/Mimir queries
    - Loki log queries
    - Tempo trace queries
    - Dashboard management
    - Alert management
    """
    
    def __init__(self, endpoint: str = "http://localhost:8686"):
        """
        Initialize MCP client.
        
        Args:
            endpoint: MCP server endpoint
        """
        self.endpoint = endpoint
        self._session = None
    
    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call an MCP tool.
        
        Note: This is a simplified implementation. In production, you'd use
        the full MCP protocol with proper transport handling.
        """
        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not installed, MCP calls will be mocked")
            return self._mock_response(tool_name, arguments)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.endpoint}/tools/{tool_name}",
                json=arguments,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise RuntimeError(f"MCP call failed: {error}")
    
    def _mock_response(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Mock response for when MCP is not available."""
        return {
            "status": "mocked",
            "tool": tool_name,
            "arguments": arguments,
            "message": "MCP client is running in mock mode",
        }
    
    async def query_prometheus(
        self,
        query: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        step: str = "1m",
    ) -> Dict[str, Any]:
        """
        Execute PromQL query.
        
        Args:
            query: PromQL query string
            start: Start time (default: 1h ago)
            end: End time (default: now)
            step: Query resolution step
        
        Returns:
            Prometheus query result
        """
        now = datetime.now(timezone.utc)
        return await self._call_tool("query_prometheus", {
            "query": query,
            "start": (start or now - timedelta(hours=1)).isoformat(),
            "end": (end or now).isoformat(),
            "step": step,
        })
    
    async def query_loki(
        self,
        query: str,
        limit: int = 100,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Execute LogQL query.
        
        Args:
            query: LogQL query string
            limit: Maximum number of log lines
            start: Start time
            end: End time
        
        Returns:
            Loki query result
        """
        now = datetime.now(timezone.utc)
        return await self._call_tool("query_loki", {
            "query": query,
            "limit": limit,
            "start": (start or now - timedelta(hours=1)).isoformat(),
            "end": (end or now).isoformat(),
        })
    
    async def query_tempo(self, trace_id: str) -> Dict[str, Any]:
        """
        Get trace by ID.
        
        Args:
            trace_id: Trace ID to retrieve
        
        Returns:
            Tempo trace data
        """
        return await self._call_tool("query_tempo", {
            "traceId": trace_id,
        })
    
    async def search_traces(
        self,
        service_name: Optional[str] = None,
        operation: Optional[str] = None,
        min_duration: Optional[str] = None,
        max_duration: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Search for traces.
        
        Args:
            service_name: Filter by service
            operation: Filter by operation
            min_duration: Minimum duration (e.g., "100ms")
            max_duration: Maximum duration
            limit: Maximum results
        
        Returns:
            List of matching traces
        """
        params = {"limit": limit}
        if service_name:
            params["serviceName"] = service_name
        if operation:
            params["operation"] = operation
        if min_duration:
            params["minDuration"] = min_duration
        if max_duration:
            params["maxDuration"] = max_duration
        
        return await self._call_tool("search_traces", params)
    
    async def create_annotation(
        self,
        dashboard_uid: str,
        text: str,
        tags: Optional[List[str]] = None,
        time_from: Optional[datetime] = None,
        time_to: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Create dashboard annotation.
        
        Args:
            dashboard_uid: Dashboard UID
            text: Annotation text
            tags: Optional tags
            time_from: Start time
            time_to: End time (optional, for range annotations)
        
        Returns:
            Created annotation
        """
        now = datetime.now(timezone.utc)
        params = {
            "dashboardUid": dashboard_uid,
            "text": text,
            "time": int((time_from or now).timestamp() * 1000),
        }
        if tags:
            params["tags"] = tags
        if time_to:
            params["timeEnd"] = int(time_to.timestamp() * 1000)
        
        return await self._call_tool("create_annotation", params)
    
    async def list_alert_rules(
        self,
        folder: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List alert rules.
        
        Args:
            folder: Optional folder filter
        
        Returns:
            List of alert rules
        """
        params = {}
        if folder:
            params["folder"] = folder
        return await self._call_tool("list_alert_rules", params)
    
    async def get_dashboard(self, uid: str) -> Dict[str, Any]:
        """
        Get dashboard by UID.
        
        Args:
            uid: Dashboard UID
        
        Returns:
            Dashboard definition
        """
        return await self._call_tool("get_dashboard_by_uid", {"uid": uid})


# =============================================================================
# Watcher Agent
# =============================================================================


class WatcherAgent:
    """
    AWF Watcher Agent - Autonomous observability and remediation.
    
    The Watcher Agent:
    1. Receives alerts from Grafana
    2. Investigates by querying metrics, logs, and traces via MCP
    3. Determines the best remediation action
    4. Executes or requests approval based on risk level
    5. Logs all actions for audit trail
    
    Example:
        watcher = WatcherAgent(WatcherConfig(
            mcp_endpoint="http://localhost:8686",
            auto_remediation_enabled=True,
        ))
        
        # Register remediation scripts
        watcher.register_script(RemediationScript(
            id="restart_workflow",
            name="Restart Failed Workflow",
            description="Restart a workflow that failed due to transient error",
            risk_level=RemediationRisk.LOW,
            python_callable=restart_workflow_handler,
        ))
        
        await watcher.start()
    """
    
    def __init__(self, config: Optional[WatcherConfig] = None):
        """
        Initialize the Watcher Agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config or WatcherConfig()
        self.mcp = GrafanaMCPClient(self.config.mcp_endpoint)
        
        # Script registry
        self._scripts: Dict[str, RemediationScript] = {}
        
        # Pending approvals
        self._pending_approvals: Dict[str, ApprovalRequest] = {}
        
        # State
        self._started = False
        self._processing_lock = asyncio.Lock()
        self._active_investigations: Dict[str, asyncio.Task] = {}
        
        # Register built-in scripts
        self._register_builtin_scripts()
    
    def _register_builtin_scripts(self) -> None:
        """Register built-in remediation scripts."""
        
        # Restart workflow
        self.register_script(RemediationScript(
            id="restart_workflow",
            name="Restart Failed Workflow",
            description="Restart a workflow that failed due to a transient error",
            risk_level=RemediationRisk.LOW,
            python_callable=self._remediate_restart_workflow,
            parameters_schema={
                "execution_id": {"type": "string", "required": True},
            },
        ))
        
        # Retry step
        self.register_script(RemediationScript(
            id="retry_step",
            name="Retry Failed Step",
            description="Retry a specific step that failed",
            risk_level=RemediationRisk.LOW,
            python_callable=self._remediate_retry_step,
            parameters_schema={
                "execution_id": {"type": "string", "required": True},
                "step_id": {"type": "string", "required": True},
            },
        ))
        
        # Increase timeout
        self.register_script(RemediationScript(
            id="increase_timeout",
            name="Increase Step Timeout",
            description="Increase timeout for steps that are timing out",
            risk_level=RemediationRisk.MEDIUM,
            python_callable=self._remediate_increase_timeout,
            parameters_schema={
                "step_id": {"type": "string", "required": True},
                "new_timeout_ms": {"type": "integer", "required": True},
            },
        ))
        
        # Notify on-call
        self.register_script(RemediationScript(
            id="notify_oncall",
            name="Notify On-Call",
            description="Page the on-call engineer",
            risk_level=RemediationRisk.LOW,
            python_callable=self._remediate_notify_oncall,
            parameters_schema={
                "message": {"type": "string", "required": True},
                "severity": {"type": "string", "required": False},
            },
        ))
        
        # Disable agent (high risk)
        self.register_script(RemediationScript(
            id="disable_agent",
            name="Disable Problematic Agent",
            description="Temporarily disable an agent that is causing issues",
            risk_level=RemediationRisk.HIGH,
            python_callable=self._remediate_disable_agent,
            parameters_schema={
                "agent_id": {"type": "string", "required": True},
                "reason": {"type": "string", "required": True},
            },
        ))
    
    def register_script(self, script: RemediationScript) -> None:
        """
        Register a remediation script.
        
        Args:
            script: Remediation script to register
        """
        self._scripts[script.id] = script
        logger.info(f"Registered remediation script: {script.id}")
    
    async def start(self) -> None:
        """Start the Watcher Agent."""
        if self._started:
            logger.warning("Watcher Agent already started")
            return
        
        self._started = True
        logger.info("Watcher Agent started")
    
    async def stop(self) -> None:
        """Stop the Watcher Agent."""
        if not self._started:
            return
        
        # Cancel active investigations
        for task in self._active_investigations.values():
            task.cancel()
        
        self._started = False
        logger.info("Watcher Agent stopped")
    
    # =========================================================================
    # Alert Handling
    # =========================================================================
    
    async def handle_alert(self, alert: GrafanaAlert) -> Dict[str, Any]:
        """
        Handle an incoming alert from Grafana.
        
        This is the main entry point for alert processing.
        
        Args:
            alert: Grafana alert to process
        
        Returns:
            Response with action taken
        """
        logger.info(f"Handling alert: {alert.title} (state={alert.state})")
        
        # Skip resolved alerts
        if alert.state == "ok":
            return {"status": "ignored", "reason": "alert_resolved"}
        
        # Check concurrent investigation limit
        if len(self._active_investigations) >= self.config.max_concurrent_investigations:
            return {"status": "queued", "reason": "max_concurrent_reached"}
        
        # Start investigation
        investigation = await self._investigate_alert(alert)
        
        # Determine action
        action = await self._determine_action(investigation)
        
        if action is None:
            return {
                "status": "no_action",
                "investigation": investigation.to_dict(),
            }
        
        # Execute or request approval
        if action.requires_approval(self.config):
            approval_request = await self._request_hitl_approval(action)
            return {
                "status": "pending_approval",
                "approval_id": approval_request.id,
                "action": action.name,
            }
        else:
            result = await self._execute_remediation(action)
            return {
                "status": "remediated" if result.success else "remediation_failed",
                "action": action.name,
                "result": {
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                },
            }
    
    async def _investigate_alert(self, alert: GrafanaAlert) -> Investigation:
        """
        Investigate an alert by gathering context.
        
        Args:
            alert: Alert to investigate
        
        Returns:
            Investigation result with gathered context
        """
        investigation = Investigation(
            alert=alert,
            started_at=datetime.now(timezone.utc),
        )
        
        try:
            # Extract context from alert labels
            execution_id = alert.labels.get("execution_id")
            workflow_id = alert.labels.get("workflow_id")
            step_id = alert.labels.get("step_id")
            agent_id = alert.labels.get("agent_id")
            trace_id = alert.labels.get("trace_id")
            
            # Query metrics
            if agent_id:
                investigation.metrics["agent_success_rate"] = await self.mcp.query_prometheus(
                    f'awf:agent_success_rate:ratio_rate5m{{agent_id="{agent_id}"}}'
                )
                investigation.metrics["agent_latency_p99"] = await self.mcp.query_prometheus(
                    f'awf:step_latency_p99:5m:by_agent{{agent_id="{agent_id}"}}'
                )
            
            if workflow_id:
                investigation.metrics["workflow_success_rate"] = await self.mcp.query_prometheus(
                    f'awf:workflow_success_rate:ratio_rate5m:by_workflow{{workflow_id="{workflow_id}"}}'
                )
            
            # Query logs
            log_query_parts = ['{service_namespace="awf"}']
            if execution_id:
                log_query_parts.append(f'|= "{execution_id}"')
            if step_id:
                log_query_parts.append(f'|= "{step_id}"')
            
            log_query = " ".join(log_query_parts)
            investigation.logs = await self.mcp.query_loki(log_query, limit=50)
            
            # Query traces
            if trace_id:
                investigation.traces = [await self.mcp.query_tempo(trace_id)]
            
            # Analyze and determine root cause
            investigation.root_cause = self._analyze_root_cause(investigation)
            investigation.confidence = self._calculate_confidence(investigation)
            investigation.affected_components = self._identify_affected_components(investigation)
            
            investigation.completed_at = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Error during investigation: {e}")
            investigation.root_cause = f"Investigation failed: {str(e)}"
            investigation.confidence = 0.0
        
        return investigation
    
    def _analyze_root_cause(self, investigation: Investigation) -> str:
        """
        Analyze gathered data to determine root cause.
        
        In a production system, this would use an LLM for intelligent analysis.
        For now, we use simple heuristics.
        """
        alert = investigation.alert
        
        # Check for common patterns
        if "timeout" in alert.title.lower() or "timeout" in alert.message.lower():
            return "Step execution exceeded configured timeout threshold"
        
        if "error_rate" in alert.title.lower():
            return "Agent error rate exceeded acceptable threshold"
        
        if "latency" in alert.title.lower():
            return "Step latency exceeded P99 threshold"
        
        if "budget" in alert.title.lower():
            return "Error budget exhausted or burning too fast"
        
        return f"Alert triggered: {alert.title}"
    
    def _calculate_confidence(self, investigation: Investigation) -> float:
        """Calculate confidence in root cause analysis."""
        confidence = 0.5  # Base confidence
        
        # More data = higher confidence
        if investigation.metrics:
            confidence += 0.2
        if investigation.logs:
            confidence += 0.15
        if investigation.traces:
            confidence += 0.15
        
        return min(confidence, 1.0)
    
    def _identify_affected_components(self, investigation: Investigation) -> List[str]:
        """Identify components affected by the issue."""
        components = []
        
        labels = investigation.alert.labels
        if labels.get("agent_id"):
            components.append(f"agent:{labels['agent_id']}")
        if labels.get("workflow_id"):
            components.append(f"workflow:{labels['workflow_id']}")
        if labels.get("step_id"):
            components.append(f"step:{labels['step_id']}")
        
        return components
    
    async def _determine_action(
        self,
        investigation: Investigation,
    ) -> Optional[RemediationAction]:
        """
        Determine the best remediation action.
        
        In production, this would use an LLM to reason about the best action.
        For now, we use rule-based matching.
        """
        alert = investigation.alert
        root_cause = investigation.root_cause or ""
        
        # Match alert to remediation
        if "timeout" in root_cause.lower():
            return RemediationAction(
                id=str(uuid.uuid4()),
                script_id="increase_timeout",
                name="Increase Step Timeout",
                description=f"Increase timeout for step due to: {root_cause}",
                risk_level=RemediationRisk.MEDIUM,
                parameters={
                    "step_id": alert.labels.get("step_id", "unknown"),
                    "new_timeout_ms": 60000,  # 60s
                },
                investigation=investigation,
            )
        
        if "error_rate" in alert.title.lower() and investigation.confidence > 0.7:
            agent_id = alert.labels.get("agent_id")
            if agent_id:
                return RemediationAction(
                    id=str(uuid.uuid4()),
                    script_id="disable_agent",
                    name="Disable Problematic Agent",
                    description=f"Disable agent {agent_id} due to high error rate",
                    risk_level=RemediationRisk.HIGH,
                    parameters={
                        "agent_id": agent_id,
                        "reason": root_cause,
                    },
                    investigation=investigation,
                )
        
        # Default: notify on-call
        return RemediationAction(
            id=str(uuid.uuid4()),
            script_id="notify_oncall",
            name="Notify On-Call",
            description=f"Alert requires human attention: {alert.title}",
            risk_level=RemediationRisk.LOW,
            parameters={
                "message": f"Alert: {alert.title}\nRoot Cause: {root_cause}",
                "severity": "warning",
            },
            investigation=investigation,
        )
    
    # =========================================================================
    # Remediation Execution
    # =========================================================================
    
    async def _execute_remediation(
        self,
        action: RemediationAction,
    ) -> RemediationResult:
        """
        Execute a remediation action.
        
        Args:
            action: Action to execute
        
        Returns:
            Execution result
        """
        script = self._scripts.get(action.script_id)
        if not script:
            return RemediationResult(
                action_id=action.id,
                success=False,
                output="",
                error=f"Unknown remediation script: {action.script_id}",
            )
        
        logger.info(f"Executing remediation: {action.name}")
        
        result = await script.execute(
            parameters=action.parameters,
            timeout_ms=action.timeout_ms,
            dry_run=self.config.dry_run_mode,
        )
        
        # Log to Grafana
        await self._log_remediation(action, result)
        
        return result
    
    async def _log_remediation(
        self,
        action: RemediationAction,
        result: RemediationResult,
    ) -> None:
        """Log remediation to Grafana as annotation."""
        try:
            status = "success" if result.success else "failed"
            await self.mcp.create_annotation(
                dashboard_uid="awf-overview",
                text=f"Remediation {status}: {action.name}",
                tags=["remediation", action.script_id, status],
            )
        except Exception as e:
            logger.warning(f"Failed to create annotation: {e}")
    
    # =========================================================================
    # Built-in Remediation Handlers
    # =========================================================================
    
    async def _remediate_restart_workflow(
        self,
        execution_id: str,
    ) -> str:
        """Restart a failed workflow."""
        logger.info(f"Restarting workflow: {execution_id}")
        # In production, this would call the orchestrator API
        return f"Workflow {execution_id} restart initiated"
    
    async def _remediate_retry_step(
        self,
        execution_id: str,
        step_id: str,
    ) -> str:
        """Retry a failed step."""
        logger.info(f"Retrying step: {step_id} in execution {execution_id}")
        return f"Step {step_id} retry initiated"
    
    async def _remediate_increase_timeout(
        self,
        step_id: str,
        new_timeout_ms: int,
    ) -> str:
        """Increase step timeout."""
        logger.info(f"Increasing timeout for step {step_id} to {new_timeout_ms}ms")
        return f"Timeout for {step_id} increased to {new_timeout_ms}ms"
    
    async def _remediate_notify_oncall(
        self,
        message: str,
        severity: str = "warning",
    ) -> str:
        """Notify on-call engineer."""
        logger.info(f"Notifying on-call: {message}")
        
        # Send to Slack if configured
        if self.config.slack_webhook_url:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        self.config.slack_webhook_url,
                        json={
                            "text": f"*AWF Alert ({severity})*\n{message}",
                        },
                    )
            except Exception as e:
                logger.error(f"Failed to send Slack notification: {e}")
        
        return f"On-call notified: {message[:100]}..."
    
    async def _remediate_disable_agent(
        self,
        agent_id: str,
        reason: str,
    ) -> str:
        """Disable a problematic agent."""
        logger.warning(f"Disabling agent {agent_id}: {reason}")
        # In production, this would update the registry
        return f"Agent {agent_id} disabled: {reason}"
    
    # =========================================================================
    # HITL Approval
    # =========================================================================
    
    async def _request_hitl_approval(
        self,
        action: RemediationAction,
    ) -> ApprovalRequest:
        """
        Request human-in-the-loop approval for high-risk action.
        
        Args:
            action: Action requiring approval
        
        Returns:
            Approval request
        """
        now = datetime.now(timezone.utc)
        
        request = ApprovalRequest(
            id=str(uuid.uuid4()),
            action=action,
            investigation=action.investigation,
            requested_at=now,
            expires_at=now + timedelta(seconds=self.config.approval_timeout_seconds),
        )
        
        self._pending_approvals[request.id] = request
        
        # Notify approvers
        await self._notify_approvers(request)
        
        logger.info(f"Created approval request: {request.id} for action: {action.name}")
        
        return request
    
    async def _notify_approvers(self, request: ApprovalRequest) -> None:
        """Notify approvers of pending request."""
        message = (
            f"*Approval Required*\n"
            f"Action: {request.action.name}\n"
            f"Risk Level: {request.action.risk_level.value}\n"
            f"Description: {request.action.description}\n"
            f"Expires: {request.expires_at.isoformat()}\n"
            f"Approval ID: `{request.id}`"
        )
        
        if self.config.slack_webhook_url:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        self.config.slack_webhook_url,
                        json={"text": message},
                    )
            except Exception as e:
                logger.error(f"Failed to send approval notification: {e}")
    
    async def approve_action(
        self,
        approval_id: str,
        approver: str,
    ) -> Dict[str, Any]:
        """
        Approve a pending action.
        
        Args:
            approval_id: Approval request ID
            approver: Username/ID of approver
        
        Returns:
            Execution result
        """
        request = self._pending_approvals.get(approval_id)
        if not request:
            return {"error": "Approval request not found"}
        
        if request.is_expired():
            request.status = "expired"
            return {"error": "Approval request has expired"}
        
        if request.status != "pending":
            return {"error": f"Request already {request.status}"}
        
        # Mark as approved
        request.status = "approved"
        request.approved_by = approver
        
        # Execute the action
        result = await self._execute_remediation(request.action)
        
        return {
            "status": "executed",
            "approved_by": approver,
            "result": {
                "success": result.success,
                "output": result.output,
                "error": result.error,
            },
        }
    
    async def reject_action(
        self,
        approval_id: str,
        rejector: str,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Reject a pending action.
        
        Args:
            approval_id: Approval request ID
            rejector: Username/ID of rejector
            reason: Rejection reason
        
        Returns:
            Rejection status
        """
        request = self._pending_approvals.get(approval_id)
        if not request:
            return {"error": "Approval request not found"}
        
        if request.status != "pending":
            return {"error": f"Request already {request.status}"}
        
        request.status = "rejected"
        request.rejected_by = rejector
        request.rejection_reason = reason
        
        logger.info(f"Approval {approval_id} rejected by {rejector}: {reason}")
        
        return {
            "status": "rejected",
            "rejected_by": rejector,
            "reason": reason,
        }
    
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests."""
        # Clean up expired requests
        now = datetime.now(timezone.utc)
        for request in list(self._pending_approvals.values()):
            if request.status == "pending" and request.is_expired():
                request.status = "expired"
        
        # Return pending only
        return [
            request.to_dict()
            for request in self._pending_approvals.values()
            if request.status == "pending"
        ]
