# Watcher Agent Guide

The Watcher Agent is AWF's autonomous observability agent. It acts as a specialized AI SRE (Site Reliability Engineer) that monitors your agent fleet, investigates anomalies, and takes remediation actions to maintain system health.

## Overview

Unlike traditional observability where a human must respond to alerts, the Watcher Agent provides **Agentic Observability**:

1. **Alert Reception**: Receives webhooks from Grafana Alerting.
2. **Autonomous Investigation**: Queries metrics (Mimir), logs (Loki), and traces (Tempo) via the Grafana MCP Server to identify the root cause.
3. **Reasoning**: Determines the best remediation action based on the gathered evidence.
4. **Remediation**: Executes pre-approved scripts (e.g., restarting a workflow, retrying a step).
5. **HITL Integration**: Requests human approval for high-risk actions.

## Configuration

The Watcher Agent is configured using the `WatcherConfig` dataclass.

```python
from awf.agents.watcher import WatcherAgent, WatcherConfig

config = WatcherConfig(
    # Endpoint for the Grafana MCP Server
    mcp_endpoint="http://localhost:8686",
    
    # Actions that ALWAYS require human-in-the-loop approval
    hitl_required_actions=["agent.disable", "workflow.cancel"],
    
    # Directory containing custom remediation scripts
    remediation_scripts_dir="./remediations",
    
    # Enable autonomous action (if False, only investigations are performed)
    auto_remediation_enabled=True,
    
    # If True, no actions are actually executed
    dry_run_mode=False
)

watcher = WatcherAgent(config)
await watcher.start()
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mcp_endpoint` | `str` | `http://localhost:8686` | Grafana MCP Server URL |
| `hitl_required_actions` | `List[str]` | `["agent.disable", ...]` | Actions requiring approval |
| `auto_remediation_enabled` | `bool` | `True` | Enable autonomous actions |
| `approval_timeout_seconds` | `int` | `3600` | How long to wait for HITL approval |
| `slack_webhook_url` | `str` | `None` | Optional Slack URL for notifications |

## Built-in Remediation Scripts

AWF includes several built-in remediation scripts that the Watcher Agent can use:

| Script ID | Action | Risk Level | Description |
|-----------|--------|------------|-------------|
| `restart_workflow` | Restart Workflow | LOW | Restarts a failed workflow with the same input. |
| `retry_step` | Retry Step | LOW | Retries a specific failed step within an execution. |
| `increase_timeout` | Increase Timeout | MEDIUM | Increases the timeout threshold for a specific step. |
| `notify_oncall` | Notify On-Call | LOW | Sends a notification to Slack or other configured channels. |
| `disable_agent` | Disable Agent | HIGH | Temporarily disables a problematic agent in the registry. |

## Creating Custom Remediation Scripts

You can extend the Watcher Agent by registering custom remediation scripts.

### Python-based Remediation

```python
from awf.agents.watcher import RemediationScript, RemediationRisk

async def my_custom_handler(execution_id: str, reason: str):
    # Custom logic here...
    return f"Successfully handled {execution_id}"

watcher.register_script(RemediationScript(
    id="custom_remediation",
    name="Custom Fix",
    description="Applies a custom fix to the system",
    risk_level=RemediationRisk.MEDIUM,
    python_callable=my_custom_handler,
    parameters_schema={
        "execution_id": {"type": "string", "required": True},
        "reason": {"type": "string", "required": True}
    }
))
```

### Shell-based Remediation

Place scripts in the `remediation_scripts_dir`. Parameters are passed as environment variables with the `PARAM_` prefix.

```bash
#!/bin/bash
# remediations/clear_cache.sh
echo "Clearing cache for agent: $PARAM_AGENT_ID"
# execution logic...
```

## HITL Approval Workflow

When the Watcher Agent determines that a **HIGH** risk action is needed (or an action listed in `hitl_required_actions`), it creates an approval request.

### Approval Process

1. **Request Created**: The agent creates an `ApprovalRequest` and pauses remediation.
2. **Notification**: If configured, a notification is sent (e.g., to Slack).
3. **Human Review**: An operator reviews the investigation and the proposed action via the AWF API or UI.
4. **Decision**: The operator approves or rejects the action.
5. **Resume**: If approved, the agent executes the action and logs the results.

### API Endpoints for Approvals

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/watcher/approvals` | `GET` | List all pending approval requests. |
| `/watcher/approvals/{id}` | `GET` | Get detailed investigation for a request. |
| `/watcher/approvals/{id}/approve` | `POST` | Approve the action. |
| `/watcher/approvals/{id}/reject` | `POST` | Reject the action (reason required). |

## Example Usage Scenario

**Scenario: An agent is timing out due to a temporary spike in latency.**

1. **Grafana Alert**: A "Step Latency P99 > 30s" alert triggers.
2. **Watcher Investigates**: The agent queries Loki and sees "TimeoutError" in logs. It queries Mimir and sees a spike in provider latency.
3. **Action Determined**: The agent decides to use `increase_timeout`.
4. **Execution**: Since `increase_timeout` is MEDIUM risk and not in `hitl_required_actions`, the agent executes it immediately.
5. **Verification**: The agent creates an annotation in the Grafana dashboard: "Remediation success: Increase Step Timeout".
