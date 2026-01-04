# Grafana Integration - Product Requirements Document

**Version:** 1.0.0
**Status:** Approved
**Author:** AWF Team
**Created:** January 2025
**Last Updated:** January 2025

---

## 1. Executive Summary

This PRD defines the integration of Grafana OSS as AWF's observability layer, replacing the conceptual "Agent Observability Dashboard" (AOD) that was planned but never built. The integration includes a novel "Watcher Agent" that uses Grafana's MCP server to provide agentic observability capabilities.

### 1.1 Vision Statement

Provide production-grade observability for AI agent workflows through:
- **Battle-tested Infrastructure**: Grafana LGTM stack (Loki, Grafana, Tempo, Mimir)
- **OpenTelemetry Native**: Standard instrumentation bridging AWF's EventEmitter
- **Agentic Observability**: Watcher Agent that autonomously monitors, investigates, and remediates issues
- **OSS-First**: No vendor lock-in; enterprise Cloud features approximated in OSS

### 1.2 Success Criteria

1. All AWF workflow events visible in Grafana dashboards
2. Distributed traces across multi-step workflows
3. Watcher Agent successfully investigates and remediates 80%+ of alerts autonomously
4. Zero additional runtime dependencies for AWF core (observability is optional)
5. One-command deployment via Docker Compose

---

## 2. Problem Statement

### 2.1 Current State

AWF has:
- `awf/orchestration/events.py` - EventEmitter with pub/sub architecture
- `awf/api/app.py` - SSE endpoint at `/executions/{id}/events`
- `awf/core/types.py` - TaskMetrics (execution_time_ms, token_usage, etc.)
- 473 passing tests with comprehensive coverage

**What's Missing:**
- No persistent metrics storage
- No distributed tracing across workflow steps
- No dashboards or visualization
- No alerting on anomalies
- No autonomous investigation or remediation

### 2.2 Why Grafana Over Custom AOD

| Factor | Custom AOD | Grafana OSS |
|--------|------------|-------------|
| Time to Production | 6+ months | 2 weeks |
| Battle-tested | No | Yes (10+ years) |
| Ecosystem | Build from scratch | 4000+ plugins |
| Maintenance | Full burden | Community-supported |
| AI Features | Build custom | MCP Server + LLM Plugin |

**Decision**: Leverage Grafana's mature ecosystem instead of building from scratch.

### 2.3 Target Users

1. **Platform Engineers**: Deploying AWF in production
2. **DevOps Teams**: Monitoring agent fleet health
3. **Application Developers**: Debugging workflow failures
4. **SRE Teams**: Managing SLOs and incident response

---

## 3. Solution Overview

### 3.1 Architecture

```
AWF Orchestration Events ─────────────────────────────┐
    (existing EventEmitter)                           │
         │                                            │
         ▼                                            ▼
OTel Exporter (NEW) ──→ Grafana Alloy ──→ LGTM Stack ──→ Grafana UI
    awf/orchestration/       │               │              │
    otel_exporter.py         │               │              │
                             │               ▼              │
                             │    ┌─────────────────────┐   │
                             │    │  Mimir (metrics)    │   │
                             │    │  Loki (logs)        │   │
                             │    │  Tempo (traces)     │   │
                             │    └─────────────────────┘   │
                             │                              │
                             ▼                              │
                    Grafana MCP Server ◄────────────────────┘
                    (52 tools for AI integration)
                             │
                             ▼
                    AWF Watcher Agent (NEW)
                    - Query Grafana via MCP
                    - Receive alert webhooks
                    - Execute remediation scripts
                    - HITL approval workflow
```

### 3.2 Key Components

| Component | Responsibility | Location |
|-----------|----------------|----------|
| **OTel Exporter** | Bridge EventEmitter → OpenTelemetry | `awf/orchestration/otel_exporter.py` |
| **Grafana Alloy** | OTLP receiver, routing to backends | Docker container |
| **LGTM Stack** | Metrics, logs, traces storage | Docker containers |
| **Pre-built Dashboards** | AWF-specific visualizations | `dashboards/*.json` |
| **Watcher Agent** | Autonomous monitoring + remediation | `awf/agents/watcher.py` |
| **MCP Client** | Interface to Grafana MCP Server | Part of Watcher Agent |

### 3.3 What Already Exists (Leverage Points)

```python
# awf/orchestration/events.py - EventEmitter with pub/sub
class EventEmitter:
    async def emit(self, event: WorkflowEvent) -> None:
        for callback in self._subscribers.get(event.type, []):
            await callback(event)

# awf/core/types.py - TaskMetrics
@dataclass
class TaskMetrics:
    execution_time_ms: int
    token_usage: TokenUsage | None
    cost_usd: float | None
    retry_count: int
    
# awf/api/app.py - SSE endpoint
@app.get("/executions/{execution_id}/events")
async def stream_events(execution_id: str):
    async def event_generator():
        async for event in orchestrator.subscribe(execution_id):
            yield f"data: {event.json()}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## 4. Detailed Requirements

### 4.1 Epic 1: OpenTelemetry Instrumentation (13 Story Points)

#### 4.1.1 OTel Exporter Bridge

Create `awf/orchestration/otel_exporter.py` that:
- Subscribes to EventEmitter
- Converts WorkflowEvents to OTel spans/metrics
- Exports via OTLP to Grafana Alloy

```python
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

class OTelExporter:
    """Bridges AWF EventEmitter to OpenTelemetry."""
    
    def __init__(self, endpoint: str = "http://localhost:4317"):
        self.tracer = trace.get_tracer("awf.orchestration")
        self.meter = metrics.get_meter("awf.orchestration")
        
        # Custom AWF metrics
        self.step_duration = self.meter.create_histogram(
            "awf.step.duration",
            unit="ms",
            description="Step execution duration"
        )
        self.token_usage = self.meter.create_counter(
            "awf.tokens.total",
            description="Total tokens used"
        )
        self.workflow_cost = self.meter.create_counter(
            "awf.cost.usd",
            description="Total cost in USD"
        )
    
    async def on_event(self, event: WorkflowEvent) -> None:
        """Handle AWF events and export to OTel."""
        match event.type:
            case "workflow.started":
                self._start_trace(event)
            case "workflow.step.completed":
                self._record_step_metrics(event)
            case "workflow.completed":
                self._end_trace(event)
```

#### 4.1.2 Trace Spans

| Span | Attributes | When |
|------|------------|------|
| `awf.workflow` | `workflow_id`, `execution_id` | Workflow start/end |
| `awf.step` | `step_id`, `agent_id`, `framework` | Step start/end |
| `awf.agent.invoke` | `agent_id`, `input_tokens`, `output_tokens` | Agent invocation |

#### 4.1.3 Custom Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `awf_step_duration_ms` | Histogram | `step_id`, `agent_id`, `status` | Step execution time |
| `awf_tokens_total` | Counter | `agent_id`, `direction` (input/output) | Token usage |
| `awf_cost_usd_total` | Counter | `agent_id`, `provider` | Cost tracking |
| `awf_workflow_total` | Counter | `workflow_id`, `status` | Workflow completions |
| `awf_retry_total` | Counter | `step_id`, `agent_id` | Retry attempts |
| `awf_fallback_total` | Counter | `step_id` | Fallback executions |

### 4.2 Epic 2: Grafana Stack Deployment (21 Story Points)

#### 4.2.1 Docker Compose Configuration

```yaml
# docker/grafana/docker-compose.yml
version: '3.8'

services:
  # Grafana UI
  grafana:
    image: grafana/grafana-oss:11.4.0
    ports:
      - "3000:3000"
    volumes:
      - ./provisioning:/etc/grafana/provisioning
      - ./dashboards:/var/lib/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-llm-app
    depends_on:
      - mimir
      - loki
      - tempo

  # OpenTelemetry Collector
  alloy:
    image: grafana/alloy:v1.5.1
    ports:
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
    volumes:
      - ./alloy-config.river:etc/alloy/config.river
    command: run --server.http.listen-addr=0.0.0.0:12345 /etc/alloy/config.river

  # Metrics storage
  mimir:
    image: grafana/mimir:2.14.0
    ports:
      - "9009:9009"
    command: -config.file=/etc/mimir/config.yaml
    volumes:
      - ./mimir-config.yaml:/etc/mimir/config.yaml

  # Log storage
  loki:
    image: grafana/loki:3.2.0
    ports:
      - "3100:3100"
    command: -config.file=/etc/loki/config.yaml
    volumes:
      - ./loki-config.yaml:/etc/loki/config.yaml

  # Trace storage
  tempo:
    image: grafana/tempo:2.6.0
    ports:
      - "3200:3200"
    command: -config.file=/etc/tempo/config.yaml
    volumes:
      - ./tempo-config.yaml:/etc/tempo/config.yaml
```

#### 4.2.2 Alloy Configuration

```river
# alloy-config.river
otelcol.receiver.otlp "default" {
  grpc {
    endpoint = "0.0.0.0:4317"
  }
  http {
    endpoint = "0.0.0.0:4318"
  }
  output {
    metrics = [otelcol.processor.batch.default.input]
    traces  = [otelcol.processor.batch.default.input]
    logs    = [otelcol.processor.batch.default.input]
  }
}

otelcol.processor.batch "default" {
  output {
    metrics = [otelcol.exporter.prometheus.mimir.input]
    traces  = [otelcol.exporter.otlp.tempo.input]
    logs    = [otelcol.exporter.loki.default.input]
  }
}

otelcol.exporter.prometheus "mimir" {
  forward_to = [prometheus.remote_write.mimir.receiver]
}

prometheus.remote_write "mimir" {
  endpoint {
    url = "http://mimir:9009/api/v1/push"
  }
}

otelcol.exporter.otlp "tempo" {
  client {
    endpoint = "tempo:4317"
    tls {
      insecure = true
    }
  }
}

otelcol.exporter.loki "default" {
  forward_to = [loki.write.default.receiver]
}

loki.write "default" {
  endpoint {
    url = "http://loki:3100/loki/api/v1/push"
  }
}
```

#### 4.2.3 Pre-built Dashboards

**Dashboard 1: AWF Overview**
- Workflow success/failure rates (time series)
- Average step duration (histogram)
- Token usage trends (counter)
- Cost accumulation (gauge)
- Active executions (stat)

**Dashboard 2: Agent Fleet Health**
- Per-agent success rate (table)
- Agent latency percentiles (heatmap)
- Agent error rates (time series)
- Framework distribution (pie chart)
- Trust score distribution (bar gauge)

#### 4.2.4 SLO Recording Rules

```yaml
# prometheus/slo-rules.yml
groups:
  - name: awf_slos
    rules:
      # SLI: Workflow success rate
      - record: awf:workflow_success_rate:ratio_rate5m
        expr: |
          sum(rate(awf_workflow_total{status="completed"}[5m]))
          /
          sum(rate(awf_workflow_total[5m]))
      
      # SLI: Step latency P99
      - record: awf:step_latency_p99:5m
        expr: |
          histogram_quantile(0.99, 
            sum(rate(awf_step_duration_ms_bucket[5m])) by (le)
          )
      
      # Error budget remaining (target: 99.9%)
      - record: awf:error_budget_remaining:ratio
        expr: |
          1 - (
            (1 - awf:workflow_success_rate:ratio_rate5m)
            /
            (1 - 0.999)
          )
```

### 4.3 Epic 3: Watcher Agent (21 Story Points)

The Watcher Agent is AWF's differentiator - an AI agent that monitors the observability stack and takes autonomous action.

#### 4.3.1 Agent Manifest

```python
# awf/agents/watcher.py
from awf.core.types import AgentManifest, Capability, CapabilityType

WATCHER_MANIFEST = AgentManifest(
    id="awf-watcher",
    name="AWF Watcher Agent",
    version="1.0.0",
    description="Autonomous monitoring and remediation agent for AWF workflows",
    framework="native",  # Built into AWF
    capabilities=[
        Capability(
            name="grafana_query",
            type=CapabilityType.TOOL,
            description="Query Grafana via MCP server",
        ),
        Capability(
            name="alert_investigation",
            type=CapabilityType.SKILL,
            description="Investigate alerts by correlating metrics, logs, and traces",
        ),
        Capability(
            name="remediation_execution",
            type=CapabilityType.SKILL,
            description="Execute pre-approved remediation scripts",
        ),
    ],
    permissions=[
        "grafana:read",
        "grafana:query",
        "awf:workflow:restart",
        "awf:agent:reconfigure",
    ],
)
```

#### 4.3.2 Watcher Agent Implementation

```python
class WatcherAgent:
    """AWF agent that uses Grafana MCP Server for observability queries."""
    
    def __init__(
        self,
        mcp_endpoint: str = "http://localhost:8080",
        hitl_required_actions: list[str] = None,
    ):
        self.mcp = GrafanaMCPClient(mcp_endpoint)
        self.hitl_required = hitl_required_actions or [
            "workflow.restart",
            "agent.disable",
            "config.change",
        ]
        self.remediation_scripts: dict[str, RemediationScript] = {}
    
    async def handle_alert(self, alert: GrafanaAlert) -> AlertResponse:
        """Main entry point for Grafana alert webhooks."""
        
        # Step 1: Investigate
        investigation = await self.investigate_alert(alert)
        
        # Step 2: Determine action
        action = await self.determine_action(investigation)
        
        # Step 3: Execute or request approval
        if action.type in self.hitl_required:
            return await self.request_hitl_approval(action)
        else:
            return await self.execute_remediation(action)
    
    async def investigate_alert(self, alert: GrafanaAlert) -> Investigation:
        """Query Grafana for context around the alert."""
        
        # Get relevant metrics
        metrics = await self.mcp.query_prometheus(
            query=f'awf_step_duration_ms{{step_id="{alert.labels.get("step_id")}"}}[1h]',
        )
        
        # Get correlated logs
        logs = await self.mcp.query_loki(
            query=f'{{execution_id="{alert.labels.get("execution_id")}"}}',
            limit=100,
        )
        
        # Get trace context
        traces = await self.mcp.query_tempo(
            trace_id=alert.labels.get("trace_id"),
        )
        
        return Investigation(
            alert=alert,
            metrics=metrics,
            logs=logs,
            traces=traces,
            root_cause=self._analyze_root_cause(metrics, logs, traces),
        )
    
    async def determine_action(self, investigation: Investigation) -> RemediationAction:
        """Use LLM to determine best remediation action."""
        
        # Build context for LLM
        context = f"""
        Alert: {investigation.alert.summary}
        Root Cause Analysis: {investigation.root_cause}
        
        Available Remediation Scripts:
        {self._format_available_scripts()}
        
        What action should be taken?
        """
        
        # Query LLM (via Grafana LLM App or direct)
        response = await self.mcp.query_llm(context)
        
        return self._parse_action_response(response)
    
    async def execute_remediation(self, action: RemediationAction) -> RemediationResult:
        """Execute a remediation script."""
        
        script = self.remediation_scripts.get(action.script_id)
        if not script:
            raise ValueError(f"Unknown remediation script: {action.script_id}")
        
        # Execute in sandbox
        result = await script.execute(
            parameters=action.parameters,
            timeout_ms=action.timeout_ms or 30000,
        )
        
        # Log result to Grafana
        await self.mcp.create_annotation(
            dashboard_uid="awf-overview",
            time=datetime.now(),
            text=f"Remediation executed: {action.script_id}",
            tags=["remediation", action.script_id],
        )
        
        return result
    
    async def request_hitl_approval(self, action: RemediationAction) -> ApprovalRequest:
        """Request human approval for high-risk actions."""
        
        # Create approval request
        request = ApprovalRequest(
            id=str(uuid4()),
            action=action,
            investigation=action.investigation,
            requested_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # Store for later retrieval
        await self._store_approval_request(request)
        
        # Notify via configured channels (Slack, PagerDuty, etc.)
        await self._notify_approvers(request)
        
        return request
```

#### 4.3.3 Grafana MCP Client

Interface to Grafana's MCP Server (52 available tools):

```python
class GrafanaMCPClient:
    """Client for Grafana MCP Server."""
    
    SUPPORTED_TOOLS = [
        # Query tools
        "query_prometheus",
        "query_loki", 
        "query_tempo",
        "get_datasource_by_name",
        
        # Dashboard tools
        "list_dashboards",
        "get_dashboard_by_uid",
        "create_annotation",
        
        # Alert tools
        "list_alert_rules",
        "get_alert_rule",
        
        # Incident tools (via OnCall)
        "list_incidents",
        "create_incident",
        "resolve_incident",
    ]
    
    async def query_prometheus(self, query: str, start: datetime = None, end: datetime = None) -> PrometheusResult:
        """Execute PromQL query via MCP."""
        return await self._call_tool("query_prometheus", {
            "query": query,
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
        })
    
    async def query_loki(self, query: str, limit: int = 100) -> LokiResult:
        """Execute LogQL query via MCP."""
        return await self._call_tool("query_loki", {
            "query": query,
            "limit": limit,
        })
    
    async def query_tempo(self, trace_id: str) -> TempoResult:
        """Get trace by ID via MCP."""
        return await self._call_tool("query_tempo", {
            "traceId": trace_id,
        })
```

#### 4.3.4 Alert Webhook Receiver

```python
# awf/api/app.py (addition)
from awf.agents.watcher import WatcherAgent

watcher = WatcherAgent()

@app.post("/webhooks/grafana-alerts")
async def receive_grafana_alert(alert: GrafanaAlertWebhook):
    """Receive alerts from Grafana and route to Watcher Agent."""
    
    response = await watcher.handle_alert(alert)
    
    return {
        "status": "received",
        "action": response.action_taken,
        "hitl_required": response.hitl_required,
    }
```

#### 4.3.5 Remediation Scripts

Pre-approved remediation actions:

| Script ID | Action | HITL Required |
|-----------|--------|---------------|
| `restart_workflow` | Restart failed workflow with same input | No |
| `retry_step` | Retry specific failed step | No |
| `switch_fallback` | Switch to fallback agent | No |
| `increase_timeout` | Increase step timeout by 50% | No |
| `disable_agent` | Temporarily disable problematic agent | Yes |
| `scale_workers` | Increase concurrent worker count | Yes |
| `notify_oncall` | Page on-call engineer | No |

#### 4.3.6 HITL Approval Workflow

```python
@app.get("/approvals/pending")
async def list_pending_approvals():
    """List pending HITL approval requests."""
    return await watcher.get_pending_approvals()

@app.post("/approvals/{approval_id}/approve")
async def approve_action(approval_id: str, approver: str):
    """Approve a pending remediation action."""
    return await watcher.approve_action(approval_id, approver)

@app.post("/approvals/{approval_id}/reject")
async def reject_action(approval_id: str, approver: str, reason: str):
    """Reject a pending remediation action."""
    return await watcher.reject_action(approval_id, approver, reason)
```

### 4.4 Epic 4: Documentation (5 Story Points)

#### 4.4.1 Deployment Guide

`docs/deploying-with-grafana.md`:
- Prerequisites
- One-command deployment
- Accessing dashboards
- Configuring AWF to export telemetry

#### 4.4.2 Watcher Agent Configuration

`docs/watcher-agent-configuration.md`:
- Enabling the Watcher Agent
- Configuring MCP endpoint
- Adding custom remediation scripts
- Setting HITL requirements

#### 4.4.3 Dashboard Usage

`docs/grafana-dashboards.md`:
- AWF Overview dashboard walkthrough
- Agent Fleet Health dashboard walkthrough
- Creating custom dashboards
- Setting up alerts

---

## 5. Non-Functional Requirements

### 5.1 Performance

| Metric | Target |
|--------|--------|
| OTel export latency | <10ms per event |
| Grafana query latency | <500ms P99 |
| Watcher investigation time | <5s |
| Dashboard refresh rate | 5s minimum |

### 5.2 Reliability

| Metric | Target |
|--------|--------|
| Telemetry durability | 30 days retention |
| Watcher availability | 99.9% |
| Zero data loss | Buffered export with retry |

### 5.3 Scalability

| Metric | Target |
|--------|--------|
| Events per second | 10,000+ |
| Concurrent dashboards | 100+ |
| Metric cardinality | 100,000 series |

---

## 6. Security Considerations

### 6.1 Watcher Agent Permissions

The Watcher Agent operates with least privilege:
- Read-only access to Grafana data
- Execute only pre-approved remediation scripts
- High-risk actions require HITL approval
- All actions logged and auditable

### 6.2 Network Security

- OTLP uses TLS in production
- Grafana protected by authentication
- Webhook endpoints require secret token
- MCP server runs locally (no external exposure)

### 6.3 Data Privacy

- No PII in telemetry by default
- Workflow inputs can be redacted
- Log retention configurable
- Data export compliant with regulations

---

## 7. Out of Scope (v1.0)

1. **Multi-cluster Grafana** - Single cluster only
2. **Custom ML models** - Use Grafana's built-in LLM
3. **Complex runbooks** - Simple remediation scripts only
4. **Cross-organization** - Single tenant only
5. **Historical trend analysis** - Real-time focus

---

## 8. Implementation Phases

### Phase 1: Infrastructure (Week 1)
- Docker Compose for LGTM stack
- Alloy configuration
- Basic connectivity test

### Phase 2: Instrumentation (Week 2)
- OTel exporter implementation
- Trace spans for workflows
- Custom metrics

### Phase 3: Dashboards (Week 3)
- AWF Overview dashboard
- Agent Fleet Health dashboard
- SLO recording rules

### Phase 4: Watcher Agent (Week 4)
- Agent manifest and registration
- MCP client implementation
- Alert webhook handler
- HITL workflow

### Phase 5: Documentation & Polish (Week 5)
- Deployment guide
- Configuration docs
- Dashboard docs
- End-to-end testing

---

## 9. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to first dashboard | <15 min | User testing |
| Alert investigation time | <30s | Watcher logs |
| Autonomous remediation rate | 80%+ | Approval requests vs auto |
| Mean time to recovery | 50% reduction | Before/after |
| User satisfaction | 4.5/5 | Survey |

---

## 10. Appendix

### 10.1 Glossary

| Term | Definition |
|------|------------|
| **LGTM Stack** | Loki, Grafana, Tempo, Mimir - the full observability stack |
| **Alloy** | Grafana's OpenTelemetry collector |
| **MCP** | Model Context Protocol - Grafana's AI integration layer |
| **HITL** | Human-in-the-loop approval workflow |
| **Watcher Agent** | AWF's autonomous monitoring agent |

### 10.2 References

- [Grafana MCP Server](https://github.com/grafana/mcp-grafana)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
- [Grafana Alloy](https://grafana.com/docs/alloy/)
- [AWF Orchestration PRD](./orchestration-engine-prd.md)

### 10.3 Grafana OSS vs Cloud Feature Comparison

| Feature | OSS | Cloud | Our Approach |
|---------|-----|-------|--------------|
| LGTM Stack | Yes | Yes | Use OSS |
| Dashboards | Yes | Yes | Use OSS |
| Alerting | Yes | Yes | Use OSS |
| SLOs | No | Yes | Prometheus recording rules |
| AI Assistant | No | Yes | Watcher Agent + LLM App |
| Anomaly Detection | No | Yes | Future: custom ML |
| OnCall | Deprecated | Yes | Webhook to PagerDuty/Slack |
