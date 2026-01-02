# Orchestration Engine - Product Requirements Document

**Version:** 1.0.0
**Status:** Planning
**Author:** AWF Team
**Created:** January 2025
**Last Updated:** January 2025

---

## 1. Executive Summary

The Orchestration Engine is the core component that enables AWF to compose multi-agent workflows with automatic routing, retries, fallbacks, and observability. It is the "brain" that coordinates agents from different frameworks (LangGraph, CrewAI, AutoGen) into cohesive workflows.

### 1.1 Vision Statement

Enable developers to declaratively compose multi-agent workflows that are:
- **Reliable**: Automatic retries, fallbacks, and timeout handling
- **Observable**: Real-time streaming of workflow progress
- **Flexible**: Sequential, parallel, and conditional execution patterns
- **Framework-agnostic**: Mix agents from different frameworks seamlessly

### 1.2 Success Criteria

1. Execute multi-step workflows with 3+ agents from different frameworks
2. Handle failures gracefully with configurable retry/fallback strategies
3. Stream real-time progress events during workflow execution
4. Achieve <100ms overhead per step (excluding agent execution time)
5. Support both sync and async execution modes

---

## 2. Problem Statement

### 2.1 Current State

AWF currently has:
- Framework adapters (LangGraph, CrewAI, AutoGen) that can execute individual agents
- Registry for discovering agents by capability
- Trust scoring and policy enforcement for security
- REST API for agent management

**What's Missing:**
- No way to compose multiple agents into a workflow
- No automatic routing based on agent capabilities
- No retry/fallback logic when agents fail
- No streaming of workflow progress
- No state management across workflow steps

### 2.2 User Pain Points

| Pain Point | Impact |
|------------|--------|
| Manual agent orchestration | Hours of glue code per workflow |
| No cross-framework composition | Locked into single framework |
| No failure handling | Workflows fail completely on any error |
| No visibility into execution | Debugging is guesswork |
| State passing is manual | Complex data plumbing between steps |

### 2.3 Target Users

1. **Application Developers**: Building AI-powered applications with multiple agents
2. **ML Engineers**: Creating complex AI pipelines
3. **Platform Teams**: Deploying agent infrastructure for their organizations

---

## 3. Solution Overview

### 3.1 Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Workflow Definition         │
                    │   (JSON/YAML/Python API)            │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATION ENGINE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │
│  │  Workflow Parser │  │  Step Executor   │  │  State Manager       │   │
│  │  - Validate def  │  │  - Route to agent│  │  - Input mapping     │   │
│  │  - Build DAG     │  │  - Apply timeout │  │  - Output capture    │   │
│  │  - Check deps    │  │  - Handle errors │  │  - Context passing   │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘   │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │
│  │ Reliability Layer│  │  Event Emitter   │  │  Metrics Collector   │   │
│  │  - Retry logic   │  │  - Step started  │  │  - Execution time    │   │
│  │  - Fallback      │  │  - Step completed│  │  - Token usage       │   │
│  │  - Circuit break │  │  - Workflow done │  │  - Cost tracking     │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
            ┌─────────────┐┌─────────────┐┌─────────────┐
            │  LangGraph  ││   CrewAI    ││  AutoGen    │
            │   Adapter   ││   Adapter   ││   Adapter   │
            └─────────────┘└─────────────┘└─────────────┘
```

### 3.2 Key Components

| Component | Responsibility |
|-----------|----------------|
| **Workflow Parser** | Validates workflow definitions, builds execution DAG |
| **Step Executor** | Routes steps to correct adapters, applies timeouts |
| **State Manager** | Handles input mapping, output capture, context passing |
| **Reliability Layer** | Retry logic, fallback execution, circuit breaking |
| **Event Emitter** | Real-time streaming of workflow progress |
| **Metrics Collector** | Aggregates execution metrics across steps |

### 3.3 Execution Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Sequential** | Steps execute one after another | Most workflows |
| **Parallel** | Multiple steps execute concurrently | Independent research tasks |
| **Conditional** | Steps execute based on conditions | Branching logic |
| **Fan-out/Fan-in** | Parallel with aggregation | Map-reduce patterns |
| **Loop** | Repeat steps until condition met | Iterative refinement |

---

## 4. Detailed Requirements

### 4.1 Workflow Definition

```yaml
# Example workflow definition
id: research-and-write
name: "Research and Write Article"
version: "1.0.0"

input_schema:
  type: object
  properties:
    topic:
      type: string
    style:
      type: string
      enum: [academic, blog, technical]
  required: [topic]

steps:
  - id: research
    agent_id: research-agent
    input_map:
      query: $.input.topic
    timeout_ms: 60000
    retry:
      max_attempts: 3
      backoff_ms: 1000
    
  - id: outline
    agent_id: outline-agent
    input_map:
      research: $.steps.research.output.findings
      topic: $.input.topic
    depends_on: [research]
    
  - id: write
    agent_id: writer-agent
    input_map:
      outline: $.steps.outline.output.outline
      style: $.input.style
    depends_on: [outline]
    fallback:
      agent_id: backup-writer-agent
    
  - id: review
    agent_id: review-agent
    input_map:
      article: $.steps.write.output.article
    depends_on: [write]
    condition: $.input.style == "academic"

output_map:
  article: $.steps.write.output.article
  review: $.steps.review.output
```

### 4.2 Input Mapping (JSONPath)

Steps receive inputs via JSONPath expressions:

| Expression | Description |
|------------|-------------|
| `$.input.X` | Access workflow input field X |
| `$.steps.Y.output.Z` | Access field Z from step Y's output |
| `$.steps.Y.metrics.execution_time_ms` | Access step Y's execution time |
| `$.context.user_id` | Access workflow context |

### 4.3 Retry Policies

```yaml
retry:
  max_attempts: 3          # Maximum retry attempts
  backoff_ms: 1000         # Initial backoff delay
  backoff_multiplier: 2    # Exponential backoff multiplier
  max_backoff_ms: 30000    # Maximum backoff delay
  retry_on:                # Which errors to retry
    - TIMEOUT
    - EXTERNAL_SERVICE_ERROR
  no_retry_on:             # Which errors NOT to retry
    - INVALID_INPUT
    - PERMISSION_DENIED
```

### 4.4 Fallback Strategies

```yaml
fallback:
  # Option 1: Use a different agent
  agent_id: backup-agent
  
  # Option 2: Use a static value
  static_value:
    article: "Unable to generate article"
    
  # Option 3: Skip this step (for optional steps)
  skip: true
```

### 4.5 Conditions

Conditions use a simple expression language:

| Expression | Description |
|------------|-------------|
| `$.input.style == "academic"` | Equality check |
| `$.steps.research.output.confidence > 0.8` | Numeric comparison |
| `$.steps.research.status == "completed"` | Status check |
| `"academic" in $.input.tags` | Contains check |
| `$.input.priority >= 5 && $.input.urgent` | Boolean logic |

### 4.6 Events

| Event | Payload | When |
|-------|---------|------|
| `workflow.started` | `{execution_id, workflow_id, input}` | Workflow begins |
| `workflow.step.started` | `{execution_id, step_id, agent_id}` | Step begins |
| `workflow.step.completed` | `{execution_id, step_id, output, metrics}` | Step succeeds |
| `workflow.step.failed` | `{execution_id, step_id, error, attempt}` | Step fails |
| `workflow.step.retrying` | `{execution_id, step_id, attempt, delay_ms}` | Step retrying |
| `workflow.step.skipped` | `{execution_id, step_id, reason}` | Step skipped |
| `workflow.completed` | `{execution_id, output, metrics}` | Workflow succeeds |
| `workflow.failed` | `{execution_id, error, completed_steps}` | Workflow fails |

### 4.7 Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `total_execution_time_ms` | int | Total workflow duration |
| `step_execution_times` | dict | Per-step durations |
| `total_token_usage` | dict | Aggregated input/output tokens |
| `total_cost_usd` | float | Aggregated cost |
| `retry_count` | int | Total retries across all steps |
| `fallback_count` | int | Times fallback was used |

---

## 5. API Design

### 5.1 Python API

```python
from awf.orchestration import Orchestrator, Workflow, WorkflowStep

# Create orchestrator
orchestrator = Orchestrator(
    registry=registry,
    adapters={"langgraph": lg_adapter, "crewai": crew_adapter}
)

# Define workflow programmatically
workflow = Workflow(
    id="research-workflow",
    name="Research Workflow",
    steps=[
        WorkflowStep(
            id="research",
            agent_id="research-agent",
            input_map={"query": "$.input.topic"},
            timeout_ms=60000,
        ),
        WorkflowStep(
            id="write",
            agent_id="writer-agent",
            input_map={"research": "$.steps.research.output"},
            depends_on=["research"],
        ),
    ],
)

# Execute synchronously
result = await orchestrator.execute(
    workflow=workflow,
    input={"topic": "AI safety"},
)

# Execute with streaming
async for event in orchestrator.execute_streaming(
    workflow=workflow,
    input={"topic": "AI safety"},
):
    print(f"Event: {event.type} - {event.data}")
```

### 5.2 REST API

```http
# Register a workflow
POST /workflows
{
  "id": "research-workflow",
  "name": "Research Workflow",
  "steps": [...]
}

# Execute a workflow
POST /workflows/{workflow_id}/execute
{
  "input": {"topic": "AI safety"},
  "context": {"user_id": "user-123"}
}

# Response
{
  "execution_id": "exec-456",
  "status": "running"
}

# Get execution status
GET /executions/{execution_id}

# Stream execution events (SSE)
GET /executions/{execution_id}/events
Accept: text/event-stream
```

---

## 6. Non-Functional Requirements

### 6.1 Performance

| Metric | Target |
|--------|--------|
| Orchestration overhead per step | <100ms |
| Maximum workflow steps | 100 |
| Maximum parallel branches | 10 |
| Event streaming latency | <50ms |

### 6.2 Reliability

| Metric | Target |
|--------|--------|
| Workflow state durability | Persisted to disk |
| Crash recovery | Resume from last checkpoint |
| Maximum retry delay | 5 minutes |

### 6.3 Scalability

| Metric | Target |
|--------|--------|
| Concurrent workflow executions | 100+ (single instance) |
| Steps per second | 50+ |

---

## 7. Out of Scope (v1.0)

1. **Distributed execution** - Single-node only for v1.0
2. **Dynamic workflow modification** - Workflows are immutable during execution
3. **Human-in-the-loop** - Requires separate approval service
4. **Long-running workflows** - Maximum 1 hour execution time
5. **Workflow versioning/migration** - Manual handling required

---

## 8. Implementation Phases

### Phase 1: Core Engine (Week 1)
- Workflow definition types
- Sequential step execution
- Basic input mapping
- Sync execution

### Phase 2: Reliability (Week 2)
- Retry logic with exponential backoff
- Timeout handling
- Fallback execution
- Error categorization

### Phase 3: Advanced Patterns (Week 3)
- Parallel execution
- Conditional steps
- Fan-out/fan-in
- Step dependencies (DAG)

### Phase 4: Observability (Week 4)
- Event streaming
- Metrics collection
- Execution history
- REST API integration

---

## 9. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Developer adoption | 100+ workflows created | Analytics |
| Reliability | 99%+ workflow success rate | Monitoring |
| Performance | <100ms overhead per step | Profiling |
| Test coverage | 90%+ code coverage | CI/CD |

---

## 10. Appendix

### 10.1 Glossary

| Term | Definition |
|------|------------|
| **Workflow** | A composition of multiple agent invocations |
| **Step** | A single agent invocation within a workflow |
| **Execution** | A running instance of a workflow |
| **DAG** | Directed Acyclic Graph of step dependencies |
| **Fallback** | Alternative action when a step fails |

### 10.2 References

- [ASP Specification](../spec/asp-specification.md)
- [Core Types](../awf/core/types.py)
- [REST API](../awf/api/app.py)
