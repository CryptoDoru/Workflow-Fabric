# Agent State Protocol (ASP) Specification

**Version:** 1.0.0-draft  
**Status:** Draft for Review  
**License:** Apache 2.0  
**Authors:** AI Workflow Fabric Team  
**Last Updated:** January 2025  

---

## Abstract

The Agent State Protocol (ASP) defines a standard interface for AI agents to 
interoperate across different frameworks, runtimes, and orchestration systems. 
ASP enables agents built with LangGraph, CrewAI, AutoGen, or any other framework 
to be discovered, composed, and orchestrated through a common protocol.

ASP is designed to be:
- **Framework-agnostic**: Works with any agent implementation
- **Runtime-independent**: Agents can run locally, in containers, or serverless
- **Interoperable**: Agents from different frameworks can work together
- **Observable**: All agent behavior is traceable and auditable

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Terminology](#2-terminology)
3. [Protocol Overview](#3-protocol-overview)
4. [Agent Manifest](#4-agent-manifest)
5. [Task Interface](#5-task-interface)
6. [Status Reporting](#6-status-reporting)
7. [Event Streaming](#7-event-streaming)
8. [Memory Interface](#8-memory-interface)
9. [Inter-Agent Messaging](#9-inter-agent-messaging)
10. [Security Considerations](#10-security-considerations)
11. [Conformance](#11-conformance)
12. [Extension Mechanisms](#12-extension-mechanisms)

---

## 1. Introduction

### 1.1 Background

The AI agent ecosystem is fragmented across multiple frameworks, each with its 
own concepts, APIs, and execution models. This fragmentation creates barriers:

- Agents from different frameworks cannot easily work together
- Orchestration systems must understand each framework's internals
- Observability and debugging require framework-specific tooling
- Trust and security properties cannot be uniformly evaluated

### 1.2 Goals

ASP addresses these challenges by defining a common protocol that:

1. **Abstracts agent capabilities** into a uniform interface
2. **Standardizes task execution** across frameworks
3. **Provides consistent observability** through events
4. **Enables trust evaluation** through metadata

### 1.3 Non-Goals

ASP does NOT:

- Define how agents are implemented internally
- Replace framework-specific APIs for advanced use cases
- Mandate specific security or isolation mechanisms
- Require specific transport protocols (HTTP, gRPC, etc.)

### 1.4 Document Conventions

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD",
"SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be
interpreted as described in RFC 2119.

---

## 2. Terminology

**Agent**: A software component that accepts structured tasks and produces 
structured results, typically using one or more language models.

**Capability**: A specific function an agent can perform, with defined input 
and output schemas.

**Task**: A request for an agent to perform a capability with specific inputs.

**Result**: The output produced by an agent after completing a task.

**Workflow**: A composition of multiple tasks across one or more agents.

**Adapter**: A component that translates between ASP and a specific agent 
framework.

**Event**: A record of something that happened during agent execution.

**Manifest**: A declaration of an agent's identity, capabilities, and 
requirements.

---

## 3. Protocol Overview

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ASP Protocol Layers                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DISCOVERY LAYER                                                │
│  ├── Agent Manifest                                             │
│  └── Registry Interface                                         │
│                                                                 │
│  EXECUTION LAYER                                                │
│  ├── Task Interface                                             │
│  ├── Result Interface                                           │
│  └── Status Reporting                                           │
│                                                                 │
│  OBSERVABILITY LAYER                                            │
│  ├── Event Streaming                                            │
│  └── Tracing                                                    │
│                                                                 │
│  COORDINATION LAYER (Optional)                                  │
│  ├── Memory Interface                                           │
│  └── Inter-Agent Messaging                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Protocol Tiers

ASP defines two conformance tiers:

**ASP Core** (Required for conformance):
- Agent Manifest
- Task Interface
- Status Reporting
- Event Streaming

**ASP Extended** (Optional, for advanced coordination):
- Memory Interface
- Inter-Agent Messaging

Implementations MUST support ASP Core. Implementations MAY support ASP Extended.

### 3.3 Transport Independence

ASP is transport-agnostic. This specification defines data structures and 
semantics. Bindings for specific transports (HTTP/REST, gRPC, WebSocket) are 
defined in separate documents.

---

## 4. Agent Manifest

The Agent Manifest declares an agent's identity, capabilities, and requirements.

### 4.1 Manifest Structure

```yaml
# Required fields
asp_version: "1.0"                    # ASP version this manifest conforms to
id: uuid                              # Globally unique identifier
name: string                          # Human-readable name
version: semver                       # Semantic version (e.g., "1.2.3")

# Capabilities (at least one required)
capabilities:
  - capability_id: string             # Unique within this agent
    description: string               # Human-readable description
    input_schema: JSONSchema          # JSON Schema for inputs
    output_schema: JSONSchema         # JSON Schema for outputs
    
    # Optional capability metadata
    idempotent: boolean               # Safe to retry? (default: false)
    estimated_latency_ms: integer     # Typical execution time
    estimated_cost_usd: number        # Typical cost per invocation
    confidence: number                # Self-assessed reliability (0.0-1.0)

# Requirements
requirements:
  runtime: enum                       # python | node | container | wasm | remote
  min_memory_mb: integer              # Minimum memory required
  timeout_seconds: integer            # Maximum execution time
  
  # Optional requirements
  gpu_required: boolean               # Needs GPU? (default: false)
  network_required: boolean           # Needs network? (default: true)
  secrets: array[string]              # Required secret names

# Trust metadata
trust:
  publisher: string                   # Publisher identifier
  signature: string                   # Cryptographic signature (optional)
  audit_status: enum                  # unaudited | community | verified | certified
  permissions: array[Permission]      # Requested permissions

# Optional metadata
metadata:
  description: string                 # Agent description
  homepage: uri                       # Documentation URL
  repository: uri                     # Source code URL
  license: string                     # SPDX license identifier
  tags: array[string]                 # Categorization tags
```

### 4.2 Field Definitions

#### 4.2.1 asp_version

The ASP specification version this manifest conforms to. MUST be a valid ASP 
version string (e.g., "1.0").

Implementations MUST reject manifests with unsupported asp_version values.

#### 4.2.2 id

A globally unique identifier for this agent. MUST be a valid UUID (RFC 4122).

The id MUST remain constant across versions of the same agent. Different agents 
MUST have different ids.

#### 4.2.3 name

A human-readable name for the agent. MUST be a non-empty string. SHOULD be 
descriptive and unique within a namespace.

Format: `[namespace/]agent-name`

Examples:
- `code-analyzer`
- `awf-official/sql-optimizer`
- `acme-corp/internal-reviewer`

#### 4.2.4 version

The agent version. MUST follow Semantic Versioning 2.0.0 (semver.org).

- MAJOR: Breaking changes to capability schemas
- MINOR: New capabilities or backward-compatible changes
- PATCH: Bug fixes without schema changes

#### 4.2.5 capabilities

An array of capabilities this agent provides. MUST contain at least one 
capability.

Each capability MUST have:
- `capability_id`: Unique identifier within this agent
- `input_schema`: Valid JSON Schema defining accepted inputs
- `output_schema`: Valid JSON Schema defining produced outputs

#### 4.2.6 requirements.runtime

The execution environment required:

| Value | Description |
|-------|-------------|
| `python` | Python interpreter (version in metadata) |
| `node` | Node.js runtime |
| `container` | OCI-compatible container |
| `wasm` | WebAssembly module |
| `remote` | Agent runs on remote server (URL in metadata) |

#### 4.2.7 trust.permissions

Requested permissions from the Permission enum:

```
Permission:
  # Network
  NETWORK_NONE              # No network access
  NETWORK_LLM_PROVIDERS     # Access to known LLM API endpoints only
  NETWORK_ALLOW_LIST        # Access to specified domains
  NETWORK_UNRESTRICTED      # Unrestricted network access
  
  # Filesystem
  FS_NONE                   # No filesystem access
  FS_READ_WORKSPACE         # Read from workflow workspace
  FS_WRITE_WORKSPACE        # Write to workflow workspace
  FS_READ_SYSTEM            # Read system files
  
  # Secrets
  SECRETS_NONE              # No secret access
  SECRETS_OWN               # Access to agent's assigned secrets
  SECRETS_WORKFLOW          # Access to workflow-level secrets
  
  # Execution
  EXEC_CODE                 # Execute generated code
  EXEC_SHELL                # Execute shell commands
  EXEC_SUBPROCESS           # Spawn subprocesses
  
  # Inter-Agent
  AGENT_RECEIVE             # Receive messages from other agents
  AGENT_SEND                # Send messages to other agents
  AGENT_DELEGATE            # Delegate tasks to other agents
```

### 4.3 Manifest Validation

Implementations MUST validate manifests before accepting them:

1. All required fields MUST be present
2. `asp_version` MUST be supported
3. `id` MUST be valid UUID
4. `version` MUST be valid semver
5. `capabilities` MUST be non-empty
6. All `input_schema` and `output_schema` MUST be valid JSON Schema
7. All enum values MUST be recognized

Invalid manifests MUST be rejected with descriptive error messages.

### 4.4 Manifest Examples

**Minimal Manifest:**

```yaml
asp_version: "1.0"
id: "550e8400-e29b-41d4-a716-446655440000"
name: "simple-agent"
version: "1.0.0"

capabilities:
  - capability_id: "greet"
    description: "Returns a greeting"
    input_schema:
      type: object
      properties:
        name:
          type: string
      required: [name]
    output_schema:
      type: object
      properties:
        greeting:
          type: string

requirements:
  runtime: python
  min_memory_mb: 128
  timeout_seconds: 30

trust:
  publisher: "anonymous"
  audit_status: "unaudited"
  permissions: [NETWORK_NONE]
```

**Full Manifest:**

```yaml
asp_version: "1.0"
id: "550e8400-e29b-41d4-a716-446655440001"
name: "awf-official/sql-optimizer"
version: "2.1.0"

capabilities:
  - capability_id: "optimize_query"
    description: "Analyzes and optimizes SQL queries for performance"
    input_schema:
      type: object
      properties:
        query:
          type: string
          description: "SQL query to optimize"
        dialect:
          type: string
          enum: [postgresql, mysql, sqlite]
          default: postgresql
        context:
          type: object
          properties:
            table_schemas:
              type: array
              items:
                type: object
            indexes:
              type: array
              items:
                type: string
      required: [query]
    output_schema:
      type: object
      properties:
        optimized_query:
          type: string
        explanation:
          type: string
        improvements:
          type: array
          items:
            type: object
            properties:
              type:
                type: string
              description:
                type: string
              impact:
                type: string
                enum: [low, medium, high]
        estimated_speedup:
          type: number
      required: [optimized_query, explanation]
    idempotent: true
    estimated_latency_ms: 2000
    estimated_cost_usd: 0.02
    confidence: 0.85

  - capability_id: "explain_query"
    description: "Explains what a SQL query does in plain English"
    input_schema:
      type: object
      properties:
        query:
          type: string
      required: [query]
    output_schema:
      type: object
      properties:
        explanation:
          type: string
        complexity:
          type: string
          enum: [simple, moderate, complex]
    idempotent: true
    estimated_latency_ms: 1500
    estimated_cost_usd: 0.01
    confidence: 0.90

requirements:
  runtime: container
  min_memory_mb: 512
  timeout_seconds: 60
  gpu_required: false
  network_required: true
  secrets:
    - OPENAI_API_KEY

trust:
  publisher: "awf-official"
  signature: "sha256:abc123..."
  audit_status: "verified"
  permissions:
    - NETWORK_LLM_PROVIDERS
    - SECRETS_OWN

metadata:
  description: |
    Professional SQL query optimizer powered by GPT-4.
    Supports PostgreSQL, MySQL, and SQLite dialects.
    Provides detailed explanations and improvement suggestions.
  homepage: "https://awf.dev/agents/sql-optimizer"
  repository: "https://github.com/awf/agents/sql-optimizer"
  license: "Apache-2.0"
  tags:
    - sql
    - database
    - optimization
    - performance
```

---

## 5. Task Interface

The Task Interface defines how agents receive work and return results.

### 5.1 Task Structure

```yaml
Task:
  # Identity
  id: uuid                          # Unique task identifier
  capability_id: string             # Which capability to invoke
  
  # Input
  input: object                     # Matches capability's input_schema
  
  # Context
  context:
    workflow_id: uuid               # Parent workflow (if any)
    parent_task_id: uuid | null     # Parent task (for subtasks)
    correlation_id: uuid            # For distributed tracing
    
    # Execution constraints
    deadline: datetime | null       # Hard deadline (ISO 8601)
    timeout_seconds: integer        # Task timeout
    budget_usd: number | null       # Maximum cost
    
    # Reliability
    reliability_tier: enum          # best_effort | standard | high | critical
    retry_policy:
      max_attempts: integer
      backoff_base_ms: integer
      backoff_max_ms: integer
    
    # Human interaction
    allow_human_escalation: boolean
    human_timeout_seconds: integer
  
  # Memory (ASP Extended)
  memory_refs: array[MemoryRef]     # References to shared memory
  
  # Conversation (ASP Extended)
  messages: array[Message]          # Prior conversation context
```

### 5.2 TaskResult Structure

```yaml
TaskResult:
  # Identity
  task_id: uuid                     # Matches Task.id
  
  # Outcome
  status: enum                      # completed | failed | partial | needs_human
  output: object | null             # Matches capability's output_schema (if completed)
  error: Error | null               # Error details (if failed)
  
  # Metrics
  metrics:
    started_at: datetime            # When execution began
    completed_at: datetime          # When execution ended
    duration_ms: integer            # Total duration
    tokens_input: integer           # LLM input tokens
    tokens_output: integer          # LLM output tokens
    cost_usd: number                # Actual cost
    retries: integer                # Number of retry attempts
  
  # Provenance
  provenance:
    agent_id: uuid                  # Agent that executed
    agent_version: semver           # Agent version used
    model_calls: array[ModelCall]   # LLM calls made
    reasoning_trace: array[Step]    # Optional reasoning steps
    sources: array[Source]          # Referenced sources
  
  # Confidence
  confidence: number                # 0.0-1.0, agent's self-assessment
  warnings: array[string]           # Non-fatal issues
```

### 5.3 Status Values

| Status | Description | Output Present | Error Present |
|--------|-------------|----------------|---------------|
| `completed` | Task finished successfully | Yes | No |
| `failed` | Task failed after all retries | Optional | Yes |
| `partial` | Task partially completed | Partial | Optional |
| `needs_human` | Awaiting human input | Optional | No |

### 5.4 Error Structure

```yaml
Error:
  code: string                      # Machine-readable error code
  message: string                   # Human-readable description
  category: enum                    # See below
  retriable: boolean                # Safe to retry?
  details: object                   # Additional context

ErrorCategory:
  INVALID_INPUT                     # Input failed schema validation
  CAPABILITY_NOT_FOUND              # Requested capability doesn't exist
  PERMISSION_DENIED                 # Insufficient permissions
  RESOURCE_EXHAUSTED                # Budget, quota, or time exceeded
  EXTERNAL_SERVICE_ERROR            # LLM or tool failure
  INTERNAL_ERROR                    # Agent implementation error
  HUMAN_INTERVENTION_REQUIRED       # Needs human decision
  CANCELLED                         # Task was cancelled
```

### 5.5 Reliability Tiers

| Tier | Max Retries | Backoff | Fallback | Human Escalation |
|------|-------------|---------|----------|------------------|
| `best_effort` | 0 | N/A | No | No |
| `standard` | 3 | Exponential | If configured | No |
| `high` | 5 | Exponential | Required | If configured |
| `critical` | 5 | Exponential | Required | Required |

### 5.6 Task Lifecycle

```
                    ┌─────────────┐
                    │   created   │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
          ┌────────│   queued    │
          │        └──────┬──────┘
          │               │
          │               ▼
          │        ┌─────────────┐
          │   ┌───│   running   │───┐
          │   │   └──────┬──────┘   │
          │   │          │          │
          │   ▼          ▼          ▼
          │ retry    completed    failed
          │   │          │          │
          │   │          ▼          │
          │   │   ┌─────────────┐   │
          │   └──▶│   running   │   │
          │       └─────────────┘   │
          │                         │
          ▼                         ▼
    ┌─────────────┐          ┌─────────────┐
    │ needs_human │          │   failed    │
    └──────┬──────┘          └─────────────┘
           │
           ▼
    ┌─────────────┐
    │  completed  │
    └─────────────┘
```

---

## 6. Status Reporting

Agents MUST report their status to enable orchestration and monitoring.

### 6.1 AgentStatus Structure

```yaml
AgentStatus:
  agent_id: uuid
  timestamp: datetime               # ISO 8601
  
  state: enum                       # See below
  current_task: uuid | null         # Task being processed
  queue_depth: integer              # Tasks waiting
  
  health:
    healthy: boolean
    last_success: datetime
    last_failure: datetime | null
    success_rate_1h: number         # 0.0-1.0
    success_rate_24h: number        # 0.0-1.0
    avg_latency_ms: integer
    
  resources:
    memory_used_mb: integer
    memory_limit_mb: integer
    cpu_percent: number
    active_connections: integer

AgentState:
  idle                              # Ready to accept tasks
  busy                              # Processing a task
  waiting_input                     # Blocked on human/agent input
  waiting_resource                  # Blocked on external resource
  draining                          # Finishing current task, accepting no new
  error                             # Recoverable error state
  terminated                        # Shut down
```

### 6.2 Status Reporting Requirements

1. Agents MUST report status when state changes
2. Agents MUST report status at least every 30 seconds when `busy`
3. Agents SHOULD report status every 60 seconds when `idle`
4. Status reports MUST include accurate timestamps

### 6.3 Health Calculation

Implementations SHOULD calculate health metrics as follows:

```python
success_rate = successful_tasks / total_tasks  # Over time window
healthy = (
    success_rate >= 0.5 and
    last_failure is None or (now - last_failure) > 5 minutes
)
```

---

## 7. Event Streaming

Agents MUST emit events for observability and auditing.

### 7.1 Event Structure

```yaml
Event:
  event_id: uuid                    # Unique event identifier
  timestamp: datetime               # When event occurred (ISO 8601)
  agent_id: uuid                    # Agent that emitted
  
  event_type: string                # See event types below
  payload: object                   # Event-specific data
  
  # Tracing
  causation_id: uuid                # Event/task that caused this
  correlation_id: uuid              # Workflow trace ID
  
  # Optional
  severity: enum                    # debug | info | warning | error
  tags: object                      # Arbitrary key-value metadata
```

### 7.2 Standard Event Types

**Lifecycle Events:**

| Event Type | Payload | Description |
|------------|---------|-------------|
| `agent.registered` | `{manifest}` | Agent registered with registry |
| `agent.updated` | `{old_version, new_version}` | Agent version updated |
| `agent.deregistered` | `{reason}` | Agent removed from registry |
| `agent.status_changed` | `{old_state, new_state}` | State transition |

**Task Events:**

| Event Type | Payload | Description |
|------------|---------|-------------|
| `task.received` | `{task_id, capability_id}` | Task received by agent |
| `task.started` | `{task_id}` | Task execution began |
| `task.progress` | `{task_id, progress, message}` | Progress update |
| `task.completed` | `{task_id, result_summary}` | Task finished successfully |
| `task.failed` | `{task_id, error}` | Task failed |
| `task.retrying` | `{task_id, attempt, reason}` | Retry attempt |
| `task.delegated` | `{task_id, target_agent}` | Task delegated |

**LLM Events:**

| Event Type | Payload | Description |
|------------|---------|-------------|
| `llm.call_started` | `{model, prompt_tokens}` | LLM call initiated |
| `llm.call_completed` | `{model, response_tokens, latency_ms}` | LLM call finished |
| `llm.call_failed` | `{model, error}` | LLM call failed |
| `llm.token_usage` | `{input, output, cost_usd}` | Token accounting |

**Security Events:**

| Event Type | Payload | Description |
|------------|---------|-------------|
| `security.permission_denied` | `{permission, context}` | Access denied |
| `security.sandbox_violation` | `{violation_type, details}` | Sandbox constraint hit |
| `security.secret_accessed` | `{secret_name}` | Secret was read |

### 7.3 Event Ordering

Events MUST be emitted in causal order within an agent. Cross-agent ordering 
is NOT guaranteed; use `correlation_id` to reconstruct workflow order.

### 7.4 Event Retention

This specification does not mandate event retention. Implementations SHOULD:
- Retain events for at least 24 hours for debugging
- Provide configurable retention policies
- Support event export for long-term storage

---

## 8. Memory Interface (ASP Extended)

The Memory Interface enables agents to share state within a workflow.

### 8.1 Memory Model

```
┌─────────────────────────────────────────────────────────────────┐
│                        MEMORY SCOPES                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TASK SCOPE (working memory)                                    │
│  ├── Lifetime: Single task execution                            │
│  ├── Visibility: Current agent only                             │
│  └── Use: Intermediate computation state                        │
│                                                                 │
│  WORKFLOW SCOPE (shared memory)                                 │
│  ├── Lifetime: Workflow execution                               │
│  ├── Visibility: All agents in workflow                         │
│  └── Use: Cross-agent data sharing                              │
│                                                                 │
│  AGENT SCOPE (episodic memory)                                  │
│  ├── Lifetime: Persistent across workflows                      │
│  ├── Visibility: Single agent                                   │
│  └── Use: Learning, adaptation, context                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Memory Entry Structure

```yaml
MemoryEntry:
  key: string                       # Unique within scope
  value: any                        # JSON-serializable value
  metadata:
    created_at: datetime
    updated_at: datetime
    created_by: uuid                # Agent that created
    ttl_seconds: integer | null     # Time-to-live
    tags: object                    # Arbitrary metadata
```

### 8.3 Memory Operations

```yaml
MemoryOperations:
  # Read
  get(scope, key) -> MemoryEntry | null
  list(scope, prefix?) -> array[MemoryEntry]
  
  # Write
  set(scope, key, value, ttl?) -> void
  delete(scope, key) -> void
  
  # Atomic
  compare_and_set(scope, key, expected, new) -> boolean
```

### 8.4 Memory Consistency

Workflow-scoped memory provides **read-your-writes** consistency within an 
agent and **eventual consistency** across agents. Implementations MAY provide 
stronger guarantees.

---

## 9. Inter-Agent Messaging (ASP Extended)

Enables direct communication between agents.

### 9.1 Message Structure

```yaml
Message:
  id: uuid
  timestamp: datetime
  
  from_agent: uuid
  to_agent: uuid | "broadcast"
  
  type: enum                        # See below
  content: object                   # Type-specific payload
  
  reply_to: uuid | null             # For responses
  correlation_id: uuid              # Workflow trace

MessageType:
  request                           # Asking for something
  response                          # Answering a request
  inform                            # Sharing information
  delegate                          # Handing off a task
  query                             # Asking about state
  event                             # Notification
```

### 9.2 Messaging Patterns

**Request/Response:**
```
Agent A                     Agent B
   │                           │
   │──── request (id=1) ──────▶│
   │                           │
   │◀─── response (reply=1) ───│
   │                           │
```

**Delegation:**
```
Agent A                     Agent B
   │                           │
   │──── delegate(task) ──────▶│
   │                           │
   │◀─── event(completed) ─────│
   │                           │
```

**Broadcast:**
```
Agent A                     Agents B, C, D
   │                           │
   │──── inform(to="broadcast")──▶│
   │                           │
```

### 9.3 Delivery Guarantees

Messages are delivered **at-least-once**. Agents MUST handle duplicate 
messages idempotently when possible.

---

## 10. Security Considerations

### 10.1 Threat Model

ASP assumes:
- Agents may be malicious or compromised
- Networks are untrusted
- LLM outputs are unpredictable

### 10.2 Required Mitigations

Implementations MUST:

1. **Validate all inputs** against declared schemas before processing
2. **Enforce permissions** declared in manifests
3. **Sandbox agent execution** to prevent unauthorized access
4. **Rate limit** API calls to prevent denial of service
5. **Audit log** all security-relevant events

### 10.3 Recommended Mitigations

Implementations SHOULD:

1. **Sign manifests** cryptographically
2. **Verify publisher identity** through trusted registries
3. **Scan agent code** for known vulnerabilities
4. **Monitor agent behavior** for anomalies
5. **Encrypt data** in transit and at rest

### 10.4 Permission Enforcement

The minimum permission principle applies. Agents MUST NOT be granted 
permissions beyond what they request in their manifest.

Orchestrators SHOULD deny agents with excessive permission requests unless 
explicitly approved by users or policy.

---

## 11. Conformance

### 11.1 Conformance Levels

**ASP Core Conformant:**
- MUST implement Agent Manifest (Section 4)
- MUST implement Task Interface (Section 5)
- MUST implement Status Reporting (Section 6)
- MUST implement Event Streaming (Section 7)

**ASP Extended Conformant:**
- MUST be ASP Core Conformant
- MUST implement Memory Interface (Section 8)
- MUST implement Inter-Agent Messaging (Section 9)

### 11.2 Conformance Testing

Conformance test suites will be published separately. Implementations claiming 
conformance SHOULD pass the relevant test suite.

### 11.3 Versioning and Compatibility

ASP follows semantic versioning:
- MAJOR: Breaking changes to required fields
- MINOR: New optional fields or features
- PATCH: Clarifications and bug fixes

Implementations MUST support the current major version. Implementations SHOULD 
support the previous major version for at least 12 months after a new major 
version release.

---

## 12. Extension Mechanisms

### 12.1 Custom Metadata

Manifests, Tasks, and Events MAY include additional fields in their `metadata` 
or `tags` objects. Custom fields MUST NOT conflict with standard fields.

Custom field names SHOULD be prefixed with a namespace:

```yaml
metadata:
  x-acme/priority: high
  x-acme/department: engineering
```

### 12.2 Custom Event Types

Custom event types MUST be prefixed with `x-`:

```yaml
event_type: "x-acme/custom-event"
```

### 12.3 Extension Specifications

Formal extensions to ASP (e.g., bindings for specific transports) MUST be 
published as separate specification documents referencing this base 
specification.

---

## Appendix A: JSON Schema Definitions

Full JSON schemas for all structures are published in the `/spec/schemas/` directory.

## Appendix B: Transport Bindings

See `asp-http-binding.md` for HTTP/REST binding.

## Appendix C: Reference Implementations

- LangGraph Adapter: `/awf/adapters/langgraph/`

---

## Changelog

### v1.0.0-draft (January 2025)
- Initial draft specification
