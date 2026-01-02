# ASP HTTP Transport Binding

**Version:** 1.0.0-draft  
**Status:** Draft  
**Companion to:** ASP Core Specification v1.0  

---

## 1. Overview

This document defines how ASP protocol operations map to HTTP/REST endpoints.
Implementations following this binding are interoperable with any ASP HTTP client.

---

## 2. Base URL and Versioning

```
Base URL: https://{host}/asp/v1
```

API version is encoded in the URL path. Breaking changes require new version.

---

## 3. Common Headers

### Request Headers (Required)

| Header | Value | Description |
|--------|-------|-------------|
| `Content-Type` | `application/json` | All request bodies are JSON |
| `Accept` | `application/json` | All responses are JSON |
| `X-ASP-Correlation-ID` | UUID | Distributed tracing ID |

### Request Headers (Optional)

| Header | Value | Description |
|--------|-------|-------------|
| `Authorization` | `Bearer {token}` | Authentication token |
| `X-ASP-Idempotency-Key` | UUID | For idempotent POST requests |
| `X-ASP-Timeout-Seconds` | Integer | Request timeout override |

### Response Headers

| Header | Value | Description |
|--------|-------|-------------|
| `X-ASP-Request-ID` | UUID | Unique request identifier |
| `X-ASP-Correlation-ID` | UUID | Echo of request correlation ID |
| `X-ASP-Processing-Time-Ms` | Integer | Server processing time |
| `X-ASP-Cost-USD` | Number | Cost incurred by this request |

---

## 4. Agent Registry Endpoints

### 4.1 Register Agent

```http
POST /asp/v1/agents
Content-Type: application/json

{
  "manifest": { ... }
}
```

**Response (201 Created):**
```json
{
  "agent_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "my-agent",
  "version": "1.0.0",
  "registered_at": "2025-01-15T10:30:00Z",
  "trust_score": 0.45,
  "sandbox_tier": "standard"
}
```

**Error Responses:**

| Status | Code | Description |
|--------|------|-------------|
| 400 | `ASP-REG-001` | Invalid manifest schema |
| 409 | `ASP-REG-002` | Agent ID already exists |
| 422 | `ASP-REG-003` | Capability schema validation failed |

### 4.2 Get Agent

```http
GET /asp/v1/agents/{agent_id}
```

**Response (200 OK):**
```json
{
  "manifest": { ... },
  "trust_score": 0.72,
  "trust_breakdown": {
    "publisher": 0.6,
    "audit": 0.5,
    "community": 0.85,
    "permissions": 0.9,
    "history": 0.75
  },
  "sandbox_tier": "standard",
  "stats": {
    "total_runs": 1523,
    "success_rate": 0.94,
    "avg_latency_ms": 2100,
    "last_run": "2025-01-15T09:45:00Z"
  }
}
```

### 4.3 Search Agents

```http
GET /asp/v1/agents?capability={capability_id}&min_trust={score}&limit={n}
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `capability` | string | Filter by capability ID |
| `q` | string | Natural language search |
| `publisher` | string | Filter by publisher |
| `min_trust` | number | Minimum trust score |
| `audit_status` | string | Required audit status |
| `limit` | integer | Max results (default: 20) |
| `offset` | integer | Pagination offset |

**Response (200 OK):**
```json
{
  "agents": [
    {
      "agent_id": "...",
      "name": "sql-optimizer",
      "version": "2.1.0",
      "trust_score": 0.72,
      "capability_match": 0.95,
      "estimated_latency_ms": 2000,
      "estimated_cost_usd": 0.02
    }
  ],
  "total": 47,
  "limit": 20,
  "offset": 0
}
```

### 4.4 Update Agent

```http
PUT /asp/v1/agents/{agent_id}
Content-Type: application/json

{
  "manifest": { ... }
}
```

**Response (200 OK):**
```json
{
  "agent_id": "...",
  "old_version": "1.0.0",
  "new_version": "1.1.0",
  "breaking_change": false,
  "updated_at": "2025-01-15T11:00:00Z"
}
```

### 4.5 Deprecate Agent Version

```http
POST /asp/v1/agents/{agent_id}/deprecate
Content-Type: application/json

{
  "version": "1.0.0",
  "sunset_date": "2025-06-01",
  "migration_guide": "Upgrade to 2.0.0. See https://...",
  "replacement": "my-agent@2.0.0"
}
```

### 4.6 Delete Agent

```http
DELETE /asp/v1/agents/{agent_id}?version={version}
```

---

## 5. Task Execution Endpoints

### 5.1 Submit Task (Async)

```http
POST /asp/v1/tasks
Content-Type: application/json
X-ASP-Idempotency-Key: {uuid}

{
  "agent_id": "550e8400-...",
  "capability_id": "optimize_query",
  "input": {
    "query": "SELECT * FROM users WHERE active = true"
  },
  "context": {
    "timeout_seconds": 60,
    "budget_usd": 0.50,
    "reliability_tier": "standard"
  }
}
```

**Response (202 Accepted):**
```json
{
  "task_id": "660e8400-...",
  "status": "queued",
  "created_at": "2025-01-15T10:30:00Z",
  "estimated_start": "2025-01-15T10:30:05Z",
  "estimated_duration_ms": 2000,
  "estimated_cost_usd": 0.02,
  "status_url": "/asp/v1/tasks/660e8400-...",
  "events_url": "/asp/v1/tasks/660e8400-.../events"
}
```

### 5.2 Submit Task (Synchronous)

```http
POST /asp/v1/tasks?sync=true&timeout=30
```

**Response (200 OK):** Returns full TaskResult directly.

### 5.3 Get Task Status

```http
GET /asp/v1/tasks/{task_id}
```

**Response (200 OK):**
```json
{
  "task_id": "660e8400-...",
  "status": "running",
  "agent_id": "550e8400-...",
  "capability_id": "optimize_query",
  "created_at": "2025-01-15T10:30:00Z",
  "started_at": "2025-01-15T10:30:02Z",
  "progress": {
    "percent": 60,
    "message": "Analyzing query structure..."
  },
  "metrics": {
    "elapsed_ms": 1200,
    "tokens_so_far": 523,
    "cost_so_far_usd": 0.008
  }
}
```

### 5.4 Get Task Result

```http
GET /asp/v1/tasks/{task_id}/result
```

### 5.5 Cancel Task

```http
POST /asp/v1/tasks/{task_id}/cancel
Content-Type: application/json

{
  "reason": "User requested cancellation"
}
```

### 5.6 Stream Task Events (SSE)

```http
GET /asp/v1/tasks/{task_id}/events
Accept: text/event-stream
```

**Response:**
```
event: task.started
data: {"task_id": "660e8400-...", "agent_id": "550e8400-..."}

event: task.progress
data: {"task_id": "660e8400-...", "percent": 30, "message": "Parsing query..."}

event: llm.call_completed
data: {"model": "gpt-4-turbo", "tokens_output": 189, "duration_ms": 1823}

event: task.completed
data: {"task_id": "660e8400-...", "status": "completed"}
```

---

## 6. Agent Status Endpoints

### 6.1 Get Agent Status

```http
GET /asp/v1/agents/{agent_id}/status
```

### 6.2 Subscribe to Agent Status (SSE)

```http
GET /asp/v1/agents/{agent_id}/status/stream
Accept: text/event-stream
```

---

## 7. Trust Endpoints

### 7.1 Get Trust Score

```http
GET /asp/v1/agents/{agent_id}/trust
```

**Response (200 OK):**
```json
{
  "agent_id": "550e8400-...",
  "overall_score": 0.72,
  "sandbox_tier": "standard",
  "components": {
    "publisher": {"score": 0.60, "details": "Publisher not verified"},
    "audit": {"score": 0.50, "details": "Community reviewed"},
    "community": {"score": 0.85, "details": "4.2/5.0 rating, 5432 runs"},
    "permissions": {"score": 0.90, "details": "No dangerous permissions"},
    "history": {"score": 0.75, "details": "94.2% success rate"}
  },
  "warnings": [
    "Publisher not verified",
    "No professional security audit"
  ],
  "recommendations": [
    {"agent_id": "...", "name": "sql-master", "trust_score": 0.91}
  ]
}
```

---

## 8. Workflow Endpoints

### 8.1 Register Workflow

```http
POST /asp/v1/workflows
Content-Type: application/json

{
  "name": "code-review-pipeline",
  "version": "1.0.0",
  "definition": { ... }
}
```

### 8.2 Execute Workflow

```http
POST /asp/v1/workflows/{workflow_id}/runs
Content-Type: application/json

{
  "inputs": {"code": "def hello(): ..."},
  "context": {"environment": "staging", "budget_usd": 1.00}
}
```

### 8.3 Get Workflow Run Status

```http
GET /asp/v1/workflows/{workflow_id}/runs/{run_id}
```

### 8.4 Stream Workflow Events

```http
GET /asp/v1/workflows/{workflow_id}/runs/{run_id}/events
Accept: text/event-stream
```

---

## 9. Memory Endpoints (ASP Extended)

### 9.1 Read Memory

```http
GET /asp/v1/memory/{scope}/{namespace}/{key}
```

### 9.2 Write Memory

```http
PUT /asp/v1/memory/{scope}/{namespace}/{key}
Content-Type: application/json

{
  "value": { ... },
  "ttl_seconds": 3600
}
```

### 9.3 List Memory Keys

```http
GET /asp/v1/memory/{scope}/{namespace}?prefix={prefix}
```

---

## 10. Error Responses

All errors follow consistent format:

```json
{
  "error": {
    "code": "ASP-TASK-001",
    "message": "Task timeout exceeded",
    "category": "TIMEOUT",
    "retriable": true,
    "details": {
      "timeout_seconds": 60,
      "elapsed_seconds": 61
    },
    "documentation_url": "https://awf.dev/docs/errors/ASP-TASK-001"
  },
  "request_id": "...",
  "correlation_id": "..."
}
```

### Standard Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| ASP-AUTH-001 | 401 | Missing authentication |
| ASP-AUTH-002 | 403 | Insufficient permissions |
| ASP-REG-001 | 400 | Invalid manifest |
| ASP-REG-002 | 409 | Agent already exists |
| ASP-TASK-001 | 408 | Task timeout |
| ASP-TASK-002 | 402 | Budget exceeded |
| ASP-TASK-003 | 404 | Task not found |
| ASP-TRUST-001 | 403 | Trust score too low |
| ASP-TRUST-002 | 403 | Policy violation |
| ASP-RATE-001 | 429 | Rate limit exceeded |
| ASP-SYS-001 | 500 | Internal server error |

---

## 11. Rate Limiting

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1673784000
```

---

## 12. Pagination

```http
GET /asp/v1/agents?limit=20&cursor=eyJsYXN0X2lkIjoiYWJjMTIzIn0=
```

```json
{
  "data": [...],
  "pagination": {
    "total": 147,
    "limit": 20,
    "has_more": true,
    "next_cursor": "eyJsYXN0X2lkIjoiZGVmNDU2In0="
  }
}
```

---

## 13. Webhooks

```http
POST /asp/v1/webhooks
Content-Type: application/json

{
  "url": "https://myapp.com/webhooks/asp",
  "events": ["task.completed", "task.failed"],
  "secret": "whsec_..."
}
```

Webhook signatures:
```
X-ASP-Webhook-Signature: sha256=abc123...
X-ASP-Webhook-Timestamp: 1673784000
```
