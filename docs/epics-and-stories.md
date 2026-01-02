# Orchestration Engine - Epics & User Stories

**Version:** 1.0.0
**Status:** Planning
**Created:** January 2025

---

## Epic Overview

| Epic | Priority | Effort | Description |
|------|----------|--------|-------------|
| E1 | P0 | L | Core Workflow Engine - Basic workflow execution |
| E2 | P0 | M | Step Execution & Routing - Execute steps via adapters |
| E3 | P0 | L | State Management - Input mapping and context |
| E4 | P1 | L | Reliability - Retries, fallbacks, timeouts |
| E5 | P1 | M | Advanced Patterns - Parallel, conditional, DAG |
| E6 | P2 | M | Events & Observability - Streaming and metrics |
| E7 | P2 | S | REST API Integration - HTTP endpoints |

**Effort Key:** S = 1-2 days, M = 3-5 days, L = 5-10 days

---

# Epic 1: Core Workflow Engine

## Overview
Build the foundational workflow execution engine that can parse workflow definitions, validate them, and execute steps sequentially.

## User Stories

### E1-S1: Define Workflow Data Types
**As a** developer
**I want to** define workflows using Python dataclasses
**So that** I have type-safe workflow definitions with validation

**Acceptance Criteria:**
- [ ] `WorkflowDefinition` dataclass with id, name, version, steps, input_schema, output_map
- [ ] `StepDefinition` dataclass with id, agent_id, input_map, timeout_ms, depends_on
- [ ] `RetryPolicy` dataclass with max_attempts, backoff_ms, backoff_multiplier
- [ ] `FallbackPolicy` dataclass with agent_id, static_value, skip
- [ ] All types have `to_dict()` and `from_dict()` methods
- [ ] Full type hints on all fields
- [ ] JSON Schema generation for workflow definitions

**Technical Notes:**
- Place in `awf/orchestration/types.py`
- Extend existing `Workflow` and `WorkflowStep` in `awf/core/types.py`
- Use `@dataclass` with `field(default_factory=...)` for mutable defaults

---

### E1-S2: Workflow Validation
**As a** developer
**I want** workflows to be validated before execution
**So that** I catch configuration errors early

**Acceptance Criteria:**
- [ ] Validate all required fields are present
- [ ] Validate step IDs are unique within workflow
- [ ] Validate agent_id references exist in registry (optional, at runtime)
- [ ] Validate input_map JSONPath expressions are syntactically valid
- [ ] Validate depends_on references existing step IDs
- [ ] Validate no circular dependencies in step graph
- [ ] Return clear, actionable error messages

**Technical Notes:**
- Create `WorkflowValidator` class
- Use topological sort to detect cycles
- Validate JSONPath syntax with simple regex (full validation at runtime)

---

### E1-S3: Workflow Execution Context
**As a** developer
**I want** each workflow execution to have isolated context
**So that** concurrent executions don't interfere

**Acceptance Criteria:**
- [ ] `ExecutionContext` class with execution_id, workflow_id, input, start_time
- [ ] Store step results in context as they complete
- [ ] Store workflow state (pending, running, completed, failed)
- [ ] Thread-safe access for concurrent step execution
- [ ] Context is serializable for persistence

**Technical Notes:**
- Use `asyncio.Lock` for thread safety
- Generate UUID for execution_id
- Store in `awf/orchestration/context.py`

---

### E1-S4: Basic Orchestrator Class
**As a** developer
**I want** an Orchestrator class to manage workflow execution
**So that** I have a clean API to run workflows

**Acceptance Criteria:**
- [ ] `Orchestrator` class with `execute(workflow, input)` method
- [ ] Accept registry and adapter configuration in constructor
- [ ] Return `WorkflowResult` with output, metrics, status
- [ ] Raise `WorkflowValidationError` for invalid workflows
- [ ] Raise `WorkflowExecutionError` for runtime failures
- [ ] Support async execution with `await`

**Technical Notes:**
- Create `awf/orchestration/orchestrator.py`
- Use dependency injection for registry and adapters
- Make execute() an async method

---

### E1-S5: Sequential Step Execution
**As a** developer
**I want** steps to execute in order based on dependencies
**So that** data flows correctly through the workflow

**Acceptance Criteria:**
- [ ] Steps with no dependencies execute first
- [ ] Steps execute only after all `depends_on` steps complete
- [ ] Step output is available to subsequent steps
- [ ] Workflow fails fast if any step fails (no retries yet)
- [ ] All steps receive correct input based on input_map

**Technical Notes:**
- Use topological sort to determine execution order
- For sequential (no parallel yet), execute in sorted order
- Pass execution context through each step

---

## Definition of Done (Epic 1)
- [ ] All stories completed and tested
- [ ] Unit tests with >90% coverage
- [ ] Integration test: 3-step sequential workflow
- [ ] Documentation in docstrings
- [ ] No type errors (mypy clean)

---

# Epic 2: Step Execution & Routing

## Overview
Execute individual steps by routing to the correct framework adapter based on the agent's framework.

## User Stories

### E2-S1: Step Executor Class
**As a** developer
**I want** a dedicated class to execute individual steps
**So that** step execution logic is encapsulated

**Acceptance Criteria:**
- [ ] `StepExecutor` class with `execute(step, context)` method
- [ ] Look up agent manifest from registry
- [ ] Determine correct adapter based on agent's framework
- [ ] Create Task with mapped inputs
- [ ] Execute via adapter and return TaskResult
- [ ] Convert TaskResult to step result format

**Technical Notes:**
- Create `awf/orchestration/executor.py`
- Accept dict of adapters keyed by framework name
- Handle missing agent gracefully

---

### E2-S2: Input Mapping (JSONPath)
**As a** developer
**I want** to map workflow/step data to agent inputs using JSONPath
**So that** I can flexibly wire data between steps

**Acceptance Criteria:**
- [ ] Support `$.input.X` to access workflow input
- [ ] Support `$.steps.Y.output.Z` to access step output
- [ ] Support `$.steps.Y.status` to access step status
- [ ] Support `$.context.X` to access execution context
- [ ] Support nested paths like `$.steps.Y.output.items[0].name`
- [ ] Return None for missing paths (configurable: error vs None)

**Technical Notes:**
- Use `jsonpath-ng` library or implement simple subset
- Create `InputMapper` class in `awf/orchestration/mapping.py`
- Cache compiled JSONPath expressions

---

### E2-S3: Timeout Handling
**As a** developer
**I want** steps to respect timeout configuration
**So that** hung agents don't block the workflow

**Acceptance Criteria:**
- [ ] Step timeout from step definition takes precedence
- [ ] Fall back to workflow default timeout if step has none
- [ ] Fall back to global default (60s) if neither specified
- [ ] Cancel agent execution on timeout
- [ ] Return TaskStatus.TIMEOUT on timeout
- [ ] Emit timeout event

**Technical Notes:**
- Use `asyncio.wait_for()` for timeout
- Adapters must support cancellation
- Create custom `StepTimeoutError`

---

### E2-S4: Agent Resolution
**As a** developer
**I want** steps to resolve agents by ID from the registry
**So that** I can reference agents declaratively

**Acceptance Criteria:**
- [ ] Look up agent manifest by agent_id
- [ ] Verify agent status is ACTIVE
- [ ] Check agent satisfies any required capabilities
- [ ] Cache agent lookups for performance
- [ ] Clear error if agent not found or inactive

**Technical Notes:**
- Use registry.get() method
- Consider caching with TTL for performance
- Raise `AgentNotFoundError` or `AgentInactiveError`

---

### E2-S5: Adapter Routing
**As a** developer
**I want** steps to route to the correct adapter based on agent framework
**So that** workflows can mix agents from different frameworks

**Acceptance Criteria:**
- [ ] Determine adapter from agent's `framework` field
- [ ] Support langgraph, crewai, autogen frameworks
- [ ] Raise error if no adapter registered for framework
- [ ] Support registering custom adapters
- [ ] Log which adapter is being used for debugging

**Technical Notes:**
- Store adapters in dict keyed by framework name
- Create `AdapterRegistry` if needed
- Support lazy adapter initialization

---

## Definition of Done (Epic 2)
- [ ] All stories completed and tested
- [ ] Unit tests with >90% coverage
- [ ] Integration test: workflow with agents from 2 different frameworks
- [ ] Timeout test: verify timeout works correctly
- [ ] Documentation in docstrings

---

# Epic 3: State Management

## Overview
Manage workflow state, step results, and data flow between steps.

## User Stories

### E3-S1: Workflow State Machine
**As a** developer
**I want** workflow execution to follow a clear state machine
**So that** I can understand and track workflow progress

**Acceptance Criteria:**
- [ ] States: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
- [ ] Valid transitions enforced (can't go COMPLETED → RUNNING)
- [ ] State changes emit events
- [ ] Terminal states (COMPLETED, FAILED, CANCELLED) are final
- [ ] State accessible throughout execution

**Technical Notes:**
- Use enum for states
- Create state machine helper class
- Log all state transitions

---

### E3-S2: Step Result Storage
**As a** developer
**I want** step results stored in execution context
**So that** subsequent steps can access them

**Acceptance Criteria:**
- [ ] Store step output in context.steps[step_id].output
- [ ] Store step status in context.steps[step_id].status
- [ ] Store step metrics in context.steps[step_id].metrics
- [ ] Store step error in context.steps[step_id].error (if failed)
- [ ] Results available immediately after step completes

**Technical Notes:**
- Use dict[str, StepResult] in ExecutionContext
- StepResult dataclass with output, status, metrics, error
- Thread-safe updates

---

### E3-S3: Output Mapping
**As a** developer
**I want** to define how workflow output is constructed from step outputs
**So that** I can control the final workflow result

**Acceptance Criteria:**
- [ ] Define `output_map` in workflow definition
- [ ] Map step outputs to final workflow output
- [ ] Support nested output construction
- [ ] Default: return last step's output if no output_map
- [ ] Validate output_map references valid steps

**Technical Notes:**
- Reuse InputMapper for output mapping
- Apply after all steps complete
- Create OutputBuilder class

---

### E3-S4: Context Variables
**As a** developer
**I want** to pass context variables through the workflow
**So that** I can share metadata like user_id, trace_id

**Acceptance Criteria:**
- [ ] Accept `context` parameter in execute()
- [ ] Context available via `$.context.X` in input_map
- [ ] Context passed to all step executions
- [ ] Support trace_id and correlation_id for tracing
- [ ] Context is immutable during execution

**Technical Notes:**
- Store in ExecutionContext
- Include trace_id, correlation_id, user context
- Pass to Task creation for agents

---

### E3-S5: Execution Persistence (Optional)
**As a** developer
**I want** execution state to be persisted
**So that** I can recover from crashes

**Acceptance Criteria:**
- [ ] Persist execution state to SQLite
- [ ] Load execution state on startup
- [ ] Resume failed executions from last checkpoint
- [ ] Clean up completed executions after TTL
- [ ] Support both in-memory and persistent modes

**Technical Notes:**
- Create ExecutionStore class
- Use SQLite for persistence
- This is optional for v1.0, can be in-memory only

---

## Definition of Done (Epic 3)
- [ ] All stories completed and tested
- [ ] State machine tested thoroughly
- [ ] Input/output mapping tested with complex paths
- [ ] Context propagation verified

---

# Epic 4: Reliability

## Overview
Make workflow execution reliable with retries, fallbacks, and error handling.

## User Stories

### E4-S1: Retry Logic
**As a** developer
**I want** failed steps to be retried automatically
**So that** transient failures don't fail the entire workflow

**Acceptance Criteria:**
- [ ] Retry on configurable error types (TIMEOUT, EXTERNAL_SERVICE_ERROR)
- [ ] Don't retry on non-retryable errors (INVALID_INPUT, PERMISSION_DENIED)
- [ ] Respect max_attempts from retry policy
- [ ] Emit retry events
- [ ] Track retry count in metrics

**Technical Notes:**
- Check TaskError.retryable flag
- Default max_attempts = 3
- Create RetryHandler class

---

### E4-S2: Exponential Backoff
**As a** developer
**I want** retries to use exponential backoff
**So that** we don't overwhelm failing services

**Acceptance Criteria:**
- [ ] Initial delay from backoff_ms (default 1000)
- [ ] Multiply by backoff_multiplier (default 2) each retry
- [ ] Cap at max_backoff_ms (default 30000)
- [ ] Add jitter (0-25% random) to prevent thundering herd
- [ ] Log delay before each retry

**Technical Notes:**
- Use asyncio.sleep() for delay
- Formula: min(backoff_ms * multiplier^attempt + jitter, max_backoff_ms)
- Create BackoffCalculator utility

---

### E4-S3: Fallback Execution
**As a** developer
**I want** to define fallback behavior for failed steps
**So that** the workflow can continue despite failures

**Acceptance Criteria:**
- [ ] Support fallback.agent_id - use alternate agent
- [ ] Support fallback.static_value - return fixed value
- [ ] Support fallback.skip - mark step as skipped
- [ ] Fallback executes only after all retries exhausted
- [ ] Fallback agent has same input as original
- [ ] Emit fallback event

**Technical Notes:**
- Create FallbackHandler class
- Fallback agent can also have retries
- Track fallback_count in metrics

---

### E4-S4: Error Categorization
**As a** developer
**I want** errors to be categorized consistently
**So that** I can handle different error types appropriately

**Acceptance Criteria:**
- [ ] Map adapter errors to standard error codes
- [ ] Categories: INVALID_INPUT, AGENT_NOT_FOUND, TIMEOUT, EXTERNAL_SERVICE_ERROR, INTERNAL_ERROR
- [ ] Include original error details
- [ ] Include stack trace for debugging
- [ ] Indicate if error is retryable

**Technical Notes:**
- Extend TaskError with consistent codes
- Create error mapping per adapter
- Use existing TaskError from core types

---

### E4-S5: Workflow-Level Timeout
**As a** developer
**I want** workflows to have an overall timeout
**So that** stuck workflows don't run forever

**Acceptance Criteria:**
- [ ] Define timeout_ms at workflow level
- [ ] Cancel all running steps on workflow timeout
- [ ] Return partial results if available
- [ ] Set status to TIMEOUT
- [ ] Default workflow timeout = 1 hour

**Technical Notes:**
- Wrap entire execute() in asyncio.wait_for()
- Track time remaining for subsequent steps
- Create WorkflowTimeoutError

---

## Definition of Done (Epic 4)
- [ ] All stories completed and tested
- [ ] Retry tested with mock failing agent
- [ ] Fallback tested with all three modes
- [ ] Timeout tested at step and workflow level
- [ ] Integration test: resilient workflow

---

# Epic 5: Advanced Execution Patterns

## Overview
Support parallel execution, conditional steps, and complex dependency graphs.

## User Stories

### E5-S1: Parallel Step Execution
**As a** developer
**I want** independent steps to execute in parallel
**So that** workflows complete faster

**Acceptance Criteria:**
- [ ] Steps with same dependencies execute concurrently
- [ ] Use asyncio.gather() for parallel execution
- [ ] Limit concurrent steps (configurable, default 10)
- [ ] Handle partial failures in parallel steps
- [ ] Aggregate results from parallel steps

**Technical Notes:**
- Identify steps that can run in parallel from DAG
- Use semaphore to limit concurrency
- Create ParallelExecutor class

---

### E5-S2: Conditional Steps
**As a** developer
**I want** steps to execute conditionally based on expressions
**So that** I can build branching workflows

**Acceptance Criteria:**
- [ ] Define `condition` as expression on step
- [ ] Evaluate condition before executing step
- [ ] Skip step if condition is false
- [ ] Support comparison operators: ==, !=, <, >, <=, >=
- [ ] Support boolean operators: &&, ||, !
- [ ] Support `in` operator for lists
- [ ] Access workflow input and step outputs in conditions

**Technical Notes:**
- Create ConditionEvaluator class
- Use safe expression parser (no eval!)
- Return SKIPPED status if condition false

---

### E5-S3: Dependency Graph (DAG)
**As a** developer
**I want** to define complex step dependencies
**So that** I can model real-world workflow patterns

**Acceptance Criteria:**
- [ ] Build DAG from step depends_on
- [ ] Detect and reject cycles
- [ ] Compute execution order via topological sort
- [ ] Identify steps that can run in parallel
- [ ] Visualize DAG (optional, for debugging)

**Technical Notes:**
- Use Kahn's algorithm for topological sort
- Store DAG in ExecutionContext
- Create DAGBuilder class

---

### E5-S4: Fan-Out Pattern
**As a** developer
**I want** to fan out work to multiple parallel steps
**So that** I can process lists in parallel

**Acceptance Criteria:**
- [ ] Define fan_out on step to iterate over list
- [ ] Create parallel step instance per list item
- [ ] Each instance receives one item as input
- [ ] Support limiting concurrency
- [ ] Results aggregated as list

**Technical Notes:**
- Expand single step into N steps at runtime
- Use input_map to specify list path
- Create FanOutHandler class

---

### E5-S5: Fan-In Pattern
**As a** developer
**I want** to aggregate results from parallel steps
**So that** I can combine results

**Acceptance Criteria:**
- [ ] Define fan_in step that waits for multiple steps
- [ ] Collect outputs from all fan-out steps
- [ ] Support aggregation functions: list, merge, first
- [ ] Handle partial results if some steps fail
- [ ] Pass aggregated result to downstream steps

**Technical Notes:**
- Pair with fan_out steps
- Create FanInHandler class
- Store aggregated results

---

## Definition of Done (Epic 5)
- [ ] All stories completed and tested
- [ ] Parallel execution tested with timing verification
- [ ] Conditional branching tested
- [ ] Fan-out/fan-in tested with list processing
- [ ] Performance test: parallel vs sequential

---

# Epic 6: Events & Observability

## Overview
Stream real-time events and collect metrics during workflow execution.

## User Stories

### E6-S1: Event Emitter
**As a** developer
**I want** workflow events emitted during execution
**So that** I can monitor progress in real-time

**Acceptance Criteria:**
- [ ] Emit events for workflow: started, completed, failed
- [ ] Emit events for steps: started, completed, failed, retrying, skipped
- [ ] Events include execution_id, workflow_id, timestamp
- [ ] Events include relevant payload (step_id, output, error)
- [ ] Support multiple event listeners

**Technical Notes:**
- Create EventEmitter class
- Use async callbacks for listeners
- Events should be non-blocking

---

### E6-S2: Streaming Execution
**As a** developer
**I want** to receive events as async iterator
**So that** I can stream progress to clients

**Acceptance Criteria:**
- [ ] `execute_streaming()` method returns AsyncIterator[Event]
- [ ] Events yielded as they occur
- [ ] Final event is workflow.completed or workflow.failed
- [ ] Iterator completes when workflow completes
- [ ] Support cancellation via iterator close

**Technical Notes:**
- Use async generator
- Buffer events in queue
- Create StreamingExecutor wrapper

---

### E6-S3: Metrics Collection
**As a** developer
**I want** execution metrics collected automatically
**So that** I can analyze performance and costs

**Acceptance Criteria:**
- [ ] Collect per-step: execution_time_ms, token_usage, retries
- [ ] Aggregate workflow: total_time, total_tokens, total_cost
- [ ] Include in workflow result
- [ ] Track retry_count and fallback_count
- [ ] Calculate p50/p99 latencies (optional)

**Technical Notes:**
- Create MetricsCollector class
- Aggregate from TaskResult.metrics
- Include in WorkflowResult

---

### E6-S4: Execution History
**As a** developer
**I want** to query past workflow executions
**So that** I can debug and audit

**Acceptance Criteria:**
- [ ] Store execution records with results
- [ ] Query by workflow_id, execution_id, status
- [ ] Query by time range
- [ ] Include full event history
- [ ] Configurable retention period

**Technical Notes:**
- Store in SQLite
- Create ExecutionHistory class
- Include events as JSON array

---

### E6-S5: Tracing Integration
**As a** developer
**I want** workflows to integrate with distributed tracing
**So that** I can trace across services

**Acceptance Criteria:**
- [ ] Accept trace_id and parent_span_id in context
- [ ] Generate span_id for each step
- [ ] Include trace IDs in all events
- [ ] Support OpenTelemetry format
- [ ] Pass trace context to agents

**Technical Notes:**
- Use trace_id from context or generate new
- Create spans per step
- Optional OpenTelemetry integration

---

## Definition of Done (Epic 6)
- [ ] All stories completed and tested
- [ ] Streaming tested with async consumer
- [ ] Metrics verified against actual execution
- [ ] History query tested

---

# Epic 7: REST API Integration

## Overview
Expose orchestration capabilities through the REST API.

## User Stories

### E7-S1: Workflow CRUD Endpoints
**As an** API consumer
**I want** to manage workflows via REST API
**So that** I can integrate with external systems

**Acceptance Criteria:**
- [ ] POST /workflows - Create workflow
- [ ] GET /workflows - List workflows
- [ ] GET /workflows/{id} - Get workflow
- [ ] PUT /workflows/{id} - Update workflow
- [ ] DELETE /workflows/{id} - Delete workflow
- [ ] Proper error responses for all endpoints

**Technical Notes:**
- Add to awf/api/app.py
- Create Pydantic models for request/response
- Store workflows in registry or separate store

---

### E7-S2: Workflow Execution Endpoints
**As an** API consumer
**I want** to execute workflows via REST API
**So that** I can trigger workflows externally

**Acceptance Criteria:**
- [ ] POST /workflows/{id}/execute - Start execution
- [ ] GET /executions/{id} - Get execution status
- [ ] GET /executions/{id}/events - Stream events (SSE)
- [ ] DELETE /executions/{id} - Cancel execution
- [ ] Return execution_id on start

**Technical Notes:**
- Use Server-Sent Events for streaming
- Store executions for status queries
- Support async execution (return immediately)

---

### E7-S3: Execution Query Endpoints
**As an** API consumer
**I want** to query execution history
**So that** I can analyze past executions

**Acceptance Criteria:**
- [ ] GET /executions - List executions with filters
- [ ] Filter by workflow_id, status, time range
- [ ] Pagination support
- [ ] Include summary metrics
- [ ] Sort by start_time

**Technical Notes:**
- Query ExecutionHistory
- Return paginated results
- Include links to individual executions

---

## Definition of Done (Epic 7)
- [ ] All endpoints implemented and tested
- [ ] OpenAPI schema updated
- [ ] Integration tests with TestClient
- [ ] Documentation updated

---

# Technical Architecture Decisions

## Decision 1: Async-First Design
**Decision:** All orchestration is async using asyncio
**Rationale:** Workflows involve I/O-bound operations; async enables efficient concurrency
**Implications:** All public methods are async; use `await` for execution

## Decision 2: JSONPath for Data Mapping
**Decision:** Use JSONPath expressions for input/output mapping
**Rationale:** Familiar syntax, flexible, handles nested data well
**Implications:** Need JSONPath library; validate expressions at definition time

## Decision 3: Event-Driven Architecture
**Decision:** Emit events for all significant state changes
**Rationale:** Enables real-time monitoring, debugging, and integration
**Implications:** Event emitter infrastructure; async event dispatch

## Decision 4: Pluggable Adapters
**Decision:** Framework adapters are pluggable via dependency injection
**Rationale:** Support new frameworks without core changes
**Implications:** Adapter interface must be stable; registry for adapters

## Decision 5: In-Memory First, Persistence Optional
**Decision:** v1.0 works entirely in-memory; persistence is optional
**Rationale:** Simpler implementation; most workflows complete quickly
**Implications:** Executions lost on crash; add persistence in v1.1

---

# File Structure

```
awf/orchestration/
├── __init__.py              # Public exports
├── types.py                 # WorkflowDefinition, StepDefinition, etc.
├── orchestrator.py          # Main Orchestrator class
├── executor.py              # StepExecutor class
├── context.py               # ExecutionContext, state management
├── mapping.py               # InputMapper, OutputBuilder (JSONPath)
├── validation.py            # WorkflowValidator
├── reliability.py           # RetryHandler, FallbackHandler, BackoffCalculator
├── dag.py                   # DAGBuilder, topological sort
├── conditions.py            # ConditionEvaluator
├── parallel.py              # ParallelExecutor, FanOutHandler, FanInHandler
├── events.py                # EventEmitter, event types
├── metrics.py               # MetricsCollector
└── errors.py                # Custom exceptions

tests/
├── test_orchestration_types.py
├── test_orchestrator.py
├── test_step_executor.py
├── test_mapping.py
├── test_reliability.py
├── test_parallel.py
├── test_conditions.py
└── test_orchestration_integration.py
```

---

# Implementation Order

1. **E1: Core Engine** (Foundation)
2. **E2: Step Execution** (Depends on E1)
3. **E3: State Management** (Depends on E1)
4. **E4: Reliability** (Depends on E2)
5. **E5: Advanced Patterns** (Depends on E2, E3)
6. **E6: Observability** (Can parallel with E4, E5)
7. **E7: REST API** (Depends on all above)

---

# Test Strategy

| Level | Coverage Target | Focus |
|-------|-----------------|-------|
| Unit | 90% | Individual classes, edge cases |
| Integration | Core flows | Multi-step workflows, cross-framework |
| E2E | Happy paths | API to execution to result |

**Key Test Scenarios:**
1. 3-step sequential workflow (happy path)
2. Parallel step execution with timing verification
3. Retry with eventual success
4. Fallback execution on failure
5. Conditional step skip
6. Fan-out/fan-in with list processing
7. Workflow timeout
8. Cross-framework workflow (LangGraph + CrewAI)
