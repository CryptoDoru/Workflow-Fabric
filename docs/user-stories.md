# AI Workflow Fabric - User Stories

## Overview

This document contains the prioritized user stories for AI Workflow Fabric (AWF), organized by epic. Each story follows the format:

- **As a** [persona]
- **I want** [capability]
- **So that** [benefit]

Stories include detailed acceptance criteria and are sized using story points (1, 2, 3, 5, 8, 13).

---

## Epic 1: Agent Registration & Discovery

**Goal**: Enable users to register agents from any supported framework and discover them through a unified interface.

**Value Proposition**: Developers can find and use agents regardless of which framework they were built with, eliminating silos and enabling cross-framework collaboration.

**Personas Served**: Vibecoder (primary), Orchestrator (primary), Governor (secondary)

---

### Story 1.1: Register a LangGraph Agent

**As a** developer using LangGraph  
**I want** to register my LangGraph agent with AWF  
**So that** it can be discovered and invoked through the unified protocol

**Story Points**: 5

#### Acceptance Criteria

1. **Given** I have a LangGraph StateGraph or CompiledGraph
   **When** I call `adapter.register(graph)`
   **Then** an ASP-compliant manifest is generated and stored

2. **Given** a LangGraph graph with nodes and edges
   **When** the manifest is generated
   **Then** it includes:
   - Agent ID (generated or user-specified)
   - Version (semantic versioning)
   - All nodes mapped to ASP capabilities
   - Input/output schemas inferred from state type
   - Required tools declared as capabilities

3. **Given** a registered agent
   **When** I query the registry by agent ID
   **Then** I receive the complete manifest

4. **Given** a LangGraph graph using tools (e.g., Tavily search)
   **When** registered
   **Then** tools are mapped to ASP capability declarations with proper permission requirements

5. **Given** an invalid graph (no nodes, invalid structure)
   **When** I attempt to register
   **Then** a descriptive error is raised before registration

#### Technical Notes

- Use LangGraph's introspection to extract graph structure
- Infer input/output schemas from TypedDict or Pydantic models
- Generate deterministic agent IDs based on graph structure hash
- Support explicit metadata override via optional parameters

---

### Story 1.2: Register a CrewAI Agent

**As a** developer using CrewAI  
**I want** to register my CrewAI agents with AWF  
**So that** they can participate in workflows with agents from other frameworks

**Story Points**: 5

#### Acceptance Criteria

1. **Given** I have a CrewAI Agent object
   **When** I call `adapter.register(agent)`
   **Then** an ASP-compliant manifest is generated

2. **Given** a CrewAI agent with role, goal, and backstory
   **When** the manifest is generated
   **Then** it includes:
   - Agent ID derived from agent name/role
   - Capabilities mapped from agent's tools
   - Role and goal as metadata
   - Memory requirements if agent uses memory

3. **Given** a CrewAI Crew (team of agents)
   **When** I register the crew
   **Then** each agent is registered individually with team relationship metadata

4. **Given** an agent with custom tools
   **When** registered
   **Then** tool names and expected inputs/outputs are captured as capabilities

#### Technical Notes

- CrewAI agents have less structured schemas; use heuristics for inference
- Capture process type (sequential, hierarchical) in crew metadata
- Support registering individual agents or entire crews

---

### Story 1.3: Discover Agents by Capability

**As a** developer building a multi-agent workflow  
**I want** to search for agents by their capabilities  
**So that** I can find the right agent for my task without knowing which framework it was built with

**Story Points**: 3

#### Acceptance Criteria

1. **Given** multiple agents registered with various capabilities
   **When** I query `registry.search(capability="web_search")`
   **Then** I receive all agents that declare web search capability

2. **Given** agents from different frameworks (LangGraph, CrewAI)
   **When** I search by capability
   **Then** results include agents from all frameworks, indistinguishable in format

3. **Given** a capability search
   **When** results are returned
   **Then** each result includes:
   - Agent ID
   - Agent name
   - Framework origin
   - Trust score
   - Capability match details

4. **Given** no agents match the query
   **When** I search
   **Then** an empty list is returned (not an error)

5. **Given** a search with multiple capability filters
   **When** I query `registry.search(capabilities=["web_search", "summarization"])`
   **Then** only agents with ALL specified capabilities are returned

#### Technical Notes

- Support fuzzy matching for capability names
- Consider adding semantic similarity for capability discovery
- Cache search results for performance
- Support pagination for large registries

---

## Epic 2: Workflow Orchestration

**Goal**: Enable developers to compose and execute workflows that span multiple agents across different frameworks.

**Value Proposition**: Build complex multi-agent systems without writing custom integration code, while maintaining observability and reliability.

**Personas Served**: Orchestrator (primary), Vibecoder (primary), Governor (secondary)

---

### Story 2.1: Execute a Task on a Registered Agent

**As a** developer  
**I want** to execute a task on any registered agent using the ASP protocol  
**So that** I don't need to know the agent's native framework API

**Story Points**: 5

#### Acceptance Criteria

1. **Given** a registered agent with ID "research-agent"
   **When** I call `awf.execute(agent_id="research-agent", task=task)`
   **Then** the task is translated to the native framework format and executed

2. **Given** a task with input payload
   **When** executed
   **Then** input is validated against the agent's declared input schema

3. **Given** a successful execution
   **When** the agent completes
   **Then** I receive an ASP TaskResult with:
   - Status: "completed"
   - Output: agent's response in standardized format
   - Metrics: execution time, token usage (if applicable)

4. **Given** an execution failure
   **When** the agent errors
   **Then** I receive an ASP TaskResult with:
   - Status: "failed"
   - Error: structured error with code, message, and details
   - Partial output (if any)

5. **Given** a long-running task
   **When** execution takes more than a threshold
   **Then** I can poll for status using a task ID

6. **Given** a task with a timeout specified
   **When** the timeout is exceeded
   **Then** the task is cancelled and returns with status "timeout"

#### Technical Notes

- Implement adapter pattern to translate ASP → native → ASP
- Use async execution with task ID for long-running operations
- Capture execution metrics for observability
- Support streaming responses where native framework allows

---

### Story 2.2: Compose a Multi-Agent Workflow

**As a** developer  
**I want** to define a workflow that orchestrates multiple agents  
**So that** I can build complex pipelines without manual integration

**Story Points**: 8

#### Acceptance Criteria

1. **Given** a workflow definition with sequential steps
   **When** I define:
   ```python
   workflow = Workflow([
       Step(agent="research-agent", input_map={"query": "$.input.topic"}),
       Step(agent="writer-agent", input_map={"research": "$.steps[0].output"})
   ])
   ```
   **Then** the workflow is validated and stored

2. **Given** a valid workflow
   **When** I execute it with input
   **Then** steps execute in order, each receiving mapped inputs from previous steps

3. **Given** a step failure
   **When** an agent fails
   **Then** the workflow stops (or follows defined error handling) and returns partial results

4. **Given** a workflow with parallel steps
   **When** I define independent branches
   **Then** they execute concurrently for improved performance

5. **Given** a workflow with conditional logic
   **When** I define:
   ```python
   Step(agent="router-agent", 
        branches={
            "technical": Step(agent="tech-writer"),
            "creative": Step(agent="creative-writer")
        })
   ```
   **Then** the appropriate branch executes based on router output

6. **Given** a workflow execution
   **When** it completes
   **Then** I receive:
   - Final output
   - Execution trace with each step's input/output
   - Aggregate metrics

#### Technical Notes

- Use JSONPath for input/output mapping between steps
- Consider DAG representation for complex workflows
- Support workflow versioning
- Enable workflow pause/resume for human-in-the-loop

---

### Story 2.3: Unified Observability Across Agents

**As a** developer operating multi-agent workflows  
**I want** unified tracing and logging across all agents  
**So that** I can debug issues without stitching together different log formats

**Story Points**: 5

#### Acceptance Criteria

1. **Given** a workflow execution spanning multiple agents
   **When** I request the trace
   **Then** I receive a single trace with spans for each agent invocation

2. **Given** agents from different frameworks
   **When** traced
   **Then** all spans follow the same format (OpenTelemetry compatible)

3. **Given** a trace span for an agent
   **When** I inspect it
   **Then** it includes:
   - Agent ID and name
   - Start/end timestamps
   - Input/output (or references)
   - Status and any errors
   - Framework-specific metadata

4. **Given** a failed step
   **When** I view the trace
   **Then** the failure point is clearly marked with error details

5. **Given** nested agent invocations (agent calling another agent)
   **When** traced
   **Then** parent-child relationships are preserved in the trace

6. **Given** trace data
   **When** I export it
   **Then** it can be sent to standard observability backends (Jaeger, Datadog, etc.)

#### Technical Notes

- Use OpenTelemetry for trace format
- Generate trace context at workflow start, propagate through steps
- Support configurable verbosity (minimal, standard, verbose)
- Consider privacy: option to exclude sensitive input/output

---

## Epic 3: Security & Trust

**Goal**: Enable safe execution of agents with appropriate isolation and trust-based policies.

**Value Proposition**: Run agents—including third-party ones—with confidence, knowing they are sandboxed and their trustworthiness is objectively assessed.

**Personas Served**: Governor (primary), Orchestrator (primary), Vibecoder (secondary)

---

### Story 3.1: Compute Trust Score for Agent

**As a** platform administrator  
**I want** an automatic trust score computed for each registered agent  
**So that** I can make informed decisions about where and how to run it

**Story Points**: 5

#### Acceptance Criteria

1. **Given** a newly registered agent
   **When** registration completes
   **Then** a trust score (0.00-1.00) is computed and stored

2. **Given** the trust scoring algorithm
   **When** a score is computed
   **Then** it factors in:
   - Publisher trust (verified publisher, reputation)
   - Audit status (third-party security audits)
   - Community trust (usage, ratings, reports)
   - Permission analysis (scope of capabilities requested)
   - Historical behavior (if previously run)

3. **Given** a trust score
   **When** I query agent metadata
   **Then** I see the score and contributing factors breakdown

4. **Given** an agent from an unverified publisher with broad permissions
   **When** scored
   **Then** it receives a lower score reflecting higher risk

5. **Given** an agent whose behavior changes (new version, abuse reports)
   **When** events occur
   **Then** the trust score is recalculated

6. **Given** configurable trust policies
   **When** I set minimum trust threshold for production
   **Then** agents below threshold cannot be invoked in production

#### Technical Notes

- See trust-scoring.md specification for algorithm details
- Store historical scores for trend analysis
- Support organization-specific trust overrides
- Enable trust score explanations for transparency

---

### Story 3.2: Execute Agent in Sandboxed Environment

**As a** developer running agents from various sources  
**I want** agents to execute in isolated sandboxes  
**So that** a misbehaving agent cannot compromise the system or access unauthorized resources

**Story Points**: 8

#### Acceptance Criteria

1. **Given** an agent with trust score 0.90+
   **When** executed
   **Then** it runs in WASM sandbox (minimal overhead, ~10ms)

2. **Given** an agent with trust score 0.70-0.89
   **When** executed
   **Then** it runs in gVisor sandbox (moderate isolation, ~100ms overhead)

3. **Given** an agent with trust score 0.40-0.69
   **When** executed
   **Then** it runs in strict gVisor sandbox (~150ms overhead)

4. **Given** an agent with trust score below 0.40
   **When** execution is attempted
   **Then** it is BLOCKED by default (configurable)

5. **Given** a sandbox execution
   **When** the agent attempts to access resources beyond declared capabilities
   **Then** the attempt is denied and logged

6. **Given** sandbox tiers
   **When** an administrator configures custom policies
   **Then** threshold mappings can be adjusted per environment

7. **Given** a sandbox execution
   **When** the agent completes
   **Then** resource usage (CPU, memory, network) is captured in metrics

#### Technical Notes

- WASM for high-trust: use Wasmtime or WasmEdge
- gVisor for lower trust: container-based isolation
- All sandboxes enforce capability-based security
- Network access controlled per declared capabilities
- Support timeout and resource limits per sandbox

---

### Story 3.3: Define and Enforce Agent Policies

**As a** platform administrator  
**I want** to define policies that govern agent execution  
**So that** I can enforce organizational security and compliance requirements

**Story Points**: 5

#### Acceptance Criteria

1. **Given** policy configuration
   **When** I define:
   ```yaml
   policies:
     - name: production-minimum-trust
       environment: production
       min_trust_score: 0.80
     - name: no-external-network
       environments: [staging, production]
       deny_capabilities: [network:external]
   ```
   **Then** policies are validated and stored

2. **Given** an active policy
   **When** an agent execution violates it
   **Then** execution is blocked and a policy violation error is returned

3. **Given** multiple applicable policies
   **When** evaluating an execution
   **Then** all policies must pass (AND logic)

4. **Given** a policy violation
   **When** blocked
   **Then** the violation is logged with:
   - Agent ID
   - Policy name
   - Violation details
   - Requesting user/context

5. **Given** policy changes
   **When** updated
   **Then** changes are applied immediately to new executions (not retroactive)

6. **Given** an override requirement
   **When** an admin grants exception
   **Then** exception is logged and time-bounded

#### Technical Notes

- Use YAML or JSON for policy definitions
- Support policy inheritance and composition
- Integrate with external policy engines (OPA) for complex rules
- Provide policy simulation mode for testing

---

## Epic Summary

| Epic | Stories | Total Points | Priority |
|------|---------|--------------|----------|
| 1. Agent Registration & Discovery | 3 | 13 | High |
| 2. Workflow Orchestration | 3 | 18 | High |
| 3. Security & Trust | 3 | 18 | High |
| **Total** | **9** | **49** | - |

---

## Prioritized Backlog

Based on user research and strategic value, the recommended implementation order:

### Sprint 1: Foundation (21 points)
1. Story 1.1: Register a LangGraph Agent (5 pts)
2. Story 2.1: Execute a Task on a Registered Agent (5 pts)
3. Story 1.3: Discover Agents by Capability (3 pts)
4. Story 2.3: Unified Observability Across Agents (5 pts)
5. Story 3.1: Compute Trust Score for Agent (5 pts) - partial

### Sprint 2: Multi-Framework & Orchestration (16 points)
1. Story 1.2: Register a CrewAI Agent (5 pts)
2. Story 2.2: Compose a Multi-Agent Workflow (8 pts)
3. Story 3.1: Compute Trust Score for Agent (complete)

### Sprint 3: Security & Enterprise (13 points)
1. Story 3.2: Execute Agent in Sandboxed Environment (8 pts)
2. Story 3.3: Define and Enforce Agent Policies (5 pts)

---

## Definition of Done

A story is considered done when:

- [ ] All acceptance criteria pass
- [ ] Code is reviewed and merged
- [ ] Unit tests cover core functionality (>80% coverage)
- [ ] Integration tests pass
- [ ] Documentation is updated
- [ ] No regressions in existing functionality
- [ ] Performance within acceptable bounds (defined per story if applicable)
