# AI Workflow Fabric - Epics & Stories

## Vision Statement

**"Kubernetes for AI Agents"** - A unified middleware layer that enables:
1. **Agent Framework Interoperability** - LangGraph, CrewAI, AutoGen, etc.
2. **LLM Provider Abstraction** - OpenAI, Anthropic, Google, etc.
3. **Production Reliability** - Trust scoring, sandboxing, observability
4. **Multi-Agent Orchestration** - Compose agents from any source

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           YOUR APPLICATION                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AWF ORCHESTRATOR                                     │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐ │
│  │ Registry  │  │  Router   │  │   Trust   │  │  Policy   │  │  Sandbox  │ │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
          │                              │                              │
          ▼                              ▼                              ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  FRAMEWORK ADAPTERS │    │   LLM PROVIDERS     │    │   TOOL PROVIDERS    │
│  ┌───────────────┐  │    │  ┌───────────────┐  │    │  ┌───────────────┐  │
│  │   LangGraph   │  │    │  │    OpenAI     │  │    │  │  Web Search   │  │
│  │    CrewAI     │  │    │  │   Anthropic   │  │    │  │  Code Exec    │  │
│  │   AutoGen     │  │    │  │    Google     │  │    │  │   Database    │  │
│  │   Assistants  │  │    │  │    Mistral    │  │    │  │     MCP       │  │
│  │    DSPy       │  │    │  │    Ollama     │  │    │  │   Custom      │  │
│  └───────────────┘  │    │  └───────────────┘  │    │  └───────────────┘  │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

---

## Epic 1: Testing Infrastructure
**Priority:** HIGH | **Effort:** M | **Dependencies:** None

Complete test suite for production readiness and CI/CD integration.

### Story 1.1: Registry Tests
**As a** developer  
**I want** comprehensive tests for InMemory and SQLite registries  
**So that** I can trust the registry works correctly in production

**Acceptance Criteria:**
- [ ] Test agent registration, retrieval, update, delete
- [ ] Test search by capability, framework, tags, trust score
- [ ] Test SQLite persistence across restarts
- [ ] Test concurrent access with async locks
- [ ] Test schema migrations
- [ ] 90%+ code coverage

### Story 1.2: API Tests
**As a** developer  
**I want** tests for all REST API endpoints  
**So that** the API contract is guaranteed

**Acceptance Criteria:**
- [ ] Test all CRUD operations for agents
- [ ] Test task submission and status
- [ ] Test trust score endpoints
- [ ] Test error handling and validation
- [ ] Test authentication/authorization (when added)
- [ ] Use pytest-asyncio and httpx

### Story 1.3: Security Tests
**As a** developer  
**I want** tests for trust, policy, and sandbox modules  
**So that** security guarantees are verified

**Acceptance Criteria:**
- [ ] Test trust score computation for various scenarios
- [ ] Test policy evaluation and violation detection
- [ ] Test sandbox tier selection based on trust
- [ ] Test sandbox execution isolation
- [ ] Test resource limit enforcement

### Story 1.4: Integration Tests
**As a** developer  
**I want** end-to-end tests of the full workflow  
**So that** all components work together correctly

**Acceptance Criteria:**
- [ ] Test: Register agent → Compute trust → Submit task → Execute
- [ ] Test multi-agent workflow execution
- [ ] Test failure scenarios and recovery
- [ ] Test with real framework adapters (mocked LLMs)

---

## Epic 2: LLM Provider Adapters
**Priority:** HIGH | **Effort:** L | **Dependencies:** Core complete

Enable any LLM to be exposed as an AWF agent for simple use cases.

### Story 2.1: LLM Provider Base Class
**As a** developer  
**I want** a base class for LLM providers  
**So that** new providers can be added consistently

**Acceptance Criteria:**
- [ ] Create `awf/providers/__init__.py`
- [ ] Create `awf/providers/base.py` with `LLMProvider` ABC
- [ ] Define standard interface: `complete()`, `stream()`, `embed()`
- [ ] Support for messages format (system, user, assistant)
- [ ] Token counting and cost estimation
- [ ] Rate limiting support

### Story 2.2: OpenAI Provider
**As a** user  
**I want** to use OpenAI models (GPT-4, GPT-4o, o1) as AWF agents  
**So that** I can leverage OpenAI's capabilities in my workflows

**Acceptance Criteria:**
- [ ] Create `awf/providers/openai.py`
- [ ] Support all GPT-4 variants, GPT-4o, o1 models
- [ ] Support function calling / tool use
- [ ] Support streaming responses
- [ ] Support embeddings (text-embedding-3-small/large)
- [ ] Automatic retry with exponential backoff
- [ ] Cost tracking per request

### Story 2.3: Anthropic Provider
**As a** user  
**I want** to use Claude models as AWF agents  
**So that** I can leverage Anthropic's capabilities in my workflows

**Acceptance Criteria:**
- [ ] Create `awf/providers/anthropic.py`
- [ ] Support Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku
- [ ] Support tool use (Anthropic's tool format)
- [ ] Support streaming responses
- [ ] Support extended context (200K tokens)
- [ ] Cost tracking per request

### Story 2.4: Google Provider
**As a** user  
**I want** to use Gemini models as AWF agents  
**So that** I can leverage Google's capabilities in my workflows

**Acceptance Criteria:**
- [ ] Create `awf/providers/google.py`
- [ ] Support Gemini Pro, Gemini Ultra, Gemini Flash
- [ ] Support function calling
- [ ] Support multimodal inputs (images, video)
- [ ] Support streaming responses
- [ ] Cost tracking per request

### Story 2.5: Mistral Provider
**As a** user  
**I want** to use Mistral models as AWF agents  
**So that** I can leverage Mistral's capabilities

**Acceptance Criteria:**
- [ ] Create `awf/providers/mistral.py`
- [ ] Support Mistral Large, Medium, Small, Codestral
- [ ] Support function calling
- [ ] Support streaming

### Story 2.6: Ollama/Local Provider
**As a** user  
**I want** to use local LLMs via Ollama  
**So that** I can run agents without cloud dependencies

**Acceptance Criteria:**
- [ ] Create `awf/providers/ollama.py`
- [ ] Auto-detect available models
- [ ] Support any Ollama-compatible model
- [ ] Support streaming
- [ ] GPU acceleration when available

### Story 2.7: LLM Agent Wrapper
**As a** user  
**I want** to wrap any LLM provider as an AWF agent  
**So that** I can use LLMs directly without a framework

**Acceptance Criteria:**
- [ ] Create `awf/adapters/llm/` package
- [ ] `LLMAgent` class that wraps any `LLMProvider`
- [ ] Automatic manifest generation from provider capabilities
- [ ] Support for system prompts, temperature, max_tokens
- [ ] Tool/function calling passthrough
- [ ] Conversation history management

---

## Epic 3: Framework Adapters
**Priority:** HIGH | **Effort:** L | **Dependencies:** Core complete

Expand framework support beyond LangGraph and CrewAI.

### Story 3.1: OpenAI Assistants Adapter
**As a** user  
**I want** to use OpenAI Assistants as AWF agents  
**So that** I can leverage Assistant API features (code interpreter, retrieval)

**Acceptance Criteria:**
- [ ] Create `awf/adapters/openai_assistants/`
- [ ] Support assistant creation and management
- [ ] Support threads and runs
- [ ] Support code interpreter tool
- [ ] Support file search/retrieval tool
- [ ] Map to ASP task/result format

### Story 3.2: AutoGen Adapter
**As a** user  
**I want** to use Microsoft AutoGen agents in AWF  
**So that** I can leverage AutoGen's multi-agent conversations

**Acceptance Criteria:**
- [ ] Create `awf/adapters/autogen/`
- [ ] Support AssistantAgent, UserProxyAgent
- [ ] Support GroupChat orchestration
- [ ] Map AutoGen message format to ASP
- [ ] Support code execution (with sandboxing)

### Story 3.3: Semantic Kernel Adapter
**As a** user  
**I want** to use Microsoft Semantic Kernel plugins as AWF agents  
**So that** I can leverage SK's plugin ecosystem

**Acceptance Criteria:**
- [ ] Create `awf/adapters/semantic_kernel/`
- [ ] Support SK plugins as capabilities
- [ ] Support SK planners
- [ ] Support SK memory

### Story 3.4: DSPy Adapter
**As a** user  
**I want** to use DSPy modules as AWF agents  
**So that** I can leverage DSPy's optimization capabilities

**Acceptance Criteria:**
- [ ] Create `awf/adapters/dspy/`
- [ ] Support DSPy modules as agents
- [ ] Support signature-based I/O mapping
- [ ] Support compiled/optimized modules

---

## Epic 4: CLI Tool
**Priority:** HIGH | **Effort:** M | **Dependencies:** API complete

Full-featured command-line interface for AWF management.

### Story 4.1: CLI Foundation
**As a** user  
**I want** a CLI tool to manage AWF  
**So that** I can interact without writing code

**Acceptance Criteria:**
- [ ] Create `awf/cli/` package with Typer
- [ ] Create entry point `awf` command
- [ ] Support `--version`, `--help`
- [ ] Support configuration file (`~/.awf/config.yaml`)
- [ ] Colorized output with Rich

### Story 4.2: Agent Commands
**As a** user  
**I want** CLI commands to manage agents  
**So that** I can register and discover agents easily

**Acceptance Criteria:**
- [ ] `awf agent register <manifest.yaml>` - Register from file
- [ ] `awf agent list [--framework] [--capability]` - List/search agents
- [ ] `awf agent get <id>` - Get agent details
- [ ] `awf agent delete <id>` - Remove agent
- [ ] `awf agent trust <id>` - Show trust score breakdown
- [ ] JSON and table output formats

### Story 4.3: Task Commands
**As a** user  
**I want** CLI commands to submit and monitor tasks  
**So that** I can run agents from the command line

**Acceptance Criteria:**
- [ ] `awf task submit <agent-id> --input '{"query": "..."}' ` - Submit task
- [ ] `awf task status <task-id>` - Check task status
- [ ] `awf task wait <task-id>` - Wait for completion
- [ ] `awf task list [--agent] [--status]` - List tasks
- [ ] Streaming output for long-running tasks

### Story 4.4: Server Commands
**As a** user  
**I want** CLI commands to run the AWF server  
**So that** I can start the API easily

**Acceptance Criteria:**
- [ ] `awf server start [--host] [--port]` - Start API server
- [ ] `awf server status` - Check server status
- [ ] Background/daemon mode support
- [ ] Auto-reload in development mode

### Story 4.5: Workflow Commands
**As a** user  
**I want** CLI commands to manage workflows  
**So that** I can compose multi-agent pipelines

**Acceptance Criteria:**
- [ ] `awf workflow create <workflow.yaml>` - Create workflow
- [ ] `awf workflow run <id> --input '{...}'` - Execute workflow
- [ ] `awf workflow status <execution-id>` - Check execution status
- [ ] `awf workflow list` - List workflows

---

## Epic 5: Sandbox Implementations
**Priority:** HIGH | **Effort:** L | **Dependencies:** Sandbox orchestrator

Real isolation environments for secure agent execution.

### Story 5.1: Process Sandbox
**As a** developer  
**I want** a subprocess-based sandbox  
**So that** agents can be isolated without special dependencies

**Acceptance Criteria:**
- [ ] Create `awf/security/process_sandbox.py`
- [ ] Execute agent code in subprocess
- [ ] Resource limits via `resource` module (Unix) or `joblib` (cross-platform)
- [ ] Timeout enforcement
- [ ] Capture stdout/stderr
- [ ] Works on Windows, Mac, Linux

### Story 5.2: Docker Sandbox
**As a** developer  
**I want** a Docker-based sandbox  
**So that** agents can be fully containerized

**Acceptance Criteria:**
- [ ] Create `awf/security/docker_sandbox.py`
- [ ] Build minimal container images for agent execution
- [ ] Memory and CPU limits via Docker
- [ ] Network isolation options
- [ ] Volume mounting for inputs/outputs
- [ ] Auto-cleanup of containers

### Story 5.3: WASM Sandbox (Future)
**As a** developer  
**I want** a WASM-based sandbox using Wasmtime  
**So that** agents have minimal overhead isolation

**Acceptance Criteria:**
- [ ] Create `awf/security/wasm_sandbox.py`
- [ ] Compile Python to WASM (via RustPython or similar)
- [ ] Memory isolation via WASM linear memory
- [ ] Capability-based permissions
- [ ] ~10ms overhead target

---

## Epic 6: Observability & Monitoring
**Priority:** MEDIUM | **Effort:** M | **Dependencies:** Core complete

Production observability for agent execution.

### Story 6.1: Structured Logging
**As an** operator  
**I want** structured JSON logging  
**So that** I can aggregate logs in production

**Acceptance Criteria:**
- [ ] Create `awf/observability/logging.py`
- [ ] JSON log format with trace IDs
- [ ] Log levels configurable per component
- [ ] Integration with Python logging

### Story 6.2: Metrics Export
**As an** operator  
**I want** Prometheus metrics  
**So that** I can monitor AWF in production

**Acceptance Criteria:**
- [ ] Create `awf/observability/metrics.py`
- [ ] Counters: tasks_submitted, tasks_completed, tasks_failed
- [ ] Histograms: execution_time, trust_score_distribution
- [ ] Gauges: active_agents, active_tasks
- [ ] `/metrics` endpoint

### Story 6.3: Distributed Tracing
**As an** operator  
**I want** OpenTelemetry tracing  
**So that** I can trace requests across agents

**Acceptance Criteria:**
- [ ] Create `awf/observability/tracing.py`
- [ ] OpenTelemetry SDK integration
- [ ] Automatic span creation for tasks
- [ ] Trace context propagation to agents
- [ ] Export to Jaeger/Zipkin/OTLP

### Story 6.4: Cost Tracking
**As a** user  
**I want** to track LLM costs per task  
**So that** I can budget and optimize

**Acceptance Criteria:**
- [ ] Track token usage per LLM call
- [ ] Cost calculation per provider (using current pricing)
- [ ] Aggregate costs by agent, task, user
- [ ] Cost reports via API/CLI

---

## Epic 7: DevOps & Packaging
**Priority:** HIGH | **Effort:** S | **Dependencies:** Tests complete

Production-ready packaging and deployment.

### Story 7.1: Package Configuration
**As a** developer  
**I want** proper pyproject.toml setup  
**So that** AWF can be installed from PyPI

**Acceptance Criteria:**
- [ ] Update pyproject.toml with all dependencies
- [ ] Optional dependency groups: [api], [cli], [langgraph], [crewai], [openai], [anthropic], [dev]
- [ ] Entry points for CLI
- [ ] Package metadata (description, classifiers, URLs)

### Story 7.2: Docker Support
**As an** operator  
**I want** Docker images for AWF  
**So that** I can deploy easily

**Acceptance Criteria:**
- [ ] Create `Dockerfile` for API server
- [ ] Create `docker-compose.yml` for full stack
- [ ] Multi-stage build for small images
- [ ] Health check configuration
- [ ] Publish to Docker Hub/GHCR

### Story 7.3: CI/CD Pipeline
**As a** developer  
**I want** automated CI/CD  
**So that** changes are tested and deployed automatically

**Acceptance Criteria:**
- [ ] Create `.github/workflows/ci.yml`
- [ ] Run tests on PR
- [ ] Run linting (ruff) and type checking (mypy)
- [ ] Publish to PyPI on release tag
- [ ] Build and push Docker images on release

---

## Epic 8: Examples & Documentation
**Priority:** MEDIUM | **Effort:** M | **Dependencies:** Features complete

Comprehensive examples and documentation for adoption.

### Story 8.1: Framework Examples
**As a** user  
**I want** working examples for each framework  
**So that** I can learn how to integrate

**Acceptance Criteria:**
- [ ] `examples/langgraph_example.py` - Full LangGraph integration
- [ ] `examples/crewai_example.py` - Full CrewAI integration
- [ ] `examples/openai_assistants_example.py` - Assistants API
- [ ] `examples/multi_framework.py` - Cross-framework workflow

### Story 8.2: LLM Provider Examples
**As a** user  
**I want** examples for each LLM provider  
**So that** I can use LLMs directly

**Acceptance Criteria:**
- [ ] `examples/llm_openai.py` - OpenAI direct usage
- [ ] `examples/llm_anthropic.py` - Anthropic direct usage
- [ ] `examples/llm_google.py` - Google Gemini usage
- [ ] `examples/llm_ollama.py` - Local LLM usage

### Story 8.3: Architecture Documentation
**As a** developer  
**I want** architecture documentation  
**So that** I can understand and contribute

**Acceptance Criteria:**
- [ ] `docs/architecture.md` - System design
- [ ] `docs/adapters.md` - How to write adapters
- [ ] `docs/providers.md` - How to add LLM providers
- [ ] `docs/security.md` - Trust and sandboxing details

### Story 8.4: API Documentation
**As a** user  
**I want** API reference documentation  
**So that** I can use AWF programmatically

**Acceptance Criteria:**
- [ ] Auto-generated from docstrings (mkdocs + mkdocstrings)
- [ ] OpenAPI spec published
- [ ] Interactive API explorer

---

## Priority Matrix

| Epic | Priority | Effort | Value | Order |
|------|----------|--------|-------|-------|
| 1. Testing | HIGH | M | Reliability | 1 |
| 7. DevOps | HIGH | S | Deployability | 2 |
| 2. LLM Providers | HIGH | L | Adoption | 3 |
| 4. CLI | HIGH | M | Usability | 4 |
| 5. Sandboxes | HIGH | L | Security | 5 |
| 3. Framework Adapters | HIGH | L | Compatibility | 6 |
| 6. Observability | MED | M | Operations | 7 |
| 8. Documentation | MED | M | Adoption | 8 |

---

## Implementation Order

### Phase 1: Production Foundation (Current Sprint)
1. E1.1 - Registry Tests
2. E1.2 - API Tests  
3. E1.3 - Security Tests
4. E7.1 - Package Configuration
5. E7.3 - CI/CD Pipeline

### Phase 2: LLM Provider Support
6. E2.1 - LLM Provider Base
7. E2.2 - OpenAI Provider
8. E2.3 - Anthropic Provider
9. E2.4 - Google Provider
10. E2.7 - LLM Agent Wrapper

### Phase 3: CLI & Usability
11. E4.1 - CLI Foundation
12. E4.2 - Agent Commands
13. E4.3 - Task Commands
14. E4.4 - Server Commands

### Phase 4: Sandbox & Security
15. E5.1 - Process Sandbox
16. E5.2 - Docker Sandbox
17. E1.4 - Integration Tests

### Phase 5: Framework Expansion
18. E3.1 - OpenAI Assistants Adapter
19. E3.2 - AutoGen Adapter
20. E2.5 - Mistral Provider
21. E2.6 - Ollama Provider

### Phase 6: Polish
22. E6.1-6.4 - Observability
23. E8.1-8.4 - Documentation
24. E7.2 - Docker Support

---

## Definition of Done

A story is complete when:
- [ ] Code implemented and follows existing patterns
- [ ] Unit tests written (90%+ coverage)
- [ ] Documentation updated (docstrings, README if needed)
- [ ] No regressions in existing tests
- [ ] Code reviewed (self-review checklist)
- [ ] Committed with conventional commit message
