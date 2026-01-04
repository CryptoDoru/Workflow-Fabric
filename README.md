# AI Workflow Fabric (AWF)

**"Kubernetes for AI Agents"**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/ai-workflow-fabric.svg)](https://pypi.org/project/ai-workflow-fabric/)
[![Tests](https://github.com/CryptoDoru/Workflow-Fabric/actions/workflows/ci.yml/badge.svg)](https://github.com/CryptoDoru/Workflow-Fabric/actions)
[![codecov](https://codecov.io/gh/CryptoDoru/Workflow-Fabric/branch/main/graph/badge.svg)](https://codecov.io/gh/CryptoDoru/Workflow-Fabric)

---

## What is AWF?

**AI Workflow Fabric (AWF)** is an open-source middleware that lets you **orchestrate AI agents from any framework** (LangGraph, CrewAI, AutoGen, OpenAI) through a single, unified platform.

Think of it as **Kubernetes for AI Agents**: just as Kubernetes manages containers regardless of where they came from, AWF manages AI agents regardless of which framework created them.

### What Problems Does It Solve?

| Problem | How AWF Solves It |
|---------|-------------------|
| **Framework Lock-in** | Use LangGraph for complex workflows, CrewAI for team collaboration, and AutoGen for conversations—all in the same pipeline |
| **No Visibility** | Full observability with Grafana dashboards showing metrics, logs, and traces for every agent execution |
| **Unreliable in Production** | Built-in retries, fallbacks, timeouts, and circuit breakers for production-grade reliability |
| **Security Concerns** | Trust scoring system evaluates every agent before execution; untrusted agents run in sandboxes |
| **Manual Incident Response** | Watcher Agent automatically detects issues and remediates them (restart agents, retry workflows, adjust timeouts) |
| **Scattered Agent Discovery** | Central registry to discover, search, and manage all your agents in one place |

---

## Who Is This For?

| User | Use Case |
|------|----------|
| **AI/ML Teams** | Manage agents from multiple frameworks without custom glue code |
| **Platform Engineers** | Get enterprise observability and reliability for AI workloads |
| **Startups** | Ship production-ready multi-agent products faster |
| **Enterprises** | Security, compliance, and governance for AI agents at scale |

---

## Quick Start (5 Minutes)

### 1. Install

```bash
pip install ai-workflow-fabric[api]
```

### 2. Start the API Server

```bash
uvicorn awf.api.app:app --reload
```

Open http://localhost:8000/docs to see the interactive API documentation.

### 3. Register Your First Agent

```bash
curl -X POST http://localhost:8000/agents \
  -H "Content-Type: application/json" \
  -d '{
    "id": "my-search-agent",
    "name": "Web Search Agent",
    "version": "1.0.0",
    "framework": "langgraph",
    "capabilities": [
      {"name": "web_search", "type": "tool"},
      {"name": "summarize", "type": "reasoning"}
    ]
  }'
```

### 4. Check Its Trust Score

```bash
curl http://localhost:8000/agents/my-search-agent/trust
```

Response:
```json
{
  "score": 0.54,
  "sandbox_tier": "gvisor_strict",
  "factors": {
    "publisher_trust": 0.50,
    "audit_status": 0.30,
    "community_trust": 0.50,
    "permission_analysis": 0.90,
    "historical_behavior": 0.70
  }
}
```

### 5. Create a Multi-Agent Workflow

```bash
curl -X POST http://localhost:8000/workflows \
  -H "Content-Type: application/json" \
  -d '{
    "id": "research-pipeline",
    "name": "Research Pipeline",
    "steps": [
      {"id": "search", "agentId": "my-search-agent", "timeoutMs": 30000},
      {"id": "analyze", "agentId": "analysis-agent", "dependsOn": ["search"]},
      {"id": "report", "agentId": "report-agent", "dependsOn": ["analyze"]}
    ]
  }'
```

### 6. Execute the Workflow

```bash
curl -X POST http://localhost:8000/workflows/research-pipeline/execute \
  -H "Content-Type: application/json" \
  -d '{"input": {"query": "AI safety best practices"}}'
```

---

## Common Use Cases

### Research Pipeline
Chain agents to search → analyze → summarize information:

```python
from awf.orchestration import WorkflowDefinition, StepDefinition

workflow = WorkflowDefinition(
    id="research-pipeline",
    name="Research Pipeline",
    steps=[
        StepDefinition(id="search", agent_id="web-search-agent"),
        StepDefinition(id="analyze", agent_id="analysis-agent", depends_on=["search"]),
        StepDefinition(id="summarize", agent_id="summarizer-agent", depends_on=["analyze"]),
    ]
)
```

### Code Review Pipeline
Scan code for security issues and quality problems:

```python
workflow = WorkflowDefinition(
    id="code-review",
    name="Code Review",
    steps=[
        StepDefinition(id="parse", agent_id="code-parser"),
        StepDefinition(id="security", agent_id="security-scanner", depends_on=["parse"]),
        StepDefinition(id="quality", agent_id="quality-reviewer", depends_on=["parse"]),
        StepDefinition(id="report", agent_id="report-generator", depends_on=["security", "quality"]),
    ]
)
```

### Customer Support Routing
Classify tickets and route to specialist agents:

```python
workflow = WorkflowDefinition(
    id="support-routing",
    name="Support Ticket Router",
    steps=[
        StepDefinition(id="classify", agent_id="ticket-classifier"),
        StepDefinition(id="handle", agent_id="support-agent", depends_on=["classify"]),
        StepDefinition(id="respond", agent_id="response-generator", depends_on=["handle"]),
    ]
)
```

See the complete examples in the [`examples/`](examples/) directory.

---

## Installation Options

```bash
# Core only (registry, trust engine, types)
pip install ai-workflow-fabric

# With REST API server
pip install ai-workflow-fabric[api]

# With CLI tool
pip install ai-workflow-fabric[cli]

# With OpenTelemetry observability
pip install ai-workflow-fabric[otel]

# With LLM providers (OpenAI, Anthropic, etc.)
pip install ai-workflow-fabric[providers]

# With specific framework adapters
pip install ai-workflow-fabric[langgraph]
pip install ai-workflow-fabric[crewai]
pip install ai-workflow-fabric[autogen]

# Everything
pip install ai-workflow-fabric[all]
```

---

## Core Concepts

### Agents
An **Agent** is any AI component that can perform tasks. AWF doesn't care what framework it's built with—it just needs a manifest describing its capabilities.

```python
from awf.core.types import AgentManifest, Capability, CapabilityType, AgentStatus

agent = AgentManifest(
    id="my-agent",
    name="My Agent",
    version="1.0.0",
    framework="langgraph",  # or "crewai", "autogen", "openai", "custom"
    capabilities=[
        Capability(name="web_search", type=CapabilityType.TOOL),
        Capability(name="summarize", type=CapabilityType.REASONING),
    ],
    status=AgentStatus.ACTIVE,
)
```

### Registry
The **Registry** stores and discovers agents. Search by capability, framework, tags, or trust score.

```python
from awf.registry.memory import InMemoryRegistry

registry = InMemoryRegistry()
await registry.register(agent)

# Find agents that can search the web
results = await registry.search(capabilities=["web_search"])

# Find high-trust LangGraph agents
results = await registry.search(framework="langgraph", min_trust_score=0.8)
```

### Trust Scoring
Every agent gets a **trust score** (0.0 - 1.0) based on:
- Publisher reputation
- Security audit status
- Community feedback
- Permissions requested
- Historical behavior

Low-trust agents run in stricter sandboxes. Blocked agents (score < 0.4) can't execute at all.

```python
from awf.security.trust import TrustScoringEngine

trust_engine = TrustScoringEngine()
score = await trust_engine.compute_score(agent)

print(f"Trust: {score.score:.2f}")
print(f"Sandbox: {score.sandbox_tier.value}")
```

### Workflows
A **Workflow** chains multiple agents together with dependencies, retries, and fallbacks.

```python
from awf.orchestration import WorkflowDefinition, StepDefinition, RetryPolicy

workflow = WorkflowDefinition(
    id="my-workflow",
    name="My Workflow",
    steps=[
        StepDefinition(
            id="step1",
            agent_id="agent-a",
            timeout_ms=30000,
            retry=RetryPolicy(max_attempts=3, backoff_multiplier=2.0),
        ),
        StepDefinition(
            id="step2",
            agent_id="agent-b",
            depends_on=["step1"],
        ),
    ]
)
```

### Policies
**Policies** control what agents can do in different environments.

```python
from awf.security.policy import PolicyEngine

policy_engine = PolicyEngine()
policy_engine.create_default_policies()

# Check if agent can run in production
result = policy_engine.evaluate(
    manifest=agent,
    task=task,
    trust_score=score,
    environment="production",
)

if result.allowed:
    print("Agent approved for production")
else:
    print(f"Denied: {result.violations}")
```

---

## Observability with Grafana

AWF includes a complete observability stack using Grafana's LGTM (Loki, Grafana, Tempo, Mimir).

### Start the Stack

```bash
cd docker/grafana
docker compose up -d
```

### Access Grafana

Open http://localhost:3000 (username: `admin`, password: `admin`)

### Included Dashboards

| Dashboard | What It Shows |
|-----------|---------------|
| **AWF Overview** | Workflow executions, success rates, latency percentiles, error rates |
| **Agent Fleet Health** | Per-agent metrics, trust scores, resource usage, failure rates |
| **Cost Tracking** | Token usage and estimated costs per agent and workflow |

### Automatic Alerting

The Watcher Agent receives Grafana alerts and can automatically:
- Restart failed agents
- Retry failed workflows
- Adjust timeouts
- Disable problematic agents
- Escalate to humans

See [docs/deploying-with-grafana.md](docs/deploying-with-grafana.md) for full setup instructions.

---

## CLI Reference

```bash
# Show version
awf version

# Agents
awf agents list                    # List all agents
awf agents get <id>                # Get agent details
awf agents register manifest.json  # Register from file
awf agents trust <id>              # Show trust score
awf agents delete <id>             # Delete agent

# Workflows
awf workflows list                 # List workflows
awf workflows run workflow.yaml    # Execute workflow
awf workflows status <id>          # Check execution status
awf workflows create --name "X"    # Create template

# Server
awf server start                   # Start API server
awf server start --reload          # With auto-reload
awf server health                  # Check server health
```

---

## API Reference

The REST API runs on port 8000 by default. Full OpenAPI documentation is available at `/docs`.

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/agents` | Register an agent |
| `GET` | `/agents` | List/search agents |
| `GET` | `/agents/{id}` | Get agent details |
| `GET` | `/agents/{id}/trust` | Get trust score |
| `DELETE` | `/agents/{id}` | Unregister agent |
| `POST` | `/workflows` | Create workflow |
| `GET` | `/workflows` | List workflows |
| `POST` | `/workflows/{id}/execute` | Execute workflow |
| `GET` | `/executions/{id}` | Get execution status |
| `POST` | `/webhooks/grafana-alerts` | Receive Grafana alerts |
| `GET` | `/watcher/approvals` | List pending approvals |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            YOUR APPLICATION                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AWF REST API (:8000)                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────────────┐   │
│  │   Registry   │ │    Trust     │ │   Workflow   │ │   Watcher Agent   │   │
│  │              │ │    Engine    │ │ Orchestrator │ │ (Auto-Remediation)│   │
│  └──────────────┘ └──────────────┘ └──────┬───────┘ └─────────┬─────────┘   │
└──────────────────────────────────────────┼────────────────────┼─────────────┘
                                           │                    │
                                      OTel │      Webhooks      │
                                           ▼                    ▼
                              ┌─────────────────────────────────────────────┐
                              │           Grafana LGTM Stack                │
                              │  Alloy → Mimir/Loki/Tempo → Grafana         │
                              └─────────────────────────────────────────────┘
                                                    │
                    ┌───────────────────────────────┼───────────────────────────┐
                    │                               │                           │
                    ▼                               ▼                           ▼
           ┌─────────────┐                 ┌─────────────┐              ┌─────────────┐
           │  LangGraph  │                 │   CrewAI    │              │   AutoGen   │
           │   Adapter   │                 │   Adapter   │              │   Adapter   │
           └─────────────┘                 └─────────────┘              └─────────────┘
```

---

## Examples

| Example | Description | Run Command |
|---------|-------------|-------------|
| [quickstart.py](examples/quickstart.py) | Basic registry, trust, and policy demo | `python examples/quickstart.py` |
| [llm_providers.py](examples/llm_providers.py) | Using OpenAI, Anthropic, etc. | `python examples/llm_providers.py` |
| [research_pipeline/](examples/research_pipeline/) | Multi-agent research workflow | `python examples/research_pipeline/main.py` |
| [code_review/](examples/code_review/) | Security and quality code review | `python examples/code_review/main.py` |
| [customer_support/](examples/customer_support/) | Ticket classification and routing | `python examples/customer_support/main.py` |

---

## LLM Providers

AWF includes unified providers for major LLM services:

```python
from awf.providers import OpenAIProvider, Message, Role

provider = OpenAIProvider()

response = await provider.complete([
    Message(role=Role.SYSTEM, content="You are a helpful assistant."),
    Message(role=Role.USER, content="What is 2+2?"),
])

print(response.content)  # "4"
print(f"Cost: ${response.usage.total_cost:.6f}")
```

**Supported:** OpenAI, Anthropic, Google (Gemini), Mistral, Ollama (local)

---

## Project Status

| Component | Status |
|-----------|--------|
| Core Types & Registry | ✅ Complete |
| Trust & Policy Engine | ✅ Complete |
| Framework Adapters | ✅ LangGraph, CrewAI, AutoGen |
| Workflow Orchestrator | ✅ Complete |
| REST API | ✅ Complete |
| Grafana Observability | ✅ Complete |
| Watcher Agent | ✅ Complete |
| LLM Providers | ✅ Complete |
| CLI Tool | ✅ Complete |
| Test Suite | ✅ 534 tests |
| Web UI | 🚧 Planned |

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Clone
git clone https://github.com/CryptoDoru/Workflow-Fabric.git
cd Workflow-Fabric

# Install dev dependencies
pip install -e ".[dev,api,cli]"

# Run tests
pytest

# Run linting
ruff check awf tests
```

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

## Links

- **GitHub**: https://github.com/CryptoDoru/Workflow-Fabric
- **Issues**: https://github.com/CryptoDoru/Workflow-Fabric/issues
- **Discussions**: https://github.com/CryptoDoru/Workflow-Fabric/discussions

---

<p align="center">
  <strong>Built for teams who need AI agents to work together reliably in production.</strong>
</p>
