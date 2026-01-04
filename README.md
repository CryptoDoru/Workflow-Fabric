# AI Workflow Fabric (AWF)

**"Kubernetes for AI Agents"**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/ai-workflow-fabric.svg)](https://pypi.org/project/ai-workflow-fabric/)
[![Tests](https://github.com/CryptoDoru/Workflow-Fabric/actions/workflows/ci.yml/badge.svg)](https://github.com/CryptoDoru/Workflow-Fabric/actions)
[![codecov](https://codecov.io/gh/CryptoDoru/Workflow-Fabric/branch/main/graph/badge.svg)](https://codecov.io/gh/CryptoDoru/Workflow-Fabric)
[![ASP Version](https://img.shields.io/badge/ASP-v1.0--draft-green.svg)](spec/asp-specification.md)

<p align="center">
  <strong>Orchestrate LangGraph, CrewAI, AutoGen, and OpenAI agents with unified observability, trust scoring, and production reliability.</strong>
</p>

---

## Why AWF?

The AI agent ecosystem is fragmented. Each framework has its own API, execution model, and tooling:

| Framework | Mental Model | Strength |
|-----------|--------------|----------|
| **LangGraph** | Graphs & State | Complex workflows |
| **CrewAI** | Roles & Teams | Agent collaboration |
| **AutoGen** | Conversations | Multi-agent chat |
| **OpenAI Assistants** | Threads | Simple deployment |

**AWF bridges them all** with:

- **Unified Registry**: Discover agents from any framework
- **Multi-Framework Orchestration**: Mix LangGraph + CrewAI + AutoGen in one workflow
- **Production Reliability**: Retries, fallbacks, timeouts, circuit breakers
- **Trust & Security**: Trust scoring, sandboxed execution, policy enforcement
- **Full Observability**: Grafana dashboards, distributed tracing, cost tracking
- **Auto-Remediation**: Watcher Agent monitors and fixes issues automatically

---

## Quick Start

### Installation

```bash
# Basic installation
pip install ai-workflow-fabric

# With API server
pip install ai-workflow-fabric[api]

# With observability (OpenTelemetry + Grafana)
pip install ai-workflow-fabric[otel]

# With everything
pip install ai-workflow-fabric[all]
```

### 5-Minute Demo

```python
import asyncio
from awf.core.types import AgentManifest, Capability, CapabilityType, AgentStatus
from awf.registry.memory import InMemoryRegistry
from awf.security.trust import TrustScoringEngine

async def main():
    # 1. Create registry and trust engine
    registry = InMemoryRegistry()
    trust_engine = TrustScoringEngine()
    
    # 2. Define an agent
    agent = AgentManifest(
        id="research-agent",
        name="Research Agent",
        version="1.0.0",
        framework="langgraph",
        capabilities=[
            Capability(name="web_search", type=CapabilityType.TOOL),
            Capability(name="summarize", type=CapabilityType.REASONING),
        ],
        status=AgentStatus.ACTIVE,
    )
    
    # 3. Compute trust score
    trust = await trust_engine.compute_score(agent)
    print(f"Trust: {trust.score:.2f} -> Sandbox: {trust.sandbox_tier.value}")
    
    # 4. Register and discover
    await registry.register(agent)
    found = await registry.search(capabilities=["web_search"])
    print(f"Found {len(found)} agents with web_search capability")

asyncio.run(main())
```

### REST API

```bash
# Start the API server
pip install ai-workflow-fabric[api]
uvicorn awf.api.app:app --reload

# Register an agent
curl -X POST http://localhost:8000/agents \
  -H "Content-Type: application/json" \
  -d '{
    "id": "my-agent",
    "name": "My Agent", 
    "version": "1.0.0",
    "framework": "langgraph",
    "capabilities": [{"name": "search", "type": "tool"}]
  }'

# Check trust score
curl http://localhost:8000/agents/my-agent/trust

# Create a workflow
curl -X POST http://localhost:8000/workflows \
  -H "Content-Type: application/json" \
  -d '{
    "id": "research-flow",
    "name": "Research Pipeline",
    "steps": [
      {"id": "search", "agent_id": "searcher"},
      {"id": "summarize", "agent_id": "summarizer", "depends_on": ["search"]}
    ]
  }'
```

### CLI

```bash
# Install with CLI support
pip install ai-workflow-fabric[cli]

# Manage agents
awf agents list
awf agents register manifest.json
awf agents trust my-agent

# Run workflows
awf run workflow.yaml --input '{"query": "AI safety"}'

# Start API server
awf server start --reload
```

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
                    ┌──────────────────────┼────────────────────┼──────────────┐
                    │                 OTel │       Webhooks     │ MCP          │
                    ▼                      ▼                    ▼              │
         ┌───────────────────────────────────────────────────────────────────┐ │
         │                    Grafana LGTM Stack                              │ │
         │   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐          │ │
         │   │  Alloy  │──▶│  Mimir  │   │  Loki   │   │  Tempo  │          │ │
         │   │ (OTel)  │   │(Metrics)│   │ (Logs)  │   │(Traces) │          │ │
         │   └─────────┘   └────┬────┘   └────┬────┘   └────┬────┘          │ │
         │                      └─────────────┴─────────────┘               │ │
         │                                 │                                 │ │
         │                         ┌───────▼───────┐                        │ │
         │                         │    Grafana    │                        │ │
         │                         │  Dashboards   │                        │ │
         │                         └───────────────┘                        │ │
         └───────────────────────────────────────────────────────────────────┘ │
                                                                               │
┌──────────────────────────────────────────────────────────────────────────────┘
│
│  Framework Adapters
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  │  LangGraph  │    │   CrewAI    │    │   AutoGen   │    │   OpenAI    │
│  │   Adapter   │    │   Adapter   │    │   Adapter   │    │   Adapter   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## Key Features

### Trust Scoring

AWF computes trust scores for agents based on 5 factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| Publisher Trust | 25% | Verified publisher, reputation |
| Audit Status | 25% | Security audits, findings |
| Community Trust | 20% | Usage, ratings, reports |
| Permission Analysis | 15% | Requested permissions risk |
| Historical Behavior | 15% | Past execution history |

Based on trust score, agents are assigned to sandbox tiers:

| Score Range | Sandbox Tier | Overhead |
|-------------|--------------|----------|
| 0.90 - 1.00 | WASM | ~10ms |
| 0.70 - 0.89 | gVisor | ~100ms |
| 0.40 - 0.69 | gVisor Strict | ~150ms |
| 0.00 - 0.39 | BLOCKED | N/A |

### Workflow Orchestration

Define multi-agent workflows with DAG-based execution:

```yaml
# workflow.yaml
id: research-pipeline
name: Research Pipeline
steps:
  - id: search
    agent_id: web-search-agent
    timeout_ms: 30000
    retry:
      max_attempts: 3
      backoff_multiplier: 2.0

  - id: analyze
    agent_id: analysis-agent
    depends_on: [search]
    
  - id: summarize
    agent_id: summarizer-agent
    depends_on: [analyze]
    fallback_agent_id: backup-summarizer
```

### Observability with Grafana

Full LGTM stack (Loki, Grafana, Tempo, Mimir) for enterprise observability:

```bash
# Start the observability stack
cd docker/grafana
docker compose up -d

# Open Grafana
open http://localhost:3000  # admin/admin
```

**Included Dashboards:**
- **AWF Overview**: Workflow executions, success rates, latency
- **Agent Fleet Health**: Per-agent metrics, trust scores, resource usage
- **Cost Tracking**: Token usage and costs per agent/workflow

### Watcher Agent (Auto-Remediation)

The Watcher Agent monitors your agent fleet and automatically fixes issues:

```python
from awf.agents.watcher import WatcherAgent, WatcherConfig

# Configure the watcher
config = WatcherConfig(
    grafana_url="http://localhost:3000",
    investigation_timeout=300,
    auto_approve_low_risk=True,  # Automatically fix low-risk issues
)

watcher = WatcherAgent(config)

# Process an alert (usually via webhook)
result = await watcher.handle_alert(grafana_alert)
print(f"Recommended: {result.recommended_action}")
print(f"Risk: {result.risk_level}")
```

**Built-in Remediation Scripts:**
- `restart_agent`: Restart a failed agent
- `retry_workflow`: Retry a failed workflow with backoff
- `increase_timeout`: Adjust timeout for slow agents
- `disable_agent`: Disable a problematic agent
- `notify_oncall`: Escalate to human operator

---

## LLM Provider Support

AWF includes unified LLM providers for all major services:

```python
from awf.providers import OpenAIProvider, AnthropicProvider, Message, Role

# Use any provider with the same interface
provider = OpenAIProvider()  # or AnthropicProvider(), GoogleProvider(), etc.

response = await provider.complete([
    Message(role=Role.USER, content="Hello!")
])

print(f"Response: {response.content}")
print(f"Cost: ${response.usage.total_cost:.6f}")
```

**Supported Providers:**
- OpenAI (GPT-4, GPT-4o, etc.)
- Anthropic (Claude 3.5, Claude 3)
- Google (Gemini Pro, Gemini Flash)
- Mistral (Mistral Large, Codestral)
- Ollama (Local models)

---

## Project Status

**Current Phase:** v1.0.0-alpha

| Component | Status |
|-----------|--------|
| ASP Specification | ✅ Complete |
| Core Types | ✅ Complete |
| LangGraph Adapter | ✅ Complete |
| CrewAI Adapter | ✅ Complete |
| AutoGen Adapter | ✅ Complete |
| In-Memory Registry | ✅ Complete |
| SQLite Registry | ✅ Complete |
| Trust Engine | ✅ Complete |
| Policy Engine | ✅ Complete |
| Sandbox Orchestrator | ✅ Complete |
| REST API | ✅ Complete |
| Workflow Orchestrator | ✅ Complete |
| Grafana Integration | ✅ Complete |
| Watcher Agent | ✅ Complete |
| LLM Providers | ✅ Complete |
| Test Suite | ✅ 534 tests |
| CLI Tool | ✅ Complete |
| Web UI | 🚧 Planned |

---

## Documentation

- **[ASP Specification](spec/asp-specification.md)**: The Agent State Protocol
- **[HTTP Binding](spec/asp-http-binding.md)**: REST API specification
- **[Trust Scoring](spec/trust-scoring.md)**: How agents are evaluated
- **[Deploying with Grafana](docs/deploying-with-grafana.md)**: Observability setup
- **[Watcher Agent Guide](docs/watcher-agent-guide.md)**: Auto-remediation configuration
- **[Grafana Dashboards](docs/grafana-dashboards.md)**: Dashboard usage guide

---

## Examples

See the `examples/` directory:

| Example | Description |
|---------|-------------|
| [quickstart.py](examples/quickstart.py) | Basic usage demo |
| [llm_providers.py](examples/llm_providers.py) | LLM provider examples |
| [research_pipeline/](examples/research_pipeline/) | Multi-agent research workflow |
| [code_review/](examples/code_review/) | Code analysis pipeline |
| [customer_support/](examples/customer_support/) | Support ticket routing |

```bash
# Run the quickstart
python examples/quickstart.py

# Run with observability
cd docker/grafana && docker compose up -d
python examples/research_pipeline/main.py
```

---

## Package Structure

```
awf/
├── core/              # Core ASP types and protocols
├── adapters/          # Framework adapters (LangGraph, CrewAI, AutoGen)
├── registry/          # Agent registry (memory, SQLite)
├── security/          # Trust scoring, policy engine, sandbox
├── orchestration/     # Workflow DAG execution
├── providers/         # LLM providers (OpenAI, Anthropic, etc.)
├── agents/            # Built-in agents (Watcher)
├── api/               # FastAPI REST API
└── cli/               # Typer CLI
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/CryptoDoru/Workflow-Fabric.git
cd Workflow-Fabric

# Install all dependencies
pip install -e ".[dev,api,cli]"

# Run tests
pytest

# Run linting
ruff check awf tests

# Type checking
mypy awf
```

### Running the Full Stack

```bash
# Terminal 1: Start observability
cd docker/grafana && docker compose up -d

# Terminal 2: Start API server
uvicorn awf.api.app:app --reload

# Terminal 3: Run tests
pytest --cov=awf
```

---

## Roadmap

### v1.0.0 (Current)
- [x] Core registry and trust engine
- [x] Framework adapters (LangGraph, CrewAI, AutoGen)
- [x] REST API with FastAPI
- [x] Workflow orchestration
- [x] Grafana observability
- [x] Watcher agent

### v1.1.0 (Q1 2025)
- [ ] Grafana MCP Server integration
- [ ] LLM-powered root cause analysis
- [ ] Slack/PagerDuty integrations
- [ ] Web UI for workflow builder

### v2.0.0 (Q2 2025)
- [ ] Distributed execution
- [ ] Kubernetes operator
- [ ] Multi-tenant SaaS mode

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

## Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/CryptoDoru/Workflow-Fabric/issues)
- **Discussions**: [Join the conversation](https://github.com/CryptoDoru/Workflow-Fabric/discussions)

---

<p align="center">
  <strong>Built with love for the AI agent community</strong>
</p>
