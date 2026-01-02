# AI Workflow Fabric (AWF)

**"Kubernetes for AI Agents"**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![ASP Version](https://img.shields.io/badge/ASP-v1.0--draft-green.svg)](spec/asp-specification.md)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)

---

## Overview

AI Workflow Fabric (AWF) is an open-source middleware abstraction layer designed to orchestrate and federate AI agents, tools, and data flows across the fragmented LLM-assisted coding and agentic AI ecosystem.

**Key Features:**

- **Unified Agent Registry**: Register and discover agents from any framework (LangGraph, CrewAI, AutoGen, OpenAI)
- **Multi-Framework Orchestration**: Compose workflows using agents from different frameworks
- **Built-in Reliability**: Automatic retries, fallbacks, timeouts, and error handling
- **Trust & Security**: Trust scoring, capability-based permissions, and sandboxed execution
- **Full Observability**: Event streaming, distributed tracing, and cost tracking

---

## Installation

```bash
# Basic installation
pip install awf

# With API server support
pip install awf[api]

# With all framework adapters
pip install awf[langgraph,crewai]

# Development installation
pip install awf[dev]
```

---

## Quick Start

### Python API

```python
import asyncio
from awf.core.types import AgentManifest, Capability, CapabilityType, AgentStatus
from awf.registry.memory import InMemoryRegistry
from awf.security.trust import TrustScoringEngine

async def main():
    # Create registry and trust engine
    registry = InMemoryRegistry()
    trust_engine = TrustScoringEngine()
    
    # Create an agent manifest
    agent = AgentManifest(
        id="my-search-agent",
        name="Search Agent",
        version="1.0.0",
        framework="langgraph",
        capabilities=[
            Capability(
                name="web_search",
                type=CapabilityType.TOOL,
                description="Search the web",
            ),
        ],
        status=AgentStatus.ACTIVE,
    )
    
    # Compute trust score
    trust = await trust_engine.compute_score(agent)
    print(f"Trust Score: {trust.score:.2f}")
    print(f"Sandbox Tier: {trust.sandbox_tier.value}")
    
    # Register the agent
    await registry.register(agent)
    
    # Search for agents
    results = await registry.search(capabilities=["web_search"])
    for result in results:
        print(f"Found: {result.name}")

asyncio.run(main())
```

### REST API

```bash
# Start the API server
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

# Search for agents
curl "http://localhost:8000/agents?capabilities=search"

# Get agent trust score
curl http://localhost:8000/agents/my-agent/trust
```

---

## The Problem

The AI agent ecosystem is fragmented:

- **LangGraph** thinks in graphs and state
- **CrewAI** thinks in roles and teams
- **AutoGen** thinks in conversations
- **OpenAI Assistants** thinks in threads

Each framework has its own API, execution model, and tooling. Connecting them requires custom glue code. Production reliability is hard. There's no standard way to discover, trust, or orchestrate agents.

---

## The Solution

AWF provides:

1. **Agent State Protocol (ASP)**: An open standard for agent interoperability
2. **Framework Adapters**: Translate between ASP and native frameworks
3. **Orchestration Engine**: Compose multi-agent workflows with reliability
4. **Security Layer**: Trust scoring and sandboxed execution

```
┌─────────────────────────────────────────────────────────────────┐
│                        YOUR APPLICATION                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AWF ORCHESTRATOR                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Registry  │  │   Router    │  │  Trust & Security       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │                  │                    │
         ▼                  ▼                    ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐
│  LangGraph  │    │   CrewAI    │    │        AutoGen          │
│   Adapter   │    │   Adapter   │    │        Adapter          │
└─────────────┘    └─────────────┘    └─────────────────────────┘
```

---

## Trust Scoring

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

---

## Project Status

**Current Phase:** v1.0.0-alpha (Core Implementation)

| Component | Status |
|-----------|--------|
| ASP Specification | ✅ Complete |
| JSON Schemas | ✅ Complete |
| HTTP Binding | ✅ Complete |
| Trust Algorithm | ✅ Complete |
| Core Types | ✅ Complete |
| LangGraph Adapter | ✅ Complete |
| CrewAI Adapter | ✅ Complete |
| In-Memory Registry | ✅ Complete |
| SQLite Registry | ✅ Complete |
| Trust Engine | ✅ Complete |
| Policy Engine | ✅ Complete |
| Sandbox Orchestrator | ✅ Complete |
| REST API | ✅ Complete |
| Test Suite | ✅ Complete |
| CLI Tool | 🚧 Planned |
| Web UI | 🚧 Planned |

---

## Documentation

- **[ASP Specification](spec/asp-specification.md)**: The Agent State Protocol
- **[HTTP Binding](spec/asp-http-binding.md)**: REST API specification
- **[Trust Scoring](spec/trust-scoring.md)**: How agents are evaluated
- **[User Stories](docs/user-stories.md)**: Detailed requirements

---

## Examples

See the `examples/` directory for complete examples:

- **[quickstart.py](examples/quickstart.py)**: Basic usage demonstration

```bash
# Run the quickstart example
python examples/quickstart.py
```

---

## Architecture

AWF follows a **protocol-first** approach:

1. **ASP Core** (Required): Task, Result, Status, Events
2. **ASP Extended** (Optional): Memory, Messaging
3. **Transport Bindings**: HTTP/REST, gRPC, WebSocket

This separation allows different implementations to interoperate as long as they conform to ASP.

### Package Structure

```
awf/
├── core/           # Core ASP types and protocols
│   └── types.py    # Dataclasses for Manifest, Task, Result, etc.
├── adapters/       # Framework adapters
│   ├── base.py     # Abstract adapter interface
│   ├── langgraph/  # LangGraph adapter
│   └── crewai/     # CrewAI adapter
├── registry/       # Agent registry implementations
│   ├── memory.py   # In-memory registry
│   ├── persistence.py  # SQLite registry
│   └── search.py   # Capability search engine
├── security/       # Security and trust
│   ├── trust.py    # Trust scoring engine
│   ├── policy.py   # Policy enforcement
│   └── sandbox.py  # Sandbox orchestrator
└── api/            # REST API
    ├── app.py      # FastAPI application
    └── models.py   # Pydantic models
```

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Protocol First | ASP spec before code | Establish standard for ecosystem |
| Sandbox Everything | All agents sandboxed | Consistent security guarantees |
| Fully Open Source | Apache 2.0 | Maximum adoption, community trust |
| Bootstrap Funding | Self-funded | Quality over speed |

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/CryptoDoru/Workflow-Fabric.git
cd Workflow-Fabric

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Type checking
mypy awf
```

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

## Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/CryptoDoru/Workflow-Fabric/issues)
- **Discussions**: [Join the conversation](https://github.com/CryptoDoru/Workflow-Fabric/discussions)
