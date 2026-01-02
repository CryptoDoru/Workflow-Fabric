# AI Workflow Fabric (AWF)

**"Kubernetes for AI Agents"**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![ASP Version](https://img.shields.io/badge/ASP-v1.0--draft-green.svg)](spec/asp-specification.md)

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

## Quick Start

```bash
# Install AWF
pip install awf

# Register an agent
awf agent register ./my-agent

# Run a workflow
awf workflow run my-workflow --input data="hello world"
```

---

## Documentation

- **[ASP Specification](spec/asp-specification.md)**: The Agent State Protocol
- **[HTTP Binding](spec/asp-http-binding.md)**: REST API specification
- **[Trust Scoring](spec/trust-scoring.md)**: How agents are evaluated
- **[User Stories](docs/user-stories.md)**: Detailed requirements

---

## Project Status

**Current Phase:** v1.0.0-draft (Protocol Specification)

| Component | Status |
|-----------|--------|
| ASP Specification | Draft Complete |
| JSON Schemas | Draft Complete |
| HTTP Binding | Draft Complete |
| Trust Algorithm | Draft Complete |
| LangGraph Adapter | In Progress |
| CLI Tool | Planned |
| Web UI | Planned |

---

## Architecture

AWF follows a **protocol-first** approach:

1. **ASP Core** (Required): Task, Result, Status, Events
2. **ASP Extended** (Optional): Memory, Messaging
3. **Transport Bindings**: HTTP/REST, gRPC, WebSocket

This separation allows different implementations to interoperate as long as they conform to ASP.

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
```

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

## Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/CryptoDoru/Workflow-Fabric/issues)
- **Discussions**: [Join the conversation](https://github.com/CryptoDoru/Workflow-Fabric/discussions)
