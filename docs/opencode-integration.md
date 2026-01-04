# OpenCode & Claude Integration Guide

This guide explains how to integrate AI Workflow Fabric (AWF) with OpenCode, Claude, and other MCP-compatible AI assistants.

## Overview

AWF provides an **MCP (Model Context Protocol) server** that allows AI assistants to:

- **Register agents** from any framework (LangGraph, CrewAI, AutoGen)
- **Create workflows** that chain multiple agents together
- **Execute workflows** and get real-time results
- **Monitor trust scores** and security policies
- **Debug failures** with detailed execution traces

## Quick Setup

### 1. Install AWF

```bash
pip install ai-workflow-fabric[api,cli]
```

### 2. Configure MCP Server

Add AWF to your MCP configuration file:

**For OpenCode** (`opencode.json` or `.opencode/mcp.json`):

```json
{
  "mcpServers": {
    "awf": {
      "command": "python",
      "args": ["-m", "awf.mcp.server"]
    }
  }
}
```

**For Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "awf": {
      "command": "python",
      "args": ["-m", "awf.mcp.server"]
    }
  }
}
```

### 3. Verify Installation

After configuring, you should see AWF tools available:

- `awf_register_agent` - Register AI agents
- `awf_list_agents` - List registered agents
- `awf_create_workflow` - Create multi-agent workflows
- `awf_execute_workflow` - Run workflows
- `awf_get_trust_score` - Check agent trust scores

## Available MCP Tools

### Agent Management

| Tool | Description |
|------|-------------|
| `awf_register_agent` | Register a new agent with capabilities |
| `awf_list_agents` | List all agents, filter by capability/framework |
| `awf_get_agent` | Get detailed agent information |
| `awf_get_trust_score` | Get trust score and sandbox tier |
| `awf_delete_agent` | Remove an agent from registry |

### Workflow Management

| Tool | Description |
|------|-------------|
| `awf_create_workflow` | Create a workflow definition |
| `awf_list_workflows` | List all workflows |
| `awf_get_workflow` | Get workflow details |
| `awf_execute_workflow` | Run a workflow with input |
| `awf_get_execution` | Get execution status and results |
| `awf_delete_workflow` | Remove a workflow |

## Example Conversations

### Register an Agent

```
User: Register a web search agent that can search the web and summarize results