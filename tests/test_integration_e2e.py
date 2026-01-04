"""
AI Workflow Fabric - End-to-End Integration Tests

These tests verify the complete workflow of AWF with real LLM calls.
They use the Antigravity provider (Google OAuth) for free model access.

To run these tests:
    # First authenticate with Google (one-time setup)
    python -c "from awf.providers.antigravity import AntigravityProvider; import asyncio; asyncio.run(AntigravityProvider().authenticate())"
    
    # Run integration tests
    pytest tests/test_integration_e2e.py -v -m integration
    
    # Run all tests including integration
    pytest tests/ -v --run-integration
"""

from __future__ import annotations

import asyncio
import os
import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from awf.core.types import (
    AgentManifest,
    AgentStatus,
    Capability,
    CapabilityType,
    Task,
    TaskResult,
    TaskStatus,
)
from awf.registry.memory import InMemoryRegistry
from awf.security.trust import TrustScoringEngine
from awf.security.policy import PolicyEngine
from awf.orchestration.types import (
    WorkflowDefinition,
    StepDefinition,
    RetryPolicy,
    WorkflowResult,
    ExecutionStatus,
    StepStatus,
)
from awf.orchestration.orchestrator import Orchestrator, OrchestratorConfig
from awf.orchestration.registry import OrchestrationAdapterRegistry


# =============================================================================
# Test Configuration
# =============================================================================


def has_antigravity_tokens() -> bool:
    """Check if Antigravity tokens are available."""
    token_path = os.path.expanduser("~/.config/awf/antigravity-tokens.json")
    return os.path.exists(token_path)


def has_openai_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


def has_anthropic_key() -> bool:
    """Check if Anthropic API key is available."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


# Skip markers for tests requiring authentication
skip_without_antigravity = pytest.mark.skipif(
    not has_antigravity_tokens(),
    reason="Antigravity tokens not found - run: python -c \"from awf.providers.antigravity import AntigravityProvider; import asyncio; asyncio.run(AntigravityProvider().authenticate())\""
)

skip_without_openai = pytest.mark.skipif(
    not has_openai_key(),
    reason="OPENAI_API_KEY not set"
)

skip_without_anthropic = pytest.mark.skipif(
    not has_anthropic_key(),
    reason="ANTHROPIC_API_KEY not set"
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def registry():
    """Create an in-memory registry."""
    return InMemoryRegistry()


@pytest.fixture
def trust_engine():
    """Create a trust scoring engine."""
    return TrustScoringEngine()


@pytest.fixture
def policy_engine():
    """Create a policy engine."""
    return PolicyEngine()


@pytest.fixture
def adapter_registry():
    """Create an adapter registry."""
    return OrchestrationAdapterRegistry()


@pytest.fixture
def orchestrator(adapter_registry, registry):
    """Create an orchestrator."""
    return Orchestrator(
        adapter_registry=adapter_registry,
        agent_registry=registry,
        config=OrchestratorConfig(
            default_timeout_ms=60000,
            max_parallel_steps=5,
            emit_events=True,
        ),
    )


@pytest.fixture
def sample_agent() -> AgentManifest:
    """Create a sample agent manifest."""
    return AgentManifest(
        id="test-llm-agent",
        name="Test LLM Agent",
        version="1.0.0",
        framework="custom",
        capabilities=[
            Capability(
                name="text_generation",
                type=CapabilityType.REASONING,
                description="Generate text responses",
            ),
            Capability(
                name="summarization",
                type=CapabilityType.REASONING,
                description="Summarize content",
            ),
        ],
        description="A test agent for E2E testing",
        status=AgentStatus.ACTIVE,
    )


# =============================================================================
# Unit Tests (Always Run)
# =============================================================================


class TestRegistryIntegration:
    """Test registry operations."""
    
    @pytest.mark.asyncio
    async def test_register_and_retrieve_agent(self, registry, sample_agent):
        """Test agent registration and retrieval."""
        await registry.register(sample_agent)
        
        retrieved = await registry.get(sample_agent.id)
        assert retrieved is not None
        assert retrieved.id == sample_agent.id
        assert retrieved.name == sample_agent.name
    
    @pytest.mark.asyncio
    async def test_search_by_capability(self, registry, sample_agent):
        """Test searching agents by capability."""
        await registry.register(sample_agent)
        
        results = await registry.search(capabilities=["text_generation"])
        assert len(results) == 1
        assert results[0].id == sample_agent.id
    
    @pytest.mark.asyncio
    async def test_search_by_framework(self, registry, sample_agent):
        """Test searching agents by framework."""
        await registry.register(sample_agent)
        
        results = await registry.search(framework="custom")
        assert len(results) == 1
        
        results = await registry.search(framework="langgraph")
        assert len(results) == 0


class TestTrustScoringIntegration:
    """Test trust scoring operations."""
    
    @pytest.mark.asyncio
    async def test_compute_trust_score(self, trust_engine, sample_agent):
        """Test trust score computation."""
        score = await trust_engine.compute_score(sample_agent)
        
        assert score.score >= 0.0
        assert score.score <= 1.0
        assert score.sandbox_tier is not None
    
    @pytest.mark.asyncio
    async def test_trust_score_factors(self, trust_engine, sample_agent):
        """Test that trust factors are populated."""
        score = await trust_engine.compute_score(sample_agent)
        
        assert score.factors is not None
        assert hasattr(score.factors, 'publisher_trust')
        assert hasattr(score.factors, 'audit_status')
        assert hasattr(score.factors, 'community_trust')
        assert hasattr(score.factors, 'permission_analysis')
        assert hasattr(score.factors, 'historical_behavior')


class TestPolicyIntegration:
    """Test policy engine operations."""
    
    @pytest.mark.asyncio
    async def test_evaluate_policy(self, policy_engine, sample_agent):
        """Test policy evaluation."""
        task = Task(
            agent_id=sample_agent.id,
            input={"query": "test"},
        )
        
        result = policy_engine.evaluate(
            manifest=sample_agent,
            task=task,
            environment="development",
        )
        
        assert result.allowed is True
    
    @pytest.mark.asyncio
    async def test_policy_with_trust_score(self, policy_engine, trust_engine, sample_agent):
        """Test policy evaluation with trust score."""
        trust = await trust_engine.compute_score(sample_agent)
        
        task = Task(
            agent_id=sample_agent.id,
            input={"query": "test"},
        )
        
        result = policy_engine.evaluate(
            manifest=sample_agent,
            task=task,
            trust_score=trust,
            environment="production",
        )
        
        # Result depends on trust score threshold
        assert result is not None


class TestWorkflowDefinition:
    """Test workflow definition creation."""
    
    def test_create_simple_workflow(self):
        """Test creating a simple workflow."""
        workflow = WorkflowDefinition(
            id="test-workflow",
            name="Test Workflow",
            steps=[
                StepDefinition(
                    id="step1",
                    agent_id="agent-a",
                ),
            ],
        )
        
        assert workflow.id == "test-workflow"
        assert len(workflow.steps) == 1
    
    def test_create_workflow_with_dependencies(self):
        """Test creating a workflow with step dependencies."""
        workflow = WorkflowDefinition(
            id="dependent-workflow",
            name="Dependent Workflow",
            steps=[
                StepDefinition(id="step1", agent_id="agent-a"),
                StepDefinition(id="step2", agent_id="agent-b", depends_on=["step1"]),
                StepDefinition(id="step3", agent_id="agent-c", depends_on=["step1", "step2"]),
            ],
        )
        
        assert len(workflow.steps) == 3
        assert workflow.steps[1].depends_on == ["step1"]
        assert workflow.steps[2].depends_on == ["step1", "step2"]
    
    def test_create_workflow_with_retry_policy(self):
        """Test creating a workflow with retry policy."""
        workflow = WorkflowDefinition(
            id="retry-workflow",
            name="Retry Workflow",
            steps=[
                StepDefinition(
                    id="step1",
                    agent_id="agent-a",
                    retry=RetryPolicy(
                        max_attempts=3,
                        backoff_ms=1000,
                        backoff_multiplier=2.0,
                    ),
                ),
            ],
        )
        
        assert workflow.steps[0].retry is not None
        assert workflow.steps[0].retry.max_attempts == 3


# =============================================================================
# Real LLM Integration Tests - Antigravity (Primary - FREE)
# =============================================================================


@pytest.mark.integration
class TestAntigravityIntegration:
    """Integration tests with Antigravity API (Claude/Gemini via Google OAuth)."""
    
    @skip_without_antigravity
    @pytest.mark.asyncio
    async def test_antigravity_claude_completion(self):
        """Test Antigravity Claude completion with real API call."""
        from awf.providers.antigravity import AntigravityProvider
        from awf.providers.base import Message, Role
        
        provider = AntigravityProvider()
        
        try:
            response = await provider.complete(
                messages=[
                    Message(role=Role.USER, content="What is 2+2? Reply with just the number."),
                ],
                model="antigravity-claude-sonnet-4-5",
                max_tokens=50,
            )
            
            assert response.content is not None
            assert len(response.content) > 0
            assert "4" in response.content
            assert response.usage is not None
            
            print(f"\nClaude response: {response.content}")
            print(f"Tokens used: {response.usage.total_tokens}")
        finally:
            await provider.close()
    
    @skip_without_antigravity
    @pytest.mark.asyncio
    async def test_antigravity_claude_with_thinking(self):
        """Test Antigravity Claude with thinking enabled."""
        from awf.providers.antigravity import AntigravityProvider
        from awf.providers.base import Message, Role
        
        provider = AntigravityProvider()
        
        try:
            response = await provider.complete(
                messages=[
                    Message(role=Role.USER, content="What is the square root of 144? Think step by step."),
                ],
                model="antigravity-claude-sonnet-4-5-thinking-low",
                max_tokens=500,
            )
            
            assert response.content is not None
            assert len(response.content) > 0
            assert "12" in response.content
            
            print(f"\nClaude thinking response: {response.content[:200]}...")
        finally:
            await provider.close()
    
    @skip_without_antigravity
    @pytest.mark.asyncio
    async def test_antigravity_gemini_completion(self):
        """Test Antigravity Gemini completion with real API call."""
        from awf.providers.antigravity import AntigravityProvider
        from awf.providers.base import Message, Role
        
        provider = AntigravityProvider()
        
        try:
            response = await provider.complete(
                messages=[
                    Message(role=Role.USER, content="List 3 programming languages. Be brief."),
                ],
                model="antigravity-gemini-3-flash",
                max_tokens=100,
            )
            
            assert response.content is not None
            assert len(response.content) > 0
            
            print(f"\nGemini response: {response.content}")
        finally:
            await provider.close()
    
    @skip_without_antigravity
    @pytest.mark.asyncio
    async def test_antigravity_streaming(self):
        """Test Antigravity streaming with real API call."""
        from awf.providers.antigravity import AntigravityProvider
        from awf.providers.base import Message, Role
        
        provider = AntigravityProvider()
        
        try:
            chunks = []
            async for chunk in provider.stream(
                messages=[
                    Message(role=Role.USER, content="Count from 1 to 5."),
                ],
                model="antigravity-claude-sonnet-4-5",
                max_tokens=50,
            ):
                chunks.append(chunk)
            
            assert len(chunks) > 0
            assert chunks[-1].is_final is True
            assert chunks[-1].accumulated_content is not None
            
            print(f"\nStreamed content: {chunks[-1].accumulated_content}")
        finally:
            await provider.close()
    
    @skip_without_antigravity
    @pytest.mark.asyncio
    async def test_antigravity_opus_thinking(self):
        """Test Antigravity with high thinking - using Sonnet as backup if Opus is rate limited."""
        from awf.providers.antigravity import AntigravityProvider
        from awf.providers.base import Message, Role
        
        provider = AntigravityProvider()
        
        try:
            # Try Opus first, fall back to Sonnet if rate limited
            models_to_try = [
                "antigravity-claude-opus-4-5-thinking-high",
                "antigravity-claude-sonnet-4-5-thinking-high",
            ]
            
            response = None
            for model in models_to_try:
                try:
                    response = await provider.complete(
                        messages=[
                            Message(
                                role=Role.USER, 
                                content="Write a haiku about programming. Just the haiku, no explanation."
                            ),
                        ],
                        model=model,
                        max_tokens=500,
                    )
                    print(f"\nUsed model: {model}")
                    break
                except RuntimeError as e:
                    if "Rate limited" in str(e) and model != models_to_try[-1]:
                        print(f"\n{model} rate limited, trying fallback...")
                        continue
                    raise
            
            assert response is not None
            assert response.content is not None
            assert len(response.content) > 0
            
            print(f"\nHaiku: {response.content}")
        finally:
            await provider.close()


@pytest.mark.integration
class TestFullWorkflowE2E:
    """End-to-end tests with complete workflow execution."""
    
    @skip_without_antigravity
    @pytest.mark.asyncio
    async def test_research_summarize_workflow(self, registry, trust_engine):
        """Test a complete research and summarize workflow using Antigravity."""
        from awf.providers.antigravity import AntigravityProvider
        from awf.providers.base import Message, Role
        
        provider = AntigravityProvider()
        
        try:
            # Step 1: Research (get facts)
            research_response = await provider.complete(
                messages=[
                    Message(role=Role.USER, content="Tell me 3 brief facts about Python programming language."),
                ],
                model="antigravity-claude-sonnet-4-5",
                max_tokens=300,
            )
            
            assert research_response.content is not None
            research_facts = research_response.content
            print(f"\n[Step 1 - Research]: {research_facts[:200]}...")
            
            # Step 2: Summarize the research
            summary_response = await provider.complete(
                messages=[
                    Message(role=Role.USER, content=f"Summarize in one sentence: {research_facts}"),
                ],
                model="antigravity-claude-sonnet-4-5",
                max_tokens=100,
            )
            
            assert summary_response.content is not None
            assert len(summary_response.content) > 0
            print(f"[Step 2 - Summary]: {summary_response.content}")
            
            # Both steps completed successfully
            print("\nWorkflow completed successfully!")
        finally:
            await provider.close()
    
    @skip_without_antigravity
    @pytest.mark.asyncio
    async def test_multi_model_workflow(self, registry):
        """Test a workflow using multiple models from Antigravity."""
        from awf.providers.antigravity import AntigravityProvider
        from awf.providers.base import Message, Role
        
        provider = AntigravityProvider()
        
        try:
            # Step 1: Fast response with Gemini Flash (needs more tokens for internal thinking)
            fast_response = await provider.complete(
                messages=[
                    Message(role=Role.USER, content="What is the capital of France? One word answer."),
                ],
                model="antigravity-gemini-3-flash",
                max_tokens=100,  # Gemini uses ~17 tokens for thinking
            )
            
            assert fast_response.content is not None
            assert "Paris" in fast_response.content
            print(f"\n[Gemini Flash]: {fast_response.content}")
            
            # Step 2: Deeper analysis with Claude
            analysis_response = await provider.complete(
                messages=[
                    Message(role=Role.USER, content=f"Why is {fast_response.content.strip()} significant historically? 2 sentences max."),
                ],
                model="antigravity-claude-sonnet-4-5",
                max_tokens=150,
            )
            
            assert analysis_response.content is not None
            print(f"[Claude Sonnet]: {analysis_response.content}")
            
            print("\nMulti-model workflow completed!")
        finally:
            await provider.close()


# =============================================================================
# MCP Server Integration Tests
# =============================================================================


class TestMCPServerIntegration:
    """Test MCP server functionality."""
    
    @pytest.mark.asyncio
    async def test_mcp_server_initialization(self):
        """Test MCP server can be initialized."""
        from awf.mcp.server import AWFMCPServer
        
        server = AWFMCPServer(use_local=True)
        
        assert server._tools is not None
        assert len(server._tools) > 0
        assert "awf_register_agent" in server._tools
        assert "awf_create_workflow" in server._tools
    
    @pytest.mark.asyncio
    async def test_mcp_register_agent_tool(self):
        """Test MCP register agent tool."""
        from awf.mcp.server import AWFMCPServer
        
        server = AWFMCPServer(use_local=True)
        
        result = await server._handle_register_agent(
            id="mcp-test-agent",
            name="MCP Test Agent",
            framework="custom",
            capabilities=[
                {"name": "test_cap", "type": "tool", "description": "A test capability"},
            ],
        )
        
        assert result["success"] is True
        assert result["agent_id"] == "mcp-test-agent"
        assert "trust_score" in result
    
    @pytest.mark.asyncio
    async def test_mcp_list_agents_tool(self):
        """Test MCP list agents tool."""
        from awf.mcp.server import AWFMCPServer
        
        server = AWFMCPServer(use_local=True)
        
        # Register an agent first
        await server._handle_register_agent(
            id="list-test-agent",
            name="List Test Agent",
            framework="langgraph",
            capabilities=[{"name": "cap1", "type": "tool"}],
        )
        
        result = await server._handle_list_agents()
        
        assert "agents" in result
        assert len(result["agents"]) >= 1
    
    @pytest.mark.asyncio
    async def test_mcp_create_workflow_tool(self):
        """Test MCP create workflow tool."""
        from awf.mcp.server import AWFMCPServer
        
        server = AWFMCPServer(use_local=True)
        
        # Register agents first
        await server._handle_register_agent(
            id="wf-agent-1",
            name="Workflow Agent 1",
            framework="custom",
            capabilities=[{"name": "cap1", "type": "tool"}],
        )
        
        result = await server._handle_create_workflow(
            id="mcp-test-workflow",
            name="MCP Test Workflow",
            steps=[
                {"id": "step1", "agent_id": "wf-agent-1"},
            ],
        )
        
        assert result["success"] is True
        assert result["workflow_id"] == "mcp-test-workflow"
        assert result["step_count"] == 1
    
    @pytest.mark.asyncio
    async def test_mcp_handle_request(self):
        """Test MCP protocol request handling."""
        from awf.mcp.server import AWFMCPServer
        
        server = AWFMCPServer(use_local=True)
        
        # Test initialize
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        })
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["protocolVersion"] == "2024-11-05"
        
        # Test tools/list
        response = await server.handle_request({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        })
        
        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) > 0


# =============================================================================
# API Integration Tests
# =============================================================================


class TestAPIIntegration:
    """Test API endpoints integration."""
    
    @pytest.mark.asyncio
    async def test_api_health_check(self):
        """Test API health check endpoint."""
        from fastapi.testclient import TestClient
        from awf.api.app import create_app
        
        app = create_app(debug=True)
        
        with TestClient(app) as client:
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_api_agent_crud(self):
        """Test API agent CRUD operations."""
        from fastapi.testclient import TestClient
        from awf.api.app import create_app
        
        app = create_app(debug=True)
        
        with TestClient(app) as client:
            # Create agent
            response = client.post("/agents", json={
                "id": "api-test-agent",
                "name": "API Test Agent",
                "version": "1.0.0",
                "framework": "custom",
                "capabilities": [
                    {"name": "test", "type": "tool"},
                ],
            })
            
            assert response.status_code == 201
            data = response.json()
            assert data["id"] == "api-test-agent"
            
            # Get agent
            response = client.get("/agents/api-test-agent")
            assert response.status_code == 200
            
            # Delete agent
            response = client.delete("/agents/api-test-agent")
            assert response.status_code == 204


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestPerformance:
    """Performance and load tests."""
    
    @pytest.mark.asyncio
    async def test_registry_bulk_operations(self, registry):
        """Test registry performance with bulk operations."""
        import time
        
        # Register 100 agents
        start = time.time()
        for i in range(100):
            agent = AgentManifest(
                id=f"perf-agent-{i}",
                name=f"Performance Agent {i}",
                version="1.0.0",
                framework="custom",
                capabilities=[Capability(name=f"cap_{i}", type=CapabilityType.TOOL)],
                status=AgentStatus.ACTIVE,
            )
            await registry.register(agent)
        
        register_time = time.time() - start
        
        # Search operations
        start = time.time()
        for i in range(100):
            await registry.search(framework="custom")
        
        search_time = time.time() - start
        
        print(f"\nBulk register (100 agents): {register_time:.3f}s")
        print(f"Bulk search (100 queries): {search_time:.3f}s")
        
        assert register_time < 5.0  # Should complete in under 5 seconds
        assert search_time < 5.0
    
    @skip_without_antigravity
    @pytest.mark.asyncio
    async def test_concurrent_antigravity_calls(self):
        """Test concurrent Antigravity API calls."""
        from awf.providers.antigravity import AntigravityProvider
        from awf.providers.base import Message, Role
        import time
        
        provider = AntigravityProvider()
        
        async def make_call(i: int):
            return await provider.complete(
                messages=[
                    Message(role=Role.USER, content=f"Reply with just the number: {i}"),
                ],
                model="antigravity-gemini-3-flash",  # Fast model for concurrency test
                max_tokens=10,
            )
        
        try:
            start = time.time()
            results = await asyncio.gather(*[make_call(i) for i in range(3)])
            elapsed = time.time() - start
            
            print(f"\n3 concurrent calls completed in: {elapsed:.2f}s")
            
            assert len(results) == 3
            assert all(r.content is not None for r in results)
        finally:
            await provider.close()


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require API keys",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="Need --run-integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
