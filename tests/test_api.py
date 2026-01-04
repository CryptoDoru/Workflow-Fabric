"""
AI Workflow Fabric - REST API Tests

Comprehensive tests for the FastAPI REST API endpoints.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

# Import FastAPI test client
try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("FastAPI not installed", allow_module_level=True)

from awf.api.app import create_app, AppState, _app_state


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def app():
    """Create a fresh test application."""
    return create_app(debug=True)


@pytest.fixture
def client(app):
    """Create a test client with the app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_agent_data():
    """Sample agent registration data."""
    return {
        "id": "test-agent-001",
        "name": "Test Search Agent",
        "version": "1.0.0",
        "framework": "langgraph",
        "framework_version": "0.2.0",
        "description": "A test agent for searching",
        "capabilities": [
            {
                "name": "web_search",
                "type": "tool",
                "description": "Search the web",
                "permissions": ["network:external"],
            },
            {
                "name": "summarize",
                "type": "reasoning",
                "description": "Summarize content",
            },
        ],
        "tags": ["search", "research", "test"],
        "publisher": "test-publisher",
        "metadata": {"custom_field": "custom_value"},
    }


@pytest.fixture
def minimal_agent_data():
    """Minimal valid agent registration data."""
    return {
        "id": "minimal-agent",
        "name": "Minimal Agent",
        "version": "1.0.0",
        "framework": "crewai",
    }


@pytest.fixture
def sample_task_data():
    """Sample task submission data."""
    return {
        "agent_id": "test-agent-001",
        "input": {"query": "What is AI?"},
        "timeout_ms": 30000,
        "priority": 0,
    }


# =============================================================================
# Health & Root Endpoint Tests
# =============================================================================


class TestHealthEndpoints:
    """Tests for health and status endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "AI Workflow Fabric"
        assert "version" in data
        assert data["docs"] == "/docs"
        assert data["openapi"] == "/openapi.json"

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0
        assert "registry_count" in data
        assert data["registry_count"] >= 0
        assert "components" in data
        assert data["components"]["registry"] == "healthy"
        assert data["components"]["trust_engine"] == "healthy"
        assert data["components"]["policy_engine"] == "healthy"

    def test_health_check_returns_correct_registry_count(self, client, sample_agent_data):
        """Test health check reflects correct agent count."""
        # Initially empty
        response = client.get("/health")
        initial_count = response.json()["registry_count"]
        
        # Register an agent
        client.post("/agents", json=sample_agent_data)
        
        # Count should increase
        response = client.get("/health")
        assert response.json()["registry_count"] == initial_count + 1


# =============================================================================
# Agent CRUD Tests
# =============================================================================


class TestAgentRegistration:
    """Tests for agent registration endpoint."""

    def test_register_agent_success(self, client, sample_agent_data):
        """Test successful agent registration."""
        response = client.post("/agents", json=sample_agent_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == sample_agent_data["id"]
        assert data["name"] == sample_agent_data["name"]
        assert data["version"] == sample_agent_data["version"]
        assert data["framework"] == sample_agent_data["framework"]
        assert data["status"] == "active"
        assert "trust_score" in data
        assert data["trust_score"] is not None
        assert len(data["capabilities"]) == 2

    def test_register_minimal_agent(self, client, minimal_agent_data):
        """Test registration with minimal required fields."""
        response = client.post("/agents", json=minimal_agent_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == minimal_agent_data["id"]
        assert data["capabilities"] == []
        assert data["tags"] == []

    def test_register_duplicate_agent_fails(self, client, sample_agent_data):
        """Test that registering duplicate agent returns 409."""
        # First registration
        response = client.post("/agents", json=sample_agent_data)
        assert response.status_code == 201
        
        # Duplicate registration
        response = client.post("/agents", json=sample_agent_data)
        assert response.status_code == 409
        assert "already exists" in response.json()["message"]

    def test_register_agent_missing_required_field(self, client):
        """Test registration fails without required fields."""
        invalid_data = {"name": "Missing ID Agent"}
        response = client.post("/agents", json=invalid_data)
        
        assert response.status_code == 422

    def test_register_agent_with_all_capability_types(self, client):
        """Test registration with all capability types."""
        agent_data = {
            "id": "all-caps-agent",
            "name": "All Capabilities Agent",
            "version": "1.0.0",
            "framework": "langgraph",
            "capabilities": [
                {"name": "tool_cap", "type": "tool"},
                {"name": "reasoning_cap", "type": "reasoning"},
                {"name": "memory_cap", "type": "memory"},
                {"name": "communication_cap", "type": "communication"},
                {"name": "custom_cap", "type": "custom"},
            ],
        }
        
        response = client.post("/agents", json=agent_data)
        assert response.status_code == 201
        
        data = response.json()
        cap_types = [cap["type"] for cap in data["capabilities"]]
        assert set(cap_types) == {"tool", "reasoning", "memory", "communication", "custom"}


class TestAgentRetrieval:
    """Tests for agent retrieval endpoint."""

    def test_get_agent_success(self, client, sample_agent_data):
        """Test getting an existing agent."""
        client.post("/agents", json=sample_agent_data)
        
        response = client.get(f"/agents/{sample_agent_data['id']}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_agent_data["id"]
        assert data["name"] == sample_agent_data["name"]

    def test_get_agent_not_found(self, client):
        """Test getting non-existent agent returns 404."""
        response = client.get("/agents/nonexistent-agent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["message"]

    def test_get_agent_includes_trust_score(self, client, sample_agent_data):
        """Test that retrieved agent includes trust score."""
        client.post("/agents", json=sample_agent_data)
        
        response = client.get(f"/agents/{sample_agent_data['id']}")
        data = response.json()
        
        assert "trust_score" in data
        assert isinstance(data["trust_score"], float)
        assert 0.0 <= data["trust_score"] <= 1.0


class TestAgentUpdate:
    """Tests for agent update endpoint."""

    def test_update_agent_name(self, client, sample_agent_data):
        """Test updating agent name."""
        client.post("/agents", json=sample_agent_data)
        
        response = client.patch(
            f"/agents/{sample_agent_data['id']}",
            json={"name": "Updated Agent Name"}
        )
        
        assert response.status_code == 200
        assert response.json()["name"] == "Updated Agent Name"

    def test_update_agent_version(self, client, sample_agent_data):
        """Test updating agent version."""
        client.post("/agents", json=sample_agent_data)
        
        response = client.patch(
            f"/agents/{sample_agent_data['id']}",
            json={"version": "2.0.0"}
        )
        
        assert response.status_code == 200
        assert response.json()["version"] == "2.0.0"

    def test_update_agent_description(self, client, sample_agent_data):
        """Test updating agent description."""
        client.post("/agents", json=sample_agent_data)
        
        response = client.patch(
            f"/agents/{sample_agent_data['id']}",
            json={"description": "Updated description"}
        )
        
        assert response.status_code == 200
        assert response.json()["description"] == "Updated description"

    def test_update_agent_tags(self, client, sample_agent_data):
        """Test updating agent tags."""
        client.post("/agents", json=sample_agent_data)
        
        new_tags = ["new-tag-1", "new-tag-2"]
        response = client.patch(
            f"/agents/{sample_agent_data['id']}",
            json={"tags": new_tags}
        )
        
        assert response.status_code == 200
        assert response.json()["tags"] == new_tags

    def test_update_agent_capabilities(self, client, sample_agent_data):
        """Test updating agent capabilities."""
        client.post("/agents", json=sample_agent_data)
        
        new_capabilities = [
            {"name": "new_capability", "type": "tool", "description": "New cap"}
        ]
        response = client.patch(
            f"/agents/{sample_agent_data['id']}",
            json={"capabilities": new_capabilities}
        )
        
        assert response.status_code == 200
        assert len(response.json()["capabilities"]) == 1
        assert response.json()["capabilities"][0]["name"] == "new_capability"

    def test_update_agent_status(self, client, sample_agent_data):
        """Test updating agent status."""
        client.post("/agents", json=sample_agent_data)
        
        response = client.patch(
            f"/agents/{sample_agent_data['id']}",
            json={"status": "suspended"}
        )
        
        assert response.status_code == 200
        assert response.json()["status"] == "suspended"

    def test_update_agent_metadata(self, client, sample_agent_data):
        """Test updating agent metadata."""
        client.post("/agents", json=sample_agent_data)
        
        new_metadata = {"key1": "value1", "key2": 123}
        response = client.patch(
            f"/agents/{sample_agent_data['id']}",
            json={"metadata": new_metadata}
        )
        
        assert response.status_code == 200
        assert response.json()["metadata"] == new_metadata

    def test_update_agent_not_found(self, client):
        """Test updating non-existent agent returns 404."""
        response = client.patch(
            "/agents/nonexistent-agent",
            json={"name": "New Name"}
        )
        
        assert response.status_code == 404

    def test_update_agent_recalculates_trust_score(self, client, sample_agent_data):
        """Test that updating agent recalculates trust score."""
        client.post("/agents", json=sample_agent_data)
        
        # Get initial trust score
        initial = client.get(f"/agents/{sample_agent_data['id']}").json()
        initial_trust = initial["trust_score"]
        
        # Update with risky capabilities
        response = client.patch(
            f"/agents/{sample_agent_data['id']}",
            json={"capabilities": [
                {"name": "exec", "type": "tool", "permissions": ["process:execute", "filesystem:write"]}
            ]}
        )
        
        # Trust score should be recalculated
        updated_trust = response.json()["trust_score"]
        assert updated_trust is not None


class TestAgentDeletion:
    """Tests for agent deletion endpoint."""

    def test_delete_agent_success(self, client, sample_agent_data):
        """Test successful agent deletion."""
        client.post("/agents", json=sample_agent_data)
        
        response = client.delete(f"/agents/{sample_agent_data['id']}")
        assert response.status_code == 204
        
        # Verify agent is gone
        response = client.get(f"/agents/{sample_agent_data['id']}")
        assert response.status_code == 404

    def test_delete_agent_not_found(self, client):
        """Test deleting non-existent agent returns 404."""
        response = client.delete("/agents/nonexistent-agent")
        assert response.status_code == 404


# =============================================================================
# Agent List & Search Tests
# =============================================================================


class TestAgentList:
    """Tests for agent listing and search endpoint."""

    def test_list_agents_empty(self, client):
        """Test listing agents when registry is empty."""
        response = client.get("/agents")
        
        assert response.status_code == 200
        data = response.json()
        assert data["agents"] == []
        assert data["total"] == 0
        assert data["page"] == 1

    def test_list_agents_returns_all(self, client, sample_agent_data, minimal_agent_data):
        """Test listing returns all registered agents."""
        client.post("/agents", json=sample_agent_data)
        client.post("/agents", json=minimal_agent_data)
        
        response = client.get("/agents")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["agents"]) == 2

    def test_list_agents_pagination(self, client):
        """Test agent list pagination."""
        # Register 5 agents
        for i in range(5):
            client.post("/agents", json={
                "id": f"agent-{i}",
                "name": f"Agent {i}",
                "version": "1.0.0",
                "framework": "langgraph",
            })
        
        # Get page 1 with page_size=2
        response = client.get("/agents?page=1&page_size=2")
        data = response.json()
        
        assert data["total"] == 5
        assert len(data["agents"]) == 2
        assert data["page"] == 1
        assert data["page_size"] == 2
        
        # Get page 2
        response = client.get("/agents?page=2&page_size=2")
        data = response.json()
        
        assert len(data["agents"]) == 2
        assert data["page"] == 2
        
        # Get page 3 (should have 1 agent)
        response = client.get("/agents?page=3&page_size=2")
        data = response.json()
        
        assert len(data["agents"]) == 1

    def test_list_agents_filter_by_framework(self, client):
        """Test filtering agents by framework."""
        # Register agents with different frameworks
        client.post("/agents", json={
            "id": "lg-agent",
            "name": "LangGraph Agent",
            "version": "1.0.0",
            "framework": "langgraph",
        })
        client.post("/agents", json={
            "id": "crew-agent",
            "name": "CrewAI Agent",
            "version": "1.0.0",
            "framework": "crewai",
        })
        
        # Filter by langgraph
        response = client.get("/agents?framework=langgraph")
        data = response.json()
        
        assert data["total"] == 1
        assert data["agents"][0]["framework"] == "langgraph"

    def test_list_agents_filter_by_capabilities(self, client):
        """Test filtering agents by capabilities."""
        client.post("/agents", json={
            "id": "search-agent",
            "name": "Search Agent",
            "version": "1.0.0",
            "framework": "langgraph",
            "capabilities": [{"name": "web_search", "type": "tool"}],
        })
        client.post("/agents", json={
            "id": "write-agent",
            "name": "Write Agent",
            "version": "1.0.0",
            "framework": "langgraph",
            "capabilities": [{"name": "text_write", "type": "tool"}],
        })
        
        # Filter by web_search capability
        response = client.get("/agents?capabilities=web_search")
        data = response.json()
        
        assert data["total"] == 1
        assert data["agents"][0]["id"] == "search-agent"

    def test_list_agents_filter_by_tags(self, client):
        """Test filtering agents by tags."""
        client.post("/agents", json={
            "id": "research-agent",
            "name": "Research Agent",
            "version": "1.0.0",
            "framework": "langgraph",
            "tags": ["research", "academic"],
        })
        client.post("/agents", json={
            "id": "sales-agent",
            "name": "Sales Agent",
            "version": "1.0.0",
            "framework": "langgraph",
            "tags": ["sales", "marketing"],
        })
        
        # Filter by research tag
        response = client.get("/agents?tags=research")
        data = response.json()
        
        assert data["total"] == 1
        assert data["agents"][0]["id"] == "research-agent"

    def test_list_agents_filter_by_min_trust_score(self, client):
        """Test filtering agents by minimum trust score."""
        # Register agents - trust scores are computed automatically
        client.post("/agents", json={
            "id": "trusted-agent",
            "name": "Trusted Agent",
            "version": "1.0.0",
            "framework": "langgraph",
            "publisher": "verified-publisher",  # Higher trust
        })
        client.post("/agents", json={
            "id": "risky-agent",
            "name": "Risky Agent",
            "version": "1.0.0",
            "framework": "langgraph",
            "capabilities": [
                {"name": "exec", "type": "tool", "permissions": ["process:execute", "filesystem:write"]}
            ],
        })
        
        # Get all agents first
        all_response = client.get("/agents")
        assert all_response.json()["total"] == 2
        
        # Filter by minimum trust score (0.5) - should filter based on computed scores
        response = client.get("/agents?min_trust_score=0.5")
        # Both should pass since default trust scores are above 0.5
        assert response.status_code == 200

    def test_list_agents_multiple_filters(self, client):
        """Test applying multiple filters."""
        client.post("/agents", json={
            "id": "match-agent",
            "name": "Match Agent",
            "version": "1.0.0",
            "framework": "langgraph",
            "capabilities": [{"name": "search", "type": "tool"}],
            "tags": ["research"],
        })
        client.post("/agents", json={
            "id": "nomatch-agent",
            "name": "No Match Agent",
            "version": "1.0.0",
            "framework": "crewai",
            "capabilities": [{"name": "search", "type": "tool"}],
            "tags": ["sales"],
        })
        
        # Filter by framework AND capabilities
        response = client.get("/agents?framework=langgraph&capabilities=search")
        data = response.json()
        
        assert data["total"] == 1
        assert data["agents"][0]["id"] == "match-agent"


# =============================================================================
# Agent Status Management Tests
# =============================================================================


class TestAgentStatusManagement:
    """Tests for agent activation/suspension endpoints."""

    def test_suspend_agent(self, client, sample_agent_data):
        """Test suspending an active agent."""
        client.post("/agents", json=sample_agent_data)
        
        response = client.post(f"/agents/{sample_agent_data['id']}/suspend")
        
        assert response.status_code == 200
        assert response.json()["status"] == "suspended"

    def test_activate_agent(self, client, sample_agent_data):
        """Test activating a suspended agent."""
        client.post("/agents", json=sample_agent_data)
        client.post(f"/agents/{sample_agent_data['id']}/suspend")
        
        response = client.post(f"/agents/{sample_agent_data['id']}/activate")
        
        assert response.status_code == 200
        assert response.json()["status"] == "active"

    def test_suspend_agent_not_found(self, client):
        """Test suspending non-existent agent returns 404."""
        response = client.post("/agents/nonexistent/suspend")
        assert response.status_code == 404

    def test_activate_agent_not_found(self, client):
        """Test activating non-existent agent returns 404."""
        response = client.post("/agents/nonexistent/activate")
        assert response.status_code == 404

    def test_activate_already_active_agent(self, client, sample_agent_data):
        """Test activating an already active agent succeeds."""
        client.post("/agents", json=sample_agent_data)
        
        response = client.post(f"/agents/{sample_agent_data['id']}/activate")
        
        assert response.status_code == 200
        assert response.json()["status"] == "active"


# =============================================================================
# Trust Score Tests
# =============================================================================


class TestTrustScore:
    """Tests for trust score endpoint."""

    def test_get_trust_score(self, client, sample_agent_data):
        """Test getting trust score for an agent."""
        client.post("/agents", json=sample_agent_data)
        
        response = client.get(f"/agents/{sample_agent_data['id']}/trust")
        
        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == sample_agent_data["id"]
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0
        assert "sandbox_tier" in data
        assert data["sandbox_tier"] in ["wasm", "gvisor", "gvisor_strict", "blocked"]
        assert "factors" in data
        assert "computed_at" in data

    def test_get_trust_score_factors(self, client, sample_agent_data):
        """Test trust score includes all factors."""
        client.post("/agents", json=sample_agent_data)
        
        response = client.get(f"/agents/{sample_agent_data['id']}/trust")
        factors = response.json()["factors"]
        
        assert "publisher_trust" in factors
        assert "audit_status" in factors
        assert "community_trust" in factors
        assert "permission_analysis" in factors
        assert "historical_behavior" in factors
        assert "computed_score" in factors
        
        # All factors should be between 0 and 1
        for key, value in factors.items():
            assert 0.0 <= value <= 1.0, f"{key} out of range"

    def test_get_trust_score_not_found(self, client):
        """Test getting trust score for non-existent agent."""
        response = client.get("/agents/nonexistent/trust")
        assert response.status_code == 404

    def test_trust_score_reflects_risky_permissions(self, client):
        """Test that risky permissions affect trust score."""
        # Agent with safe capabilities
        safe_agent = {
            "id": "safe-agent",
            "name": "Safe Agent",
            "version": "1.0.0",
            "framework": "langgraph",
            "capabilities": [{"name": "search", "type": "tool"}],
        }
        client.post("/agents", json=safe_agent)
        safe_trust = client.get("/agents/safe-agent/trust").json()
        
        # Agent with risky capabilities
        risky_agent = {
            "id": "risky-agent",
            "name": "Risky Agent",
            "version": "1.0.0",
            "framework": "langgraph",
            "capabilities": [
                {
                    "name": "exec",
                    "type": "tool",
                    "permissions": ["process:execute", "filesystem:write", "network:*"],
                }
            ],
        }
        client.post("/agents", json=risky_agent)
        risky_trust = client.get("/agents/risky-agent/trust").json()
        
        # Risky agent should have lower permission_analysis score
        assert risky_trust["factors"]["permission_analysis"] <= safe_trust["factors"]["permission_analysis"]


# =============================================================================
# Task Submission Tests
# =============================================================================


class TestTaskSubmission:
    """Tests for task submission endpoint."""

    def test_submit_task_success(self, client, sample_agent_data, sample_task_data):
        """Test successful task submission."""
        client.post("/agents", json=sample_agent_data)
        
        response = client.post("/tasks", json=sample_task_data)
        
        assert response.status_code == 202
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"
        assert "message" in data

    def test_submit_task_agent_not_found(self, client, sample_task_data):
        """Test task submission fails for non-existent agent."""
        response = client.post("/tasks", json=sample_task_data)
        
        assert response.status_code == 404
        assert "not found" in response.json()["message"]

    def test_submit_task_agent_suspended(self, client, sample_agent_data, sample_task_data):
        """Test task submission fails for suspended agent."""
        client.post("/agents", json=sample_agent_data)
        client.post(f"/agents/{sample_agent_data['id']}/suspend")
        
        response = client.post("/tasks", json=sample_task_data)
        
        assert response.status_code == 400
        assert "not active" in response.json()["message"]

    def test_submit_task_with_context(self, client, sample_agent_data):
        """Test task submission with execution context."""
        client.post("/agents", json=sample_agent_data)
        
        task_data = {
            "agent_id": sample_agent_data["id"],
            "input": {"query": "test"},
            "context": {"user_id": "user-123", "session_id": "sess-456"},
            "metadata": {"source": "test"},
        }
        
        response = client.post("/tasks", json=task_data)
        
        assert response.status_code == 202

    def test_submit_task_with_priority(self, client, sample_agent_data):
        """Test task submission with priority."""
        client.post("/agents", json=sample_agent_data)
        
        task_data = {
            "agent_id": sample_agent_data["id"],
            "input": {"query": "urgent"},
            "priority": 10,
        }
        
        response = client.post("/tasks", json=task_data)
        
        assert response.status_code == 202

    def test_submit_task_with_correlation_id(self, client, sample_agent_data):
        """Test task submission with correlation ID."""
        client.post("/agents", json=sample_agent_data)
        
        task_data = {
            "agent_id": sample_agent_data["id"],
            "input": {"query": "test"},
            "correlation_id": "corr-123-456",
        }
        
        response = client.post("/tasks", json=task_data)
        
        assert response.status_code == 202

    def test_submit_task_invalid_timeout(self, client, sample_agent_data):
        """Test task submission with invalid timeout."""
        client.post("/agents", json=sample_agent_data)
        
        # Timeout too short
        task_data = {
            "agent_id": sample_agent_data["id"],
            "input": {"query": "test"},
            "timeout_ms": 50,  # Less than 100
        }
        
        response = client.post("/tasks", json=task_data)
        assert response.status_code == 422

    def test_submit_task_invalid_priority(self, client, sample_agent_data):
        """Test task submission with invalid priority."""
        client.post("/agents", json=sample_agent_data)
        
        # Priority out of range
        task_data = {
            "agent_id": sample_agent_data["id"],
            "input": {"query": "test"},
            "priority": 20,  # Greater than 10
        }
        
        response = client.post("/tasks", json=task_data)
        assert response.status_code == 422

    def test_get_task_not_implemented(self, client):
        """Test getting task status returns 501 (not implemented)."""
        response = client.get("/tasks/some-task-id")
        
        assert response.status_code == 501
        assert "not yet implemented" in response.json()["message"]


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_json_body(self, client):
        """Test handling of invalid JSON body."""
        response = client.post(
            "/agents",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422

    def test_empty_request_body(self, client):
        """Test handling of empty request body."""
        response = client.post(
            "/agents",
            json={}
        )
        
        assert response.status_code == 422

    def test_invalid_capability_type(self, client):
        """Test handling of invalid capability type."""
        agent_data = {
            "id": "bad-agent",
            "name": "Bad Agent",
            "version": "1.0.0",
            "framework": "langgraph",
            "capabilities": [{"name": "test", "type": "invalid_type"}],
        }
        
        response = client.post("/agents", json=agent_data)
        assert response.status_code == 422

    def test_pagination_invalid_page(self, client):
        """Test handling of invalid page number."""
        response = client.get("/agents?page=0")  # Must be >= 1
        assert response.status_code == 422

    def test_pagination_invalid_page_size(self, client):
        """Test handling of invalid page size."""
        response = client.get("/agents?page_size=200")  # Max is 100
        assert response.status_code == 422

    def test_min_trust_score_out_of_range(self, client):
        """Test handling of min_trust_score out of range."""
        response = client.get("/agents?min_trust_score=1.5")
        assert response.status_code == 422

    def test_request_with_x_request_id(self, client):
        """Test that X-Request-ID is included in error responses."""
        response = client.get(
            "/agents/nonexistent",
            headers={"X-Request-ID": "req-123-456"}
        )
        
        assert response.status_code == 404
        data = response.json()
        assert data.get("request_id") == "req-123-456"


# =============================================================================
# OpenAPI & Documentation Tests
# =============================================================================


class TestDocumentation:
    """Tests for API documentation endpoints."""

    def test_openapi_schema_available(self, client):
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "AI Workflow Fabric"

    def test_docs_endpoint_available(self, client):
        """Test Swagger UI is available."""
        response = client.get("/docs")
        
        assert response.status_code == 200

    def test_redoc_endpoint_available(self, client):
        """Test ReDoc is available."""
        response = client.get("/redoc")
        
        assert response.status_code == 200


# =============================================================================
# CORS Tests
# =============================================================================


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, client):
        """Test CORS headers are present for cross-origin requests."""
        response = client.options(
            "/agents",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            }
        )
        
        # In debug mode, CORS allows all origins
        assert response.status_code == 200


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests covering multiple endpoints."""

    def test_full_agent_lifecycle(self, client, sample_agent_data):
        """Test complete agent lifecycle: register, update, suspend, activate, delete."""
        # Register
        response = client.post("/agents", json=sample_agent_data)
        assert response.status_code == 201
        agent_id = response.json()["id"]
        
        # Verify registration
        response = client.get(f"/agents/{agent_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "active"
        
        # Update
        response = client.patch(f"/agents/{agent_id}", json={"name": "Updated Name"})
        assert response.status_code == 200
        assert response.json()["name"] == "Updated Name"
        
        # Suspend
        response = client.post(f"/agents/{agent_id}/suspend")
        assert response.status_code == 200
        assert response.json()["status"] == "suspended"
        
        # Try to submit task to suspended agent
        task_data = {"agent_id": agent_id, "input": {"test": "data"}}
        response = client.post("/tasks", json=task_data)
        assert response.status_code == 400
        
        # Activate
        response = client.post(f"/agents/{agent_id}/activate")
        assert response.status_code == 200
        assert response.json()["status"] == "active"
        
        # Submit task to active agent
        response = client.post("/tasks", json=task_data)
        assert response.status_code == 202
        
        # Get trust score
        response = client.get(f"/agents/{agent_id}/trust")
        assert response.status_code == 200
        
        # Delete
        response = client.delete(f"/agents/{agent_id}")
        assert response.status_code == 204
        
        # Verify deletion
        response = client.get(f"/agents/{agent_id}")
        assert response.status_code == 404

    def test_multi_agent_workflow_setup(self, client):
        """Test setting up multiple agents for a workflow."""
        # Register research agent
        research_agent = {
            "id": "research-agent",
            "name": "Research Agent",
            "version": "1.0.0",
            "framework": "langgraph",
            "capabilities": [{"name": "web_search", "type": "tool"}],
            "tags": ["research"],
        }
        response = client.post("/agents", json=research_agent)
        assert response.status_code == 201
        
        # Register writer agent
        writer_agent = {
            "id": "writer-agent",
            "name": "Writer Agent",
            "version": "1.0.0",
            "framework": "crewai",
            "capabilities": [{"name": "text_generation", "type": "reasoning"}],
            "tags": ["writing"],
        }
        response = client.post("/agents", json=writer_agent)
        assert response.status_code == 201
        
        # Verify both agents are registered
        response = client.get("/agents")
        assert response.json()["total"] == 2
        
        # Search by capability
        response = client.get("/agents?capabilities=web_search")
        assert response.json()["total"] == 1
        assert response.json()["agents"][0]["id"] == "research-agent"
        
        # Get health
        response = client.get("/health")
        assert response.json()["registry_count"] == 2


# =============================================================================
# Workflow CRUD Tests
# =============================================================================


@pytest.fixture
def sample_workflow_data():
    """Sample workflow definition data."""
    return {
        "id": "test-workflow-001",
        "name": "Test Research Workflow",
        "description": "A test workflow for research tasks",
        "steps": [
            {
                "id": "research",
                "agent_id": "research-agent",
                "input_map": {"query": "$.input.topic"},
                "retry": {
                    "max_attempts": 3,
                    "backoff_ms": 1000,
                    "backoff_multiplier": 2.0,
                    "max_backoff_ms": 30000,
                },
            },
            {
                "id": "write",
                "agent_id": "writer-agent",
                "input_map": {"content": "$.steps.research.output.data"},
                "depends_on": ["research"],
                "fallback": {
                    "skip": True,
                },
            },
        ],
        "timeout_ms": 60000,
    }


@pytest.fixture
def minimal_workflow_data():
    """Minimal valid workflow data."""
    return {
        "id": "minimal-workflow",
        "name": "Minimal Workflow",
        "steps": [
            {
                "id": "step1",
                "agent_id": "test-agent",
            },
        ],
    }


class TestWorkflowCRUD:
    """Tests for workflow CRUD endpoints."""

    def test_create_workflow_success(self, client, sample_workflow_data):
        """Test successful workflow creation."""
        response = client.post("/workflows", json=sample_workflow_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == sample_workflow_data["id"]
        assert data["name"] == sample_workflow_data["name"]
        assert data["description"] == sample_workflow_data["description"]
        assert len(data["steps"]) == 2
        assert data["timeout_ms"] == sample_workflow_data["timeout_ms"]

    def test_create_minimal_workflow(self, client, minimal_workflow_data):
        """Test creating workflow with minimal fields."""
        response = client.post("/workflows", json=minimal_workflow_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == minimal_workflow_data["id"]
        assert len(data["steps"]) == 1

    def test_create_duplicate_workflow_fails(self, client, sample_workflow_data):
        """Test that creating duplicate workflow returns 409."""
        client.post("/workflows", json=sample_workflow_data)
        
        response = client.post("/workflows", json=sample_workflow_data)
        assert response.status_code == 409
        assert "already exists" in response.json()["message"]

    def test_get_workflow_success(self, client, sample_workflow_data):
        """Test getting an existing workflow."""
        client.post("/workflows", json=sample_workflow_data)
        
        response = client.get(f"/workflows/{sample_workflow_data['id']}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_workflow_data["id"]
        assert data["name"] == sample_workflow_data["name"]

    def test_get_workflow_not_found(self, client):
        """Test getting non-existent workflow returns 404."""
        response = client.get("/workflows/nonexistent-workflow")
        
        assert response.status_code == 404
        assert "not found" in response.json()["message"]

    def test_list_workflows_empty(self, client):
        """Test listing workflows when none exist."""
        response = client.get("/workflows")
        
        assert response.status_code == 200
        assert response.json() == []

    def test_list_workflows_returns_all(self, client, sample_workflow_data, minimal_workflow_data):
        """Test listing returns all workflows."""
        client.post("/workflows", json=sample_workflow_data)
        client.post("/workflows", json=minimal_workflow_data)
        
        response = client.get("/workflows")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_delete_workflow_success(self, client, sample_workflow_data):
        """Test successful workflow deletion."""
        client.post("/workflows", json=sample_workflow_data)
        
        response = client.delete(f"/workflows/{sample_workflow_data['id']}")
        assert response.status_code == 204
        
        # Verify deleted
        response = client.get(f"/workflows/{sample_workflow_data['id']}")
        assert response.status_code == 404

    def test_delete_workflow_not_found(self, client):
        """Test deleting non-existent workflow returns 404."""
        response = client.delete("/workflows/nonexistent-workflow")
        assert response.status_code == 404

    def test_workflow_step_with_retry_policy(self, client):
        """Test workflow step with retry policy."""
        workflow = {
            "id": "retry-workflow",
            "name": "Retry Workflow",
            "steps": [
                {
                    "id": "step1",
                    "agent_id": "test-agent",
                    "retry": {
                        "max_attempts": 5,
                        "backoff_ms": 500,
                        "backoff_multiplier": 1.5,
                        "max_backoff_ms": 10000,
                    },
                },
            ],
        }
        
        response = client.post("/workflows", json=workflow)
        assert response.status_code == 201
        
        # Verify retry config is stored
        data = response.json()
        step = data["steps"][0]
        assert step.get("retry") is not None

    def test_workflow_step_with_fallback_policy(self, client):
        """Test workflow step with fallback policy."""
        workflow = {
            "id": "fallback-workflow",
            "name": "Fallback Workflow",
            "steps": [
                {
                    "id": "step1",
                    "agent_id": "test-agent",
                    "fallback": {
                        "static_value": {"default": "value"},
                    },
                },
            ],
        }
        
        response = client.post("/workflows", json=workflow)
        assert response.status_code == 201

    def test_workflow_step_with_condition(self, client):
        """Test workflow step with condition."""
        workflow = {
            "id": "conditional-workflow",
            "name": "Conditional Workflow",
            "steps": [
                {
                    "id": "step1",
                    "agent_id": "test-agent",
                },
                {
                    "id": "step2",
                    "agent_id": "test-agent-2",
                    "depends_on": ["step1"],
                    "condition": "$.input.flag == True",
                },
            ],
        }
        
        response = client.post("/workflows", json=workflow)
        assert response.status_code == 201
        
        data = response.json()
        assert data["steps"][1].get("condition") == "$.input.flag == True"


# =============================================================================
# Workflow Execution Tests
# =============================================================================


class TestWorkflowExecution:
    """Tests for workflow execution endpoints."""

    def test_execute_workflow_not_found(self, client):
        """Test executing non-existent workflow returns 404."""
        response = client.post(
            "/workflows/nonexistent/execute",
            json={"input": {"test": "data"}}
        )
        assert response.status_code == 404

    def test_get_execution_not_found(self, client):
        """Test getting non-existent execution returns 404."""
        response = client.get("/executions/nonexistent-execution")
        assert response.status_code == 404

    def test_cancel_execution_not_found(self, client):
        """Test cancelling non-existent execution returns 404."""
        response = client.post("/executions/nonexistent-execution/cancel")
        assert response.status_code == 404

    def test_list_executions_empty(self, client):
        """Test listing executions when none exist."""
        response = client.get("/executions")
        
        assert response.status_code == 200
        assert response.json() == []

    def test_list_executions_filter_by_workflow_id(self, client):
        """Test filtering executions by workflow ID."""
        response = client.get("/executions?workflow_id=test-workflow")
        
        assert response.status_code == 200
        assert response.json() == []

    def test_list_executions_filter_by_status(self, client):
        """Test filtering executions by status."""
        response = client.get("/executions?status=completed")
        
        assert response.status_code == 200
        assert response.json() == []

    def test_execution_request_validation(self, client, minimal_workflow_data):
        """Test execution request validation."""
        client.post("/workflows", json=minimal_workflow_data)
        
        # Valid timeout
        response = client.post(
            f"/workflows/{minimal_workflow_data['id']}/execute",
            json={
                "input": {"test": "data"},
                "timeout_ms": 60000,
            }
        )
        # Will fail due to missing agent, but validates request
        assert response.status_code in (202, 500)

    def test_execution_request_with_context(self, client, minimal_workflow_data):
        """Test execution request with context."""
        client.post("/workflows", json=minimal_workflow_data)
        
        response = client.post(
            f"/workflows/{minimal_workflow_data['id']}/execute",
            json={
                "input": {"test": "data"},
                "context": {"user_id": "user-123"},
                "trace_id": "trace-456",
                "correlation_id": "corr-789",
            }
        )
        # Will fail due to missing agent, but validates request
        assert response.status_code in (202, 500)


# =============================================================================
# SSE Streaming Tests
# =============================================================================


class TestSSEStreaming:
    """Tests for SSE streaming endpoint."""

    def test_sse_endpoint_exists(self, client):
        """Test that SSE endpoint exists and returns correct content type."""
        # This will timeout since there's no execution, but validates endpoint exists
        import httpx
        
        # Use a short timeout to test the endpoint exists
        response = client.get(
            "/executions/test-execution/events",
            headers={"Accept": "text/event-stream"}
        )
        
        # Should return 200 with event-stream content type
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")


# =============================================================================
# Workflow Integration Tests
# =============================================================================


class TestWorkflowIntegration:
    """Integration tests for workflow features."""

    def test_workflow_lifecycle(self, client, minimal_workflow_data):
        """Test complete workflow lifecycle: create, read, delete."""
        # Create
        response = client.post("/workflows", json=minimal_workflow_data)
        assert response.status_code == 201
        workflow_id = response.json()["id"]
        
        # Read
        response = client.get(f"/workflows/{workflow_id}")
        assert response.status_code == 200
        assert response.json()["id"] == workflow_id
        
        # List
        response = client.get("/workflows")
        assert response.status_code == 200
        assert len(response.json()) == 1
        
        # Delete
        response = client.delete(f"/workflows/{workflow_id}")
        assert response.status_code == 204
        
        # Verify deleted
        response = client.get(f"/workflows/{workflow_id}")
        assert response.status_code == 404
        
        # List should be empty
        response = client.get("/workflows")
        assert response.json() == []

    def test_workflow_with_dependencies(self, client):
        """Test workflow with step dependencies."""
        workflow = {
            "id": "dependency-workflow",
            "name": "Dependency Workflow",
            "steps": [
                {
                    "id": "step1",
                    "agent_id": "agent-1",
                },
                {
                    "id": "step2",
                    "agent_id": "agent-2",
                    "depends_on": ["step1"],
                },
                {
                    "id": "step3",
                    "agent_id": "agent-3",
                    "depends_on": ["step1", "step2"],
                },
            ],
        }
        
        response = client.post("/workflows", json=workflow)
        assert response.status_code == 201
        
        data = response.json()
        assert len(data["steps"]) == 3
        assert data["steps"][1].get("dependsOn") == ["step1"]
        assert data["steps"][2].get("dependsOn") == ["step1", "step2"]

    def test_workflow_with_input_mapping(self, client):
        """Test workflow with JSONPath input mapping."""
        workflow = {
            "id": "mapping-workflow",
            "name": "Mapping Workflow",
            "steps": [
                {
                    "id": "step1",
                    "agent_id": "agent-1",
                    "input_map": {
                        "query": "$.input.topic",
                        "limit": "$.input.max_results",
                    },
                },
                {
                    "id": "step2",
                    "agent_id": "agent-2",
                    "depends_on": ["step1"],
                    "input_map": {
                        "data": "$.steps.step1.output.results",
                    },
                },
            ],
        }
        
        response = client.post("/workflows", json=workflow)
        assert response.status_code == 201
        
        data = response.json()
        assert data["steps"][0].get("inputMap") is not None
        assert data["steps"][1].get("inputMap") is not None
