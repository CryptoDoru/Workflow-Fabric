"""
Tests for AWF Registry implementations.

Tests both InMemoryRegistry and SQLiteRegistry for:
- CRUD operations (register, get, update, delete)
- Search functionality (capabilities, framework, tags, trust score)
- Concurrent access
- Persistence (SQLite only)
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from awf.core.types import (
    AgentManifest,
    AgentStatus,
    Capability,
    CapabilityType,
)
from awf.registry.memory import InMemoryRegistry
from awf.registry.persistence import SQLiteRegistry


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_manifest() -> AgentManifest:
    """Create a sample agent manifest for testing."""
    return AgentManifest(
        id="test-agent-1",
        name="Test Agent",
        version="1.0.0",
        framework="langgraph",
        framework_version="0.2.0",
        description="A test agent",
        capabilities=[
            Capability(
                name="web_search",
                type=CapabilityType.TOOL,
                description="Search the web",
                permissions=["network:external"],
            ),
            Capability(
                name="summarize",
                type=CapabilityType.REASONING,
                description="Summarize text",
            ),
        ],
        tags=["test", "search", "demo"],
        publisher="test-publisher",
        trust_score=0.85,
        status=AgentStatus.ACTIVE,
    )


@pytest.fixture
def sample_manifest_2() -> AgentManifest:
    """Create a second sample agent manifest."""
    return AgentManifest(
        id="test-agent-2",
        name="Code Agent",
        version="2.0.0",
        framework="crewai",
        description="A code analysis agent",
        capabilities=[
            Capability(
                name="code_review",
                type=CapabilityType.REASONING,
                description="Review code",
            ),
            Capability(
                name="security_scan",
                type=CapabilityType.TOOL,
                description="Scan for vulnerabilities",
                permissions=["filesystem:read"],
            ),
        ],
        tags=["code", "security"],
        trust_score=0.92,
        status=AgentStatus.ACTIVE,
    )


@pytest.fixture
def sample_manifest_inactive() -> AgentManifest:
    """Create an inactive agent manifest."""
    return AgentManifest(
        id="test-agent-inactive",
        name="Inactive Agent",
        version="1.0.0",
        framework="langgraph",
        status=AgentStatus.SUSPENDED,
    )


@pytest.fixture
def memory_registry() -> InMemoryRegistry:
    """Create an in-memory registry."""
    return InMemoryRegistry()


@pytest.fixture
async def sqlite_registry(tmp_path: Path) -> SQLiteRegistry:
    """Create a SQLite registry with temporary database."""
    db_path = tmp_path / "test_registry.db"
    registry = SQLiteRegistry(db_path=str(db_path))
    await registry.initialize()
    yield registry
    await registry.close()


# =============================================================================
# InMemoryRegistry Tests
# =============================================================================


class TestInMemoryRegistry:
    """Tests for InMemoryRegistry."""

    @pytest.mark.asyncio
    async def test_register_and_get(
        self, memory_registry: InMemoryRegistry, sample_manifest: AgentManifest
    ):
        """Test registering and retrieving an agent."""
        await memory_registry.register(sample_manifest)
        
        result = await memory_registry.get(sample_manifest.id)
        
        assert result is not None
        assert result.id == sample_manifest.id
        assert result.name == sample_manifest.name
        assert result.version == sample_manifest.version
        assert result.framework == sample_manifest.framework

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, memory_registry: InMemoryRegistry):
        """Test getting a non-existent agent returns None."""
        result = await memory_registry.get("nonexistent-agent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update(
        self, memory_registry: InMemoryRegistry, sample_manifest: AgentManifest
    ):
        """Test updating an agent manifest."""
        await memory_registry.register(sample_manifest)
        
        sample_manifest.version = "2.0.0"
        sample_manifest.description = "Updated description"
        
        result = await memory_registry.update(sample_manifest)
        assert result is True
        
        updated = await memory_registry.get(sample_manifest.id)
        assert updated is not None
        assert updated.version == "2.0.0"
        assert updated.description == "Updated description"

    @pytest.mark.asyncio
    async def test_update_nonexistent(
        self, memory_registry: InMemoryRegistry, sample_manifest: AgentManifest
    ):
        """Test updating a non-existent agent returns False."""
        result = await memory_registry.update(sample_manifest)
        assert result is False

    @pytest.mark.asyncio
    async def test_delete(
        self, memory_registry: InMemoryRegistry, sample_manifest: AgentManifest
    ):
        """Test deleting an agent."""
        await memory_registry.register(sample_manifest)
        
        result = await memory_registry.delete(sample_manifest.id)
        assert result is True
        
        # Verify deletion
        agent = await memory_registry.get(sample_manifest.id)
        assert agent is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, memory_registry: InMemoryRegistry):
        """Test deleting a non-existent agent returns False."""
        result = await memory_registry.delete("nonexistent-agent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_all(
        self,
        memory_registry: InMemoryRegistry,
        sample_manifest: AgentManifest,
        sample_manifest_2: AgentManifest,
    ):
        """Test listing all agents."""
        await memory_registry.register(sample_manifest)
        await memory_registry.register(sample_manifest_2)
        
        results = await memory_registry.list_all()
        
        assert len(results) == 2
        ids = {r.id for r in results}
        assert sample_manifest.id in ids
        assert sample_manifest_2.id in ids

    @pytest.mark.asyncio
    async def test_count(
        self,
        memory_registry: InMemoryRegistry,
        sample_manifest: AgentManifest,
        sample_manifest_2: AgentManifest,
    ):
        """Test counting agents."""
        assert await memory_registry.count() == 0
        
        await memory_registry.register(sample_manifest)
        assert await memory_registry.count() == 1
        
        await memory_registry.register(sample_manifest_2)
        assert await memory_registry.count() == 2

    @pytest.mark.asyncio
    async def test_clear(
        self,
        memory_registry: InMemoryRegistry,
        sample_manifest: AgentManifest,
        sample_manifest_2: AgentManifest,
    ):
        """Test clearing all agents."""
        await memory_registry.register(sample_manifest)
        await memory_registry.register(sample_manifest_2)
        
        await memory_registry.clear()
        
        assert await memory_registry.count() == 0

    @pytest.mark.asyncio
    async def test_set_status(
        self, memory_registry: InMemoryRegistry, sample_manifest: AgentManifest
    ):
        """Test setting agent status."""
        await memory_registry.register(sample_manifest)
        
        result = await memory_registry.set_status(
            sample_manifest.id, AgentStatus.SUSPENDED
        )
        assert result is True
        
        agent = await memory_registry.get(sample_manifest.id)
        assert agent is not None
        assert agent.status == AgentStatus.SUSPENDED

    # Search Tests
    
    @pytest.mark.asyncio
    async def test_search_by_capability(
        self,
        memory_registry: InMemoryRegistry,
        sample_manifest: AgentManifest,
        sample_manifest_2: AgentManifest,
    ):
        """Test searching by capability."""
        await memory_registry.register(sample_manifest)
        await memory_registry.register(sample_manifest_2)
        
        # Search for web_search capability
        results = await memory_registry.search(capabilities=["web_search"])
        assert len(results) == 1
        assert results[0].id == sample_manifest.id
        
        # Search for code_review capability
        results = await memory_registry.search(capabilities=["code_review"])
        assert len(results) == 1
        assert results[0].id == sample_manifest_2.id

    @pytest.mark.asyncio
    async def test_search_by_framework(
        self,
        memory_registry: InMemoryRegistry,
        sample_manifest: AgentManifest,
        sample_manifest_2: AgentManifest,
    ):
        """Test searching by framework."""
        await memory_registry.register(sample_manifest)
        await memory_registry.register(sample_manifest_2)
        
        results = await memory_registry.search(framework="langgraph")
        assert len(results) == 1
        assert results[0].id == sample_manifest.id
        
        results = await memory_registry.search(framework="crewai")
        assert len(results) == 1
        assert results[0].id == sample_manifest_2.id

    @pytest.mark.asyncio
    async def test_search_by_tags(
        self,
        memory_registry: InMemoryRegistry,
        sample_manifest: AgentManifest,
        sample_manifest_2: AgentManifest,
    ):
        """Test searching by tags."""
        await memory_registry.register(sample_manifest)
        await memory_registry.register(sample_manifest_2)
        
        results = await memory_registry.search(tags=["search"])
        assert len(results) == 1
        assert results[0].id == sample_manifest.id
        
        results = await memory_registry.search(tags=["security"])
        assert len(results) == 1
        assert results[0].id == sample_manifest_2.id

    @pytest.mark.asyncio
    async def test_search_by_trust_score(
        self,
        memory_registry: InMemoryRegistry,
        sample_manifest: AgentManifest,
        sample_manifest_2: AgentManifest,
    ):
        """Test searching by minimum trust score."""
        await memory_registry.register(sample_manifest)  # 0.85
        await memory_registry.register(sample_manifest_2)  # 0.92
        
        results = await memory_registry.search(min_trust_score=0.90)
        assert len(results) == 1
        assert results[0].id == sample_manifest_2.id
        
        results = await memory_registry.search(min_trust_score=0.80)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_excludes_inactive(
        self,
        memory_registry: InMemoryRegistry,
        sample_manifest: AgentManifest,
        sample_manifest_inactive: AgentManifest,
    ):
        """Test that search excludes inactive agents."""
        await memory_registry.register(sample_manifest)
        await memory_registry.register(sample_manifest_inactive)
        
        results = await memory_registry.search()
        assert len(results) == 1
        assert results[0].id == sample_manifest.id

    @pytest.mark.asyncio
    async def test_search_combined_filters(
        self,
        memory_registry: InMemoryRegistry,
        sample_manifest: AgentManifest,
        sample_manifest_2: AgentManifest,
    ):
        """Test searching with multiple filters."""
        await memory_registry.register(sample_manifest)
        await memory_registry.register(sample_manifest_2)
        
        # No results - conflicting filters
        results = await memory_registry.search(
            framework="langgraph",
            capabilities=["code_review"],
        )
        assert len(results) == 0
        
        # One result - matching filters
        results = await memory_registry.search(
            framework="langgraph",
            capabilities=["web_search"],
        )
        assert len(results) == 1

    # Concurrency Tests
    
    @pytest.mark.asyncio
    async def test_concurrent_registration(self, memory_registry: InMemoryRegistry):
        """Test concurrent agent registration."""
        agents = [
            AgentManifest(
                id=f"concurrent-agent-{i}",
                name=f"Concurrent Agent {i}",
                version="1.0.0",
                framework="langgraph",
                status=AgentStatus.ACTIVE,
            )
            for i in range(100)
        ]
        
        # Register all agents concurrently
        await asyncio.gather(*[
            memory_registry.register(agent) for agent in agents
        ])
        
        # Verify all were registered
        assert await memory_registry.count() == 100


# =============================================================================
# SQLiteRegistry Tests
# =============================================================================


class TestSQLiteRegistry:
    """Tests for SQLiteRegistry."""

    @pytest.mark.asyncio
    async def test_register_and_get(
        self, sqlite_registry: SQLiteRegistry, sample_manifest: AgentManifest
    ):
        """Test registering and retrieving an agent."""
        await sqlite_registry.register(sample_manifest)
        
        result = await sqlite_registry.get(sample_manifest.id)
        
        assert result is not None
        assert result.id == sample_manifest.id
        assert result.name == sample_manifest.name
        assert result.version == sample_manifest.version
        assert result.framework == sample_manifest.framework
        assert len(result.capabilities) == len(sample_manifest.capabilities)

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, sqlite_registry: SQLiteRegistry):
        """Test getting a non-existent agent returns None."""
        result = await sqlite_registry.get("nonexistent-agent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update(
        self, sqlite_registry: SQLiteRegistry, sample_manifest: AgentManifest
    ):
        """Test updating an agent manifest."""
        await sqlite_registry.register(sample_manifest)
        original_registered_at = sample_manifest.registered_at
        
        sample_manifest.version = "2.0.0"
        sample_manifest.description = "Updated description"
        
        result = await sqlite_registry.update(sample_manifest)
        assert result is True
        
        updated = await sqlite_registry.get(sample_manifest.id)
        assert updated is not None
        assert updated.version == "2.0.0"
        assert updated.description == "Updated description"
        # Registration time should be preserved
        assert updated.registered_at == original_registered_at

    @pytest.mark.asyncio
    async def test_delete(
        self, sqlite_registry: SQLiteRegistry, sample_manifest: AgentManifest
    ):
        """Test deleting an agent."""
        await sqlite_registry.register(sample_manifest)
        
        result = await sqlite_registry.delete(sample_manifest.id)
        assert result is True
        
        agent = await sqlite_registry.get(sample_manifest.id)
        assert agent is None

    @pytest.mark.asyncio
    async def test_search_by_capability(
        self,
        sqlite_registry: SQLiteRegistry,
        sample_manifest: AgentManifest,
        sample_manifest_2: AgentManifest,
    ):
        """Test searching by capability in SQLite."""
        await sqlite_registry.register(sample_manifest)
        await sqlite_registry.register(sample_manifest_2)
        
        results = await sqlite_registry.search(capabilities=["web_search"])
        assert len(results) == 1
        assert results[0].id == sample_manifest.id

    @pytest.mark.asyncio
    async def test_search_by_framework(
        self,
        sqlite_registry: SQLiteRegistry,
        sample_manifest: AgentManifest,
        sample_manifest_2: AgentManifest,
    ):
        """Test searching by framework in SQLite."""
        await sqlite_registry.register(sample_manifest)
        await sqlite_registry.register(sample_manifest_2)
        
        results = await sqlite_registry.search(framework="crewai")
        assert len(results) == 1
        assert results[0].id == sample_manifest_2.id

    @pytest.mark.asyncio
    async def test_search_by_tags(
        self,
        sqlite_registry: SQLiteRegistry,
        sample_manifest: AgentManifest,
        sample_manifest_2: AgentManifest,
    ):
        """Test searching by tags in SQLite."""
        await sqlite_registry.register(sample_manifest)
        await sqlite_registry.register(sample_manifest_2)
        
        results = await sqlite_registry.search(tags=["code"])
        assert len(results) == 1
        assert results[0].id == sample_manifest_2.id

    @pytest.mark.asyncio
    async def test_search_by_capability_specific(
        self, sqlite_registry: SQLiteRegistry, sample_manifest: AgentManifest
    ):
        """Test search_by_capability method."""
        await sqlite_registry.register(sample_manifest)
        
        results = await sqlite_registry.search_by_capability("web_search")
        assert len(results) == 1
        assert results[0].id == sample_manifest.id
        
        results = await sqlite_registry.search_by_capability(
            "web_search", CapabilityType.TOOL
        )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_persistence(self, tmp_path: Path, sample_manifest: AgentManifest):
        """Test that data persists across registry instances."""
        db_path = tmp_path / "persistence_test.db"
        
        # Create first registry and register agent
        registry1 = SQLiteRegistry(db_path=str(db_path))
        await registry1.initialize()
        await registry1.register(sample_manifest)
        await registry1.close()
        
        # Create second registry and verify agent exists
        registry2 = SQLiteRegistry(db_path=str(db_path))
        await registry2.initialize()
        
        result = await registry2.get(sample_manifest.id)
        assert result is not None
        assert result.id == sample_manifest.id
        assert result.name == sample_manifest.name
        
        await registry2.close()

    @pytest.mark.asyncio
    async def test_not_initialized_error(self, tmp_path: Path):
        """Test that operations fail if not initialized."""
        db_path = tmp_path / "not_init.db"
        registry = SQLiteRegistry(db_path=str(db_path))
        
        with pytest.raises(RuntimeError, match="not initialized"):
            await registry.get("test")

    @pytest.mark.asyncio
    async def test_vacuum(
        self, sqlite_registry: SQLiteRegistry, sample_manifest: AgentManifest
    ):
        """Test vacuum operation."""
        await sqlite_registry.register(sample_manifest)
        await sqlite_registry.delete(sample_manifest.id)
        
        # Should not raise
        await sqlite_registry.vacuum()

    @pytest.mark.asyncio
    async def test_count_and_clear(
        self,
        sqlite_registry: SQLiteRegistry,
        sample_manifest: AgentManifest,
        sample_manifest_2: AgentManifest,
    ):
        """Test count and clear operations."""
        await sqlite_registry.register(sample_manifest)
        await sqlite_registry.register(sample_manifest_2)
        
        assert await sqlite_registry.count() == 2
        
        await sqlite_registry.clear()
        
        assert await sqlite_registry.count() == 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_capabilities(self, memory_registry: InMemoryRegistry):
        """Test agent with no capabilities."""
        agent = AgentManifest(
            id="no-caps-agent",
            name="No Capabilities",
            version="1.0.0",
            framework="langgraph",
            capabilities=[],
            status=AgentStatus.ACTIVE,
        )
        
        await memory_registry.register(agent)
        result = await memory_registry.get(agent.id)
        
        assert result is not None
        assert len(result.capabilities) == 0

    @pytest.mark.asyncio
    async def test_special_characters_in_id(self, memory_registry: InMemoryRegistry):
        """Test agent ID with special characters."""
        agent = AgentManifest(
            id="agent-with-special_chars.v1",
            name="Special Agent",
            version="1.0.0",
            framework="langgraph",
            status=AgentStatus.ACTIVE,
        )
        
        await memory_registry.register(agent)
        result = await memory_registry.get("agent-with-special_chars.v1")
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_unicode_content(self, memory_registry: InMemoryRegistry):
        """Test agent with unicode content."""
        agent = AgentManifest(
            id="unicode-agent",
            name="Агент 日本語",
            version="1.0.0",
            framework="langgraph",
            description="描述 описание 🤖",
            status=AgentStatus.ACTIVE,
        )
        
        await memory_registry.register(agent)
        result = await memory_registry.get(agent.id)
        
        assert result is not None
        assert result.name == "Агент 日本語"
        assert "🤖" in result.description

    @pytest.mark.asyncio
    async def test_overwrite_on_register(
        self, memory_registry: InMemoryRegistry, sample_manifest: AgentManifest
    ):
        """Test that register overwrites existing agent."""
        await memory_registry.register(sample_manifest)
        
        # Modify and re-register
        sample_manifest.version = "3.0.0"
        await memory_registry.register(sample_manifest)
        
        result = await memory_registry.get(sample_manifest.id)
        assert result is not None
        assert result.version == "3.0.0"

    @pytest.mark.asyncio
    async def test_none_trust_score_filter(
        self, memory_registry: InMemoryRegistry
    ):
        """Test search with None trust score."""
        agent = AgentManifest(
            id="no-trust-agent",
            name="No Trust Score",
            version="1.0.0",
            framework="langgraph",
            trust_score=None,
            status=AgentStatus.ACTIVE,
        )
        
        await memory_registry.register(agent)
        
        # Should not be included when filtering by trust score
        results = await memory_registry.search(min_trust_score=0.5)
        assert len(results) == 0
        
        # Should be included when not filtering by trust score
        results = await memory_registry.search()
        assert len(results) == 1
