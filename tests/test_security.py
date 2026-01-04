"""
Tests for AWF Security module.

Tests for:
- Trust scoring engine
- Policy engine
- Sandbox orchestrator
"""

import pytest
from datetime import datetime, timedelta, timezone

from awf.core.types import (
    AgentManifest,
    AgentStatus,
    Capability,
    CapabilityType,
    Event,
    EventType,
    Policy,
    SandboxTier,
    Task,
    TrustFactors,
    TrustScore,
)
from awf.security.trust import (
    TrustScoringEngine,
    PublisherInfo,
    AuditInfo,
    CommunityMetrics,
)
from awf.security.policy import PolicyEngine, PolicyEvaluationResult
from awf.security.sandbox import (
    SandboxOrchestrator,
    SandboxConfig,
    SandboxPool,
    StubSandbox,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def trust_engine() -> TrustScoringEngine:
    """Create a trust scoring engine."""
    return TrustScoringEngine()


@pytest.fixture
def policy_engine() -> PolicyEngine:
    """Create a policy engine."""
    engine = PolicyEngine()
    engine.create_default_policies()
    return engine


@pytest.fixture
def sandbox_orchestrator() -> SandboxOrchestrator:
    """Create a sandbox orchestrator with stub sandbox."""
    orchestrator = SandboxOrchestrator()
    orchestrator.register_sandbox_type(SandboxTier.WASM, StubSandbox)
    return orchestrator


@pytest.fixture
def high_trust_manifest() -> AgentManifest:
    """Create a high-trust agent manifest."""
    return AgentManifest(
        id="high-trust-agent",
        name="High Trust Agent",
        version="1.0.0",
        framework="langgraph",
        publisher="verified-publisher",
        audit_status="audited",
        capabilities=[
            Capability(
                name="simple_task",
                type=CapabilityType.REASONING,
                description="Simple reasoning",
            ),
        ],
        status=AgentStatus.ACTIVE,
    )


@pytest.fixture
def low_trust_manifest() -> AgentManifest:
    """Create a low-trust agent manifest."""
    return AgentManifest(
        id="low-trust-agent",
        name="Low Trust Agent",
        version="1.0.0",
        framework="unknown",
        capabilities=[
            Capability(
                name="dangerous_task",
                type=CapabilityType.TOOL,
                description="Dangerous operation",
                permissions=["process:execute", "filesystem:write", "network:external"],
            ),
        ],
        status=AgentStatus.ACTIVE,
    )


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task."""
    return Task(
        agent_id="test-agent",
        input={"query": "test"},
        timeout_ms=30000,
    )


# =============================================================================
# Trust Scoring Engine Tests
# =============================================================================


class TestTrustScoringEngine:
    """Tests for TrustScoringEngine."""

    @pytest.mark.asyncio
    async def test_compute_score_basic(
        self, trust_engine: TrustScoringEngine, high_trust_manifest: AgentManifest
    ):
        """Test basic trust score computation."""
        score = await trust_engine.compute_score(high_trust_manifest)
        
        assert score is not None
        assert isinstance(score, TrustScore)
        assert 0.0 <= score.score <= 1.0
        assert score.sandbox_tier is not None

    @pytest.mark.asyncio
    async def test_score_factors(
        self, trust_engine: TrustScoringEngine, high_trust_manifest: AgentManifest
    ):
        """Test that all trust factors are computed."""
        score = await trust_engine.compute_score(high_trust_manifest)
        
        assert score.factors is not None
        assert 0.0 <= score.factors.publisher_trust <= 1.0
        assert 0.0 <= score.factors.audit_status <= 1.0
        assert 0.0 <= score.factors.community_trust <= 1.0
        assert 0.0 <= score.factors.permission_analysis <= 1.0
        assert 0.0 <= score.factors.historical_behavior <= 1.0

    @pytest.mark.asyncio
    async def test_sandbox_tier_wasm(self, trust_engine: TrustScoringEngine):
        """Test WASM tier for high trust score."""
        factors = TrustFactors(
            publisher_trust=1.0,
            audit_status=1.0,
            community_trust=1.0,
            permission_analysis=1.0,
            historical_behavior=1.0,
        )
        score = TrustScore.compute(factors)
        
        assert score.score >= 0.90
        assert score.sandbox_tier == SandboxTier.WASM

    @pytest.mark.asyncio
    async def test_sandbox_tier_gvisor(self, trust_engine: TrustScoringEngine):
        """Test gVisor tier for medium trust score."""
        factors = TrustFactors(
            publisher_trust=0.8,
            audit_status=0.7,
            community_trust=0.8,
            permission_analysis=0.8,
            historical_behavior=0.8,
        )
        score = TrustScore.compute(factors)
        
        assert 0.70 <= score.score < 0.90
        assert score.sandbox_tier == SandboxTier.GVISOR

    @pytest.mark.asyncio
    async def test_sandbox_tier_gvisor_strict(self, trust_engine: TrustScoringEngine):
        """Test gVisor Strict tier for low trust score."""
        factors = TrustFactors(
            publisher_trust=0.5,
            audit_status=0.4,
            community_trust=0.5,
            permission_analysis=0.6,
            historical_behavior=0.5,
        )
        score = TrustScore.compute(factors)
        
        assert 0.40 <= score.score < 0.70
        assert score.sandbox_tier == SandboxTier.GVISOR_STRICT

    @pytest.mark.asyncio
    async def test_sandbox_tier_blocked(self, trust_engine: TrustScoringEngine):
        """Test BLOCKED tier for very low trust score."""
        factors = TrustFactors(
            publisher_trust=0.2,
            audit_status=0.1,
            community_trust=0.2,
            permission_analysis=0.3,
            historical_behavior=0.2,
        )
        score = TrustScore.compute(factors)
        
        assert score.score < 0.40
        assert score.sandbox_tier == SandboxTier.BLOCKED

    @pytest.mark.asyncio
    async def test_publisher_info_affects_score(
        self, trust_engine: TrustScoringEngine, high_trust_manifest: AgentManifest
    ):
        """Test that publisher info affects trust score."""
        # Register verified publisher
        trust_engine.register_publisher(PublisherInfo(
            id="verified-publisher",
            verified=True,
            reputation_score=0.9,
            agent_count=50,
            violations=0,
        ))
        
        score = await trust_engine.compute_score(high_trust_manifest)
        verified_publisher_trust = score.factors.publisher_trust
        
        # Should have high publisher trust
        assert verified_publisher_trust > 0.7

    @pytest.mark.asyncio
    async def test_audit_info_affects_score(
        self, trust_engine: TrustScoringEngine, high_trust_manifest: AgentManifest
    ):
        """Test that audit info affects trust score."""
        # Register recent audit
        trust_engine.register_audit(
            high_trust_manifest.id,
            AuditInfo(
                audited=True,
                auditor="security-firm",
                audit_date=datetime.now(timezone.utc) - timedelta(days=30),
                findings=0,
                critical_findings=0,
            ),
        )
        
        score = await trust_engine.compute_score(high_trust_manifest)
        
        # Should have high audit status
        assert score.factors.audit_status > 0.8

    @pytest.mark.asyncio
    async def test_community_metrics_affects_score(
        self, trust_engine: TrustScoringEngine, high_trust_manifest: AgentManifest
    ):
        """Test that community metrics affect trust score."""
        trust_engine.update_community_metrics(
            high_trust_manifest.id,
            CommunityMetrics(
                total_uses=10000,
                unique_users=500,
                positive_ratings=450,
                negative_ratings=10,
                abuse_reports=0,
            ),
        )
        
        score = await trust_engine.compute_score(high_trust_manifest)
        
        # Should have high community trust
        assert score.factors.community_trust > 0.8

    @pytest.mark.asyncio
    async def test_risky_permissions_lower_score(
        self, trust_engine: TrustScoringEngine, low_trust_manifest: AgentManifest
    ):
        """Test that risky permissions lower trust score."""
        score = await trust_engine.compute_score(low_trust_manifest)
        
        # Should have low permission analysis score
        assert score.factors.permission_analysis < 0.7

    @pytest.mark.asyncio
    async def test_update_score_on_event(
        self, trust_engine: TrustScoringEngine, high_trust_manifest: AgentManifest
    ):
        """Test updating score based on events."""
        # Compute initial score
        initial_score = await trust_engine.compute_score(high_trust_manifest)
        
        # Report abuse event
        abuse_event = Event(
            type=EventType.POLICY_VIOLATION,
            source=high_trust_manifest.id,
            data={"reason": "abuse"},
        )
        
        updated_score = await trust_engine.update_score(
            high_trust_manifest.id, abuse_event
        )
        
        assert updated_score is not None
        assert updated_score.score < initial_score.score

    @pytest.mark.asyncio
    async def test_score_caching(
        self, trust_engine: TrustScoringEngine, high_trust_manifest: AgentManifest
    ):
        """Test that scores are cached."""
        await trust_engine.compute_score(high_trust_manifest)
        
        cached = trust_engine.get_cached_score(high_trust_manifest.id)
        assert cached is not None
        
        trust_engine.invalidate_cache(high_trust_manifest.id)
        assert trust_engine.get_cached_score(high_trust_manifest.id) is None


# =============================================================================
# Policy Engine Tests
# =============================================================================


class TestPolicyEngine:
    """Tests for PolicyEngine."""

    def test_default_policies_created(self, policy_engine: PolicyEngine):
        """Test that default policies are created."""
        policies = policy_engine.list_policies()
        
        assert len(policies) >= 3
        ids = {p.id for p in policies}
        assert "default-production" in ids
        assert "default-staging" in ids
        assert "default-development" in ids

    def test_register_policy(self, policy_engine: PolicyEngine):
        """Test registering a custom policy."""
        custom_policy = Policy(
            id="custom-policy",
            name="Custom Policy",
            environments=["test"],
            min_trust_score=0.8,
        )
        
        policy_engine.register_policy(custom_policy)
        
        retrieved = policy_engine.get_policy("custom-policy")
        assert retrieved is not None
        assert retrieved.name == "Custom Policy"

    def test_unregister_policy(self, policy_engine: PolicyEngine):
        """Test unregistering a policy."""
        result = policy_engine.unregister_policy("default-development")
        assert result is True
        
        assert policy_engine.get_policy("default-development") is None

    @pytest.mark.asyncio
    async def test_evaluate_allowed(
        self,
        policy_engine: PolicyEngine,
        high_trust_manifest: AgentManifest,
        sample_task: Task,
    ):
        """Test policy evaluation - allowed."""
        high_trust_manifest.trust_score = 0.85
        
        result = policy_engine.evaluate(
            manifest=high_trust_manifest,
            task=sample_task,
            environment="development",
        )
        
        assert result.allowed is True
        assert len(result.violations) == 0

    @pytest.mark.asyncio
    async def test_evaluate_denied_trust_score(
        self,
        policy_engine: PolicyEngine,
        low_trust_manifest: AgentManifest,
        sample_task: Task,
    ):
        """Test policy denial due to low trust score."""
        low_trust_manifest.trust_score = 0.5
        
        result = policy_engine.evaluate(
            manifest=low_trust_manifest,
            task=sample_task,
            environment="production",  # Production requires 0.7
        )
        
        assert result.allowed is False
        assert any(
            v.violation_type == "trust_score_below_minimum"
            for v in result.violations
        )

    @pytest.mark.asyncio
    async def test_evaluate_denied_capability(
        self, policy_engine: PolicyEngine, sample_task: Task
    ):
        """Test policy denial due to denied capability."""
        # Create policy denying a capability
        policy = Policy(
            id="deny-search",
            name="Deny Search",
            environments=["test"],
            deny_capabilities=["web_search"],
        )
        policy_engine.register_policy(policy)
        
        manifest = AgentManifest(
            id="search-agent",
            name="Search Agent",
            version="1.0.0",
            framework="langgraph",
            capabilities=[
                Capability(
                    name="web_search",
                    type=CapabilityType.TOOL,
                ),
            ],
            trust_score=0.9,
            status=AgentStatus.ACTIVE,
        )
        
        result = policy_engine.evaluate(
            manifest=manifest,
            task=sample_task,
            environment="test",
        )
        
        assert result.allowed is False
        assert any(
            v.violation_type == "denied_capability"
            for v in result.violations
        )

    @pytest.mark.asyncio
    async def test_evaluate_network_denied(
        self, policy_engine: PolicyEngine, sample_task: Task
    ):
        """Test policy denial due to network access."""
        policy = Policy(
            id="no-network",
            name="No Network",
            environments=["isolated"],
            allow_network=False,
        )
        policy_engine.register_policy(policy)
        
        manifest = AgentManifest(
            id="network-agent",
            name="Network Agent",
            version="1.0.0",
            framework="langgraph",
            capabilities=[
                Capability(
                    name="api_call",
                    type=CapabilityType.TOOL,
                    permissions=["network:external"],
                ),
            ],
            trust_score=0.9,
            status=AgentStatus.ACTIVE,
        )
        
        result = policy_engine.evaluate(
            manifest=manifest,
            task=sample_task,
            environment="isolated",
        )
        
        assert result.allowed is False
        assert any(
            v.violation_type == "network_access_denied"
            for v in result.violations
        )

    def test_violation_log(
        self,
        policy_engine: PolicyEngine,
        low_trust_manifest: AgentManifest,
        sample_task: Task,
    ):
        """Test violation logging."""
        low_trust_manifest.trust_score = 0.3
        
        policy_engine.evaluate(
            manifest=low_trust_manifest,
            task=sample_task,
            environment="production",
        )
        
        log = policy_engine.get_violation_log(agent_id=low_trust_manifest.id)
        assert len(log) > 0

    def test_clear_violation_log(self, policy_engine: PolicyEngine):
        """Test clearing violation log."""
        policy_engine.clear_violation_log()
        log = policy_engine.get_violation_log()
        assert len(log) == 0


# =============================================================================
# Sandbox Orchestrator Tests
# =============================================================================


class TestSandboxOrchestrator:
    """Tests for SandboxOrchestrator."""

    @pytest.mark.asyncio
    async def test_select_tier_wasm(self, sandbox_orchestrator: SandboxOrchestrator):
        """Test tier selection for high trust."""
        factors = TrustFactors(
            publisher_trust=1.0,
            audit_status=1.0,
            community_trust=1.0,
            permission_analysis=1.0,
            historical_behavior=1.0,
        )
        trust_score = TrustScore.compute(factors)
        
        tier = sandbox_orchestrator.select_tier(trust_score)
        assert tier == SandboxTier.WASM

    @pytest.mark.asyncio
    async def test_select_tier_blocked(self, sandbox_orchestrator: SandboxOrchestrator):
        """Test tier selection for very low trust."""
        factors = TrustFactors(
            publisher_trust=0.1,
            audit_status=0.1,
            community_trust=0.1,
            permission_analysis=0.1,
            historical_behavior=0.1,
        )
        trust_score = TrustScore.compute(factors)
        
        tier = sandbox_orchestrator.select_tier(trust_score)
        assert tier == SandboxTier.BLOCKED

    @pytest.mark.asyncio
    async def test_execute_blocked(
        self,
        sandbox_orchestrator: SandboxOrchestrator,
        low_trust_manifest: AgentManifest,
        sample_task: Task,
    ):
        """Test execution blocked for low trust."""
        await sandbox_orchestrator.initialize()
        
        # Create very low trust score
        factors = TrustFactors(
            publisher_trust=0.1,
            audit_status=0.1,
            community_trust=0.1,
            permission_analysis=0.1,
            historical_behavior=0.1,
        )
        trust_score = TrustScore.compute(factors)
        
        result = await sandbox_orchestrator.execute(
            agent=low_trust_manifest,
            task=sample_task,
            trust_score=trust_score,
        )
        
        assert result.status.value == "failed"
        assert result.error is not None
        assert result.error.code == "EXECUTION_BLOCKED"
        
        await sandbox_orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_execute_success(
        self,
        sandbox_orchestrator: SandboxOrchestrator,
        high_trust_manifest: AgentManifest,
        sample_task: Task,
    ):
        """Test successful execution."""
        await sandbox_orchestrator.initialize()
        
        # Create high trust score
        factors = TrustFactors(
            publisher_trust=1.0,
            audit_status=1.0,
            community_trust=1.0,
            permission_analysis=1.0,
            historical_behavior=1.0,
        )
        trust_score = TrustScore.compute(factors)
        
        result = await sandbox_orchestrator.execute(
            agent=high_trust_manifest,
            task=sample_task,
            trust_score=trust_score,
        )
        
        assert result.status.value == "completed"
        assert result.output is not None
        assert result.metrics is not None
        
        await sandbox_orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_get_available_tiers(
        self, sandbox_orchestrator: SandboxOrchestrator
    ):
        """Test getting available tiers."""
        tiers = sandbox_orchestrator.get_available_tiers()
        assert SandboxTier.WASM in tiers

    @pytest.mark.asyncio
    async def test_is_tier_available(
        self, sandbox_orchestrator: SandboxOrchestrator
    ):
        """Test checking tier availability."""
        assert sandbox_orchestrator.is_tier_available(SandboxTier.WASM) is True
        assert sandbox_orchestrator.is_tier_available(SandboxTier.GVISOR) is False


# =============================================================================
# Stub Sandbox Tests
# =============================================================================


class TestStubSandbox:
    """Tests for StubSandbox."""

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test sandbox initialization."""
        sandbox = StubSandbox()
        await sandbox.initialize(SandboxConfig())
        assert sandbox.is_available() is True

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test sandbox execution."""
        sandbox = StubSandbox()
        await sandbox.initialize(SandboxConfig())
        
        result = await sandbox.execute(
            code="test-agent",
            input_data={"query": "hello"},
        )
        
        assert result.success is True
        assert result.output is not None
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_terminate(self):
        """Test sandbox termination."""
        sandbox = StubSandbox()
        await sandbox.initialize(SandboxConfig())
        await sandbox.terminate()
        # Should not raise

    def test_expected_overhead(self):
        """Test expected overhead property."""
        sandbox = StubSandbox()
        assert sandbox.expected_overhead_ms == 1


# =============================================================================
# Sandbox Pool Tests
# =============================================================================


class TestSandboxPool:
    """Tests for SandboxPool."""

    @pytest.mark.asyncio
    async def test_pool_initialization(self):
        """Test pool initialization."""
        pool = SandboxPool(wasm_pool_size=2)
        pool.register_factory(SandboxTier.WASM, StubSandbox)
        
        await pool.initialize()
        
        # Should be able to acquire
        sandbox = await pool.acquire(SandboxTier.WASM)
        assert sandbox is not None
        
        await pool.release(sandbox)
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_acquire_release(self):
        """Test acquiring and releasing sandboxes."""
        pool = SandboxPool(wasm_pool_size=2)
        pool.register_factory(SandboxTier.WASM, StubSandbox)
        await pool.initialize()
        
        # Acquire two sandboxes
        s1 = await pool.acquire(SandboxTier.WASM)
        s2 = await pool.acquire(SandboxTier.WASM)
        
        assert s1 is not None
        assert s2 is not None
        
        # Release both
        await pool.release(s1)
        await pool.release(s2)
        
        await pool.shutdown()

    @pytest.mark.asyncio
    async def test_pool_unavailable_tier(self):
        """Test acquiring unavailable tier raises error."""
        pool = SandboxPool()
        await pool.initialize()
        
        with pytest.raises(ValueError):
            await pool.acquire(SandboxTier.GVISOR)
        
        await pool.shutdown()


# =============================================================================
# Sandbox Config Tests
# =============================================================================


class TestSandboxConfig:
    """Tests for SandboxConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SandboxConfig()
        
        assert config.max_memory_bytes == 512 * 1024 * 1024
        assert config.max_cpu_time_ms == 60000
        assert config.max_execution_time_ms == 120000
        assert config.allow_network is False
        assert config.allow_filesystem is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SandboxConfig(
            max_memory_bytes=1024 * 1024 * 1024,
            allow_network=True,
            allowed_hosts=["api.example.com"],
        )
        
        assert config.max_memory_bytes == 1024 * 1024 * 1024
        assert config.allow_network is True
        assert "api.example.com" in config.allowed_hosts
