"""
AI Workflow Fabric - Trust Scoring Engine

This module implements the trust scoring algorithm defined in the
ASP specification for computing agent trustworthiness.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from awf.adapters.base import TrustScorer
from awf.core.types import (
    AgentManifest,
    Event,
    SandboxTier,
    TrustFactors,
    TrustScore,
)


@dataclass
class PublisherInfo:
    """Information about a publisher for trust computation."""
    
    id: str
    verified: bool = False
    reputation_score: float = 0.5  # 0-1
    agent_count: int = 0
    violations: int = 0


@dataclass
class AuditInfo:
    """Information about security audits."""
    
    audited: bool = False
    auditor: Optional[str] = None
    audit_date: Optional[datetime] = None
    findings: int = 0
    critical_findings: int = 0


@dataclass
class CommunityMetrics:
    """Community trust metrics."""
    
    total_uses: int = 0
    unique_users: int = 0
    positive_ratings: int = 0
    negative_ratings: int = 0
    abuse_reports: int = 0


class TrustScoringEngine(TrustScorer):
    """
    Engine for computing and managing agent trust scores.
    
    Implements the trust scoring algorithm from the ASP specification:
    - Publisher Trust (25%)
    - Audit Status (25%)
    - Community Trust (20%)
    - Permission Analysis (15%)
    - Historical Behavior (15%)
    
    Example usage:
        ```python
        engine = TrustScoringEngine()
        
        # Compute trust score for a manifest
        trust_score = await engine.compute_score(manifest)
        
        print(f"Score: {trust_score.score}")
        print(f"Sandbox Tier: {trust_score.sandbox_tier}")
        
        # Update score based on behavior
        new_score = await engine.update_score(
            agent_id="my-agent",
            event=abuse_report_event,
        )
        ```
    """
    
    # Weight factors as defined in spec
    WEIGHT_PUBLISHER = 0.25
    WEIGHT_AUDIT = 0.25
    WEIGHT_COMMUNITY = 0.20
    WEIGHT_PERMISSIONS = 0.15
    WEIGHT_BEHAVIOR = 0.15
    
    # Sandbox tier thresholds
    TIER_WASM = 0.90
    TIER_GVISOR = 0.70
    TIER_GVISOR_STRICT = 0.40
    
    def __init__(self):
        """Initialize the trust scoring engine."""
        self._publisher_cache: Dict[str, PublisherInfo] = {}
        self._audit_cache: Dict[str, AuditInfo] = {}
        self._community_cache: Dict[str, CommunityMetrics] = {}
        self._behavior_cache: Dict[str, List[Event]] = {}
        self._score_cache: Dict[str, TrustScore] = {}
    
    async def compute_score(self, manifest: AgentManifest) -> TrustScore:
        """
        Compute trust score for an agent manifest.
        
        Args:
            manifest: The agent manifest to score
        
        Returns:
            TrustScore with computed values
        """
        # Compute individual factors
        publisher_score = await self._compute_publisher_trust(manifest)
        audit_score = await self._compute_audit_status(manifest)
        community_score = await self._compute_community_trust(manifest)
        permission_score = self._compute_permission_analysis(manifest)
        behavior_score = await self._compute_historical_behavior(manifest)
        
        # Build trust factors
        factors = TrustFactors(
            publisher_trust=publisher_score,
            audit_status=audit_score,
            community_trust=community_score,
            permission_analysis=permission_score,
            historical_behavior=behavior_score,
        )
        
        # Compute final score
        trust_score = TrustScore.compute(factors)
        
        # Cache the score
        self._score_cache[manifest.id] = trust_score
        
        return trust_score
    
    async def update_score(
        self, agent_id: str, event: Event
    ) -> Optional[TrustScore]:
        """
        Update trust score based on a new event.
        
        Args:
            agent_id: The agent to update
            event: The event that triggered the update
        
        Returns:
            Updated TrustScore, or None if agent not found
        """
        # Store event in behavior history
        if agent_id not in self._behavior_cache:
            self._behavior_cache[agent_id] = []
        self._behavior_cache[agent_id].append(event)
        
        # If we have a cached score, update it
        if agent_id in self._score_cache:
            old_score = self._score_cache[agent_id]
            
            # Adjust behavior score based on event
            behavior_adjustment = self._get_behavior_adjustment(event)
            new_behavior = max(0.0, min(1.0, 
                old_score.factors.historical_behavior + behavior_adjustment
            ))
            
            # Recompute with adjusted behavior
            new_factors = TrustFactors(
                publisher_trust=old_score.factors.publisher_trust,
                audit_status=old_score.factors.audit_status,
                community_trust=old_score.factors.community_trust,
                permission_analysis=old_score.factors.permission_analysis,
                historical_behavior=new_behavior,
            )
            
            new_score = TrustScore.compute(new_factors)
            self._score_cache[agent_id] = new_score
            
            return new_score
        
        return None
    
    async def _compute_publisher_trust(self, manifest: AgentManifest) -> float:
        """Compute publisher trust score (0-1)."""
        if not manifest.publisher:
            return 0.3  # Unknown publisher gets low score
        
        # Check cache
        if manifest.publisher in self._publisher_cache:
            info = self._publisher_cache[manifest.publisher]
        else:
            # In production, this would query a publisher database
            info = PublisherInfo(id=manifest.publisher)
        
        score = 0.5  # Base score
        
        # Verified publishers get bonus
        if info.verified:
            score += 0.3
        
        # Reputation adjustment
        score += (info.reputation_score - 0.5) * 0.2
        
        # Violation penalty
        if info.violations > 0:
            score -= min(0.3, info.violations * 0.1)
        
        return max(0.0, min(1.0, score))
    
    async def _compute_audit_status(self, manifest: AgentManifest) -> float:
        """Compute audit status score (0-1)."""
        agent_id = manifest.id
        
        # Check cache
        if agent_id in self._audit_cache:
            info = self._audit_cache[agent_id]
        else:
            # Check manifest for audit info
            audit_status = manifest.audit_status
            if audit_status == "audited":
                info = AuditInfo(audited=True)
            else:
                info = AuditInfo(audited=False)
        
        if not info.audited:
            return 0.3  # Unaudited agents get low score
        
        score = 0.8  # Audited base score
        
        # Recent audit bonus
        if info.audit_date:
            days_since = (datetime.utcnow() - info.audit_date).days
            if days_since < 90:
                score += 0.15
            elif days_since < 180:
                score += 0.1
            elif days_since > 365:
                score -= 0.1
        
        # Findings penalty
        if info.critical_findings > 0:
            score -= min(0.3, info.critical_findings * 0.15)
        score -= min(0.2, info.findings * 0.05)
        
        return max(0.0, min(1.0, score))
    
    async def _compute_community_trust(self, manifest: AgentManifest) -> float:
        """Compute community trust score (0-1)."""
        agent_id = manifest.id
        
        # Check cache
        if agent_id in self._community_cache:
            metrics = self._community_cache[agent_id]
        else:
            metrics = CommunityMetrics()
        
        # New agents start with neutral score
        if metrics.total_uses == 0:
            return 0.5
        
        # Calculate rating ratio
        total_ratings = metrics.positive_ratings + metrics.negative_ratings
        if total_ratings > 0:
            positive_ratio = metrics.positive_ratings / total_ratings
        else:
            positive_ratio = 0.5
        
        score = 0.3 + (positive_ratio * 0.5)
        
        # Usage bonus (more usage = more trust)
        if metrics.unique_users >= 100:
            score += 0.15
        elif metrics.unique_users >= 10:
            score += 0.1
        elif metrics.unique_users >= 3:
            score += 0.05
        
        # Abuse report penalty
        if metrics.abuse_reports > 0:
            score -= min(0.4, metrics.abuse_reports * 0.1)
        
        return max(0.0, min(1.0, score))
    
    def _compute_permission_analysis(self, manifest: AgentManifest) -> float:
        """Compute permission analysis score (0-1)."""
        # Start with high score
        score = 1.0
        
        # Analyze capabilities for risky permissions
        risky_permissions = {
            "network:external": 0.1,
            "filesystem:write": 0.15,
            "filesystem:read": 0.05,
            "process:execute": 0.25,
            "database:write": 0.1,
        }
        
        for cap in manifest.capabilities:
            for permission in cap.permissions:
                if permission in risky_permissions:
                    score -= risky_permissions[permission]
        
        # Many capabilities = higher risk
        cap_count = len(manifest.capabilities)
        if cap_count > 10:
            score -= 0.1
        elif cap_count > 5:
            score -= 0.05
        
        return max(0.0, min(1.0, score))
    
    async def _compute_historical_behavior(self, manifest: AgentManifest) -> float:
        """Compute historical behavior score (0-1)."""
        agent_id = manifest.id
        
        # Check behavior history
        if agent_id not in self._behavior_cache:
            return 0.7  # New agents get moderate score
        
        events = self._behavior_cache[agent_id]
        if not events:
            return 0.7
        
        score = 0.8  # Base score for agents with history
        
        # Analyze recent events
        for event in events[-100:]:  # Last 100 events
            adjustment = self._get_behavior_adjustment(event)
            score += adjustment
        
        return max(0.0, min(1.0, score))
    
    def _get_behavior_adjustment(self, event: Event) -> float:
        """Get trust score adjustment for an event."""
        event_adjustments = {
            "task.completed": 0.001,  # Small positive for successful execution
            "task.failed": -0.002,  # Small negative for failures
            "policy.violation": -0.05,  # Larger negative for policy violations
            "abuse.reported": -0.1,  # Significant negative for abuse
            "audit.passed": 0.1,  # Bonus for passing audit
        }
        
        return event_adjustments.get(event.type.value, 0.0)
    
    def get_sandbox_tier(self, score: float) -> SandboxTier:
        """
        Determine sandbox tier based on trust score.
        
        Args:
            score: Trust score (0-1)
        
        Returns:
            Appropriate SandboxTier
        """
        if score >= self.TIER_WASM:
            return SandboxTier.WASM
        elif score >= self.TIER_GVISOR:
            return SandboxTier.GVISOR
        elif score >= self.TIER_GVISOR_STRICT:
            return SandboxTier.GVISOR_STRICT
        else:
            return SandboxTier.BLOCKED
    
    def register_publisher(self, info: PublisherInfo) -> None:
        """Register publisher information for trust computation."""
        self._publisher_cache[info.id] = info
    
    def register_audit(self, agent_id: str, info: AuditInfo) -> None:
        """Register audit information for an agent."""
        self._audit_cache[agent_id] = info
    
    def update_community_metrics(
        self, agent_id: str, metrics: CommunityMetrics
    ) -> None:
        """Update community metrics for an agent."""
        self._community_cache[agent_id] = metrics
    
    def get_cached_score(self, agent_id: str) -> Optional[TrustScore]:
        """Get cached trust score if available."""
        return self._score_cache.get(agent_id)
    
    def invalidate_cache(self, agent_id: str) -> None:
        """Invalidate cached score for an agent."""
        if agent_id in self._score_cache:
            del self._score_cache[agent_id]
