"""
AI Workflow Fabric - Policy Engine

This module implements the policy enforcement engine for controlling
agent execution based on security and governance rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from awf.core.types import (
    AgentManifest,
    Policy,
    PolicyViolation,
    Task,
    TrustScore,
)


@dataclass
class PolicyEvaluationResult:
    """Result of policy evaluation."""
    
    allowed: bool
    policy_id: Optional[str] = None
    violations: List[PolicyViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "allowed": self.allowed,
            "policyId": self.policy_id,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": self.warnings,
        }


class PolicyEngine:
    """
    Engine for evaluating and enforcing policies on agent execution.
    
    Policies can restrict:
    - Minimum trust score requirements
    - Allowed/denied capabilities
    - Execution time limits
    - Resource limits (memory, network, filesystem)
    - Environment-specific rules
    
    Example usage:
        ```python
        engine = PolicyEngine()
        
        # Register policies
        engine.register_policy(production_policy)
        engine.register_policy(staging_policy)
        
        # Evaluate before execution
        result = engine.evaluate(
            manifest=agent_manifest,
            task=task,
            trust_score=trust_score,
            environment="production",
        )
        
        if result.allowed:
            # Proceed with execution
            pass
        else:
            # Block execution
            for violation in result.violations:
                print(f"Violation: {violation.violation_type}")
        ```
    """
    
    def __init__(self):
        """Initialize the policy engine."""
        self._policies: Dict[str, Policy] = {}
        self._violation_log: List[PolicyViolation] = []
    
    def register_policy(self, policy: Policy) -> None:
        """
        Register a policy.
        
        Args:
            policy: The policy to register
        """
        self._policies[policy.id] = policy
    
    def unregister_policy(self, policy_id: str) -> bool:
        """
        Unregister a policy.
        
        Args:
            policy_id: The ID of the policy to remove
        
        Returns:
            True if removed, False if not found
        """
        if policy_id in self._policies:
            del self._policies[policy_id]
            return True
        return False
    
    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Get a policy by ID."""
        return self._policies.get(policy_id)
    
    def list_policies(self) -> List[Policy]:
        """List all registered policies."""
        return list(self._policies.values())
    
    def evaluate(
        self,
        manifest: AgentManifest,
        task: Task,
        trust_score: Optional[TrustScore] = None,
        environment: str = "development",
    ) -> PolicyEvaluationResult:
        """
        Evaluate all applicable policies for an execution request.
        
        Args:
            manifest: The agent manifest
            task: The task to execute
            trust_score: The agent's trust score
            environment: The execution environment
        
        Returns:
            PolicyEvaluationResult with allow/deny decision
        """
        violations: List[PolicyViolation] = []
        warnings: List[str] = []
        blocking_policy: Optional[str] = None
        
        # Find applicable policies
        applicable_policies = self._get_applicable_policies(
            manifest, environment
        )
        
        for policy in applicable_policies:
            if not policy.enabled:
                continue
            
            # Check trust score
            if policy.min_trust_score is not None:
                score = trust_score.score if trust_score else manifest.trust_score
                if score is None or score < policy.min_trust_score:
                    violations.append(PolicyViolation(
                        policy_id=policy.id,
                        policy_name=policy.name,
                        agent_id=manifest.id,
                        violation_type="trust_score_below_minimum",
                        details={
                            "required": policy.min_trust_score,
                            "actual": score,
                        },
                    ))
                    blocking_policy = policy.id
            
            # Check required capabilities
            for required_cap in policy.require_capabilities:
                agent_caps = {c.name for c in manifest.capabilities}
                if required_cap not in agent_caps:
                    violations.append(PolicyViolation(
                        policy_id=policy.id,
                        policy_name=policy.name,
                        agent_id=manifest.id,
                        violation_type="missing_required_capability",
                        details={"capability": required_cap},
                    ))
                    blocking_policy = policy.id
            
            # Check denied capabilities
            for denied_cap in policy.deny_capabilities:
                for cap in manifest.capabilities:
                    if denied_cap in cap.permissions or cap.name == denied_cap:
                        violations.append(PolicyViolation(
                            policy_id=policy.id,
                            policy_name=policy.name,
                            agent_id=manifest.id,
                            violation_type="denied_capability",
                            details={"capability": denied_cap},
                        ))
                        blocking_policy = policy.id
            
            # Check execution time limits
            if policy.max_execution_time_ms and task.timeout_ms:
                if task.timeout_ms > policy.max_execution_time_ms:
                    warnings.append(
                        f"Task timeout ({task.timeout_ms}ms) exceeds policy "
                        f"limit ({policy.max_execution_time_ms}ms). "
                        "Will be capped to policy limit."
                    )
            
            # Check network access
            if not policy.allow_network:
                for cap in manifest.capabilities:
                    if any("network" in p for p in cap.permissions):
                        violations.append(PolicyViolation(
                            policy_id=policy.id,
                            policy_name=policy.name,
                            agent_id=manifest.id,
                            violation_type="network_access_denied",
                            details={"capability": cap.name},
                        ))
                        blocking_policy = policy.id
            
            # Check filesystem access
            if not policy.allow_filesystem:
                for cap in manifest.capabilities:
                    if any("filesystem" in p for p in cap.permissions):
                        violations.append(PolicyViolation(
                            policy_id=policy.id,
                            policy_name=policy.name,
                            agent_id=manifest.id,
                            violation_type="filesystem_access_denied",
                            details={"capability": cap.name},
                        ))
                        blocking_policy = policy.id
        
        # Log violations
        self._violation_log.extend(violations)
        
        # Determine final result
        allowed = len(violations) == 0
        
        return PolicyEvaluationResult(
            allowed=allowed,
            policy_id=blocking_policy,
            violations=violations,
            warnings=warnings,
        )
    
    def _get_applicable_policies(
        self, manifest: AgentManifest, environment: str
    ) -> List[Policy]:
        """Get policies applicable to a specific agent and environment."""
        applicable: List[Policy] = []
        
        for policy in self._policies.values():
            # Check environment match
            if policy.environments:
                if environment not in policy.environments:
                    continue
            
            # Check agent match
            if policy.agent_ids:
                if manifest.id not in policy.agent_ids:
                    continue
            
            applicable.append(policy)
        
        return applicable
    
    def get_violation_log(
        self,
        agent_id: Optional[str] = None,
        policy_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[PolicyViolation]:
        """
        Get violation history.
        
        Args:
            agent_id: Filter by agent ID
            policy_id: Filter by policy ID
            limit: Maximum number of results
        
        Returns:
            List of PolicyViolation objects
        """
        violations = self._violation_log.copy()
        
        if agent_id:
            violations = [v for v in violations if v.agent_id == agent_id]
        
        if policy_id:
            violations = [v for v in violations if v.policy_id == policy_id]
        
        # Most recent first
        violations.reverse()
        
        return violations[:limit]
    
    def clear_violation_log(self) -> None:
        """Clear the violation log."""
        self._violation_log.clear()
    
    def create_default_policies(self) -> None:
        """Create default security policies."""
        # Production policy - strict
        production = Policy(
            id="default-production",
            name="Default Production Policy",
            environments=["production"],
            min_trust_score=0.7,
            deny_capabilities=["process:execute"],
            max_execution_time_ms=300000,  # 5 minutes
            allow_network=True,
            allow_filesystem=False,
            description="Default strict policy for production environment",
        )
        self.register_policy(production)
        
        # Staging policy - moderate
        staging = Policy(
            id="default-staging",
            name="Default Staging Policy",
            environments=["staging"],
            min_trust_score=0.5,
            max_execution_time_ms=600000,  # 10 minutes
            allow_network=True,
            allow_filesystem=True,
            description="Default moderate policy for staging environment",
        )
        self.register_policy(staging)
        
        # Development policy - permissive
        development = Policy(
            id="default-development",
            name="Default Development Policy",
            environments=["development", "local"],
            min_trust_score=0.0,
            allow_network=True,
            allow_filesystem=True,
            description="Permissive policy for development",
        )
        self.register_policy(development)
