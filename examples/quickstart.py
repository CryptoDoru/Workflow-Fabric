#!/usr/bin/env python3
"""
AI Workflow Fabric - Quickstart Example

This example demonstrates the basic usage of AWF:
1. Create an agent manifest
2. Register it with the in-memory registry
3. Compute trust score
4. Search for agents by capability

Run with:
    python examples/quickstart.py
"""

import asyncio
from datetime import datetime

# AWF imports
from awf.core.types import (
    AgentManifest,
    AgentStatus,
    Capability,
    CapabilityType,
    Task,
)
from awf.registry.memory import InMemoryRegistry
from awf.security.trust import TrustScoringEngine
from awf.security.policy import PolicyEngine


async def main():
    """Main quickstart example."""
    print("=" * 60)
    print("AI Workflow Fabric - Quickstart Example")
    print("=" * 60)
    print()
    
    # -------------------------------------------------------------------------
    # Step 1: Create the registry and engines
    # -------------------------------------------------------------------------
    print("Step 1: Initializing AWF components...")
    
    registry = InMemoryRegistry()
    trust_engine = TrustScoringEngine()
    policy_engine = PolicyEngine()
    
    # Create default policies
    policy_engine.create_default_policies()
    
    print("  - In-memory registry: OK")
    print("  - Trust scoring engine: OK")
    print("  - Policy engine: OK (3 default policies)")
    print()
    
    # -------------------------------------------------------------------------
    # Step 2: Create agent manifests
    # -------------------------------------------------------------------------
    print("Step 2: Creating agent manifests...")
    
    # Web Search Agent
    web_search_agent = AgentManifest(
        id="web-search-agent",
        name="Web Search Agent",
        version="1.0.0",
        framework="langgraph",
        framework_version="0.2.0",
        description="Searches the web for information using multiple search engines",
        capabilities=[
            Capability(
                name="web_search",
                type=CapabilityType.TOOL,
                description="Search the web for information",
                permissions=["network:external"],
            ),
            Capability(
                name="summarize",
                type=CapabilityType.REASONING,
                description="Summarize search results",
            ),
        ],
        tags=["search", "web", "research"],
        publisher="awf-examples",
        status=AgentStatus.ACTIVE,
    )
    
    # Code Analysis Agent
    code_analysis_agent = AgentManifest(
        id="code-analysis-agent",
        name="Code Analysis Agent",
        version="2.1.0",
        framework="crewai",
        framework_version="0.30.0",
        description="Analyzes code for bugs, security issues, and improvements",
        capabilities=[
            Capability(
                name="code_review",
                type=CapabilityType.REASONING,
                description="Review code for issues",
            ),
            Capability(
                name="security_scan",
                type=CapabilityType.TOOL,
                description="Scan for security vulnerabilities",
                permissions=["filesystem:read"],
            ),
        ],
        tags=["code", "analysis", "security"],
        publisher="awf-examples",
        audit_status="audited",
        status=AgentStatus.ACTIVE,
    )
    
    # Data Processing Agent
    data_agent = AgentManifest(
        id="data-processing-agent",
        name="Data Processing Agent",
        version="1.5.0",
        framework="langgraph",
        description="Processes and transforms data from various sources",
        capabilities=[
            Capability(
                name="data_transform",
                type=CapabilityType.TOOL,
                description="Transform data between formats",
            ),
            Capability(
                name="data_validate",
                type=CapabilityType.REASONING,
                description="Validate data quality",
            ),
        ],
        tags=["data", "etl", "processing"],
        status=AgentStatus.ACTIVE,
    )
    
    print(f"  - Created: {web_search_agent.name}")
    print(f"  - Created: {code_analysis_agent.name}")
    print(f"  - Created: {data_agent.name}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 3: Compute trust scores
    # -------------------------------------------------------------------------
    print("Step 3: Computing trust scores...")
    
    for agent in [web_search_agent, code_analysis_agent, data_agent]:
        trust_score = await trust_engine.compute_score(agent)
        agent.trust_score = trust_score.score
        
        print(f"  - {agent.name}:")
        print(f"      Score: {trust_score.score:.2f}")
        print(f"      Sandbox Tier: {trust_score.sandbox_tier.value}")
        print(f"      Factors:")
        print(f"        Publisher: {trust_score.factors.publisher_trust:.2f}")
        print(f"        Audit: {trust_score.factors.audit_status:.2f}")
        print(f"        Community: {trust_score.factors.community_trust:.2f}")
        print(f"        Permissions: {trust_score.factors.permission_analysis:.2f}")
        print(f"        Behavior: {trust_score.factors.historical_behavior:.2f}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 4: Register agents
    # -------------------------------------------------------------------------
    print("Step 4: Registering agents in registry...")
    
    await registry.register(web_search_agent)
    await registry.register(code_analysis_agent)
    await registry.register(data_agent)
    
    count = await registry.count()
    print(f"  - Registered {count} agents")
    print()
    
    # -------------------------------------------------------------------------
    # Step 5: Search for agents
    # -------------------------------------------------------------------------
    print("Step 5: Searching for agents...")
    
    # Search by capability
    print("  - Searching for agents with 'web_search' capability:")
    results = await registry.search(capabilities=["web_search"])
    for agent in results:
        print(f"      Found: {agent.name} (trust: {agent.trust_score:.2f})")
    
    # Search by framework
    print("  - Searching for LangGraph agents:")
    results = await registry.search(framework="langgraph")
    for agent in results:
        print(f"      Found: {agent.name}")
    
    # Search by tag
    print("  - Searching for agents tagged 'security':")
    results = await registry.search(tags=["security"])
    for agent in results:
        print(f"      Found: {agent.name}")
    
    # Search by trust score
    print("  - Searching for agents with trust >= 0.6:")
    results = await registry.search(min_trust_score=0.6)
    for agent in results:
        print(f"      Found: {agent.name} (trust: {agent.trust_score:.2f})")
    print()
    
    # -------------------------------------------------------------------------
    # Step 6: Check policies before execution
    # -------------------------------------------------------------------------
    print("Step 6: Checking policies before execution...")
    
    # Create a sample task
    task = Task(
        agent_id=web_search_agent.id,
        input={"query": "What is AI Workflow Fabric?"},
        timeout_ms=30000,
    )
    
    # Evaluate against production policy
    trust_score = await trust_engine.compute_score(web_search_agent)
    result = policy_engine.evaluate(
        manifest=web_search_agent,
        task=task,
        trust_score=trust_score,
        environment="production",
    )
    
    print(f"  - Agent: {web_search_agent.name}")
    print(f"  - Environment: production")
    print(f"  - Policy Result: {'ALLOWED' if result.allowed else 'DENIED'}")
    if result.violations:
        print(f"  - Violations:")
        for v in result.violations:
            print(f"      {v.violation_type}: {v.details}")
    if result.warnings:
        print(f"  - Warnings:")
        for w in result.warnings:
            print(f"      {w}")
    print()
    
    # Try development environment (more permissive)
    result = policy_engine.evaluate(
        manifest=web_search_agent,
        task=task,
        trust_score=trust_score,
        environment="development",
    )
    
    print(f"  - Environment: development")
    print(f"  - Policy Result: {'ALLOWED' if result.allowed else 'DENIED'}")
    print()
    
    # -------------------------------------------------------------------------
    # Step 7: List all agents
    # -------------------------------------------------------------------------
    print("Step 7: Listing all registered agents...")
    
    all_agents = await registry.list_all()
    print(f"  Total agents: {len(all_agents)}")
    print()
    for agent in all_agents:
        print(f"  [{agent.status.value.upper()}] {agent.name}")
        print(f"           ID: {agent.id}")
        print(f"      Version: {agent.version}")
        print(f"    Framework: {agent.framework}")
        print(f"        Trust: {agent.trust_score:.2f}" if agent.trust_score else "        Trust: N/A")
        print(f" Capabilities: {', '.join(c.name for c in agent.capabilities)}")
        print()
    
    # -------------------------------------------------------------------------
    # Done
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Quickstart complete!")
    print()
    print("Next steps:")
    print("  - See examples/langgraph_example.py for LangGraph integration")
    print("  - See examples/crewai_example.py for CrewAI integration")
    print("  - Run 'uvicorn awf.api.app:app --reload' for REST API")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
