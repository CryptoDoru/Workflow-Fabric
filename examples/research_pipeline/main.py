#!/usr/bin/env python3
"""
AWF Research Pipeline Example

This example demonstrates a multi-agent research workflow that:
1. Searches the web for information
2. Analyzes and extracts key insights
3. Summarizes findings into a report

Run with:
    python examples/research_pipeline/main.py

With observability:
    cd docker/grafana && docker compose up -d
    python examples/research_pipeline/main.py
"""

import asyncio
from datetime import datetime

from awf.core.types import (
    AgentManifest,
    AgentStatus,
    Capability,
    CapabilityType,
)
from awf.registry.memory import InMemoryRegistry
from awf.security.trust import TrustScoringEngine
from awf.orchestration import (
    WorkflowDefinition,
    StepDefinition,
    RetryPolicy,
)


async def create_agents(registry: InMemoryRegistry) -> None:
    """Create and register the research pipeline agents."""
    trust_engine = TrustScoringEngine()
    
    # Web Search Agent - finds relevant information
    search_agent = AgentManifest(
        id="web-search-agent",
        name="Web Search Agent",
        version="1.0.0",
        framework="langgraph",
        description="Searches the web for information using multiple search engines",
        capabilities=[
            Capability(
                name="web_search",
                type=CapabilityType.TOOL,
                description="Search the web for information",
                permissions=["network:external"],
            ),
        ],
        tags=["search", "web", "research"],
        publisher="awf-examples",
        status=AgentStatus.ACTIVE,
    )
    
    # Analysis Agent - extracts insights from search results
    analysis_agent = AgentManifest(
        id="analysis-agent",
        name="Analysis Agent",
        version="1.0.0",
        framework="langgraph",
        description="Analyzes text and extracts key insights",
        capabilities=[
            Capability(
                name="analyze",
                type=CapabilityType.REASONING,
                description="Analyze and extract insights from text",
            ),
            Capability(
                name="categorize",
                type=CapabilityType.REASONING,
                description="Categorize information by topic",
            ),
        ],
        tags=["analysis", "insights", "reasoning"],
        publisher="awf-examples",
        status=AgentStatus.ACTIVE,
    )
    
    # Summarizer Agent - creates final report
    summarizer_agent = AgentManifest(
        id="summarizer-agent",
        name="Summarizer Agent",
        version="1.0.0",
        framework="crewai",
        description="Summarizes information into concise reports",
        capabilities=[
            Capability(
                name="summarize",
                type=CapabilityType.REASONING,
                description="Create concise summaries",
            ),
            Capability(
                name="format_report",
                type=CapabilityType.TOOL,
                description="Format output as structured report",
            ),
        ],
        tags=["summarization", "reports", "writing"],
        publisher="awf-examples",
        status=AgentStatus.ACTIVE,
    )
    
    # Compute trust scores and register
    for agent in [search_agent, analysis_agent, summarizer_agent]:
        trust = await trust_engine.compute_score(agent)
        agent.trust_score = trust.score
        await registry.register(agent)
        print(f"  Registered: {agent.name} (trust: {trust.score:.2f})")


def create_workflow() -> WorkflowDefinition:
    """Create the research pipeline workflow definition."""
    return WorkflowDefinition(
        id="research-pipeline",
        name="Research Pipeline",
        description="Multi-agent workflow for research tasks",
        version="1.0.0",
        steps=[
            # Step 1: Search for information
            StepDefinition(
                id="search",
                agent_id="web-search-agent",
                timeout_ms=30000,
                retry=RetryPolicy(
                    max_attempts=3,
                    backoff_ms=1000,
                    backoff_multiplier=2.0,
                ),
                input_map={
                    "query": "$.input.query",
                    "max_results": "$.input.max_results",
                },
            ),
            # Step 2: Analyze results
            StepDefinition(
                id="analyze",
                agent_id="analysis-agent",
                depends_on=["search"],
                timeout_ms=60000,
                input_map={
                    "search_results": "$.steps.search.output.results",
                    "focus_areas": "$.input.focus_areas",
                },
            ),
            # Step 3: Summarize into report
            StepDefinition(
                id="summarize",
                agent_id="summarizer-agent",
                depends_on=["analyze"],
                timeout_ms=45000,
                input_map={
                    "insights": "$.steps.analyze.output.insights",
                    "format": "$.input.report_format",
                },
            ),
        ],
        metadata={
            "category": "research",
            "estimated_duration_ms": 135000,
        },
    )


async def simulate_agent_execution(step_id: str, input_data: dict) -> dict:
    """Simulate agent execution (replace with real agent calls in production)."""
    await asyncio.sleep(0.5)  # Simulate processing time
    
    if step_id == "search":
        return {
            "results": [
                {"title": "AI Safety Overview", "snippet": "Key concepts in AI safety..."},
                {"title": "Recent AI Developments", "snippet": "Latest advances in AI..."},
                {"title": "AI Ethics Guidelines", "snippet": "Ethical considerations..."},
            ],
            "total_results": 3,
            "search_time_ms": 450,
        }
    elif step_id == "analyze":
        return {
            "insights": [
                {"topic": "Safety", "summary": "AI safety is critical for responsible deployment"},
                {"topic": "Ethics", "summary": "Ethical frameworks guide AI development"},
                {"topic": "Progress", "summary": "Rapid advances require careful consideration"},
            ],
            "confidence": 0.85,
        }
    elif step_id == "summarize":
        return {
            "report": """
# AI Safety Research Summary

## Key Findings
1. AI safety is a critical concern for responsible AI deployment
2. Ethical frameworks provide essential guidance
3. Rapid advances require careful human oversight

## Recommendations
- Implement safety testing before deployment
- Follow established ethical guidelines
- Maintain human oversight mechanisms

Generated: {timestamp}
            """.format(timestamp=datetime.now().isoformat()),
            "word_count": 87,
            "format": "markdown",
        }
    
    return {"status": "completed"}


async def main():
    """Run the research pipeline example."""
    print("=" * 70)
    print("AWF Research Pipeline Example")
    print("=" * 70)
    print()
    
    # Initialize components
    print("1. Initializing AWF components...")
    registry = InMemoryRegistry()
    print("   - Registry: OK")
    print()
    
    # Create and register agents
    print("2. Creating and registering agents...")
    await create_agents(registry)
    print()
    
    # Create workflow
    print("3. Creating workflow definition...")
    workflow = create_workflow()
    print(f"   - Workflow: {workflow.name}")
    print(f"   - Steps: {len(workflow.steps)}")
    for step in workflow.steps:
        deps = f" (depends on: {step.depends_on})" if step.depends_on else ""
        print(f"     - {step.id}: {step.agent_id}{deps}")
    print()
    
    # Define input
    workflow_input = {
        "query": "AI safety best practices",
        "max_results": 10,
        "focus_areas": ["safety", "ethics", "governance"],
        "report_format": "markdown",
    }
    
    print("4. Executing workflow...")
    print(f"   Input: {workflow_input}")
    print()
    
    # Execute steps (simulated)
    results = {}
    for step in workflow.steps:
        print(f"   Executing step: {step.id}...")
        
        # Build input from mappings
        step_input = {}
        for key, mapping in (step.input_map or {}).items():
            if mapping.startswith("$.input."):
                field = mapping.replace("$.input.", "")
                step_input[key] = workflow_input.get(field)
            elif mapping.startswith("$.steps."):
                # Parse step reference
                parts = mapping.replace("$.steps.", "").split(".")
                step_ref = parts[0]
                if step_ref in results:
                    value = results[step_ref]
                    for part in parts[1:]:
                        if isinstance(value, dict):
                            value = value.get(part)
                    step_input[key] = value
        
        # Execute step
        result = await simulate_agent_execution(step.id, step_input)
        results[step.id] = result
        print(f"   - {step.id}: Complete")
    
    print()
    print("5. Workflow completed successfully!")
    print()
    print("-" * 70)
    print("FINAL REPORT:")
    print("-" * 70)
    print(results.get("summarize", {}).get("report", "No report generated"))
    print("-" * 70)
    print()
    
    # Show execution summary
    print("Execution Summary:")
    print(f"  - Steps executed: {len(results)}")
    print(f"  - Search results: {results.get('search', {}).get('total_results', 0)}")
    print(f"  - Insights extracted: {len(results.get('analyze', {}).get('insights', []))}")
    print(f"  - Report format: {results.get('summarize', {}).get('format', 'unknown')}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
