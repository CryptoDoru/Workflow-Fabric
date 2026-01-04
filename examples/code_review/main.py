#!/usr/bin/env python3
"""
AWF Code Review Pipeline Example

This example demonstrates a multi-agent code review workflow that:
1. Parses code files for analysis
2. Checks for security vulnerabilities
3. Reviews code quality and style
4. Generates a comprehensive review report

Run with:
    python examples/code_review/main.py
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


async def create_agents(registry: InMemoryRegistry) -> None:
    """Create and register the code review agents."""
    trust_engine = TrustScoringEngine()
    
    # Code Parser Agent
    parser_agent = AgentManifest(
        id="code-parser-agent",
        name="Code Parser Agent",
        version="1.0.0",
        framework="langgraph",
        description="Parses code files and extracts structure",
        capabilities=[
            Capability(
                name="parse_code",
                type=CapabilityType.TOOL,
                description="Parse code into AST representation",
                permissions=["filesystem:read"],
            ),
            Capability(
                name="extract_functions",
                type=CapabilityType.TOOL,
                description="Extract function signatures and bodies",
            ),
        ],
        tags=["parsing", "code", "analysis"],
        publisher="awf-examples",
        status=AgentStatus.ACTIVE,
    )
    
    # Security Scanner Agent
    security_agent = AgentManifest(
        id="security-scanner-agent",
        name="Security Scanner Agent",
        version="2.0.0",
        framework="crewai",
        description="Scans code for security vulnerabilities",
        capabilities=[
            Capability(
                name="scan_vulnerabilities",
                type=CapabilityType.TOOL,
                description="Scan for common vulnerabilities",
            ),
            Capability(
                name="check_dependencies",
                type=CapabilityType.TOOL,
                description="Check for vulnerable dependencies",
            ),
        ],
        tags=["security", "vulnerabilities", "scanning"],
        publisher="awf-examples",
        audit_status="audited",  # Higher trust due to audit
        status=AgentStatus.ACTIVE,
    )
    
    # Quality Reviewer Agent
    quality_agent = AgentManifest(
        id="quality-reviewer-agent",
        name="Quality Reviewer Agent",
        version="1.5.0",
        framework="langgraph",
        description="Reviews code quality and suggests improvements",
        capabilities=[
            Capability(
                name="review_quality",
                type=CapabilityType.REASONING,
                description="Review code quality patterns",
            ),
            Capability(
                name="suggest_improvements",
                type=CapabilityType.REASONING,
                description="Suggest code improvements",
            ),
        ],
        tags=["quality", "review", "best-practices"],
        publisher="awf-examples",
        status=AgentStatus.ACTIVE,
    )
    
    # Report Generator Agent
    reporter_agent = AgentManifest(
        id="report-generator-agent",
        name="Report Generator Agent",
        version="1.0.0",
        framework="crewai",
        description="Generates comprehensive review reports",
        capabilities=[
            Capability(
                name="generate_report",
                type=CapabilityType.TOOL,
                description="Generate formatted review report",
            ),
            Capability(
                name="prioritize_issues",
                type=CapabilityType.REASONING,
                description="Prioritize issues by severity",
            ),
        ],
        tags=["reporting", "documentation"],
        publisher="awf-examples",
        status=AgentStatus.ACTIVE,
    )
    
    # Register all agents
    for agent in [parser_agent, security_agent, quality_agent, reporter_agent]:
        trust = await trust_engine.compute_score(agent)
        agent.trust_score = trust.score
        await registry.register(agent)
        print(f"  Registered: {agent.name} (trust: {trust.score:.2f})")


async def simulate_code_review(code: str) -> dict:
    """Simulate a complete code review process."""
    # Simulate parsing
    await asyncio.sleep(0.3)
    parse_result = {
        "file": "example.py",
        "language": "python",
        "lines": code.count("\n") + 1,
        "functions": ["process_data", "validate_input", "save_to_db"],
        "classes": ["DataProcessor"],
    }
    
    # Simulate security scan
    await asyncio.sleep(0.3)
    security_result = {
        "vulnerabilities": [
            {
                "id": "SEC-001",
                "severity": "HIGH",
                "type": "SQL Injection",
                "line": 42,
                "description": "Unsanitized user input in SQL query",
                "recommendation": "Use parameterized queries",
            },
            {
                "id": "SEC-002",
                "severity": "MEDIUM",
                "type": "Hardcoded Secret",
                "line": 15,
                "description": "API key hardcoded in source",
                "recommendation": "Use environment variables",
            },
        ],
        "dependencies_checked": 12,
        "vulnerable_dependencies": ["requests==2.25.0"],
    }
    
    # Simulate quality review
    await asyncio.sleep(0.3)
    quality_result = {
        "issues": [
            {
                "id": "QUAL-001",
                "type": "Code Smell",
                "severity": "LOW",
                "description": "Function too long (> 50 lines)",
                "line": 25,
            },
            {
                "id": "QUAL-002",
                "type": "Missing Documentation",
                "severity": "MEDIUM",
                "description": "Public function lacks docstring",
                "line": 10,
            },
        ],
        "suggestions": [
            "Consider splitting process_data into smaller functions",
            "Add type hints for better code clarity",
            "Use dataclasses for data structures",
        ],
        "complexity_score": 7.2,
    }
    
    return {
        "parse": parse_result,
        "security": security_result,
        "quality": quality_result,
    }


def generate_report(results: dict) -> str:
    """Generate a formatted code review report."""
    security = results["security"]
    quality = results["quality"]
    parse = results["parse"]
    
    high_severity = len([v for v in security["vulnerabilities"] if v["severity"] == "HIGH"])
    medium_severity = len([v for v in security["vulnerabilities"] if v["severity"] == "MEDIUM"])
    
    report = f"""
================================================================================
                        CODE REVIEW REPORT
================================================================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
File: {parse["file"]}
Language: {parse["language"]}
Lines: {parse["lines"]}

--------------------------------------------------------------------------------
                          EXECUTIVE SUMMARY
--------------------------------------------------------------------------------
Security Issues:  {len(security["vulnerabilities"])} ({high_severity} HIGH, {medium_severity} MEDIUM)
Quality Issues:   {len(quality["issues"])}
Complexity Score: {quality["complexity_score"]}/10

Overall Status: {"NEEDS ATTENTION" if high_severity > 0 else "REVIEW RECOMMENDED"}

--------------------------------------------------------------------------------
                        SECURITY FINDINGS
--------------------------------------------------------------------------------
"""
    
    for vuln in security["vulnerabilities"]:
        report += f"""
[{vuln["severity"]}] {vuln["type"]} (Line {vuln["line"]})
  Issue: {vuln["description"]}
  Fix:   {vuln["recommendation"]}
"""
    
    if security["vulnerable_dependencies"]:
        report += f"""
Vulnerable Dependencies:
  - {", ".join(security["vulnerable_dependencies"])}
"""
    
    report += """
--------------------------------------------------------------------------------
                        QUALITY ISSUES
--------------------------------------------------------------------------------
"""
    
    for issue in quality["issues"]:
        report += f"""
[{issue["severity"]}] {issue["type"]} (Line {issue["line"]})
  {issue["description"]}
"""
    
    report += """
--------------------------------------------------------------------------------
                        RECOMMENDATIONS
--------------------------------------------------------------------------------
"""
    
    for i, suggestion in enumerate(quality["suggestions"], 1):
        report += f"  {i}. {suggestion}\n"
    
    report += """
================================================================================
                           END OF REPORT
================================================================================
"""
    
    return report


async def main():
    """Run the code review pipeline example."""
    print("=" * 70)
    print("AWF Code Review Pipeline Example")
    print("=" * 70)
    print()
    
    # Initialize
    print("1. Initializing AWF components...")
    registry = InMemoryRegistry()
    print("   - Registry: OK")
    print()
    
    # Create agents
    print("2. Creating and registering agents...")
    await create_agents(registry)
    print()
    
    # Sample code to review
    sample_code = '''
import requests

API_KEY = "sk-secret-key-12345"  # TODO: move to env

class DataProcessor:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def process_data(self, user_input):
        # Fetch data from external API
        response = requests.get(f"https://api.example.com/data?key={API_KEY}")
        data = response.json()
        
        # Process and store (vulnerable!)
        query = f"INSERT INTO data VALUES ('{user_input}')"
        self.db.execute(query)
        
        return data
    
    def validate_input(self, data):
        return len(data) > 0
    
    def save_to_db(self, data):
        self.db.insert(data)
'''
    
    print("3. Sample code to review:")
    print("-" * 70)
    for i, line in enumerate(sample_code.strip().split("\n"), 1):
        print(f"{i:3}: {line}")
    print("-" * 70)
    print()
    
    # Run review
    print("4. Running code review pipeline...")
    print("   - Parsing code structure...")
    print("   - Scanning for security issues...")
    print("   - Reviewing code quality...")
    
    results = await simulate_code_review(sample_code)
    print("   - Generating report...")
    print()
    
    # Generate and display report
    report = generate_report(results)
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
