#!/usr/bin/env python3
"""
AWF Customer Support Pipeline Example

This example demonstrates a multi-agent customer support workflow that:
1. Classifies incoming support tickets
2. Routes to appropriate specialist agents
3. Generates responses or escalates to humans

Run with:
    python examples/customer_support/main.py
"""

import asyncio
import random
from datetime import datetime
from enum import Enum

from awf.core.types import (
    AgentManifest,
    AgentStatus,
    Capability,
    CapabilityType,
)
from awf.registry.memory import InMemoryRegistry
from awf.security.trust import TrustScoringEngine


class TicketCategory(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    ESCALATION = "escalation"


class TicketPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


async def create_agents(registry: InMemoryRegistry) -> None:
    """Create and register the support agents."""
    trust_engine = TrustScoringEngine()
    
    # Classifier Agent
    classifier_agent = AgentManifest(
        id="ticket-classifier-agent",
        name="Ticket Classifier Agent",
        version="1.0.0",
        framework="langgraph",
        description="Classifies support tickets by category and priority",
        capabilities=[
            Capability(
                name="classify_ticket",
                type=CapabilityType.REASONING,
                description="Classify ticket category and priority",
            ),
            Capability(
                name="extract_entities",
                type=CapabilityType.REASONING,
                description="Extract key entities from ticket",
            ),
        ],
        tags=["classification", "nlp", "support"],
        publisher="awf-examples",
        status=AgentStatus.ACTIVE,
    )
    
    # Billing Support Agent
    billing_agent = AgentManifest(
        id="billing-support-agent",
        name="Billing Support Agent",
        version="1.0.0",
        framework="crewai",
        description="Handles billing-related support queries",
        capabilities=[
            Capability(
                name="billing_inquiry",
                type=CapabilityType.REASONING,
                description="Answer billing questions",
            ),
            Capability(
                name="refund_process",
                type=CapabilityType.TOOL,
                description="Process refund requests",
                permissions=["billing:read", "billing:write"],
            ),
        ],
        tags=["billing", "support", "payments"],
        publisher="awf-examples",
        status=AgentStatus.ACTIVE,
    )
    
    # Technical Support Agent
    technical_agent = AgentManifest(
        id="technical-support-agent",
        name="Technical Support Agent",
        version="2.0.0",
        framework="langgraph",
        description="Handles technical support and troubleshooting",
        capabilities=[
            Capability(
                name="troubleshoot",
                type=CapabilityType.REASONING,
                description="Troubleshoot technical issues",
            ),
            Capability(
                name="knowledge_search",
                type=CapabilityType.TOOL,
                description="Search knowledge base",
            ),
        ],
        tags=["technical", "troubleshooting", "support"],
        publisher="awf-examples",
        status=AgentStatus.ACTIVE,
    )
    
    # Response Generator Agent
    response_agent = AgentManifest(
        id="response-generator-agent",
        name="Response Generator Agent",
        version="1.0.0",
        framework="crewai",
        description="Generates customer-friendly responses",
        capabilities=[
            Capability(
                name="generate_response",
                type=CapabilityType.REASONING,
                description="Generate helpful responses",
            ),
            Capability(
                name="personalize",
                type=CapabilityType.REASONING,
                description="Personalize responses to customer",
            ),
        ],
        tags=["response", "communication", "support"],
        publisher="awf-examples",
        status=AgentStatus.ACTIVE,
    )
    
    # Register all agents
    for agent in [classifier_agent, billing_agent, technical_agent, response_agent]:
        trust = await trust_engine.compute_score(agent)
        agent.trust_score = trust.score
        await registry.register(agent)
        print(f"  Registered: {agent.name} (trust: {trust.score:.2f})")


# Sample tickets for demonstration
SAMPLE_TICKETS = [
    {
        "id": "TKT-001",
        "subject": "Can't log into my account",
        "body": "I've been trying to log in for the past hour but keep getting an error message. I've reset my password twice already. This is urgent as I need to access my reports.",
        "customer": "john.doe@example.com",
    },
    {
        "id": "TKT-002",
        "subject": "Refund request for double charge",
        "body": "I was charged twice for my subscription last month. Order #12345. Please refund the duplicate charge of $29.99.",
        "customer": "jane.smith@example.com",
    },
    {
        "id": "TKT-003",
        "subject": "How do I export my data?",
        "body": "I'd like to export all my project data to CSV format. Where can I find this option?",
        "customer": "bob.wilson@example.com",
    },
]


async def classify_ticket(ticket: dict) -> dict:
    """Simulate ticket classification."""
    await asyncio.sleep(0.2)
    
    body_lower = ticket["body"].lower()
    subject_lower = ticket["subject"].lower()
    
    # Simple keyword-based classification
    if any(word in body_lower for word in ["charge", "refund", "payment", "billing", "invoice"]):
        category = TicketCategory.BILLING
    elif any(word in body_lower for word in ["error", "bug", "crash", "login", "password", "not working"]):
        category = TicketCategory.TECHNICAL
    else:
        category = TicketCategory.GENERAL
    
    # Priority based on urgency keywords
    if any(word in body_lower for word in ["urgent", "asap", "immediately", "critical"]):
        priority = TicketPriority.URGENT
    elif any(word in body_lower for word in ["important", "soon", "quickly"]):
        priority = TicketPriority.HIGH
    else:
        priority = TicketPriority.MEDIUM
    
    return {
        "ticket_id": ticket["id"],
        "category": category.value,
        "priority": priority.value,
        "entities": {
            "order_id": "12345" if "12345" in ticket["body"] else None,
            "amount": "$29.99" if "$29.99" in ticket["body"] else None,
        },
        "confidence": random.uniform(0.85, 0.98),
    }


async def route_and_process(classification: dict, ticket: dict) -> dict:
    """Route ticket to appropriate agent and process."""
    await asyncio.sleep(0.3)
    
    category = classification["category"]
    
    if category == "billing":
        agent_used = "billing-support-agent"
        resolution = {
            "action": "refund_initiated",
            "details": f"Refund of {classification['entities'].get('amount', 'N/A')} initiated for order {classification['entities'].get('order_id', 'N/A')}",
            "requires_approval": True,
        }
    elif category == "technical":
        agent_used = "technical-support-agent"
        resolution = {
            "action": "troubleshooting_steps",
            "details": "Password reset confirmed. Account unlocked. Sent new login instructions.",
            "kb_articles": ["KB-001: Account Recovery", "KB-015: Login Troubleshooting"],
        }
    else:
        agent_used = "technical-support-agent"
        resolution = {
            "action": "information_provided",
            "details": "Export feature is available under Settings > Data > Export. CSV format is supported.",
            "kb_articles": ["KB-042: Data Export Guide"],
        }
    
    return {
        "agent_used": agent_used,
        "resolution": resolution,
        "processing_time_ms": random.randint(500, 2000),
    }


async def generate_response(ticket: dict, classification: dict, processing: dict) -> str:
    """Generate a customer-friendly response."""
    await asyncio.sleep(0.2)
    
    customer_name = ticket["customer"].split("@")[0].replace(".", " ").title()
    
    responses = {
        "billing": f"""Dear {customer_name},

Thank you for contacting our support team regarding your billing concern.

We have reviewed your account and confirmed the duplicate charge. A refund of {classification['entities'].get('amount', 'the duplicate amount')} has been initiated for order #{classification['entities'].get('order_id', 'your order')}.

Please allow 5-7 business days for the refund to appear in your account.

If you have any further questions, please don't hesitate to reach out.

Best regards,
Customer Support Team
Ticket #{ticket['id']}""",
        
        "technical": f"""Dear {customer_name},

Thank you for reaching out about your login issue.

We understand how frustrating this can be. We've taken the following steps to resolve your issue:

1. Your password has been successfully reset
2. Your account has been unlocked
3. A new login link has been sent to your email

Please try logging in again using the link in your email. If you continue to experience issues, please reply to this ticket.

Related Help Articles:
- Account Recovery Guide
- Login Troubleshooting Steps

Best regards,
Technical Support Team
Ticket #{ticket['id']}""",
        
        "general": f"""Dear {customer_name},

Thank you for your question about data export.

You can export your data by following these steps:
1. Navigate to Settings in the top-right menu
2. Click on "Data" in the sidebar
3. Select "Export" 
4. Choose CSV as your format
5. Click "Export All Data"

Your download will begin automatically.

For more detailed instructions, please see our Data Export Guide in the Help Center.

Best regards,
Customer Support Team
Ticket #{ticket['id']}"""
    }
    
    return responses.get(classification["category"], responses["general"])


async def process_ticket(ticket: dict) -> dict:
    """Process a single support ticket through the pipeline."""
    # Step 1: Classify
    classification = await classify_ticket(ticket)
    
    # Step 2: Route and process
    processing = await route_and_process(classification, ticket)
    
    # Step 3: Generate response
    response = await generate_response(ticket, classification, processing)
    
    return {
        "ticket": ticket,
        "classification": classification,
        "processing": processing,
        "response": response,
        "status": "resolved",
        "timestamp": datetime.now().isoformat(),
    }


async def main():
    """Run the customer support pipeline example."""
    print("=" * 70)
    print("AWF Customer Support Pipeline Example")
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
    
    # Process tickets
    print("3. Processing support tickets...")
    print()
    
    for ticket in SAMPLE_TICKETS:
        print(f"   Processing: {ticket['id']} - {ticket['subject']}")
        result = await process_ticket(ticket)
        
        print(f"   - Category: {result['classification']['category'].upper()}")
        print(f"   - Priority: {result['classification']['priority'].upper()}")
        print(f"   - Agent: {result['processing']['agent_used']}")
        print(f"   - Status: {result['status'].upper()}")
        print()
        print("-" * 70)
        print("CUSTOMER RESPONSE:")
        print("-" * 70)
        print(result["response"])
        print("-" * 70)
        print()
    
    # Summary
    print("=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total tickets processed: {len(SAMPLE_TICKETS)}")
    print(f"Agents available: 4")
    print(f"  - Ticket Classifier Agent")
    print(f"  - Billing Support Agent")
    print(f"  - Technical Support Agent")
    print(f"  - Response Generator Agent")
    print()
    print("Pipeline steps:")
    print("  1. Ticket Classification -> Categorize and prioritize")
    print("  2. Intelligent Routing -> Route to specialist agent")
    print("  3. Response Generation -> Create personalized response")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
