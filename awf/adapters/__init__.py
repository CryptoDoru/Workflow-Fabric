"""
AI Workflow Fabric - Adapters Package

This package contains framework-specific adapters that translate between
native agent frameworks and the Agent State Protocol (ASP).

Supported Frameworks:
    - LangGraph: Graph-based agent orchestration
    - CrewAI: Role-based agent teams
    - AutoGen: Conversation-based multi-agent systems
"""

from awf.adapters.base import (
    BaseAdapter,
    AgentRegistry,
    TrustScorer,
    AdapterError,
    AgentNotFoundError,
    RegistrationError,
    ExecutionError,
    ValidationError,
)

# Lazy imports to avoid requiring all frameworks to be installed
__all__ = [
    # Base classes and interfaces
    "BaseAdapter",
    "AgentRegistry",
    "TrustScorer",
    # Exceptions
    "AdapterError",
    "AgentNotFoundError",
    "RegistrationError",
    "ExecutionError",
    "ValidationError",
    # Framework adapters (lazy loaded)
    "LangGraphAdapter",
    "CrewAIAdapter",
    "AutoGenAdapter",
]


def __getattr__(name: str):
    """Lazy load framework adapters to avoid import errors when frameworks aren't installed."""
    if name == "LangGraphAdapter":
        from awf.adapters.langgraph import LangGraphAdapter
        return LangGraphAdapter
    elif name == "CrewAIAdapter":
        from awf.adapters.crewai import CrewAIAdapter
        return CrewAIAdapter
    elif name == "AutoGenAdapter":
        from awf.adapters.autogen import AutoGenAdapter
        return AutoGenAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
