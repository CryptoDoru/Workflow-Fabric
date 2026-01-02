"""
AI Workflow Fabric - REST API Package

This package provides a FastAPI-based REST API for the AWF registry
and orchestration services.

Usage:
    from awf.api import create_app
    
    app = create_app()
    # Run with: uvicorn awf.api:app --reload
"""

from awf.api.app import create_app
from awf.api.models import (
    AgentCreate,
    AgentResponse,
    AgentSearchQuery,
    TaskCreate,
    TaskResponse,
    HealthResponse,
)

__all__ = [
    "create_app",
    "AgentCreate",
    "AgentResponse", 
    "AgentSearchQuery",
    "TaskCreate",
    "TaskResponse",
    "HealthResponse",
]
