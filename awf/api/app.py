"""
AI Workflow Fabric - FastAPI Application

This module provides the main FastAPI application for the AWF REST API.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

try:
    from fastapi import FastAPI, HTTPException, Query, Depends, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError(
        "FastAPI is required for the AWF API. "
        "Install with: pip install awf[api]"
    )

from awf.api.models import (
    AgentCreate,
    AgentListResponse,
    AgentResponse,
    AgentSearchQuery,
    AgentUpdate,
    CapabilityResponse,
    ErrorResponse,
    HealthResponse,
    PolicyCreate,
    PolicyResponse,
    TaskCreate,
    TaskResponse,
    TaskSubmitResponse,
    TrustScoreResponse,
    WorkflowCreate,
    WorkflowExecutionResponse,
    WorkflowResponse,
)
from awf.core.types import (
    AgentManifest,
    AgentStatus,
    Capability,
    CapabilityType,
    Task,
    TaskResult,
)
from awf.registry.memory import InMemoryRegistry
from awf.security.trust import TrustScoringEngine
from awf.security.policy import PolicyEngine


# =============================================================================
# Application State
# =============================================================================


class AppState:
    """Application state container."""
    
    def __init__(self):
        self.registry = InMemoryRegistry()
        self.trust_engine = TrustScoringEngine()
        self.policy_engine = PolicyEngine()
        self.start_time = time.time()
        self.version = "1.0.0"


_app_state: Optional[AppState] = None


def get_app_state() -> AppState:
    """Get the application state."""
    if _app_state is None:
        raise RuntimeError("Application not initialized")
    return _app_state


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    global _app_state
    
    # Startup
    _app_state = AppState()
    
    yield
    
    # Shutdown
    _app_state = None


# =============================================================================
# Application Factory
# =============================================================================


def create_app(
    *,
    title: str = "AI Workflow Fabric",
    version: str = "1.0.0",
    debug: bool = False,
    cors_origins: Optional[list[str]] = None,
) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        title: API title
        version: API version
        debug: Enable debug mode
        cors_origins: Allowed CORS origins (None = allow all in debug mode)
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title=title,
        description="Kubernetes for AI Agents - Unified agent orchestration and federation",
        version=version,
        debug=debug,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # CORS middleware
    origins = cors_origins or (["*"] if debug else [])
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Register routes
    _register_routes(app)
    
    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "message": exc.detail,
                "request_id": request.headers.get("X-Request-ID"),
            },
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": str(exc) if debug else "An internal error occurred",
                "request_id": request.headers.get("X-Request-ID"),
            },
        )
    
    return app


# =============================================================================
# Route Registration
# =============================================================================


def _register_routes(app: FastAPI) -> None:
    """Register all API routes."""
    
    # -------------------------------------------------------------------------
    # Health & Status
    # -------------------------------------------------------------------------
    
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Health check",
    )
    async def health_check() -> HealthResponse:
        """Check the health of the AWF service."""
        state = get_app_state()
        count = await state.registry.count()
        
        return HealthResponse(
            status="healthy",
            version=state.version,
            uptime_seconds=time.time() - state.start_time,
            registry_count=count,
            components={
                "registry": "healthy",
                "trust_engine": "healthy",
                "policy_engine": "healthy",
            },
        )
    
    @app.get(
        "/",
        tags=["Health"],
        summary="Root endpoint",
    )
    async def root() -> Dict[str, Any]:
        """Root endpoint with API information."""
        return {
            "name": "AI Workflow Fabric",
            "version": get_app_state().version,
            "docs": "/docs",
            "openapi": "/openapi.json",
        }
    
    # -------------------------------------------------------------------------
    # Agent Registry
    # -------------------------------------------------------------------------
    
    @app.post(
        "/agents",
        response_model=AgentResponse,
        status_code=201,
        tags=["Agents"],
        summary="Register an agent",
    )
    async def register_agent(agent: AgentCreate) -> AgentResponse:
        """Register a new agent in the registry."""
        state = get_app_state()
        
        # Check if agent already exists
        existing = await state.registry.get(agent.id)
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Agent with ID '{agent.id}' already exists"
            )
        
        # Convert to manifest
        capabilities = [
            Capability(
                name=cap.name,
                type=CapabilityType(cap.type.value if hasattr(cap.type, 'value') else cap.type),
                description=cap.description,
                permissions=cap.permissions,
                metadata=cap.metadata,
            )
            for cap in agent.capabilities
        ]
        
        manifest = AgentManifest(
            id=agent.id,
            name=agent.name,
            version=agent.version,
            framework=agent.framework,
            framework_version=agent.framework_version,
            description=agent.description,
            capabilities=capabilities,
            tags=agent.tags,
            publisher=agent.publisher,
            documentation_url=agent.documentation_url,
            source_url=agent.source_url,
            metadata=agent.metadata,
            status=AgentStatus.ACTIVE,
        )
        
        # Compute trust score
        trust_score = await state.trust_engine.compute_score(manifest)
        manifest.trust_score = trust_score.score
        
        # Register
        await state.registry.register(manifest)
        
        return _manifest_to_response(manifest)
    
    @app.get(
        "/agents/{agent_id}",
        response_model=AgentResponse,
        tags=["Agents"],
        summary="Get an agent",
    )
    async def get_agent(agent_id: str) -> AgentResponse:
        """Retrieve an agent by ID."""
        state = get_app_state()
        manifest = await state.registry.get(agent_id)
        
        if manifest is None:
            raise HTTPException(
                status_code=404,
                detail=f"Agent with ID '{agent_id}' not found"
            )
        
        return _manifest_to_response(manifest)
    
    @app.get(
        "/agents",
        response_model=AgentListResponse,
        tags=["Agents"],
        summary="List agents",
    )
    async def list_agents(
        capabilities: Optional[str] = Query(None, description="Comma-separated capabilities"),
        framework: Optional[str] = Query(None),
        tags: Optional[str] = Query(None, description="Comma-separated tags"),
        min_trust_score: Optional[float] = Query(None, ge=0.0, le=1.0),
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=100),
    ) -> AgentListResponse:
        """List and search for agents."""
        state = get_app_state()
        
        # Parse comma-separated values
        cap_list = capabilities.split(",") if capabilities else None
        tag_list = tags.split(",") if tags else None
        
        # Search
        results = await state.registry.search(
            capabilities=cap_list,
            framework=framework,
            tags=tag_list,
            min_trust_score=min_trust_score,
        )
        
        # Paginate
        total = len(results)
        start = (page - 1) * page_size
        end = start + page_size
        page_results = results[start:end]
        
        return AgentListResponse(
            agents=[_manifest_to_response(m) for m in page_results],
            total=total,
            page=page,
            page_size=page_size,
        )
    
    @app.patch(
        "/agents/{agent_id}",
        response_model=AgentResponse,
        tags=["Agents"],
        summary="Update an agent",
    )
    async def update_agent(agent_id: str, update: AgentUpdate) -> AgentResponse:
        """Update an existing agent."""
        state = get_app_state()
        manifest = await state.registry.get(agent_id)
        
        if manifest is None:
            raise HTTPException(
                status_code=404,
                detail=f"Agent with ID '{agent_id}' not found"
            )
        
        # Apply updates
        if update.name is not None:
            manifest.name = update.name
        if update.version is not None:
            manifest.version = update.version
        if update.description is not None:
            manifest.description = update.description
        if update.tags is not None:
            manifest.tags = update.tags
        if update.status is not None:
            manifest.status = AgentStatus(update.status)
        if update.metadata is not None:
            manifest.metadata = update.metadata
        if update.capabilities is not None:
            manifest.capabilities = [
                Capability(
                    name=cap.name,
                    type=CapabilityType(cap.type.value if hasattr(cap.type, 'value') else cap.type),
                    description=cap.description,
                    permissions=cap.permissions,
                    metadata=cap.metadata,
                )
                for cap in update.capabilities
            ]
        
        # Re-compute trust score
        trust_score = await state.trust_engine.compute_score(manifest)
        manifest.trust_score = trust_score.score
        
        # Save
        await state.registry.update(manifest)
        
        return _manifest_to_response(manifest)
    
    @app.delete(
        "/agents/{agent_id}",
        status_code=204,
        tags=["Agents"],
        summary="Delete an agent",
    )
    async def delete_agent(agent_id: str) -> None:
        """Remove an agent from the registry."""
        state = get_app_state()
        deleted = await state.registry.delete(agent_id)
        
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Agent with ID '{agent_id}' not found"
            )
    
    @app.post(
        "/agents/{agent_id}/activate",
        response_model=AgentResponse,
        tags=["Agents"],
        summary="Activate an agent",
    )
    async def activate_agent(agent_id: str) -> AgentResponse:
        """Set an agent's status to active."""
        state = get_app_state()
        updated = await state.registry.set_status(agent_id, AgentStatus.ACTIVE)
        
        if not updated:
            raise HTTPException(
                status_code=404,
                detail=f"Agent with ID '{agent_id}' not found"
            )
        
        manifest = await state.registry.get(agent_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        return _manifest_to_response(manifest)
    
    @app.post(
        "/agents/{agent_id}/suspend",
        response_model=AgentResponse,
        tags=["Agents"],
        summary="Suspend an agent",
    )
    async def suspend_agent(agent_id: str) -> AgentResponse:
        """Suspend an agent."""
        state = get_app_state()
        updated = await state.registry.set_status(agent_id, AgentStatus.SUSPENDED)
        
        if not updated:
            raise HTTPException(
                status_code=404,
                detail=f"Agent with ID '{agent_id}' not found"
            )
        
        manifest = await state.registry.get(agent_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        return _manifest_to_response(manifest)
    
    # -------------------------------------------------------------------------
    # Trust
    # -------------------------------------------------------------------------
    
    @app.get(
        "/agents/{agent_id}/trust",
        response_model=TrustScoreResponse,
        tags=["Trust"],
        summary="Get trust score",
    )
    async def get_trust_score(agent_id: str) -> TrustScoreResponse:
        """Get the trust score for an agent."""
        state = get_app_state()
        manifest = await state.registry.get(agent_id)
        
        if manifest is None:
            raise HTTPException(
                status_code=404,
                detail=f"Agent with ID '{agent_id}' not found"
            )
        
        trust = await state.trust_engine.compute_score(manifest)
        
        return TrustScoreResponse(
            agent_id=agent_id,
            score=trust.score,
            sandbox_tier=trust.sandbox_tier.value,
            factors={
                "publisher_trust": trust.factors.publisher_trust,
                "audit_status": trust.factors.audit_status,
                "community_trust": trust.factors.community_trust,
                "permission_analysis": trust.factors.permission_analysis,
                "historical_behavior": trust.factors.historical_behavior,
                "computed_score": trust.factors.compute_score(),
            },
            computed_at=trust.computed_at,
            expires_at=trust.expires_at,
        )
    
    # -------------------------------------------------------------------------
    # Tasks
    # -------------------------------------------------------------------------
    
    @app.post(
        "/tasks",
        response_model=TaskSubmitResponse,
        status_code=202,
        tags=["Tasks"],
        summary="Submit a task",
    )
    async def submit_task(task: TaskCreate) -> TaskSubmitResponse:
        """Submit a task for execution."""
        state = get_app_state()
        
        # Verify agent exists
        agent = await state.registry.get(task.agent_id)
        if agent is None:
            raise HTTPException(
                status_code=404,
                detail=f"Agent with ID '{task.agent_id}' not found"
            )
        
        # Check agent status
        if agent.status != AgentStatus.ACTIVE:
            raise HTTPException(
                status_code=400,
                detail=f"Agent '{task.agent_id}' is not active (status: {agent.status.value})"
            )
        
        # Check policy
        # Create a dummy task for policy evaluation
        dummy_task = Task(
            agent_id=task.agent_id,
            input=task.input,
            timeout_ms=task.timeout_ms,
        )
        policy_result = state.policy_engine.evaluate(
            manifest=agent,
            task=dummy_task,
            environment="default",
        )
        
        if not policy_result.allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Policy violation: {policy_result.violations[0].violation_type}" if policy_result.violations else "Policy check failed",
            )
        
        # Create task
        asp_task = Task(
            agent_id=task.agent_id,
            input=task.input,
            correlation_id=task.correlation_id,
            timeout_ms=task.timeout_ms,
            priority=task.priority,
            context=task.context,
            metadata=task.metadata,
        )
        
        # TODO: Submit to execution engine
        # For now, just return the task ID
        
        return TaskSubmitResponse(
            task_id=asp_task.id,
            status="pending",
            message="Task submitted successfully",
        )
    
    @app.get(
        "/tasks/{task_id}",
        response_model=TaskResponse,
        tags=["Tasks"],
        summary="Get task status",
    )
    async def get_task(task_id: str) -> TaskResponse:
        """Get the status of a task."""
        # TODO: Implement task storage and retrieval
        raise HTTPException(
            status_code=501,
            detail="Task retrieval not yet implemented"
        )


# =============================================================================
# Helpers
# =============================================================================


def _manifest_to_response(manifest: AgentManifest) -> AgentResponse:
    """Convert an AgentManifest to an AgentResponse."""
    return AgentResponse(
        id=manifest.id,
        name=manifest.name,
        version=manifest.version,
        framework=manifest.framework,
        framework_version=manifest.framework_version,
        status=manifest.status.value,
        trust_score=manifest.trust_score,
        description=manifest.description,
        capabilities=[
            CapabilityResponse(
                name=cap.name,
                type=cap.type.value,
                description=cap.description,
                permissions=cap.permissions,
                metadata=cap.metadata,
            )
            for cap in manifest.capabilities
        ],
        tags=manifest.tags,
        publisher=manifest.publisher,
        documentation_url=manifest.documentation_url,
        source_url=manifest.source_url,
        registered_at=manifest.registered_at,
        updated_at=manifest.updated_at,
        metadata=manifest.metadata,
    )


# =============================================================================
# Default App Instance
# =============================================================================


# Create default app for `uvicorn awf.api.app:app`
app = create_app(debug=True)
