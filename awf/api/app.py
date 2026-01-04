"""
AI Workflow Fabric - FastAPI Application

This module provides the main FastAPI application for the AWF REST API.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Set

try:
    from fastapi import FastAPI, HTTPException, Query, Depends, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
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
    ApprovalActionRequest,
    ApprovalActionResponse,
    ApprovalRequestResponse,
    CapabilityResponse,
    ErrorResponse,
    ExecutionStatusEnum,
    GrafanaAlertResponse,
    GrafanaAlertWebhook,
    HealthResponse,
    PolicyCreate,
    PolicyResponse,
    StepResultResponse,
    StepStatusEnum,
    TaskCreate,
    TaskResponse,
    TaskSubmitResponse,
    TrustScoreResponse,
    WorkflowCreate,
    WorkflowEventResponse,
    WorkflowEventTypeEnum,
    WorkflowExecuteRequest,
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
from awf.orchestration.types import (
    WorkflowDefinition,
    StepDefinition,
    RetryPolicy,
    FallbackPolicy,
    WorkflowResult,
    StepResult,
    StepStatus,
    ExecutionStatus,
    WorkflowEvent,
)
from awf.orchestration.orchestrator import Orchestrator, OrchestratorConfig
from awf.orchestration.registry import OrchestrationAdapterRegistry
from awf.agents.watcher import WatcherAgent, WatcherConfig, GrafanaAlert


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
        
        # Workflow storage (in-memory for now)
        self.workflows: Dict[str, WorkflowDefinition] = {}
        
        # Execution storage (in-memory for now)
        self.executions: Dict[str, WorkflowResult] = {}
        
        # Event storage for SSE (per execution)
        self.execution_events: Dict[str, list] = {}
        
        # WebSocket connections for real-time updates
        self.websocket_connections: Set[WebSocket] = set()
        self.execution_subscribers: Dict[str, Set[WebSocket]] = {}
        
        # Orchestrator setup
        self.adapter_registry = OrchestrationAdapterRegistry()
        self.orchestrator = Orchestrator(
            adapter_registry=self.adapter_registry,
            agent_registry=self.registry,
            config=OrchestratorConfig(
                default_timeout_ms=300000,  # 5 minutes
                max_parallel_steps=10,
                emit_events=True,
            ),
            event_callback=self._on_workflow_event,
        )
        
        # Watcher Agent for autonomous observability
        self.watcher = WatcherAgent(WatcherConfig())
    
    async def _on_workflow_event(self, event: WorkflowEvent) -> None:
        """Handle workflow events for SSE streaming and WebSocket broadcast."""
        execution_id = event.execution_id
        if execution_id not in self.execution_events:
            self.execution_events[execution_id] = []
        self.execution_events[execution_id].append(event)
        
        # Broadcast to WebSocket subscribers
        await self._broadcast_event(execution_id, event)
    
    async def _broadcast_event(self, execution_id: str, event: WorkflowEvent) -> None:
        """Broadcast event to WebSocket subscribers."""
        subscribers = self.execution_subscribers.get(execution_id, set()).copy()
        # Also notify global subscribers
        all_subscribers = self.execution_subscribers.get("__all__", set())
        subscribers.update(all_subscribers)
        
        event_data = {
            "type": event.type.value,
            "execution_id": event.execution_id,
            "workflow_id": event.workflow_id,
            "step_id": event.step_id,
            "timestamp": event.timestamp.isoformat() if event.timestamp else None,
            "data": event.data,
        }
        message = json.dumps(event_data)
        
        disconnected = set()
        for ws in subscribers:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.add(ws)
        
        # Clean up disconnected clients
        for ws in disconnected:
            subscribers.discard(ws)
    
    def subscribe_to_execution(self, execution_id: str, ws: WebSocket) -> None:
        """Subscribe a WebSocket to execution events."""
        if execution_id not in self.execution_subscribers:
            self.execution_subscribers[execution_id] = set()
        self.execution_subscribers[execution_id].add(ws)
    
    def unsubscribe_from_execution(self, execution_id: str, ws: WebSocket) -> None:
        """Unsubscribe a WebSocket from execution events."""
        if execution_id in self.execution_subscribers:
            self.execution_subscribers[execution_id].discard(ws)


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
    
    # -------------------------------------------------------------------------
    # Workflows
    # -------------------------------------------------------------------------
    
    @app.post(
        "/workflows",
        response_model=WorkflowResponse,
        status_code=201,
        tags=["Workflows"],
        summary="Create a workflow",
    )
    async def create_workflow(workflow: WorkflowCreate) -> WorkflowResponse:
        """Create a new workflow definition."""
        state = get_app_state()
        
        # Check if workflow already exists
        if workflow.id in state.workflows:
            raise HTTPException(
                status_code=409,
                detail=f"Workflow with ID '{workflow.id}' already exists"
            )
        
        # Convert Pydantic model to WorkflowDefinition
        steps = []
        for step in workflow.steps:
            retry = None
            if step.retry:
                retry = RetryPolicy(
                    max_attempts=step.retry.max_attempts,
                    backoff_ms=step.retry.backoff_ms,
                    backoff_multiplier=step.retry.backoff_multiplier,
                    max_backoff_ms=step.retry.max_backoff_ms,
                )
            
            fallback = None
            if step.fallback:
                fallback = FallbackPolicy(
                    skip=step.fallback.skip,
                    static_value=step.fallback.static_value,
                    agent_id=step.fallback.agent_id,
                )
            
            steps.append(StepDefinition(
                id=step.id,
                agent_id=step.agent_id,
                input_map=step.input_map,
                depends_on=step.depends_on,
                condition=step.condition,
                timeout_ms=step.timeout_ms,
                retry=retry,
                fallback=fallback,
                metadata=step.metadata,
            ))
        
        workflow_def = WorkflowDefinition(
            id=workflow.id,
            name=workflow.name,
            description=workflow.description,
            steps=steps,
            input_schema=workflow.input_schema,
            timeout_ms=workflow.timeout_ms,
        )
        
        # Store workflow
        state.workflows[workflow.id] = workflow_def
        
        return _workflow_to_response(workflow_def)
    
    @app.get(
        "/workflows/{workflow_id}",
        response_model=WorkflowResponse,
        tags=["Workflows"],
        summary="Get a workflow",
    )
    async def get_workflow(workflow_id: str) -> WorkflowResponse:
        """Retrieve a workflow definition by ID."""
        state = get_app_state()
        
        if workflow_id not in state.workflows:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow with ID '{workflow_id}' not found"
            )
        
        return _workflow_to_response(state.workflows[workflow_id])
    
    @app.get(
        "/workflows",
        response_model=list[WorkflowResponse],
        tags=["Workflows"],
        summary="List workflows",
    )
    async def list_workflows() -> list[WorkflowResponse]:
        """List all workflow definitions."""
        state = get_app_state()
        return [_workflow_to_response(w) for w in state.workflows.values()]
    
    @app.delete(
        "/workflows/{workflow_id}",
        status_code=204,
        tags=["Workflows"],
        summary="Delete a workflow",
    )
    async def delete_workflow(workflow_id: str) -> None:
        """Delete a workflow definition."""
        state = get_app_state()
        
        if workflow_id not in state.workflows:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow with ID '{workflow_id}' not found"
            )
        
        del state.workflows[workflow_id]
    
    # -------------------------------------------------------------------------
    # Workflow Execution
    # -------------------------------------------------------------------------
    
    @app.post(
        "/workflows/{workflow_id}/execute",
        response_model=WorkflowExecutionResponse,
        status_code=202,
        tags=["Executions"],
        summary="Execute a workflow",
    )
    async def execute_workflow(
        workflow_id: str,
        request: WorkflowExecuteRequest,
    ) -> WorkflowExecutionResponse:
        """Start executing a workflow."""
        state = get_app_state()
        
        # Get workflow
        if workflow_id not in state.workflows:
            raise HTTPException(
                status_code=404,
                detail=f"Workflow with ID '{workflow_id}' not found"
            )
        
        workflow = state.workflows[workflow_id]
        
        # Execute workflow
        try:
            result = await state.orchestrator.execute(
                workflow=workflow,
                input_data=request.input,
                context=request.context,
                timeout_ms=request.timeout_ms,
                trace_id=request.trace_id,
                correlation_id=request.correlation_id,
            )
            
            # Store result
            state.executions[result.execution_id] = result
            
            return _result_to_response(result)
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Workflow execution failed: {str(e)}"
            )
    
    @app.get(
        "/executions/{execution_id}",
        response_model=WorkflowExecutionResponse,
        tags=["Executions"],
        summary="Get execution status",
    )
    async def get_execution(execution_id: str) -> WorkflowExecutionResponse:
        """Get the status of a workflow execution."""
        state = get_app_state()
        
        if execution_id not in state.executions:
            raise HTTPException(
                status_code=404,
                detail=f"Execution with ID '{execution_id}' not found"
            )
        
        return _result_to_response(state.executions[execution_id])
    
    @app.post(
        "/executions/{execution_id}/cancel",
        response_model=WorkflowExecutionResponse,
        tags=["Executions"],
        summary="Cancel an execution",
    )
    async def cancel_execution(execution_id: str) -> WorkflowExecutionResponse:
        """Cancel a running workflow execution."""
        state = get_app_state()
        
        # Try to cancel
        cancelled = await state.orchestrator.cancel(execution_id)
        
        if not cancelled:
            # Check if execution exists but is already completed
            if execution_id in state.executions:
                result = state.executions[execution_id]
                if result.status in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED):
                    return _result_to_response(result)
            
            raise HTTPException(
                status_code=404,
                detail=f"Execution with ID '{execution_id}' not found or already completed"
            )
        
        # Wait a moment for cancellation to propagate
        import asyncio
        await asyncio.sleep(0.1)
        
        if execution_id in state.executions:
            return _result_to_response(state.executions[execution_id])
        
        # Return minimal response if not yet stored
        return WorkflowExecutionResponse(
            execution_id=execution_id,
            workflow_id="unknown",
            status=ExecutionStatusEnum.CANCELLED,
            input={},
        )
    
    @app.get(
        "/executions",
        response_model=list[WorkflowExecutionResponse],
        tags=["Executions"],
        summary="List executions",
    )
    async def list_executions(
        workflow_id: Optional[str] = Query(None, description="Filter by workflow ID"),
        status: Optional[str] = Query(None, description="Filter by status"),
    ) -> list[WorkflowExecutionResponse]:
        """List workflow executions with optional filtering."""
        state = get_app_state()
        
        results = []
        for result in state.executions.values():
            if workflow_id and result.workflow_id != workflow_id:
                continue
            if status and result.status.value != status:
                continue
            results.append(_result_to_response(result))
        
        return results
    
    @app.get(
        "/executions/{execution_id}/events",
        tags=["Executions"],
        summary="Stream execution events (SSE)",
    )
    async def stream_execution_events(execution_id: str) -> StreamingResponse:
        """
        Stream workflow execution events using Server-Sent Events (SSE).
        
        Events are formatted as:
        ```
        event: workflow.started
        data: {"executionId": "...", "workflowId": "...", ...}
        
        event: step.completed
        data: {"stepId": "...", ...}
        ```
        """
        import asyncio
        import json
        
        state = get_app_state()
        
        async def event_generator():
            """Generate SSE events."""
            # Track which events we've sent
            sent_count = 0
            max_wait_iterations = 600  # 60 seconds max wait
            iterations = 0
            
            while iterations < max_wait_iterations:
                # Get events for this execution
                events = state.execution_events.get(execution_id, [])
                
                # Send any new events
                while sent_count < len(events):
                    event = events[sent_count]
                    event_data = event.to_dict()
                    event_type = event.type.value
                    
                    yield f"event: {event_type}\n"
                    yield f"data: {json.dumps(event_data)}\n\n"
                    
                    sent_count += 1
                    
                    # Check if this is a terminal event
                    if event_type in (
                        "workflow.completed",
                        "workflow.failed",
                        "workflow.cancelled",
                    ):
                        return
                
                # Wait a bit before checking for more events
                await asyncio.sleep(0.1)
                iterations += 1
            
            # Timeout - send a timeout event
            yield f"event: timeout\n"
            yield f"data: {json.dumps({'message': 'Event stream timed out'})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    
    # -------------------------------------------------------------------------
    # WebSocket Real-Time Updates
    # -------------------------------------------------------------------------
    
    @app.websocket("/ws/executions/{execution_id}")
    async def websocket_execution_events(websocket: WebSocket, execution_id: str):
        """
        WebSocket endpoint for real-time execution event streaming.
        
        Connect to receive live updates for a specific workflow execution.
        Events are sent as JSON messages in the same format as the SSE endpoint.
        
        Example client connection:
        ```javascript
        const ws = new WebSocket('ws://localhost:8000/ws/executions/exec_123');
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('Event:', data.type, data);
        };
        ```
        """
        state = get_app_state()
        
        # Accept the connection
        await websocket.accept()
        
        # Track connection
        state.websocket_connections.add(websocket)
        state.subscribe_to_execution(execution_id, websocket)
        
        try:
            # Send any existing events for this execution
            existing_events = state.execution_events.get(execution_id, [])
            for event in existing_events:
                event_data = {
                    "type": event.type.value,
                    "execution_id": event.execution_id,
                    "workflow_id": event.workflow_id,
                    "step_id": event.step_id,
                    "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                    "data": event.data,
                }
                await websocket.send_json(event_data)
            
            # Keep connection alive and wait for disconnect
            while True:
                # Wait for any message (ping/pong or close)
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send ping to keep alive
                    await websocket.send_json({"type": "ping"})
                    
        except WebSocketDisconnect:
            pass
        finally:
            # Clean up
            state.websocket_connections.discard(websocket)
            state.unsubscribe_from_execution(execution_id, websocket)
    
    @app.websocket("/ws/events")
    async def websocket_all_events(websocket: WebSocket):
        """
        WebSocket endpoint for receiving ALL workflow events.
        
        Useful for dashboards that need to monitor all executions.
        """
        state = get_app_state()
        
        await websocket.accept()
        state.websocket_connections.add(websocket)
        
        # Subscribe to all executions
        all_subscription_id = "__all__"
        if all_subscription_id not in state.execution_subscribers:
            state.execution_subscribers[all_subscription_id] = set()
        state.execution_subscribers[all_subscription_id].add(websocket)
        
        try:
            while True:
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "ping"})
        except WebSocketDisconnect:
            pass
        finally:
            state.websocket_connections.discard(websocket)
            if all_subscription_id in state.execution_subscribers:
                state.execution_subscribers[all_subscription_id].discard(websocket)
    
    # -------------------------------------------------------------------------
    # Watcher Agent Webhooks
    # -------------------------------------------------------------------------
    
    @app.post(
        "/webhooks/grafana-alerts",
        response_model=GrafanaAlertResponse,
        tags=["Watcher"],
        summary="Receive Grafana alerts",
    )
    async def receive_grafana_alert(
        payload: GrafanaAlertWebhook,
    ) -> GrafanaAlertResponse:
        """
        Receive alerts from Grafana Alerting and route to the Watcher Agent.
        
        The Watcher Agent will:
        1. Investigate the alert by querying metrics, logs, and traces
        2. Determine the best remediation action
        3. Execute low-risk actions automatically
        4. Request human approval for high-risk actions
        
        Configure Grafana to send webhooks to this endpoint:
        ```
        POST http://<awf-host>:8000/webhooks/grafana-alerts
        ```
        """
        state = get_app_state()
        
        # Convert webhook payload to GrafanaAlert
        alert = GrafanaAlert.from_webhook(payload.model_dump())
        
        # Handle via Watcher Agent
        result = await state.watcher.handle_alert(alert)
        
        return GrafanaAlertResponse(
            status=result.get("status", "unknown"),
            alert_id=alert.id,
            reason=result.get("reason"),
            approval_id=result.get("approval_id"),
            action=result.get("action"),
            result=result.get("result"),
            investigation=result.get("investigation"),
        )
    
    @app.get(
        "/watcher/approvals",
        response_model=list[ApprovalRequestResponse],
        tags=["Watcher"],
        summary="List pending approvals",
    )
    async def list_pending_approvals() -> list[ApprovalRequestResponse]:
        """
        List all pending HITL (Human-in-the-Loop) approval requests.
        
        High-risk remediation actions require human approval before execution.
        Use this endpoint to view pending requests, then approve or reject them.
        """
        state = get_app_state()
        pending = state.watcher.get_pending_approvals()
        
        responses = []
        for req in pending:
            responses.append(ApprovalRequestResponse(
                id=req["id"],
                action=req["action"],
                investigation=req["investigation"],
                requested_at=req["requested_at"],
                expires_at=req["expires_at"],
                status=req["status"],
                approved_by=req.get("approved_by"),
                rejected_by=req.get("rejected_by"),
                rejection_reason=req.get("rejection_reason"),
            ))
        
        return responses
    
    @app.get(
        "/watcher/approvals/{approval_id}",
        response_model=ApprovalRequestResponse,
        tags=["Watcher"],
        summary="Get approval request details",
    )
    async def get_approval_request(approval_id: str) -> ApprovalRequestResponse:
        """Get details of a specific approval request."""
        state = get_app_state()
        
        # Find the request
        request = state.watcher._pending_approvals.get(approval_id)
        if not request:
            raise HTTPException(
                status_code=404,
                detail=f"Approval request '{approval_id}' not found"
            )
        
        return ApprovalRequestResponse(
            id=request.id,
            action={
                "script_id": request.action.script_id,
                "name": request.action.name,
                "description": request.action.description,
                "risk_level": request.action.risk_level.value,
                "parameters": request.action.parameters,
            },
            investigation=request.investigation.to_dict(),
            requested_at=request.requested_at,
            expires_at=request.expires_at,
            status=request.status,
            approved_by=request.approved_by,
            rejected_by=request.rejected_by,
            rejection_reason=request.rejection_reason,
        )
    
    @app.post(
        "/watcher/approvals/{approval_id}/approve",
        response_model=ApprovalActionResponse,
        tags=["Watcher"],
        summary="Approve a remediation action",
    )
    async def approve_action(
        approval_id: str,
        request: ApprovalActionRequest,
    ) -> ApprovalActionResponse:
        """
        Approve a pending remediation action.
        
        Once approved, the Watcher Agent will execute the action immediately.
        The execution result will be returned in the response.
        """
        state = get_app_state()
        
        result = await state.watcher.approve_action(
            approval_id=approval_id,
            approver=request.user,
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=400 if result["error"] != "Approval request not found" else 404,
                detail=result["error"]
            )
        
        return ApprovalActionResponse(
            status=result.get("status", "executed"),
            approval_id=approval_id,
            approved_by=result.get("approved_by"),
            result=result.get("result"),
        )
    
    @app.post(
        "/watcher/approvals/{approval_id}/reject",
        response_model=ApprovalActionResponse,
        tags=["Watcher"],
        summary="Reject a remediation action",
    )
    async def reject_action(
        approval_id: str,
        request: ApprovalActionRequest,
    ) -> ApprovalActionResponse:
        """
        Reject a pending remediation action.
        
        A reason must be provided explaining why the action was rejected.
        The action will not be executed.
        """
        state = get_app_state()
        
        if not request.reason:
            raise HTTPException(
                status_code=400,
                detail="Rejection reason is required"
            )
        
        result = await state.watcher.reject_action(
            approval_id=approval_id,
            rejector=request.user,
            reason=request.reason,
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=400 if result["error"] != "Approval request not found" else 404,
                detail=result["error"]
            )
        
        return ApprovalActionResponse(
            status=result.get("status", "rejected"),
            approval_id=approval_id,
            rejected_by=result.get("rejected_by"),
            reason=result.get("reason"),
        )
    
    @app.get(
        "/watcher/health",
        tags=["Watcher"],
        summary="Watcher Agent health check",
    )
    async def watcher_health() -> Dict[str, Any]:
        """Check the health and status of the Watcher Agent."""
        state = get_app_state()
        
        return {
            "status": "active" if state.watcher._started else "stopped",
            "mcp_endpoint": state.watcher.config.mcp_endpoint,
            "auto_remediation_enabled": state.watcher.config.auto_remediation_enabled,
            "dry_run_mode": state.watcher.config.dry_run_mode,
            "pending_approvals": len(state.watcher.get_pending_approvals()),
            "registered_scripts": list(state.watcher._scripts.keys()),
        }


# =============================================================================
# Helpers
# =============================================================================


def _workflow_to_response(workflow: WorkflowDefinition) -> WorkflowResponse:
    """Convert a WorkflowDefinition to a WorkflowResponse."""
    return WorkflowResponse(
        id=workflow.id,
        name=workflow.name,
        description=workflow.description,
        version=workflow.version,
        steps=[step.to_dict() for step in workflow.steps],
        input_schema=workflow.input_schema,
        output_schema=None,  # Not stored in WorkflowDefinition
        timeout_ms=workflow.timeout_ms,
        max_retries=0,
        created_at=workflow.created_at,
    )


def _result_to_response(result: WorkflowResult) -> WorkflowExecutionResponse:
    """Convert a WorkflowResult to a WorkflowExecutionResponse."""
    step_results = {}
    for step_id, step_result in result.step_results.items():
        step_results[step_id] = StepResultResponse(
            step_id=step_result.step_id,
            status=StepStatusEnum(step_result.status.value),
            output=step_result.output,
            error=step_result.error,
            error_category=step_result.error_category.value if step_result.error_category else None,
            started_at=step_result.started_at,
            completed_at=step_result.completed_at,
            execution_time_ms=step_result.execution_time_ms,
            retry_count=step_result.retry_count,
            used_fallback=step_result.used_fallback,
            metadata=step_result.metadata,
        )
    
    return WorkflowExecutionResponse(
        execution_id=result.execution_id,
        workflow_id=result.workflow_id,
        status=ExecutionStatusEnum(result.status.value),
        input=result.input,
        output=result.output,
        error=result.error,
        step_results=step_results,
        started_at=result.started_at,
        completed_at=result.completed_at,
        total_execution_time_ms=result.total_execution_time_ms,
        total_retry_count=result.total_retry_count,
        total_fallback_count=result.total_fallback_count,
        trace_id=result.metadata.get("trace_id") if result.metadata else None,
    )


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
