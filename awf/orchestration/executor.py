"""
AI Workflow Fabric - Step Executor

This module provides step execution with retry and fallback handling.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Protocol, Set

from awf.core.types import Task, TaskResult, TaskStatus, Event
from awf.orchestration.types import (
    ErrorCategory,
    ExecutionContext,
    FallbackPolicy,
    RetryPolicy,
    StepDefinition,
    StepResult,
    StepStatus,
    WorkflowEvent,
    WorkflowEventType,
)
from awf.orchestration.mapping import InputMapper, JSONPathError
from awf.orchestration.errors import (
    AgentNotFoundError,
    AgentInactiveError,
    StepTimeoutError,
    StepExecutionError,
    InputMappingError,
    FallbackError,
    NoFallbackAvailableError,
    AdapterNotFoundError,
    categorize_exception,
)


# =============================================================================
# Adapter Protocol
# =============================================================================


class AdapterProtocol(Protocol):
    """Protocol defining the interface adapters must implement."""
    
    async def execute(self, task: Task) -> TaskResult:
        """Execute a task on an agent."""
        ...
    
    async def execute_streaming(self, task: Task) -> AsyncIterator[Event]:
        """Execute a task with streaming events."""
        ...
    
    async def cancel(self, task_id: str) -> bool:
        """Cancel a running task."""
        ...


class AdapterRegistry(Protocol):
    """Protocol for adapter registry."""
    
    def get_adapter(self, framework: str) -> Optional[AdapterProtocol]:
        """Get an adapter by framework name."""
        ...
    
    def get_agent_framework(self, agent_id: str) -> Optional[str]:
        """Get the framework for an agent."""
        ...


class AgentRegistryProtocol(Protocol):
    """Protocol for agent registry."""
    
    async def get(self, agent_id: str) -> Optional[Any]:
        """Get an agent manifest by ID."""
        ...


# =============================================================================
# Event Callback
# =============================================================================


EventCallback = Callable[[WorkflowEvent], None]
AsyncEventCallback = Callable[[WorkflowEvent], Any]


# =============================================================================
# Step Executor
# =============================================================================


@dataclass
class StepExecutorConfig:
    """Configuration for step execution."""
    
    default_timeout_ms: int = 60000  # 1 minute
    default_retry: Optional[RetryPolicy] = None
    max_parallel_steps: int = 10
    emit_events: bool = True


class StepExecutor:
    """
    Executes workflow steps with retry and fallback handling.
    
    The StepExecutor is responsible for:
    - Mapping inputs from execution context
    - Calling the appropriate adapter for the agent
    - Handling timeouts
    - Retry with exponential backoff
    - Fallback execution
    - Event emission
    """
    
    def __init__(
        self,
        adapter_registry: AdapterRegistry,
        agent_registry: AgentRegistryProtocol,
        config: Optional[StepExecutorConfig] = None,
        event_callback: Optional[AsyncEventCallback] = None,
    ):
        """
        Initialize the step executor.
        
        Args:
            adapter_registry: Registry of framework adapters
            agent_registry: Registry of agents
            config: Executor configuration
            event_callback: Optional callback for workflow events
        """
        self.adapter_registry = adapter_registry
        self.agent_registry = agent_registry
        self.config = config or StepExecutorConfig()
        self.event_callback = event_callback
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._cancelled: Set[str] = set()
    
    async def execute_step(
        self,
        step: StepDefinition,
        context: ExecutionContext,
        default_retry: Optional[RetryPolicy] = None,
    ) -> StepResult:
        """
        Execute a single workflow step.
        
        Args:
            step: The step definition to execute
            context: The workflow execution context
            default_retry: Default retry policy if step doesn't define one
        
        Returns:
            StepResult with execution outcome
        """
        started_at = datetime.now(timezone.utc)
        
        # Check if cancelled
        if step.id in self._cancelled:
            return StepResult(
                step_id=step.id,
                status=StepStatus.SKIPPED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                metadata={"reason": "cancelled"},
            )
        
        # Emit step started event
        await self._emit_event(WorkflowEvent(
            type=WorkflowEventType.STEP_STARTED,
            execution_id=context.execution_id,
            workflow_id=context.workflow.id,
            step_id=step.id,
            data={"agent_id": step.agent_id},
        ))
        
        try:
            # Map inputs
            input_data = await self._map_inputs(step, context)
            
            # Get retry policy
            retry_policy = step.retry or default_retry or self.config.default_retry
            
            # Execute with retry
            result = await self._execute_with_retry(
                step=step,
                input_data=input_data,
                context=context,
                retry_policy=retry_policy,
                started_at=started_at,
            )
            
            return result
            
        except Exception as e:
            # Handle fallback if available
            if step.fallback:
                return await self._execute_fallback(
                    step=step,
                    context=context,
                    original_error=e,
                    started_at=started_at,
                )
            
            # No fallback, create error result
            completed_at = datetime.now(timezone.utc)
            error_category = categorize_exception(e)
            
            await self._emit_event(WorkflowEvent(
                type=WorkflowEventType.STEP_FAILED,
                execution_id=context.execution_id,
                workflow_id=context.workflow.id,
                step_id=step.id,
                data={
                    "error": str(e),
                    "category": error_category.value,
                },
            ))
            
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error={
                    "message": str(e),
                    "type": type(e).__name__,
                },
                error_category=error_category,
                started_at=started_at,
                completed_at=completed_at,
                execution_time_ms=int(
                    (completed_at - started_at).total_seconds() * 1000
                ),
            )
    
    async def cancel_step(self, step_id: str) -> bool:
        """
        Cancel a running step.
        
        Args:
            step_id: The step ID to cancel
        
        Returns:
            True if cancellation was requested
        """
        self._cancelled.add(step_id)
        
        if step_id in self._active_tasks:
            task = self._active_tasks[step_id]
            task.cancel()
            return True
        
        return False
    
    async def _map_inputs(
        self,
        step: StepDefinition,
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """Map step inputs from context."""
        if not step.input_map:
            return {}
        
        try:
            mapper = InputMapper(context)
            return mapper.map_inputs(step.input_map, strict=True)
        except JSONPathError as e:
            raise InputMappingError(
                step_id=step.id,
                expression=str(e),
                reason=str(e),
                execution_id=context.execution_id,
                workflow_id=context.workflow.id,
            ) from e
    
    async def _execute_with_retry(
        self,
        step: StepDefinition,
        input_data: Dict[str, Any],
        context: ExecutionContext,
        retry_policy: Optional[RetryPolicy],
        started_at: datetime,
    ) -> StepResult:
        """Execute step with retry logic."""
        max_attempts = retry_policy.max_attempts if retry_policy else 1
        last_error: Optional[Exception] = None
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Check cancellation
                if step.id in self._cancelled:
                    raise asyncio.CancelledError()
                
                # Execute the step
                result = await self._execute_single_attempt(
                    step=step,
                    input_data=input_data,
                    context=context,
                )
                
                # Success
                completed_at = datetime.now(timezone.utc)
                
                await self._emit_event(WorkflowEvent(
                    type=WorkflowEventType.STEP_COMPLETED,
                    execution_id=context.execution_id,
                    workflow_id=context.workflow.id,
                    step_id=step.id,
                    data={"output": result},
                ))
                
                return StepResult(
                    step_id=step.id,
                    status=StepStatus.COMPLETED,
                    output=result,
                    started_at=started_at,
                    completed_at=completed_at,
                    execution_time_ms=int(
                        (completed_at - started_at).total_seconds() * 1000
                    ),
                    retry_count=attempt,
                )
                
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_error = e
                error_category = categorize_exception(e)
                
                # Check if we should retry
                if retry_policy and retry_policy.should_retry(error_category, attempt + 1):
                    attempt += 1
                    delay_ms = retry_policy.get_delay_ms(attempt)
                    
                    await self._emit_event(WorkflowEvent(
                        type=WorkflowEventType.STEP_RETRYING,
                        execution_id=context.execution_id,
                        workflow_id=context.workflow.id,
                        step_id=step.id,
                        data={
                            "attempt": attempt + 1,
                            "max_attempts": max_attempts,
                            "delay_ms": delay_ms,
                            "error": str(e),
                        },
                    ))
                    
                    await asyncio.sleep(delay_ms / 1000.0)
                    continue
                
                # Can't retry, re-raise
                raise
        
        # Exhausted retries
        if last_error:
            raise last_error
        
        raise StepExecutionError(
            message=f"Step '{step.id}' failed after {max_attempts} attempts",
            step_id=step.id,
            execution_id=context.execution_id,
            workflow_id=context.workflow.id,
        )
    
    async def _execute_single_attempt(
        self,
        step: StepDefinition,
        input_data: Dict[str, Any],
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """Execute a single attempt of a step."""
        # Get the agent's framework
        agent = await self.agent_registry.get(step.agent_id)
        if agent is None:
            raise AgentNotFoundError(
                agent_id=step.agent_id,
                step_id=step.id,
                execution_id=context.execution_id,
                workflow_id=context.workflow.id,
            )
        
        # Get the adapter
        framework = getattr(agent, 'framework', None)
        if not framework:
            raise AdapterNotFoundError(
                framework="unknown",
                agent_id=step.agent_id,
            )
        
        adapter = self.adapter_registry.get_adapter(framework)
        if adapter is None:
            raise AdapterNotFoundError(
                framework=framework,
                agent_id=step.agent_id,
            )
        
        # Create task
        task = Task(
            id=f"{context.execution_id}-{step.id}",
            agent_id=step.agent_id,
            input=input_data,
            correlation_id=context.correlation_id,
            metadata={
                "workflow_id": context.workflow.id,
                "execution_id": context.execution_id,
                "step_id": step.id,
            },
        )
        
        # Execute with timeout
        timeout_ms = step.timeout_ms or self.config.default_timeout_ms
        timeout_seconds = timeout_ms / 1000.0
        
        try:
            # Track active task
            coro = adapter.execute(task)
            async_task = asyncio.create_task(coro)
            self._active_tasks[step.id] = async_task
            
            try:
                result = await asyncio.wait_for(async_task, timeout=timeout_seconds)
            finally:
                self._active_tasks.pop(step.id, None)
            
            # Check result status
            if result.status == TaskStatus.FAILED:
                error_msg = result.error.get("message", "Unknown error") if result.error else "Unknown error"
                raise StepExecutionError(
                    message=error_msg,
                    step_id=step.id,
                    execution_id=context.execution_id,
                    workflow_id=context.workflow.id,
                    category=ErrorCategory.EXTERNAL_SERVICE_ERROR,
                    details=result.error,
                )
            
            return result.output or {}
            
        except asyncio.TimeoutError:
            # Cancel the task
            if step.id in self._active_tasks:
                self._active_tasks[step.id].cancel()
                self._active_tasks.pop(step.id, None)
            
            await self._emit_event(WorkflowEvent(
                type=WorkflowEventType.STEP_TIMEOUT,
                execution_id=context.execution_id,
                workflow_id=context.workflow.id,
                step_id=step.id,
                data={"timeout_ms": timeout_ms},
            ))
            
            raise StepTimeoutError(
                step_id=step.id,
                timeout_ms=timeout_ms,
                execution_id=context.execution_id,
                workflow_id=context.workflow.id,
            )
    
    async def _execute_fallback(
        self,
        step: StepDefinition,
        context: ExecutionContext,
        original_error: Exception,
        started_at: datetime,
    ) -> StepResult:
        """Execute fallback for a failed step."""
        fallback = step.fallback
        if not fallback:
            raise NoFallbackAvailableError(
                step_id=step.id,
                original_error=str(original_error),
                execution_id=context.execution_id,
                workflow_id=context.workflow.id,
            )
        
        await self._emit_event(WorkflowEvent(
            type=WorkflowEventType.STEP_FALLBACK,
            execution_id=context.execution_id,
            workflow_id=context.workflow.id,
            step_id=step.id,
            data={
                "original_error": str(original_error),
                "fallback_type": self._get_fallback_type(fallback),
            },
        ))
        
        try:
            if fallback.skip:
                # Skip this step
                completed_at = datetime.now(timezone.utc)
                
                await self._emit_event(WorkflowEvent(
                    type=WorkflowEventType.STEP_SKIPPED,
                    execution_id=context.execution_id,
                    workflow_id=context.workflow.id,
                    step_id=step.id,
                    data={"reason": "fallback_skip"},
                ))
                
                return StepResult(
                    step_id=step.id,
                    status=StepStatus.SKIPPED,
                    started_at=started_at,
                    completed_at=completed_at,
                    execution_time_ms=int(
                        (completed_at - started_at).total_seconds() * 1000
                    ),
                    used_fallback=True,
                    metadata={"original_error": str(original_error)},
                )
            
            elif fallback.static_value is not None:
                # Use static value
                completed_at = datetime.now(timezone.utc)
                
                await self._emit_event(WorkflowEvent(
                    type=WorkflowEventType.STEP_COMPLETED,
                    execution_id=context.execution_id,
                    workflow_id=context.workflow.id,
                    step_id=step.id,
                    data={"output": fallback.static_value, "fallback": True},
                ))
                
                return StepResult(
                    step_id=step.id,
                    status=StepStatus.COMPLETED,
                    output=fallback.static_value,
                    started_at=started_at,
                    completed_at=completed_at,
                    execution_time_ms=int(
                        (completed_at - started_at).total_seconds() * 1000
                    ),
                    used_fallback=True,
                    metadata={"original_error": str(original_error)},
                )
            
            elif fallback.agent_id:
                # Execute fallback agent
                fallback_step = StepDefinition(
                    id=f"{step.id}_fallback",
                    agent_id=fallback.agent_id,
                    input_map=step.input_map,
                    timeout_ms=step.timeout_ms,
                )
                
                # Execute without retry or further fallback
                result = await self._execute_with_retry(
                    step=fallback_step,
                    input_data=await self._map_inputs(step, context),
                    context=context,
                    retry_policy=None,
                    started_at=started_at,
                )
                
                result.step_id = step.id
                result.used_fallback = True
                result.metadata["original_error"] = str(original_error)
                result.metadata["fallback_agent"] = fallback.agent_id
                
                return result
            
            else:
                raise NoFallbackAvailableError(
                    step_id=step.id,
                    original_error=str(original_error),
                    execution_id=context.execution_id,
                    workflow_id=context.workflow.id,
                )
                
        except Exception as e:
            if isinstance(e, NoFallbackAvailableError):
                raise
            
            raise FallbackError(
                step_id=step.id,
                original_error=str(original_error),
                fallback_error=str(e),
                execution_id=context.execution_id,
                workflow_id=context.workflow.id,
            ) from e
    
    def _get_fallback_type(self, fallback: FallbackPolicy) -> str:
        """Get a descriptive type for a fallback policy."""
        if fallback.skip:
            return "skip"
        elif fallback.static_value is not None:
            return "static_value"
        elif fallback.agent_id:
            return "agent"
        return "unknown"
    
    async def _emit_event(self, event: WorkflowEvent) -> None:
        """Emit a workflow event."""
        if not self.config.emit_events:
            return
        
        if self.event_callback:
            try:
                result = self.event_callback(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                # Don't let event emission failures affect execution
                pass
