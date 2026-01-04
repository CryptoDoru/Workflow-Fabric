"""
AI Workflow Fabric - LangGraph Task Executor

This module provides the task execution engine for LangGraph agents,
handling execution, streaming, cancellation, and observability.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union
from uuid import uuid4

from awf.core.types import (
    Event,
    EventType,
    SandboxTier,
    Task,
    TaskError,
    TaskMetrics,
    TaskResult,
    TaskStatus,
)

# Type hints for LangGraph
try:
    from langgraph.graph.graph import CompiledGraph
    
    LANGGRAPH_AVAILABLE = True
except ImportError:
    CompiledGraph = Any
    LANGGRAPH_AVAILABLE = False


@dataclass
class ExecutionContext:
    """Context for a single task execution."""
    
    task: Task
    graph: CompiledGraph
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    span_id: str = field(default_factory=lambda: str(uuid4()))
    cancelled: bool = False
    
    # Execution state
    current_node: Optional[str] = None
    events: List[Event] = field(default_factory=list)
    
    # Metrics collection
    start_time: float = field(default_factory=time.perf_counter)
    token_usage: Dict[str, int] = field(default_factory=dict)


class TaskExecutor:
    """
    Executes tasks on LangGraph agents.
    
    This class handles:
    - Synchronous and streaming execution
    - Timeout and cancellation
    - Error handling and retries
    - Metrics collection
    - Event emission
    """
    
    def __init__(
        self,
        default_timeout_ms: int = 300000,  # 5 minutes
        max_retries: int = 0,
        retry_delay_ms: int = 1000,
    ):
        """
        Initialize the task executor.
        
        Args:
            default_timeout_ms: Default timeout for tasks
            max_retries: Maximum number of retries on failure
            retry_delay_ms: Delay between retries
        """
        self.default_timeout_ms = default_timeout_ms
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms
        
        # Track running tasks for cancellation
        self._running_tasks: Dict[str, ExecutionContext] = {}
        self._event_callbacks: List[Callable[[Event], None]] = []
    
    def on_event(self, callback: Callable[[Event], None]) -> None:
        """Register a callback for execution events."""
        self._event_callbacks.append(callback)
    
    async def execute(
        self,
        task: Task,
        graph: CompiledGraph,
        sandbox_tier: Optional[SandboxTier] = None,
    ) -> TaskResult:
        """
        Execute a task on a LangGraph graph.
        
        Args:
            task: The task to execute
            graph: The compiled LangGraph graph
            sandbox_tier: Optional sandbox tier for isolation
        
        Returns:
            TaskResult with execution outcome
        """
        context = ExecutionContext(task=task, graph=graph)
        self._running_tasks[task.id] = context
        
        try:
            # Emit task started event
            self._emit_event(Event(
                type=EventType.TASK_STARTED,
                source=task.agent_id,
                correlation_id=task.id,
                trace_id=task.trace_id,
                data={"input": task.input},
            ))
            
            # Execute with retry logic
            result = await self._execute_with_retry(context, sandbox_tier)
            
            # Emit completion/failure event
            if result.status == TaskStatus.COMPLETED:
                self._emit_event(Event(
                    type=EventType.TASK_COMPLETED,
                    source=task.agent_id,
                    correlation_id=task.id,
                    trace_id=task.trace_id,
                    data={"output": result.output, "metrics": result.metrics.to_dict() if result.metrics else None},
                ))
            else:
                self._emit_event(Event(
                    type=EventType.TASK_FAILED,
                    source=task.agent_id,
                    correlation_id=task.id,
                    trace_id=task.trace_id,
                    data={"error": result.error.to_dict() if result.error else None},
                ))
            
            return result
        
        finally:
            del self._running_tasks[task.id]
    
    async def execute_streaming(
        self,
        task: Task,
        graph: CompiledGraph,
        sandbox_tier: Optional[SandboxTier] = None,
    ) -> AsyncIterator[Event]:
        """
        Execute a task with streaming events.
        
        Yields events as they occur during execution.
        """
        context = ExecutionContext(task=task, graph=graph)
        self._running_tasks[task.id] = context
        
        try:
            # Emit task started
            started_event = Event(
                type=EventType.TASK_STARTED,
                source=task.agent_id,
                correlation_id=task.id,
                trace_id=task.trace_id,
                span_id=context.span_id,
                data={"input": task.input},
            )
            yield started_event
            
            # Execute with streaming
            async for event in self._stream_execution(context):
                yield event
        
        finally:
            del self._running_tasks[task.id]
    
    async def cancel(self, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            task_id: ID of the task to cancel
        
        Returns:
            True if cancelled, False if not found
        """
        context = self._running_tasks.get(task_id)
        if context:
            context.cancelled = True
            return True
        return False
    
    def get_running_tasks(self) -> List[str]:
        """Get IDs of all currently running tasks."""
        return list(self._running_tasks.keys())
    
    async def _execute_with_retry(
        self,
        context: ExecutionContext,
        sandbox_tier: Optional[SandboxTier],
    ) -> TaskResult:
        """Execute with retry logic."""
        last_error: Optional[Exception] = None
        
        for attempt in range(self.max_retries + 1):
            if context.cancelled:
                return self._create_cancelled_result(context)
            
            try:
                return await self._execute_once(context, sandbox_tier)
            
            except Exception as e:
                last_error = e
                
                if not self._is_retryable(e) or attempt >= self.max_retries:
                    break
                
                # Wait before retry
                await asyncio.sleep(self.retry_delay_ms / 1000.0)
        
        # All retries exhausted
        return self._create_error_result(context, last_error)
    
    async def _execute_once(
        self,
        context: ExecutionContext,
        sandbox_tier: Optional[SandboxTier],
    ) -> TaskResult:
        """Execute a single attempt."""
        task = context.task
        graph = context.graph
        
        # Determine timeout
        timeout_ms = task.timeout_ms or self.default_timeout_ms
        timeout_sec = timeout_ms / 1000.0
        
        # Prepare config
        config = self._build_config(context)
        
        try:
            # Run in thread pool with timeout
            loop = asyncio.get_event_loop()
            
            output = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: graph.invoke(task.input, config=config)
                ),
                timeout=timeout_sec,
            )
            
            # Calculate metrics
            execution_time = int((time.perf_counter() - context.start_time) * 1000)
            
            return TaskResult(
                task_id=task.id,
                agent_id=task.agent_id,
                status=TaskStatus.COMPLETED,
                output=self._normalize_output(output),
                metrics=TaskMetrics(
                    execution_time_ms=execution_time,
                    token_usage=context.token_usage if context.token_usage else None,
                    sandbox_tier=sandbox_tier,
                ),
                trace_id=task.trace_id,
                span_id=context.span_id,
                started_at=context.started_at,
                completed_at=datetime.now(timezone.utc),
            )
        
        except asyncio.TimeoutError:
            execution_time = int((time.perf_counter() - context.start_time) * 1000)
            
            return TaskResult(
                task_id=task.id,
                agent_id=task.agent_id,
                status=TaskStatus.TIMEOUT,
                error=TaskError(
                    code="TIMEOUT",
                    message=f"Task timed out after {timeout_ms}ms",
                    retryable=True,
                ),
                metrics=TaskMetrics(execution_time_ms=execution_time),
                trace_id=task.trace_id,
                span_id=context.span_id,
                started_at=context.started_at,
                completed_at=datetime.now(timezone.utc),
            )
        
        except asyncio.CancelledError:
            return self._create_cancelled_result(context)
    
    async def _stream_execution(
        self,
        context: ExecutionContext,
    ) -> AsyncIterator[Event]:
        """Stream execution events."""
        task = context.task
        graph = context.graph
        
        config = self._build_config(context)
        
        try:
            # Get stream from graph
            loop = asyncio.get_event_loop()
            
            def get_stream():
                return list(graph.stream(task.input, config=config))
            
            chunks = await loop.run_in_executor(None, get_stream)
            
            # Yield state change events for each chunk
            for i, chunk in enumerate(chunks):
                if context.cancelled:
                    yield Event(
                        type=EventType.TASK_FAILED,
                        source=task.agent_id,
                        correlation_id=task.id,
                        trace_id=task.trace_id,
                        span_id=context.span_id,
                        data={"error": {"code": "CANCELLED", "message": "Task was cancelled"}},
                    )
                    return
                
                # Extract node name from chunk if available
                node_name = None
                if isinstance(chunk, dict):
                    node_name = list(chunk.keys())[0] if chunk else None
                
                yield Event(
                    type=EventType.STATE_CHANGED,
                    source=task.agent_id,
                    correlation_id=task.id,
                    trace_id=task.trace_id,
                    span_id=context.span_id,
                    data={
                        "chunk_index": i,
                        "node": node_name,
                        "state": chunk,
                    },
                )
            
            # Emit completion
            execution_time = int((time.perf_counter() - context.start_time) * 1000)
            final_output = chunks[-1] if chunks else {}
            
            yield Event(
                type=EventType.TASK_COMPLETED,
                source=task.agent_id,
                correlation_id=task.id,
                trace_id=task.trace_id,
                span_id=context.span_id,
                data={
                    "output": self._normalize_output(final_output),
                    "metrics": {"executionTimeMs": execution_time},
                },
            )
        
        except Exception as e:
            execution_time = int((time.perf_counter() - context.start_time) * 1000)
            
            yield Event(
                type=EventType.TASK_FAILED,
                source=task.agent_id,
                correlation_id=task.id,
                trace_id=task.trace_id,
                span_id=context.span_id,
                data={
                    "error": {
                        "code": "EXECUTION_ERROR",
                        "message": str(e),
                        "stackTrace": traceback.format_exc(),
                    },
                    "metrics": {"executionTimeMs": execution_time},
                },
            )
    
    def _build_config(self, context: ExecutionContext) -> Dict[str, Any]:
        """Build LangGraph config from execution context."""
        return {
            "configurable": {
                "task_id": context.task.id,
                "trace_id": context.task.trace_id,
                "span_id": context.span_id,
                "awf_context": context.task.context,
            },
            "callbacks": self._create_callbacks(context),
        }
    
    def _create_callbacks(self, context: ExecutionContext) -> List[Any]:
        """Create LangGraph callbacks for observability."""
        # This could be extended to integrate with LangSmith or custom callbacks
        return []
    
    def _normalize_output(self, output: Any) -> Dict[str, Any]:
        """Normalize graph output to a dictionary."""
        if isinstance(output, dict):
            return output
        return {"result": output}
    
    def _create_cancelled_result(self, context: ExecutionContext) -> TaskResult:
        """Create a TaskResult for a cancelled task."""
        execution_time = int((time.perf_counter() - context.start_time) * 1000)
        
        return TaskResult(
            task_id=context.task.id,
            agent_id=context.task.agent_id,
            status=TaskStatus.CANCELLED,
            error=TaskError(
                code="CANCELLED",
                message="Task was cancelled",
                retryable=False,
            ),
            metrics=TaskMetrics(execution_time_ms=execution_time),
            trace_id=context.task.trace_id,
            span_id=context.span_id,
            started_at=context.started_at,
            completed_at=datetime.now(timezone.utc),
        )
    
    def _create_error_result(
        self,
        context: ExecutionContext,
        error: Optional[Exception],
    ) -> TaskResult:
        """Create a TaskResult for an error."""
        execution_time = int((time.perf_counter() - context.start_time) * 1000)
        
        return TaskResult(
            task_id=context.task.id,
            agent_id=context.task.agent_id,
            status=TaskStatus.FAILED,
            error=TaskError(
                code="EXECUTION_ERROR",
                message=str(error) if error else "Unknown error",
                stack_trace=traceback.format_exc() if error else None,
                retryable=self._is_retryable(error) if error else False,
            ),
            metrics=TaskMetrics(execution_time_ms=execution_time),
            trace_id=context.task.trace_id,
            span_id=context.span_id,
            started_at=context.started_at,
            completed_at=datetime.now(timezone.utc),
        )
    
    def _is_retryable(self, error: Optional[Exception]) -> bool:
        """Determine if an error is retryable."""
        if error is None:
            return False
        
        retryable_types = (
            TimeoutError,
            ConnectionError,
            asyncio.TimeoutError,
        )
        
        # Check error type
        if isinstance(error, retryable_types):
            return True
        
        # Check error message for retryable patterns
        error_msg = str(error).lower()
        retryable_patterns = [
            "timeout",
            "connection reset",
            "connection refused",
            "temporary failure",
            "rate limit",
            "too many requests",
            "service unavailable",
            "502",
            "503",
            "504",
        ]
        
        return any(pattern in error_msg for pattern in retryable_patterns)
    
    def _emit_event(self, event: Event) -> None:
        """Emit an event to all registered callbacks."""
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception:
                pass  # Don't let callback errors affect execution
