"""
AI Workflow Fabric - Workflow Orchestrator

This module provides the main Orchestrator class for executing workflows.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set
from uuid import uuid4

from awf.orchestration.types import (
    ExecutionContext,
    ExecutionStatus,
    StepDefinition,
    StepResult,
    StepStatus,
    WorkflowDefinition,
    WorkflowEvent,
    WorkflowEventType,
    WorkflowResult,
)
from awf.orchestration.dag import build_dag, WorkflowDAG
from awf.orchestration.mapping import OutputMapper
from awf.orchestration.validation import validate_workflow, ValidationResult
from awf.orchestration.executor import (
    StepExecutor,
    StepExecutorConfig,
    AdapterRegistry,
    AgentRegistryProtocol,
    AsyncEventCallback,
)
from awf.orchestration.errors import (
    WorkflowDefinitionError,
    WorkflowTimeoutError,
    WorkflowCancelledError,
    ConditionEvaluationError,
    DependencyFailedError,
    ExecutionError,
)


# =============================================================================
# Orchestrator Configuration
# =============================================================================


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    
    # Execution settings
    default_timeout_ms: int = 300000  # 5 minutes
    max_parallel_steps: int = 10
    
    # Validation settings
    validate_on_execute: bool = True
    strict_validation: bool = False
    
    # Event settings
    emit_events: bool = True
    
    # Step executor config
    step_executor_config: Optional[StepExecutorConfig] = None


# =============================================================================
# Condition Evaluator
# =============================================================================


class ConditionEvaluator:
    """
    Evaluates step conditions to determine if a step should run.
    
    Conditions are Python expressions that can reference:
    - $.input.X - workflow input
    - $.steps.Y.output.Z - step outputs
    - $.steps.Y.status - step status
    - $.context.X - context values
    """
    
    def __init__(self, context: ExecutionContext):
        """Initialize with execution context."""
        self.context = context
    
    def evaluate(self, condition: str, step_id: str) -> bool:
        """
        Evaluate a condition expression.
        
        Args:
            condition: The condition expression
            step_id: The step ID (for error messages)
        
        Returns:
            True if condition is met, False otherwise
        
        Raises:
            ConditionEvaluationError: If condition evaluation fails
        """
        if not condition:
            return True
        
        try:
            # Build evaluation context
            eval_context = self._build_eval_context()
            
            # Replace JSONPath-like references with Python dict access
            python_expr = self._convert_to_python_expr(condition)
            
            # Evaluate safely
            result = eval(python_expr, {"__builtins__": {}}, eval_context)
            
            return bool(result)
            
        except Exception as e:
            raise ConditionEvaluationError(
                step_id=step_id,
                condition=condition,
                reason=str(e),
                execution_id=self.context.execution_id,
                workflow_id=self.context.workflow.id,
            ) from e
    
    def _build_eval_context(self) -> Dict[str, Any]:
        """Build the evaluation context dict."""
        # Build steps data
        steps_data: Dict[str, Any] = {}
        for step_id, result in self.context.step_results.items():
            steps_data[step_id] = {
                "output": result.output or {},
                "status": result.status.value,
                "error": result.error,
            }
        
        return {
            "input": self.context.input,
            "steps": steps_data,
            "context": self.context.context,
            # Safe built-ins
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "True": True,
            "False": False,
            "None": None,
        }
    
    def _convert_to_python_expr(self, condition: str) -> str:
        """Convert JSONPath-like expressions to Python dict access."""
        import re
        
        # Replace $.X.Y with X["Y"]
        def replacer(match: re.Match) -> str:
            path = match.group(1)
            parts = path.split(".")
            result = parts[0]
            for part in parts[1:]:
                # Handle array notation
                if "[" in part:
                    base = part.split("[")[0]
                    index = part.split("[")[1].rstrip("]")
                    result += f'["{base}"][{index}]'
                else:
                    result += f'["{part}"]'
            return result
        
        # Match $. followed by path
        return re.sub(r'\$\.([a-zA-Z_][a-zA-Z0-9_.\[\]]*)', replacer, condition)


# =============================================================================
# Orchestrator
# =============================================================================


class Orchestrator:
    """
    Main workflow orchestrator.
    
    The Orchestrator is responsible for:
    - Validating workflow definitions
    - Building execution DAGs
    - Executing steps in correct order (sequential or parallel)
    - Managing execution state
    - Handling workflow-level timeouts
    - Emitting events for observability
    """
    
    def __init__(
        self,
        adapter_registry: AdapterRegistry,
        agent_registry: AgentRegistryProtocol,
        config: Optional[OrchestratorConfig] = None,
        event_callback: Optional[AsyncEventCallback] = None,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            adapter_registry: Registry of framework adapters
            agent_registry: Registry of agents
            config: Orchestrator configuration
            event_callback: Optional callback for workflow events
        """
        self.adapter_registry = adapter_registry
        self.agent_registry = agent_registry
        self.config = config or OrchestratorConfig()
        self.event_callback = event_callback
        
        # Create step executor
        step_config = self.config.step_executor_config or StepExecutorConfig(
            default_timeout_ms=self.config.default_timeout_ms,
            max_parallel_steps=self.config.max_parallel_steps,
            emit_events=self.config.emit_events,
        )
        self.step_executor = StepExecutor(
            adapter_registry=adapter_registry,
            agent_registry=agent_registry,
            config=step_config,
            event_callback=event_callback,
        )
        
        # Active executions
        self._active_executions: Dict[str, ExecutionContext] = {}
        self._cancelled_executions: Set[str] = set()
    
    async def execute(
        self,
        workflow: WorkflowDefinition,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        timeout_ms: Optional[int] = None,
        trace_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Execute a workflow.
        
        Args:
            workflow: The workflow definition to execute
            input_data: Input data for the workflow
            context: Optional execution context data
            timeout_ms: Optional timeout override
            trace_id: Optional trace ID for observability
            correlation_id: Optional correlation ID
        
        Returns:
            WorkflowResult with execution outcome
        
        Raises:
            WorkflowDefinitionError: If workflow is invalid
            WorkflowTimeoutError: If execution times out
            WorkflowCancelledError: If execution is cancelled
        """
        # Create execution context
        exec_context = ExecutionContext.create(
            workflow=workflow,
            input=input_data,
            context=context,
            trace_id=trace_id,
            correlation_id=correlation_id,
        )
        
        # Validate workflow
        if self.config.validate_on_execute:
            validation = validate_workflow(workflow, strict=self.config.strict_validation)
            if not validation.valid:
                error_messages = [e.message for e in validation.errors]
                raise WorkflowDefinitionError(
                    message=f"Workflow validation failed: {'; '.join(error_messages)}",
                    workflow_id=workflow.id,
                    details={"errors": [e.to_dict() for e in validation.errors]},
                )
        
        # Track active execution
        self._active_executions[exec_context.execution_id] = exec_context
        
        try:
            # Execute with timeout
            effective_timeout = timeout_ms or workflow.timeout_ms or self.config.default_timeout_ms
            timeout_seconds = effective_timeout / 1000.0
            
            result = await asyncio.wait_for(
                self._execute_workflow(exec_context),
                timeout=timeout_seconds,
            )
            
            return result
            
        except asyncio.TimeoutError:
            return await self._handle_timeout(exec_context, effective_timeout)
        except asyncio.CancelledError:
            return await self._handle_cancellation(exec_context)
        finally:
            self._active_executions.pop(exec_context.execution_id, None)
    
    async def execute_streaming(
        self,
        workflow: WorkflowDefinition,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[WorkflowEvent]:
        """
        Execute a workflow with streaming events.
        
        Yields events as they occur during execution.
        
        Args:
            workflow: The workflow definition
            input_data: Input data for the workflow
            context: Optional execution context
        
        Yields:
            WorkflowEvent objects as execution progresses
        """
        # Create event queue
        event_queue: asyncio.Queue[Optional[WorkflowEvent]] = asyncio.Queue()
        
        async def queue_callback(event: WorkflowEvent) -> None:
            await event_queue.put(event)
        
        # Store original callback
        original_callback = self.event_callback
        self.event_callback = queue_callback
        self.step_executor.event_callback = queue_callback
        
        try:
            # Start execution in background
            exec_task = asyncio.create_task(
                self.execute(workflow, input_data, context)
            )
            
            # Yield events as they come
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                    if event is None:
                        break
                    yield event
                except asyncio.TimeoutError:
                    if exec_task.done():
                        # Drain remaining events
                        while not event_queue.empty():
                            event = event_queue.get_nowait()
                            if event:
                                yield event
                        break
            
            # Await the result
            await exec_task
            
        finally:
            self.event_callback = original_callback
            self.step_executor.event_callback = original_callback
    
    async def cancel(self, execution_id: str, reason: Optional[str] = None) -> bool:
        """
        Cancel a running workflow execution.
        
        Args:
            execution_id: The execution ID to cancel
            reason: Optional reason for cancellation
        
        Returns:
            True if cancellation was requested
        """
        if execution_id in self._active_executions:
            self._cancelled_executions.add(execution_id)
            
            # Cancel all active steps
            context = self._active_executions[execution_id]
            for step in context.workflow.steps:
                await self.step_executor.cancel_step(step.id)
            
            return True
        
        return False
    
    def get_execution_status(self, execution_id: str) -> Optional[ExecutionContext]:
        """
        Get the current status of an execution.
        
        Args:
            execution_id: The execution ID
        
        Returns:
            ExecutionContext if found, None otherwise
        """
        return self._active_executions.get(execution_id)
    
    async def _execute_workflow(self, context: ExecutionContext) -> WorkflowResult:
        """Execute the workflow steps."""
        started_at = datetime.now(timezone.utc)
        context.status = ExecutionStatus.RUNNING
        context.started_at = started_at
        
        # Emit workflow started event
        await self._emit_event(WorkflowEvent(
            type=WorkflowEventType.WORKFLOW_STARTED,
            execution_id=context.execution_id,
            workflow_id=context.workflow.id,
            data={
                "input": context.input,
                "step_count": len(context.workflow.steps),
            },
        ))
        
        try:
            # Build DAG
            dag = build_dag(context.workflow)
            
            # Execute steps
            await self._execute_steps(context, dag)
            
            # Check for failures
            failed_steps = [
                step_id for step_id, result in context.step_results.items()
                if result.status == StepStatus.FAILED
            ]
            
            if failed_steps:
                context.status = ExecutionStatus.FAILED
                error_msg = f"Steps failed: {', '.join(failed_steps)}"
                
                await self._emit_event(WorkflowEvent(
                    type=WorkflowEventType.WORKFLOW_FAILED,
                    execution_id=context.execution_id,
                    workflow_id=context.workflow.id,
                    data={
                        "failed_steps": failed_steps,
                        "error": error_msg,
                    },
                ))
                
                return self._build_result(context, started_at, error={"message": error_msg})
            
            # Map output
            output = self._map_output(context)
            
            context.status = ExecutionStatus.COMPLETED
            context.completed_at = datetime.now(timezone.utc)
            
            await self._emit_event(WorkflowEvent(
                type=WorkflowEventType.WORKFLOW_COMPLETED,
                execution_id=context.execution_id,
                workflow_id=context.workflow.id,
                data={"output": output},
            ))
            
            return self._build_result(context, started_at, output=output)
            
        except Exception as e:
            context.status = ExecutionStatus.FAILED
            
            await self._emit_event(WorkflowEvent(
                type=WorkflowEventType.WORKFLOW_FAILED,
                execution_id=context.execution_id,
                workflow_id=context.workflow.id,
                data={"error": str(e)},
            ))
            
            return self._build_result(
                context, started_at,
                error={"message": str(e), "type": type(e).__name__},
            )
    
    async def _execute_steps(
        self,
        context: ExecutionContext,
        dag: WorkflowDAG,
    ) -> None:
        """Execute all steps in the workflow."""
        completed: Set[str] = set()
        
        while len(completed) < len(dag.nodes):
            # Check for cancellation
            if context.execution_id in self._cancelled_executions:
                raise WorkflowCancelledError(
                    execution_id=context.execution_id,
                    workflow_id=context.workflow.id,
                )
            
            # Get ready steps
            ready_steps = dag.get_ready_steps(completed)
            
            if not ready_steps:
                # Check if we're stuck due to failures
                pending = set(dag.nodes.keys()) - completed
                for step_id in pending:
                    step = dag.get_node(step_id)
                    if step:
                        for dep in step.dependencies:
                            dep_result = context.step_results.get(dep)
                            if dep_result and dep_result.status == StepStatus.FAILED:
                                # Skip this step due to dependency failure
                                context.set_step_result(StepResult(
                                    step_id=step_id,
                                    status=StepStatus.SKIPPED,
                                    error={"message": f"Dependency '{dep}' failed"},
                                ))
                                completed.add(step_id)
                
                # If still no ready steps, we're done
                ready_steps = dag.get_ready_steps(completed)
                if not ready_steps:
                    break
            
            # Execute ready steps in parallel (up to max)
            batch_size = min(len(ready_steps), self.config.max_parallel_steps)
            batch = ready_steps[:batch_size]
            
            tasks = []
            for step_id in batch:
                step_node = dag.get_node(step_id)
                if step_node:
                    task = asyncio.create_task(
                        self._execute_step(context, step_node.step)
                    )
                    tasks.append((step_id, task))
            
            # Wait for all tasks in batch
            for step_id, task in tasks:
                try:
                    result = await task
                    context.set_step_result(result)
                except Exception as e:
                    context.set_step_result(StepResult(
                        step_id=step_id,
                        status=StepStatus.FAILED,
                        error={"message": str(e), "type": type(e).__name__},
                    ))
                completed.add(step_id)
    
    async def _execute_step(
        self,
        context: ExecutionContext,
        step: StepDefinition,
    ) -> StepResult:
        """Execute a single step."""
        # Evaluate condition
        if step.condition:
            evaluator = ConditionEvaluator(context)
            should_run = evaluator.evaluate(step.condition, step.id)
            
            if not should_run:
                await self._emit_event(WorkflowEvent(
                    type=WorkflowEventType.STEP_SKIPPED,
                    execution_id=context.execution_id,
                    workflow_id=context.workflow.id,
                    step_id=step.id,
                    data={"reason": "condition_not_met"},
                ))
                
                return StepResult(
                    step_id=step.id,
                    status=StepStatus.SKIPPED,
                    metadata={"reason": "condition_not_met"},
                )
        
        # Execute the step
        return await self.step_executor.execute_step(
            step=step,
            context=context,
            default_retry=context.workflow.default_retry,
        )
    
    def _map_output(self, context: ExecutionContext) -> Dict[str, Any]:
        """Map workflow output from step results."""
        mapper = OutputMapper(context)
        return mapper.map_output(context.workflow.output_map)
    
    def _build_result(
        self,
        context: ExecutionContext,
        started_at: datetime,
        output: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """Build the final workflow result."""
        completed_at = datetime.now(timezone.utc)
        
        # Aggregate metrics
        total_retries = sum(
            r.retry_count for r in context.step_results.values()
        )
        total_fallbacks = sum(
            1 for r in context.step_results.values() if r.used_fallback
        )
        
        # Aggregate token usage
        total_tokens: Dict[str, int] = {}
        for result in context.step_results.values():
            if result.token_usage:
                for key, value in result.token_usage.items():
                    total_tokens[key] = total_tokens.get(key, 0) + value
        
        return WorkflowResult(
            execution_id=context.execution_id,
            workflow_id=context.workflow.id,
            status=context.status,
            input=context.input,
            output=output,
            error=error,
            step_results=context.step_results,
            started_at=started_at,
            completed_at=completed_at,
            total_execution_time_ms=int(
                (completed_at - started_at).total_seconds() * 1000
            ),
            total_token_usage=total_tokens if total_tokens else None,
            total_retry_count=total_retries,
            total_fallback_count=total_fallbacks,
            metadata={
                "trace_id": context.trace_id,
                "correlation_id": context.correlation_id,
            },
        )
    
    async def _handle_timeout(
        self,
        context: ExecutionContext,
        timeout_ms: int,
    ) -> WorkflowResult:
        """Handle workflow timeout."""
        context.status = ExecutionStatus.TIMEOUT
        
        await self._emit_event(WorkflowEvent(
            type=WorkflowEventType.WORKFLOW_FAILED,
            execution_id=context.execution_id,
            workflow_id=context.workflow.id,
            data={
                "error": f"Workflow timed out after {timeout_ms}ms",
                "current_step": context.current_step,
            },
        ))
        
        return self._build_result(
            context,
            context.started_at or datetime.now(timezone.utc),
            error={
                "message": f"Workflow timed out after {timeout_ms}ms",
                "type": "TimeoutError",
                "current_step": context.current_step,
            },
        )
    
    async def _handle_cancellation(
        self,
        context: ExecutionContext,
        reason: Optional[str] = None,
    ) -> WorkflowResult:
        """Handle workflow cancellation."""
        context.status = ExecutionStatus.CANCELLED
        
        await self._emit_event(WorkflowEvent(
            type=WorkflowEventType.WORKFLOW_CANCELLED,
            execution_id=context.execution_id,
            workflow_id=context.workflow.id,
            data={"reason": reason or "User cancelled"},
        ))
        
        return self._build_result(
            context,
            context.started_at or datetime.now(timezone.utc),
            error={
                "message": "Workflow was cancelled",
                "type": "CancelledError",
                "reason": reason,
            },
        )
    
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
