"""
AI Workflow Fabric - Orchestration Engine

This module provides workflow orchestration capabilities for executing
multi-agent workflows with reliability features like retry, fallback,
and event streaming.

Example usage:

    from awf.orchestration import (
        Orchestrator,
        OrchestratorConfig,
        WorkflowDefinition,
        StepDefinition,
    )
    
    # Define a workflow
    workflow = WorkflowDefinition(
        id="research-workflow",
        name="Research and Write",
        steps=[
            StepDefinition(
                id="research",
                agent_id="research-agent",
                input_map={"query": "$.input.topic"},
            ),
            StepDefinition(
                id="write",
                agent_id="writer-agent",
                input_map={"research": "$.steps.research.output"},
                depends_on=["research"],
            ),
        ],
    )
    
    # Create orchestrator
    orchestrator = Orchestrator(
        adapter_registry=adapter_registry,
        agent_registry=agent_registry,
    )
    
    # Execute workflow
    result = await orchestrator.execute(
        workflow=workflow,
        input_data={"topic": "AI safety"},
    )
"""

from awf.orchestration.types import (
    # Enums
    ExecutionStatus,
    StepStatus,
    ErrorCategory,
    WorkflowEventType,
    # Data classes
    RetryPolicy,
    FallbackPolicy,
    StepDefinition,
    WorkflowDefinition,
    StepResult,
    WorkflowResult,
    WorkflowEvent,
    ExecutionContext,
)

from awf.orchestration.errors import (
    # Base exceptions
    OrchestrationError,
    # Workflow definition errors
    WorkflowDefinitionError,
    CyclicDependencyError,
    InvalidStepReferenceError,
    DuplicateStepIdError,
    InvalidInputMapError,
    # Execution errors
    ExecutionError,
    StepExecutionError,
    AgentNotFoundError,
    AgentInactiveError,
    StepTimeoutError,
    WorkflowTimeoutError,
    WorkflowCancelledError,
    ConditionEvaluationError,
    InputMappingError,
    DependencyFailedError,
    # Fallback errors
    FallbackError,
    NoFallbackAvailableError,
    # Registry errors
    RegistryError,
    AdapterNotFoundError,
    # Utilities
    categorize_exception,
)

from awf.orchestration.mapping import (
    JSONPathError,
    InputMapper,
    OutputMapper,
    ExpressionValidator,
)

from awf.orchestration.dag import (
    DAGNode,
    WorkflowDAG,
    build_dag,
    get_execution_order,
    get_parallel_execution_plan,
    detect_cycle,
    get_affected_steps,
)

from awf.orchestration.validation import (
    ValidationError,
    ValidationResult,
    WorkflowValidator,
    validate_workflow,
    is_valid_workflow,
)

from awf.orchestration.executor import (
    StepExecutorConfig,
    StepExecutor,
    AdapterProtocol,
    AdapterRegistry,
    AgentRegistryProtocol,
    EventCallback,
    AsyncEventCallback,
)

from awf.orchestration.orchestrator import (
    OrchestratorConfig,
    ConditionEvaluator,
    Orchestrator,
)

from awf.orchestration.registry import (
    AdapterEntry,
    OrchestrationAdapterRegistry,
    create_adapter_registry,
    auto_discover_adapters,
)

from awf.orchestration.events import (
    EventFilter,
    Subscription,
    EventEmitter,
)

from awf.orchestration.state import (
    StateManager,
    InMemoryStateManager,
    SQLiteStateManager,
)


__all__ = [
    # Types - Enums
    "ExecutionStatus",
    "StepStatus",
    "ErrorCategory",
    "WorkflowEventType",
    # Types - Data classes
    "RetryPolicy",
    "FallbackPolicy",
    "StepDefinition",
    "WorkflowDefinition",
    "StepResult",
    "WorkflowResult",
    "WorkflowEvent",
    "ExecutionContext",
    # Errors
    "OrchestrationError",
    "WorkflowDefinitionError",
    "CyclicDependencyError",
    "InvalidStepReferenceError",
    "DuplicateStepIdError",
    "InvalidInputMapError",
    "ExecutionError",
    "StepExecutionError",
    "AgentNotFoundError",
    "AgentInactiveError",
    "StepTimeoutError",
    "WorkflowTimeoutError",
    "WorkflowCancelledError",
    "ConditionEvaluationError",
    "InputMappingError",
    "DependencyFailedError",
    "FallbackError",
    "NoFallbackAvailableError",
    "RegistryError",
    "AdapterNotFoundError",
    "categorize_exception",
    # Mapping
    "JSONPathError",
    "InputMapper",
    "OutputMapper",
    "ExpressionValidator",
    # DAG
    "DAGNode",
    "WorkflowDAG",
    "build_dag",
    "get_execution_order",
    "get_parallel_execution_plan",
    "detect_cycle",
    "get_affected_steps",
    # Validation
    "ValidationError",
    "ValidationResult",
    "WorkflowValidator",
    "validate_workflow",
    "is_valid_workflow",
    # Executor
    "StepExecutorConfig",
    "StepExecutor",
    "AdapterProtocol",
    "AdapterRegistry",
    "AgentRegistryProtocol",
    "EventCallback",
    "AsyncEventCallback",
    # Orchestrator
    "OrchestratorConfig",
    "ConditionEvaluator",
    "Orchestrator",
    # Registry
    "AdapterEntry",
    "OrchestrationAdapterRegistry",
    "create_adapter_registry",
    "auto_discover_adapters",
    # Events
    "EventFilter",
    "Subscription",
    "EventEmitter",
    # State
    "StateManager",
    "InMemoryStateManager",
    "SQLiteStateManager",
]
