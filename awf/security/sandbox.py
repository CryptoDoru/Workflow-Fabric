"""
AI Workflow Fabric - Sandbox Orchestrator

This module provides the main sandbox orchestrator that selects and manages
the appropriate sandbox tier for agent execution based on trust scores.
"""

from __future__ import annotations

import abc
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from awf.core.types import (
    AgentManifest,
    SandboxTier,
    Task,
    TaskError,
    TaskMetrics,
    TaskResult,
    TaskStatus,
    TrustScore,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Sandbox Types
# =============================================================================


class SandboxStatus(str, Enum):
    """Status of a sandbox instance."""
    INITIALIZING = "initializing"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""
    
    # Resource limits
    max_memory_bytes: int = 512 * 1024 * 1024  # 512MB
    max_cpu_time_ms: int = 60000  # 60 seconds
    max_execution_time_ms: int = 120000  # 2 minutes
    
    # Network settings
    allow_network: bool = False
    allowed_hosts: List[str] = field(default_factory=list)
    
    # Filesystem settings
    allow_filesystem: bool = False
    allowed_paths: List[str] = field(default_factory=list)
    read_only_paths: List[str] = field(default_factory=list)
    
    # Environment
    environment: Dict[str, str] = field(default_factory=dict)
    
    # Sandbox-specific options
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SandboxResult:
    """Result from sandbox execution."""
    
    success: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    
    # Metrics
    execution_time_ms: int = 0
    memory_usage_bytes: int = 0
    cpu_time_ms: int = 0
    
    # Sandbox info
    sandbox_tier: Optional[SandboxTier] = None
    sandbox_overhead_ms: int = 0


# =============================================================================
# Abstract Sandbox
# =============================================================================


class Sandbox(abc.ABC):
    """
    Abstract base class for sandbox implementations.
    
    Each sandbox tier (WASM, gVisor, etc.) implements this interface
    to provide isolated execution environments.
    """
    
    tier: SandboxTier
    
    @abc.abstractmethod
    async def initialize(self, config: SandboxConfig) -> None:
        """
        Initialize the sandbox with the given configuration.
        
        Args:
            config: Sandbox configuration
        """
        pass
    
    @abc.abstractmethod
    async def execute(
        self,
        code: str,
        input_data: Dict[str, Any],
        *,
        timeout_ms: Optional[int] = None,
    ) -> SandboxResult:
        """
        Execute code in the sandbox.
        
        Args:
            code: The code/agent to execute
            input_data: Input data for the execution
            timeout_ms: Optional execution timeout
        
        Returns:
            Execution result
        """
        pass
    
    @abc.abstractmethod
    async def terminate(self) -> None:
        """Terminate the sandbox and cleanup resources."""
        pass
    
    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if this sandbox type is available on the system."""
        pass
    
    @property
    @abc.abstractmethod
    def expected_overhead_ms(self) -> int:
        """Expected overhead in milliseconds for this sandbox tier."""
        pass


# =============================================================================
# Sandbox Pool
# =============================================================================


class SandboxPool:
    """
    Pool of pre-warmed sandbox instances for reduced latency.
    
    Maintains a pool of ready sandboxes for each tier to minimize
    cold-start overhead.
    """
    
    def __init__(
        self,
        *,
        wasm_pool_size: int = 10,
        gvisor_pool_size: int = 5,
        gvisor_strict_pool_size: int = 2,
    ):
        self._wasm_pool_size = wasm_pool_size
        self._gvisor_pool_size = gvisor_pool_size
        self._gvisor_strict_pool_size = gvisor_strict_pool_size
        
        self._pools: Dict[SandboxTier, asyncio.Queue[Sandbox]] = {}
        self._factories: Dict[SandboxTier, Callable[[], Sandbox]] = {}
        self._initialized = False
    
    def register_factory(
        self,
        tier: SandboxTier,
        factory: Callable[[], Sandbox],
    ) -> None:
        """
        Register a sandbox factory for a tier.
        
        Args:
            tier: The sandbox tier
            factory: Factory function that creates sandbox instances
        """
        self._factories[tier] = factory
    
    async def initialize(self) -> None:
        """Initialize the sandbox pools with pre-warmed instances."""
        if self._initialized:
            return
        
        for tier in [SandboxTier.WASM, SandboxTier.GVISOR, SandboxTier.GVISOR_STRICT]:
            if tier not in self._factories:
                continue
            
            pool_size = self._get_pool_size(tier)
            self._pools[tier] = asyncio.Queue(maxsize=pool_size)
            
            # Pre-warm the pool
            for _ in range(pool_size):
                try:
                    sandbox = self._factories[tier]()
                    await sandbox.initialize(SandboxConfig())
                    await self._pools[tier].put(sandbox)
                except Exception as e:
                    logger.warning(f"Failed to pre-warm {tier.value} sandbox: {e}")
        
        self._initialized = True
    
    async def acquire(self, tier: SandboxTier) -> Sandbox:
        """
        Acquire a sandbox from the pool.
        
        Args:
            tier: The sandbox tier to acquire
        
        Returns:
            A ready sandbox instance
        
        Raises:
            ValueError: If tier is not available
        """
        if tier not in self._pools:
            raise ValueError(f"No pool available for tier: {tier.value}")
        
        try:
            # Try to get from pool without blocking
            sandbox = self._pools[tier].get_nowait()
            return sandbox
        except asyncio.QueueEmpty:
            # Create new instance
            if tier not in self._factories:
                raise ValueError(f"No factory registered for tier: {tier.value}")
            
            sandbox = self._factories[tier]()
            await sandbox.initialize(SandboxConfig())
            return sandbox
    
    async def release(self, sandbox: Sandbox) -> None:
        """
        Return a sandbox to the pool.
        
        Args:
            sandbox: The sandbox to return
        """
        tier = sandbox.tier
        if tier not in self._pools:
            await sandbox.terminate()
            return
        
        try:
            self._pools[tier].put_nowait(sandbox)
        except asyncio.QueueFull:
            # Pool is full, terminate this instance
            await sandbox.terminate()
    
    async def shutdown(self) -> None:
        """Shutdown all pools and terminate all sandboxes."""
        for tier, pool in self._pools.items():
            while not pool.empty():
                try:
                    sandbox = pool.get_nowait()
                    await sandbox.terminate()
                except asyncio.QueueEmpty:
                    break
        
        self._pools.clear()
        self._initialized = False
    
    def _get_pool_size(self, tier: SandboxTier) -> int:
        """Get the pool size for a tier."""
        if tier == SandboxTier.WASM:
            return self._wasm_pool_size
        elif tier == SandboxTier.GVISOR:
            return self._gvisor_pool_size
        elif tier == SandboxTier.GVISOR_STRICT:
            return self._gvisor_strict_pool_size
        return 0


# =============================================================================
# Sandbox Orchestrator
# =============================================================================


class SandboxOrchestrator:
    """
    Main orchestrator for sandbox-based agent execution.
    
    This class is responsible for:
    1. Selecting the appropriate sandbox tier based on trust score
    2. Managing sandbox lifecycle (creation, execution, cleanup)
    3. Enforcing resource limits and security policies
    4. Collecting execution metrics
    
    Example usage:
        ```python
        orchestrator = SandboxOrchestrator()
        await orchestrator.initialize()
        
        # Execute a task in appropriate sandbox
        result = await orchestrator.execute(
            agent=manifest,
            task=task,
            trust_score=trust_score,
        )
        
        await orchestrator.shutdown()
        ```
    """
    
    def __init__(
        self,
        *,
        pool: Optional[SandboxPool] = None,
        default_config: Optional[SandboxConfig] = None,
    ):
        self._pool = pool or SandboxPool()
        self._default_config = default_config or SandboxConfig()
        self._sandbox_types: Dict[SandboxTier, Type[Sandbox]] = {}
        self._initialized = False
    
    def register_sandbox_type(
        self,
        tier: SandboxTier,
        sandbox_class: Type[Sandbox],
    ) -> None:
        """
        Register a sandbox implementation for a tier.
        
        Args:
            tier: The sandbox tier
            sandbox_class: The sandbox class to use
        """
        self._sandbox_types[tier] = sandbox_class
        self._pool.register_factory(tier, sandbox_class)
    
    async def initialize(self) -> None:
        """Initialize the orchestrator and sandbox pools."""
        if self._initialized:
            return
        
        await self._pool.initialize()
        self._initialized = True
        logger.info("Sandbox orchestrator initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the orchestrator and cleanup resources."""
        await self._pool.shutdown()
        self._initialized = False
        logger.info("Sandbox orchestrator shutdown")
    
    def select_tier(self, trust_score: TrustScore) -> SandboxTier:
        """
        Select the appropriate sandbox tier based on trust score.
        
        Args:
            trust_score: The agent's trust score
        
        Returns:
            The selected sandbox tier
        """
        return trust_score.sandbox_tier
    
    async def execute(
        self,
        agent: AgentManifest,
        task: Task,
        trust_score: TrustScore,
        *,
        config: Optional[SandboxConfig] = None,
    ) -> TaskResult:
        """
        Execute a task in an appropriate sandbox.
        
        Args:
            agent: The agent manifest
            task: The task to execute
            trust_score: The agent's trust score
            config: Optional sandbox configuration override
        
        Returns:
            Task execution result
        """
        started_at = datetime.utcnow()
        
        # Select sandbox tier
        tier = self.select_tier(trust_score)
        
        # Check if execution is blocked
        if tier == SandboxTier.BLOCKED:
            return TaskResult(
                task_id=task.id,
                agent_id=agent.id,
                status=TaskStatus.FAILED,
                error=TaskError(
                    code="EXECUTION_BLOCKED",
                    message=f"Agent execution blocked due to low trust score: {trust_score.score:.2f}",
                    details={"trust_score": trust_score.score},
                    retryable=False,
                ),
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )
        
        # Check if tier is available
        if tier not in self._sandbox_types:
            logger.warning(f"Sandbox tier {tier.value} not available, falling back")
            tier = self._get_fallback_tier(tier)
            
            if tier is None or tier == SandboxTier.BLOCKED:
                return TaskResult(
                    task_id=task.id,
                    agent_id=agent.id,
                    status=TaskStatus.FAILED,
                    error=TaskError(
                        code="NO_SANDBOX_AVAILABLE",
                        message="No suitable sandbox tier available for execution",
                        retryable=True,
                    ),
                    started_at=started_at,
                    completed_at=datetime.utcnow(),
                )
        
        # Acquire sandbox
        sandbox: Optional[Sandbox] = None
        try:
            sandbox = await self._pool.acquire(tier)
        except ValueError as e:
            return TaskResult(
                task_id=task.id,
                agent_id=agent.id,
                status=TaskStatus.FAILED,
                error=TaskError(
                    code="SANDBOX_ACQUISITION_FAILED",
                    message=str(e),
                    retryable=True,
                ),
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )
        
        try:
            # Merge configuration
            exec_config = self._merge_config(config)
            
            # Execute in sandbox
            timeout = task.timeout_ms or exec_config.max_execution_time_ms
            
            result = await sandbox.execute(
                code=agent.id,  # Agent ID serves as reference
                input_data=task.input,
                timeout_ms=timeout,
            )
            
            completed_at = datetime.utcnow()
            
            # Build metrics
            metrics = TaskMetrics(
                execution_time_ms=result.execution_time_ms,
                memory_usage_bytes=result.memory_usage_bytes,
                cpu_time_ms=result.cpu_time_ms,
                sandbox_tier=tier,
                sandbox_overhead_ms=result.sandbox_overhead_ms,
            )
            
            if result.success:
                return TaskResult(
                    task_id=task.id,
                    agent_id=agent.id,
                    status=TaskStatus.COMPLETED,
                    output=result.output,
                    metrics=metrics,
                    trace_id=task.trace_id,
                    started_at=started_at,
                    completed_at=completed_at,
                )
            else:
                return TaskResult(
                    task_id=task.id,
                    agent_id=agent.id,
                    status=TaskStatus.FAILED,
                    error=TaskError(
                        code=result.error_code or "EXECUTION_FAILED",
                        message=result.error or "Unknown error",
                        retryable=True,
                    ),
                    metrics=metrics,
                    trace_id=task.trace_id,
                    started_at=started_at,
                    completed_at=completed_at,
                )
        
        except asyncio.TimeoutError:
            return TaskResult(
                task_id=task.id,
                agent_id=agent.id,
                status=TaskStatus.TIMEOUT,
                error=TaskError(
                    code="EXECUTION_TIMEOUT",
                    message=f"Task execution timed out after {task.timeout_ms}ms",
                    retryable=True,
                ),
                trace_id=task.trace_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )
        
        except Exception as e:
            logger.exception(f"Sandbox execution error: {e}")
            return TaskResult(
                task_id=task.id,
                agent_id=agent.id,
                status=TaskStatus.FAILED,
                error=TaskError(
                    code="SANDBOX_ERROR",
                    message=str(e),
                    retryable=True,
                ),
                trace_id=task.trace_id,
                started_at=started_at,
                completed_at=datetime.utcnow(),
            )
        
        finally:
            # Return sandbox to pool
            if sandbox:
                await self._pool.release(sandbox)
    
    def _get_fallback_tier(self, tier: SandboxTier) -> Optional[SandboxTier]:
        """Get fallback tier if preferred is unavailable."""
        # Fallback chain: WASM -> gVisor -> gVisor Strict -> None
        fallback_order = [
            SandboxTier.WASM,
            SandboxTier.GVISOR,
            SandboxTier.GVISOR_STRICT,
        ]
        
        try:
            current_idx = fallback_order.index(tier)
        except ValueError:
            return None
        
        # Try more restrictive tiers
        for fallback_tier in fallback_order[current_idx:]:
            if fallback_tier in self._sandbox_types:
                return fallback_tier
        
        return None
    
    def _merge_config(self, override: Optional[SandboxConfig]) -> SandboxConfig:
        """Merge override config with default config."""
        if override is None:
            return self._default_config
        
        # Use override values if provided, otherwise default
        return SandboxConfig(
            max_memory_bytes=override.max_memory_bytes or self._default_config.max_memory_bytes,
            max_cpu_time_ms=override.max_cpu_time_ms or self._default_config.max_cpu_time_ms,
            max_execution_time_ms=override.max_execution_time_ms or self._default_config.max_execution_time_ms,
            allow_network=override.allow_network,
            allowed_hosts=override.allowed_hosts or self._default_config.allowed_hosts,
            allow_filesystem=override.allow_filesystem,
            allowed_paths=override.allowed_paths or self._default_config.allowed_paths,
            read_only_paths=override.read_only_paths or self._default_config.read_only_paths,
            environment={**self._default_config.environment, **override.environment},
            options={**self._default_config.options, **override.options},
        )
    
    def get_available_tiers(self) -> List[SandboxTier]:
        """Get list of available sandbox tiers."""
        return list(self._sandbox_types.keys())
    
    def is_tier_available(self, tier: SandboxTier) -> bool:
        """Check if a specific tier is available."""
        return tier in self._sandbox_types


# =============================================================================
# Stub Sandbox (for testing)
# =============================================================================


class StubSandbox(Sandbox):
    """
    Stub sandbox implementation for testing.
    
    This sandbox executes tasks directly without isolation,
    useful for development and testing purposes only.
    
    WARNING: Do not use in production!
    """
    
    tier = SandboxTier.WASM  # Pretend to be WASM tier
    
    def __init__(self):
        self._config: Optional[SandboxConfig] = None
        self._status = SandboxStatus.INITIALIZING
    
    async def initialize(self, config: SandboxConfig) -> None:
        """Initialize the stub sandbox."""
        self._config = config
        self._status = SandboxStatus.READY
    
    async def execute(
        self,
        code: str,
        input_data: Dict[str, Any],
        *,
        timeout_ms: Optional[int] = None,
    ) -> SandboxResult:
        """Execute in stub sandbox (no real isolation)."""
        import time
        start = time.perf_counter()
        
        self._status = SandboxStatus.EXECUTING
        
        try:
            # Simulate some execution time
            await asyncio.sleep(0.001)  # 1ms
            
            # Return input as output (echo behavior)
            output = {"echo": input_data, "agent": code}
            
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            
            self._status = SandboxStatus.COMPLETED
            
            return SandboxResult(
                success=True,
                output=output,
                execution_time_ms=elapsed_ms,
                memory_usage_bytes=1024 * 1024,  # Fake 1MB usage
                sandbox_tier=self.tier,
                sandbox_overhead_ms=1,
            )
        
        except Exception as e:
            self._status = SandboxStatus.FAILED
            return SandboxResult(
                success=False,
                error=str(e),
                error_code="STUB_ERROR",
            )
    
    async def terminate(self) -> None:
        """Terminate the stub sandbox."""
        self._status = SandboxStatus.TERMINATED
    
    def is_available(self) -> bool:
        """Stub is always available."""
        return True
    
    @property
    def expected_overhead_ms(self) -> int:
        """Stub has minimal overhead."""
        return 1
