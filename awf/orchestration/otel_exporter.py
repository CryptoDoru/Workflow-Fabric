"""
AI Workflow Fabric - OpenTelemetry Exporter

This module bridges AWF's EventEmitter to OpenTelemetry, enabling
integration with Grafana's LGTM stack (Loki, Grafana, Tempo, Mimir).

Usage:
    from awf.orchestration.otel_exporter import OTelExporter, setup_otel_exporter
    
    # Quick setup (connects to default Alloy endpoint)
    exporter = setup_otel_exporter()
    
    # Or manual setup with custom endpoint
    exporter = OTelExporter(endpoint="http://alloy:4317")
    exporter.start()
"""

from __future__ import annotations

import atexit
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional

from awf.orchestration.events import EventEmitter, get_event_emitter
from awf.orchestration.types import (
    ExecutionStatus,
    StepStatus,
    WorkflowEvent,
    WorkflowEventType,
)

# OpenTelemetry imports - these are optional dependencies
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.trace import Status, StatusCode, SpanKind
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Stub types for when OTel is not installed
    TracerProvider = None  # type: ignore
    MeterProvider = None  # type: ignore

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class OTelConfig:
    """Configuration for OpenTelemetry exporter."""
    
    # OTLP endpoint (Grafana Alloy by default)
    endpoint: str = "http://localhost:4317"
    
    # Service identification
    service_name: str = "awf-orchestration"
    service_version: str = "1.0.0"
    deployment_environment: str = "development"
    
    # Export settings
    export_traces: bool = True
    export_metrics: bool = True
    
    # Metric export interval (seconds)
    metric_export_interval_ms: int = 5000
    
    # Enable/disable specific metric types
    enable_step_duration_histogram: bool = True
    enable_token_counter: bool = True
    enable_cost_counter: bool = True
    enable_workflow_counter: bool = True
    enable_retry_counter: bool = True
    
    # Histogram boundaries for step duration (milliseconds)
    step_duration_boundaries: tuple = (
        10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000
    )
    
    def to_resource(self) -> Any:
        """Create OTel Resource from config."""
        if not OTEL_AVAILABLE:
            return None
        return Resource.create({
            SERVICE_NAME: self.service_name,
            SERVICE_VERSION: self.service_version,
            "deployment.environment": self.deployment_environment,
            "service.namespace": "awf",
        })


# =============================================================================
# Span Context Manager
# =============================================================================


@dataclass
class SpanContext:
    """Tracks active spans for correlation."""
    
    # Active workflow spans: execution_id -> span
    workflow_spans: Dict[str, Any] = field(default_factory=dict)
    
    # Active step spans: (execution_id, step_id) -> span
    step_spans: Dict[tuple, Any] = field(default_factory=dict)
    
    def get_workflow_span(self, execution_id: str) -> Optional[Any]:
        """Get active workflow span."""
        return self.workflow_spans.get(execution_id)
    
    def set_workflow_span(self, execution_id: str, span: Any) -> None:
        """Store workflow span."""
        self.workflow_spans[execution_id] = span
    
    def remove_workflow_span(self, execution_id: str) -> Optional[Any]:
        """Remove and return workflow span."""
        return self.workflow_spans.pop(execution_id, None)
    
    def get_step_span(self, execution_id: str, step_id: str) -> Optional[Any]:
        """Get active step span."""
        return self.step_spans.get((execution_id, step_id))
    
    def set_step_span(self, execution_id: str, step_id: str, span: Any) -> None:
        """Store step span."""
        self.step_spans[(execution_id, step_id)] = span
    
    def remove_step_span(self, execution_id: str, step_id: str) -> Optional[Any]:
        """Remove and return step span."""
        return self.step_spans.pop((execution_id, step_id), None)


# =============================================================================
# OpenTelemetry Exporter
# =============================================================================


class OTelExporter:
    """
    Bridges AWF EventEmitter to OpenTelemetry.
    
    Subscribes to workflow events and exports:
    - Traces: Distributed tracing for workflows and steps
    - Metrics: Counters and histograms for observability
    
    Works with any OTLP-compatible backend (Grafana Alloy, Jaeger, etc.)
    """
    
    def __init__(self, config: Optional[OTelConfig] = None):
        """
        Initialize the OTel exporter.
        
        Args:
            config: Optional configuration (uses defaults if not provided)
        """
        if not OTEL_AVAILABLE:
            logger.warning(
                "OpenTelemetry packages not installed. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk "
                "opentelemetry-exporter-otlp-proto-grpc"
            )
            self._enabled = False
            return
        
        self._config = config or OTelConfig()
        self._enabled = True
        self._started = False
        self._subscription_id: Optional[str] = None
        self._span_context = SpanContext()
        
        # OTel components (initialized in start())
        self._tracer: Optional[Any] = None
        self._meter: Optional[Any] = None
        self._tracer_provider: Optional[Any] = None
        self._meter_provider: Optional[Any] = None
        
        # Metrics instruments (initialized in _setup_metrics())
        self._workflow_counter: Optional[Any] = None
        self._step_duration_histogram: Optional[Any] = None
        self._token_counter: Optional[Any] = None
        self._cost_counter: Optional[Any] = None
        self._retry_counter: Optional[Any] = None
        self._fallback_counter: Optional[Any] = None
        self._active_workflows_gauge: Optional[Any] = None
    
    @property
    def enabled(self) -> bool:
        """Check if exporter is enabled."""
        return self._enabled
    
    @property
    def started(self) -> bool:
        """Check if exporter is started."""
        return self._started
    
    def start(self, emitter: Optional[EventEmitter] = None) -> None:
        """
        Start the exporter and subscribe to events.
        
        Args:
            emitter: EventEmitter to subscribe to (uses global if not provided)
        """
        if not self._enabled:
            logger.warning("OTelExporter is disabled (OpenTelemetry not available)")
            return
        
        if self._started:
            logger.warning("OTelExporter already started")
            return
        
        # Setup OTel SDK
        self._setup_tracing()
        self._setup_metrics()
        
        # Subscribe to events
        target_emitter = emitter or get_event_emitter()
        self._subscription_id = target_emitter.subscribe(
            callback=self._on_event,
            async_callback=True,
        )
        
        self._started = True
        logger.info(
            f"OTelExporter started, exporting to {self._config.endpoint}"
        )
    
    def stop(self) -> None:
        """Stop the exporter and unsubscribe from events."""
        if not self._started:
            return
        
        # Unsubscribe from events
        if self._subscription_id:
            get_event_emitter().unsubscribe(self._subscription_id)
            self._subscription_id = None
        
        # Flush and shutdown providers
        if self._tracer_provider:
            self._tracer_provider.force_flush()
            self._tracer_provider.shutdown()
        
        if self._meter_provider:
            self._meter_provider.force_flush()
            self._meter_provider.shutdown()
        
        self._started = False
        logger.info("OTelExporter stopped")
    
    def _setup_tracing(self) -> None:
        """Initialize OpenTelemetry tracing."""
        if not self._config.export_traces:
            return
        
        resource = self._config.to_resource()
        
        # Create exporter
        exporter = OTLPSpanExporter(
            endpoint=self._config.endpoint,
            insecure=True,  # For local development
        )
        
        # Create and set provider
        self._tracer_provider = TracerProvider(resource=resource)
        self._tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(self._tracer_provider)
        
        # Get tracer
        self._tracer = trace.get_tracer(
            self._config.service_name,
            self._config.service_version,
        )
    
    def _setup_metrics(self) -> None:
        """Initialize OpenTelemetry metrics."""
        if not self._config.export_metrics:
            return
        
        resource = self._config.to_resource()
        
        # Create exporter
        exporter = OTLPMetricExporter(
            endpoint=self._config.endpoint,
            insecure=True,
        )
        
        # Create reader with export interval
        reader = PeriodicExportingMetricReader(
            exporter,
            export_interval_millis=self._config.metric_export_interval_ms,
        )
        
        # Create and set provider
        self._meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[reader],
        )
        metrics.set_meter_provider(self._meter_provider)
        
        # Get meter
        self._meter = metrics.get_meter(
            self._config.service_name,
            self._config.service_version,
        )
        
        # Create instruments
        self._create_metrics_instruments()
    
    def _create_metrics_instruments(self) -> None:
        """Create metrics instruments."""
        if not self._meter:
            return
        
        # Workflow counter
        if self._config.enable_workflow_counter:
            self._workflow_counter = self._meter.create_counter(
                name="awf_workflow_total",
                description="Total number of workflow executions",
                unit="1",
            )
        
        # Step duration histogram
        if self._config.enable_step_duration_histogram:
            self._step_duration_histogram = self._meter.create_histogram(
                name="awf_step_duration_ms",
                description="Step execution duration in milliseconds",
                unit="ms",
            )
        
        # Token counter
        if self._config.enable_token_counter:
            self._token_counter = self._meter.create_counter(
                name="awf_tokens_total",
                description="Total tokens used across all steps",
                unit="1",
            )
        
        # Cost counter
        if self._config.enable_cost_counter:
            self._cost_counter = self._meter.create_counter(
                name="awf_cost_usd_total",
                description="Total cost in USD",
                unit="USD",
            )
        
        # Retry counter
        if self._config.enable_retry_counter:
            self._retry_counter = self._meter.create_counter(
                name="awf_retry_total",
                description="Total retry attempts",
                unit="1",
            )
            self._fallback_counter = self._meter.create_counter(
                name="awf_fallback_total",
                description="Total fallback executions",
                unit="1",
            )
        
        # Active workflows gauge
        self._active_workflows_gauge = self._meter.create_up_down_counter(
            name="awf_workflow_active",
            description="Number of currently active workflow executions",
            unit="1",
        )
    
    async def _on_event(self, event: WorkflowEvent) -> None:
        """
        Handle workflow events.
        
        Routes events to appropriate handlers for tracing and metrics.
        """
        try:
            handlers = {
                WorkflowEventType.WORKFLOW_STARTED: self._on_workflow_started,
                WorkflowEventType.WORKFLOW_COMPLETED: self._on_workflow_completed,
                WorkflowEventType.WORKFLOW_FAILED: self._on_workflow_failed,
                WorkflowEventType.WORKFLOW_CANCELLED: self._on_workflow_cancelled,
                WorkflowEventType.STEP_STARTED: self._on_step_started,
                WorkflowEventType.STEP_COMPLETED: self._on_step_completed,
                WorkflowEventType.STEP_FAILED: self._on_step_failed,
                WorkflowEventType.STEP_RETRYING: self._on_step_retrying,
                WorkflowEventType.STEP_SKIPPED: self._on_step_skipped,
                WorkflowEventType.STEP_TIMEOUT: self._on_step_timeout,
                WorkflowEventType.STEP_FALLBACK: self._on_step_fallback,
            }
            
            handler = handlers.get(event.type)
            if handler:
                handler(event)
        except Exception as e:
            logger.error(f"Error handling event {event.type}: {e}")
    
    # =========================================================================
    # Workflow Event Handlers
    # =========================================================================
    
    def _on_workflow_started(self, event: WorkflowEvent) -> None:
        """Handle workflow started event."""
        # Start trace span
        if self._tracer:
            span = self._tracer.start_span(
                name=f"workflow.{event.workflow_id}",
                kind=SpanKind.INTERNAL,
                attributes={
                    "awf.execution_id": event.execution_id,
                    "awf.workflow_id": event.workflow_id,
                    "awf.event_type": event.type.value,
                },
            )
            self._span_context.set_workflow_span(event.execution_id, span)
        
        # Update active workflows gauge
        if self._active_workflows_gauge:
            self._active_workflows_gauge.add(
                1,
                {"workflow_id": event.workflow_id},
            )
    
    def _on_workflow_completed(self, event: WorkflowEvent) -> None:
        """Handle workflow completed event."""
        self._end_workflow_span(event, StatusCode.OK)
        self._record_workflow_metric(event, "completed")
    
    def _on_workflow_failed(self, event: WorkflowEvent) -> None:
        """Handle workflow failed event."""
        self._end_workflow_span(event, StatusCode.ERROR)
        self._record_workflow_metric(event, "failed")
    
    def _on_workflow_cancelled(self, event: WorkflowEvent) -> None:
        """Handle workflow cancelled event."""
        self._end_workflow_span(event, StatusCode.ERROR, "cancelled")
        self._record_workflow_metric(event, "cancelled")
    
    def _end_workflow_span(
        self,
        event: WorkflowEvent,
        status_code: Any,
        description: Optional[str] = None,
    ) -> None:
        """End workflow span and record final metrics."""
        span = self._span_context.remove_workflow_span(event.execution_id)
        if span:
            if OTEL_AVAILABLE:
                span.set_status(Status(status_code, description))
            
            # Add final attributes from event data
            if event.data:
                if "total_execution_time_ms" in event.data:
                    span.set_attribute(
                        "awf.total_execution_time_ms",
                        event.data["total_execution_time_ms"],
                    )
                if "total_token_usage" in event.data:
                    usage = event.data["total_token_usage"]
                    span.set_attribute("awf.total_input_tokens", usage.get("input_tokens", 0))
                    span.set_attribute("awf.total_output_tokens", usage.get("output_tokens", 0))
            
            span.end()
        
        # Update active workflows gauge
        if self._active_workflows_gauge:
            self._active_workflows_gauge.add(
                -1,
                {"workflow_id": event.workflow_id},
            )
    
    def _record_workflow_metric(self, event: WorkflowEvent, status: str) -> None:
        """Record workflow completion metric."""
        if self._workflow_counter:
            self._workflow_counter.add(
                1,
                {
                    "workflow_id": event.workflow_id,
                    "status": status,
                },
            )
    
    # =========================================================================
    # Step Event Handlers
    # =========================================================================
    
    def _on_step_started(self, event: WorkflowEvent) -> None:
        """Handle step started event."""
        if not self._tracer or not event.step_id:
            return
        
        # Get parent workflow span
        parent_span = self._span_context.get_workflow_span(event.execution_id)
        parent_context = trace.set_span_in_context(parent_span) if parent_span else None
        
        # Start step span
        span = self._tracer.start_span(
            name=f"step.{event.step_id}",
            kind=SpanKind.INTERNAL,
            context=parent_context,
            attributes={
                "awf.execution_id": event.execution_id,
                "awf.workflow_id": event.workflow_id,
                "awf.step_id": event.step_id,
                "awf.agent_id": event.data.get("agent_id", "unknown"),
            },
        )
        self._span_context.set_step_span(event.execution_id, event.step_id, span)
    
    def _on_step_completed(self, event: WorkflowEvent) -> None:
        """Handle step completed event."""
        if not event.step_id:
            return
        
        self._end_step_span(event, StatusCode.OK if OTEL_AVAILABLE else None)
        self._record_step_metrics(event, "completed")
    
    def _on_step_failed(self, event: WorkflowEvent) -> None:
        """Handle step failed event."""
        if not event.step_id:
            return
        
        self._end_step_span(event, StatusCode.ERROR if OTEL_AVAILABLE else None)
        self._record_step_metrics(event, "failed")
    
    def _on_step_retrying(self, event: WorkflowEvent) -> None:
        """Handle step retrying event."""
        if self._retry_counter and event.step_id:
            self._retry_counter.add(
                1,
                {
                    "step_id": event.step_id,
                    "agent_id": event.data.get("agent_id", "unknown"),
                    "attempt": str(event.data.get("attempt", 1)),
                },
            )
        
        # Add event to span
        span = self._span_context.get_step_span(
            event.execution_id,
            event.step_id,
        ) if event.step_id else None
        if span:
            span.add_event(
                "retry",
                {
                    "attempt": event.data.get("attempt", 1),
                    "delay_ms": event.data.get("delay_ms", 0),
                },
            )
    
    def _on_step_skipped(self, event: WorkflowEvent) -> None:
        """Handle step skipped event."""
        if not event.step_id:
            return
        
        self._end_step_span(event, StatusCode.OK if OTEL_AVAILABLE else None, "skipped")
        self._record_step_metrics(event, "skipped")
    
    def _on_step_timeout(self, event: WorkflowEvent) -> None:
        """Handle step timeout event."""
        if not event.step_id:
            return
        
        self._end_step_span(event, StatusCode.ERROR if OTEL_AVAILABLE else None, "timeout")
        self._record_step_metrics(event, "timeout")
    
    def _on_step_fallback(self, event: WorkflowEvent) -> None:
        """Handle step fallback event."""
        if self._fallback_counter and event.step_id:
            self._fallback_counter.add(
                1,
                {
                    "step_id": event.step_id,
                    "original_agent_id": event.data.get("original_agent_id", "unknown"),
                    "fallback_agent_id": event.data.get("fallback_agent_id", "unknown"),
                },
            )
        
        # Add event to span
        span = self._span_context.get_step_span(
            event.execution_id,
            event.step_id,
        ) if event.step_id else None
        if span:
            span.add_event(
                "fallback",
                {
                    "original_agent_id": event.data.get("original_agent_id", "unknown"),
                    "fallback_agent_id": event.data.get("fallback_agent_id", "unknown"),
                },
            )
    
    def _end_step_span(
        self,
        event: WorkflowEvent,
        status_code: Optional[Any],
        description: Optional[str] = None,
    ) -> None:
        """End step span."""
        if not event.step_id:
            return
        
        span = self._span_context.remove_step_span(event.execution_id, event.step_id)
        if span:
            if status_code is not None and OTEL_AVAILABLE:
                span.set_status(Status(status_code, description))
            
            # Add attributes from event data
            if event.data:
                if "execution_time_ms" in event.data:
                    span.set_attribute(
                        "awf.execution_time_ms",
                        event.data["execution_time_ms"],
                    )
                if "token_usage" in event.data:
                    usage = event.data["token_usage"]
                    span.set_attribute("awf.input_tokens", usage.get("input_tokens", 0))
                    span.set_attribute("awf.output_tokens", usage.get("output_tokens", 0))
                if "error" in event.data:
                    span.set_attribute("awf.error", str(event.data["error"]))
                if "error_category" in event.data:
                    span.set_attribute("awf.error_category", event.data["error_category"])
            
            span.end()
    
    def _record_step_metrics(self, event: WorkflowEvent, status: str) -> None:
        """Record step metrics."""
        if not event.step_id:
            return
        
        attributes = {
            "step_id": event.step_id,
            "agent_id": event.data.get("agent_id", "unknown"),
            "status": status,
        }
        
        # Step duration histogram
        if self._step_duration_histogram and "execution_time_ms" in event.data:
            self._step_duration_histogram.record(
                event.data["execution_time_ms"],
                attributes,
            )
        
        # Token usage
        if self._token_counter and "token_usage" in event.data:
            usage = event.data["token_usage"]
            if "input_tokens" in usage:
                self._token_counter.add(
                    usage["input_tokens"],
                    {**attributes, "direction": "input"},
                )
            if "output_tokens" in usage:
                self._token_counter.add(
                    usage["output_tokens"],
                    {**attributes, "direction": "output"},
                )
        
        # Cost
        if self._cost_counter and "cost_usd" in event.data:
            self._cost_counter.add(
                event.data["cost_usd"],
                {
                    **attributes,
                    "provider": event.data.get("provider", "unknown"),
                },
            )


# =============================================================================
# Convenience Functions
# =============================================================================


_global_exporter: Optional[OTelExporter] = None


def setup_otel_exporter(
    endpoint: str = "http://localhost:4317",
    service_name: str = "awf-orchestration",
    **kwargs: Any,
) -> OTelExporter:
    """
    Setup and start the global OTel exporter.
    
    Args:
        endpoint: OTLP endpoint (Grafana Alloy)
        service_name: Service name for identification
        **kwargs: Additional config options
    
    Returns:
        The configured and started exporter
    
    Example:
        exporter = setup_otel_exporter(
            endpoint="http://alloy:4317",
            deployment_environment="production",
        )
    """
    global _global_exporter
    
    if _global_exporter and _global_exporter.started:
        logger.warning("OTel exporter already setup, returning existing instance")
        return _global_exporter
    
    config = OTelConfig(
        endpoint=endpoint,
        service_name=service_name,
        **kwargs,
    )
    
    _global_exporter = OTelExporter(config)
    _global_exporter.start()
    
    # Register shutdown handler
    atexit.register(_global_exporter.stop)
    
    return _global_exporter


def get_otel_exporter() -> Optional[OTelExporter]:
    """Get the global OTel exporter if configured."""
    return _global_exporter


def shutdown_otel_exporter() -> None:
    """Shutdown the global OTel exporter."""
    global _global_exporter
    if _global_exporter:
        _global_exporter.stop()
        _global_exporter = None


@contextmanager
def otel_context(
    endpoint: str = "http://localhost:4317",
    **kwargs: Any,
) -> Iterator[OTelExporter]:
    """
    Context manager for OTel exporter.
    
    Useful for testing or temporary observability.
    
    Example:
        with otel_context() as exporter:
            # Run workflows, traces will be exported
            await orchestrator.execute(workflow, input)
    """
    exporter = OTelExporter(OTelConfig(endpoint=endpoint, **kwargs))
    exporter.start()
    try:
        yield exporter
    finally:
        exporter.stop()
