"""
AI Workflow Fabric - Event System

This module provides event emission and subscription for workflow observability.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum
import json

from awf.orchestration.types import WorkflowEvent, WorkflowEventType


# =============================================================================
# Event Subscription
# =============================================================================


class EventFilter:
    """Filters events based on criteria."""
    
    def __init__(
        self,
        event_types: Optional[List[WorkflowEventType]] = None,
        execution_ids: Optional[Set[str]] = None,
        workflow_ids: Optional[Set[str]] = None,
        step_ids: Optional[Set[str]] = None,
    ):
        """
        Initialize event filter.
        
        Args:
            event_types: Filter by event types (None = all)
            execution_ids: Filter by execution IDs (None = all)
            workflow_ids: Filter by workflow IDs (None = all)
            step_ids: Filter by step IDs (None = all)
        """
        self.event_types = set(event_types) if event_types else None
        self.execution_ids = execution_ids
        self.workflow_ids = workflow_ids
        self.step_ids = step_ids
    
    def matches(self, event: WorkflowEvent) -> bool:
        """Check if event matches filter criteria."""
        if self.event_types and event.type not in self.event_types:
            return False
        if self.execution_ids and event.execution_id not in self.execution_ids:
            return False
        if self.workflow_ids and event.workflow_id not in self.workflow_ids:
            return False
        if self.step_ids and event.step_id and event.step_id not in self.step_ids:
            return False
        return True


@dataclass
class Subscription:
    """An event subscription."""
    
    id: str
    callback: Callable[[WorkflowEvent], Any]
    filter: Optional[EventFilter] = None
    async_callback: bool = False
    
    def should_receive(self, event: WorkflowEvent) -> bool:
        """Check if this subscription should receive the event."""
        if self.filter is None:
            return True
        return self.filter.matches(event)


# =============================================================================
# Event Emitter
# =============================================================================


class EventEmitter:
    """
    Central event hub for workflow events.
    
    Provides pub/sub pattern for workflow observability:
    - Multiple subscribers can listen for events
    - Filters allow targeted subscriptions
    - Supports both sync and async callbacks
    - Thread-safe event emission
    """
    
    def __init__(self):
        """Initialize the event emitter."""
        self._subscriptions: Dict[str, Subscription] = {}
        self._subscription_counter = 0
        self._lock = asyncio.Lock()
        self._event_queue: asyncio.Queue[WorkflowEvent] = asyncio.Queue()
        self._history: List[WorkflowEvent] = []
        self._history_limit = 1000
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
    
    def subscribe(
        self,
        callback: Callable[[WorkflowEvent], Any],
        filter: Optional[EventFilter] = None,
        async_callback: bool = False,
    ) -> str:
        """
        Subscribe to events.
        
        Args:
            callback: Function to call when event occurs
            filter: Optional filter for events
            async_callback: If True, callback is async
        
        Returns:
            Subscription ID for unsubscribing
        """
        self._subscription_counter += 1
        sub_id = f"sub_{self._subscription_counter}"
        
        self._subscriptions[sub_id] = Subscription(
            id=sub_id,
            callback=callback,
            filter=filter,
            async_callback=async_callback,
        )
        
        return sub_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            subscription_id: The subscription ID to remove
        
        Returns:
            True if subscription was found and removed
        """
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            return True
        return False
    
    async def emit(self, event: WorkflowEvent) -> None:
        """
        Emit an event to all matching subscribers.
        
        Args:
            event: The event to emit
        """
        # Store in history
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]
        
        # Notify subscribers
        for subscription in list(self._subscriptions.values()):
            if subscription.should_receive(event):
                try:
                    if subscription.async_callback:
                        result = subscription.callback(event)
                        if asyncio.iscoroutine(result):
                            await result
                    else:
                        subscription.callback(event)
                except Exception:
                    # Don't let subscriber errors affect emission
                    pass
    
    def emit_sync(self, event: WorkflowEvent) -> None:
        """
        Emit an event synchronously (for non-async contexts).
        
        Only calls synchronous callbacks.
        """
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]
        
        for subscription in list(self._subscriptions.values()):
            if not subscription.async_callback and subscription.should_receive(event):
                try:
                    subscription.callback(event)
                except Exception:
                    pass
    
    def get_history(
        self,
        execution_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[WorkflowEvent]:
        """
        Get event history.
        
        Args:
            execution_id: Optional filter by execution ID
            limit: Maximum events to return
        
        Returns:
            List of events (most recent first)
        """
        events = self._history
        
        if execution_id:
            events = [e for e in events if e.execution_id == execution_id]
        
        return list(reversed(events[-limit:]))
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()
    
    @property
    def subscriber_count(self) -> int:
        """Number of active subscribers."""
        return len(self._subscriptions)


# =============================================================================
# Event Serialization
# =============================================================================


def event_to_json(event: WorkflowEvent) -> str:
    """Serialize event to JSON string."""
    return json.dumps(event.to_dict())


def event_to_sse(event: WorkflowEvent) -> str:
    """Format event for Server-Sent Events."""
    data = json.dumps(event.to_dict())
    return f"event: {event.type.value}\ndata: {data}\n\n"


# =============================================================================
# Global Event Emitter
# =============================================================================


_global_emitter: Optional[EventEmitter] = None


def get_event_emitter() -> EventEmitter:
    """Get or create the global event emitter."""
    global _global_emitter
    if _global_emitter is None:
        _global_emitter = EventEmitter()
    return _global_emitter


def reset_event_emitter() -> None:
    """Reset the global event emitter (for testing)."""
    global _global_emitter
    _global_emitter = None
