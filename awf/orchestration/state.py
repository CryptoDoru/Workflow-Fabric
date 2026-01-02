"""
AI Workflow Fabric - State Management

This module provides state persistence and recovery for workflow executions.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

from awf.orchestration.types import (
    ExecutionContext,
    ExecutionStatus,
    StepResult,
    StepStatus,
    WorkflowDefinition,
    WorkflowResult,
)


# =============================================================================
# Checkpoint Types
# =============================================================================


@dataclass
class ExecutionCheckpoint:
    """A checkpoint of workflow execution state."""
    
    execution_id: str
    workflow_id: str
    status: ExecutionStatus
    input_data: Dict[str, Any]
    context_data: Dict[str, Any]
    step_results: Dict[str, Dict[str, Any]]
    current_step: Optional[str]
    created_at: datetime
    updated_at: datetime
    workflow_json: str  # Serialized workflow definition
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "executionId": self.execution_id,
            "workflowId": self.workflow_id,
            "status": self.status.value,
            "inputData": self.input_data,
            "contextData": self.context_data,
            "stepResults": self.step_results,
            "currentStep": self.current_step,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat(),
            "workflowJson": self.workflow_json,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_context(cls, context: ExecutionContext) -> ExecutionCheckpoint:
        """Create checkpoint from execution context."""
        step_results = {}
        for step_id, result in context.step_results.items():
            step_results[step_id] = result.to_dict()
        
        now = datetime.now(timezone.utc)
        return cls(
            execution_id=context.execution_id,
            workflow_id=context.workflow.id,
            status=context.status,
            input_data=context.input,
            context_data=context.context,
            step_results=step_results,
            current_step=context.current_step,
            created_at=context.started_at or now,
            updated_at=now,
            workflow_json=json.dumps(context.workflow.to_dict()),
            metadata={
                "trace_id": context.trace_id,
                "correlation_id": context.correlation_id,
            },
        )


# =============================================================================
# State Manager Interface
# =============================================================================


class StateManager:
    """
    Abstract interface for state management.
    
    Provides checkpoint and recovery functionality for workflows.
    """
    
    async def save_checkpoint(self, context: ExecutionContext) -> None:
        """Save execution state checkpoint."""
        raise NotImplementedError
    
    async def load_checkpoint(self, execution_id: str) -> Optional[ExecutionCheckpoint]:
        """Load checkpoint by execution ID."""
        raise NotImplementedError
    
    async def delete_checkpoint(self, execution_id: str) -> bool:
        """Delete a checkpoint."""
        raise NotImplementedError
    
    async def list_checkpoints(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        limit: int = 100,
    ) -> List[ExecutionCheckpoint]:
        """List checkpoints with optional filtering."""
        raise NotImplementedError
    
    async def get_resumable_executions(
        self,
        workflow_id: Optional[str] = None,
    ) -> List[ExecutionCheckpoint]:
        """Get executions that can be resumed."""
        raise NotImplementedError


# =============================================================================
# In-Memory State Manager
# =============================================================================


class InMemoryStateManager(StateManager):
    """In-memory state manager for testing and development."""
    
    def __init__(self):
        """Initialize the in-memory state manager."""
        self._checkpoints: Dict[str, ExecutionCheckpoint] = {}
        self._lock = asyncio.Lock()
    
    async def save_checkpoint(self, context: ExecutionContext) -> None:
        """Save execution state checkpoint."""
        async with self._lock:
            checkpoint = ExecutionCheckpoint.from_context(context)
            self._checkpoints[context.execution_id] = checkpoint
    
    async def load_checkpoint(self, execution_id: str) -> Optional[ExecutionCheckpoint]:
        """Load checkpoint by execution ID."""
        return self._checkpoints.get(execution_id)
    
    async def delete_checkpoint(self, execution_id: str) -> bool:
        """Delete a checkpoint."""
        async with self._lock:
            if execution_id in self._checkpoints:
                del self._checkpoints[execution_id]
                return True
            return False
    
    async def list_checkpoints(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        limit: int = 100,
    ) -> List[ExecutionCheckpoint]:
        """List checkpoints with optional filtering."""
        result = []
        for checkpoint in self._checkpoints.values():
            if workflow_id and checkpoint.workflow_id != workflow_id:
                continue
            if status and checkpoint.status != status:
                continue
            result.append(checkpoint)
            if len(result) >= limit:
                break
        return result
    
    async def get_resumable_executions(
        self,
        workflow_id: Optional[str] = None,
    ) -> List[ExecutionCheckpoint]:
        """Get executions that can be resumed."""
        resumable_statuses = {ExecutionStatus.RUNNING, ExecutionStatus.PENDING}
        return await self.list_checkpoints(
            workflow_id=workflow_id,
            status=None,  # Filter in loop
        )


# =============================================================================
# SQLite State Manager
# =============================================================================


class SQLiteStateManager(StateManager):
    """SQLite-based state manager for persistence."""
    
    def __init__(self, db_path: str = "awf_state.db"):
        """
        Initialize SQLite state manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._lock = asyncio.Lock()
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    execution_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    context_data TEXT NOT NULL,
                    step_results TEXT NOT NULL,
                    current_step TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    workflow_json TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflow_id 
                ON checkpoints(workflow_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status 
                ON checkpoints(status)
            """)
            conn.commit()
        finally:
            conn.close()
    
    async def save_checkpoint(self, context: ExecutionContext) -> None:
        """Save execution state checkpoint."""
        checkpoint = ExecutionCheckpoint.from_context(context)
        
        async with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO checkpoints 
                    (execution_id, workflow_id, status, input_data, context_data,
                     step_results, current_step, created_at, updated_at, 
                     workflow_json, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    checkpoint.execution_id,
                    checkpoint.workflow_id,
                    checkpoint.status.value,
                    json.dumps(checkpoint.input_data),
                    json.dumps(checkpoint.context_data),
                    json.dumps(checkpoint.step_results),
                    checkpoint.current_step,
                    checkpoint.created_at.isoformat(),
                    checkpoint.updated_at.isoformat(),
                    checkpoint.workflow_json,
                    json.dumps(checkpoint.metadata),
                ))
                conn.commit()
            finally:
                conn.close()
    
    async def load_checkpoint(self, execution_id: str) -> Optional[ExecutionCheckpoint]:
        """Load checkpoint by execution ID."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "SELECT * FROM checkpoints WHERE execution_id = ?",
                (execution_id,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_checkpoint(row)
            return None
        finally:
            conn.close()
    
    async def delete_checkpoint(self, execution_id: str) -> bool:
        """Delete a checkpoint."""
        async with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    "DELETE FROM checkpoints WHERE execution_id = ?",
                    (execution_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()
    
    async def list_checkpoints(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[ExecutionStatus] = None,
        limit: int = 100,
    ) -> List[ExecutionCheckpoint]:
        """List checkpoints with optional filtering."""
        conn = sqlite3.connect(self.db_path)
        try:
            query = "SELECT * FROM checkpoints WHERE 1=1"
            params: List[Any] = []
            
            if workflow_id:
                query += " AND workflow_id = ?"
                params.append(workflow_id)
            if status:
                query += " AND status = ?"
                params.append(status.value)
            
            query += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            return [self._row_to_checkpoint(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    async def get_resumable_executions(
        self,
        workflow_id: Optional[str] = None,
    ) -> List[ExecutionCheckpoint]:
        """Get executions that can be resumed."""
        conn = sqlite3.connect(self.db_path)
        try:
            query = """
                SELECT * FROM checkpoints 
                WHERE status IN ('running', 'pending')
            """
            params: List[Any] = []
            
            if workflow_id:
                query += " AND workflow_id = ?"
                params.append(workflow_id)
            
            query += " ORDER BY updated_at DESC"
            
            cursor = conn.execute(query, params)
            return [self._row_to_checkpoint(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def _row_to_checkpoint(self, row: tuple) -> ExecutionCheckpoint:
        """Convert database row to checkpoint."""
        return ExecutionCheckpoint(
            execution_id=row[0],
            workflow_id=row[1],
            status=ExecutionStatus(row[2]),
            input_data=json.loads(row[3]),
            context_data=json.loads(row[4]),
            step_results=json.loads(row[5]),
            current_step=row[6],
            created_at=datetime.fromisoformat(row[7]),
            updated_at=datetime.fromisoformat(row[8]),
            workflow_json=row[9],
            metadata=json.loads(row[10]),
        )
    
    async def cleanup_old_checkpoints(self, days: int = 7) -> int:
        """
        Clean up checkpoints older than specified days.
        
        Args:
            days: Delete checkpoints older than this many days
        
        Returns:
            Number of deleted checkpoints
        """
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        async with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    "DELETE FROM checkpoints WHERE updated_at < ?",
                    (cutoff.isoformat(),)
                )
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()


# =============================================================================
# Checkpoint Recovery
# =============================================================================


def restore_context_from_checkpoint(
    checkpoint: ExecutionCheckpoint,
) -> ExecutionContext:
    """
    Restore an ExecutionContext from a checkpoint.
    
    Args:
        checkpoint: The checkpoint to restore from
    
    Returns:
        Restored ExecutionContext
    """
    # Deserialize workflow
    workflow_data = json.loads(checkpoint.workflow_json)
    workflow = WorkflowDefinition.from_dict(workflow_data)
    
    # Create context
    context = ExecutionContext(
        execution_id=checkpoint.execution_id,
        workflow=workflow,
        input=checkpoint.input_data,
        context=checkpoint.context_data,
        status=checkpoint.status,
        step_results={},
        current_step=checkpoint.current_step,
        started_at=checkpoint.created_at,
        trace_id=checkpoint.metadata.get("trace_id"),
        correlation_id=checkpoint.metadata.get("correlation_id"),
    )
    
    # Restore step results
    for step_id, result_data in checkpoint.step_results.items():
        context.step_results[step_id] = StepResult(
            step_id=step_id,
            status=StepStatus(result_data["status"]),
            output=result_data.get("output"),
            error=result_data.get("error"),
        )
    
    return context
