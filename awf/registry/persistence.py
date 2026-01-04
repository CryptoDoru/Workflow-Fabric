"""
AI Workflow Fabric - SQLite Registry Persistence

This module provides SQLite-based persistence for the agent registry,
enabling durable storage across restarts.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from awf.adapters.base import AgentRegistry
from awf.core.types import (
    AgentManifest,
    AgentStatus,
    Capability,
    CapabilityType,
    Schema,
    SchemaProperty,
)


class SQLiteRegistry(AgentRegistry):
    """
    SQLite-backed implementation of the AgentRegistry.
    
    This registry provides persistent storage with full ACID guarantees.
    Suitable for production single-instance or small-scale deployments.
    
    Example usage:
        ```python
        # Create with default path
        registry = SQLiteRegistry()
        await registry.initialize()
        
        # Or specify custom path
        registry = SQLiteRegistry(db_path="/var/lib/awf/registry.db")
        await registry.initialize()
        
        # Use normally
        await registry.register(manifest)
        agents = await registry.search(capabilities=["web_search"])
        
        # Close when done
        await registry.close()
        ```
    
    The registry stores manifests as JSON blobs with indexed metadata for
    efficient querying. Schema:
    
        agents:
            - id (PRIMARY KEY)
            - name
            - version
            - framework
            - status
            - trust_score
            - tags (JSON array)
            - capabilities (JSON array of names)
            - manifest_json (full manifest as JSON)
            - registered_at
            - updated_at
    """
    
    DEFAULT_DB_PATH = ".awf/registry.db"
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        create_tables: bool = True,
    ):
        """
        Initialize the SQLite registry.
        
        Args:
            db_path: Path to SQLite database file. If None, uses DEFAULT_DB_PATH.
            create_tables: Whether to create tables on initialization.
        """
        self._db_path = Path(db_path) if db_path else Path(self.DEFAULT_DB_PATH)
        self._create_tables = create_tables
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        Initialize the database connection and create tables if needed.
        
        This must be called before using the registry.
        """
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            # Create parent directory if needed
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create connection (SQLite is sync, but we wrap for async API)
            self._connection = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            
            if self._create_tables:
                await self._create_schema()
            
            self._initialized = True
    
    async def _create_schema(self) -> None:
        """Create the database schema."""
        assert self._connection is not None
        
        cursor = self._connection.cursor()
        
        # Main agents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                framework TEXT NOT NULL,
                framework_version TEXT,
                status TEXT NOT NULL DEFAULT 'registered',
                trust_score REAL,
                publisher TEXT,
                audit_status TEXT,
                description TEXT,
                documentation_url TEXT,
                source_url TEXT,
                tags_json TEXT DEFAULT '[]',
                capabilities_json TEXT DEFAULT '[]',
                manifest_json TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Index for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agents_framework
            ON agents(framework)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agents_status
            ON agents(status)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agents_trust_score
            ON agents(trust_score)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_agents_updated_at
            ON agents(updated_at)
        """)
        
        # Capabilities junction table for efficient capability search
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_capabilities (
                agent_id TEXT NOT NULL,
                capability_name TEXT NOT NULL,
                capability_type TEXT NOT NULL,
                PRIMARY KEY (agent_id, capability_name),
                FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_capabilities_name
            ON agent_capabilities(capability_name)
        """)
        
        # Tags junction table for efficient tag search
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_tags (
                agent_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (agent_id, tag),
                FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_tag
            ON agent_tags(tag)
        """)
        
        self._connection.commit()
    
    async def close(self) -> None:
        """Close the database connection."""
        async with self._lock:
            if self._connection:
                self._connection.close()
                self._connection = None
            self._initialized = False
    
    @asynccontextmanager
    async def _get_cursor(self) -> AsyncIterator[sqlite3.Cursor]:
        """Get a database cursor with automatic commit/rollback."""
        if not self._initialized or self._connection is None:
            raise RuntimeError("Registry not initialized. Call initialize() first.")
        
        async with self._lock:
            cursor = self._connection.cursor()
            try:
                yield cursor
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
    
    def _manifest_to_row(self, manifest: AgentManifest) -> Dict[str, Any]:
        """Convert an AgentManifest to a database row."""
        return {
            "id": manifest.id,
            "name": manifest.name,
            "version": manifest.version,
            "framework": manifest.framework,
            "framework_version": manifest.framework_version,
            "status": manifest.status.value,
            "trust_score": manifest.trust_score,
            "publisher": manifest.publisher,
            "audit_status": manifest.audit_status,
            "description": manifest.description,
            "documentation_url": manifest.documentation_url,
            "source_url": manifest.source_url,
            "tags_json": json.dumps(manifest.tags),
            "capabilities_json": json.dumps([cap.name for cap in manifest.capabilities]),
            "manifest_json": json.dumps(manifest.to_dict()),
            "registered_at": manifest.registered_at.isoformat() if manifest.registered_at else datetime.now(timezone.utc).isoformat(),
            "updated_at": manifest.updated_at.isoformat() if manifest.updated_at else datetime.now(timezone.utc).isoformat(),
        }
    
    def _row_to_manifest(self, row: sqlite3.Row) -> AgentManifest:
        """Convert a database row to an AgentManifest."""
        manifest_data = json.loads(row["manifest_json"])
        
        # Parse capabilities
        capabilities = []
        for cap_data in manifest_data.get("capabilities", []):
            capabilities.append(Capability(
                name=cap_data["name"],
                type=CapabilityType(cap_data.get("type", "custom")),
                description=cap_data.get("description"),
                permissions=cap_data.get("permissions", []),
                metadata=cap_data.get("metadata", {}),
            ))
        
        # Parse schemas if present
        input_schema = None
        output_schema = None
        
        if "inputSchema" in manifest_data:
            input_schema = self._parse_schema(manifest_data["inputSchema"])
        
        if "outputSchema" in manifest_data:
            output_schema = self._parse_schema(manifest_data["outputSchema"])
        
        return AgentManifest(
            id=row["id"],
            name=row["name"],
            version=row["version"],
            framework=row["framework"],
            framework_version=row["framework_version"],
            status=AgentStatus(row["status"]),
            trust_score=row["trust_score"],
            publisher=row["publisher"],
            audit_status=row["audit_status"],
            description=row["description"],
            documentation_url=row["documentation_url"],
            source_url=row["source_url"],
            tags=json.loads(row["tags_json"]),
            capabilities=capabilities,
            input_schema=input_schema,
            output_schema=output_schema,
            registered_at=datetime.fromisoformat(row["registered_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=manifest_data.get("metadata", {}),
        )
    
    def _parse_schema(self, schema_data: Dict[str, Any]) -> Schema:
        """Parse a JSON schema dict into a Schema object."""
        properties = {}
        for name, prop_data in schema_data.get("properties", {}).items():
            properties[name] = SchemaProperty(
                name=name,
                type=prop_data.get("type", "string"),
                description=prop_data.get("description"),
                required=name in schema_data.get("required", []),
                default=prop_data.get("default"),
                enum=prop_data.get("enum"),
                items=prop_data.get("items"),
            )
        
        return Schema(
            type=schema_data.get("type", "object"),
            properties=properties,
            required=schema_data.get("required", []),
            additional_properties=schema_data.get("additionalProperties", False),
            description=schema_data.get("description"),
        )
    
    async def register(self, manifest: AgentManifest) -> None:
        """
        Store an agent manifest.
        
        Args:
            manifest: The AgentManifest to store
        """
        manifest.updated_at = datetime.now(timezone.utc)
        row = self._manifest_to_row(manifest)
        
        async with self._get_cursor() as cursor:
            # Insert or replace the main record
            cursor.execute("""
                INSERT OR REPLACE INTO agents (
                    id, name, version, framework, framework_version,
                    status, trust_score, publisher, audit_status,
                    description, documentation_url, source_url,
                    tags_json, capabilities_json, manifest_json,
                    registered_at, updated_at
                ) VALUES (
                    :id, :name, :version, :framework, :framework_version,
                    :status, :trust_score, :publisher, :audit_status,
                    :description, :documentation_url, :source_url,
                    :tags_json, :capabilities_json, :manifest_json,
                    :registered_at, :updated_at
                )
            """, row)
            
            # Update capabilities junction table
            cursor.execute(
                "DELETE FROM agent_capabilities WHERE agent_id = ?",
                (manifest.id,)
            )
            for cap in manifest.capabilities:
                cursor.execute("""
                    INSERT INTO agent_capabilities (agent_id, capability_name, capability_type)
                    VALUES (?, ?, ?)
                """, (manifest.id, cap.name, cap.type.value))
            
            # Update tags junction table
            cursor.execute(
                "DELETE FROM agent_tags WHERE agent_id = ?",
                (manifest.id,)
            )
            for tag in manifest.tags:
                cursor.execute("""
                    INSERT INTO agent_tags (agent_id, tag)
                    VALUES (?, ?)
                """, (manifest.id, tag))
    
    async def get(self, agent_id: str) -> Optional[AgentManifest]:
        """
        Retrieve an agent manifest by ID.
        
        Args:
            agent_id: The ID of the agent
        
        Returns:
            The AgentManifest if found, None otherwise
        """
        async with self._get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM agents WHERE id = ?",
                (agent_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return self._row_to_manifest(row)
    
    async def search(
        self,
        capabilities: Optional[List[str]] = None,
        framework: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_trust_score: Optional[float] = None,
    ) -> List[AgentManifest]:
        """
        Search for agents matching criteria.
        
        Args:
            capabilities: Required capabilities (agent must have ALL)
            framework: Filter by framework name
            tags: Required tags (agent must have ALL)
            min_trust_score: Minimum trust score
        
        Returns:
            List of matching AgentManifest objects
        """
        query = "SELECT DISTINCT a.* FROM agents a"
        conditions = ["a.status = 'active'"]
        params: List[Any] = []
        
        # Join with capabilities table if filtering by capabilities
        if capabilities:
            for i, cap in enumerate(capabilities):
                alias = f"c{i}"
                query += f" INNER JOIN agent_capabilities {alias} ON a.id = {alias}.agent_id AND {alias}.capability_name = ?"
                params.append(cap)
        
        # Join with tags table if filtering by tags
        if tags:
            for i, tag in enumerate(tags):
                alias = f"t{i}"
                query += f" INNER JOIN agent_tags {alias} ON a.id = {alias}.agent_id AND {alias}.tag = ?"
                params.append(tag)
        
        # Framework filter
        if framework:
            conditions.append("a.framework = ?")
            params.append(framework)
        
        # Trust score filter
        if min_trust_score is not None:
            conditions.append("a.trust_score >= ?")
            params.append(min_trust_score)
        
        # Build final query
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY a.updated_at DESC"
        
        async with self._get_cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_manifest(row) for row in rows]
    
    async def delete(self, agent_id: str) -> bool:
        """
        Remove an agent from the registry.
        
        Args:
            agent_id: The ID of the agent to remove
        
        Returns:
            True if removed, False if not found
        """
        async with self._get_cursor() as cursor:
            cursor.execute(
                "DELETE FROM agents WHERE id = ?",
                (agent_id,)
            )
            return cursor.rowcount > 0
    
    async def list_all(self) -> List[AgentManifest]:
        """
        List all registered agents.
        
        Returns:
            List of all AgentManifest objects
        """
        async with self._get_cursor() as cursor:
            cursor.execute("SELECT * FROM agents ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            
            return [self._row_to_manifest(row) for row in rows]
    
    async def update(self, manifest: AgentManifest) -> bool:
        """
        Update an existing agent manifest.
        
        Args:
            manifest: The updated manifest
        
        Returns:
            True if updated, False if not found
        """
        # Check if exists first
        existing = await self.get(manifest.id)
        if existing is None:
            return False
        
        # Preserve original registration time
        manifest.registered_at = existing.registered_at
        manifest.updated_at = datetime.now(timezone.utc)
        
        await self.register(manifest)
        return True
    
    async def set_status(self, agent_id: str, status: AgentStatus) -> bool:
        """
        Update the status of an agent.
        
        Args:
            agent_id: The ID of the agent
            status: The new status
        
        Returns:
            True if updated, False if not found
        """
        now = datetime.now(timezone.utc).isoformat()
        
        async with self._get_cursor() as cursor:
            cursor.execute("""
                UPDATE agents 
                SET status = ?, updated_at = ?
                WHERE id = ?
            """, (status.value, now, agent_id))
            
            return cursor.rowcount > 0
    
    async def count(self) -> int:
        """Get the number of registered agents."""
        async with self._get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM agents")
            row = cursor.fetchone()
            return row[0] if row else 0
    
    async def clear(self) -> None:
        """Remove all agents from the registry."""
        async with self._get_cursor() as cursor:
            cursor.execute("DELETE FROM agent_tags")
            cursor.execute("DELETE FROM agent_capabilities")
            cursor.execute("DELETE FROM agents")
    
    async def search_by_capability(
        self,
        capability_name: str,
        capability_type: Optional[CapabilityType] = None,
    ) -> List[AgentManifest]:
        """
        Search for agents with a specific capability.
        
        Args:
            capability_name: The capability name to search for
            capability_type: Optional type filter
        
        Returns:
            List of matching AgentManifest objects
        """
        query = """
            SELECT DISTINCT a.* FROM agents a
            INNER JOIN agent_capabilities c ON a.id = c.agent_id
            WHERE a.status = 'active' AND c.capability_name = ?
        """
        params: List[Any] = [capability_name]
        
        if capability_type:
            query += " AND c.capability_type = ?"
            params.append(capability_type.value)
        
        query += " ORDER BY a.trust_score DESC NULLS LAST"
        
        async with self._get_cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_manifest(row) for row in rows]
    
    async def vacuum(self) -> None:
        """
        Optimize the database by running VACUUM.
        
        This reclaims space from deleted rows and optimizes the database file.
        Should be run periodically in production.
        """
        if self._connection:
            self._connection.execute("VACUUM")
    
    @property
    def db_path(self) -> Path:
        """Get the database file path."""
        return self._db_path
    
    @property
    def is_initialized(self) -> bool:
        """Check if the registry is initialized."""
        return self._initialized


class RegistryMigrator:
    """
    Handles database schema migrations for SQLiteRegistry.
    
    This class manages versioned migrations to upgrade the schema
    as the application evolves.
    """
    
    MIGRATIONS = [
        # (version, description, sql_up, sql_down)
        (
            1,
            "Initial schema",
            None,  # Already created by SQLiteRegistry._create_schema
            "DROP TABLE IF EXISTS agent_tags; DROP TABLE IF EXISTS agent_capabilities; DROP TABLE IF EXISTS agents;"
        ),
    ]
    
    def __init__(self, registry: SQLiteRegistry):
        self.registry = registry
    
    async def get_version(self) -> int:
        """Get the current schema version."""
        if not self.registry._connection:
            return 0
        
        cursor = self.registry._connection.cursor()
        
        # Check if version table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='schema_version'
        """)
        
        if cursor.fetchone() is None:
            # Create version table
            cursor.execute("""
                CREATE TABLE schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                INSERT INTO schema_version (version, applied_at)
                VALUES (1, ?)
            """, (datetime.now(timezone.utc).isoformat(),))
            self.registry._connection.commit()
            return 1
        
        cursor.execute("SELECT MAX(version) FROM schema_version")
        row = cursor.fetchone()
        return row[0] if row and row[0] else 0
    
    async def migrate(self, target_version: Optional[int] = None) -> int:
        """
        Run migrations up to target version.
        
        Args:
            target_version: Target version (None for latest)
        
        Returns:
            The new schema version
        """
        current = await self.get_version()
        target = target_version or len(self.MIGRATIONS)
        
        if current >= target:
            return current
        
        if self.registry._connection is None:
            raise RuntimeError("Registry not initialized")
        
        cursor = self.registry._connection.cursor()
        
        for version, description, sql_up, sql_down in self.MIGRATIONS:
            if version <= current:
                continue
            if version > target:
                break
            
            if sql_up:
                cursor.executescript(sql_up)
            
            cursor.execute("""
                INSERT INTO schema_version (version, applied_at)
                VALUES (?, ?)
            """, (version, datetime.now(timezone.utc).isoformat()))
        
        self.registry._connection.commit()
        return await self.get_version()
