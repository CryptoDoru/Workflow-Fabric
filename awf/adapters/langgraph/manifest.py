"""
AI Workflow Fabric - LangGraph Manifest Generator

This module provides utilities for generating ASP-compliant manifests
from LangGraph agents with advanced introspection.
"""

from __future__ import annotations

import hashlib
import inspect
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints

from awf.core.types import (
    AgentManifest,
    AgentStatus,
    Capability,
    CapabilityType,
    Schema,
    SchemaProperty,
)

# Type hints for LangGraph
try:
    from langgraph.graph import StateGraph
    from langgraph.graph.graph import CompiledGraph
    
    LANGGRAPH_AVAILABLE = True
except ImportError:
    StateGraph = Any
    CompiledGraph = Any
    LANGGRAPH_AVAILABLE = False


class ManifestGenerator:
    """
    Generates ASP-compliant manifests from LangGraph graphs.
    
    This class provides deep introspection of LangGraph StateGraph and
    CompiledGraph objects to extract metadata for ASP registration.
    """
    
    def __init__(self, framework_version: Optional[str] = None):
        """
        Initialize the manifest generator.
        
        Args:
            framework_version: Version of LangGraph being used
        """
        self.framework_version = framework_version
    
    def generate(
        self,
        graph: Union[StateGraph, CompiledGraph],
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        version: str = "1.0.0",
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        publisher: Optional[str] = None,
        documentation_url: Optional[str] = None,
        source_url: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentManifest:
        """
        Generate an ASP manifest from a LangGraph graph.
        
        Args:
            graph: The LangGraph StateGraph or CompiledGraph
            agent_id: Optional custom agent ID
            name: Optional human-readable name
            version: Semantic version string
            description: Optional description
            tags: Optional tags for discovery
            publisher: Optional publisher identifier
            documentation_url: Optional documentation URL
            source_url: Optional source code URL
            extra_metadata: Additional metadata to include
        
        Returns:
            AgentManifest: The generated manifest
        """
        # Ensure we have a compiled graph for full introspection
        compiled = self._ensure_compiled(graph)
        
        # Generate agent ID if not provided
        if agent_id is None:
            agent_id = self._generate_id(compiled)
        
        # Extract graph structure
        nodes = self._extract_nodes(compiled)
        edges = self._extract_edges(compiled)
        
        # Extract capabilities
        capabilities = self._extract_capabilities(compiled, nodes)
        
        # Extract schemas
        input_schema = self._extract_input_schema(compiled)
        output_schema = self._extract_output_schema(compiled)
        
        # Build metadata
        metadata: Dict[str, Any] = {
            "graph_structure": {
                "nodes": list(nodes.keys()),
                "edges": edges,
                "entry_point": self._get_entry_point(compiled),
                "end_points": self._get_end_points(compiled),
            },
        }
        
        if extra_metadata:
            metadata.update(extra_metadata)
        
        # Create manifest
        return AgentManifest(
            id=agent_id,
            name=name or agent_id,
            version=version,
            framework="langgraph",
            framework_version=self.framework_version,
            capabilities=capabilities,
            input_schema=input_schema,
            output_schema=output_schema,
            description=description,
            tags=tags or [],
            publisher=publisher,
            documentation_url=documentation_url,
            source_url=source_url,
            status=AgentStatus.REGISTERED,
            registered_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata=metadata,
        )
    
    def _ensure_compiled(
        self, graph: Union[StateGraph, CompiledGraph]
    ) -> CompiledGraph:
        """Ensure we have a compiled graph."""
        if hasattr(graph, "invoke"):
            return graph  # Already compiled
        return graph.compile()
    
    def _generate_id(self, compiled: CompiledGraph) -> str:
        """Generate a deterministic ID from graph structure."""
        nodes = sorted(self._extract_nodes(compiled).keys())
        edges = sorted([f"{e['source']}->{e['target']}" for e in self._extract_edges(compiled)])
        content = f"{nodes}|{edges}"
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"langgraph-{hash_val}"
    
    def _extract_nodes(self, compiled: CompiledGraph) -> Dict[str, Any]:
        """Extract node information from the graph."""
        nodes: Dict[str, Any] = {}
        
        try:
            graph_nodes = getattr(compiled, "nodes", {})
            for name, node in graph_nodes.items():
                nodes[name] = {
                    "name": name,
                    "callable": node,
                    "docstring": inspect.getdoc(node) if callable(node) else None,
                    "signature": self._get_signature(node) if callable(node) else None,
                }
        except Exception:
            pass
        
        return nodes
    
    def _extract_edges(self, compiled: CompiledGraph) -> List[Dict[str, Any]]:
        """Extract edge information from the graph."""
        edges: List[Dict[str, Any]] = []
        
        try:
            graph_edges = getattr(compiled, "edges", [])
            for edge in graph_edges:
                if isinstance(edge, tuple) and len(edge) >= 2:
                    edges.append({
                        "source": edge[0],
                        "target": edge[1],
                        "conditional": len(edge) > 2,
                    })
                elif isinstance(edge, dict):
                    edges.append(edge)
        except Exception:
            pass
        
        return edges
    
    def _get_entry_point(self, compiled: CompiledGraph) -> Optional[str]:
        """Get the entry point of the graph."""
        try:
            return getattr(compiled, "entry_point", None)
        except Exception:
            return None
    
    def _get_end_points(self, compiled: CompiledGraph) -> List[str]:
        """Get the end points of the graph."""
        try:
            # Look for nodes that have no outgoing edges
            nodes = set(self._extract_nodes(compiled).keys())
            edges = self._extract_edges(compiled)
            sources = {e["source"] for e in edges}
            end_points = nodes - sources
            return list(end_points)
        except Exception:
            return []
    
    def _extract_capabilities(
        self,
        compiled: CompiledGraph,
        nodes: Dict[str, Any],
    ) -> List[Capability]:
        """Extract capabilities from graph nodes."""
        capabilities: List[Capability] = []
        
        for name, node_info in nodes.items():
            # Create capability for each node
            cap = Capability(
                name=name,
                type=CapabilityType.REASONING,
                description=node_info.get("docstring") or f"Graph node: {name}",
                metadata={
                    "node_name": name,
                    "signature": node_info.get("signature"),
                },
            )
            capabilities.append(cap)
            
            # Extract tools from node
            node_callable = node_info.get("callable")
            if node_callable:
                tools = self._extract_node_tools(node_callable)
                for tool in tools:
                    tool_cap = Capability(
                        name=tool["name"],
                        type=CapabilityType.TOOL,
                        description=tool.get("description"),
                        permissions=tool.get("permissions", []),
                        metadata={
                            "source_node": name,
                            "tool_type": tool.get("type", "unknown"),
                        },
                    )
                    capabilities.append(tool_cap)
        
        return capabilities
    
    def _extract_node_tools(self, node_callable: Callable) -> List[Dict[str, Any]]:
        """Extract tools used by a node callable."""
        tools: List[Dict[str, Any]] = []
        
        try:
            # Check for tools attribute
            node_tools = getattr(node_callable, "tools", None)
            if node_tools:
                for tool in node_tools:
                    tools.append({
                        "name": getattr(tool, "name", str(tool)),
                        "description": getattr(tool, "description", None),
                        "type": "langgraph_tool",
                        "permissions": self._infer_permissions(tool),
                    })
            
            # Check for bound tools in closure
            if hasattr(node_callable, "__closure__") and node_callable.__closure__:
                for cell in node_callable.__closure__:
                    cell_content = cell.cell_contents
                    if hasattr(cell_content, "tools"):
                        for tool in cell_content.tools:
                            tools.append({
                                "name": getattr(tool, "name", str(tool)),
                                "description": getattr(tool, "description", None),
                                "type": "bound_tool",
                                "permissions": self._infer_permissions(tool),
                            })
        except Exception:
            pass
        
        return tools
    
    def _infer_permissions(self, tool: Any) -> List[str]:
        """Infer permissions required by a tool."""
        permissions: List[str] = []
        
        try:
            name = getattr(tool, "name", str(tool)).lower()
            
            # Network permissions
            if any(kw in name for kw in ["web", "http", "api", "fetch", "search", "scrape"]):
                permissions.append("network:external")
            
            # Filesystem permissions
            if any(kw in name for kw in ["file", "read", "write", "save", "load", "disk"]):
                permissions.append("filesystem:read")
            if any(kw in name for kw in ["write", "save", "create", "delete"]):
                permissions.append("filesystem:write")
            
            # Process permissions
            if any(kw in name for kw in ["exec", "shell", "command", "subprocess", "run"]):
                permissions.append("process:execute")
            
            # Database permissions
            if any(kw in name for kw in ["sql", "database", "db", "query", "mongo", "redis"]):
                permissions.append("database:query")
        except Exception:
            pass
        
        return permissions
    
    def _extract_input_schema(self, compiled: CompiledGraph) -> Optional[Schema]:
        """Extract input schema from the graph's state type."""
        try:
            state_type = self._get_state_type(compiled)
            if state_type:
                return self._type_to_schema(state_type)
        except Exception:
            pass
        return None
    
    def _extract_output_schema(self, compiled: CompiledGraph) -> Optional[Schema]:
        """Extract output schema from the graph's state type."""
        # In LangGraph, input and output typically use the same state type
        return self._extract_input_schema(compiled)
    
    def _get_state_type(self, compiled: CompiledGraph) -> Optional[Type]:
        """Get the state type from the compiled graph."""
        try:
            # Try different ways to access the state schema
            if hasattr(compiled, "schema"):
                return compiled.schema
            
            if hasattr(compiled, "builder"):
                builder = compiled.builder
                if hasattr(builder, "schema"):
                    return builder.schema
            
            # Check for state_schema attribute
            if hasattr(compiled, "state_schema"):
                return compiled.state_schema
        except Exception:
            pass
        
        return None
    
    def _type_to_schema(self, type_hint: Type) -> Schema:
        """Convert a Python type to an ASP Schema."""
        properties: Dict[str, SchemaProperty] = {}
        required: List[str] = []
        
        try:
            # Handle Pydantic models
            if hasattr(type_hint, "model_fields"):
                for field_name, field_info in type_hint.model_fields.items():
                    properties[field_name] = SchemaProperty(
                        name=field_name,
                        type=self._get_json_type(field_info.annotation),
                        description=field_info.description,
                        required=field_info.is_required(),
                        default=field_info.default if field_info.default is not None else None,
                    )
                    if field_info.is_required():
                        required.append(field_name)
            
            # Handle TypedDict
            elif hasattr(type_hint, "__annotations__"):
                annotations = get_type_hints(type_hint) if hasattr(type_hint, "__module__") else type_hint.__annotations__
                for field_name, field_type in annotations.items():
                    is_optional = self._is_optional_type(field_type)
                    properties[field_name] = SchemaProperty(
                        name=field_name,
                        type=self._get_json_type(field_type),
                        required=not is_optional,
                    )
                    if not is_optional:
                        required.append(field_name)
        except Exception:
            pass
        
        return Schema(
            type="object",
            properties=properties,
            required=required,
            description=type_hint.__doc__ if hasattr(type_hint, "__doc__") else None,
        )
    
    def _get_json_type(self, python_type: Type) -> str:
        """Map Python type to JSON schema type."""
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null",
        }
        
        # Handle Optional and Union types
        origin = getattr(python_type, "__origin__", None)
        if origin is Union:
            args = getattr(python_type, "__args__", ())
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return self._get_json_type(non_none[0])
            return "string"  # Fallback for complex unions
        
        if origin is list:
            return "array"
        if origin is dict:
            return "object"
        
        return type_mapping.get(python_type, "string")
    
    def _is_optional_type(self, type_hint: Type) -> bool:
        """Check if a type is Optional."""
        origin = getattr(type_hint, "__origin__", None)
        if origin is Union:
            args = getattr(type_hint, "__args__", ())
            return type(None) in args
        return False
    
    def _get_signature(self, callable_obj: Callable) -> Optional[str]:
        """Get the string representation of a callable's signature."""
        try:
            sig = inspect.signature(callable_obj)
            return str(sig)
        except Exception:
            return None


# Import Union for type checking
from typing import Union
