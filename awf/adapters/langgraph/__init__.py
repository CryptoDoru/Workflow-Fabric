"""
AI Workflow Fabric - LangGraph Adapter Package

This package provides the adapter for integrating LangGraph agents
with AI Workflow Fabric.

Example usage:
    ```python
    from langgraph.graph import StateGraph
    from awf.adapters.langgraph import LangGraphAdapter
    from awf.core import Task
    
    # Create your LangGraph agent
    class MyState(TypedDict):
        messages: list
        context: str
    
    graph = StateGraph(MyState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_edge("researcher", "writer")
    compiled = graph.compile()
    
    # Register with AWF
    adapter = LangGraphAdapter()
    manifest = adapter.register(
        compiled,
        agent_id="my-research-agent",
        metadata={
            "name": "Research Agent",
            "description": "An agent that researches topics and writes summaries",
            "tags": ["research", "writing"],
        }
    )
    
    print(f"Registered agent: {manifest.id}")
    print(f"Capabilities: {[c.name for c in manifest.capabilities]}")
    
    # Execute via ASP
    task = Task(
        agent_id="my-research-agent",
        input={"messages": [], "context": "AI Safety"},
    )
    result = await adapter.execute(task)
    
    if result.status == TaskStatus.COMPLETED:
        print(f"Output: {result.output}")
    else:
        print(f"Error: {result.error}")
    ```
"""

from awf.adapters.langgraph.adapter import LangGraphAdapter
from awf.adapters.langgraph.manifest import ManifestGenerator
from awf.adapters.langgraph.executor import TaskExecutor, ExecutionContext

__all__ = [
    "LangGraphAdapter",
    "ManifestGenerator",
    "TaskExecutor",
    "ExecutionContext",
]
