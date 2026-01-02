"""
AI Workflow Fabric - CrewAI Adapter Package

This package provides the adapter for integrating CrewAI agents
with AI Workflow Fabric.

Example usage:
    ```python
    from crewai import Agent, Crew, Task
    from awf.adapters.crewai import CrewAIAdapter
    from awf.core import Task as AWFTask
    
    # Create your CrewAI agents
    researcher = Agent(
        role="Senior Researcher",
        goal="Research AI topics thoroughly",
        backstory="An expert AI researcher",
        tools=[search_tool],
    )
    
    writer = Agent(
        role="Technical Writer",
        goal="Write clear technical content",
        backstory="A skilled technical writer",
    )
    
    # Create a crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[
            Task(description="Research AI safety", agent=researcher),
            Task(description="Write summary", agent=writer),
        ],
    )
    
    # Register with AWF
    adapter = CrewAIAdapter()
    manifest = adapter.register(
        crew,
        agent_id="research-crew",
        metadata={
            "name": "Research & Writing Crew",
            "description": "A crew that researches and writes",
            "tags": ["research", "writing"],
        }
    )
    
    print(f"Registered crew: {manifest.id}")
    print(f"Capabilities: {[c.name for c in manifest.capabilities]}")
    
    # Execute via ASP
    task = AWFTask(
        agent_id="research-crew",
        input={"task_description": "Research AI safety best practices"},
    )
    result = await adapter.execute(task)
    
    if result.status == TaskStatus.COMPLETED:
        print(f"Output: {result.output}")
    else:
        print(f"Error: {result.error}")
    ```
"""

from awf.adapters.crewai.adapter import CrewAIAdapter

__all__ = [
    "CrewAIAdapter",
]
