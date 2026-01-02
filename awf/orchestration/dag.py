"""
AI Workflow Fabric - DAG Builder

This module provides DAG (Directed Acyclic Graph) construction and 
topological sorting for workflow step execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from awf.orchestration.types import StepDefinition, WorkflowDefinition
from awf.orchestration.errors import CyclicDependencyError, InvalidStepReferenceError


# =============================================================================
# DAG Node
# =============================================================================


@dataclass
class DAGNode:
    """A node in the workflow DAG."""
    
    step_id: str
    step: StepDefinition
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    
    def __hash__(self) -> int:
        return hash(self.step_id)
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, DAGNode):
            return self.step_id == other.step_id
        return False


# =============================================================================
# DAG Builder
# =============================================================================


@dataclass
class WorkflowDAG:
    """
    Directed Acyclic Graph representation of a workflow.
    
    Provides efficient access to step execution order, parallel groups,
    and dependency tracking.
    """
    
    workflow_id: str
    nodes: Dict[str, DAGNode] = field(default_factory=dict)
    _topo_order: Optional[List[str]] = field(default=None, repr=False)
    _parallel_groups: Optional[List[List[str]]] = field(default=None, repr=False)
    
    @property
    def step_count(self) -> int:
        """Number of steps in the DAG."""
        return len(self.nodes)
    
    @property
    def root_steps(self) -> List[str]:
        """Steps with no dependencies (entry points)."""
        return [
            node.step_id 
            for node in self.nodes.values() 
            if not node.dependencies
        ]
    
    @property
    def leaf_steps(self) -> List[str]:
        """Steps with no dependents (exit points)."""
        return [
            node.step_id 
            for node in self.nodes.values() 
            if not node.dependents
        ]
    
    def get_node(self, step_id: str) -> Optional[DAGNode]:
        """Get a node by step ID."""
        return self.nodes.get(step_id)
    
    def get_dependencies(self, step_id: str) -> Set[str]:
        """Get direct dependencies of a step."""
        node = self.nodes.get(step_id)
        return node.dependencies if node else set()
    
    def get_dependents(self, step_id: str) -> Set[str]:
        """Get direct dependents of a step (steps that depend on this one)."""
        node = self.nodes.get(step_id)
        return node.dependents if node else set()
    
    def get_all_ancestors(self, step_id: str) -> Set[str]:
        """Get all ancestors (transitive dependencies) of a step."""
        ancestors: Set[str] = set()
        to_visit = list(self.get_dependencies(step_id))
        
        while to_visit:
            current = to_visit.pop()
            if current not in ancestors:
                ancestors.add(current)
                to_visit.extend(self.get_dependencies(current))
        
        return ancestors
    
    def get_all_descendants(self, step_id: str) -> Set[str]:
        """Get all descendants (transitive dependents) of a step."""
        descendants: Set[str] = set()
        to_visit = list(self.get_dependents(step_id))
        
        while to_visit:
            current = to_visit.pop()
            if current not in descendants:
                descendants.add(current)
                to_visit.extend(self.get_dependents(current))
        
        return descendants
    
    def get_topological_order(self) -> List[str]:
        """
        Get steps in topological order (dependencies before dependents).
        
        Uses Kahn's algorithm for deterministic ordering.
        
        Returns:
            List of step IDs in valid execution order
        
        Raises:
            CyclicDependencyError: If the graph contains a cycle
        """
        if self._topo_order is not None:
            return self._topo_order
        
        # Calculate in-degree for each node
        in_degree: Dict[str, int] = {
            step_id: len(node.dependencies)
            for step_id, node in self.nodes.items()
        }
        
        # Start with nodes that have no dependencies
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        queue.sort()  # Sort for deterministic ordering
        
        result: List[str] = []
        
        while queue:
            # Pop the first (alphabetically) node
            current = queue.pop(0)
            result.append(current)
            
            # Decrease in-degree for all dependents
            for dependent in sorted(self.get_dependents(current)):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
                    queue.sort()  # Keep sorted for determinism
        
        # Check for cycles
        if len(result) != len(self.nodes):
            # Find the cycle for error reporting
            remaining = set(self.nodes.keys()) - set(result)
            cycle = self._find_cycle(remaining)
            raise CyclicDependencyError(cycle=cycle, workflow_id=self.workflow_id)
        
        self._topo_order = result
        return result
    
    def get_parallel_groups(self) -> List[List[str]]:
        """
        Group steps into parallel execution levels.
        
        Each group contains steps that can execute in parallel
        (all their dependencies are in previous groups).
        
        Returns:
            List of groups, where each group is a list of step IDs
        """
        if self._parallel_groups is not None:
            return self._parallel_groups
        
        # Track which steps have been assigned to groups
        assigned: Set[str] = set()
        groups: List[List[str]] = []
        
        while len(assigned) < len(self.nodes):
            # Find all steps whose dependencies are all assigned
            current_group: List[str] = []
            
            for step_id, node in self.nodes.items():
                if step_id not in assigned:
                    if node.dependencies <= assigned:
                        current_group.append(step_id)
            
            if not current_group:
                # This shouldn't happen if there are no cycles
                break
            
            # Sort for deterministic ordering
            current_group.sort()
            groups.append(current_group)
            assigned.update(current_group)
        
        self._parallel_groups = groups
        return groups
    
    def get_ready_steps(self, completed: Set[str]) -> List[str]:
        """
        Get steps that are ready to execute.
        
        A step is ready if all its dependencies are in the completed set.
        
        Args:
            completed: Set of completed step IDs
        
        Returns:
            List of step IDs ready to execute
        """
        ready: List[str] = []
        
        for step_id, node in self.nodes.items():
            if step_id not in completed:
                if node.dependencies <= completed:
                    ready.append(step_id)
        
        return sorted(ready)
    
    def _find_cycle(self, remaining: Set[str]) -> List[str]:
        """Find a cycle in the remaining nodes for error reporting."""
        if not remaining:
            return []
        
        # Start from any remaining node
        start = next(iter(remaining))
        visited: Set[str] = set()
        path: List[str] = []
        
        def dfs(node: str) -> Optional[List[str]]:
            if node in path:
                # Found cycle
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]
            
            if node in visited:
                return None
            
            visited.add(node)
            path.append(node)
            
            for dep in self.get_dependencies(node):
                if dep in remaining:
                    result = dfs(dep)
                    if result:
                        return result
            
            path.pop()
            return None
        
        result = dfs(start)
        return result or [start]
    
    def validate(self) -> List[str]:
        """
        Validate the DAG structure.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors: List[str] = []
        
        # Check for cycles
        try:
            self.get_topological_order()
        except CyclicDependencyError as e:
            errors.append(str(e))
        
        # Check for invalid references
        for step_id, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    errors.append(
                        f"Step '{step_id}' references unknown step '{dep}'"
                    )
        
        return errors


# =============================================================================
# DAG Builder Functions
# =============================================================================


def build_dag(workflow: WorkflowDefinition) -> WorkflowDAG:
    """
    Build a DAG from a workflow definition.
    
    Args:
        workflow: The workflow definition
    
    Returns:
        WorkflowDAG representing the workflow structure
    
    Raises:
        InvalidStepReferenceError: If a step references an unknown step
    """
    dag = WorkflowDAG(workflow_id=workflow.id)
    step_ids = {step.id for step in workflow.steps}
    
    # Create nodes
    for step in workflow.steps:
        node = DAGNode(
            step_id=step.id,
            step=step,
            dependencies=set(step.depends_on),
        )
        dag.nodes[step.id] = node
    
    # Validate references and build dependents
    for step in workflow.steps:
        for dep_id in step.depends_on:
            if dep_id not in step_ids:
                raise InvalidStepReferenceError(
                    step_id=step.id,
                    referenced_step_id=dep_id,
                    workflow_id=workflow.id,
                )
            dag.nodes[dep_id].dependents.add(step.id)
    
    return dag


def get_execution_order(workflow: WorkflowDefinition) -> List[str]:
    """
    Get the execution order for workflow steps.
    
    Convenience function that builds a DAG and returns topological order.
    
    Args:
        workflow: The workflow definition
    
    Returns:
        List of step IDs in valid execution order
    """
    dag = build_dag(workflow)
    return dag.get_topological_order()


def get_parallel_execution_plan(workflow: WorkflowDefinition) -> List[List[str]]:
    """
    Get a parallel execution plan for workflow steps.
    
    Steps in the same group can be executed in parallel.
    
    Args:
        workflow: The workflow definition
    
    Returns:
        List of groups, where each group contains step IDs
        that can be executed in parallel
    """
    dag = build_dag(workflow)
    return dag.get_parallel_groups()


def detect_cycle(workflow: WorkflowDefinition) -> Optional[List[str]]:
    """
    Detect if the workflow contains a cycle.
    
    Args:
        workflow: The workflow definition
    
    Returns:
        List of step IDs forming the cycle, or None if no cycle
    """
    dag = build_dag(workflow)
    try:
        dag.get_topological_order()
        return None
    except CyclicDependencyError as e:
        return e.cycle


def get_affected_steps(workflow: WorkflowDefinition, step_id: str) -> Set[str]:
    """
    Get all steps affected if a given step fails.
    
    Returns all steps that depend (directly or transitively) on the given step.
    
    Args:
        workflow: The workflow definition
        step_id: The step to check
    
    Returns:
        Set of step IDs that would be affected
    """
    dag = build_dag(workflow)
    return dag.get_all_descendants(step_id)
