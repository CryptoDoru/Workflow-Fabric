"""
Tests for AWF Orchestration DAG Module

Comprehensive tests for DAG building, topological sorting, cycle detection,
and parallel execution planning. These tests verify actual graph algorithms
work correctly, not just that methods run without errors.
"""

import pytest

from awf.orchestration.types import StepDefinition, WorkflowDefinition
from awf.orchestration.dag import (
    DAGNode,
    WorkflowDAG,
    build_dag,
    get_execution_order,
    get_parallel_execution_plan,
    detect_cycle,
    get_affected_steps,
)
from awf.orchestration.errors import CyclicDependencyError, InvalidStepReferenceError


# =============================================================================
# Helper Functions
# =============================================================================


def make_workflow(steps_config: list[tuple[str, str, list[str]]]) -> WorkflowDefinition:
    """
    Create a workflow from a simple config.
    
    Args:
        steps_config: List of (step_id, agent_id, depends_on) tuples
    
    Returns:
        WorkflowDefinition
    """
    steps = [
        StepDefinition(id=step_id, agent_id=agent_id, depends_on=depends_on)
        for step_id, agent_id, depends_on in steps_config
    ]
    return WorkflowDefinition(id="test-workflow", name="Test Workflow", steps=steps)


# =============================================================================
# DAGNode Tests
# =============================================================================


class TestDAGNode:
    """Tests for DAGNode dataclass."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        step = StepDefinition(id="step1", agent_id="agent1")
        node = DAGNode(step_id="step1", step=step)
        
        assert node.step_id == "step1"
        assert node.step == step
        assert node.dependencies == set()
        assert node.dependents == set()
    
    def test_node_with_dependencies(self):
        """Test node with dependencies."""
        step = StepDefinition(id="step2", agent_id="agent2", depends_on=["step1"])
        node = DAGNode(
            step_id="step2",
            step=step,
            dependencies={"step1"},
        )
        
        assert "step1" in node.dependencies
    
    def test_node_equality(self):
        """Test node equality is based on step_id."""
        step1 = StepDefinition(id="step1", agent_id="agent1")
        step2 = StepDefinition(id="step1", agent_id="agent2")  # Different agent, same ID
        
        node1 = DAGNode(step_id="step1", step=step1)
        node2 = DAGNode(step_id="step1", step=step2)
        
        assert node1 == node2
    
    def test_node_hash(self):
        """Test node hashing for set operations."""
        step = StepDefinition(id="step1", agent_id="agent1")
        node = DAGNode(step_id="step1", step=step)
        
        # Should be usable in sets
        node_set = {node}
        assert len(node_set) == 1


# =============================================================================
# DAG Building Tests
# =============================================================================


class TestBuildDAG:
    """Tests for build_dag function."""
    
    def test_single_step_dag(self):
        """Test DAG with single step."""
        workflow = make_workflow([("step1", "agent1", [])])
        dag = build_dag(workflow)
        
        assert dag.step_count == 1
        assert "step1" in dag.nodes
        assert dag.root_steps == ["step1"]
        assert dag.leaf_steps == ["step1"]
    
    def test_linear_chain_dag(self):
        """Test DAG with linear chain: A -> B -> C."""
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", ["A"]),
            ("C", "agent", ["B"]),
        ])
        dag = build_dag(workflow)
        
        assert dag.step_count == 3
        assert dag.root_steps == ["A"]
        assert dag.leaf_steps == ["C"]
        
        # Verify dependencies
        assert dag.get_dependencies("A") == set()
        assert dag.get_dependencies("B") == {"A"}
        assert dag.get_dependencies("C") == {"B"}
        
        # Verify dependents
        assert dag.get_dependents("A") == {"B"}
        assert dag.get_dependents("B") == {"C"}
        assert dag.get_dependents("C") == set()
    
    def test_diamond_dag(self):
        """
        Test diamond pattern:
            A
           / \\
          B   C
           \\ /
            D
        """
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", ["A"]),
            ("C", "agent", ["A"]),
            ("D", "agent", ["B", "C"]),
        ])
        dag = build_dag(workflow)
        
        assert dag.step_count == 4
        assert dag.root_steps == ["A"]
        assert dag.leaf_steps == ["D"]
        
        # D depends on both B and C
        assert dag.get_dependencies("D") == {"B", "C"}
        
        # A has two dependents
        assert dag.get_dependents("A") == {"B", "C"}
    
    def test_multiple_roots_dag(self):
        """Test DAG with multiple entry points."""
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", []),
            ("C", "agent", ["A", "B"]),
        ])
        dag = build_dag(workflow)
        
        assert set(dag.root_steps) == {"A", "B"}
        assert dag.leaf_steps == ["C"]
    
    def test_multiple_leaves_dag(self):
        """Test DAG with multiple exit points."""
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", ["A"]),
            ("C", "agent", ["A"]),
        ])
        dag = build_dag(workflow)
        
        assert dag.root_steps == ["A"]
        assert set(dag.leaf_steps) == {"B", "C"}
    
    def test_invalid_dependency_reference(self):
        """Test that referencing non-existent step raises error."""
        # Can't use make_workflow because WorkflowDefinition validates
        steps = [
            StepDefinition(id="step1", agent_id="agent", depends_on=["nonexistent"])
        ]
        
        with pytest.raises(ValueError):
            WorkflowDefinition(id="test", name="Test", steps=steps)
    
    def test_complex_dag(self):
        """
        Test complex DAG:
            A     B
            |\\   /|
            | \\ / |
            |  X  |
            | / \\ |
            C     D
             \\   /
              \\ /
               E
        """
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", []),
            ("C", "agent", ["A", "B"]),
            ("D", "agent", ["A", "B"]),
            ("E", "agent", ["C", "D"]),
        ])
        dag = build_dag(workflow)
        
        assert dag.step_count == 5
        assert set(dag.root_steps) == {"A", "B"}
        assert dag.leaf_steps == ["E"]


# =============================================================================
# Topological Sort Tests
# =============================================================================


class TestTopologicalSort:
    """Tests for topological ordering."""
    
    def test_single_step_order(self):
        """Test ordering with single step."""
        workflow = make_workflow([("step1", "agent", [])])
        order = get_execution_order(workflow)
        
        assert order == ["step1"]
    
    def test_linear_chain_order(self):
        """Test ordering respects linear dependencies."""
        workflow = make_workflow([
            ("C", "agent", ["B"]),
            ("A", "agent", []),
            ("B", "agent", ["A"]),
        ])
        order = get_execution_order(workflow)
        
        # A must come before B, B must come before C
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")
    
    def test_diamond_order(self):
        """Test diamond pattern ordering."""
        workflow = make_workflow([
            ("D", "agent", ["B", "C"]),
            ("B", "agent", ["A"]),
            ("C", "agent", ["A"]),
            ("A", "agent", []),
        ])
        order = get_execution_order(workflow)
        
        # A must be first
        assert order[0] == "A"
        # D must be last
        assert order[-1] == "D"
        # B and C must be between A and D
        assert order.index("B") > order.index("A")
        assert order.index("C") > order.index("A")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")
    
    def test_order_is_deterministic(self):
        """Test that order is deterministic across multiple calls."""
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", []),
            ("C", "agent", ["A", "B"]),
        ])
        
        order1 = get_execution_order(workflow)
        order2 = get_execution_order(workflow)
        order3 = get_execution_order(workflow)
        
        assert order1 == order2 == order3
    
    def test_parallel_steps_alphabetical(self):
        """Test that steps with no dependencies between them are alphabetically ordered."""
        workflow = make_workflow([
            ("zebra", "agent", []),
            ("alpha", "agent", []),
            ("beta", "agent", []),
        ])
        order = get_execution_order(workflow)
        
        # Should be alphabetical since no dependencies
        assert order == ["alpha", "beta", "zebra"]


# =============================================================================
# Cycle Detection Tests
# =============================================================================


class TestCycleDetection:
    """Tests for cycle detection."""
    
    def test_no_cycle_returns_none(self):
        """Test that acyclic graph returns None."""
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", ["A"]),
            ("C", "agent", ["B"]),
        ])
        
        cycle = detect_cycle(workflow)
        assert cycle is None
    
    def test_self_loop_detected(self):
        """Test that self-loop is detected."""
        # Create step that depends on itself - need to bypass validation
        step = StepDefinition(id="A", agent_id="agent", depends_on=[])
        workflow = WorkflowDefinition(id="test", name="Test", steps=[step])
        
        # Manually add self-dependency to test DAG cycle detection
        dag = build_dag(workflow)
        dag.nodes["A"].dependencies.add("A")
        dag._topo_order = None  # Clear cache
        
        with pytest.raises(CyclicDependencyError) as exc_info:
            dag.get_topological_order()
        
        assert "A" in exc_info.value.cycle
    
    def test_two_step_cycle_detected(self):
        """Test that A -> B -> A cycle is detected."""
        # Create workflow first without the cycle
        step_a = StepDefinition(id="A", agent_id="agent", depends_on=[])
        step_b = StepDefinition(id="B", agent_id="agent", depends_on=["A"])
        workflow = WorkflowDefinition(id="test", name="Test", steps=[step_a, step_b])
        
        # Build DAG and add cycle
        dag = build_dag(workflow)
        dag.nodes["A"].dependencies.add("B")
        dag.nodes["B"].dependents.add("A")
        dag._topo_order = None  # Clear cache
        
        with pytest.raises(CyclicDependencyError) as exc_info:
            dag.get_topological_order()
        
        # Cycle should contain both A and B
        assert "A" in exc_info.value.cycle or "B" in exc_info.value.cycle
    
    def test_large_cycle_detected(self):
        """Test that A -> B -> C -> D -> A cycle is detected."""
        # Create acyclic workflow first
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", ["A"]),
            ("C", "agent", ["B"]),
            ("D", "agent", ["C"]),
        ])
        
        # Add cycle D -> A
        dag = build_dag(workflow)
        dag.nodes["A"].dependencies.add("D")
        dag.nodes["D"].dependents.add("A")
        dag._topo_order = None
        
        with pytest.raises(CyclicDependencyError):
            dag.get_topological_order()


# =============================================================================
# Parallel Groups Tests
# =============================================================================


class TestParallelGroups:
    """Tests for parallel execution group calculation."""
    
    def test_single_step_one_group(self):
        """Single step results in single group."""
        workflow = make_workflow([("step1", "agent", [])])
        groups = get_parallel_execution_plan(workflow)
        
        assert len(groups) == 1
        assert groups[0] == ["step1"]
    
    def test_independent_steps_one_group(self):
        """Independent steps can run in parallel."""
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", []),
            ("C", "agent", []),
        ])
        groups = get_parallel_execution_plan(workflow)
        
        # All steps should be in one group (can run in parallel)
        assert len(groups) == 1
        assert set(groups[0]) == {"A", "B", "C"}
    
    def test_linear_chain_sequential_groups(self):
        """Linear chain results in sequential groups."""
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", ["A"]),
            ("C", "agent", ["B"]),
        ])
        groups = get_parallel_execution_plan(workflow)
        
        # Each step in its own group
        assert len(groups) == 3
        assert groups[0] == ["A"]
        assert groups[1] == ["B"]
        assert groups[2] == ["C"]
    
    def test_diamond_groups(self):
        """
        Diamond pattern:
            A       <- Group 1
           / \\
          B   C     <- Group 2 (parallel)
           \\ /
            D       <- Group 3
        """
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", ["A"]),
            ("C", "agent", ["A"]),
            ("D", "agent", ["B", "C"]),
        ])
        groups = get_parallel_execution_plan(workflow)
        
        assert len(groups) == 3
        assert groups[0] == ["A"]
        assert set(groups[1]) == {"B", "C"}  # B and C in parallel
        assert groups[2] == ["D"]
    
    def test_complex_parallel_groups(self):
        """
        Test more complex parallel grouping:
            A   B   C      <- Group 1 (all parallel)
            |   |  /|
            D   E   F      <- Group 2 (D depends on A, E depends on B, F depends on C)
             \\ | /
               G           <- Group 3
        """
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", []),
            ("C", "agent", []),
            ("D", "agent", ["A"]),
            ("E", "agent", ["B"]),
            ("F", "agent", ["C"]),
            ("G", "agent", ["D", "E", "F"]),
        ])
        groups = get_parallel_execution_plan(workflow)
        
        assert len(groups) == 3
        assert set(groups[0]) == {"A", "B", "C"}
        assert set(groups[1]) == {"D", "E", "F"}
        assert groups[2] == ["G"]
    
    def test_groups_respect_partial_dependencies(self):
        """
        Test partial dependencies:
            A   B
            |   |
            C   |  <- C depends only on A
             \\ |
              D    <- D depends on B and C
        """
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", []),
            ("C", "agent", ["A"]),
            ("D", "agent", ["B", "C"]),
        ])
        groups = get_parallel_execution_plan(workflow)
        
        # A and B can run in parallel (group 1)
        # C can run after A, but D needs both B and C
        # This means: [A,B], [C], [D] OR [A,B], [C], [D] depending on interpretation
        # Actually: A,B in group 1, then C in group 2 (after A), then D in group 3
        # But B could theoretically run in group 1 OR group 2 since D only needs it
        # The algorithm puts all no-dep steps in group 1
        
        assert set(groups[0]) == {"A", "B"}
        assert groups[1] == ["C"]
        assert groups[2] == ["D"]


# =============================================================================
# Ready Steps Tests
# =============================================================================


class TestReadySteps:
    """Tests for get_ready_steps method."""
    
    def test_initial_ready_steps(self):
        """Test ready steps at start (no completed steps)."""
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", []),
            ("C", "agent", ["A", "B"]),
        ])
        dag = build_dag(workflow)
        
        ready = dag.get_ready_steps(completed=set())
        assert set(ready) == {"A", "B"}
    
    def test_ready_after_partial_completion(self):
        """Test ready steps after some steps complete."""
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", []),
            ("C", "agent", ["A", "B"]),
        ])
        dag = build_dag(workflow)
        
        # After A completes, B is still not done, so C not ready
        ready = dag.get_ready_steps(completed={"A"})
        assert ready == ["B"]  # Only B is ready
        
        # After both A and B complete, C is ready
        ready = dag.get_ready_steps(completed={"A", "B"})
        assert ready == ["C"]
    
    def test_ready_returns_empty_when_all_done(self):
        """Test ready returns empty when all steps completed."""
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", ["A"]),
        ])
        dag = build_dag(workflow)
        
        ready = dag.get_ready_steps(completed={"A", "B"})
        assert ready == []
    
    def test_ready_diamond_progression(self):
        """Test ready steps through diamond execution."""
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", ["A"]),
            ("C", "agent", ["A"]),
            ("D", "agent", ["B", "C"]),
        ])
        dag = build_dag(workflow)
        
        # Initially only A is ready
        assert dag.get_ready_steps(set()) == ["A"]
        
        # After A, B and C are ready
        assert set(dag.get_ready_steps({"A"})) == {"B", "C"}
        
        # After A and B, only C is ready (D needs C too)
        assert dag.get_ready_steps({"A", "B"}) == ["C"]
        
        # After A, B, C, D is ready
        assert dag.get_ready_steps({"A", "B", "C"}) == ["D"]


# =============================================================================
# Ancestor/Descendant Tests
# =============================================================================


class TestAncestorsDescendants:
    """Tests for ancestor and descendant calculation."""
    
    def test_get_all_ancestors_linear(self):
        """Test ancestors in linear chain."""
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", ["A"]),
            ("C", "agent", ["B"]),
            ("D", "agent", ["C"]),
        ])
        dag = build_dag(workflow)
        
        assert dag.get_all_ancestors("A") == set()
        assert dag.get_all_ancestors("B") == {"A"}
        assert dag.get_all_ancestors("C") == {"A", "B"}
        assert dag.get_all_ancestors("D") == {"A", "B", "C"}
    
    def test_get_all_ancestors_diamond(self):
        """Test ancestors in diamond pattern."""
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", ["A"]),
            ("C", "agent", ["A"]),
            ("D", "agent", ["B", "C"]),
        ])
        dag = build_dag(workflow)
        
        assert dag.get_all_ancestors("D") == {"A", "B", "C"}
    
    def test_get_all_descendants_linear(self):
        """Test descendants in linear chain."""
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", ["A"]),
            ("C", "agent", ["B"]),
            ("D", "agent", ["C"]),
        ])
        dag = build_dag(workflow)
        
        assert dag.get_all_descendants("A") == {"B", "C", "D"}
        assert dag.get_all_descendants("B") == {"C", "D"}
        assert dag.get_all_descendants("C") == {"D"}
        assert dag.get_all_descendants("D") == set()
    
    def test_get_affected_steps(self):
        """Test get_affected_steps for failure impact analysis."""
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", ["A"]),
            ("C", "agent", ["A"]),
            ("D", "agent", ["B"]),
            ("E", "agent", ["C"]),
        ])
        
        # If A fails, everything downstream is affected
        affected = get_affected_steps(workflow, "A")
        assert affected == {"B", "C", "D", "E"}
        
        # If B fails, only D is affected
        affected = get_affected_steps(workflow, "B")
        assert affected == {"D"}
        
        # If D fails, nothing is affected (it's a leaf)
        affected = get_affected_steps(workflow, "D")
        assert affected == set()


# =============================================================================
# DAG Validation Tests
# =============================================================================


class TestDAGValidation:
    """Tests for DAG validation method."""
    
    def test_valid_dag(self):
        """Test validation of valid DAG."""
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", ["A"]),
        ])
        dag = build_dag(workflow)
        
        errors = dag.validate()
        assert errors == []
    
    def test_validation_catches_cycle(self):
        """Test validation catches cycles."""
        workflow = make_workflow([
            ("A", "agent", []),
            ("B", "agent", ["A"]),
        ])
        dag = build_dag(workflow)
        
        # Manually add cycle
        dag.nodes["A"].dependencies.add("B")
        dag._topo_order = None
        
        errors = dag.validate()
        assert len(errors) >= 1
        assert any("Cyclic" in e or "cycle" in e.lower() for e in errors)


# =============================================================================
# Edge Cases and Stress Tests
# =============================================================================


class TestDAGEdgeCases:
    """Edge cases and stress tests for DAG."""
    
    def test_many_independent_steps(self):
        """Test DAG with many independent steps."""
        n = 50
        steps_config = [(f"step{i}", "agent", []) for i in range(n)]
        workflow = make_workflow(steps_config)
        
        dag = build_dag(workflow)
        groups = dag.get_parallel_groups()
        
        # All steps should be in one parallel group
        assert len(groups) == 1
        assert len(groups[0]) == n
    
    def test_long_linear_chain(self):
        """Test DAG with long linear chain."""
        n = 50
        steps_config = [(f"step{i}", "agent", [f"step{i-1}"] if i > 0 else []) 
                        for i in range(n)]
        workflow = make_workflow(steps_config)
        
        dag = build_dag(workflow)
        order = dag.get_topological_order()
        groups = dag.get_parallel_groups()
        
        # Should have n groups (no parallelism)
        assert len(groups) == n
        
        # Order should be step0, step1, step2, ...
        for i, step_id in enumerate(order):
            assert step_id == f"step{i}"
    
    def test_wide_then_narrow(self):
        """
        Test pattern:
            A B C D E F G H  <- 8 parallel steps
                  |
                  X          <- Single step depending on all
        """
        n = 8
        parallel_steps = [(f"p{i}", "agent", []) for i in range(n)]
        final_step = [("final", "agent", [f"p{i}" for i in range(n)])]
        
        workflow = make_workflow(parallel_steps + final_step)
        dag = build_dag(workflow)
        groups = dag.get_parallel_groups()
        
        assert len(groups) == 2
        assert len(groups[0]) == n
        assert groups[1] == ["final"]
