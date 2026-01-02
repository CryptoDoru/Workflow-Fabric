"""
AI Workflow Fabric - Input/Output Mapping

This module provides JSONPath-based input and output mapping for workflows.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from awf.orchestration.types import ExecutionContext, StepResult


# =============================================================================
# JSONPath Parser (Simple Subset)
# =============================================================================


class JSONPathError(Exception):
    """Error parsing or evaluating JSONPath expression."""
    pass


class InputMapper:
    """
    Maps workflow data to step inputs using JSONPath-like expressions.
    
    Supported expressions:
    - $.input.X - Access workflow input field
    - $.steps.Y.output.Z - Access step Y's output field Z
    - $.steps.Y.status - Access step Y's status
    - $.steps.Y.metrics.X - Access step Y's metrics
    - $.context.X - Access execution context
    
    Supports nested paths like $.steps.Y.output.items[0].name
    """
    
    # Pattern to match JSONPath expressions
    PATH_PATTERN = re.compile(
        r'^\$\.(input|steps|context)\.(.+)$'
    )
    
    # Pattern to match array index
    ARRAY_INDEX_PATTERN = re.compile(r'(.+)\[(\d+)\]$')
    
    def __init__(self, context: ExecutionContext):
        """
        Initialize mapper with execution context.
        
        Args:
            context: The workflow execution context
        """
        self.context = context
    
    def map_inputs(
        self,
        input_map: Dict[str, str],
        strict: bool = False,
    ) -> Dict[str, Any]:
        """
        Map input expressions to actual values.
        
        Args:
            input_map: Dict mapping input names to JSONPath expressions
            strict: If True, raise error on missing paths; if False, use None
        
        Returns:
            Dict with resolved values
        
        Raises:
            JSONPathError: If expression is invalid or path not found (strict mode)
        """
        result: Dict[str, Any] = {}
        
        for key, expression in input_map.items():
            try:
                value = self.resolve(expression)
                result[key] = value
            except JSONPathError:
                if strict:
                    raise
                result[key] = None
        
        return result
    
    def resolve(self, expression: str) -> Any:
        """
        Resolve a JSONPath expression to a value.
        
        Args:
            expression: JSONPath expression (e.g., $.input.topic)
        
        Returns:
            The resolved value
        
        Raises:
            JSONPathError: If expression is invalid or path not found
        """
        # Handle literal values (not starting with $)
        if not expression.startswith('$'):
            return expression
        
        match = self.PATH_PATTERN.match(expression)
        if not match:
            raise JSONPathError(f"Invalid JSONPath expression: {expression}")
        
        root = match.group(1)
        path = match.group(2)
        
        # Get the root object
        if root == "input":
            obj = self.context.input
        elif root == "steps":
            obj = self._get_steps_data()
        elif root == "context":
            obj = self.context.context
        else:
            raise JSONPathError(f"Unknown root: {root}")
        
        # Navigate the path
        return self._navigate_path(obj, path, expression)
    
    def _get_steps_data(self) -> Dict[str, Any]:
        """Build a dict of step data for path resolution."""
        result: Dict[str, Any] = {}
        
        for step_id, step_result in self.context.step_results.items():
            result[step_id] = {
                "output": step_result.output or {},
                "status": step_result.status.value,
                "error": step_result.error,
                "metrics": {
                    "executionTimeMs": step_result.execution_time_ms,
                    "retryCount": step_result.retry_count,
                    "usedFallback": step_result.used_fallback,
                    "tokenUsage": step_result.token_usage,
                },
            }
        
        return result
    
    def _navigate_path(self, obj: Any, path: str, full_expression: str) -> Any:
        """Navigate a dot-separated path through an object."""
        parts = self._split_path(path)
        current = obj
        
        for part in parts:
            if current is None:
                raise JSONPathError(
                    f"Cannot navigate path '{full_expression}': "
                    f"encountered None at '{part}'"
                )
            
            # Check for array index
            index_match = self.ARRAY_INDEX_PATTERN.match(part)
            if index_match:
                key = index_match.group(1)
                index = int(index_match.group(2))
                
                # First navigate to the key
                if isinstance(current, dict):
                    current = current.get(key)
                else:
                    raise JSONPathError(
                        f"Cannot access key '{key}' on non-dict in '{full_expression}'"
                    )
                
                # Then access the index
                if current is None:
                    raise JSONPathError(
                        f"Cannot index into None in '{full_expression}'"
                    )
                if not isinstance(current, (list, tuple)):
                    raise JSONPathError(
                        f"Cannot index into non-list in '{full_expression}'"
                    )
                if index >= len(current):
                    raise JSONPathError(
                        f"Index {index} out of range in '{full_expression}'"
                    )
                current = current[index]
            else:
                # Simple key access
                if isinstance(current, dict):
                    if part not in current:
                        raise JSONPathError(
                            f"Key '{part}' not found in '{full_expression}'"
                        )
                    current = current[part]
                else:
                    raise JSONPathError(
                        f"Cannot access key '{part}' on non-dict in '{full_expression}'"
                    )
        
        return current
    
    def _split_path(self, path: str) -> List[str]:
        """
        Split a path into parts, handling dots inside brackets.
        
        E.g., "steps.research.output.items[0].name" -> 
              ["steps", "research", "output", "items[0]", "name"]
        """
        parts: List[str] = []
        current = ""
        in_bracket = False
        
        for char in path:
            if char == '[':
                in_bracket = True
                current += char
            elif char == ']':
                in_bracket = False
                current += char
            elif char == '.' and not in_bracket:
                if current:
                    parts.append(current)
                current = ""
            else:
                current += char
        
        if current:
            parts.append(current)
        
        return parts


# =============================================================================
# Output Mapper
# =============================================================================


class OutputMapper:
    """
    Maps step outputs to final workflow output.
    
    Uses the same JSONPath syntax as InputMapper.
    """
    
    def __init__(self, context: ExecutionContext):
        """Initialize with execution context."""
        self.context = context
        self.input_mapper = InputMapper(context)
    
    def map_output(
        self,
        output_map: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Map step outputs to workflow output.
        
        Args:
            output_map: Dict mapping output field names to JSONPath expressions
                        If None, returns the last step's output
        
        Returns:
            The mapped output dict
        """
        if output_map is None:
            # Return last step's output
            return self._get_last_step_output()
        
        return self.input_mapper.map_inputs(output_map, strict=False)
    
    def _get_last_step_output(self) -> Dict[str, Any]:
        """Get the output of the last completed step."""
        # Find steps with no dependents (leaf nodes)
        step_ids = {s.id for s in self.context.workflow.steps}
        has_dependent = set()
        
        for step in self.context.workflow.steps:
            for dep in step.depends_on:
                has_dependent.add(dep)
        
        leaf_steps = step_ids - has_dependent
        
        # If there are multiple leaf steps, we need the one that completed last
        # For simplicity, if there's exactly one leaf, use it
        if len(leaf_steps) == 1:
            leaf_id = next(iter(leaf_steps))
            result = self.context.step_results.get(leaf_id)
            if result and result.output:
                return result.output
        
        # Fallback: return the last step in definition order that completed
        for step in reversed(self.context.workflow.steps):
            result = self.context.step_results.get(step.id)
            if result and result.status.value == "completed" and result.output:
                return result.output
        
        return {}


# =============================================================================
# Expression Validator
# =============================================================================


class ExpressionValidator:
    """Validates JSONPath expressions without executing them."""
    
    PATH_PATTERN = re.compile(
        r'^\$\.(input|steps|context)\.([a-zA-Z_][a-zA-Z0-9_\.\[\]]*)+$'
    )
    
    @classmethod
    def validate(cls, expression: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a JSONPath expression.
        
        Args:
            expression: The expression to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not expression:
            return False, "Expression cannot be empty"
        
        if not expression.startswith('$'):
            # Literal value, always valid
            return True, None
        
        if not cls.PATH_PATTERN.match(expression):
            return False, f"Invalid JSONPath syntax: {expression}"
        
        return True, None
    
    @classmethod
    def validate_input_map(
        cls,
        input_map: Dict[str, str],
        available_steps: List[str],
    ) -> List[str]:
        """
        Validate all expressions in an input map.
        
        Args:
            input_map: The input map to validate
            available_steps: List of step IDs that are available (defined before)
        
        Returns:
            List of error messages (empty if valid)
        """
        errors: List[str] = []
        
        for key, expression in input_map.items():
            is_valid, error = cls.validate(expression)
            if not is_valid:
                errors.append(f"Input '{key}': {error}")
                continue
            
            # Check step references
            if expression.startswith("$.steps."):
                # Extract step ID
                parts = expression.split(".")
                if len(parts) >= 3:
                    step_id = parts[2]
                    # Handle array notation
                    if "[" in step_id:
                        step_id = step_id.split("[")[0]
                    
                    if step_id not in available_steps:
                        errors.append(
                            f"Input '{key}': references unknown or not-yet-executed "
                            f"step '{step_id}'"
                        )
        
        return errors
