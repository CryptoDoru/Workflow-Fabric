# Contributing to AI Workflow Fabric

Thank you for your interest in contributing to AI Workflow Fabric! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- **Be Respectful**: Treat everyone with respect. No harassment, discrimination, or abusive behavior.
- **Be Constructive**: Provide constructive feedback. We're all here to learn and improve.
- **Be Collaborative**: Work together. Help others when you can.
- **Be Inclusive**: Welcome newcomers and help them get started.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A GitHub account

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Workflow-Fabric.git
   cd Workflow-Fabric
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/CryptoDoru/Workflow-Fabric.git
   ```

## Development Setup

### Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Dependencies

```bash
# Install in development mode with all dev dependencies
pip install -e ".[dev]"

# If you're working on a specific adapter, install those deps too
pip install -e ".[dev,langgraph]"  # For LangGraph adapter work
```

### Set Up Pre-commit Hooks

```bash
pre-commit install
```

This ensures code quality checks run before each commit.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/CryptoDoru/Workflow-Fabric/issues)
2. If not, create a new issue with:
   - A clear, descriptive title
   - Steps to reproduce the issue
   - Expected behavior
   - Actual behavior
   - Your environment (OS, Python version, AWF version)
   - Any relevant logs or screenshots

### Suggesting Features

1. Check if the feature has already been suggested in [Issues](https://github.com/CryptoDoru/Workflow-Fabric/issues)
2. If not, create a new issue with:
   - A clear, descriptive title
   - The problem this feature would solve
   - Your proposed solution
   - Any alternatives you've considered

### Contributing Code

1. Find an issue to work on or create one
2. Comment on the issue to let others know you're working on it
3. Create a branch for your work
4. Make your changes
5. Submit a pull request

## Pull Request Process

### Branch Naming

Use descriptive branch names:
- `feature/agent-discovery-api`
- `fix/langgraph-timeout-handling`
- `docs/contributing-guide`
- `refactor/trust-scoring-algorithm`

### Commit Messages

Write clear, concise commit messages:

```
type(scope): brief description

Longer description if needed. Explain the "why" not just the "what".

Closes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Before Submitting

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**:
   ```bash
   pytest
   ```

3. **Run linting**:
   ```bash
   ruff check .
   black --check .
   mypy awf
   ```

4. **Update documentation** if you've changed APIs or added features

### Pull Request Template

When creating a PR, include:

- **What**: Brief description of changes
- **Why**: Motivation and context
- **How**: Implementation approach
- **Testing**: How you tested the changes
- **Checklist**:
  - [ ] Tests pass
  - [ ] Linting passes
  - [ ] Documentation updated
  - [ ] Changelog updated (if applicable)

### Review Process

1. A maintainer will review your PR
2. Address any feedback
3. Once approved, a maintainer will merge your PR

## Coding Standards

### Python Style

We follow PEP 8 with some modifications:

- Line length: 100 characters
- Use type hints for all function signatures
- Use docstrings for all public functions and classes

### Code Formatting

We use:
- **Black** for code formatting
- **Ruff** for linting
- **mypy** for type checking

Run all checks:
```bash
black awf tests
ruff check awf tests
mypy awf
```

### Docstrings

Use Google-style docstrings:

```python
def register_agent(
    self,
    agent: T,
    agent_id: Optional[str] = None,
) -> AgentManifest:
    """Register an agent with the AWF registry.
    
    Args:
        agent: The native agent object to register
        agent_id: Optional custom identifier. If not provided,
            one will be generated.
    
    Returns:
        The generated AgentManifest for the registered agent.
    
    Raises:
        ValidationError: If the agent is invalid.
        RegistrationError: If registration fails.
    
    Example:
        >>> adapter = LangGraphAdapter()
        >>> manifest = adapter.register(my_graph)
        >>> print(manifest.id)
        'langgraph-abc123'
    """
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=awf --cov-report=html

# Run specific tests
pytest tests/test_core.py
pytest tests/test_langgraph.py::test_registration

# Run only fast tests
pytest -m "not slow"
```

### Writing Tests

- Place tests in `tests/` directory
- Mirror the source structure: `awf/core/types.py` → `tests/test_core_types.py`
- Use descriptive test names: `test_register_agent_with_valid_graph_succeeds`
- Include both positive and negative test cases
- Use fixtures for common setup

Example test:

```python
import pytest
from awf.core import Task, TaskStatus

class TestTask:
    def test_create_task_with_required_fields(self):
        task = Task(agent_id="test-agent", input={"query": "test"})
        
        assert task.agent_id == "test-agent"
        assert task.input == {"query": "test"}
        assert task.id is not None  # Auto-generated
    
    def test_create_task_with_custom_id(self):
        task = Task(
            agent_id="test-agent",
            input={"query": "test"},
            id="custom-id",
        )
        
        assert task.id == "custom-id"
    
    def test_task_to_dict_format(self):
        task = Task(agent_id="test-agent", input={"query": "test"})
        data = task.to_dict()
        
        assert "agentId" in data  # camelCase in JSON
        assert data["agentId"] == "test-agent"
```

## Documentation

### Where to Document

- **README.md**: High-level overview and quick start
- **docs/**: Detailed documentation (user guides, API reference)
- **Docstrings**: Inline API documentation
- **spec/**: Protocol specifications

### Documentation Style

- Use clear, concise language
- Include code examples
- Explain the "why" not just the "what"
- Keep it up to date with code changes

## Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and community discussions
- **Discord**: (Coming soon) For real-time chat

### Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- GitHub contributor graphs

## Thank You!

Every contribution, no matter how small, helps make AI Workflow Fabric better. Thank you for being part of our community!
