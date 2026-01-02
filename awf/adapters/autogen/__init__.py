"""
AI Workflow Fabric - AutoGen Adapter Package

This package provides the adapter for integrating Microsoft AutoGen agents with AWF.
"""

from awf.adapters.autogen.adapter import (
    AutoGenAdapter,
    AUTOGEN_AVAILABLE,
)

__all__ = [
    "AutoGenAdapter",
    "AUTOGEN_AVAILABLE",
]
