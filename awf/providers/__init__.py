"""
AI Workflow Fabric - LLM Provider Package

This package provides unified interfaces for multiple LLM providers,
enabling agents to use any supported LLM through a consistent API.

Supported Providers:
- OpenAI (GPT-4, GPT-4o, o1, etc.)
- Anthropic (Claude 3.5, Claude 3)
- Google (Gemini Pro, Gemini Flash)
- Mistral (Large, Medium, Small)
- Ollama (Local LLMs)
- Antigravity (Claude/Gemini via Google OAuth - free!)

Example usage:
    from awf.providers import OpenAIProvider, Message, Role
    
    provider = OpenAIProvider(api_key="...")
    
    response = await provider.complete([
        Message(role=Role.USER, content="Hello!")
    ])
    
    print(response.content)

Antigravity Usage (free models via Google):
    from awf.providers import AntigravityProvider, Message, Role
    
    provider = AntigravityProvider()
    await provider.authenticate()  # Opens browser for Google sign-in
    
    response = await provider.complete([
        Message(role=Role.USER, content="Hello!")
    ], model="antigravity-claude-opus-4-5-thinking-high")
"""

from awf.providers.base import (
    LLMProvider,
    Message,
    Role,
    CompletionResponse,
    StreamChunk,
    ToolCall,
    ToolDefinition,
    Usage,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
)

__all__ = [
    # Base classes
    "LLMProvider",
    "Message",
    "Role",
    "CompletionResponse",
    "StreamChunk",
    "ToolCall",
    "ToolDefinition",
    "Usage",
    # Errors
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "InvalidRequestError",
    # Providers (lazy loaded)
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "MistralProvider",
    "OllamaProvider",
    "AntigravityProvider",
]

# Lazy imports for providers to avoid requiring all dependencies
def __getattr__(name: str):
    if name == "OpenAIProvider":
        from awf.providers.openai import OpenAIProvider
        return OpenAIProvider
    elif name == "AnthropicProvider":
        from awf.providers.anthropic import AnthropicProvider
        return AnthropicProvider
    elif name == "GoogleProvider":
        from awf.providers.google import GoogleProvider
        return GoogleProvider
    elif name == "MistralProvider":
        from awf.providers.mistral import MistralProvider
        return MistralProvider
    elif name == "OllamaProvider":
        from awf.providers.ollama import OllamaProvider
        return OllamaProvider
    elif name == "AntigravityProvider":
        from awf.providers.antigravity import AntigravityProvider
        return AntigravityProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
