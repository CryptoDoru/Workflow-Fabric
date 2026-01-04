"""
AI Workflow Fabric - LLM Provider Base Classes

This module defines the abstract base class and common types for all
LLM provider implementations.
"""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)


# =============================================================================
# Enumerations
# =============================================================================


class Role(str, Enum):
    """Message role in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Reason for completion finishing."""
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ToolDefinition:
    """Definition of a tool that the LLM can call."""
    
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class ToolCall:
    """A tool call made by the LLM."""
    
    id: str
    name: str
    arguments: Dict[str, Any]
    
    @classmethod
    def from_openai(cls, tool_call: Dict[str, Any]) -> ToolCall:
        """Create from OpenAI format."""
        import json
        return cls(
            id=tool_call["id"],
            name=tool_call["function"]["name"],
            arguments=json.loads(tool_call["function"]["arguments"]),
        )


@dataclass
class Message:
    """A message in a conversation."""
    
    role: Role
    content: Optional[str] = None
    name: Optional[str] = None  # For tool messages
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None  # For tool responses
    
    # Multimodal content
    images: Optional[List[str]] = None  # Base64 or URLs
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI message format."""
        msg: Dict[str, Any] = {"role": self.role.value}
        
        if self.content:
            if self.images:
                # Multimodal
                content = [{"type": "text", "text": self.content}]
                for img in self.images:
                    if img.startswith("http"):
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": img}
                        })
                    else:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                        })
                msg["content"] = content
            else:
                msg["content"] = self.content
        
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": str(tc.arguments),
                    }
                }
                for tc in self.tool_calls
            ]
        
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        
        if self.name:
            msg["name"] = self.name
        
        return msg
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic message format."""
        msg: Dict[str, Any] = {"role": self.role.value}
        
        if self.role == Role.SYSTEM:
            # Anthropic handles system messages separately
            return {"type": "system", "text": self.content or ""}
        
        if self.images:
            content = []
            for img in self.images:
                if img.startswith("http"):
                    content.append({
                        "type": "image",
                        "source": {"type": "url", "url": img}
                    })
                else:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img,
                        }
                    })
            if self.content:
                content.append({"type": "text", "text": self.content})
            msg["content"] = content
        else:
            msg["content"] = self.content or ""
        
        return msg


@dataclass
class Usage:
    """Token usage information."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Cost estimation (in USD)
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    total_cost: float = 0.0
    
    def __add__(self, other: Usage) -> Usage:
        """Add two usage objects."""
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            prompt_cost=self.prompt_cost + other.prompt_cost,
            completion_cost=self.completion_cost + other.completion_cost,
            total_cost=self.total_cost + other.total_cost,
        )


@dataclass
class CompletionResponse:
    """Response from a completion request."""
    
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: FinishReason = FinishReason.STOP
    usage: Optional[Usage] = None
    
    # Metadata
    model: Optional[str] = None
    provider: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Raw response for debugging
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class StreamChunk:
    """A chunk from a streaming response."""
    
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[FinishReason] = None
    
    # For accumulating
    is_final: bool = False
    accumulated_content: str = ""


# =============================================================================
# Exceptions
# =============================================================================


class ProviderError(Exception):
    """Base exception for provider errors."""
    
    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        status_code: Optional[int] = None,
        response: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.response = response


class RateLimitError(ProviderError):
    """Rate limit exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        retry_after: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class AuthenticationError(ProviderError):
    """Authentication failed."""
    pass


class InvalidRequestError(ProviderError):
    """Invalid request parameters."""
    pass


class ModelNotFoundError(ProviderError):
    """Requested model not found."""
    pass


# =============================================================================
# Abstract Base Class
# =============================================================================


class LLMProvider(abc.ABC):
    """
    Abstract base class for LLM providers.
    
    All provider implementations must inherit from this class and implement
    the required methods.
    
    Example implementation:
        class MyProvider(LLMProvider):
            provider_name = "my_provider"
            
            async def complete(self, messages, **kwargs):
                # Implementation
                pass
            
            async def stream(self, messages, **kwargs):
                # Implementation
                yield chunk
    """
    
    # Provider identification
    provider_name: str = "base"
    
    # Default model
    default_model: str = ""
    
    # Supported features
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    supports_embeddings: bool = False
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the provider.
        
        Args:
            api_key: API key for authentication
            base_url: Override base URL for API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    @abc.abstractmethod
    async def complete(
        self,
        messages: List[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[str] = None,  # "auto", "none", or tool name
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> CompletionResponse:
        """
        Generate a completion for the given messages.
        
        Args:
            messages: List of messages in the conversation
            model: Model to use (defaults to provider's default)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            tools: Available tools for the model
            tool_choice: How to handle tool selection
            stop: Stop sequences
            **kwargs: Provider-specific options
        
        Returns:
            CompletionResponse with generated content
        """
        pass
    
    @abc.abstractmethod
    async def stream(
        self,
        messages: List[Message],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[str] = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a completion for the given messages.
        
        Args:
            Same as complete()
        
        Yields:
            StreamChunk objects as they arrive
        """
        pass
    
    async def embed(
        self,
        texts: List[str],
        *,
        model: Optional[str] = None,
        **kwargs,
    ) -> List[List[float]]:
        """
        Generate embeddings for the given texts.
        
        Args:
            texts: Texts to embed
            model: Embedding model to use
        
        Returns:
            List of embedding vectors
        
        Raises:
            NotImplementedError if provider doesn't support embeddings
        """
        raise NotImplementedError(
            f"{self.provider_name} does not support embeddings"
        )
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in a text string.
        
        This is a rough estimate; providers may override with accurate counting.
        
        Args:
            text: Text to count tokens for
            model: Model to use for counting
        
        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token for English
        return len(text) // 4
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: Optional[str] = None,
    ) -> Usage:
        """
        Estimate cost for a request.
        
        Providers should override this with accurate pricing.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model used
        
        Returns:
            Usage object with cost estimates
        """
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
    
    async def _retry_with_backoff(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute a function with exponential backoff retry.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        
        Raises:
            Last exception if all retries fail
        """
        last_exception: Optional[Exception] = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except RateLimitError as e:
                last_exception = e
                if e.retry_after:
                    await asyncio.sleep(e.retry_after)
                else:
                    await asyncio.sleep(delay)
                delay *= 2
            except (ProviderError, Exception) as e:
                last_exception = e
                if attempt < self.max_retries:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise
        
        if last_exception:
            raise last_exception
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models for this provider.
        
        Returns:
            List of model identifiers
        """
        return []
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider_name})"
