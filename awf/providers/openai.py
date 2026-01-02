"""
AI Workflow Fabric - OpenAI Provider

This module provides the OpenAI LLM provider implementation supporting:
- GPT-4, GPT-4 Turbo, GPT-4o
- o1, o1-mini
- GPT-3.5 Turbo
- Embeddings (text-embedding-3-small/large)
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Dict, List, Optional

from awf.providers.base import (
    LLMProvider,
    Message,
    Role,
    CompletionResponse,
    StreamChunk,
    ToolCall,
    ToolDefinition,
    Usage,
    FinishReason,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    ModelNotFoundError,
)


# Model pricing (USD per 1M tokens) as of late 2024
MODEL_PRICING = {
    # GPT-4o
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # GPT-4 Turbo
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    # GPT-4
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-32k": {"input": 60.00, "output": 120.00},
    # GPT-3.5
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-16k": {"input": 3.00, "output": 4.00},
    # o1 models
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o1-preview": {"input": 15.00, "output": 60.00},
    # Embeddings
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
}


class OpenAIProvider(LLMProvider):
    """
    OpenAI LLM provider.
    
    Supports all GPT models including GPT-4, GPT-4o, o1, and embeddings.
    
    Example:
        provider = OpenAIProvider(api_key="sk-...")
        
        response = await provider.complete([
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Hello!"),
        ])
        
        print(response.content)
    """
    
    provider_name = "openai"
    default_model = "gpt-4o-mini"
    
    supports_streaming = True
    supports_tools = True
    supports_vision = True
    supports_embeddings = True
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            organization: OpenAI organization ID
            base_url: Override API base URL
            timeout: Request timeout
            max_retries: Max retry attempts
            retry_delay: Initial retry delay
        """
        super().__init__(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or "https://api.openai.com/v1",
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self.organization = organization or os.getenv("OPENAI_ORG_ID")
        self._client: Any = None
    
    def _get_client(self):
        """Get or create the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAIProvider. "
                    "Install with: pip install openai"
                )
            
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client
    
    async def complete(
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
    ) -> CompletionResponse:
        """Generate a completion."""
        client = self._get_client()
        model = model or self.default_model
        
        # Build request
        request: Dict[str, Any] = {
            "model": model,
            "messages": [m.to_openai_format() for m in messages],
            "temperature": temperature,
        }
        
        if max_tokens:
            request["max_tokens"] = max_tokens
        
        if tools:
            request["tools"] = [t.to_openai_format() for t in tools]
            if tool_choice:
                if tool_choice in ("auto", "none", "required"):
                    request["tool_choice"] = tool_choice
                else:
                    request["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tool_choice}
                    }
        
        if stop:
            request["stop"] = stop
        
        # Add any extra kwargs
        request.update(kwargs)
        
        try:
            response = await self._retry_with_backoff(
                client.chat.completions.create,
                **request,
            )
        except Exception as e:
            raise self._handle_error(e)
        
        # Parse response
        choice = response.choices[0]
        message = choice.message
        
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in message.tool_calls
            ]
        
        finish_reason = self._map_finish_reason(choice.finish_reason)
        
        usage = None
        if response.usage:
            usage = self.estimate_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                model,
            )
        
        return CompletionResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=model,
            provider=self.provider_name,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
        )
    
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
        """Stream a completion."""
        client = self._get_client()
        model = model or self.default_model
        
        request: Dict[str, Any] = {
            "model": model,
            "messages": [m.to_openai_format() for m in messages],
            "temperature": temperature,
            "stream": True,
        }
        
        if max_tokens:
            request["max_tokens"] = max_tokens
        
        if tools:
            request["tools"] = [t.to_openai_format() for t in tools]
            if tool_choice:
                if tool_choice in ("auto", "none", "required"):
                    request["tool_choice"] = tool_choice
                else:
                    request["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tool_choice}
                    }
        
        if stop:
            request["stop"] = stop
        
        request.update(kwargs)
        
        try:
            stream = await client.chat.completions.create(**request)
        except Exception as e:
            raise self._handle_error(e)
        
        accumulated = ""
        async for chunk in stream:
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason
            
            content = delta.content if delta.content else None
            if content:
                accumulated += content
            
            yield StreamChunk(
                content=content,
                finish_reason=self._map_finish_reason(finish_reason) if finish_reason else None,
                is_final=finish_reason is not None,
                accumulated_content=accumulated,
            )
    
    async def embed(
        self,
        texts: List[str],
        *,
        model: Optional[str] = None,
        **kwargs,
    ) -> List[List[float]]:
        """Generate embeddings."""
        client = self._get_client()
        model = model or "text-embedding-3-small"
        
        try:
            response = await client.embeddings.create(
                model=model,
                input=texts,
                **kwargs,
            )
        except Exception as e:
            raise self._handle_error(e)
        
        return [item.embedding for item in response.data]
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: Optional[str] = None,
    ) -> Usage:
        """Estimate cost based on OpenAI pricing."""
        model = model or self.default_model
        
        # Find pricing (use base model name)
        pricing = None
        for model_name, prices in MODEL_PRICING.items():
            if model.startswith(model_name) or model_name in model:
                pricing = prices
                break
        
        if pricing is None:
            pricing = {"input": 0.0, "output": 0.0}
        
        prompt_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        completion_cost = (completion_tokens / 1_000_000) * pricing["output"]
        
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            total_cost=prompt_cost + completion_cost,
        )
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens using tiktoken if available."""
        try:
            import tiktoken
            model = model or self.default_model
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            # Fallback to rough estimate
            return super().count_tokens(text, model)
    
    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models."""
        return list(MODEL_PRICING.keys())
    
    def _map_finish_reason(self, reason: Optional[str]) -> FinishReason:
        """Map OpenAI finish reason to our enum."""
        if reason is None:
            return FinishReason.STOP
        mapping = {
            "stop": FinishReason.STOP,
            "length": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALLS,
            "content_filter": FinishReason.CONTENT_FILTER,
        }
        return mapping.get(reason, FinishReason.STOP)
    
    def _handle_error(self, error: Exception) -> ProviderError:
        """Convert OpenAI errors to our error types."""
        error_str = str(error)
        
        if "rate_limit" in error_str.lower():
            return RateLimitError(
                error_str,
                provider=self.provider_name,
            )
        elif "authentication" in error_str.lower() or "api_key" in error_str.lower():
            return AuthenticationError(
                error_str,
                provider=self.provider_name,
            )
        elif "model" in error_str.lower() and "not found" in error_str.lower():
            return ModelNotFoundError(
                error_str,
                provider=self.provider_name,
            )
        elif "invalid" in error_str.lower():
            return InvalidRequestError(
                error_str,
                provider=self.provider_name,
            )
        else:
            return ProviderError(
                error_str,
                provider=self.provider_name,
            )
