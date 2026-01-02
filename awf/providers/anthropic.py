"""
AI Workflow Fabric - Anthropic Provider

This module provides the Anthropic LLM provider implementation supporting:
- Claude 3.5 Sonnet
- Claude 3 Opus, Sonnet, Haiku
- Extended context (200K tokens)
- Tool use
- Vision (multimodal)
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
    # Claude 3.5
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00},
    # Claude 3
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-opus-latest": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude LLM provider.
    
    Supports Claude 3.5 and Claude 3 models with tool use and vision.
    
    Example:
        provider = AnthropicProvider(api_key="sk-ant-...")
        
        response = await provider.complete([
            Message(role=Role.USER, content="Hello Claude!"),
        ])
        
        print(response.content)
    """
    
    provider_name = "anthropic"
    default_model = "claude-3-5-sonnet-latest"
    
    supports_streaming = True
    supports_tools = True
    supports_vision = True
    supports_embeddings = False  # Anthropic doesn't offer embeddings
    
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
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            base_url: Override API base URL
            timeout: Request timeout
            max_retries: Max retry attempts
            retry_delay: Initial retry delay
        """
        super().__init__(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self._client: Any = None
    
    def _get_client(self):
        """Get or create the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError(
                    "anthropic package is required for AnthropicProvider. "
                    "Install with: pip install anthropic"
                )
            
            kwargs: Dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            
            self._client = AsyncAnthropic(**kwargs)
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
        
        # Extract system message
        system_content = None
        chat_messages = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_content = msg.content
            else:
                chat_messages.append(msg.to_anthropic_format())
        
        # Build request
        request: Dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens or 4096,
        }
        
        if system_content:
            request["system"] = system_content
        
        if temperature != 1.0:  # Anthropic default is 1.0
            request["temperature"] = temperature
        
        if tools:
            request["tools"] = [t.to_anthropic_format() for t in tools]
            if tool_choice:
                if tool_choice == "auto":
                    request["tool_choice"] = {"type": "auto"}
                elif tool_choice == "none":
                    request["tool_choice"] = {"type": "none"}
                elif tool_choice == "required":
                    request["tool_choice"] = {"type": "any"}
                else:
                    request["tool_choice"] = {"type": "tool", "name": tool_choice}
        
        if stop:
            request["stop_sequences"] = stop
        
        request.update(kwargs)
        
        try:
            response = await self._retry_with_backoff(
                client.messages.create,
                **request,
            )
        except Exception as e:
            raise self._handle_error(e)
        
        # Parse response
        content = None
        tool_calls = None
        
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))
        
        finish_reason = self._map_finish_reason(response.stop_reason)
        
        usage = self.estimate_cost(
            response.usage.input_tokens,
            response.usage.output_tokens,
            model,
        )
        
        return CompletionResponse(
            content=content,
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
        
        # Extract system message
        system_content = None
        chat_messages = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_content = msg.content
            else:
                chat_messages.append(msg.to_anthropic_format())
        
        request: Dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens or 4096,
        }
        
        if system_content:
            request["system"] = system_content
        
        if temperature != 1.0:
            request["temperature"] = temperature
        
        if tools:
            request["tools"] = [t.to_anthropic_format() for t in tools]
            if tool_choice:
                if tool_choice == "auto":
                    request["tool_choice"] = {"type": "auto"}
                elif tool_choice == "none":
                    request["tool_choice"] = {"type": "none"}
                else:
                    request["tool_choice"] = {"type": "tool", "name": tool_choice}
        
        if stop:
            request["stop_sequences"] = stop
        
        request.update(kwargs)
        
        try:
            async with client.messages.stream(**request) as stream:
                accumulated = ""
                async for event in stream:
                    if hasattr(event, 'type'):
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, 'text'):
                                text = event.delta.text
                                accumulated += text
                                yield StreamChunk(
                                    content=text,
                                    accumulated_content=accumulated,
                                )
                        elif event.type == "message_stop":
                            yield StreamChunk(
                                is_final=True,
                                finish_reason=FinishReason.STOP,
                                accumulated_content=accumulated,
                            )
        except Exception as e:
            raise self._handle_error(e)
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: Optional[str] = None,
    ) -> Usage:
        """Estimate cost based on Anthropic pricing."""
        model = model or self.default_model
        
        # Find pricing
        pricing = None
        for model_name, prices in MODEL_PRICING.items():
            if model_name in model or model in model_name:
                pricing = prices
                break
        
        if pricing is None:
            # Default to sonnet pricing
            pricing = {"input": 3.00, "output": 15.00}
        
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
    
    def get_available_models(self) -> List[str]:
        """Get list of available Anthropic models."""
        return list(MODEL_PRICING.keys())
    
    def _map_finish_reason(self, reason: Optional[str]) -> FinishReason:
        """Map Anthropic stop reason to our enum."""
        if reason is None:
            return FinishReason.STOP
        mapping = {
            "end_turn": FinishReason.STOP,
            "stop_sequence": FinishReason.STOP,
            "max_tokens": FinishReason.LENGTH,
            "tool_use": FinishReason.TOOL_CALLS,
        }
        return mapping.get(reason, FinishReason.STOP)
    
    def _handle_error(self, error: Exception) -> ProviderError:
        """Convert Anthropic errors to our error types."""
        error_str = str(error)
        
        if "rate_limit" in error_str.lower() or "overloaded" in error_str.lower():
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
