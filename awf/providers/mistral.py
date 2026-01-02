"""
AI Workflow Fabric - Mistral AI Provider

This module provides the Mistral AI LLM provider implementation supporting:
- Mistral Large (mistral-large-latest)
- Mistral Medium (mistral-medium-latest)
- Mistral Small (mistral-small-latest)
- Codestral (codestral-latest)
- Mixtral 8x7B (open-mixtral-8x7b)
- Mistral 7B (open-mistral-7b)
- Embeddings (mistral-embed)
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
    # Mistral Large
    "mistral-large-latest": {"input": 3.00, "output": 9.00},
    "mistral-large-2411": {"input": 3.00, "output": 9.00},
    # Mistral Small
    "mistral-small-latest": {"input": 0.20, "output": 0.60},
    "mistral-small-2409": {"input": 0.20, "output": 0.60},
    # Codestral
    "codestral-latest": {"input": 0.30, "output": 0.90},
    "codestral-2405": {"input": 0.30, "output": 0.90},
    # Open models (via Mistral API)
    "open-mixtral-8x7b": {"input": 0.70, "output": 0.70},
    "open-mixtral-8x22b": {"input": 2.00, "output": 6.00},
    "open-mistral-7b": {"input": 0.25, "output": 0.25},
    "open-mistral-nemo": {"input": 0.15, "output": 0.15},
    # Pixtral (multimodal)
    "pixtral-large-latest": {"input": 3.00, "output": 9.00},
    "pixtral-12b-2409": {"input": 0.15, "output": 0.15},
    # Embeddings
    "mistral-embed": {"input": 0.10, "output": 0.0},
}


class MistralProvider(LLMProvider):
    """
    Mistral AI LLM provider.
    
    Supports all Mistral models including Large, Small, Codestral, and embeddings.
    
    Example:
        provider = MistralProvider(api_key="...")
        
        response = await provider.complete([
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Hello!"),
        ])
        
        print(response.content)
    """
    
    provider_name = "mistral"
    default_model = "mistral-small-latest"
    
    supports_streaming = True
    supports_tools = True
    supports_vision = True  # Pixtral models
    supports_embeddings = True
    
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
        Initialize Mistral provider.
        
        Args:
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env var)
            base_url: Override API base URL
            timeout: Request timeout
            max_retries: Max retry attempts
            retry_delay: Initial retry delay
        """
        super().__init__(
            api_key=api_key or os.getenv("MISTRAL_API_KEY"),
            base_url=base_url or "https://api.mistral.ai/v1",
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self._client: Any = None
    
    def _get_client(self):
        """Get or create the Mistral client."""
        if self._client is None:
            try:
                from mistralai import Mistral
            except ImportError:
                raise ImportError(
                    "mistralai package is required for MistralProvider. "
                    "Install with: pip install mistralai"
                )
            
            self._client = Mistral(
                api_key=self.api_key,
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
        
        # Convert messages to Mistral format
        mistral_messages = self._convert_messages(messages)
        
        # Build request kwargs
        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": mistral_messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            request_kwargs["max_tokens"] = max_tokens
        
        if tools:
            request_kwargs["tools"] = [self._convert_tool(t) for t in tools]
            if tool_choice:
                request_kwargs["tool_choice"] = tool_choice
        
        if stop:
            request_kwargs["stop"] = stop
        
        request_kwargs.update(kwargs)
        
        try:
            response = await self._retry_with_backoff(
                client.chat.complete_async,
                **request_kwargs,
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
                    arguments=json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments,
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
        
        mistral_messages = self._convert_messages(messages)
        
        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": mistral_messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            request_kwargs["max_tokens"] = max_tokens
        
        if tools:
            request_kwargs["tools"] = [self._convert_tool(t) for t in tools]
            if tool_choice:
                request_kwargs["tool_choice"] = tool_choice
        
        if stop:
            request_kwargs["stop"] = stop
        
        request_kwargs.update(kwargs)
        
        try:
            stream = await client.chat.stream_async(**request_kwargs)
        except Exception as e:
            raise self._handle_error(e)
        
        accumulated = ""
        async for event in stream:
            if not event.data or not event.data.choices:
                continue
            
            choice = event.data.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason
            
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
        model = model or "mistral-embed"
        
        try:
            response = await client.embeddings.create_async(
                model=model,
                inputs=texts,
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
        """Estimate cost based on Mistral pricing."""
        model = model or self.default_model
        
        # Find pricing
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
    
    def get_available_models(self) -> List[str]:
        """Get list of available Mistral models."""
        return list(MODEL_PRICING.keys())
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert our messages to Mistral format."""
        result = []
        for msg in messages:
            converted: Dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content or "",
            }
            
            if msg.tool_call_id:
                converted["tool_call_id"] = msg.tool_call_id
            
            if msg.tool_calls:
                converted["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        }
                    }
                    for tc in msg.tool_calls
                ]
            
            result.append(converted)
        return result
    
    def _convert_tool(self, tool: ToolDefinition) -> Dict[str, Any]:
        """Convert tool definition to Mistral format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        }
    
    def _map_finish_reason(self, reason: Optional[str]) -> FinishReason:
        """Map Mistral finish reason to our enum."""
        if reason is None:
            return FinishReason.STOP
        mapping = {
            "stop": FinishReason.STOP,
            "length": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALLS,
            "model_length": FinishReason.LENGTH,
        }
        return mapping.get(reason, FinishReason.STOP)
    
    def _handle_error(self, error: Exception) -> ProviderError:
        """Convert Mistral errors to our error types."""
        error_str = str(error)
        
        if "rate" in error_str.lower() and "limit" in error_str.lower():
            return RateLimitError(
                error_str,
                provider=self.provider_name,
            )
        elif "authentication" in error_str.lower() or "api_key" in error_str.lower() or "unauthorized" in error_str.lower():
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
