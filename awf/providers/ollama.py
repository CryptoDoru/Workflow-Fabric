"""
AI Workflow Fabric - Ollama Provider

This module provides the Ollama LLM provider implementation for running
local LLMs. Supports any model available through Ollama.

Common models:
- llama3.2, llama3.1, llama2
- codellama
- mistral, mixtral
- phi3
- qwen2.5
- deepseek-coder
- nomic-embed-text (embeddings)
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
    InvalidRequestError,
    ModelNotFoundError,
)


class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider for local model inference.
    
    Ollama must be running locally (default: http://localhost:11434).
    Models must be pulled first with `ollama pull <model>`.
    
    Example:
        provider = OllamaProvider()  # Uses default localhost URL
        
        response = await provider.complete([
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Hello!"),
        ], model="llama3.2")
        
        print(response.content)
    
    Note: Ollama runs locally, so there's no API key and no per-token cost.
    """
    
    provider_name = "ollama"
    default_model = "llama3.2"
    
    supports_streaming = True
    supports_tools = True  # Ollama supports tools with compatible models
    supports_vision = True  # LLaVA and other vision models
    supports_embeddings = True
    
    def __init__(
        self,
        api_key: Optional[str] = None,  # Not used, kept for interface compatibility
        *,
        base_url: Optional[str] = None,
        timeout: float = 120.0,  # Longer timeout for local inference
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize Ollama provider.
        
        Args:
            api_key: Ignored (Ollama doesn't require authentication)
            base_url: Ollama server URL (defaults to OLLAMA_HOST or localhost:11434)
            timeout: Request timeout (default 120s for slower local inference)
            max_retries: Max retry attempts
            retry_delay: Initial retry delay
        """
        super().__init__(
            api_key=None,  # Ollama doesn't use API keys
            base_url=base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self._client: Any = None
    
    def _get_client(self):
        """Get or create the Ollama client."""
        if self._client is None:
            try:
                from ollama import AsyncClient
            except ImportError:
                raise ImportError(
                    "ollama package is required for OllamaProvider. "
                    "Install with: pip install ollama"
                )
            
            self._client = AsyncClient(host=self.base_url)
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
        
        # Convert messages to Ollama format
        ollama_messages = self._convert_messages(messages)
        
        # Build options
        options: Dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens:
            options["num_predict"] = max_tokens
        if stop:
            options["stop"] = stop
        
        # Build request
        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "options": options,
            "stream": False,
        }
        
        if tools:
            request_kwargs["tools"] = [self._convert_tool(t) for t in tools]
        
        request_kwargs.update(kwargs)
        
        try:
            response = await self._retry_with_backoff(
                client.chat,
                **request_kwargs,
            )
        except Exception as e:
            raise self._handle_error(e)
        
        # Parse response
        message = response.get("message", {})
        content = message.get("content", "")
        
        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    id=f"call_{i}",
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"] if isinstance(tc["function"]["arguments"], dict) else json.loads(tc["function"]["arguments"]),
                )
                for i, tc in enumerate(message["tool_calls"])
            ]
        
        # Determine finish reason
        finish_reason = FinishReason.STOP
        if tool_calls:
            finish_reason = FinishReason.TOOL_CALLS
        elif response.get("done_reason") == "length":
            finish_reason = FinishReason.LENGTH
        
        # Calculate usage (Ollama provides token counts)
        usage = Usage(
            prompt_tokens=response.get("prompt_eval_count", 0),
            completion_tokens=response.get("eval_count", 0),
            total_tokens=response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
            # Local inference has no cost
            prompt_cost=0.0,
            completion_cost=0.0,
            total_cost=0.0,
        )
        
        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=model,
            provider=self.provider_name,
            raw_response=response,
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
        
        ollama_messages = self._convert_messages(messages)
        
        options: Dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens:
            options["num_predict"] = max_tokens
        if stop:
            options["stop"] = stop
        
        request_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "options": options,
            "stream": True,
        }
        
        if tools:
            request_kwargs["tools"] = [self._convert_tool(t) for t in tools]
        
        request_kwargs.update(kwargs)
        
        try:
            stream = await client.chat(**request_kwargs)
        except Exception as e:
            raise self._handle_error(e)
        
        accumulated = ""
        async for chunk in stream:
            message = chunk.get("message", {})
            content = message.get("content", "")
            is_done = chunk.get("done", False)
            
            if content:
                accumulated += content
            
            finish_reason = None
            if is_done:
                done_reason = chunk.get("done_reason")
                if done_reason == "length":
                    finish_reason = FinishReason.LENGTH
                else:
                    finish_reason = FinishReason.STOP
            
            yield StreamChunk(
                content=content if content else None,
                finish_reason=finish_reason,
                is_final=is_done,
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
        model = model or "nomic-embed-text"
        
        embeddings = []
        for text in texts:
            try:
                response = await client.embeddings(
                    model=model,
                    prompt=text,
                )
                embeddings.append(response["embedding"])
            except Exception as e:
                raise self._handle_error(e)
        
        return embeddings
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List locally available models.
        
        Returns:
            List of model info dicts with name, size, etc.
        """
        client = self._get_client()
        try:
            response = await client.list()
            return response.get("models", [])
        except Exception as e:
            raise self._handle_error(e)
    
    async def pull_model(self, model: str) -> None:
        """
        Pull a model from Ollama registry.
        
        Args:
            model: Model name to pull (e.g., "llama3.2")
        """
        client = self._get_client()
        try:
            await client.pull(model)
        except Exception as e:
            raise self._handle_error(e)
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: Optional[str] = None,
    ) -> Usage:
        """
        Estimate cost - always zero for local inference.
        
        Ollama runs locally, so there's no per-token cost.
        """
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_cost=0.0,
            completion_cost=0.0,
            total_cost=0.0,
        )
    
    def get_available_models(self) -> List[str]:
        """
        Get list of common Ollama models.
        
        Note: Actual availability depends on what's pulled locally.
        Use list_models() to see what's actually available.
        """
        return [
            "llama3.2",
            "llama3.1",
            "llama2",
            "codellama",
            "mistral",
            "mixtral",
            "phi3",
            "qwen2.5",
            "deepseek-coder",
            "gemma2",
            "llava",  # Vision model
            "nomic-embed-text",  # Embeddings
        ]
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert our messages to Ollama format."""
        result = []
        for msg in messages:
            converted: Dict[str, Any] = {
                "role": msg.role.value,
                "content": msg.content or "",
            }
            
            # Handle images for vision models
            if msg.images:
                converted["images"] = msg.images
            
            # Handle tool calls
            if msg.tool_calls:
                converted["tool_calls"] = [
                    {
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    }
                    for tc in msg.tool_calls
                ]
            
            result.append(converted)
        return result
    
    def _convert_tool(self, tool: ToolDefinition) -> Dict[str, Any]:
        """Convert tool definition to Ollama format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        }
    
    def _handle_error(self, error: Exception) -> ProviderError:
        """Convert Ollama errors to our error types."""
        error_str = str(error)
        
        if "model" in error_str.lower() and ("not found" in error_str.lower() or "does not exist" in error_str.lower()):
            return ModelNotFoundError(
                f"{error_str}. Run 'ollama pull <model>' to download it.",
                provider=self.provider_name,
            )
        elif "connection" in error_str.lower() or "refused" in error_str.lower():
            return ProviderError(
                f"Cannot connect to Ollama server at {self.base_url}. "
                "Make sure Ollama is running with 'ollama serve'.",
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
