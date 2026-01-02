"""
AI Workflow Fabric - Google Provider

This module provides the Google Gemini LLM provider implementation supporting:
- Gemini 2.0 Flash
- Gemini 1.5 Pro, Flash
- Gemini 1.0 Pro
- Multimodal (images, video, audio)
- Function calling
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
# Note: Pricing varies by context length
MODEL_PRICING = {
    # Gemini 2.0
    "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Free during preview
    # Gemini 1.5
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},  # <128K context
    "gemini-1.5-pro-latest": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},  # <128K context
    "gemini-1.5-flash-latest": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
    # Gemini 1.0
    "gemini-1.0-pro": {"input": 0.50, "output": 1.50},
    "gemini-pro": {"input": 0.50, "output": 1.50},
    # Embeddings
    "text-embedding-004": {"input": 0.00, "output": 0.00},  # Free
}


class GoogleProvider(LLMProvider):
    """
    Google Gemini LLM provider.
    
    Supports Gemini models with multimodal capabilities and function calling.
    
    Example:
        provider = GoogleProvider(api_key="...")
        
        response = await provider.complete([
            Message(role=Role.USER, content="Hello Gemini!"),
        ])
        
        print(response.content)
    """
    
    provider_name = "google"
    default_model = "gemini-1.5-flash"
    
    supports_streaming = True
    supports_tools = True
    supports_vision = True
    supports_embeddings = True
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize Google provider.
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            project_id: GCP project ID for Vertex AI
            location: GCP region for Vertex AI
            base_url: Override API base URL
            timeout: Request timeout
            max_retries: Max retry attempts
            retry_delay: Initial retry delay
        """
        super().__init__(
            api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self._client: Any = None
        self._use_vertex = project_id is not None
    
    def _get_client(self):
        """Get or create the Google AI client."""
        if self._client is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package is required for GoogleProvider. "
                    "Install with: pip install google-generativeai"
                )
            
            genai.configure(api_key=self.api_key)
            self._client = genai
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
        genai = self._get_client()
        model_name = model or self.default_model
        
        # Get model
        gen_model = genai.GenerativeModel(model_name)
        
        # Build generation config
        config: Dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens:
            config["max_output_tokens"] = max_tokens
        if stop:
            config["stop_sequences"] = stop
        
        # Convert messages to Gemini format
        contents = self._convert_messages(messages)
        
        # Build tool config if tools provided
        tool_config = None
        if tools:
            tool_config = self._build_tools(tools)
        
        try:
            # Use synchronous API wrapped in asyncio
            import asyncio
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: gen_model.generate_content(
                    contents,
                    generation_config=config,
                    tools=tool_config,
                )
            )
        except Exception as e:
            raise self._handle_error(e)
        
        # Parse response
        content = None
        tool_calls = None
        
        if response.candidates:
            candidate = response.candidates[0]
            for part in candidate.content.parts:
                if hasattr(part, 'text'):
                    content = part.text
                elif hasattr(part, 'function_call'):
                    if tool_calls is None:
                        tool_calls = []
                    fc = part.function_call
                    tool_calls.append(ToolCall(
                        id=f"call_{len(tool_calls)}",
                        name=fc.name,
                        arguments=dict(fc.args),
                    ))
        
        # Get finish reason
        finish_reason = FinishReason.STOP
        if response.candidates:
            reason = response.candidates[0].finish_reason
            finish_reason = self._map_finish_reason(reason)
        
        # Estimate usage
        usage = None
        if hasattr(response, 'usage_metadata'):
            usage = self.estimate_cost(
                response.usage_metadata.prompt_token_count,
                response.usage_metadata.candidates_token_count,
                model_name,
            )
        
        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=model_name,
            provider=self.provider_name,
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
        genai = self._get_client()
        model_name = model or self.default_model
        
        gen_model = genai.GenerativeModel(model_name)
        
        config: Dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens:
            config["max_output_tokens"] = max_tokens
        if stop:
            config["stop_sequences"] = stop
        
        contents = self._convert_messages(messages)
        
        tool_config = None
        if tools:
            tool_config = self._build_tools(tools)
        
        try:
            import asyncio
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: gen_model.generate_content(
                    contents,
                    generation_config=config,
                    tools=tool_config,
                    stream=True,
                )
            )
            
            accumulated = ""
            for chunk in response:
                if chunk.text:
                    accumulated += chunk.text
                    yield StreamChunk(
                        content=chunk.text,
                        accumulated_content=accumulated,
                    )
            
            yield StreamChunk(
                is_final=True,
                finish_reason=FinishReason.STOP,
                accumulated_content=accumulated,
            )
        except Exception as e:
            raise self._handle_error(e)
    
    async def embed(
        self,
        texts: List[str],
        *,
        model: Optional[str] = None,
        **kwargs,
    ) -> List[List[float]]:
        """Generate embeddings."""
        genai = self._get_client()
        model_name = model or "text-embedding-004"
        
        try:
            import asyncio
            results = []
            for text in texts:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda t=text: genai.embed_content(
                        model=f"models/{model_name}",
                        content=t,
                    )
                )
                results.append(result['embedding'])
            return results
        except Exception as e:
            raise self._handle_error(e)
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to Gemini format."""
        contents = []
        system_instruction = None
        
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content
                continue
            
            parts = []
            
            if msg.content:
                parts.append({"text": msg.content})
            
            if msg.images:
                for img in msg.images:
                    if img.startswith("http"):
                        # URL - would need to fetch
                        parts.append({"text": f"[Image: {img}]"})
                    else:
                        # Base64
                        import base64
                        parts.append({
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img,
                            }
                        })
            
            role = "user" if msg.role == Role.USER else "model"
            contents.append({
                "role": role,
                "parts": parts,
            })
        
        # Prepend system instruction if present
        if system_instruction and contents:
            contents[0]["parts"].insert(0, {"text": f"Instructions: {system_instruction}\n\n"})
        
        return contents
    
    def _build_tools(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Build Gemini tool configuration."""
        function_declarations = []
        for tool in tools:
            function_declarations.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
        return [{"function_declarations": function_declarations}]
    
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: Optional[str] = None,
    ) -> Usage:
        """Estimate cost based on Google pricing."""
        model = model or self.default_model
        
        pricing = None
        for model_name, prices in MODEL_PRICING.items():
            if model_name in model or model in model_name:
                pricing = prices
                break
        
        if pricing is None:
            pricing = {"input": 0.075, "output": 0.30}  # Default to flash
        
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
        """Get list of available Google models."""
        return list(MODEL_PRICING.keys())
    
    def _map_finish_reason(self, reason: Any) -> FinishReason:
        """Map Gemini finish reason to our enum."""
        if reason is None:
            return FinishReason.STOP
        
        reason_str = str(reason).lower()
        if "stop" in reason_str:
            return FinishReason.STOP
        elif "max_tokens" in reason_str or "length" in reason_str:
            return FinishReason.LENGTH
        elif "safety" in reason_str:
            return FinishReason.CONTENT_FILTER
        else:
            return FinishReason.STOP
    
    def _handle_error(self, error: Exception) -> ProviderError:
        """Convert Google errors to our error types."""
        error_str = str(error)
        
        if "quota" in error_str.lower() or "rate" in error_str.lower():
            return RateLimitError(
                error_str,
                provider=self.provider_name,
            )
        elif "api_key" in error_str.lower() or "authentication" in error_str.lower():
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
