"""
AI Workflow Fabric - LLM Provider Examples

This example demonstrates how to use the various LLM providers
included in AWF for chat completions, streaming, and embeddings.

Requirements:
    pip install ai-workflow-fabric[providers]
    
    Set environment variables:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    - GOOGLE_API_KEY
    - MISTRAL_API_KEY
    
    For Ollama, ensure the server is running:
    ollama serve
"""

import asyncio
import os
from typing import Optional

# Import providers and types
from awf.providers import (
    Message,
    Role,
    ToolDefinition,
)


async def demo_openai():
    """Demonstrate OpenAI provider usage."""
    print("\n" + "=" * 60)
    print("OpenAI Provider Demo")
    print("=" * 60)
    
    from awf.providers import OpenAIProvider
    
    # Initialize provider (uses OPENAI_API_KEY env var by default)
    provider = OpenAIProvider()
    
    print(f"\nProvider: {provider.provider_name}")
    print(f"Default model: {provider.default_model}")
    print(f"Available models: {provider.get_available_models()[:5]}...")
    
    # Simple completion
    print("\n--- Simple Completion ---")
    response = await provider.complete([
        Message(role=Role.SYSTEM, content="You are a helpful assistant."),
        Message(role=Role.USER, content="What is 2 + 2? Answer in one word."),
    ])
    
    print(f"Response: {response.content}")
    print(f"Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
    print(f"Cost: ${response.usage.total_cost:.6f}" if response.usage else "Cost: N/A")
    
    # Streaming completion
    print("\n--- Streaming Completion ---")
    print("Response: ", end="", flush=True)
    async for chunk in provider.stream([
        Message(role=Role.USER, content="Count from 1 to 5."),
    ]):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()
    
    # Token counting
    print("\n--- Token Counting ---")
    text = "Hello, world! This is a test of token counting."
    tokens = provider.count_tokens(text)
    print(f"Text: '{text}'")
    print(f"Tokens: {tokens}")


async def demo_anthropic():
    """Demonstrate Anthropic provider usage."""
    print("\n" + "=" * 60)
    print("Anthropic Provider Demo")
    print("=" * 60)
    
    from awf.providers import AnthropicProvider
    
    provider = AnthropicProvider()
    
    print(f"\nProvider: {provider.provider_name}")
    print(f"Default model: {provider.default_model}")
    
    response = await provider.complete([
        Message(role=Role.USER, content="What is the capital of France? One word answer."),
    ])
    
    print(f"Response: {response.content}")
    print(f"Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
    print(f"Cost: ${response.usage.total_cost:.6f}" if response.usage else "Cost: N/A")


async def demo_google():
    """Demonstrate Google provider usage."""
    print("\n" + "=" * 60)
    print("Google (Gemini) Provider Demo")
    print("=" * 60)
    
    from awf.providers import GoogleProvider
    
    provider = GoogleProvider()
    
    print(f"\nProvider: {provider.provider_name}")
    print(f"Default model: {provider.default_model}")
    
    response = await provider.complete([
        Message(role=Role.USER, content="What is Python? Answer in one sentence."),
    ])
    
    print(f"Response: {response.content}")
    print(f"Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")


async def demo_mistral():
    """Demonstrate Mistral provider usage."""
    print("\n" + "=" * 60)
    print("Mistral Provider Demo")
    print("=" * 60)
    
    from awf.providers import MistralProvider
    
    provider = MistralProvider()
    
    print(f"\nProvider: {provider.provider_name}")
    print(f"Default model: {provider.default_model}")
    
    response = await provider.complete([
        Message(role=Role.USER, content="What is AI? Answer briefly."),
    ])
    
    print(f"Response: {response.content}")
    print(f"Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
    print(f"Cost: ${response.usage.total_cost:.6f}" if response.usage else "Cost: N/A")


async def demo_ollama():
    """Demonstrate Ollama provider usage (local models)."""
    print("\n" + "=" * 60)
    print("Ollama Provider Demo (Local)")
    print("=" * 60)
    
    from awf.providers import OllamaProvider
    
    provider = OllamaProvider()
    
    print(f"\nProvider: {provider.provider_name}")
    print(f"Default model: {provider.default_model}")
    print(f"Server URL: {provider.base_url}")
    
    try:
        # List available local models
        models = await provider.list_models()
        if models:
            print(f"Available local models: {[m.get('name', 'unknown') for m in models[:5]]}")
        
        response = await provider.complete([
            Message(role=Role.USER, content="Hello! Who are you?"),
        ], model="llama3.2")  # Use a model you have pulled
        
        print(f"Response: {response.content}")
        print(f"Tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
        print("Cost: $0.00 (local inference)")
    except Exception as e:
        print(f"Ollama not available: {e}")
        print("Start Ollama with: ollama serve")


async def demo_tools():
    """Demonstrate tool/function calling."""
    print("\n" + "=" * 60)
    print("Tool Calling Demo")
    print("=" * 60)
    
    from awf.providers import OpenAIProvider
    
    provider = OpenAIProvider()
    
    # Define a weather tool
    weather_tool = ToolDefinition(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g., 'San Francisco'",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    )
    
    print("\n--- Tool Calling ---")
    response = await provider.complete(
        messages=[
            Message(role=Role.USER, content="What's the weather in Tokyo?"),
        ],
        tools=[weather_tool],
        tool_choice="auto",
    )
    
    if response.tool_calls:
        print("Tool calls requested:")
        for tc in response.tool_calls:
            print(f"  - {tc.name}({tc.arguments})")
    else:
        print(f"Response: {response.content}")


async def demo_embeddings():
    """Demonstrate embedding generation."""
    print("\n" + "=" * 60)
    print("Embeddings Demo")
    print("=" * 60)
    
    from awf.providers import OpenAIProvider
    
    provider = OpenAIProvider()
    
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn fox leaps above a sleepy canine.",
        "Python is a programming language.",
    ]
    
    print(f"\nGenerating embeddings for {len(texts)} texts...")
    embeddings = await provider.embed(texts)
    
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        print(f"\nText {i+1}: '{text[:50]}...'")
        print(f"Embedding dim: {len(emb)}")
        print(f"First 5 values: {emb[:5]}")
    
    # Calculate similarity
    import math
    
    def cosine_similarity(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        return dot / (mag_a * mag_b) if mag_a and mag_b else 0
    
    sim_01 = cosine_similarity(embeddings[0], embeddings[1])
    sim_02 = cosine_similarity(embeddings[0], embeddings[2])
    
    print(f"\nSimilarity (text 1 vs 2): {sim_01:.4f} (semantically similar)")
    print(f"Similarity (text 1 vs 3): {sim_02:.4f} (different topics)")


async def demo_cost_estimation():
    """Demonstrate cost estimation."""
    print("\n" + "=" * 60)
    print("Cost Estimation Demo")
    print("=" * 60)
    
    from awf.providers import OpenAIProvider, AnthropicProvider
    
    # Compare costs across providers
    providers = [
        ("OpenAI (GPT-4o)", OpenAIProvider(), "gpt-4o"),
        ("OpenAI (GPT-4o-mini)", OpenAIProvider(), "gpt-4o-mini"),
        ("Anthropic (Claude 3.5 Sonnet)", AnthropicProvider(), "claude-3-5-sonnet-20241022"),
    ]
    
    # Simulate a typical request
    prompt_tokens = 1000
    completion_tokens = 500
    
    print(f"\nEstimated costs for {prompt_tokens} prompt + {completion_tokens} completion tokens:\n")
    
    for name, provider, model in providers:
        usage = provider.estimate_cost(prompt_tokens, completion_tokens, model)
        print(f"{name}:")
        print(f"  Prompt cost:     ${usage.prompt_cost:.6f}")
        print(f"  Completion cost: ${usage.completion_cost:.6f}")
        print(f"  Total cost:      ${usage.total_cost:.6f}")
        print()


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("AI Workflow Fabric - LLM Provider Examples")
    print("=" * 60)
    
    # Check for API keys
    providers_available = {
        "OpenAI": bool(os.getenv("OPENAI_API_KEY")),
        "Anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "Google": bool(os.getenv("GOOGLE_API_KEY")),
        "Mistral": bool(os.getenv("MISTRAL_API_KEY")),
        "Ollama": True,  # Local, no API key needed
    }
    
    print("\nProvider availability:")
    for name, available in providers_available.items():
        status = "[OK]" if available else "[Missing API Key]"
        print(f"  {name}: {status}")
    
    # Run demos for available providers
    try:
        if providers_available["OpenAI"]:
            await demo_openai()
            await demo_tools()
            await demo_embeddings()
    except Exception as e:
        print(f"OpenAI demo error: {e}")
    
    try:
        if providers_available["Anthropic"]:
            await demo_anthropic()
    except Exception as e:
        print(f"Anthropic demo error: {e}")
    
    try:
        if providers_available["Google"]:
            await demo_google()
    except Exception as e:
        print(f"Google demo error: {e}")
    
    try:
        if providers_available["Mistral"]:
            await demo_mistral()
    except Exception as e:
        print(f"Mistral demo error: {e}")
    
    try:
        await demo_ollama()
    except Exception as e:
        print(f"Ollama demo error: {e}")
    
    # Cost estimation doesn't need API calls
    await demo_cost_estimation()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
