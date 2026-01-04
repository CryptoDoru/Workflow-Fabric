"""
AI Workflow Fabric - Antigravity Provider

Provides access to Claude and Gemini models through Google's Antigravity API.
Uses Google OAuth with PKCE for authentication.

This enables free access to models like:
- Claude Opus 4.5 (with thinking)
- Claude Sonnet 4.5 (with thinking)  
- Gemini 3 Pro (low/high thinking)
- Gemini 3 Flash

Reference: https://github.com/NoeFabris/opencode-antigravity-auth
"""

from __future__ import annotations

import asyncio
import hashlib
import base64
import secrets
import json
import os
import re
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urlparse, parse_qs
import threading

import httpx

from awf.providers.base import (
    LLMProvider,
    Message,
    Role,
    CompletionResponse,
    StreamChunk,
    ToolDefinition,
    ToolCall,
    Usage,
    FinishReason,
)


# =============================================================================
# Constants
# =============================================================================


OAUTH_CONFIG = {
    "client_id": "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com",
    "client_secret": "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",
    "callback_port": 51121,
    "callback_path": "/oauth-callback",
    "auth_endpoint": "https://accounts.google.com/o/oauth2/auth",
    "token_endpoint": "https://oauth2.googleapis.com/token",
    "scopes": [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/cclog",
        "https://www.googleapis.com/auth/experimentsandconfigs",
    ],
}

ANTIGRAVITY_ENDPOINTS = {
    "daily": "https://daily-cloudcode-pa.sandbox.googleapis.com",
    "autopush": "https://autopush-cloudcode-pa.sandbox.googleapis.com",
    "production": "https://cloudcode-pa.googleapis.com",
    "stream_generate": "/v1internal:streamGenerateContent?alt=sse",
    "generate": "/v1internal:generateContent",
    "load_code_assist": "/v1internal:loadCodeAssist",
}

ANTIGRAVITY_HEADERS = {
    "User-Agent": "antigravity/1.11.5 windows/amd64",
    "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "Client-Metadata": '{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}',
}

# Model mappings from user-friendly names to Antigravity model IDs
MODEL_MAPPINGS = {
    # Claude models
    "antigravity-claude-sonnet-4-5": "claude-sonnet-4-5",
    "antigravity-claude-sonnet-4-5-thinking-low": "claude-sonnet-4-5-thinking-low",
    "antigravity-claude-sonnet-4-5-thinking-medium": "claude-sonnet-4-5-thinking-medium",
    "antigravity-claude-sonnet-4-5-thinking-high": "claude-sonnet-4-5-thinking-high",
    "antigravity-claude-opus-4-5-thinking-low": "claude-opus-4-5-thinking-low",
    "antigravity-claude-opus-4-5-thinking-medium": "claude-opus-4-5-thinking-medium",
    "antigravity-claude-opus-4-5-thinking-high": "claude-opus-4-5-thinking-high",
    # Gemini models
    "antigravity-gemini-3-flash": "gemini-3-flash",
    "antigravity-gemini-3-pro-low": "gemini-3-pro-low",
    "antigravity-gemini-3-pro-high": "gemini-3-pro-high",
}

THINKING_BUDGETS = {
    "low": 8192,
    "medium": 16384,
    "high": 32768,
}


# =============================================================================
# PKCE Utilities
# =============================================================================


def generate_pkce_params() -> Tuple[str, str, str]:
    """Generate PKCE code verifier, challenge, and state."""
    # Generate code verifier (43-128 chars, URL-safe base64)
    code_verifier = secrets.token_urlsafe(64)[:128]
    
    # Generate code challenge (SHA256 of verifier, base64url encoded)
    digest = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    
    # Generate state
    state = secrets.token_urlsafe(32)
    
    return code_verifier, code_challenge, state


# =============================================================================
# OAuth Callback Server
# =============================================================================


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OAuth callback."""
    
    def log_message(self, format: str, *args) -> None:
        """Suppress default logging."""
        pass
    
    def do_GET(self) -> None:
        """Handle GET request (OAuth callback)."""
        parsed = urlparse(self.path)
        
        if parsed.path == OAUTH_CONFIG["callback_path"]:
            query = parse_qs(parsed.query)
            
            code = query.get("code", [None])[0]
            state = query.get("state", [None])[0]
            error = query.get("error", [None])[0]
            
            if error:
                self.server.oauth_result = {"error": error}  # type: ignore
            elif code and state:
                self.server.oauth_result = {"code": code, "state": state}  # type: ignore
            else:
                self.server.oauth_result = {"error": "missing_params"}  # type: ignore
            
            # Send success response
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            
            html = """
            <!DOCTYPE html>
            <html>
            <head><title>Authentication Complete</title></head>
            <body style="font-family: system-ui; text-align: center; padding: 50px;">
                <h1>Authentication Successful!</h1>
                <p>You can close this window and return to the application.</p>
                <script>window.close();</script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()


# =============================================================================
# Token Storage
# =============================================================================


@dataclass
class TokenInfo:
    """Stored OAuth tokens."""
    access_token: str
    refresh_token: str
    expires_at: datetime
    project_id: str
    email: Optional[str] = None


def get_token_path() -> str:
    """Get path for token storage."""
    config_dir = os.path.expanduser("~/.config/awf")
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "antigravity-tokens.json")


def load_tokens() -> Optional[TokenInfo]:
    """Load tokens from disk."""
    path = get_token_path()
    if not os.path.exists(path):
        return None
    
    try:
        with open(path, "r") as f:
            data = json.load(f)
        
        return TokenInfo(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=datetime.fromisoformat(data["expires_at"]),
            project_id=data["project_id"],
            email=data.get("email"),
        )
    except Exception:
        return None


def save_tokens(tokens: TokenInfo) -> None:
    """Save tokens to disk."""
    path = get_token_path()
    
    data = {
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "expires_at": tokens.expires_at.isoformat(),
        "project_id": tokens.project_id,
        "email": tokens.email,
    }
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def clear_tokens() -> None:
    """Clear stored tokens."""
    path = get_token_path()
    if os.path.exists(path):
        os.remove(path)


# =============================================================================
# SSE Parser
# =============================================================================


def parse_sse_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single SSE line."""
    if not line.startswith("data:"):
        return None
    
    data = line[5:].strip()
    if not data:
        return None
    
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


# =============================================================================
# Antigravity Provider
# =============================================================================


class AntigravityProvider(LLMProvider):
    """
    LLM provider using Google's Antigravity API.
    
    Provides access to Claude and Gemini models through Google OAuth.
    
    Usage:
        provider = AntigravityProvider()
        
        # First time: authenticate
        await provider.authenticate()
        
        # Then use normally
        response = await provider.complete(
            messages=[Message(role=Role.USER, content="Hello!")],
            model="antigravity-claude-sonnet-4-5",
        )
    """
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the provider.
        
        Args:
            base_url: Override the Antigravity API URL (defaults to daily sandbox)
        """
        self.base_url = base_url or ANTIGRAVITY_ENDPOINTS["daily"]
        self._tokens: Optional[TokenInfo] = None
        self._client: Optional[httpx.AsyncClient] = None
        
        # Try to load existing tokens
        self._tokens = load_tokens()
    
    @property
    def name(self) -> str:
        return "antigravity"
    
    @property
    def is_authenticated(self) -> bool:
        """Check if we have valid tokens."""
        if not self._tokens:
            return False
        
        # Check if token is expired (with 5 min buffer)
        buffer = timedelta(minutes=5)
        return self._tokens.expires_at > datetime.now(timezone.utc) + buffer
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client
    
    async def authenticate(self) -> bool:
        """
        Start OAuth flow to authenticate with Google.
        
        Opens a browser window for Google sign-in.
        Returns True if authentication was successful.
        """
        # Generate PKCE params
        code_verifier, code_challenge, state = generate_pkce_params()
        
        # Build authorization URL
        redirect_uri = f"http://localhost:{OAUTH_CONFIG['callback_port']}{OAUTH_CONFIG['callback_path']}"
        
        auth_params = {
            "client_id": OAUTH_CONFIG["client_id"],
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": " ".join(OAUTH_CONFIG["scopes"]),
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }
        
        auth_url = f"{OAUTH_CONFIG['auth_endpoint']}?{urlencode(auth_params)}"
        
        # Start callback server
        server = HTTPServer(("localhost", OAUTH_CONFIG["callback_port"]), OAuthCallbackHandler)
        server.oauth_result = None  # type: ignore
        server.timeout = 120  # 2 minute timeout
        
        # Open browser
        print(f"Opening browser for Google sign-in...")
        webbrowser.open(auth_url)
        
        # Wait for callback
        try:
            while server.oauth_result is None:  # type: ignore
                server.handle_request()
        except Exception as e:
            print(f"OAuth callback error: {e}")
            return False
        finally:
            server.server_close()
        
        result = server.oauth_result  # type: ignore
        
        if "error" in result:
            print(f"OAuth error: {result['error']}")
            return False
        
        # Validate state
        if result["state"] != state:
            print("OAuth state mismatch - possible CSRF attack")
            return False
        
        # Exchange code for tokens
        return await self._exchange_code(result["code"], code_verifier, redirect_uri)
    
    async def _exchange_code(self, code: str, code_verifier: str, redirect_uri: str) -> bool:
        """Exchange authorization code for tokens."""
        client = await self._get_client()
        
        token_data = {
            "client_id": OAUTH_CONFIG["client_id"],
            "client_secret": OAUTH_CONFIG["client_secret"],
            "code": code,
            "code_verifier": code_verifier,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        }
        
        response = await client.post(
            OAUTH_CONFIG["token_endpoint"],
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        
        if response.status_code != 200:
            print(f"Token exchange failed: {response.text}")
            return False
        
        token_response = response.json()
        
        if "refresh_token" not in token_response:
            print("No refresh token received - please revoke access and try again")
            return False
        
        # Get user info
        email = await self._fetch_user_email(token_response["access_token"])
        
        # Discover project ID
        project_id = await self._discover_project(token_response["access_token"])
        
        # Store tokens
        self._tokens = TokenInfo(
            access_token=token_response["access_token"],
            refresh_token=token_response["refresh_token"],
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=token_response["expires_in"]),
            project_id=project_id or self._generate_synthetic_project_id(),
            email=email,
        )
        
        save_tokens(self._tokens)
        
        print(f"Authenticated as: {email or 'unknown'}")
        return True
    
    async def _fetch_user_email(self, access_token: str) -> Optional[str]:
        """Fetch user email from Google."""
        client = await self._get_client()
        
        try:
            response = await client.get(
                "https://www.googleapis.com/oauth2/v1/userinfo?alt=json",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            
            if response.status_code == 200:
                return response.json().get("email")
        except Exception:
            pass
        
        return None
    
    async def _discover_project(self, access_token: str) -> Optional[str]:
        """Discover project ID via loadCodeAssist."""
        client = await self._get_client()
        
        endpoints = [
            ANTIGRAVITY_ENDPOINTS["daily"],
            ANTIGRAVITY_ENDPOINTS["autopush"],
            ANTIGRAVITY_ENDPOINTS["production"],
        ]
        
        for base_url in endpoints:
            try:
                url = f"{base_url}{ANTIGRAVITY_ENDPOINTS['load_code_assist']}"
                
                response = await client.post(
                    url,
                    json={
                        "metadata": {
                            "ideType": "IDE_UNSPECIFIED",
                            "platform": "PLATFORM_UNSPECIFIED",
                            "pluginType": "GEMINI",
                        }
                    },
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                        **ANTIGRAVITY_HEADERS,
                    },
                )
                
                if response.status_code == 200:
                    data = response.json()
                    project = data.get("cloudaicompanionProject")
                    
                    if isinstance(project, str):
                        return project
                    elif isinstance(project, dict):
                        return project.get("id")
            except Exception:
                continue
        
        return None
    
    def _generate_synthetic_project_id(self) -> str:
        """Generate a synthetic project ID when discovery fails."""
        adjectives = ["useful", "bright", "swift", "calm", "bold"]
        nouns = ["fuze", "wave", "spark", "flow", "core"]
        
        adj = secrets.choice(adjectives)
        noun = secrets.choice(nouns)
        suffix = secrets.token_hex(4)
        
        return f"{adj}-{noun}-{suffix}"
    
    async def _ensure_authenticated(self) -> None:
        """Ensure we have valid tokens, refreshing if needed."""
        if not self._tokens:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        # Check if token needs refresh
        buffer = timedelta(minutes=5)
        if self._tokens.expires_at <= datetime.now(timezone.utc) + buffer:
            await self._refresh_token()
    
    async def _refresh_token(self) -> None:
        """Refresh the access token."""
        if not self._tokens:
            raise RuntimeError("No tokens to refresh")
        
        client = await self._get_client()
        
        token_data = {
            "client_id": OAUTH_CONFIG["client_id"],
            "client_secret": OAUTH_CONFIG["client_secret"],
            "grant_type": "refresh_token",
            "refresh_token": self._tokens.refresh_token,
        }
        
        response = await client.post(
            OAUTH_CONFIG["token_endpoint"],
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        
        if response.status_code != 200:
            # Token may have been revoked
            clear_tokens()
            self._tokens = None
            raise RuntimeError("Token refresh failed - please re-authenticate")
        
        token_response = response.json()
        
        self._tokens.access_token = token_response["access_token"]
        self._tokens.expires_at = datetime.now(timezone.utc) + timedelta(seconds=token_response["expires_in"])
        
        save_tokens(self._tokens)
    
    def _get_antigravity_model(self, model: str) -> str:
        """Map model name to Antigravity model ID."""
        return MODEL_MAPPINGS.get(model, model)
    
    def _get_thinking_budget(self, model: str) -> Optional[int]:
        """Get thinking budget for model."""
        if "-thinking-high" in model:
            return THINKING_BUDGETS["high"]
        elif "-thinking-medium" in model:
            return THINKING_BUDGETS["medium"]
        elif "-thinking-low" in model:
            return THINKING_BUDGETS["low"]
        return None
    
    def _build_request(
        self,
        messages: List[Message],
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ToolDefinition]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Build Antigravity API request."""
        # Transform messages
        contents = []
        for msg in messages:
            role = "model" if msg.role == Role.ASSISTANT else "user"
            contents.append({
                "role": role,
                "parts": [{"text": msg.content}],
            })
        
        # Build generation config
        generation_config: Dict[str, Any] = {}
        
        if max_tokens:
            generation_config["maxOutputTokens"] = max_tokens
        
        if temperature is not None:
            generation_config["temperature"] = temperature
        
        # Add thinking config
        thinking_budget = self._get_thinking_budget(model)
        if thinking_budget:
            generation_config["thinkingConfig"] = {
                "thinkingBudget": thinking_budget,
                "includeThoughts": True,
            }
            # Ensure maxOutputTokens > thinkingBudget
            if not generation_config.get("maxOutputTokens") or generation_config["maxOutputTokens"] <= thinking_budget:
                generation_config["maxOutputTokens"] = thinking_budget + 16000
        
        # Build request
        request: Dict[str, Any] = {
            "project": self._tokens.project_id if self._tokens else "",
            "model": self._get_antigravity_model(model),
            "request": {
                "contents": contents,
            },
            "userAgent": "antigravity",
            "requestId": f"awf-{secrets.token_hex(8)}",
        }
        
        if generation_config:
            request["request"]["generationConfig"] = generation_config
        
        # Add tools if provided
        if tools:
            function_declarations = []
            for tool in tools:
                function_declarations.append({
                    "name": tool.name.replace("-", "_"),
                    "description": tool.description,
                    "parameters": tool.parameters,
                })
            request["request"]["tools"] = [{"functionDeclarations": function_declarations}]
        
        return request
    
    async def complete(
        self,
        messages: List[Message],
        model: str = "antigravity-claude-sonnet-4-5",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[str] = None,
        **kwargs,
    ) -> CompletionResponse:
        """
        Generate a completion using Antigravity API.
        
        Args:
            messages: List of messages
            model: Model to use (e.g., "antigravity-claude-sonnet-4-5")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            tools: Optional tool definitions
            tool_choice: Tool choice mode
            
        Returns:
            Response with generated content
        """
        await self._ensure_authenticated()
        
        client = await self._get_client()
        
        request = self._build_request(messages, model, max_tokens, temperature, tools, stream=False)
        
        url = f"{self.base_url}{ANTIGRAVITY_ENDPOINTS['generate']}"
        
        # Tokens guaranteed by _ensure_authenticated()
        assert self._tokens is not None
        
        response = await client.post(
            url,
            json=request,
            headers={
                "Authorization": f"Bearer {self._tokens.access_token}",
                "Content-Type": "application/json",
                **ANTIGRAVITY_HEADERS,
            },
        )
        
        if response.status_code == 429:
            raise RuntimeError("Rate limited - please wait and try again")
        elif response.status_code == 401:
            clear_tokens()
            self._tokens = None
            raise RuntimeError("Authentication failed - please re-authenticate")
        elif response.status_code != 200:
            raise RuntimeError(f"API error: {response.status_code} - {response.text}")
        
        data = response.json()
        
        # Parse response
        content = ""
        tool_calls = []
        
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part and not part.get("thought"):
                    content += part["text"]
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append(ToolCall(
                        id=fc.get("id", secrets.token_hex(8)),
                        name=fc["name"],
                        arguments=fc.get("args", {}),
                    ))
        
        # Build usage
        usage_data = data.get("usageMetadata", {})
        usage = Usage(
            prompt_tokens=usage_data.get("promptTokenCount", 0),
            completion_tokens=usage_data.get("candidatesTokenCount", 0),
            total_tokens=usage_data.get("totalTokenCount", 0),
            total_cost=0.0,  # Antigravity is free
        )
        
        return CompletionResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            model=model,
            finish_reason=FinishReason.STOP,
        )
    
    async def stream(
        self,
        messages: List[Message],
        model: str = "antigravity-claude-sonnet-4-5",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream a completion using Antigravity API with SSE.
        
        Args:
            messages: List of messages
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            tools: Optional tool definitions
            tool_choice: Tool choice mode
            
        Yields:
            StreamChunk objects with incremental content
        """
        await self._ensure_authenticated()
        
        client = await self._get_client()
        
        request = self._build_request(messages, model, max_tokens, temperature, tools, stream=True)
        
        url = f"{self.base_url}{ANTIGRAVITY_ENDPOINTS['stream_generate']}"
        
        accumulated_content = ""
        
        # Tokens guaranteed by _ensure_authenticated()
        assert self._tokens is not None
        
        async with client.stream(
            "POST",
            url,
            json=request,
            headers={
                "Authorization": f"Bearer {self._tokens.access_token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                **ANTIGRAVITY_HEADERS,
            },
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(f"API error: {response.status_code}")
            
            async for line in response.aiter_lines():
                event = parse_sse_line(line)
                if not event:
                    continue
                
                candidates = event.get("candidates", [])
                if not candidates:
                    continue
                
                parts = candidates[0].get("content", {}).get("parts", [])
                
                for part in parts:
                    if "text" in part and not part.get("thought"):
                        text = part["text"]
                        accumulated_content += text
                        
                        yield StreamChunk(
                            content=text,
                            accumulated_content=accumulated_content,
                            is_final=False,
                        )
        
        # Final chunk
        yield StreamChunk(
            content="",
            accumulated_content=accumulated_content,
            is_final=True,
        )
    
    async def embed(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small",
        **kwargs,
    ) -> List[List[float]]:
        """
        Generate embeddings.
        
        Note: Antigravity doesn't support embeddings directly.
        Use a different provider for embeddings.
        """
        raise NotImplementedError("Antigravity doesn't support embeddings - use OpenAI or another provider")
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def sign_out(self) -> None:
        """Sign out and clear stored tokens."""
        clear_tokens()
        self._tokens = None
        print("Signed out successfully")


# =============================================================================
# Factory
# =============================================================================


def create_antigravity_provider(base_url: Optional[str] = None) -> AntigravityProvider:
    """Create an Antigravity provider instance."""
    return AntigravityProvider(base_url=base_url)
