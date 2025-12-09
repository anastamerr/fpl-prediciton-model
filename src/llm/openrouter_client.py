"""
Lightweight OpenRouter API client wrapper.

Uses the OpenAI-compatible chat completions endpoint; callers provide a list of
messages. This client centralizes headers, retries, and basic error handling.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import time
import requests


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

SUPPORTED_MODELS = [
    "mistralai/mistral-7b-instruct:free",
    "moonshotai/kimi-k2:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
]


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter returns an error response."""


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Dict[str, Any]
    latency_ms: int
    raw: Dict[str, Any]


class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        default_model: str = SUPPORTED_MODELS[0],
        timeout: int = 60,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required.")
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        attempt = 0
        start = time.time()
        last_error: Optional[str] = None
        while attempt <= self.max_retries:
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                latency_ms = int((time.time() - start) * 1000)
                if response.status_code >= 400:
                    last_error = response.text
                    attempt += 1
                    time.sleep(1.5 * attempt)
                    continue
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                model_used = data.get("model", payload["model"])
                usage = data.get("usage", {})
                return LLMResponse(content, model_used, usage, latency_ms, data)
            except requests.RequestException as exc:
                last_error = str(exc)
                attempt += 1
                time.sleep(1.5 * attempt)

        raise OpenRouterError(f"Failed to complete chat request: {last_error}")
