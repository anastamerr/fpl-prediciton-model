"""
LLM response generator that wraps OpenRouterClient with simple logging hooks.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
import time

from .openrouter_client import OpenRouterClient, SUPPORTED_MODELS, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    model: str
    content: str
    usage: Dict[str, Any]
    latency_ms: int
    raw: Dict[str, Any]


class LLMGenerator:
    def __init__(self, client: OpenRouterClient):
        self.client = client

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> GenerationResult:
        model_id = model or self.client.default_model
        if model_id not in SUPPORTED_MODELS:
            logger.warning("Model %s not in supported list; proceeding anyway.", model_id)

        start = time.time()
        response: LLMResponse = self.client.chat(
            messages=messages, model=model_id, temperature=temperature, max_tokens=max_tokens
        )
        latency_ms = max(response.latency_ms, int((time.time() - start) * 1000))

        return GenerationResult(
            model=response.model,
            content=response.content,
            usage=response.usage,
            latency_ms=latency_ms,
            raw=response.raw,
        )
