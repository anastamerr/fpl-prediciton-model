"""
Quantitative evaluator scaffold for comparing multiple LLMs on a shared test set.

Intended metrics:
- latency_ms
- token usage (prompt/completion/total)
- response length
- (optional) accuracy/hallucination flags when gold answers are provided
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
import time

from .llm_generator import LLMGenerator, GenerationResult
from .prompt_builder import PromptBuilder


@dataclass
class EvaluationSample:
    question: str
    context: Any
    gold_answer: Optional[str] = None


@dataclass
class QuantResult:
    model: str
    question: str
    latency_ms: int
    tokens_total: Optional[int]
    tokens_prompt: Optional[int]
    tokens_completion: Optional[int]
    response_length: int
    content: str
    gold_answer: Optional[str]
    correctness: Optional[bool]


class QuantitativeEvaluator:
    def __init__(
        self,
        generator: LLMGenerator,
        prompt_builder: Optional[PromptBuilder] = None,
        metric_fn: Optional[Callable[[str, str], bool]] = None,
    ):
        self.generator = generator
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.metric_fn = metric_fn

    def evaluate(self, samples: List[EvaluationSample], models: List[str]) -> List[QuantResult]:
        results: List[QuantResult] = []
        for sample in samples:
            messages = self.prompt_builder.build_messages(user_query=sample.question, kg_context=sample.context)
            for model in models:
                start = time.time()
                gen: GenerationResult = self.generator.generate(messages=messages, model=model)
                latency_ms = max(gen.latency_ms, int((time.time() - start) * 1000))
                usage = gen.usage or {}
                correctness = None
                if sample.gold_answer and self.metric_fn:
                    correctness = self.metric_fn(gen.content, sample.gold_answer)
                results.append(
                    QuantResult(
                        model=gen.model,
                        question=sample.question,
                        latency_ms=latency_ms,
                        tokens_total=usage.get("total_tokens"),
                        tokens_prompt=usage.get("prompt_tokens"),
                        tokens_completion=usage.get("completion_tokens"),
                        response_length=len(gen.content.split()),
                        content=gen.content,
                        gold_answer=sample.gold_answer,
                        correctness=correctness,
                    )
                )
        return results
