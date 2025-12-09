"""
Run quantitative evaluation across models using eval dataset.

Usage:
    python scripts/run_quant_eval.py --eval-file data/eval_questions.json --models mistralai/mistral-7b-instruct:free

Requires ENABLE_LLM=1 and OPENROUTER_API_KEY in env.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.llm.openrouter_client import OpenRouterClient  # noqa: E402
from src.llm.llm_generator import LLMGenerator  # noqa: E402
from src.llm.prompt_builder import PromptBuilder  # noqa: E402
from src.llm.quantitative_evaluator import QuantitativeEvaluator, EvaluationSample  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quantitative evaluation over LLMs.")
    parser.add_argument("--eval-file", default="data/eval_questions.json", help="Path to eval JSON.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "mistralai/mistral-7b-instruct:free",
            "moonshotai/kimi-k2:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemini-2.0-flash-exp:free",
        ],
    )
    return parser.parse_args()


def load_samples(path: Path):
    data = json.loads(path.read_text())
    return [EvaluationSample(question=item["question"], context=item.get("context"), gold_answer=item.get("gold_answer")) for item in data]


def main() -> None:
    args = parse_args()
    eval_path = Path(args.eval_file)
    samples = load_samples(eval_path)

    client = OpenRouterClient()
    generator = LLMGenerator(client)
    evaluator = QuantitativeEvaluator(generator, PromptBuilder())

    results = evaluator.evaluate(samples, models=args.models)
    for r in results:
        print(
            f"{r.model} | {r.latency_ms} ms | prompt {r.tokens_prompt} | completion {r.tokens_completion} | total {r.tokens_total} | q: {r.question}"
        )


if __name__ == "__main__":
    main()
