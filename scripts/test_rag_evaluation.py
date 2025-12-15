"""
RAG Evaluation Script - Tests different LLMs with different retrieval methods.

This script tests the FPL RAG system by:
1. Running 5 FPL-related questions
2. Testing 3 models: GPT 5.1 Nano, Gemini Flash Lite, Llama 3.3
3. Testing 5 retrieval modes: Baseline, Hybrid (BGE), Hybrid (MPNet), Embedding (BGE), Embedding (MPNet)
4. Saving all results to a CSV file

Usage:
    python scripts/test_rag_evaluation.py
"""

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Make src importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

load_dotenv()

from src.preprocessing.intent_classifier import FPLIntentClassifier
from src.preprocessing.entity_extractor import FPLEntityExtractor
from src.retrieval.baseline_retriever import BaselineRetriever
from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.neo4j_client import get_driver, verify_connection
from src.llm.prompt_builder import PromptBuilder
from src.llm.llm_generator import LLMGenerator
from src.llm.openrouter_client import OpenRouterClient, SUPPORTED_MODELS


# Test configuration
MODELS = [
    "openai/gpt-5-nano",
    "google/gemini-2.0-flash-lite-001",
    "meta-llama/llama-3.3-70b-instruct",
]

RETRIEVAL_METHODS = [
    "Baseline Only",
    "Hybrid (Baseline + BGE)",
    "Hybrid (Baseline + MPNet)",
    "Embeddings Only (BGE)",
    "Embeddings Only (MPNet)",
]

# 5 FPL questions derived from the fpl_two_seasons.csv data
TEST_QUESTIONS = [
    "Who scored the most goals in the 2021-22 season?",
    "Which midfielder had the highest total points in 2022-23?",
    "Who had the most assists among forwards in 2021-22?",
    "Which defenders had the most clean sheets in 2022-23?",
    "Which players had the highest ICT index in 2021-22?",
]


def load_player_index(driver) -> List[str]:
    """Load all player names from Neo4j for entity extraction."""
    if not driver:
        return []
    try:
        with driver.session() as session:
            result = session.run("MATCH (p:Player) RETURN p.player_name AS name")
            return [row["name"] for row in result if row["name"]]
    except Exception:
        return []


def load_team_index(driver) -> List[str]:
    """Load all team names from Neo4j for entity extraction."""
    if not driver:
        return []
    try:
        with driver.session() as session:
            result = session.run("MATCH (t:Team) RETURN t.name AS name")
            return [row["name"] for row in result if row["name"]]
    except Exception:
        return []


def run_retrieval(
    query: str,
    retrieval_method: str,
    driver,
    classifier: FPLIntentClassifier,
    extractor: FPLEntityExtractor,
) -> Dict[str, Any]:
    """Run the retrieval pipeline for a given query and method."""
    intent_result = classifier.classify(query)
    entities = extractor.extract(query)
    anchor_player = entities.players[0] if entities.players else None

    baseline_records: List[Dict[str, Any]] = []
    embedding_hits: List[Dict[str, Any]] = []
    fused: List[Dict[str, Any]] = []
    cypher_query = None
    errors: List[str] = []

    use_baseline = retrieval_method in [
        "Baseline Only",
        "Hybrid (Baseline + BGE)",
        "Hybrid (Baseline + MPNet)",
    ]
    use_embedding = retrieval_method in [
        "Hybrid (Baseline + BGE)",
        "Hybrid (Baseline + MPNet)",
        "Embeddings Only (BGE)",
        "Embeddings Only (MPNet)",
    ]
    embed_model = "mpnet" if "MPNet" in retrieval_method else "bge-small"

    # Baseline retrieval
    baseline_runner = None
    if driver and use_baseline:
        baseline_runner = BaselineRetriever(driver)
        try:
            bres = baseline_runner.retrieve(intent=intent_result.intent, entities=entities.__dict__)
            baseline_records = bres.records
            cypher_query = bres.query
            if bres.error:
                errors.append(f"Baseline error: {bres.error}")
        except Exception as exc:
            errors.append(f"Baseline retrieval failed: {exc}")

    # Embedding retrieval
    if driver and use_embedding:
        try:
            emb = EmbeddingRetriever(driver, top_k=20)
            position_filter = entities.positions[0] if entities.positions else None
            if anchor_player:
                embedding_hits = [
                    hit.__dict__
                    for hit in emb.search(
                        anchor_player=anchor_player,
                        k=20,
                        position=position_filter,
                        exclude_players=entities.players,
                    )
                ]
                if "Hybrid" in retrieval_method and baseline_runner:
                    fused = HybridRetriever(baseline_runner, emb).retrieve(
                        intent=intent_result.intent, entities=entities.__dict__, user_query=query
                    ).fused
            else:
                # For embedding search without anchor player, skip
                errors.append("Embedding search skipped: no anchor player found.")
        except Exception as exc:
            errors.append(f"Embedding retrieval failed: {exc}")

    # Fallback for embedding-only modes
    if driver and use_embedding and not baseline_records:
        if len(embedding_hits) < 3:
            try:
                baseline_runner = baseline_runner or BaselineRetriever(driver)
                positions_hint = entities.positions or ["FWD", "MID"]
                fallback_entities = {
                    "seasons": entities.seasons,
                    "positions": positions_hint,
                    "statistics": ["total_points"],
                    "numerical_values": {"limit": 5},
                }
                stat_res = baseline_runner.retrieve(intent="statistics_query", entities=fallback_entities)
                if stat_res.records:
                    baseline_records = stat_res.records
                    cypher_query = cypher_query or stat_res.query
            except Exception as exc:
                errors.append(f"Fallback retrieval failed: {exc}")

    context = {
        "baseline": baseline_records,
        "embedding_hits": embedding_hits,
        "fused": fused,
        "anchor_player": anchor_player,
    }

    return {
        "intent": intent_result,
        "entities": entities,
        "context": context,
        "cypher_query": cypher_query,
        "errors": errors,
        "retrieval_method": retrieval_method,
    }


def generate_llm_response(
    query: str,
    context: Dict[str, Any],
    model: str,
    llm_generator: LLMGenerator,
) -> Dict[str, Any]:
    """Generate LLM response for the given query and context."""
    try:
        prompt = PromptBuilder().build_messages(user_query=query, kg_context=context)
        start_time = time.time()
        gen = llm_generator.generate(messages=prompt, model=model)
        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "answer": gen.content,
            "model": gen.model,
            "latency_ms": gen.latency_ms or latency_ms,
            "usage": gen.usage,
            "error": None,
        }
    except Exception as exc:
        return {
            "answer": None,
            "model": model,
            "latency_ms": 0,
            "usage": {},
            "error": str(exc),
        }


def run_evaluation():
    """Main evaluation function."""
    print("=" * 80)
    print("FPL RAG Evaluation Script")
    print("=" * 80)

    # Initialize connections
    print("\nInitializing connections...")
    driver = None
    try:
        driver = get_driver()
        if not verify_connection(driver):
            print("WARNING: Neo4j connection failed. Retrieval will use placeholders.")
            driver = None
        else:
            print("Neo4j connection successful.")
    except Exception as exc:
        print(f"WARNING: Neo4j connection failed: {exc}")
        driver = None

    llm_generator = None
    try:
        client = OpenRouterClient()
        llm_generator = LLMGenerator(client)
        print("OpenRouter client initialized successfully.")
    except Exception as exc:
        print(f"ERROR: OpenRouter client initialization failed: {exc}")
        return

    # Initialize classifiers
    classifier = FPLIntentClassifier()
    player_index = load_player_index(driver)
    team_index = load_team_index(driver)
    extractor = FPLEntityExtractor(player_index=player_index, team_index=team_index)

    print(f"\nLoaded {len(player_index)} players and {len(team_index)} teams for entity extraction.")

    # Results storage
    results = []

    # Timestamp for the output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = ROOT / f"evaluation_results_{timestamp}.csv"

    total_tests = len(TEST_QUESTIONS) * len(RETRIEVAL_METHODS) * len(MODELS)
    current_test = 0

    print(f"\nRunning {total_tests} total tests...")
    print(f"Questions: {len(TEST_QUESTIONS)}")
    print(f"Retrieval methods: {len(RETRIEVAL_METHODS)}")
    print(f"Models: {len(MODELS)}")
    print("-" * 80)

    for question in TEST_QUESTIONS:
        print(f"\n[Question] {question}")

        for retrieval_method in RETRIEVAL_METHODS:
            # Run retrieval once per method per question
            retrieval_result = run_retrieval(
                query=question,
                retrieval_method=retrieval_method,
                driver=driver,
                classifier=classifier,
                extractor=extractor,
            )

            for model in MODELS:
                current_test += 1
                print(f"  [{current_test}/{total_tests}] {retrieval_method} + {model.split('/')[-1][:20]}...", end=" ")

                # Generate LLM response
                llm_result = generate_llm_response(
                    query=question,
                    context=retrieval_result["context"],
                    model=model,
                    llm_generator=llm_generator,
                )

                # Record result
                result = {
                    "question": question,
                    "retrieval_method": retrieval_method,
                    "model": model,
                    "intent": retrieval_result["intent"].intent,
                    "intent_confidence": retrieval_result["intent"].confidence,
                    "entities_players": str(retrieval_result["entities"].players),
                    "entities_teams": str(retrieval_result["entities"].teams),
                    "entities_seasons": str(retrieval_result["entities"].seasons),
                    "entities_positions": str(retrieval_result["entities"].positions),
                    "baseline_records_count": len(retrieval_result["context"]["baseline"]),
                    "embedding_hits_count": len(retrieval_result["context"]["embedding_hits"]),
                    "fused_count": len(retrieval_result["context"]["fused"]),
                    "cypher_query": retrieval_result["cypher_query"] or "",
                    "retrieval_errors": "; ".join(retrieval_result["errors"]),
                    "llm_answer": llm_result["answer"] or "",
                    "llm_latency_ms": llm_result["latency_ms"],
                    "llm_prompt_tokens": llm_result["usage"].get("prompt_tokens", 0),
                    "llm_completion_tokens": llm_result["usage"].get("completion_tokens", 0),
                    "llm_total_tokens": llm_result["usage"].get("total_tokens", 0),
                    "llm_error": llm_result["error"] or "",
                }
                results.append(result)

                if llm_result["error"]:
                    print(f"ERROR: {llm_result['error'][:50]}")
                else:
                    print(f"OK ({llm_result['latency_ms']}ms)")

                # Small delay to avoid rate limiting
                time.sleep(0.5)

    # Write results to CSV
    print(f"\n\nWriting results to {output_file}...")

    fieldnames = [
        "question",
        "retrieval_method",
        "model",
        "intent",
        "intent_confidence",
        "entities_players",
        "entities_teams",
        "entities_seasons",
        "entities_positions",
        "baseline_records_count",
        "embedding_hits_count",
        "fused_count",
        "cypher_query",
        "retrieval_errors",
        "llm_answer",
        "llm_latency_ms",
        "llm_prompt_tokens",
        "llm_completion_tokens",
        "llm_total_tokens",
        "llm_error",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {output_file}")
    print(f"Total tests completed: {len(results)}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = sum(1 for r in results if not r["llm_error"])
    failed = sum(1 for r in results if r["llm_error"])
    avg_latency = sum(r["llm_latency_ms"] for r in results if not r["llm_error"]) / max(successful, 1)

    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print(f"Average latency: {avg_latency:.0f}ms")

    # Per-model summary
    print("\nPer-model results:")
    for model in MODELS:
        model_results = [r for r in results if r["model"] == model]
        model_successful = sum(1 for r in model_results if not r["llm_error"])
        model_avg_latency = sum(r["llm_latency_ms"] for r in model_results if not r["llm_error"]) / max(model_successful, 1)
        print(f"  {model}: {model_successful}/{len(model_results)} successful, avg {model_avg_latency:.0f}ms")

    # Per-retrieval method summary
    print("\nPer-retrieval method results:")
    for method in RETRIEVAL_METHODS:
        method_results = [r for r in results if r["retrieval_method"] == method]
        method_successful = sum(1 for r in method_results if not r["llm_error"])
        print(f"  {method}: {method_successful}/{len(method_results)} successful")

    print("\nEvaluation complete!")

    if driver:
        driver.close()


if __name__ == "__main__":
    run_evaluation()
