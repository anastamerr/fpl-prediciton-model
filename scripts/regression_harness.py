"""
Lightweight regression harness to run end-to-end queries (intent -> entities ->
retrieval -> LLM) and perform simple sanity checks on the LLM output.

Usage:
    python scripts/regression_harness.py \
        --queries data/regression_queries.json \
        --mode baseline_first

Modes:
  - baseline_first: use Baseline for queries tagged "baseline", Hybrid(BGE) for
    "hybrid", Embedding(BGE) for "embedding".
  - hybrid_only / embedding_only: override per-query tags.

Outputs:
  - data/regression_results.json (detailed)
  - data/regression_results.csv (summary)
"""

import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Make src importable
ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.preprocessing.intent_classifier import FPLIntentClassifier  # noqa: E402
from src.preprocessing.entity_extractor import FPLEntityExtractor  # noqa: E402
from src.retrieval.baseline_retriever import BaselineRetriever  # noqa: E402
from src.retrieval.embedding_retriever import EmbeddingRetriever  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.utils.neo4j_client import get_driver, verify_connection  # noqa: E402
from src.llm.openrouter_client import OpenRouterClient, SUPPORTED_MODELS  # noqa: E402
from src.llm.llm_generator import LLMGenerator  # noqa: E402
from src.llm.prompt_builder import PromptBuilder  # noqa: E402


def load_indexes(driver) -> Tuple[List[str], List[str]]:
    """Fetch player/team names to prime entity extractor."""
    players, teams = [], []
    with driver.session() as session:
        rows = session.run("MATCH (p:Player) RETURN p.player_name AS name")
        players = [r["name"] for r in rows if r.get("name")]
        rows = session.run("MATCH (t:Team) RETURN t.name AS name")
        teams = [r["name"] for r in rows if r.get("name")]
    return players, teams


def extract_names_from_text(text: str) -> List[str]:
    """
    Naive proper-noun extractor to spot names the LLM mentions.
    Looks for capitalized tokens or two-word capitalized sequences.
    """
    if not text:
        return []
    matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text)
    # Filter common stop words that are capitalized at sentence start
    stop = {"The", "A", "An", "This", "That", "If", "In", "On", "At", "For", "With", "Best", "Top"}
    return [m.strip() for m in matches if m.strip() and m.split()[0] not in stop]


def build_context_summary(baseline_records: List[Dict[str, Any]], embedding_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    players = set()
    for row in baseline_records:
        p = row.get("player") or row.get("player_name")
        if p:
            players.add(p)
        for key, val in row.items():
            if isinstance(val, str) and "player" in key and val not in players:
                players.add(val)
    for hit in embedding_hits:
        p = hit.get("player") or hit.get("player_name")
        if p:
            players.add(p)
    return {
        "players": sorted(players),
        "baseline_count": len(baseline_records),
        "embedding_count": len(embedding_hits),
    }


def sanity_check(llm_answer: str, context_summary: Dict[str, Any], baseline_count: int, embedding_count: int) -> List[str]:
    issues: List[str] = []
    if not llm_answer or len(llm_answer.strip()) < 10:
        issues.append("LLM answer is empty or too short.")
        return issues

    if baseline_count == 0 and embedding_count == 0 and "not enough" not in llm_answer.lower():
        issues.append("Answered without context but did not acknowledge missing data.")

    mentioned = extract_names_from_text(llm_answer)
    context_players = set(context_summary.get("players", []))
    if mentioned:
        unknown = [m for m in mentioned if m not in context_players]
        if len(unknown) >= 3:
            issues.append(f"LLM mentions names not in context: {unknown[:5]}")
    return issues


def pick_retrieval(mode: str, default: str) -> str:
    if mode == "baseline":
        return "baseline"
    if mode == "hybrid":
        return "hybrid"
    if mode == "embedding":
        return "embedding"
    return default


def run_single(
    question: str,
    retrieval_mode: str,
    classifier: FPLIntentClassifier,
    extractor: FPLEntityExtractor,
    driver,
    llm: Optional[LLMGenerator],
    prompt_builder: PromptBuilder,
) -> Dict[str, Any]:
    intent = classifier.classify(question)
    entities = extractor.extract(question)

    baseline_records: List[Dict[str, Any]] = []
    embedding_hits: List[Dict[str, Any]] = []
    fused: List[Dict[str, Any]] = []
    cypher_query = None
    errors: List[str] = []

    baseline = BaselineRetriever(driver)
    if retrieval_mode in {"baseline", "hybrid"}:
        try:
            bres = baseline.retrieve(intent=intent.intent, entities=entities.__dict__)
            baseline_records = bres.records
            cypher_query = bres.query
            if bres.error:
                errors.append(f"Baseline error: {bres.error}")
        except Exception as exc:
            errors.append(f"Baseline failure: {exc}")

    if retrieval_mode in {"embedding", "hybrid"}:
        try:
            emb = EmbeddingRetriever(driver, top_k=15)
            pos_filter = entities.positions[0] if entities.positions else None
            anchor = entities.players[0] if entities.players else None
            if anchor:
                hits = emb.search(
                    anchor_player=anchor,
                    position=pos_filter,
                    exclude_players=entities.players,
                )
                embedding_hits = [h.__dict__ for h in hits]
                if retrieval_mode == "hybrid":
                    fused = HybridRetriever(baseline, emb).retrieve(
                        intent=intent.intent, entities=entities.__dict__, user_query=question
                    ).fused
            else:
                errors.append("Embedding search skipped: provide a player name to anchor similarity.")
        except Exception as exc:
            errors.append(f"Embedding failure: {exc}")

    context = {
        "baseline": baseline_records,
        "embedding_hits": embedding_hits[:5],
        "fused": fused,
        "anchor_player": entities.players[0] if entities.players else None,
        "embedding_summary": [
            {"player": h.get("player"), "score": h.get("score")} for h in embedding_hits[:5]
        ],
    }
    if context["anchor_player"] and embedding_hits:
        summary_str = ", ".join(
            f"{h.get('player')} (score {round(h.get('score', 0), 3)})" for h in embedding_hits[:5]
        )
        context["embedding_summary_text"] = f"Anchor: {context['anchor_player']}; similar: {summary_str}"
    llm_answer = ""
    llm_meta: Dict[str, Any] = {}
    if llm:
        messages = prompt_builder.build_messages(user_query=question, kg_context=context)
        env_models = os.getenv("LLM_MODELS")
        env_model_single = os.getenv("LLM_MODEL")
        candidates = []
        if env_models:
            candidates = [m.strip() for m in env_models.split(",") if m.strip()]
        elif env_model_single:
            candidates = [env_model_single]
        else:
            # Prefer less rate-limited models first.
            candidates = [
                "google/gemini-2.0-flash-lite-001",
                "google/gemini-2.0-flash-exp:free",
                "openai/gpt-5-nano",
                "moonshotai/kimi-k2:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "mistralai/mistral-7b-instruct:free",
            ]
        last_error: Optional[str] = None
        for model_id in candidates or SUPPORTED_MODELS:
            try:
                gen = llm.generate(messages=messages, model=model_id)
                llm_answer = gen.content
                llm_meta = {"model": gen.model, "latency_ms": gen.latency_ms, "usage": gen.usage}
                last_error = None
                break
            except Exception as exc:
                last_error = str(exc)
                continue
        if last_error:
            errors.append(f"LLM failure: {last_error}")
    else:
        llm_answer = "LLM disabled (no client)."

    context_summary = build_context_summary(baseline_records, embedding_hits)
    issues = sanity_check(llm_answer, context_summary, len(baseline_records), len(embedding_hits))

    return {
        "question": question,
        "intent": intent.intent,
        "confidence": intent.confidence,
        "entities": entities.__dict__,
        "retrieval_mode": retrieval_mode,
        "baseline_count": len(baseline_records),
        "embedding_count": len(embedding_hits),
        "fused_count": len(fused),
        "cypher_query": cypher_query,
        "llm_answer": llm_answer,
        "llm_meta": llm_meta,
        "context_summary": context_summary,
        "issues": issues + errors,
    }


def run_all(queries: List[Dict[str, Any]], override_mode: Optional[str] = None) -> List[Dict[str, Any]]:
    load_dotenv()
    driver = get_driver()
    if not verify_connection(driver):
        raise RuntimeError("Neo4j connection failed. Check NEO4J_* in .env.")

    player_idx, team_idx = load_indexes(driver)
    extractor = FPLEntityExtractor(player_index=player_idx, team_index=team_idx)
    classifier = FPLIntentClassifier()
    prompt_builder = PromptBuilder()

    llm = None
    try:
        client = OpenRouterClient()
        llm = LLMGenerator(client)
    except Exception as exc:
        print(f"[WARN] OpenRouter not initialized: {exc}")

    results = []
    for idx, item in enumerate(queries):
        qtext = item["question"]
        tag_mode = item.get("mode", "baseline")
        retrieval_mode = pick_retrieval(tag_mode, override_mode or "baseline")
        print(f"Running: {item.get('id', '')} [{retrieval_mode}] -> {qtext}")
        res = run_single(
            question=qtext,
            retrieval_mode=retrieval_mode,
            classifier=classifier,
            extractor=extractor,
            driver=driver,
            llm=llm,
            prompt_builder=prompt_builder,
        )
        results.append(res)
    return results


def write_outputs(results: List[Dict[str, Any]], out_json: Path, out_csv: Path) -> None:
    out_json.write_text(json.dumps(results, indent=2))
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "question",
                "intent",
                "retrieval_mode",
                "baseline_count",
                "embedding_count",
                "issues",
                "latency_ms",
                "model",
            ]
        )
        for r in results:
            meta = r.get("llm_meta", {})
            writer.writerow(
                [
                    r["question"],
                    r["intent"],
                    r["retrieval_mode"],
                    r["baseline_count"],
                    r["embedding_count"],
                    "; ".join(r.get("issues", [])),
                    meta.get("latency_ms"),
                    meta.get("model"),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run regression queries through the pipeline with sanity checks.")
    parser.add_argument("--queries", default="data/regression_queries.json", help="Path to queries JSON.")
    parser.add_argument(
        "--mode",
        choices=["baseline_first", "hybrid_only", "embedding_only"],
        default="baseline_first",
        help="Override retrieval strategy.",
    )
    parser.add_argument("--out-json", default="data/regression_results.json", help="Output JSON path.")
    parser.add_argument("--out-csv", default="data/regression_results.csv", help="Output CSV path.")
    parser.add_argument("--max", type=int, default=None, help="Limit number of queries to run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = json.loads(Path(args.queries).read_text())
    if args.max:
        queries = queries[: args.max]
    override = None
    if args.mode == "hybrid_only":
        override = "hybrid"
    elif args.mode == "embedding_only":
        override = "embedding"
    results = run_all(queries, override_mode=override)

    timestamp = datetime.utcnow().isoformat() + "Z"
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    write_outputs(results, out_json, out_csv)
    print(f"Wrote {len(results)} results at {timestamp} -> {out_json} and {out_csv}")


if __name__ == "__main__":
    main()
