"""
Streamlit UI for the FPL Team Formulation Recommender System.

Backend calls are guarded: if Neo4j or OpenRouter are not configured, the app
falls back to safe placeholders. Integrate live retrieval/LLM when credentials
and infrastructure are available.
"""

from pathlib import Path
import os
import sys
import re
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

# Optional viz dependencies
try:
    import pandas as pd
    import networkx as nx  # type: ignore
    import plotly.graph_objects as go  # type: ignore
except ImportError:  # pragma: no cover
    pd = None
    nx = None
    go = None
load_dotenv()  # ensure .env is loaded for Streamlit sessions

# Make src importable when running via `streamlit run app/streamlit_app.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.preprocessing.intent_classifier import FPLIntentClassifier  # noqa: E402
from src.preprocessing.entity_extractor import FPLEntityExtractor  # noqa: E402
from src.retrieval.baseline_retriever import BaselineRetriever  # noqa: E402
from src.retrieval.embedding_retriever import EmbeddingRetriever  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.utils.neo4j_client import get_driver, verify_connection  # noqa: E402
from src.llm.prompt_builder import PromptBuilder  # noqa: E402
from src.llm.llm_generator import LLMGenerator  # noqa: E402
from src.llm.openrouter_client import OpenRouterClient  # noqa: E402


MODELS = [
    "mistralai/mistral-7b-instruct:free",
    "moonshotai/kimi-k2:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
]

RETRIEVAL_METHODS = [
    "Hybrid (Baseline + BGE)",
    "Hybrid (Baseline + MPNet)",
    "Baseline Only",
    "Embeddings Only (BGE)",
    "Embeddings Only (MPNet)",
]

EXAMPLE_QUERIES = [
    "Recommend a squad with budget constraints",
    "Who are the top forwards this season?",
    "Compare Haaland vs Kane for 2022-23",
    "Show me fixtures for gameweek 10",
    "Players with most goals in 2021-22",
]

_classifier = FPLIntentClassifier()
_extractor = None  # Initialized after Neo4j connection
LLM_ENABLED = os.getenv("ENABLE_LLM", "1").lower() in ("1", "true", "yes")


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


def init_extractor(driver) -> FPLEntityExtractor:
    """Initialize entity extractor with player/team indexes from Neo4j."""
    player_index = load_player_index(driver)
    team_index = load_team_index(driver)
    return FPLEntityExtractor(player_index=player_index, team_index=team_index)


def init_driver():
    try:
        driver = get_driver()
        return driver if verify_connection(driver) else None
    except Exception:
        return None


def init_llm():
    try:
        client = OpenRouterClient()
        return LLMGenerator(client)
    except Exception as exc:
        st.warning(f"OpenRouter client init failed: {exc}")
        return None


def run_pipeline(
    query: str,
    retrieval_choice: str,
    model_choice: str,
    driver,
    llm_generator,
) -> Dict[str, Any]:
    intent_result = _classifier.classify(query)
    entities = _extractor.extract(query)

    baseline_records: List[Dict[str, Any]] = []
    cypher_query = None
    embedding_hits: List[Dict[str, Any]] = []
    fused: List[Dict[str, Any]] = []
    errors: List[str] = []
    baseline_runner = None

    use_baseline = retrieval_choice in [
        "Hybrid (Baseline + BGE)",
        "Hybrid (Baseline + MPNet)",
        "Baseline Only",
    ]
    use_embedding = retrieval_choice in [
        "Hybrid (Baseline + BGE)",
        "Hybrid (Baseline + MPNet)",
        "Embeddings Only (BGE)",
        "Embeddings Only (MPNet)",
    ]
    embed_model = "mpnet" if "MPNet" in retrieval_choice else "bge-small"

    # Retrieval path selection
    if driver and use_baseline:
        baseline_runner = BaselineRetriever(driver)
        try:
            bres = baseline_runner.retrieve(intent=intent_result.intent, entities=entities.__dict__)
            baseline_records = bres.records
            cypher_query = bres.query
            if bres.error:
                errors.append(f"Baseline error: {bres.error}")
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"Baseline retrieval failed: {exc}")

    elif use_baseline:
        baseline_records = [{"note": "Neo4j not connected; showing placeholder only."}]

    if driver and use_embedding:
        try:
            emb = EmbeddingRetriever(driver, model_alias=embed_model, top_k=20)
            # Extract position filter from entities (use first position if multiple)
            position_filter = entities.positions[0] if entities.positions else None
            embedding_hits = [hit.__dict__ for hit in emb.search(query_text=query, k=20, position=position_filter)]
            if "Hybrid" in retrieval_choice and baseline_runner:
                fused = HybridRetriever(baseline_runner, emb).retrieve(
                    intent=intent_result.intent, entities=entities.__dict__, user_query=query
                ).fused
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"Embedding retrieval skipped/failed: {exc}")
    elif use_embedding:
        embedding_hits = [{"note": "Neo4j not connected; no embeddings retrieved."}]

    # Fallback: for team recommendations, ensure some baseline context even if user chose embeddings-only.
    if intent_result.intent == "team_recommendation" and driver and not baseline_records:
        try:
            baseline_runner = baseline_runner or BaselineRetriever(driver)
            fallback_entities = entities.__dict__.copy()
            fallback_entities["numerical_values"] = fallback_entities.get("numerical_values", {}) or {}
            fallback_entities["numerical_values"].setdefault("budget", 100.0)
            fallback_entities["numerical_values"].setdefault("limit", 15)
            fallback_res = baseline_runner.retrieve(intent="team_recommendation", entities=fallback_entities)
            if fallback_res.records:
                baseline_records = fallback_res.records
                cypher_query = cypher_query or fallback_res.query
            # If still empty, fetch top players per position as context.
            if not baseline_records:
                position_samples = []
                for pos in ["GK", "DEF", "MID", "FWD"]:
                    pos_entities = {
                        "positions": [pos],
                        "seasons": fallback_entities.get("seasons", []),
                        "numerical_values": {"limit": 5},
                    }
                    res = baseline_runner.retrieve(intent="position_search", entities=pos_entities)
                    for row in res.records:
                        row["position"] = pos
                    position_samples.extend(res.records)
                if position_samples:
                    baseline_records = position_samples
        except Exception as exc:  # pragma: no cover
            errors.append(f"Fallback team recommendation failed: {exc}")

    llm_answer = "LLM call not executed (no OPENROUTER_API_KEY or ENABLE_LLM=0)."
    llm_meta: Dict[str, Any] = {
        "model": model_choice,
        "status": "skipped" if not LLM_ENABLED else "disabled",
    }
    context_for_llm: Any = {
        "baseline": baseline_records,
        "embedding_hits": embedding_hits,
        "fused": fused,
    }
    if not (baseline_records or embedding_hits or fused):
        llm_answer = (
            "No knowledge graph context retrieved. Try a more specific FPL question "
            "(e.g., 'top forwards 2022-23' or 'fixtures for Liverpool GW5')."
        )
        return {
            "intent": intent_result,
            "entities": entities,
            "baseline_records": baseline_records,
            "cypher_query": cypher_query,
            "embedding_hits": embedding_hits,
            "fused": fused,
            "llm_answer": llm_answer,
            "llm_meta": llm_meta,
            "errors": errors,
            "retrieval_choice": retrieval_choice,
            "model_choice": model_choice,
            "graph": _build_graph(baseline_records, embedding_hits),
        }
    if llm_generator and context_for_llm and LLM_ENABLED:
        try:
            prompt = PromptBuilder().build_messages(user_query=query, kg_context=context_for_llm)
            gen = llm_generator.generate(messages=prompt, model=model_choice)
            raw_content = gen.content
            cleaned = _clean_llm_content(raw_content)
            llm_answer = cleaned if cleaned else (raw_content or "Model returned empty content.")
            llm_meta = {
                "model": gen.model,
                "latency_ms": gen.latency_ms,
                "usage": gen.usage,
                "raw_content": raw_content,
            }
        except Exception as exc:  # pragma: no cover
            errors.append(f"LLM generation skipped/failed: {exc}")
            llm_answer = (
                "LLM call failed. Please verify OPENROUTER_API_KEY, ENABLE_LLM=1, and network access, "
                "then retry."
            )

    return {
        "intent": intent_result,
        "entities": entities,
        "baseline_records": baseline_records,
        "cypher_query": cypher_query,
        "embedding_hits": embedding_hits,
        "fused": fused,
        "llm_answer": llm_answer,
        "llm_meta": llm_meta,
        "errors": errors,
        "retrieval_choice": retrieval_choice,
        "model_choice": model_choice,
        "graph": _build_graph(baseline_records, embedding_hits),
    }


def _build_graph(baseline_records: List[Dict[str, Any]], embedding_hits: List[Dict[str, Any]]):
    """
    Create a lightweight graph from baseline player rows and embedding hits.
    Nodes: players; Edges: dummy similarity edges for embedding hits.
    This is a placeholder and should be replaced with real KG subgraph snippets.
    """
    if not go or not nx or not baseline_records:
        return None

    G = nx.Graph()
    for row in baseline_records:
        player = row.get("player") or row.get("player_name")
        if player:
            G.add_node(player, type="player", score=row.get("total_points", 0))

    for hit in embedding_hits:
        player = hit.get("player")
        score = hit.get("score", 0)
        if player:
            G.add_node(player, type="embedding_hit", score=score)
            # connect embedding hit to a pseudo root
            G.add_edge("query", player, weight=score)
    G.add_node("query", type="query", score=1.0)

    pos = nx.spring_layout(G, seed=42, k=0.8)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    texts = []
    colors = []
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        texts.append(f"{node} ({data.get('type')}) score={round(data.get('score', 0), 3)}")
        colors.append("#1f77b4" if data.get("type") == "player" else "#ff7f0e" if data.get("type") == "embedding_hit" else "#2ca02c")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=texts,
        marker=dict(
            showscale=False,
            color=colors,
            size=12,
            line_width=1,
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="closest",
    )
    return fig


def _clean_llm_content(text: str) -> str:
    """
    Remove common BOS/EOS tokens returned by some models and trim whitespace.
    """
    if not text:
        return text
    cleaned = re.sub(r"^<s>\s*", "", text.strip())
    cleaned = re.sub(r"\s*</s>$", "", cleaned)
    return cleaned.strip()


def main() -> None:
    global _extractor

    st.set_page_config(
        page_title="FPL Team Formulation Recommender System",
        page_icon="⚽",
        layout="wide",
    )
    st.title("FPL Team Formulation Recommender System")
    st.caption("Graph-RAG over Neo4j KG • OpenRouter LLMs • FPL-focused personas")

    driver = init_driver()
    llm_generator = init_llm()

    # Initialize entity extractor with player/team indexes from Neo4j
    if _extractor is None:
        _extractor = init_extractor(driver)
        if driver:
            player_count = len(load_player_index(driver))
            team_count = len(load_team_index(driver))
            st.success(f"Loaded {player_count} players and {team_count} teams for entity extraction.")

    if not driver:
        st.warning("Neo4j connection not available. Retrieval will use placeholders.")
    if not llm_generator or not LLM_ENABLED:
        st.warning("LLM calls are skipped (missing OpenRouter client or ENABLE_LLM not set).")

    with st.sidebar:
        st.header("Configuration")
        model_choice = st.selectbox("Model", MODELS, index=0)
        retrieval_choice = st.selectbox("Retrieval Method", RETRIEVAL_METHODS, index=0)
        st.markdown("**Example Queries**")
        for q in EXAMPLE_QUERIES:
            if st.button(q):
                st.session_state["user_query"] = q
        st.divider()
        st.markdown(
            "Backend is guarded. Set Neo4j creds in .env. To enable live LLM calls, set ENABLE_LLM=1 and ensure OPENROUTER_API_KEY is present."
        )

    user_query = st.text_area("Ask an FPL question", value=st.session_state.get("user_query", ""), height=120)
    ask = st.button("Ask")

    if ask and user_query.strip():
        st.session_state["last_query"] = user_query
        st.session_state["last_model"] = model_choice
        st.session_state["last_retrieval"] = retrieval_choice
        st.session_state["last_result"] = run_pipeline(
            query=user_query,
            retrieval_choice=retrieval_choice,
            model_choice=model_choice,
            driver=driver,
            llm_generator=llm_generator,
        )

    last_result = st.session_state.get("last_result")
    if last_result:
        st.subheader("› FPL Expert Answer")
        st.info(last_result["llm_answer"])
        meta = last_result.get("llm_meta", {})
        st.markdown(
            f"- Selected model: `{meta.get('model', st.session_state.get('last_model'))}`"
        )
        st.markdown(f"- Retrieval method: `{st.session_state.get('last_retrieval')}`")
        st.markdown(f"- Intent: `{last_result['intent'].intent}` (confidence {last_result['intent'].confidence})")
        st.markdown(f"- Parsed entities: `{last_result['entities'].__dict__}`")
        if meta.get("latency_ms"):
            st.markdown(f"- LLM latency: {meta['latency_ms']} ms; usage: {meta.get('usage')}")
        if meta.get("usage"):
            usage = meta["usage"]
            st.markdown(
                f"- Tokens → prompt: {usage.get('prompt_tokens')}, completion: {usage.get('completion_tokens')}, total: {usage.get('total_tokens')}"
            )
        if meta.get("raw_content") and meta.get("raw_content") != last_result["llm_answer"]:
            with st.expander("LLM Raw Response"):
                st.write(meta["raw_content"])

    with st.expander("Knowledge Graph Retrieved Context", expanded=False):
        if last_result and last_result["baseline_records"]:
            st.markdown("**Baseline (Cypher) Results**")
            st.dataframe(last_result["baseline_records"])
        if last_result and last_result["embedding_hits"]:
            st.markdown("**Embedding Hits**")
            st.dataframe(last_result["embedding_hits"])
        if last_result and last_result["fused"]:
            st.markdown("**Hybrid Fused Results**")
            st.dataframe(last_result["fused"])

    with st.expander("Cypher Queries Executed", expanded=False):
        st.code(last_result["cypher_query"] if last_result else "// No query yet.", language="cypher")

    with st.expander("Graph Visualization", expanded=False):
        if last_result and last_result.get("graph") and go:
            fig = last_result["graph"]
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Graph preview will render here when data and viz deps are available.")

    with st.expander("Team Recommendations", expanded=False):
        if last_result and last_result["intent"].intent == "team_recommendation" and last_result["baseline_records"]:
            st.markdown("**Recommended Players (from retrieval results)**")
            st.dataframe(last_result["baseline_records"])
        else:
            st.write("For team recommendation intents, ranked player picks will be displayed here.")

    if last_result and last_result["errors"]:
        st.error("\n".join(last_result["errors"]))


if __name__ == "__main__":
    main()
