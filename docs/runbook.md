# Runbook: Milestone 3 Graph-RAG System

## Prerequisites
- Python 3.10+
- Neo4j DB loaded with Milestone 2 KG
- `.env` with NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENROUTER_API_KEY (see `.env.example`)

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Connectivity Check
```bash
python - <<'PY'
from src.utils.neo4j_client import get_driver, verify_connection
d = get_driver()
print("Neo4j OK:", verify_connection(d))
PY
```

## Run Streamlit UI
```bash
streamlit run app/streamlit_app.py
```
- If Neo4j/OpenRouter are not configured, the UI shows warnings and uses placeholders.
- To enable live LLM calls, set `ENABLE_LLM=1` and ensure `OPENROUTER_API_KEY` is set.

## Generate Player Embeddings
```bash
python scripts/generate_embeddings.py --model bge-small --use-text
# or MPNet
python scripts/generate_embeddings.py --model mpnet --use-text
```
- Optional: `--limit 50` for smoke tests.
- Creates vector index (`player_embeddings` or `player_embeddings_mpnet`) and writes `p.embedding`.

## Run Tests
```bash
pytest -q
```

## Components Reference
- Preprocessing: `src/preprocessing/intent_classifier.py`, `entity_extractor.py`, `query_embedder.py`
- Retrieval: `src/retrieval/cypher_templates.py`, `baseline_retriever.py`, `embedding_retriever.py`, `hybrid_retriever.py`, `node_embeddings.py`
- LLM: `src/llm/openrouter_client.py`, `prompt_builder.py`, `llm_generator.py`, `quantitative_evaluator.py`
- UI: `app/streamlit_app.py`
- Docs: `docs/system_architecture.md`, `docs/retrieval_strategy.md`, `evaluation_rubric.md`
