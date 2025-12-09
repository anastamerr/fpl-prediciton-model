# FPL Team Formulation Recommender System (Milestone 3)

Graph-RAG system using the Neo4j KG from Milestone 2 as grounding for LLM-backed FPL answers and team recommendations.

## Quick Start
1) **Install deps**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
2) **Configure env** (copy `.env.example` → `.env`)
   - `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
   - `OPENROUTER_API_KEY` (for LLMs)
   - `ENABLE_LLM=1` to allow LLM calls (optional)
3) **Verify Neo4j**
   ```bash
   python - <<'PY'
   from src.utils.neo4j_client import get_driver, verify_connection
   d=get_driver(); print("Neo4j OK:", verify_connection(d))
   PY
   ```
4) **Generate embeddings** (required for embedding/hybrid retrieval)
   ```bash
   python scripts/generate_embeddings.py --model bge-small --use-text
   python scripts/generate_embeddings.py --model mpnet --use-text
   ```
5) **Run UI**
   ```bash
   streamlit run app/streamlit_app.py
   ```
   - Choose retrieval method/model in sidebar.
   - With `ENABLE_LLM=1`, LLM answers will be generated; otherwise placeholders show.

## Repository Map
- Preprocessing: `src/preprocessing/intent_classifier.py`, `entity_extractor.py`, `query_embedder.py`
- Retrieval: `src/retrieval/cypher_templates.py`, `baseline_retriever.py`, `embedding_retriever.py`, `hybrid_retriever.py`, `node_embeddings.py`
- LLM: `src/llm/openrouter_client.py`, `prompt_builder.py`, `llm_generator.py`, `quantitative_evaluator.py`
- UI: `app/streamlit_app.py` (tables for baseline/embedding/hybrid, optional graph viz, team reco table)
- Scripts: `scripts/generate_embeddings.py`, `scripts/run_quant_eval.py`
- Docs: `docs/system_architecture.md`, `docs/retrieval_strategy.md`, `docs/runbook.md`, `docs/evaluation_plan.md`, `docs/error_analysis.md`, `docs/improvements.md`, `docs/limitations.md`
- Data seeds: `data/eval_questions.json`, `data/sample_questions.json`

## Running Evaluations
- Quantitative: populate `context`/`gold_answer` in `data/eval_questions.json`, then run `scripts/run_quant_eval.py --eval-file data/eval_questions.json` (needs `ENABLE_LLM=1`).
- Qualitative: use `evaluation_rubric.md` with ~15 queries; score 1–5 across 7 dimensions.

## Notes / Known Gaps
- Team-level attack/defense aggregates are approximate because the KG lacks explicit Player→Team links; improvement plan documented in `docs/improvements.md`.
- Graph viz is placeholder; feed real KG subgraphs for richer visuals.
- LLM eval results are not precomputed; scripts and rubrics are provided to run them.
