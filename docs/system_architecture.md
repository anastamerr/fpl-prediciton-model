# System Architecture (Milestone 3)

## Overview
- Goal: Graph-RAG recommender for FPL built on the Neo4j KG from Milestone 2.
- Layers:
  1) Preprocessing: intent classification, entity extraction, query embedding.
  2) Retrieval: baseline Cypher templates, embedding search, hybrid fusion.
  3) LLM: prompt builder, OpenRouter client, response generator, evaluation.
  4) UI: Streamlit app exposing model/retrieval choices and KG transparency.

## Data/KG
- Schema (from Milestone 2):
  - Season(season_name) -[:HAS_GW]-> Gameweek {season, GW_number}
  - Gameweek -[:HAS_FIXTURE]-> Fixture {season, fixture_number, kickoff_time, home_team, away_team}
  - Fixture -[:HAS_HOME_TEAM]/[:HAS_AWAY_TEAM]-> Team {name}
  - Player -[:PLAYS_AS]-> Position {name}
  - Player -[:PLAYED_IN {stats...}]-> Fixture (minutes, goals_scored, assists, total_points, bonus, etc.)
- Vector indexes: `player_embeddings` (BGE, dim 384) and `player_embeddings_mpnet` (MPNet, dim 768) planned for players.

## Preprocessing
- Intent classifier: `src/preprocessing/intent_classifier.py` (rule-based, 10 intents).
- Entity extractor: `src/preprocessing/entity_extractor.py` (regex + fuzzy matching for players/teams/positions/seasons/GWs/stats/budget).
- Query embedder: `src/preprocessing/query_embedder.py` (BGE/MPNet switchable, normalized embeddings, caching).

## Retrieval
- Cypher templates: `src/retrieval/cypher_templates.py` (10+ queries including performance, top-by-position, team attack/defense, form, fixtures, aggregates, comparisons, best XI, differentials, consistency, team/position filters, budget builder).
- Baseline retriever: `src/retrieval/baseline_retriever.py` (intent→template mapping, param builders, logging).
- Embedding retriever: `src/retrieval/embedding_retriever.py` (vector search against Neo4j vector indexes).
- Node embeddings: `src/retrieval/node_embeddings.py` (aggregate stats→embedding; index creation/persist).
- Hybrid retriever: `src/retrieval/hybrid_retriever.py` (fuses baseline+embedding results via reciprocal-rank style scoring).

## LLM Layer
- OpenRouter client: `src/llm/openrouter_client.py` (4 models, retries).
- Prompt builder: `src/llm/prompt_builder.py` (persona + context + task).
- Generator: `src/llm/llm_generator.py`.
- Quantitative evaluator scaffold: `src/llm/quantitative_evaluator.py`.
- Qualitative rubric: `evaluation_rubric.md`.

## UI
- Streamlit app: `app/streamlit_app.py` with controls for model/retrieval selection, example queries, and placeholder sections for KG context, Cypher, graph viz, team recommendations, and LLM answer.

## Config/Infra
- Requirements in `requirements.txt`; Neo4j helper in `src/utils/neo4j_client.py` (loads `.env`, provides health check).
- Expected env vars: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `OPENROUTER_API_KEY`.

## Execution Flow (planned)
1) User query enters UI.
2) Preprocessing: intent classifier + entity extractor (+ embedding if embedding/hybrid).
3) Retrieval:
   - Baseline: pick Cypher template, run against KG.
   - Embedding: embed query, vector search players (optionally filter by position).
   - Hybrid: fuse results.
4) Prompting: build persona/context/task prompt from KG results.
5) LLM call via OpenRouter; return answer and metadata.
6) UI displays KG context, Cypher queries, subgraph viz, and answer with timings/tokens.
