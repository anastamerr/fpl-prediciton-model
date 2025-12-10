# Retrieval Strategy (Baseline, Numeric Embeddings, Hybrid)

## Baseline (Cypher)
- Templates in `src/retrieval/cypher_templates.py` cover: player performance by GW, top by position, team attack/defense, player form (per player), fixtures, season aggregates, position-based recommendations, comparisons, historical best XI, differentials, consistency, team/position filter, budget builder, and form leaders (recent top performers).
- Intent-to-template mapping in `BaselineRetriever` (see `src/retrieval/baseline_retriever.py`); params built from entity extraction with default season fallback.
- Logging/error capture: retrieval returns `RetrievalResult` with `error` populated on failure.

### Example: Top Forwards This Season
- Intent: `position_search`
- Entities: `{positions: ["FWD"], seasons: ["2022-23"], numerical_values: {limit: 5}}`
- Template: `top_players_by_position`
- Cypher (simplified):
  ```cypher
  MATCH (s:Season {season_name: $season})-[:HAS_GW]->(:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
  MATCH (p:Player)-[:PLAYS_AS]->(:Position {name: $position})
  MATCH (p)-[r:PLAYED_IN]->(f)
  WITH p, SUM(r.total_points) AS total_points
  ORDER BY total_points DESC
  LIMIT $limit
  RETURN p.player_name AS player, total_points
  ```

## Embeddings (Numeric-Only)
- Strategy: Numeric feature vectors (z-scored player stats) stored on each Player node.
- Index: `player_embeddings_numeric` on property `embedding_numeric` (dimension = number of stats). Text embeddings are not used in retrieval.
- `NodeEmbeddingGenerator` builds and persists these numeric vectors to Neo4j and ensures the numeric vector index exists.
- `EmbeddingRetriever` requires an anchor player; it fetches the anchor's numeric vector and runs `db.index.vector.queryNodes('player_embeddings_numeric', k, anchor_vector)` with optional position filter.

### Example: Similarity Query
- Input: "Find players similar to Erling Haaland".
- Flow: extract anchor player → load `embedding_numeric` → vector search on `player_embeddings_numeric` → return top-k players with scores. If no anchor player is present, embedding search is skipped and baseline/hybrid fallback applies.

## Hybrid
- `HybridRetriever` runs baseline + embedding, then fuses by reciprocal rank style weighted scoring (default 0.7 baseline / 0.3 embedding).
- Output includes fused list plus raw baseline and embedding hits for transparency.

## Error Handling and Missing Data
- Baseline: missing entities yield graceful errors in `RetrievalResult.error`.
- Embedding: if vector index or anchor player is unavailable, collector returns errors; hybrid falls back to baseline only.
- Streamlit UI surfaces errors in the FPL Expert Answer and expander sections, with placeholders when drivers/keys are absent.*** End Patch"---
