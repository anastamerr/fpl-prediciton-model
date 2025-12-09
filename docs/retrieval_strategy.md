# Retrieval Strategy (Baseline, Embeddings, Hybrid)

## Baseline (Cypher)
- Templates in `src/retrieval/cypher_templates.py` cover: player performance by GW, top by position, team attack/defense, player form, fixtures, season aggregates, position-based recommendations, comparisons, historical best XI, differentials, consistency, team/position filter, budget builder.
- Intent→template mapping in `BaselineRetriever` (see `src/retrieval/baseline_retriever.py`); params built from entity extraction.
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

## Embeddings
- Query embedding via `QueryEmbedder` (BGE 384d or MPNet 768d, normalized). Vector indexes expected: `player_embeddings` (BGE) and `player_embeddings_mpnet` (MPNet).
- `NodeEmbeddingGenerator` builds per-player embeddings from aggregated PLAYED_IN stats, persists to Neo4j, and creates vector index.
- `EmbeddingRetriever` runs `db.index.vector.queryNodes` with optional position filter.

### Example: Similar Forwards Query
- Input: "Best forwards in form" with position filter "FWD".
- Embed query → vector → vector search on `player_embeddings`.
- Return top-k players with scores and properties.

## Hybrid
- `HybridRetriever` runs baseline + embedding, then fuses by reciprocal rank style weighted scoring (default 0.7 baseline / 0.3 embedding).
- Output includes fused list plus raw baseline and embedding hits for transparency.

## Error Handling and Missing Data
- Baseline: missing entities yield graceful errors in `RetrievalResult.error`.
- Embedding: if vector index or model unavailable, collector returns errors; hybrid falls back to baseline only.
- Streamlit UI surfaces errors in the “FPL Expert Answer” and expander sections, with placeholders when drivers/keys are absent.
