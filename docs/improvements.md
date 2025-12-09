# Improvement Plan
- Add Player→Team relationships (or encode team on PLAYED_IN) to support accurate team attack/defense, ownership, and squad-aware analytics.
- Generate season-specific embeddings or add season as a filter to vector search; evaluate BGE vs MPNet per intent.
- Enhance hybrid fusion with learning-to-rank or weighted feature fusion; calibrate scores.
- Upgrade graph visualization to include fixtures, teams, players, and positions with legends and hover stats.
- Implement recommendation reasoning with key stats (points, form, budget) and formation-aware selection.
- Run full quantitative/qualitative LLM evaluations; track latency/tokens/cost and hallucination rate.
- Add caching for retrieval results and model outputs; add rate-limit handling dashboards.
