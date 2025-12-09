# Error Analysis (Current State)

## Retrieval
- Team attack/defense templates rely on per-fixture aggregates without explicit player→team edges, so results are approximate. Needs team linkage or per-team stats on fixtures.
- Embedding search relevance varies by model; MPNet shows different bias vs BGE. Position filter works but scores are uncalibrated.
- Hybrid fusion uses simple reciprocal-rank weighting; no learning-to-rank or feature-level fusion yet.

## LLM
- LLM path is gated by `ENABLE_LLM`; no automated hallucination checks yet.
- No guardrails on output length/format beyond prompt instructions.

## UI
- Graph viz uses placeholder nodes/edges (players + query) until true KG subgraph is passed through.
- Team recommendation table reuses baseline results; needs richer stats/reasons.

## Data Gaps
- Missing explicit Player→Team relationships in KG limits accurate team-level queries (attack/defense, ownership, etc.).
- Vector search doesn’t filter by season; embeddings are season-agnostic aggregates.

## Next Actions
- Add Player→Team edges or inject team into PLAYED_IN to fix team-level aggregates.
- Add season filters to embeddings or generate season-specific embeddings.
- Implement hallucination checks (string match to context) and add quantitative accuracy scoring.
