"""
Hybrid retriever: combines baseline Cypher results with embedding search hits.
"""

from dataclasses import dataclass
from typing import Any, Dict, List
import logging

from .baseline_retriever import BaselineRetriever, RetrievalResult
from .embedding_retriever import EmbeddingRetriever, EmbeddingHit

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    fused: List[Dict[str, Any]]
    baseline: RetrievalResult
    embedding: List[EmbeddingHit]


class HybridRetriever:
    def __init__(self, baseline: BaselineRetriever, embedding: EmbeddingRetriever, weight_baseline: float = 0.7):
        self.baseline = baseline
        self.embedding = embedding
        self.weight_baseline = weight_baseline

    def retrieve(self, intent: str, entities: Dict[str, Any], user_query: str) -> HybridResult:
        baseline_result = self.baseline.retrieve(intent=intent, entities=entities)
        embedding_hits: List[EmbeddingHit] = []
        try:
            # Extract position filter from entities if available
            positions = entities.get("positions", [])
            position_filter = positions[0] if positions else None
            embedding_hits = self.embedding.search(user_query, position=position_filter)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Embedding search failed: %s", exc)

        fused = self._fuse_player_results(baseline_result, embedding_hits)
        return HybridResult(fused=fused, baseline=baseline_result, embedding=embedding_hits)

    def _fuse_player_results(
        self, baseline_result: RetrievalResult, embedding_hits: List[EmbeddingHit]
    ) -> List[Dict[str, Any]]:
        """
        Simple reciprocal-rank style fusion on player names.
        """
        scores: Dict[str, float] = {}

        for rank, row in enumerate(baseline_result.records):
            name = row.get("player") or row.get("player_name")
            if not name:
                continue
            scores[name] = scores.get(name, 0.0) + self.weight_baseline / (rank + 1)

        for rank, hit in enumerate(embedding_hits):
            scores[hit.player] = scores.get(hit.player, 0.0) + (1 - self.weight_baseline) * (1 / (rank + 1))

        fused = [
            {
                "player": name,
                "score": round(score, 4),
                "baseline_present": any(
                    (row.get("player") == name or row.get("player_name") == name)
                    for row in baseline_result.records
                ),
                "embedding_present": any(hit.player == name for hit in embedding_hits),
            }
            for name, score in scores.items()
        ]
        return sorted(fused, key=lambda x: x["score"], reverse=True)
