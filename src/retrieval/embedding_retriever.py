"""
Numeric embedding retriever using player node feature vectors.

This retriever requires an anchor player name. It pulls the player's numeric
embedding from Neo4j (e.g., p.embedding_numeric) and performs vector similarity
search via db.index.vector.queryNodes on a numeric index.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingHit:
    player: str
    score: float
    properties: Dict[str, Any]


class EmbeddingRetriever:
    def __init__(
        self,
        driver,
        index_name: str = "player_embeddings_numeric",
        property_key: str = "embedding_numeric",
        top_k: int = 25,
        **_: Any,
    ):
        self.driver = driver
        self.index_name = index_name
        self.property_key = property_key
        self.top_k = top_k

    def _fetch_embedding(self, player_name: str) -> Optional[List[float]]:
        cypher = f"""
        MATCH (p:Player {{player_name: $player_name}})
        WHERE p.{self.property_key} IS NOT NULL
        RETURN p.{self.property_key} AS embedding
        """
        with self.driver.session() as session:
            row = session.run(cypher, {"player_name": player_name}).single()
            return row["embedding"] if row else None

    def search(
        self,
        anchor_player: str,
        k: Optional[int] = None,
        position: Optional[str] = None,
        exclude_players: Optional[List[str]] = None,
    ) -> List[EmbeddingHit]:
        if not anchor_player:
            raise ValueError("anchor_player is required for numeric embedding search.")
        embedding = self._fetch_embedding(anchor_player)
        if embedding is None:
            raise ValueError(f"No numeric embedding found for player '{anchor_player}'.")

        where_clauses = []
        if position:
            where_clauses.append("(node)-[:PLAYS_AS]->(:Position {name: $position})")
        if exclude_players:
            where_clauses.append("NOT node.player_name IN $exclude_players")
        where_block = ""
        if where_clauses:
            where_block = "WHERE " + " AND ".join(where_clauses)

        cypher = f"""
        CALL db.index.vector.queryNodes('{self.index_name}', $k, $embedding)
        YIELD node, score
        {where_block}
        RETURN node.player_name AS player, score, properties(node) AS props
        ORDER BY score DESC
        """
        params: Dict[str, Any] = {"embedding": embedding, "k": k or self.top_k}
        if position:
            params["position"] = position
        if exclude_players:
            params["exclude_players"] = exclude_players

        with self.driver.session() as session:
            rows = session.run(cypher, params).data()
        return [EmbeddingHit(row["player"], row["score"], row["props"]) for row in rows]
