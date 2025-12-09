"""
Embedding-based retriever for player similarity search in Neo4j.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

from ..preprocessing.query_embedder import QueryEmbedder

logger = logging.getLogger(__name__)


INDEX_BY_MODEL = {
    "bge-small": {"index": "player_embeddings", "property": "embedding"},
    "mpnet": {"index": "player_embeddings_mpnet", "property": "embedding_mpnet"},
}


@dataclass
class EmbeddingHit:
    player: str
    score: float
    properties: Dict[str, Any]


class EmbeddingRetriever:
    def __init__(self, driver, model_alias: str = "bge-small", top_k: int = 25):
        self.driver = driver
        self.embedder = QueryEmbedder(model_alias=model_alias, normalize=True)
        self.top_k = top_k

    def search(
        self,
        query_text: str,
        model_alias: Optional[str] = None,
        k: Optional[int] = None,
        position: Optional[str] = None,
    ) -> List[EmbeddingHit]:
        if model_alias and model_alias != self.embedder.model_alias:
            self.embedder.switch_model(model_alias)
        index_meta = INDEX_BY_MODEL.get(self.embedder.model_alias)
        if not index_meta:
            raise ValueError(f"No vector index mapping for model {self.embedder.model_alias}")

        embedding = self.embedder.embed(query_text).tolist()
        index_name = index_meta["index"]
        top_k = k or self.top_k

        position_clause = ""
        if position:
            position_clause = "WHERE (node)-[:PLAYS_AS]->(:Position {name: $position})"

        # Embed index name directly as Neo4j may not support parameterized index names
        # in db.index.vector.queryNodes for all versions
        cypher = f"""
        CALL db.index.vector.queryNodes('{index_name}', {top_k}, $embedding)
        YIELD node, score
        {position_clause}
        RETURN node.player_name AS player, score, properties(node) AS props
        ORDER BY score DESC
        """
        params = {"embedding": embedding}
        if position:
            params["position"] = position

        with self.driver.session() as session:
            rows = session.run(cypher, params).data()
        return [EmbeddingHit(row["player"], row["score"], row["props"]) for row in rows]
