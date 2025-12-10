"""
Test script to debug numeric embedding search directly.

Usage:
    python scripts/test_embedding_search.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.neo4j_client import get_driver


def test_vector_search(anchor_player: str = "Erling Haaland"):
    driver = get_driver()
    with driver.session() as session:
        row = session.run(
            """
            MATCH (p:Player {player_name: $name})
            RETURN p.embedding_numeric AS emb
            """,
            name=anchor_player,
        ).single()
        if not row or not row["emb"]:
            print(f"No numeric embedding found for {anchor_player}")
            return
        embedding = row["emb"]
        print(f"Anchor: {anchor_player}, embedding dim={len(embedding)}")

        cypher = """
        CALL db.index.vector.queryNodes('player_embeddings_numeric', 10, $embedding)
        YIELD node, score
        RETURN node.player_name AS player, score
        ORDER BY score DESC
        """
        results = session.run(cypher, embedding=embedding).data()
        print(f"Results: {len(results)} players")
        for r in results[:10]:
            print(f"  {r['player']}: {r['score']:.4f}")

    driver.close()


if __name__ == "__main__":
    test_vector_search()
