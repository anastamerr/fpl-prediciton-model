"""
Test script to debug embedding search directly.

Usage:
    python scripts/test_embedding_search.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.neo4j_client import get_driver
from src.preprocessing.query_embedder import QueryEmbedder


def test_vector_search():
    driver = get_driver()
    embedder = QueryEmbedder(model_alias="bge-small", normalize=True)

    # Test query
    query = "who are the top forwards this season"
    print(f"Query: {query}")
    print(f"Embedding model: bge-small (384 dimensions)")

    embedding = embedder.embed(query).tolist()
    print(f"Query embedding generated, length: {len(embedding)}")

    with driver.session() as session:
        # Test 1: Basic vector search without position filter
        print("\n--- Test 1: Basic vector search (no filter) ---")
        cypher1 = """
        CALL db.index.vector.queryNodes('player_embeddings', 10, $embedding)
        YIELD node, score
        RETURN node.player_name AS player, score
        ORDER BY score DESC
        """
        results1 = session.run(cypher1, embedding=embedding).data()
        print(f"Results: {len(results1)} players")
        for r in results1[:5]:
            print(f"  {r['player']}: {r['score']:.4f}")

        # Test 2: Vector search with position filter using WHERE
        print("\n--- Test 2: Vector search with WHERE position filter ---")
        cypher2 = """
        CALL db.index.vector.queryNodes('player_embeddings', 20, $embedding)
        YIELD node, score
        WHERE (node)-[:PLAYS_AS]->(:Position {name: 'FWD'})
        RETURN node.player_name AS player, score
        ORDER BY score DESC
        """
        results2 = session.run(cypher2, embedding=embedding).data()
        print(f"Results: {len(results2)} forwards")
        for r in results2[:5]:
            print(f"  {r['player']}: {r['score']:.4f}")

        # Test 3: Vector search with MATCH position filter (alternative approach)
        print("\n--- Test 3: Vector search with MATCH position filter ---")
        cypher3 = """
        CALL db.index.vector.queryNodes('player_embeddings', 20, $embedding)
        YIELD node, score
        WITH node, score
        MATCH (node)-[:PLAYS_AS]->(:Position {name: 'FWD'})
        RETURN node.player_name AS player, score
        ORDER BY score DESC
        """
        results3 = session.run(cypher3, embedding=embedding).data()
        print(f"Results: {len(results3)} forwards")
        for r in results3[:5]:
            print(f"  {r['player']}: {r['score']:.4f}")

        # Test 4: Check what a sample forward's embedding looks like
        print("\n--- Test 4: Sample forward player embedding check ---")
        sample = session.run("""
            MATCH (p:Player)-[:PLAYS_AS]->(:Position {name: 'FWD'})
            WHERE p.embedding IS NOT NULL
            RETURN p.player_name AS name, size(p.embedding) AS dim
            ORDER BY p.player_name
            LIMIT 5
        """).data()
        for s in sample:
            print(f"  {s['name']}: {s['dim']} dimensions")

    driver.close()


if __name__ == "__main__":
    test_vector_search()
