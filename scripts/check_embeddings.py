"""
Diagnostic script to verify embeddings are properly stored in Neo4j.

Usage:
    python scripts/check_embeddings.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.neo4j_client import get_driver, verify_connection


def check_embeddings():
    driver = get_driver()
    if not verify_connection(driver):
        print("ERROR: Cannot connect to Neo4j")
        return

    with driver.session() as session:
        # Check total players
        total = session.run("MATCH (p:Player) RETURN count(p) AS count").single()["count"]
        print(f"Total Player nodes: {total}")

        # Check players with BGE embeddings
        bge_count = session.run(
            "MATCH (p:Player) WHERE p.embedding IS NOT NULL RETURN count(p) AS count"
        ).single()["count"]
        print(f"Players with BGE embeddings (p.embedding): {bge_count}")

        # Check players with MPNet embeddings
        mpnet_count = session.run(
            "MATCH (p:Player) WHERE p.embedding_mpnet IS NOT NULL RETURN count(p) AS count"
        ).single()["count"]
        print(f"Players with MPNet embeddings (p.embedding_mpnet): {mpnet_count}")

        # Check vector indexes
        print("\n--- Vector Indexes ---")
        indexes = session.run("SHOW INDEXES WHERE type = 'VECTOR'").data()
        if indexes:
            for idx in indexes:
                print(f"  Index: {idx.get('name', 'N/A')}, State: {idx.get('state', 'N/A')}, "
                      f"Population: {idx.get('populationPercent', 'N/A')}%")
        else:
            print("  No vector indexes found!")

        # Sample a player with embedding
        print("\n--- Sample Player with Embedding ---")
        sample = session.run(
            "MATCH (p:Player) WHERE p.embedding IS NOT NULL "
            "RETURN p.player_name AS name, size(p.embedding) AS dim LIMIT 1"
        ).single()
        if sample:
            print(f"  Player: {sample['name']}, Embedding dimension: {sample['dim']}")
        else:
            print("  No players with embeddings found!")

        # Check positions
        print("\n--- Positions in Database ---")
        positions = session.run("MATCH (pos:Position) RETURN pos.name AS name").data()
        print(f"  Positions: {[p['name'] for p in positions]}")

        # Check FWD players count
        fwd_count = session.run(
            "MATCH (p:Player)-[:PLAYS_AS]->(:Position {name: 'FWD'}) RETURN count(p) AS count"
        ).single()["count"]
        print(f"\n  Forward players (FWD): {fwd_count}")

        # Check FWD players with embeddings
        fwd_emb_count = session.run(
            "MATCH (p:Player)-[:PLAYS_AS]->(:Position {name: 'FWD'}) "
            "WHERE p.embedding IS NOT NULL RETURN count(p) AS count"
        ).single()["count"]
        print(f"  Forward players with embeddings: {fwd_emb_count}")

    driver.close()
    print("\n--- Recommendations ---")
    if bge_count == 0:
        print("  Run: python scripts/generate_embeddings.py --model bge-small --use-text")
    if mpnet_count == 0:
        print("  Run: python scripts/generate_embeddings.py --model mpnet --use-text")
    if bge_count > 0 and mpnet_count > 0:
        print("  Embeddings look good! Try the Streamlit app.")


if __name__ == "__main__":
    check_embeddings()
