"""
CLI to generate and persist player embeddings into Neo4j.

Usage:
    python scripts/generate_embeddings.py --model bge-small --use-text --limit 100

Requires .env with NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD (and model weights available).
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.neo4j_client import get_driver  # noqa: E402
from src.retrieval.node_embeddings import NodeEmbeddingGenerator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate player embeddings and store in Neo4j.")
    parser.add_argument("--model", choices=["bge-small", "mpnet"], default="bge-small", help="Embedding model alias.")
    parser.add_argument(
        "--index-name",
        default=None,
        help="Vector index name. Defaults to player_embeddings or player_embeddings_mpnet.",
    )
    parser.add_argument("--use-text", action="store_true", help="Use textual feature descriptions instead of numeric.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of players for quick runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    is_mpnet = args.model == "mpnet"
    index_name = args.index_name or ("player_embeddings_mpnet" if is_mpnet else "player_embeddings")
    property_key = "embedding_mpnet" if is_mpnet else "embedding"

    driver = get_driver()
    generator = NodeEmbeddingGenerator(driver, model_alias=args.model)
    count = generator.persist_embeddings(
        index_name=index_name, property_key=property_key, use_text=args.use_text, limit=args.limit
    )
    print(
        f"Embedded {count} players into index '{index_name}' using model '{args.model}' "
        f"stored on property '{property_key}'."
    )


if __name__ == "__main__":
    main()
