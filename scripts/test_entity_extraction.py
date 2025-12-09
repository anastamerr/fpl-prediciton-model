"""
Test script to verify entity extraction is working correctly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.utils.neo4j_client import get_driver
from src.preprocessing.entity_extractor import FPLEntityExtractor
from src.preprocessing.intent_classifier import FPLIntentClassifier


def load_indexes(driver):
    """Load player and team names from Neo4j."""
    players = []
    teams = []
    if driver:
        with driver.session() as session:
            players = [r["name"] for r in session.run("MATCH (p:Player) RETURN p.player_name AS name").data() if r["name"]]
            teams = [r["name"] for r in session.run("MATCH (t:Team) RETURN t.name AS name").data() if r["name"]]
    return players, teams


def test_queries():
    driver = get_driver()
    players, teams = load_indexes(driver)

    print(f"Loaded {len(players)} players, {len(teams)} teams")
    print(f"Sample teams: {teams[:10]}")
    print()

    extractor = FPLEntityExtractor(player_index=players, team_index=teams)
    classifier = FPLIntentClassifier()

    test_cases = [
        "Compare Haaland and Kane",
        "Liverpool fixtures GW10",
        "Most goals 2022-23",
        "Best defenders under 5.5m",
        "Top forwards this season",
        "Who are the best midfielders under 8m?",
        "Show me Arsenal fixtures for gameweek 5",
    ]

    for query in test_cases:
        print(f"Query: {query}")
        intent = classifier.classify(query)
        entities = extractor.extract(query)
        print(f"  Intent: {intent.intent} (confidence: {intent.confidence})")
        print(f"  Players: {entities.players}")
        print(f"  Teams: {entities.teams}")
        print(f"  Positions: {entities.positions}")
        print(f"  Gameweeks: {entities.gameweeks}")
        print(f"  Seasons: {entities.seasons}")
        print(f"  Stats: {entities.statistics}")
        print(f"  Numbers: {entities.numerical_values}")
        print()

    driver.close()


if __name__ == "__main__":
    test_queries()
