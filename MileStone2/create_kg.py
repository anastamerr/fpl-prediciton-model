
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from neo4j import Driver, GraphDatabase

# Columns that should be parsed as integers for relationship properties or keys.
INT_COLUMNS = [
    "assists",
    "bonus",
    "clean_sheets",
    "element",
    "fixture",
    "goals_conceded",
    "goals_scored",
    "minutes",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "red_cards",
    "saves",
    "team_a_score",
    "team_h_score",
    "total_points",
    "transfers_balance",
    "transfers_in",
    "transfers_out",
    "value",
    "yellow_cards",
    "GW",
    "bps",
]

# Columns that should be parsed as floats.
FLOAT_COLUMNS = ["influence", "creativity", "threat", "ict_index", "form"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the FPL knowledge graph in Neo4j.")
    parser.add_argument("--config", default="config.txt", help="Path to config file with URI/USERNAME/PASSWORD.")
    parser.add_argument("--csv", default="fpl_two_seasons.csv", help="Path to the FPL CSV export.")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for write operations.")
    parser.add_argument("--reset", action="store_true", help="Clear the existing graph before loading.")
    parser.add_argument(
        "--skip-scoring", action="store_true", help="Skip the defender scoring consistency check after load."
    )
    return parser.parse_args()


def read_config(config_path: Path) -> Dict[str, str]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config: Dict[str, str] = {}
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip()

    missing = [key for key in ("URI", "USERNAME", "PASSWORD") if key not in config]
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(missing)}")
    return config


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in FLOAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    df["kickoff_time"] = (
        pd.to_datetime(df["kickoff_time"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")
    )

    str_columns = ["season", "name", "position", "home_team", "away_team"]
    for col in str_columns:
        df[col] = df[col].astype(str)
    return df


def chunked(seq: List[Dict], size: int) -> Iterable[List[Dict]]:
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def create_constraints(session) -> None:
    constraint_statements = [
        "CREATE CONSTRAINT season_name_unique IF NOT EXISTS FOR (s:Season) REQUIRE s.season_name IS UNIQUE",
        "CREATE CONSTRAINT gameweek_key IF NOT EXISTS FOR (g:Gameweek) REQUIRE (g.season, g.GW_number) IS NODE KEY",
        "CREATE CONSTRAINT fixture_key IF NOT EXISTS FOR (f:Fixture) REQUIRE (f.season, f.fixture_number) IS NODE KEY",
        "CREATE CONSTRAINT team_name_unique IF NOT EXISTS FOR (t:Team) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT position_name_unique IF NOT EXISTS FOR (p:Position) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT player_key IF NOT EXISTS FOR (p:Player) REQUIRE (p.player_name, p.player_element) IS NODE KEY",
    ]
    for statement in constraint_statements:
        session.run(statement)


def clear_database(session) -> None:
    session.run("MATCH (n) DETACH DELETE n")


def build_nodes(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    seasons = [{"season_name": season} for season in sorted(df["season"].unique())]
    positions = [{"name": pos} for pos in sorted(df["position"].unique())]

    players_df = df[["name", "element"]].drop_duplicates()
    players = [
        {"player_name": row.name, "player_element": int(row.element)} for row in players_df.itertuples(index=False)
    ]

    plays_as_df = df[["name", "element", "position"]].drop_duplicates()
    plays_as = [
        {"player_name": row.name, "player_element": int(row.element), "position": row.position}
        for row in plays_as_df.itertuples(index=False)
    ]

    gameweeks_df = df[["season", "GW"]].drop_duplicates()
    gameweeks = [{"season": row.season, "GW_number": int(row.GW)} for row in gameweeks_df.itertuples(index=False)]

    fixtures_df = df[["season", "GW", "fixture", "kickoff_time", "home_team", "away_team"]].drop_duplicates(
        subset=["season", "fixture"]
    )
    fixtures = [
        {
            "season": row.season,
            "GW_number": int(row.GW),
            "fixture_number": int(row.fixture),
            "kickoff_time": row.kickoff_time,
            "home_team": row.home_team,
            "away_team": row.away_team,
        }
        for row in fixtures_df.itertuples(index=False)
    ]

    played_in_df = df[
        [
            "season",
            "fixture",
            "name",
            "element",
            "minutes",
            "goals_scored",
            "assists",
            "total_points",
            "bonus",
            "clean_sheets",
            "goals_conceded",
            "own_goals",
            "penalties_saved",
            "penalties_missed",
            "yellow_cards",
            "red_cards",
            "saves",
            "bps",
            "influence",
            "creativity",
            "threat",
            "ict_index",
            "form",
        ]
    ]
    played_in_rows = [
        {
            "season": row.season,
            "fixture_number": int(row.fixture),
            "player_name": row.name,
            "player_element": int(row.element),
            "minutes": int(row.minutes),
            "goals_scored": int(row.goals_scored),
            "assists": int(row.assists),
            "total_points": int(row.total_points),
            "bonus": int(row.bonus),
            "clean_sheets": int(row.clean_sheets),
            "goals_conceded": int(row.goals_conceded),
            "own_goals": int(row.own_goals),
            "penalties_saved": int(row.penalties_saved),
            "penalties_missed": int(row.penalties_missed),
            "yellow_cards": int(row.yellow_cards),
            "red_cards": int(row.red_cards),
            "saves": int(row.saves),
            "bps": int(row.bps),
            "influence": float(row.influence),
            "creativity": float(row.creativity),
            "threat": float(row.threat),
            "ict_index": float(row.ict_index),
            "form": float(row.form),
        }
        for row in played_in_df.itertuples(index=False)
    ]

    return {
        "seasons": seasons,
        "positions": positions,
        "players": players,
        "plays_as": plays_as,
        "gameweeks": gameweeks,
        "fixtures": fixtures,
        "played_in": played_in_rows,
    }


def write_positions(session, positions: List[Dict]) -> None:
    session.run("UNWIND $rows AS row MERGE (:Position {name: row.name})", rows=positions)


def write_seasons(session, seasons: List[Dict]) -> None:
    session.run("UNWIND $rows AS row MERGE (:Season {season_name: row.season_name})", rows=seasons)


def write_gameweeks(session, gameweeks: List[Dict], batch_size: int) -> None:
    query = """
    UNWIND $rows AS row
    MATCH (s:Season {season_name: row.season})
    MERGE (gw:Gameweek {season: row.season, GW_number: row.GW_number})
    MERGE (s)-[:HAS_GW]->(gw)
    """
    for batch in chunked(gameweeks, batch_size):
        session.run(query, rows=batch)


def write_fixtures(session, fixtures: List[Dict], batch_size: int) -> None:
    query = """
    UNWIND $rows AS row
    MATCH (gw:Gameweek {season: row.season, GW_number: row.GW_number})
    MERGE (f:Fixture {season: row.season, fixture_number: row.fixture_number})
    SET f.kickoff_time = row.kickoff_time
    MERGE (gw)-[:HAS_FIXTURE]->(f)
    MERGE (home:Team {name: row.home_team})
    MERGE (away:Team {name: row.away_team})
    MERGE (f)-[:HAS_HOME_TEAM]->(home)
    MERGE (f)-[:HAS_AWAY_TEAM]->(away)
    """
    for batch in chunked(fixtures, batch_size):
        session.run(query, rows=batch)


def write_players(session, players: List[Dict], batch_size: int) -> None:
    query = "UNWIND $rows AS row MERGE (:Player {player_name: row.player_name, player_element: row.player_element})"
    for batch in chunked(players, batch_size):
        session.run(query, rows=batch)


def write_plays_as(session, plays_as: List[Dict], batch_size: int) -> None:
    query = """
    UNWIND $rows AS row
    MATCH (p:Player {player_name: row.player_name, player_element: row.player_element})
    MATCH (pos:Position {name: row.position})
    MERGE (p)-[:PLAYS_AS]->(pos)
    """
    for batch in chunked(plays_as, batch_size):
        session.run(query, rows=batch)


def write_played_in(session, played_in: List[Dict], batch_size: int) -> None:
    query = """
    UNWIND $rows AS row
    MATCH (p:Player {player_name: row.player_name, player_element: row.player_element})
    MATCH (f:Fixture {season: row.season, fixture_number: row.fixture_number})
    MERGE (p)-[r:PLAYED_IN]->(f)
    SET r.minutes = row.minutes,
        r.goals_scored = row.goals_scored,
        r.assists = row.assists,
        r.total_points = row.total_points,
        r.bonus = row.bonus,
        r.clean_sheets = row.clean_sheets,
        r.goals_conceded = row.goals_conceded,
        r.own_goals = row.own_goals,
        r.penalties_saved = row.penalties_saved,
        r.penalties_missed = row.penalties_missed,
        r.yellow_cards = row.yellow_cards,
        r.red_cards = row.red_cards,
        r.saves = row.saves,
        r.bps = row.bps,
        r.influence = row.influence,
        r.creativity = row.creativity,
        r.threat = row.threat,
        r.ict_index = row.ict_index,
        r.form = row.form
    """
    for batch in chunked(played_in, batch_size):
        session.run(query, rows=batch)


def calculate_defender_points(props: Dict) -> int:
    minutes = int(props.get("minutes", 0))
    playing_points = 0 if minutes == 0 else 1 if minutes < 60 else 2
    assist_points = int(props.get("assists", 0)) * 3
    penalty_miss_points = int(props.get("penalties_missed", 0)) * -2
    own_goal_points = int(props.get("own_goals", 0)) * -2
    yellow_card_points = int(props.get("yellow_cards", 0)) * -1
    red_card_points = int(props.get("red_cards", 0)) * -3
    bonus_points = int(props.get("bonus", 0))

    goal_points = int(props.get("goals_scored", 0)) * 6
    clean_sheet_points = 4 if int(props.get("clean_sheets", 0)) > 0 and minutes > 60 else 0
    goals_conceded_points = -(int(props.get("goals_conceded", 0)) // 2)

    return (
        playing_points
        + assist_points
        + penalty_miss_points
        + own_goal_points
        + yellow_card_points
        + red_card_points
        + bonus_points
        + goal_points
        + clean_sheet_points
        + goals_conceded_points
    )


def detect_position_inconsistencies(driver: Driver, limit: Optional[int] = None) -> List[Dict]:
    findings: List[Dict] = []
    with driver.session() as session:
        candidate_players = session.run(
            """
            MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
            WITH p, COLLECT(pos.name) AS positions
            WHERE 'DEF' IN positions AND size(positions) > 1
            RETURN p.player_name AS player_name, p.player_element AS player_element, positions
            """
        ).data()

        for candidate in candidate_players:
            player_name = candidate["player_name"]
            player_element = candidate["player_element"]
            fixture_rows = session.run(
                """
                MATCH (p:Player {player_name: $player_name, player_element: $player_element})-[r:PLAYED_IN]->(f:Fixture)
                MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)<-[:HAS_GW]-(s:Season)
                RETURN s.season_name AS season, f.fixture_number AS fixture_number, r{.*} AS rel
                ORDER BY s.season_name, f.fixture_number
                """,
                player_name=player_name,
                player_element=player_element,
            ).data()

            if not fixture_rows:
                continue

            mismatches = []
            for entry in fixture_rows:
                rel_props = entry["rel"]
                def_score = calculate_defender_points(rel_props)
                total_points = int(rel_props.get("total_points", 0))
                matches_def = def_score == total_points
                if not matches_def:
                    mismatches.append(
                        {
                            "season": entry["season"],
                            "fixture_number": entry["fixture_number"],
                            "total_points": total_points,
                            "def_score": def_score,
                            "matches_def": matches_def,
                        }
                    )

            if mismatches:
                findings.append(
                    {
                        "player": player_name,
                        "player_element": player_element,
                        "not_def_fixtures": len(mismatches),
                        "total_fixtures": len(fixture_rows),
                        "fixture_details": mismatches,
                    }
                )
            if limit and len(findings) >= limit:
                break
    return findings


def load_graph(driver: Driver, df: pd.DataFrame, batch_size: int, reset: bool) -> None:
    nodes = build_nodes(df)
    with driver.session() as session:
        if reset:
            clear_database(session)
        create_constraints(session)
        write_seasons(session, nodes["seasons"])
        write_gameweeks(session, nodes["gameweeks"], batch_size)
        write_fixtures(session, nodes["fixtures"], batch_size)
        write_positions(session, nodes["positions"])
        write_players(session, nodes["players"], batch_size)
        write_plays_as(session, nodes["plays_as"], batch_size)
        write_played_in(session, nodes["played_in"], batch_size)


def main() -> None:
    args = parse_args()
    config = read_config(Path(args.config))
    df = load_dataframe(Path(args.csv))

    driver = GraphDatabase.driver(config["URI"], auth=(config["USERNAME"], config["PASSWORD"]))
    try:
        load_graph(driver, df, batch_size=args.batch_size, reset=args.reset)
        if not args.skip_scoring:
            inconsistencies = detect_position_inconsistencies(driver)
            if inconsistencies:
                print(f"Potential defender mislabels found for {len(inconsistencies)} players.")
                for finding in inconsistencies[:5]:
                    print(
                        f"- {finding['player']} ({finding['player_element']}): "
                        f"{finding['not_def_fixtures']}/{finding['total_fixtures']} fixtures differ"
                    )
            else:
                print("No defender position inconsistencies detected.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
