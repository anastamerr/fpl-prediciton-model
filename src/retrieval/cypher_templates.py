"""
Collection of Cypher query templates for FPL KG retrieval.

Each template returns a (query, params) tuple and enforces required parameters.
Templates are aligned with the graph produced in Milestone 2:
- Season(season_name) -> [:HAS_GW] -> Gameweek {season, GW_number}
- Gameweek -> [:HAS_FIXTURE] -> Fixture {season, fixture_number, kickoff_time, home_team, away_team}
- Player -> [:PLAYED_IN] -> Fixture (stats on relationship)
- Player -> [:PLAYS_AS] -> Position {name}
- Fixture -> [:HAS_HOME_TEAM]/[:HAS_AWAY_TEAM] -> Team {name}
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable


class MissingParameterError(ValueError):
    """Raised when a required parameter is missing."""


def _require(params: Dict, keys: List[str]) -> None:
    missing = [k for k in keys if k not in params or params[k] in (None, "")]
    if missing:
        raise MissingParameterError(f"Missing required parameters: {', '.join(missing)}")


def player_performance_by_gameweek(player_name: str, season: str, gameweek: int) -> Tuple[str, Dict]:
    _require(locals(), ["player_name", "season", "gameweek"])
    query = """
    MATCH (s:Season {season_name: $season})-[:HAS_GW]->(gw:Gameweek {GW_number: $gameweek})
    MATCH (gw)-[:HAS_FIXTURE]->(f:Fixture)
    MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f)
    RETURN p.player_name AS player,
           gw.GW_number AS gameweek,
           f.fixture_number AS fixture,
           r.minutes AS minutes,
           r.goals_scored AS goals,
           r.assists AS assists,
           r.total_points AS points,
           r.bonus AS bonus,
           r.clean_sheets AS clean_sheets,
           r.goals_conceded AS goals_conceded,
           r.yellow_cards AS yellow_cards,
           r.red_cards AS red_cards,
           r.saves AS saves
    ORDER BY f.fixture_number
    """
    return query, {"player_name": player_name, "season": season, "gameweek": gameweek}


def top_players_by_position(position: str, season: str, limit: int = 10, max_price: float = None) -> Tuple[str, Dict]:
    _require(locals(), ["position", "season"])
    # Note: Price filtering requires 'now_cost' property on Player nodes.
    # If not available, filter is ignored and all players returned by points.
    price_filter = ""
    price_note = ""
    if max_price is not None:
        price_filter = "WHERE p.now_cost IS NOT NULL AND p.now_cost <= $max_price"
        price_note = "// Note: Price filter applied only if now_cost property exists"

    query = f"""
    MATCH (s:Season {{season_name: $season}})-[:HAS_GW]->(:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
    MATCH (p:Player)-[:PLAYS_AS]->(:Position {{name: $position}})
    MATCH (p)-[r:PLAYED_IN]->(f)
    WITH p, SUM(r.total_points) AS total_points
    {price_filter}
    RETURN p.player_name AS player, total_points
    ORDER BY total_points DESC
    LIMIT $limit
    {price_note}
    """
    params = {"season": season, "position": position, "limit": limit}
    if max_price is not None:
        params["max_price"] = max_price
    return query, params


def best_defense_teams(season: str, limit: int = 10) -> Tuple[str, Dict]:
    _require(locals(), ["season"])
    # Note: KG lacks direct Player->Team links, so we use clean sheets by defenders as a proxy
    # Returns teams with most clean sheets from their defensive players
    query = """
    MATCH (s:Season {season_name: $season})-[:HAS_GW]->(:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
    MATCH (p:Player)-[:PLAYS_AS]->(:Position {name: 'DEF'})
    MATCH (p)-[r:PLAYED_IN]->(f)
    WITH p, SUM(r.clean_sheets) AS clean_sheets, SUM(r.goals_conceded) AS goals_conceded
    RETURN p.player_name AS player, clean_sheets, goals_conceded
    ORDER BY clean_sheets DESC, goals_conceded ASC
    LIMIT $limit
    """
    return query, {"season": season, "limit": limit}


def best_attack_teams(season: str, limit: int = 10) -> Tuple[str, Dict]:
    _require(locals(), ["season"])
    # Note: KG lacks direct Player->Team links, so we return top attacking players
    # as a proxy for team attack strength
    query = """
    MATCH (s:Season {season_name: $season})-[:HAS_GW]->(:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
    MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
    WHERE pos.name IN ['FWD', 'MID']
    MATCH (p)-[r:PLAYED_IN]->(f)
    WITH p, pos.name AS position, SUM(r.goals_scored) AS goals, SUM(r.assists) AS assists,
         SUM(r.total_points) AS total_points
    RETURN p.player_name AS player, position, goals, assists, total_points
    ORDER BY goals DESC, assists DESC
    LIMIT $limit
    """
    return query, {"season": season, "limit": limit}


def player_form(player_name: str, last_n_gameweeks: int = 5) -> Tuple[str, Dict]:
    _require(locals(), ["player_name"])
    query = """
    MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f:Fixture)<-[:HAS_FIXTURE]-(gw:Gameweek)
    WITH p, gw, r
    ORDER BY gw.season DESC, gw.GW_number DESC
    LIMIT $last_n_gameweeks
    RETURN p.player_name AS player,
           collect(gw.GW_number) AS gameweeks,
           avg(r.total_points) AS avg_points,
           sum(r.total_points) AS total_points,
           avg(r.minutes) AS avg_minutes,
           avg(r.ict_index) AS avg_ict_index,
           avg(r.bps) AS avg_bps
    """
    return query, {"player_name": player_name, "last_n_gameweeks": last_n_gameweeks}


def fixtures_for_team(team_name: str, season: str, gameweek: int) -> Tuple[str, Dict]:
    _require(locals(), ["team_name", "season", "gameweek"])
    query = """
    MATCH (s:Season {season_name: $season})-[:HAS_GW]->(gw:Gameweek {GW_number: $gameweek})
    MATCH (gw)-[:HAS_FIXTURE]->(f:Fixture)
    WHERE (f)-[:HAS_HOME_TEAM]->(:Team {name: $team_name})
       OR (f)-[:HAS_AWAY_TEAM]->(:Team {name: $team_name})
    RETURN f.fixture_number AS fixture_number,
           f.kickoff_time AS kickoff_time,
           [(f)-[:HAS_HOME_TEAM]->(home) | home.name][0] AS home_team,
           [(f)-[:HAS_AWAY_TEAM]->(away) | away.name][0] AS away_team
    """
    return query, {"team_name": team_name, "season": season, "gameweek": gameweek}


def player_season_aggregate(player_name: str, season: str) -> Tuple[str, Dict]:
    _require(locals(), ["player_name", "season"])
    query = """
    MATCH (s:Season {season_name: $season})-[:HAS_GW]->(:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
    MATCH (p:Player {player_name: $player_name})-[r:PLAYED_IN]->(f)
    RETURN p.player_name AS player,
           SUM(r.goals_scored) AS goals,
           SUM(r.assists) AS assists,
           SUM(r.total_points) AS total_points,
           SUM(r.clean_sheets) AS clean_sheets,
           SUM(r.bonus) AS bonus,
           AVG(r.ict_index) AS avg_ict_index,
           AVG(r.minutes) AS avg_minutes
    """
    return query, {"player_name": player_name, "season": season}


def position_recommendations(position: str, min_points: int = 0, max_budget: float = 1000.0, limit: int = 10) -> Tuple[str, Dict]:
    _require(locals(), ["position"])
    # Note: max_budget filtering requires 'now_cost' property which may not exist
    query = """
    MATCH (p:Player)-[:PLAYS_AS]->(:Position {name: $position})
    MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
    WITH p, SUM(r.total_points) AS total_points, AVG(r.minutes) AS avg_minutes,
         AVG(r.ict_index) AS avg_ict, COUNT(r) AS games_played
    WHERE total_points >= $min_points
    RETURN p.player_name AS player,
           total_points,
           avg_minutes,
           avg_ict,
           games_played
    ORDER BY total_points DESC
    LIMIT $limit
    """
    return query, {
        "position": position,
        "min_points": min_points,
        "max_budget": max_budget,
        "limit": limit,
    }


def player_comparison(player1: str, player2: str, season: str) -> Tuple[str, Dict]:
    _require(locals(), ["player1", "player2", "season"])
    # Aggregate each player's stats separately, then combine
    query = """
    MATCH (s:Season {season_name: $season})-[:HAS_GW]->(:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
    MATCH (p1:Player {player_name: $player1})-[r1:PLAYED_IN]->(f)
    WITH $player2 AS player2_name, p1,
         SUM(r1.total_points) AS p1_points,
         SUM(r1.goals_scored) AS p1_goals,
         SUM(r1.assists) AS p1_assists,
         AVG(r1.ict_index) AS p1_ict,
         COUNT(r1) AS p1_games
    MATCH (s:Season {season_name: $season})-[:HAS_GW]->(:Gameweek)-[:HAS_FIXTURE]->(f2:Fixture)
    MATCH (p2:Player {player_name: player2_name})-[r2:PLAYED_IN]->(f2)
    RETURN p1.player_name AS player1,
           p1_points, p1_goals, p1_assists, p1_ict, p1_games,
           p2.player_name AS player2,
           SUM(r2.total_points) AS p2_points,
           SUM(r2.goals_scored) AS p2_goals,
           SUM(r2.assists) AS p2_assists,
           AVG(r2.ict_index) AS p2_ict,
           COUNT(r2) AS p2_games
    """
    return query, {"player1": player1, "player2": player2, "season": season}


def historical_best_xi(season: str, formation: str = "3-4-3") -> Tuple[str, Dict]:
    _require(locals(), ["season"])
    # Split formation string safely; expected format "3-4-3"
    parts = [int(p) for p in formation.split("-") if p.isdigit()]
    if len(parts) != 3:
        raise ValueError("Formation must look like '3-4-3'")
    defenders, midfielders, forwards = parts
    query = f"""
    MATCH (s:Season {{season_name: $season}})-[:HAS_GW]->(:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
    MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
    MATCH (p)-[r:PLAYED_IN]->(f)
    WITH p, pos.name AS position, SUM(r.total_points) AS total_points
    WITH position, p, total_points
    ORDER BY total_points DESC
    WITH position,
         CASE position WHEN 'DEF' THEN {defenders}
                       WHEN 'MID' THEN {midfielders}
                       WHEN 'FWD' THEN {forwards}
                       ELSE 1 END AS cap,
         p,
         total_points
    WITH position, cap, collect({{player: p.player_name, points: total_points}})[0..cap] AS picks
    RETURN position, picks
    """
    return query, {"season": season}


def differential_picks(max_ownership: float = 0.1, limit: int = 10) -> Tuple[str, Dict]:
    query = """
    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
    WITH p, SUM(r.total_points) AS total_points, AVG(r.form) AS avg_form, p.ownership_percent AS ownership
    WHERE ownership IS NOT NULL AND ownership <= $max_ownership
    RETURN p.player_name AS player, ownership, total_points, avg_form
    ORDER BY total_points DESC
    LIMIT $limit
    """
    return query, {"max_ownership": max_ownership, "limit": limit}


def consistent_performers(min_gameweeks: int = 5, limit: int = 10) -> Tuple[str, Dict]:
    query = """
    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)<-[:HAS_FIXTURE]-(gw:Gameweek)
    WITH p, gw.GW_number AS gw, r.total_points AS pts
    WITH p, collect(pts) AS pts_list
    WHERE size(pts_list) >= $min_gameweeks
    WITH p, stDev(pts_list) AS std_dev, avg(pts_list) AS avg_pts
    RETURN p.player_name AS player, avg_pts, std_dev
    ORDER BY std_dev ASC, avg_pts DESC
    LIMIT $limit
    """
    return query, {"min_gameweeks": min_gameweeks, "limit": limit}


def players_by_team_and_position(team_name: str, position: str, season: str) -> Tuple[str, Dict]:
    _require(locals(), ["team_name", "position", "season"])
    # Note: KG lacks Player->Team links, so this finds players who appeared in
    # fixtures involving the specified team. Results may include opposing players.
    # For more accurate results, the KG would need Player->Team relationships.
    query = """
    MATCH (s:Season {season_name: $season})-[:HAS_GW]->(:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
    WHERE (f)-[:HAS_HOME_TEAM]->(:Team {name: $team_name})
       OR (f)-[:HAS_AWAY_TEAM]->(:Team {name: $team_name})
    MATCH (p:Player)-[:PLAYS_AS]->(:Position {name: $position})
    MATCH (p)-[r:PLAYED_IN]->(f)
    WITH p, SUM(r.total_points) AS total_points, SUM(r.goals_scored) AS goals,
         SUM(r.assists) AS assists, COUNT(DISTINCT f) AS appearances
    WHERE appearances >= 5
    RETURN p.player_name AS player, total_points, goals, assists, appearances
    ORDER BY appearances DESC, total_points DESC
    LIMIT 20
    """
    return query, {"team_name": team_name, "position": position, "season": season}


def budget_team_builder(max_budget: float, season: str, formation: str = "3-4-3", limit: int = 15) -> Tuple[str, Dict]:
    _require(locals(), ["max_budget", "season"])
    query = """
    MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
    MATCH (p)-[r:PLAYED_IN]->(:Fixture)<-[:HAS_FIXTURE]-(:Gameweek)<-[:HAS_GW]-(s:Season {season_name: $season})
    WITH p, pos.name AS position, SUM(r.total_points) AS total_points, p.now_cost AS cost
    WHERE cost IS NOT NULL AND cost <= $max_budget
    RETURN position, p.player_name AS player, cost, total_points
    ORDER BY total_points DESC
    LIMIT $limit
    """
    return query, {"max_budget": max_budget, "season": season, "formation": formation, "limit": limit}


def top_players_by_stat(season: str, stat: str = "goals_scored", limit: int = 10) -> Tuple[str, Dict]:
    """Return top players by a specific statistic (goals, assists, points, etc.)."""
    _require(locals(), ["season"])
    # Map common stat names to relationship properties
    stat_map = {
        "goals": "goals_scored",
        "goals_scored": "goals_scored",
        "assists": "assists",
        "points": "total_points",
        "total_points": "total_points",
        "bonus": "bonus",
        "clean_sheets": "clean_sheets",
    }
    stat_field = stat_map.get(stat, "total_points")

    query = f"""
    MATCH (s:Season {{season_name: $season}})-[:HAS_GW]->(:Gameweek)-[:HAS_FIXTURE]->(f:Fixture)
    MATCH (p:Player)-[r:PLAYED_IN]->(f)
    OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
    WITH p, pos.name AS position, SUM(r.{stat_field}) AS stat_value, SUM(r.total_points) AS total_points
    RETURN p.player_name AS player, position, stat_value AS {stat_field}, total_points
    ORDER BY stat_value DESC
    LIMIT $limit
    """
    return query, {"season": season, "limit": limit}


TEMPLATE_REGISTRY: Dict[str, Callable] = {
    "player_performance_by_gameweek": player_performance_by_gameweek,
    "top_players_by_position": top_players_by_position,
    "top_players_by_stat": top_players_by_stat,
    "best_defense_teams": best_defense_teams,
    "best_attack_teams": best_attack_teams,
    "player_form": player_form,
    "fixtures_for_team": fixtures_for_team,
    "player_season_aggregate": player_season_aggregate,
    "position_recommendations": position_recommendations,
    "player_comparison": player_comparison,
    "historical_best_xi": historical_best_xi,
    "differential_picks": differential_picks,
    "consistent_performers": consistent_performers,
    "players_by_team_and_position": players_by_team_and_position,
    "budget_team_builder": budget_team_builder,
}


def list_templates() -> List[str]:
    """Return names of available templates."""
    return sorted(TEMPLATE_REGISTRY.keys())
