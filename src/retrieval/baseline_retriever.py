"""
Baseline retriever that maps intents to Cypher templates and executes them.

Designed to pair with the Milestone 2 Neo4j KG and the intent/entity outputs from
preprocessing. Query execution uses the Neo4j Python driver; callers can provide
their own driver/session implementation for testing.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import logging

from . import cypher_templates as templates

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    intent: str
    template_name: str
    query: str
    params: Dict[str, Any]
    records: List[Dict[str, Any]]
    error: Optional[str] = None


class BaselineRetriever:
    """
    Intent-driven Cypher retriever.

    - Uses template registry from `cypher_templates.py`
    - Maps FPL intents to default templates; `template_hint` overrides mapping
    - Handles missing entities gracefully by returning an error in RetrievalResult
    """

    def __init__(self, driver, intent_map: Optional[Dict[str, str]] = None):
        self.driver = driver
        self._default_season: Optional[str] = None
        self.intent_map = intent_map or {
            "player_performance": "player_performance_by_gameweek",
            "team_analysis": "best_defense_teams",
            "team_recommendation": "budget_team_builder",
            "position_search": "top_players_by_position",
            "fixture_analysis": "fixtures_for_team",
            "statistics_query": "top_players_by_stat",
            "comparison_query": "player_comparison",
            "form_analysis": "form_leaders",
            "historical_query": "historical_best_xi",
            "general_question": "top_players_by_position",
        }

    def retrieve(
        self,
        intent: str,
        entities: Dict[str, Any],
        template_hint: Optional[str] = None,
    ) -> RetrievalResult:
        template_name = template_hint or self.intent_map.get(intent)
        if not template_name or template_name not in templates.TEMPLATE_REGISTRY:
            msg = f"No template available for intent '{intent}'."
            logger.warning(msg)
            return RetrievalResult(intent, template_name or "none", "", {}, [], error=msg)

        try:
            params_builder = self._params_builder(template_name)
            params = params_builder(entities)
            query, bound_params = self._build_query(template_name, params)
            records = self._run_query(query, bound_params)
            return RetrievalResult(intent, template_name, query, bound_params, records)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Retrieval failed: %s", exc)
            return RetrievalResult(intent, template_name, "", {}, [], error=str(exc))

    def _build_query(self, template_name: str, params: Dict[str, Any]) -> (str, Dict[str, Any]):
        builder: Callable = templates.TEMPLATE_REGISTRY[template_name]
        query, bound = builder(**params)
        return query, bound

    def _run_query(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.driver:
            raise RuntimeError("Neo4j driver is not configured.")
        with self.driver.session() as session:
            result = session.run(query, params)
            return result.data()

    def _params_builder(self, template_name: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        builders: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "player_performance_by_gameweek": self._build_player_performance_params,
            "top_players_by_position": self._build_top_players_params,
            "top_players_by_stat": self._build_top_by_stat_params,
            "best_defense_teams": self._build_season_params,
            "best_attack_teams": self._build_season_params,
            "player_form": self._build_player_form_params,
            "fixtures_for_team": self._build_fixture_params,
            "player_season_aggregate": self._build_player_season_params,
            "position_recommendations": self._build_position_reco_params,
            "player_comparison": self._build_player_comparison_params,
            "historical_best_xi": self._build_season_formation_params,
            "differential_picks": self._build_differential_params,
            "consistent_performers": self._build_consistency_params,
            "players_by_team_and_position": self._build_team_position_params,
            "budget_team_builder": self._build_budget_params,
            "form_leaders": self._build_form_leaders_params,
        }
        if template_name not in builders:
            raise KeyError(f"No params builder for template {template_name}")
        return builders[template_name]

    @staticmethod
    def _first_value(seq: Optional[List[Any]]) -> Optional[Any]:
        return seq[0] if seq else None

    def _season_or_default(self, seasons: Optional[List[Any]]) -> Optional[Any]:
        val = self._first_value(seasons)
        if val:
            return val
        return self._get_default_season()

    def _get_default_season(self) -> Optional[str]:
        if self._default_season or not self.driver:
            return self._default_season
        try:
            with self.driver.session() as session:
                row = session.run(
                    "MATCH (s:Season) RETURN s.season_name AS season ORDER BY season DESC LIMIT 1"
                ).single()
                if row:
                    self._default_season = row["season"]
        except Exception:
            self._default_season = None
        return self._default_season

    def _build_player_performance_params(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "player_name": self._first_value(entities.get("players")),
            "season": self._season_or_default(entities.get("seasons")),
            "gameweek": self._first_value(entities.get("gameweeks")),
        }

    def _build_top_players_params(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        num = entities.get("numerical_values", {})
        params = {
            "position": self._first_value(entities.get("positions")) or "FWD",
            "season": self._season_or_default(entities.get("seasons")),
            "limit": num.get("limit", 10),
        }
        return params

    def _build_top_by_stat_params(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        # Map extracted statistics to the stat parameter
        stats = entities.get("statistics", [])
        stat = "total_points"  # default
        if stats:
            stat_map = {
                "goals_scored": "goals_scored",
                "assists": "assists",
                "total_points": "total_points",
                "bonus": "bonus",
                "clean_sheets": "clean_sheets",
            }
            stat = stat_map.get(stats[0], "total_points")
        return {
            "season": self._season_or_default(entities.get("seasons")),
            "stat": stat,
            "limit": entities.get("numerical_values", {}).get("limit", 10),
        }

    def _build_season_params(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "season": self._season_or_default(entities.get("seasons")),
            "limit": entities.get("numerical_values", {}).get("limit", 10),
        }

    def _build_player_form_params(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "player_name": self._first_value(entities.get("players")),
            "last_n_gameweeks": entities.get("numerical_values", {}).get("last_n_gameweeks", 5),
        }

    def _build_form_leaders_params(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        num = entities.get("numerical_values", {})
        return {
            "last_n_gameweeks": num.get("last_n_gameweeks", 5),
            "season": self._season_or_default(entities.get("seasons")),
            "limit": num.get("limit", 10),
        }

    def _build_fixture_params(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "team_name": self._first_value(entities.get("teams")),
            "season": self._season_or_default(entities.get("seasons")),
            "gameweek": self._first_value(entities.get("gameweeks")),
        }

    def _build_player_season_params(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "player_name": self._first_value(entities.get("players")),
            "season": self._season_or_default(entities.get("seasons")),
        }

    def _build_position_reco_params(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        num = entities.get("numerical_values", {})
        return {
            "position": self._first_value(entities.get("positions")) or "FWD",
            "min_points": num.get("min_points", 0),
            "max_budget": num.get("max_price", num.get("budget", 1000.0)),
            "limit": num.get("limit", 10),
        }

    def _build_player_comparison_params(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        players = entities.get("players") or []
        player1 = players[0] if len(players) > 0 else None
        player2 = players[1] if len(players) > 1 else None
        return {
            "player1": player1,
            "player2": player2,
            "season": self._season_or_default(entities.get("seasons")),
        }

    def _build_season_formation_params(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "season": self._season_or_default(entities.get("seasons")),
            "formation": entities.get("numerical_values", {}).get("formation", "3-4-3"),
        }

    def _build_differential_params(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        num = entities.get("numerical_values", {})
        return {
            "limit": num.get("limit", 10),
            "max_ownership": num.get("max_ownership", num.get("ownership", 0.1)),
        }

    def _build_consistency_params(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        num = entities.get("numerical_values", {})
        return {
            "min_gameweeks": num.get("min_gameweeks", 5),
            "limit": num.get("limit", 10),
        }

    def _build_team_position_params(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "team_name": self._first_value(entities.get("teams")),
            "position": self._first_value(entities.get("positions")),
            "season": self._season_or_default(entities.get("seasons")),
        }

    def _build_budget_params(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        num = entities.get("numerical_values", {})
        return {
            "max_budget": num.get("budget", 100.0),
            "season": self._season_or_default(entities.get("seasons")),
            "formation": num.get("formation", "3-4-3"),
            "limit": num.get("limit", 15),
        }
