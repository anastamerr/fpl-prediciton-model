"""
Utility to generate and persist player node embeddings in Neo4j.

Supports two strategies:
1) Numeric feature vectors derived from PLAYED_IN aggregates
2) Textual feature descriptions embedded with QueryEmbedder
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
import numpy as np

from ..preprocessing.query_embedder import QueryEmbedder

logger = logging.getLogger(__name__)


@dataclass
class PlayerFeatures:
    player_name: str
    player_element: int
    positions: List[str]
    features: Dict[str, Any]


class NodeEmbeddingGenerator:
    def __init__(self, driver, model_alias: str = "bge-small"):
        self.driver = driver
        self.embedder = QueryEmbedder(model_alias=model_alias, normalize=True)

    def fetch_player_features(self) -> List[PlayerFeatures]:
        """
        Aggregate numeric stats per player from PLAYED_IN relationships.
        """
        cypher = """
        MATCH (p:Player)
        OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
        WITH p, collect(DISTINCT pos.name) AS positions
        MATCH (p)-[r:PLAYED_IN]->(:Fixture)
        RETURN p.player_name AS player_name,
               p.player_element AS player_element,
               positions AS positions,
               SUM(r.goals_scored) AS goals,
               SUM(r.assists) AS assists,
               SUM(r.total_points) AS total_points,
               AVG(r.minutes) AS avg_minutes,
               AVG(r.ict_index) AS avg_ict,
               AVG(r.influence) AS avg_influence,
               AVG(r.creativity) AS avg_creativity,
               AVG(r.threat) AS avg_threat,
               SUM(r.clean_sheets) AS clean_sheets
        """
        with self.driver.session() as session:
            rows = session.run(cypher).data()
        return [
            PlayerFeatures(
                player_name=row["player_name"],
                player_element=row["player_element"],
                positions=row.get("positions") or [],
                features={k: row[k] for k in row.keys() if k not in {"player_name", "player_element", "positions"}},
            )
            for row in rows
        ]

    def _compute_percentiles(self, all_features: List[PlayerFeatures]) -> Dict[str, Dict[str, float]]:
        """Compute percentile thresholds for each stat across all players."""
        stats = ["total_points", "goals", "assists", "avg_ict", "avg_influence", "avg_creativity", "avg_threat"]
        percentiles = {}

        for stat in stats:
            values = [pf.features.get(stat, 0) or 0 for pf in all_features]
            values = sorted(values)
            n = len(values)
            if n == 0:
                percentiles[stat] = {"p99": 0, "p95": 0, "p90": 0, "p75": 0, "p50": 0, "p25": 0}
            else:
                percentiles[stat] = {
                    "p99": values[int(n * 0.99)] if n > 0 else 0,
                    "p95": values[int(n * 0.95)] if n > 0 else 0,
                    "p90": values[int(n * 0.90)] if n > 0 else 0,
                    "p75": values[int(n * 0.75)] if n > 0 else 0,
                    "p50": values[int(n * 0.50)] if n > 0 else 0,
                    "p25": values[int(n * 0.25)] if n > 0 else 0,
                }
        return percentiles

    def _get_performance_tier(self, value: float, thresholds: Dict[str, float]) -> str:
        """Return a qualitative tier based on percentile thresholds."""
        if value >= thresholds["p99"]:
            return "world-class"
        elif value >= thresholds["p95"]:
            return "elite"
        elif value >= thresholds["p90"]:
            return "top"
        elif value >= thresholds["p75"]:
            return "good"
        elif value >= thresholds["p50"]:
            return "above average"
        elif value >= thresholds["p25"]:
            return "average"
        else:
            return "developing"

    def _to_text_feature(self, pf: PlayerFeatures, percentiles: Optional[Dict] = None) -> str:
        feats = pf.features
        positions = ", ".join(pf.positions) if pf.positions else "Unknown"

        # Get raw values
        total_points = feats.get('total_points', 0) or 0
        goals = feats.get('goals', 0) or 0
        assists = feats.get('assists', 0) or 0
        avg_ict = feats.get('avg_ict', 0) or 0
        avg_influence = feats.get('avg_influence', 0) or 0
        avg_creativity = feats.get('avg_creativity', 0) or 0
        avg_threat = feats.get('avg_threat', 0) or 0

        # Build qualitative description
        if percentiles:
            points_tier = self._get_performance_tier(total_points, percentiles["total_points"])
            goals_tier = self._get_performance_tier(goals, percentiles["goals"])
            ict_tier = self._get_performance_tier(avg_ict, percentiles["avg_ict"])

            # Position-specific descriptors
            pos_name = pf.positions[0] if pf.positions else "player"
            pos_full = {"FWD": "forward", "MID": "midfielder", "DEF": "defender", "GK": "goalkeeper"}.get(pos_name, "player")

            qualitative = []

            # Strongest descriptors for truly exceptional players
            if points_tier == "world-class":
                qualitative.extend([
                    f"the absolute best {pos_full}",
                    f"number one {pos_full} this season",
                    f"top {pos_full} in the league",
                    "must-have player",
                    "highest scoring",
                    "world-class performer"
                ])
            elif points_tier == "elite":
                qualitative.extend([
                    f"elite {pos_full}",
                    f"one of the best {pos_full}s this season",
                    f"top tier {pos_full}",
                    "premium pick",
                    "outstanding performer"
                ])
            elif points_tier == "top":
                qualitative.extend([
                    f"top {pos_full}",
                    f"among the best {pos_full}s",
                    "strong performer"
                ])
            elif points_tier == "good":
                qualitative.append(f"solid {pos_full}")

            # Goal scoring descriptors
            if goals_tier == "world-class" and pos_name in ["FWD", "MID"]:
                qualitative.extend(["league's top scorer", "prolific goal machine", "best goal scorer"])
            elif goals_tier == "elite" and pos_name in ["FWD", "MID"]:
                qualitative.extend(["elite goal scorer", "prolific striker"])
            elif goals_tier == "top" and pos_name in ["FWD", "MID"]:
                qualitative.append("top scorer")

            # ICT descriptors
            if ict_tier in ["world-class", "elite"]:
                qualitative.append("exceptional threat")

            qual_str = ", ".join(qualitative) if qualitative else "squad rotation player"
        else:
            qual_str = ""

        return (
            f"{pf.player_name} is a {positions} player. "
            f"Performance: {qual_str}. "
            f"Season stats: {int(total_points)} total points, {int(goals)} goals, {int(assists)} assists. "
            f"ICT index: {round(avg_ict, 1)}, influence: {round(avg_influence, 1)}, "
            f"creativity: {round(avg_creativity, 1)}, threat: {round(avg_threat, 1)}."
        )

    def generate_embeddings(
        self,
        use_text: bool = True,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return list of {player_element, embedding, player_name} ready to be persisted.
        """
        all_features = self.fetch_player_features()

        # Compute percentiles from ALL players for proper ranking
        percentiles = self._compute_percentiles(all_features) if use_text else None

        features = all_features[:limit] if limit else all_features

        payload = []
        for pf in features:
            if use_text:
                text = self._to_text_feature(pf, percentiles)
                vector = self.embedder.embed(text)
            else:
                numeric = np.array(list(pf.features.values()))
                vector = self.embedder.embed(" ".join(map(str, numeric.tolist())))
            payload.append(
                {
                    "player_element": pf.player_element,
                    "player_name": pf.player_name,
                    "embedding": vector.tolist(),
                }
            )
        return payload

    def persist_embeddings(
        self,
        index_name: str = "player_embeddings",
        property_key: str = "embedding",
        use_text: bool = True,
        limit: Optional[int] = None,
    ) -> int:
        """
        Writes embeddings to Neo4j and ensures vector index exists.
        """
        payload = self.generate_embeddings(use_text=use_text, limit=limit)
        if not payload:
            return 0

        dim = self.embedder.embedding_dimension()
        index_cypher = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (p:Player) ON p.{property_key}
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dim},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """

        # Use f-string to embed property key directly since Neo4j doesn't support
        # parameterized property names in SET clauses reliably
        update_cypher = f"""
        UNWIND $rows AS row
        MATCH (p:Player {{player_element: row.player_element, player_name: row.player_name}})
        SET p.{property_key} = row.embedding
        """

        with self.driver.session() as session:
            session.run(index_cypher)
            session.run(update_cypher, rows=payload)
        return len(payload)
