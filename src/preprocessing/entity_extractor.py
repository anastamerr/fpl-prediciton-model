"""
Rule-based entity extraction for FPL queries.

Extracts players, teams, positions, seasons, gameweeks, statistics, and common
numeric constraints (budget, points) using regex and lightweight fuzzy matching.
"""

from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Sequence
import re


SEASON_PATTERN = re.compile(r"\b(20\d{2})[-/](\d{2})\b")
GAMEWEEK_PATTERN = re.compile(r"\b(?:gw|gameweek)\s*(\d{1,2})\b", flags=re.IGNORECASE)

POSITION_MAP = {
    "goalkeeper": "GK",
    "keeper": "GK",
    "gk": "GK",
    "defender": "DEF",
    "def": "DEF",
    "centre back": "DEF",
    "center back": "DEF",
    "full back": "DEF",
    "midfielder": "MID",
    "mid": "MID",
    "midfield": "MID",
    "winger": "MID",
    "forward": "FWD",
    "striker": "FWD",
    "fwd": "FWD",
}

STAT_KEYWORDS = {
    "goal": "goals_scored",
    "goals": "goals_scored",
    "assist": "assists",
    "assists": "assists",
    "points": "total_points",
    "bonus": "bonus",
    "bps": "bps",
    "ict": "ict_index",
    "threat": "threat",
    "creativity": "creativity",
    "influence": "influence",
    "clean sheet": "clean_sheets",
    "clean sheets": "clean_sheets",
    "xg": "expected_goals",
    "xa": "expected_assists",
}


@dataclass
class ExtractedEntities:
    players: List[str]
    teams: List[str]
    positions: List[str]
    seasons: List[str]
    gameweeks: List[int]
    statistics: List[str]
    numerical_values: Dict[str, Any]


class FPLEntityExtractor:
    """
    Extract FPL-specific entities with regex patterns and fuzzy matching.

    Player/team matching prefers substring detection and falls back to fuzzy
    n-gram matching using difflib when an index is provided.
    """

    def __init__(
        self,
        player_index: Optional[Sequence[str]] = None,
        team_index: Optional[Sequence[str]] = None,
        fuzzy_cutoff: float = 0.5,
    ) -> None:
        self.player_lookup = {p.lower(): p for p in player_index or []}
        self.team_lookup = {t.lower(): t for t in team_index or []}
        self.fuzzy_cutoff = fuzzy_cutoff

    def extract(self, query: Optional[str]) -> ExtractedEntities:
        if not query:
            return self._empty()

        normalized = query.lower()
        return ExtractedEntities(
            players=self._match_catalog(normalized, self.player_lookup),
            teams=self._match_catalog(normalized, self.team_lookup),
            positions=self._extract_positions(normalized),
            seasons=self._extract_seasons(normalized),
            gameweeks=self._extract_gameweeks(normalized),
            statistics=self._extract_statistics(normalized),
            numerical_values=self._extract_numbers(normalized),
        )

    def _empty(self) -> ExtractedEntities:
        return ExtractedEntities([], [], [], [], [], [], {})

    def _extract_seasons(self, text: str) -> List[str]:
        return [match.group(0) for match in SEASON_PATTERN.finditer(text)]

    def _extract_gameweeks(self, text: str) -> List[int]:
        gws = []
        for match in GAMEWEEK_PATTERN.finditer(text):
            try:
                gws.append(int(match.group(1)))
            except ValueError:
                continue
        return gws

    def _extract_positions(self, text: str) -> List[str]:
        positions = []
        for key, code in POSITION_MAP.items():
            if key in text:
                positions.append(code)
        return sorted(set(positions))

    def _extract_statistics(self, text: str) -> List[str]:
        stats = []
        for key, canonical in STAT_KEYWORDS.items():
            if key in text:
                stats.append(canonical)
        return sorted(set(stats))

    def _match_catalog(self, text: str, catalog: Dict[str, str]) -> List[str]:
        if not catalog:
            return []

        hits = []
        text_tokens = set(text.lower().split())

        # Strategy 1: Exact full name match (e.g., "erling haaland" in text)
        for key, label in catalog.items():
            if key in text:
                hits.append(label)

        # Strategy 2: Last name matching for players (more common in queries)
        # Match if any significant token from the catalog key appears as a whole word in text
        for key, label in catalog.items():
            key_parts = key.split()
            # For multi-word names, check if last name (or any part >= 4 chars) matches
            for part in key_parts:
                if len(part) >= 4 and part in text_tokens:
                    hits.append(label)
                    break

        if hits:
            return sorted(set(hits))

        # Strategy 3: Fuzzy fallback with STRICT cutoff (only if no exact matches)
        # Only match longer tokens to avoid false positives
        tokens = [t for t in re.findall(r"[a-zA-Z]+", text) if len(t) >= 5]

        for token in tokens:
            # Use high cutoff for strict matching
            matches = get_close_matches(token, catalog.keys(), n=1, cutoff=0.8)
            if matches:
                hits.append(catalog[matches[0]])

        return sorted(set(hits))

    def _extract_numbers(self, text: str) -> Dict[str, Any]:
        numbers: Dict[str, Any] = {}

        budget_match = re.search(r"(?:budget|cost|price)\s*(?:of\s*)?(\d+(?:\.\d+)?)", text)
        if budget_match:
            numbers["budget"] = float(budget_match.group(1))

        min_points = re.search(r"(?:at least|min|over)\s*(\d+)\s*(?:pts|points)?", text)
        if min_points:
            numbers["min_points"] = int(min_points.group(1))

        top_n = re.search(r"\btop\s*(\d{1,2})\b", text)
        if top_n:
            numbers["limit"] = int(top_n.group(1))

        max_price = re.search(r"\bunder\s*(\d+(?:\.\d+)?)(?:m)?\b", text)
        if max_price and "budget" not in numbers:
            numbers["max_price"] = float(max_price.group(1))

        return numbers
