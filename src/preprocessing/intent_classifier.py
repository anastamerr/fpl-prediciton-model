"""
Rule-based intent classifier for FPL queries.

The classifier covers 10 intents defined in AGENTS.md and returns a label plus
simple confidence and keyword matches. It is intentionally lightweight so it
can be swapped for an LLM-backed classifier later without changing callers.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re


INTENT_LABELS = [
    "player_performance",
    "team_analysis",
    "team_recommendation",
    "position_search",
    "fixture_analysis",
    "statistics_query",
    "comparison_query",
    "form_analysis",
    "historical_query",
    "general_question",
]


@dataclass
class IntentResult:
    intent: str
    confidence: float
    matched_keywords: List[str]
    secondary_matches: List[str]


class FPLIntentClassifier:
    """
    Keyword-driven classifier for FPL intents.

    The heuristic uses keyword tallies plus regex patterns. Confidence is a
    coarse 0-1 score based on the relative strength of matches.
    """

    def __init__(self) -> None:
        self.intent_keywords: Dict[str, List[str]] = {
            "team_recommendation": [
                "recommend",
                "suggest",
                "build",
                "draft",
                "pick",
                "select",
                "squad",
                "team",
                "lineup",
                "formation",
                "captain",
                "captaincy",
            ],
            "comparison_query": [
                "compare",
                "versus",
                "vs",
                "head to head",
                "better than",
            ],
            "fixture_analysis": [
                "fixture",
                "fixtures",
                "schedule",
                "gw",
                "gameweek",
                "next match",
            ],
            "position_search": [
                "forward",
                "striker",
                "fwd",
                "midfielder",
                "midfield",
                "mid",
                "defender",
                "def",
                "gk",
                "goalkeeper",
                "keeper",
            ],
            "statistics_query": [
                "most",
                "highest",
                "assist",
                "goals",
                "assists",
                "points",
                "bps",
                "ict",
                "xg",
                "xa",
                "stats",
            ],
            "player_performance": [
                "how did",
                "performance",
                "played in",
                "returned",
            ],
            "team_analysis": [
                "team",
                "defense",
                "defence",
                "attack",
                "goals conceded",
                "clean sheet",
            ],
            "form_analysis": [
                "form",
                "recent",
                "streak",
                "hot",
                "in form",
            ],
            "historical_query": [
                "season",
                "historical",
                "previous",
                "last year",
                "all time",
                "since",
            ],
        }
        self.intent_priority: List[str] = [
            "comparison_query",
            "team_recommendation",
            "player_performance",
            "fixture_analysis",
            "form_analysis",
            "statistics_query",
            "team_analysis",
            "position_search",
            "historical_query",
            "general_question",
        ]
        self.vs_pattern = re.compile(r"\b(?:vs|versus|v)\b", flags=re.IGNORECASE)

    def classify(self, query: Optional[str]) -> IntentResult:
        """
        Return the best-fit intent for a query with a simple confidence score.
        """
        if not query or not query.strip():
            return IntentResult("general_question", 0.0, [], [])

        normalized = query.lower()
        scores: Dict[str, float] = {}
        matched_keywords: Dict[str, List[str]] = {}

        for intent, keywords in self.intent_keywords.items():
            matches = [kw for kw in keywords if self._contains(normalized, kw)]
            if matches:
                matched_keywords[intent] = matches
                scores[intent] = float(len(matches))

        # Regex boost for comparison intent
        if self.vs_pattern.search(normalized):
            scores["comparison_query"] = scores.get("comparison_query", 0.0) + 1.5
            matched_keywords.setdefault("comparison_query", []).append("vs")

        # Gameweek mentions often imply fixture/form intent
        gameweek_mentioned = bool(re.search(r"\b(?:gw|gameweek)s?\s*\d{1,2}\b", normalized) or re.search(r"\b\d{1,2}\s*gameweeks?\b", normalized))
        if gameweek_mentioned:
            scores["fixture_analysis"] = scores.get("fixture_analysis", 0.0) + 1.0
            matched_keywords.setdefault("fixture_analysis", []).append("gameweek")
            # Performance questions with a gameweek should lean to player_performance
            scores["player_performance"] = scores.get("player_performance", 0.0) + 0.5
            # Form queries that cite gameweeks should lean to form_analysis
            if "form" in normalized or "recent" in normalized:
                scores["form_analysis"] = scores.get("form_analysis", 0.0) + 1.5

        # Formation hint pushes to team recommendation
        if re.search(r"\b\d-\d-\d\b", normalized):
            scores["team_recommendation"] = scores.get("team_recommendation", 0.0) + 1.0
            matched_keywords.setdefault("team_recommendation", []).append("formation")

        # Extra boost when explicit performance wording appears.
        if "performance" in normalized or "how did" in normalized:
            scores["player_performance"] = scores.get("player_performance", 0.0) + 1.0
        if re.search(r"score\s+in", normalized):
            scores["player_performance"] = scores.get("player_performance", 0.0) + 1.0

        # Fixture phrasing without "fixture" keyword (e.g., "play next")
        if re.search(r"play next|next match|next game", normalized):
            scores["fixture_analysis"] = scores.get("fixture_analysis", 0.0) + 1.0
            matched_keywords.setdefault("fixture_analysis", []).append("next")

        # If a position is mentioned alongside ranking language, boost position_search.
        if matched_keywords.get("position_search") and ("top" in normalized or "best" in normalized):
            scores["position_search"] = scores.get("position_search", 0.0) + 1.0

        # Historical queries mentioning seasons should dominate.
        if re.search(r"\bseason\b|\bhistorical\b|\bprevious\b", normalized):
            scores["historical_query"] = scores.get("historical_query", 0.0) + 0.25
        # Team attack/defense questions with seasonal context should favor team_analysis.
        if ("attack" in normalized or "defense" in normalized or "defence" in normalized) and "season" in normalized:
            scores["team_analysis"] = scores.get("team_analysis", 0.0) + 0.75

        if not scores:
            return IntentResult("general_question", 0.2, [], [])

        best_intent = self._pick_intent(scores)
        confidence = self._confidence(best_intent, scores)
        primary_matches = matched_keywords.get(best_intent, [])
        secondary = [
            intent for intent in scores.keys() if intent != best_intent and scores[intent] > 0
        ]

        return IntentResult(best_intent, confidence, primary_matches, secondary)

    def _pick_intent(self, scores: Dict[str, float]) -> str:
        # Highest score wins; ties resolved by priority list.
        max_score = max(scores.values())
        candidates = [intent for intent, score in scores.items() if score == max_score]
        for intent in self.intent_priority:
            if intent in candidates:
                return intent
        return candidates[0]

    def _confidence(self, winner: str, scores: Dict[str, float]) -> float:
        total = sum(scores.values())
        if total <= 0:
            return 0.0
        return round(min(1.0, scores[winner] / total), 2)

    @staticmethod
    def _contains(text: str, keyword: str) -> bool:
        """
        Safer keyword detection with word boundaries for short tokens.
        Avoids accidental matches like 'form' in 'performance'.
        """
        if keyword == "form":
            return re.search(r"\bform\b", text) is not None
        if len(keyword) <= 3 or keyword in {"gw", "vs"}:
            return re.search(rf"\b{re.escape(keyword)}\b", text) is not None
        return keyword in text


# Lightweight corpus to seed unit tests and quick manual checks.
INTENT_EXAMPLES: Dict[str, List[str]] = {
    "player_performance": [
        "How did Salah perform in gameweek 5?",
        "Show Foden's score in GW2.",
        "What was Haaland's performance last weekend?",
    ],
    "team_analysis": [
        "Which teams had the best defense in 2022-23?",
        "Top attacks last season?",
        "Strongest defenses this year?",
    ],
    "team_recommendation": [
        "Recommend a squad with budget constraints.",
        "Build me a 3-4-3 team under 100m.",
        "Suggest a team for gameweek 10.",
    ],
    "position_search": [
        "Who are the top forwards?",
        "Best defenders right now.",
        "Top GK options.",
    ],
    "fixture_analysis": [
        "Show me fixtures for gameweek 10.",
        "What's Liverpool's fixture in GW5?",
        "Who do Arsenal play next weekend?",
    ],
    "statistics_query": [
        "Players with most goals in 2021-22.",
        "Highest assist tally this season.",
        "Top points scorers overall.",
    ],
    "comparison_query": [
        "Compare Haaland vs Kane.",
        "Is Saka better than Martinelli?",
        "Haaland versus Watkins in GW3.",
    ],
    "form_analysis": [
        "Show players in best form.",
        "Who is on a hot streak lately?",
        "Best recent performers last 5 gameweeks.",
    ],
    "historical_query": [
        "Best players from season 2021-22.",
        "Historical top scorers.",
        "Who dominated two seasons ago?",
    ],
    "general_question": [
        "How does FPL scoring work?",
        "Explain chips strategy.",
        "What is bonus point system?",
    ],
}
