from src.preprocessing.entity_extractor import FPLEntityExtractor


def test_extract_positions_and_gameweek():
    extractor = FPLEntityExtractor()
    result = extractor.extract("Top forwards for GW5 under 10m")
    assert "FWD" in result.positions
    assert 5 in result.gameweeks
    assert result.numerical_values.get("max_price") == 10.0


def test_extract_season_and_stats():
    extractor = FPLEntityExtractor()
    result = extractor.extract("Most assists and goals in 2022-23 season")
    assert "2022-23" in result.seasons
    assert "assists" in result.statistics
    assert "goals_scored" in result.statistics


def test_fuzzy_player_and_team_match():
    extractor = FPLEntityExtractor(player_index=["Mohamed Salah"], team_index=["Liverpool"])
    result = extractor.extract("How did Mo Salah do for liverpool in gw3")
    assert "Mohamed Salah" in result.players
    assert "Liverpool" in result.teams
