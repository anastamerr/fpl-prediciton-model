from src.retrieval.baseline_retriever import BaselineRetriever


class DummySession:
    def run(self, query, params=None):
        # Return an object with data() that yields empty list
        class R:
            def data(self_inner):
                return []

        return R()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class DummyDriver:
    def session(self):
        return DummySession()


def test_baseline_param_building_with_sparse_entities():
    retriever = BaselineRetriever(driver=DummyDriver())
    entities = {
        "players": ["Erling Haaland"],
        "seasons": ["2022-23"],
        "gameweeks": [1],
        "positions": ["FWD"],
        "teams": ["Man City"],
        "numerical_values": {"limit": 5, "budget": 100.0},
    }
    result = retriever.retrieve(intent="player_performance", entities=entities)
    assert result.error is None
    assert result.template_name == "player_performance_by_gameweek"
    assert result.params["player_name"] == "Erling Haaland"
    assert result.params["season"] == "2022-23"
    assert result.params["gameweek"] == 1
