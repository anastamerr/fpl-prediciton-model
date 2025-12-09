import pytest

from src.preprocessing.intent_classifier import FPLIntentClassifier, INTENT_EXAMPLES


@pytest.fixture(scope="module")
def classifier():
    return FPLIntentClassifier()


@pytest.mark.parametrize("intent_label", list(INTENT_EXAMPLES.keys()))
def test_examples_cover_expected_intents(classifier, intent_label):
    # Ensure every seed example maps to its intended label.
    for query in INTENT_EXAMPLES[intent_label]:
        result = classifier.classify(query)
        assert result.intent == intent_label
        assert result.confidence >= 0.2


def test_empty_query_returns_general():
    result = FPLIntentClassifier().classify("")
    assert result.intent == "general_question"
    assert result.confidence == 0.0


def test_multi_signal_prefers_comparison():
    query = "Compare Haaland vs Kane for GW3 best forward"
    result = FPLIntentClassifier().classify(query)
    assert result.intent == "comparison_query"
    assert "vs" in result.matched_keywords
