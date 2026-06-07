from pathlib import Path

import pandas as pd

from src.fusion_model import FusionInput, MultiModalFusionEngine
from src.knowledge_base import CampusKnowledgeBase
from src.text_pipeline import IntentClassifier
from src.utils import FAQ_CSV, bootstrap_project_assets, word_error_rate


def test_bootstrap_generates_required_data():
    bootstrap_project_assets()
    assert FAQ_CSV.exists()
    faq = pd.read_csv(FAQ_CSV)
    assert len(faq) >= 300
    assert {"text", "intent", "location"}.issubset(faq.columns)


def test_knowledge_base_has_20_locations():
    kb = CampusKnowledgeBase()
    locations = kb.all_locations()
    assert len(locations) == 20
    assert kb.get_by_name("Library")["category"] == "Learning"


def test_intent_classifier_predicts_opening_hours():
    classifier = IntentClassifier()
    result = classifier.predict("What time does the library open?")
    assert result["intent"] in {"opening_hours", "find_location"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_fusion_retrieves_top3():
    engine = MultiModalFusionEngine()
    result = engine.retrieve(FusionInput(text_query="Where can I find quiet study space?"), top_k=3)
    assert result["top_prediction"] is not None
    assert len(result["top_3"]) == 3


def test_word_error_rate_exact_match():
    assert word_error_rate("where is the library", "where is the library") == 0.0
