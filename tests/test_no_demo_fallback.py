"""Small offline checks for the no-demo behavior and model bundle metadata."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from chat import ChatConfigurationError, ChatService
from service import Service


def test_models_are_trained_from_supplied_history():
    svc = Service()
    assert set(svc.models) == {2, 3, 4}
    for horizon, bundle in svc.models.items():
        assert bundle["horizon_weeks"] == horizon
        assert bundle["training_dataset"] == "data/dengue_history.csv"
        assert bundle["feature_version"] == "causal_weekly_weather_v3_fwd1"


def test_chatbot_does_not_return_template_when_key_is_missing(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    bot = ChatService(Service())
    try:
        bot.ask("hello")
    except ChatConfigurationError:
        return
    raise AssertionError("Chatbot must fail closed without a real API key")


def test_trained_models_score_historical_origin_rows():
    import engine
    svc = Service()
    history = svc.history.copy()
    for horizon, bundle in svc.models.items():
        scored = engine.score_rows(history, bundle)
        latest = scored.sort_values("date").groupby("ward_id", as_index=False).tail(1)
        assert len(latest) == 60
        assert latest.expected_cases.ge(0).all()
        probability_sum = latest[[f"p_{tier}" for tier in engine.TIERS]].sum(axis=1)
        assert ((probability_sum - 1).abs() < 1e-6).all()


def test_forecast_features_are_used_by_models():
    import engine
    svc = Service()
    for bundle in svc.models.values():
        assert set(engine.FWD_COLS) <= set(bundle["features"])
