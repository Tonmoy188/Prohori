"""FastAPI backend. Run: python -m uvicorn main:app --reload"""
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
import weather
from chat import ChatConfigurationError, ChatRateLimited, ChatService
from service import Service

app = FastAPI(title="Dengue live early-warning API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
svc = Service()
bot = ChatService(svc)


def guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown ward/city: {exc}")
    except weather.WeatherUnavailable as exc:
        raise HTTPException(503, f"Live weather unavailable: {exc}")


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=300)
    ward_id: Optional[str] = None
    horizon_weeks: Optional[int] = Field(None, ge=2, le=4)
    session_id: str = Field("default", pattern=r"^[A-Za-z0-9_\-]{1,64}$")


@app.get("/")
def root():
    return {"service": app.title, "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {
        "status": "ok",  # prediction works without the optional chatbot key
        "models": {str(h): {"trained_through": svc.models[h].get("trained_through"),
                            "training_rows": svc.models[h].get("training_rows")} for h in config.HORIZONS},
        "wards": len(svc.wards),
        "weather_provider": "Open-Meteo Forecast + Archive API (real data; no key required)",
        "chatbot": "Gemini API configured" if bot.client else "disabled: add GEMINI_API_KEY to .env",
    }


@app.get("/wards")
def wards():
    return {"count": len(svc.wards), "wards": svc.wards[["ward_id", "city"]].to_dict("records")}


@app.get("/weather/forecast")
def weather_forecast(city: Optional[str] = None):
    payload, stale = guard(weather.get_payload)
    if city and city not in weather.CITIES:
        raise HTTPException(404, f"Unknown city: {city}")
    return {"source": payload.get("source"), "fetched_at_utc": __import__("datetime").datetime.fromtimestamp(
        payload["fetched_at"], tz=__import__("datetime").timezone.utc).isoformat(),
        "stale_cache": stale, "provider_forecast_days": payload.get("forecast_days_available", 16),
        "daily_forecast": weather.forecast_daily(payload, city)}


@app.get("/weather")
def live_weather(city: Optional[str] = None):
    frames, meta = guard(svc.frames)
    data = frames[frames.dengue_cases.isna()].copy()
    if city:
        data = data[data.city == city]
        if data.empty:
            raise HTTPException(404, "Unknown city")
    columns = ["city", "ward_id", "date", "rainfall_mm", "avg_temp_c", "max_temp_c", "min_temp_c", "relative_humidity_pct"]
    # Return the latest eight live weeks per ward. Older observations remain in the
    # model's internal lag history, but are not mislabeled as the current forecast.
    recent = (data.sort_values("date").groupby("ward_id", group_keys=False).tail(8))
    latest = data.sort_values("date").groupby("ward_id", group_keys=False).tail(1)
    forecast_context = {}
    for name in sorted(data.city.unique()):
        payload, stale = guard(weather.get_payload)
        forecast_context[name] = {
            "provider": payload.get("source", "Open-Meteo"),
            "fetched_at_utc": __import__("datetime").datetime.fromtimestamp(
                payload["fetched_at"], tz=__import__("datetime").timezone.utc).isoformat(),
            "stale_cache": stale,
            "daily_forecast": weather.forecast_daily(payload, name),
        }
    return {
        "weather": meta,
        "current_week_by_ward": latest[columns].assign(
            date=lambda x: x.date.dt.strftime("%Y-%m-%d")).round(2).to_dict("records"),
        "recent_live_weekly_weather": recent[columns].assign(
            date=lambda x: x.date.dt.strftime("%Y-%m-%d")).round(2).to_dict("records"),
        # Backward-compatible key for existing dashboards; now limited to the latest eight weeks.
        "live_weekly_weather": recent[columns].assign(
            date=lambda x: x.date.dt.strftime("%Y-%m-%d")).round(2).to_dict("records"),
        "forecast_by_city": forecast_context,
    }


@app.get("/predict")
def predict(ward_id: str, horizon_weeks: int = Query(3, ge=2, le=4)):
    return guard(svc.predict_ward, ward_id, horizon_weeks)


@app.get("/outlook")
def outlook(ward_id: str):
    return {"ward_id": ward_id, "horizons": [guard(svc.predict_ward, ward_id, h) for h in config.HORIZONS]}


@app.get("/predict/all")
def predict_all(horizon_weeks: int = Query(3, ge=2, le=4), city: Optional[str] = None):
    return guard(svc.predict_all, horizon_weeks, city)


@app.post("/chat")
def chat(body: ChatIn):
    try:
        return bot.ask(body.message, body.session_id, body.ward_id, body.horizon_weeks)
    except ChatConfigurationError as exc:
        raise HTTPException(503, str(exc))
    except ChatRateLimited as exc:
        raise HTTPException(429, str(exc))
    except weather.WeatherUnavailable as exc:
        raise HTTPException(503, f"Live weather unavailable: {exc}")
    except KeyError as exc:
        raise HTTPException(404, f"Unknown ward/city: {exc}")


@app.get("/metrics")
def metrics():
    """Chronological (2024-2025 holdout) validation metrics written by scripts/train_models.py."""
    path = config.ROOT / "training_report.json"
    if not path.is_file():
        raise HTTPException(404, "Run python scripts/train_models.py first")
    return __import__("json").loads(path.read_text(encoding="utf-8"))


@app.get("/stats")
def stats():
    return {"chatbot": "Gemini API only", "configured": bool(bot.client)}
