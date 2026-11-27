"""FastAPI backend. Run: python -m uvicorn main:app --reload"""
import datetime as dt
import json
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
import dispatch
import weather
from chat import ChatRateLimited, ChatService
from ops import OpsEngine
from service import Service

app = FastAPI(title="PROHORI DengueWatch operations API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
svc = Service()
ops = OpsEngine()
bot = ChatService(svc, ops)


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
        "chatbot": "Gemini API configured" if bot.client else "offline template replies (no GEMINI_API_KEY)",
        "ops_models": ops.manifest.get("classification_scores"),
        "ops_data_through": ops.origins()["latest"],
    }


@app.get("/wards")
def wards():
    return {"count": len(svc.wards), "wards": svc.wards[["ward_id", "city"]].to_dict("records")}


@app.get("/weather/forecast")
def weather_forecast(city: Optional[str] = None):
    payload, stale = guard(weather.get_payload)
    if city and city not in weather.CITIES:
        raise HTTPException(404, f"Unknown city: {city}")
    return {"source": payload.get("source"), "fetched_at_utc": dt.datetime.fromtimestamp(payload["fetched_at"], tz=dt.timezone.utc).isoformat(),
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
    payload, stale = guard(weather.get_payload)
    for name in sorted(data.city.unique()):
        forecast_context[name] = {
            "provider": payload.get("source", "Open-Meteo"),
            "fetched_at_utc": dt.datetime.fromtimestamp(payload["fetched_at"], tz=dt.timezone.utc).isoformat(),
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
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/stats")
def stats():
    return {"chatbot": "Gemini API with offline template fallback", "configured": bool(bot.client)}


# ------------------------------------------------------------------ Phase 2 operations (fully offline)
HorizonQ = Query(3, ge=2, le=4)
OriginQ = Query(None, description="Forecast origin week (YYYY-MM-DD). Default: newest reported week.")


def op(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown ward/city: {exc}")
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.get("/ops/meta")
def ops_meta():
    return {"origin_weeks": ops.origins(), "per_case_ratios": ops.ratios, "simulator_assumptions": ops.assumptions(),
            "bed_capacity": ops.capacity.to_dict(), "models": ops.manifest}


@app.get("/ops/forecast")
def ops_forecast(horizon_weeks: int = HorizonQ, origin_week: Optional[str] = OriginQ, city: Optional[str] = None):
    f = op(ops.forecast, horizon_weeks, origin_week)
    if city:
        f = f[f.city == city]
    return {"horizon_weeks": horizon_weeks, "count": len(f), "wards": json.loads(f.to_json(orient="records"))}


@app.get("/ops/allocation")
def ops_allocation(horizon_weeks: int = HorizonQ, stock_pct: float = Query(100, ge=30, le=100),
                   crews: int = Query(20, ge=0, le=500), origin_week: Optional[str] = OriginQ,
                   scope: Literal["region", "city"] = "region"):
    return op(ops.allocate, horizon_weeks, stock_pct, crews, origin_week, scope)


@app.get("/ops/overflow")
def ops_overflow(horizon_weeks: int = HorizonQ, origin_week: Optional[str] = OriginQ,
                 capacity_factor: float = Query(1.0, ge=0.1, le=5.0)):
    return op(ops.route_overflow, horizon_weeks, origin_week, capacity_factor)


class SimulateIn(BaseModel):
    deployments: Dict[str, int] = Field(..., description="ward_id -> number of larvicide/fogging crews")
    origin_week: Optional[str] = None


@app.post("/ops/simulate")
def ops_simulate(body: SimulateIn):
    return op(ops.simulate, body.deployments, body.origin_week)


@app.get("/ops/roi")
def ops_roi(crews_per_ward: int = Query(1, ge=1, le=config.MAX_CREWS_PER_WARD), origin_week: Optional[str] = OriginQ,
            top: int = Query(60, ge=1, le=60)):
    return op(ops.roi_ranking, crews_per_ward, origin_week, top)


def _notices(horizon_weeks, stock_pct, crews, origin_week, capacity_factor):
    alloc = op(ops.allocate, horizon_weeks, stock_pct, crews, origin_week)
    routing = op(ops.route_overflow, horizon_weeks, origin_week, capacity_factor)
    return dispatch.build_notices(alloc, routing), routing


@app.get("/ops/dispatch")
def ops_dispatch(fmt: Literal["json", "sms", "whatsapp", "html", "pdf"] = Query("json", alias="format"),
                 ward_id: Optional[str] = None, city: Optional[str] = None, horizon_weeks: int = HorizonQ,
                 stock_pct: float = Query(100, ge=30, le=100), crews: int = Query(20, ge=0, le=500),
                 origin_week: Optional[str] = OriginQ, capacity_factor: float = Query(1.0, ge=0.1, le=5.0),
                 only_alerts: bool = False):
    notices, routing = _notices(horizon_weeks, stock_pct, crews, origin_week, capacity_factor)
    if ward_id:
        notices = [n for n in notices if n["ward_id"] == ward_id]
    if city:
        notices = [n for n in notices if n["city"] == city]
    if only_alerts:
        notices = [n for n in notices if n["supply_status"] != "OK" or n["beds"]["net_deficit"]]
    if not notices and (ward_id or city):
        raise HTTPException(404, f"Unknown ward/city: {ward_id or city}")
    cities = {c: v for c, v in routing["cities"].items() if not city or c == city}
    title = f"Commissioner Dispatch Brief - {ward_id or city or 'All 60 wards'}"
    if fmt == "sms":
        return PlainTextResponse("\n".join(dispatch.sms(n) for n in notices))
    if fmt == "whatsapp":
        return PlainTextResponse("\n\n".join(dispatch.whatsapp(n) for n in notices))
    if fmt == "html":
        return HTMLResponse(dispatch.html_brief(notices, title, cities))
    if fmt == "pdf":
        return Response(dispatch.pdf_brief(notices, title, cities), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="dispatch_{ward_id or city or "all"}.pdf"'})
    return {"count": len(notices), "notices": notices, "sms": [dispatch.sms(n) for n in notices]}


class CaseReport(BaseModel):
    date: str
    ward_id: str
    dengue_cases: float = Field(ge=0)
    rainfall_mm: Optional[float] = None
    avg_temp_c: Optional[float] = None
    max_temp_c: Optional[float] = None
    min_temp_c: Optional[float] = None
    relative_humidity_pct: Optional[float] = None


@app.post("/ops/case-reports")
def ops_case_reports(rows: List[CaseReport]):
    """Append new weekly case reports so forecasts can start from the newest reported week."""
    return op(ops.add_case_reports, [r.model_dump(exclude_none=True) for r in rows])
