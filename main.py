"""DengueWatch Phase 2 FastAPI app. Core endpoints are 100% local (no network, no LLM)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import config
import operations
import weather
from service import Service

app = FastAPI(title="DengueWatch Phase 2 API", version="2.1.0",
              description="Forecasting, constrained allocation, overflow routing, "
                          "vector-control simulation and dispatch. Fully offline-capable.")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

service = Service()
EXPORT_DIR = Path(__file__).resolve().parent / "exports"
_chat = None  # optional LLM assistant, created lazily; never used by core endpoints


class _Ops(BaseModel):
    horizon_weeks: int = Field(default=2, ge=2, le=4)
    city: Optional[str] = None
    available_regional_stock_pct: float = Field(default=100.0, ge=30, le=100)
    total_spraying_crews: int = Field(default=0, ge=0, le=10000)


class AllocationRequest(_Ops):
    pass

class OverflowRequest(_Ops):
    bed_capacity_overrides: Dict[str, int] = Field(default_factory=dict)


class SimulateRequest(BaseModel):
    total_crews: int = Field(default=10, ge=0, le=10000)
    city: Optional[str] = None
    deployments: Optional[Dict[str, int]] = None  # ward_id -> crews; omitted = auto (greedy by ROI)
    bed_capacity_overrides: Dict[str, int] = Field(default_factory=dict)
    effect_per_crew: Optional[float] = Field(default=None, gt=0, lt=1)
    max_effect: Optional[float] = Field(default=None, gt=0, le=1)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=300)
    session_id: str = Field(default="default", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    ward_id: Optional[str] = None
    horizon_weeks: Optional[int] = Field(default=None, ge=2, le=4)


def _city(city: Optional[str]) -> Optional[str]:
    if city is None or not city.strip():
        return None
    name = operations.ALLOWED_CITIES.get(city.strip().casefold())
    if name is None:
        raise HTTPException(404, f"Unknown city: {city}. Allowed: {sorted(operations.ALLOWED_CITIES.values())}")
    return name


def run(fn, *a, **k):
    try:
        return fn(*a, **k)
    except KeyError as exc:
        raise HTTPException(404, f"Not found: {exc}") from exc
    except ValueError as exc:  # includes operations/simulator validation errors
        raise HTTPException(422, str(exc)) from exc


@app.get("/")
def root():
    return {"service": "DengueWatch Phase 2", "status": "running", "docs": "/docs", "health": "/health"}


def _chat_status() -> str:
    if not config.GEMINI_API_KEY:
        return "local template only (no GEMINI_API_KEY)"
    return "local template only (offline switch on)" if weather.is_offline() else "gemini enabled (template fallback)"


@app.get("/health")
def health():
    return {"status": "ok", "loaded_horizons": sorted(service.models), "wards": len(service.available_wards()),
            "dataset_rows": len(service.history), "dataset_through": str(service.history["date"].max().date()),
            "offline_mode": weather.is_offline(), "core_requires_internet": False,
            "chatbot": _chat_status()}


@app.get("/wards")
def get_wards():
    w = service.available_wards()
    return {"count": len(w), "wards": w}


@app.get("/cities")
def get_cities():
    c = sorted({w["city"] for w in service.available_wards()})
    return {"count": len(c), "cities": c}


@app.get("/predict")
def predict_ward(ward_id: str, horizon_weeks: int = Query(default=2, ge=2, le=4)):
    return run(service.predict_ward, ward_id, horizon_weeks)


@app.get("/predict/all")
def predict_all(horizon_weeks: int = Query(default=2, ge=2, le=4), city: Optional[str] = None):
    return run(lambda: service.predict_all(horizon_weeks, _city(city)))


@app.get("/outlook")
def outlook(ward_id: str):
    return {"ward_id": ward_id, "forecasts": [run(service.predict_ward, ward_id, h) for h in (2, 3, 4)]}


def _ops_kwargs(b: _Ops) -> dict:
    return dict(horizon=b.horizon_weeks, city=_city(b.city),
                available_regional_stock_pct=b.available_regional_stock_pct,
                total_spraying_crews=b.total_spraying_crews)


@app.post("/operations/allocate")
def allocate(body: AllocationRequest):
    return run(lambda: service.allocate_resources(**_ops_kwargs(body)))


@app.post("/operations/overflow")
def overflow(body: OverflowRequest):
    return run(lambda: service.route_overflow(**_ops_kwargs(body), capacity_overrides=body.bed_capacity_overrides))


@app.post("/operations/simulate")
def simulate(body: SimulateRequest):
    params = {k: v for k, v in (("effect_per_crew", body.effect_per_crew),
                                ("max_effect", body.max_effect)) if v is not None}
    return run(lambda: service.simulate(body.total_crews, body.deployments, _city(body.city),
                                        body.bed_capacity_overrides, params))


@app.post("/operations/dispatch")
def dispatch(body: OverflowRequest):
    d = run(lambda: service.dispatch(**_ops_kwargs(body), capacity_overrides=body.bed_capacity_overrides))
    return {k: v for k, v in d.items() if k != "html_brief"}


@app.post("/operations/dispatch/html", response_class=HTMLResponse)
def dispatch_html(body: OverflowRequest):
    return run(lambda: service.dispatch(**_ops_kwargs(body), capacity_overrides=body.bed_capacity_overrides))["html_brief"]


@app.post("/operations/dispatch/export")
def dispatch_export(body: OverflowRequest):
    d = run(lambda: service.dispatch(**_ops_kwargs(body), capacity_overrides=body.bed_capacity_overrides))
    tag = (_city(body.city) or "all_cities").lower().replace(" ", "_") + f"_h{body.horizon_weeks}w"
    EXPORT_DIR.mkdir(exist_ok=True)
    sep = "\n" + "=" * 60 + "\n"
    files = {f"sms_{tag}.txt": sep.join(m["sms"] for m in d["ward_messages"]),
             f"whatsapp_{tag}.txt": sep.join(m["whatsapp"] for m in d["ward_messages"]),
             f"commissioner_brief_{tag}.html": d["html_brief"]}
    for name, text in files.items():
        (EXPORT_DIR / name).write_text(text, encoding="utf-8")
    return {"exported": [f"exports/{n}" for n in files], "wards": len(d["ward_messages"])}

@app.get("/weather")
def get_weather():
    return weather.weather_context()  # live Open-Meteo if online, deterministic local fallback otherwise


@app.post("/weather/offline/{state}")
def weather_offline(state: str):
    if state.lower() not in ("on", "off"):
        raise HTTPException(422, "state must be 'on' or 'off'")
    weather.set_offline(state.lower() == "on")
    return {"offline_mode": weather.is_offline()}


def _template_reply(req: ChatRequest) -> str:
    """Deterministic, no-LLM answer used when the Gemini assistant is unavailable/offline."""
    h = req.horizon_weeks or 2
    if req.ward_id:
        p = service.predict_ward(req.ward_id, h)
        return (f"{p['ward_id']} ({p['city']}): model estimate {p['expected_cases']:.1f} cases, risk "
                f"{p['risk_tier']}, for week of {p['forecast_week']} ({h}-week horizon).")
    a = service.predict_all(h)
    top = sorted(a["wards"], key=lambda w: -w["expected_cases"])[:3]
    return (f"Top model-estimated wards for week of {a['forecast_week']}: "
            + "; ".join(f"{w['ward_id']} {w['expected_cases']:.1f} cases ({w['risk_tier']})" for w in top) + ".")


@app.post("/chat")
def chat(body: ChatRequest):
    """Optional. Gemini only when configured and online; otherwise a deterministic local template.
    Rate limits (429), unknown wards (404) and bad input (422) are always reported, never hidden."""
    global _chat
    if config.GEMINI_API_KEY and not weather.is_offline():
        try:
            from chat import ChatRateLimited, ChatService  # lazy: core never imports the LLM client
            if _chat is None:
                _chat = ChatService(service)
            try:
                return _chat.ask(body.message, body.session_id, body.ward_id, body.horizon_weeks)
            except ChatRateLimited as exc:
                raise HTTPException(429, str(exc)) from exc
            except KeyError as exc:
                raise HTTPException(404, f"Unknown ward: {exc}") from exc
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        except HTTPException:
            raise
        except Exception:  # Gemini/network failure -> deterministic fallback
            pass
    return {"reply": run(_template_reply, body), "source": "deterministic_template", "cached": False}


@app.get("/stats")
def stats():
    return {"dataset_rows": len(service.history), "wards": len(service.available_wards()),
            "available_horizons": sorted(service.models), "chatbot_configured": bool(config.GEMINI_API_KEY),
            "offline_mode": weather.is_offline()}
