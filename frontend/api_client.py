"""HTTP client for the PROHORI DengueWatch FastAPI backend (main.py). No Streamlit imports."""
import os

import requests

BASE_URL = os.getenv("DENGUEWATCH_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.getenv("DENGUEWATCH_API_TIMEOUT", "30"))
CHAT_TIMEOUT = float(os.getenv("DENGUEWATCH_CHAT_TIMEOUT", "45"))
_session = requests.Session()


class ApiError(Exception):
    """kind: offline | timeout | http | invalid"""

    def __init__(self, message, status=None, kind="http"):
        super().__init__(message)
        self.message, self.status, self.kind = message, status, kind


def _detail(resp):
    try:
        d = resp.json().get("detail")
    except Exception:
        d = None
    if isinstance(d, list):  # FastAPI 422 validation errors
        d = "; ".join(str(i.get("msg", i)) for i in d if isinstance(i, dict))
    return str(d) if d else None


def _call(method, path, *, params=None, body=None, timeout=TIMEOUT, need=()):
    try:
        resp = _session.request(method, BASE_URL + path, params=params, json=body, timeout=timeout)
    except requests.exceptions.ConnectionError:
        raise ApiError(f"Cannot reach the DengueWatch backend at {BASE_URL}. "
                       "Start it with: python -m uvicorn main:app", kind="offline")
    except requests.exceptions.Timeout:
        raise ApiError(f"The backend did not answer within {timeout:.0f}s. Try again in a moment.", kind="timeout")
    except requests.exceptions.RequestException as exc:
        raise ApiError(f"Request failed: {type(exc).__name__}", kind="offline")
    if not resp.ok:
        raise ApiError(_detail(resp) or f"Backend returned HTTP {resp.status_code}.", status=resp.status_code)
    try:
        data = resp.json()
    except ValueError:
        raise ApiError("The backend returned a response that is not valid JSON.", kind="invalid")
    if not isinstance(data, dict) or any(k not in data for k in need):
        raise ApiError(f"Unexpected response from {path}: missing {', '.join(k for k in need if k not in (data or {}))}.",
                       kind="invalid")
    return data


def health():
    return _call("GET", "/health", timeout=5, need=("status",))


def stats():
    return _call("GET", "/stats", timeout=5, need=("configured",))


def wards():
    return _call("GET", "/wards", need=("wards",))


def weather(city=None):
    """GET /weather. Per-ward rows repeat the city-level values; 'weather' holds fetch/staleness metadata."""
    return _call("GET", "/weather", params={"city": city} if city else None,
                 need=("weather", "recent_live_weekly_weather", "forecast_by_city"))


def weather_forecast(city=None):
    """GET /weather/forecast. Daily provider forecast (today onward) plus source and staleness."""
    return _call("GET", "/weather/forecast", params={"city": city} if city else None,
                 need=("daily_forecast", "provider_forecast_days", "fetched_at_utc"))


def predict_all(horizon_weeks, city=None):
    params = {"horizon_weeks": horizon_weeks, **({"city": city} if city else {})}
    return _call("GET", "/predict/all", params=params, need=("wards", "forecast_week", "origin_week", "weather"))


def outlook(ward_id):
    return _call("GET", "/outlook", params={"ward_id": ward_id}, need=("horizons",))


def metrics():
    return _call("GET", "/metrics", need=("horizons",))


def chat(message, session_id, ward_id=None, horizon_weeks=None):
    body = {"message": message, "session_id": session_id}
    if ward_id:
        body["ward_id"] = ward_id
    if horizon_weeks:
        body["horizon_weeks"] = horizon_weeks
    return _call("POST", "/chat", body=body, timeout=CHAT_TIMEOUT, need=("reply",))


# ------------------------------------------------------------------ Phase 2 operations (offline backend)
def _raw(path, params):
    try:
        resp = _session.get(BASE_URL + path, params=params, timeout=TIMEOUT)
    except requests.exceptions.RequestException as exc:
        raise ApiError(f"Cannot reach the backend: {type(exc).__name__}", kind="offline")
    if not resp.ok:
        raise ApiError(_detail(resp) or f"Backend returned HTTP {resp.status_code}.", status=resp.status_code)
    return resp


def ops_meta():
    return _call("GET", "/ops/meta", need=("origin_weeks",))


def ops_allocation(horizon_weeks, stock_pct, crews, origin_week=None, scope="region"):
    p = {"horizon_weeks": horizon_weeks, "stock_pct": stock_pct, "crews": crews, "scope": scope}
    return _call("GET", "/ops/allocation", params={**p, **({"origin_week": origin_week} if origin_week else {})},
                 need=("wards", "totals"))


def ops_overflow(horizon_weeks, origin_week=None, capacity_factor=1.0):
    p = {"horizon_weeks": horizon_weeks, "capacity_factor": capacity_factor}
    return _call("GET", "/ops/overflow", params={**p, **({"origin_week": origin_week} if origin_week else {})},
                 need=("wards", "cities", "transfers"))


def ops_simulate(deployments, origin_week=None):
    return _call("POST", "/ops/simulate", body={"deployments": deployments, "origin_week": origin_week},
                 need=("totals_by_horizon",))


def ops_roi(crews_per_ward=1, origin_week=None):
    p = {"crews_per_ward": crews_per_ward, **({"origin_week": origin_week} if origin_week else {})}
    return _call("GET", "/ops/roi", params=p, need=("roi_ranking",))


def ops_dispatch(fmt, params):
    """fmt json returns dict; sms/whatsapp/html return text; pdf returns bytes."""
    params = {k: v for k, v in {**params, "format": fmt}.items() if v not in (None, "")}
    if fmt == "json":
        return _call("GET", "/ops/dispatch", params=params, need=("notices",))
    resp = _raw("/ops/dispatch", params)
    return resp.content if fmt == "pdf" else resp.text
