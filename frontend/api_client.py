"""Tiny HTTP client for the DengueWatch Phase 2 FastAPI backend (main.py). No Streamlit imports."""
import os

import requests

BASE_URL = os.getenv("DENGUEWATCH_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.getenv("DENGUEWATCH_API_TIMEOUT", "60"))
_session = requests.Session()


class ApiError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.message, self.status = message, status


def _call(method, path, params=None, body=None, as_text=False):
    try:
        r = _session.request(method, BASE_URL + path, params=params, json=body, timeout=TIMEOUT)
    except requests.exceptions.RequestException:
        raise ApiError(f"Cannot reach the backend at {BASE_URL}. Start it with: python -m uvicorn main:app")
    if not r.ok:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = None
        if isinstance(detail, list):
            detail = "; ".join(str(d.get("msg", d)) for d in detail)
        raise ApiError(str(detail or f"Backend returned HTTP {r.status_code}"), r.status_code)
    return r.text if as_text else r.json()


# ---- status & reference
def health():
    return _call("GET", "/health")


def stats():
    return _call("GET", "/stats")


def cities():
    return _call("GET", "/cities")["cities"]


def wards():
    return _call("GET", "/wards")["wards"]


# ---- forecasts
def predict_all(horizon, city=None):
    return _call("GET", "/predict/all", {"horizon_weeks": horizon, **({"city": city} if city else {})})


def outlook(ward_id):
    return _call("GET", "/outlook", {"ward_id": ward_id})


# ---- operations (all POST, JSON body)
def _ops(horizon, city, stock_pct, crews, overrides=None):
    b = {"horizon_weeks": horizon, "city": city, "available_regional_stock_pct": stock_pct,
         "total_spraying_crews": crews}
    if overrides is not None:
        b["bed_capacity_overrides"] = overrides
    return b


def allocate(horizon, city, stock_pct, crews):
    return _call("POST", "/operations/allocate", body=_ops(horizon, city, stock_pct, crews))


def overflow(horizon, city, stock_pct, crews, overrides=None):
    return _call("POST", "/operations/overflow", body=_ops(horizon, city, stock_pct, crews, overrides or {}))


def simulate(total_crews, city=None, deployments=None, effect_per_crew=None, max_effect=None):
    body = {"total_crews": total_crews, "city": city, "deployments": deployments or None,
            "effect_per_crew": effect_per_crew, "max_effect": max_effect}
    return _call("POST", "/operations/simulate", body=body)


def dispatch(horizon, city, stock_pct, crews):
    return _call("POST", "/operations/dispatch", body=_ops(horizon, city, stock_pct, crews, {}))


def dispatch_html(horizon, city, stock_pct, crews):
    return _call("POST", "/operations/dispatch/html", body=_ops(horizon, city, stock_pct, crews, {}), as_text=True)


def dispatch_export(horizon, city, stock_pct, crews):
    return _call("POST", "/operations/dispatch/export", body=_ops(horizon, city, stock_pct, crews, {}))

# ---- weather & assistant
def weather():
    return _call("GET", "/weather")


def set_offline(on):
    return _call("POST", f"/weather/offline/{'on' if on else 'off'}")


def chat(message, session_id, ward_id=None, horizon=None):
    body = {"message": message, "session_id": session_id}
    if ward_id:
        body["ward_id"] = ward_id
    if horizon:
        body["horizon_weeks"] = horizon
    return _call("POST", "/chat", body=body)
