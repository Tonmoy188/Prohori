"""Optional weather context for DengueWatch (Open-Meteo, with an offline fallback).

Design rules (hackathon requirement 5 - offline resilience):
  * Forecasting, allocation, routing and dispatch NEVER depend on this module.
  * Every public entry point used by the app (`weather_context`) is guaranteed not to
    raise and not to block longer than a few seconds, even with no internet.
  * When the network is unavailable (or offline mode is switched on) it falls back to
    deterministic values computed from the stored Phase 2 dataset.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "onsite_dengue_clean.csv"
CACHE_FILE = ROOT / "cache" / "weather.json"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

TTL_S = int(os.environ.get("WEATHER_TTL_S", 3 * 3600))
RETRY_S = int(os.environ.get("WEATHER_RETRY_S", 300))
TIMEOUT_S = float(os.environ.get("WEATHER_TIMEOUT_S", 4))  # short: fail fast when offline

DAILY_COLUMNS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "relative_humidity_2m_mean",
]
DAILY = ",".join(DAILY_COLUMNS)
RENAME = {
    "temperature_2m_max": "max_temp_c", "temperature_2m_min": "min_temp_c",
    "temperature_2m_mean": "avg_temp_c", "precipitation_sum": "rainfall_mm",
    "relative_humidity_2m_mean": "relative_humidity_pct",
}
WEATHER_COLS = ["avg_temp_c", "max_temp_c", "min_temp_c", "rainfall_mm", "relative_humidity_pct"]

CITIES = {
    "Dhaka North": (23.81, 90.41), "Dhaka South": (23.71, 90.41),
    "Chattogram": (22.34, 91.83), "Gazipur": (23.99, 90.42),
    "Narayanganj": (23.62, 90.50),
}

_lock = threading.Lock()
_memory_cache: dict = {}
_fail_until = 0.0
_force_offline = os.environ.get("DENGUEWATCH_OFFLINE", "0").strip().lower() in {"1", "true", "yes"}
_dataset_cache: pd.DataFrame | None = None


class WeatherUnavailable(Exception):
    """Raised when live weather data cannot be obtained."""


# ----------------------------------------------------------------------------
# Offline switch (usable as a live "kill switch" demo toggle in the dashboard)
# ----------------------------------------------------------------------------
def set_offline(value: bool) -> None:
    global _force_offline
    _force_offline = bool(value)


def is_offline() -> bool:
    return _force_offline


# ----------------------------------------------------------------------------
# Live download
# ----------------------------------------------------------------------------
def _dataset() -> pd.DataFrame:
    global _dataset_cache
    if _dataset_cache is None:
        cols = ["date", "city"] + WEATHER_COLS
        _dataset_cache = pd.read_csv(DATASET_PATH, usecols=cols, parse_dates=["date"])
    return _dataset_cache


def _get_json(url: str, params: dict) -> list:
    try:
        import requests  # optional dependency: only needed for live weather
    except ImportError as exc:
        raise WeatherUnavailable("requests is not installed") from exc
    response = requests.get(url, params=params, timeout=TIMEOUT_S)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else [data]


def _download() -> dict:
    today = pd.Timestamp.now(tz="Asia/Dhaka").date()
    # Dataset weeks start Monday; the next week to fetch starts 7 days after the last one.
    history_start = _dataset()["date"].max().date() + timedelta(days=7)
    archive_end = today - timedelta(days=6)  # archive API lags ~5 days
    names = list(CITIES)
    coords = {
        "latitude": ",".join(str(CITIES[n][0]) for n in names),
        "longitude": ",".join(str(CITIES[n][1]) for n in names),
        "daily": DAILY,
        "timezone": "Asia/Dhaka",
    }
    try:
        forecast = _get_json(FORECAST_URL, {**coords, "past_days": 7, "forecast_days": 16})
        if len(forecast) != len(names):
            raise ValueError("unexpected forecast response count")
        archive: list = []
        if history_start <= archive_end:
            archive = _get_json(ARCHIVE_URL, {**coords, "start_date": history_start.isoformat(),
                                              "end_date": archive_end.isoformat()})
            if len(archive) != len(names):
                raise ValueError("unexpected archive response count")

        cities = {}
        for i, city in enumerate(names):
            rows: dict = {}
            sources = [archive[i].get("daily", {}) if archive else {}, forecast[i].get("daily", {})]
            for src in sources:  # forecast overwrites archive on overlapping days
                for j, day in enumerate(src.get("time", [])):
                    rows[day] = {c: src[c][j] for c in DAILY_COLUMNS}
            days = sorted(rows)
            cities[city] = {"daily": {"time": days,
                                      **{c: [rows[d].get(c) for d in days] for c in DAILY_COLUMNS}}}
        return {"fetched_at": time.time(), "source": "Open-Meteo Forecast + Archive API",
                "cities": cities}
    except Exception as exc:
        raise WeatherUnavailable(f"Open-Meteo request failed: {type(exc).__name__}: {exc}") from exc


def _read_cache_file() -> dict:
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and "cities" in data else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write_cache_file(payload: dict) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, CACHE_FILE)  # atomic: no half-written cache
    except OSError:
        pass


def get_payload() -> tuple[dict, bool]:
    """Return (weather payload, is_stale). Raises WeatherUnavailable if nothing usable."""
    global _fail_until
    if _force_offline:
        raise WeatherUnavailable("offline mode is on")

    with _lock:
        if not _memory_cache:
            _memory_cache.update(_read_cache_file())
        cached = dict(_memory_cache)
        if cached and time.time() - float(cached.get("fetched_at", 0)) < TTL_S:
            return cached, False
        if time.time() < _fail_until:
            if cached:
                return cached, True
            raise WeatherUnavailable("weather API recently failed; retrying shortly")

    try:  # network call happens OUTSIDE the lock so other requests never queue behind it
        payload = _download()
    except WeatherUnavailable:
        with _lock:
            _fail_until = time.time() + RETRY_S
        if cached:
            return cached, True
        raise

    with _lock:
        _memory_cache.clear()
        _memory_cache.update(payload)
    _write_cache_file(payload)
    return payload, False


# ----------------------------------------------------------------------------
# Live-data helpers (same names/behaviour as the Phase 1 module)
# ----------------------------------------------------------------------------
def daily_frame(payload: dict, city: str) -> pd.DataFrame:
    if city not in payload.get("cities", {}):
        raise WeatherUnavailable(f"No weather response available for {city}")
    frame = pd.DataFrame(payload["cities"][city]["daily"]).rename(columns=RENAME)
    frame["date"] = pd.to_datetime(frame.pop("time"))
    today = pd.Timestamp.now(tz="Asia/Dhaka").date()
    frame["is_forecast"] = frame["date"].dt.date >= today
    return frame.dropna(subset=WEATHER_COLS).reset_index(drop=True)


def weekly_by_city(payload: dict) -> dict[str, pd.DataFrame]:
    """Complete Monday-start weeks (7 observed/forecast days) per city."""
    today = pd.Timestamp.now(tz="Asia/Dhaka").tz_localize(None).normalize()
    out = {}
    for city in payload.get("cities", {}):
        f = daily_frame(payload, city)
        f["date"] = f["date"] - pd.to_timedelta(f["date"].dt.dayofweek, unit="D")
        weekly = f.groupby("date").agg(
            rainfall_mm=("rainfall_mm", "sum"), avg_temp_c=("avg_temp_c", "mean"),
            max_temp_c=("max_temp_c", "mean"), min_temp_c=("min_temp_c", "mean"),
            relative_humidity_pct=("relative_humidity_pct", "mean"),
            n_days=("is_forecast", "size"), forecast_days=("is_forecast", "sum"),
        ).reset_index()
        out[city] = weekly[(weekly["n_days"] == 7) & (weekly["date"] <= today)] \
            .drop(columns=["n_days"]).reset_index(drop=True)
    return out


def forecast_daily(payload: dict, city: str | None = None) -> list[dict]:
    names = [city] if city else list(payload.get("cities", {}))
    today = pd.Timestamp.now(tz="Asia/Dhaka").date()
    results: list[dict] = []
    for name in names:
        f = daily_frame(payload, name)
        f = f[f["date"].dt.date >= today].copy()
        f.insert(0, "city", name)
        f["date"] = f["date"].dt.strftime("%Y-%m-%d")
        results.extend(f.round(2).to_dict("records"))
    return results


def next_week_forecast(payload: dict) -> dict:
    """Next Monday-Sunday forecast aggregate per city (only if all 7 days are present)."""
    today = pd.Timestamp.now(tz="Asia/Dhaka").tz_localize(None).normalize()
    start = today - pd.to_timedelta(today.dayofweek, unit="D") + pd.Timedelta(days=7)
    end = start + pd.Timedelta(days=6)
    out = {}
    for city in payload.get("cities", {}):
        f = daily_frame(payload, city)
        f = f[(f["date"] >= start) & (f["date"] <= end)]
        if len(f) == 7:
            out[city] = {"rain_fwd1": float(f["rainfall_mm"].sum()),
                         "temp_fwd1": float(f["avg_temp_c"].mean()),
                         "hum_fwd1": float(f["relative_humidity_pct"].mean())}
    return out


def forecast_last_date(payload: dict, city: str):
    f = daily_frame(payload, city)
    return None if f.empty else f["date"].max().date()


# ----------------------------------------------------------------------------
# Deterministic offline fallback (from the stored dataset only)
# ----------------------------------------------------------------------------
def _city_weekly_from_dataset() -> dict[str, pd.DataFrame]:
    d = _dataset()
    w = d.groupby(["city", "date"], as_index=False)[WEATHER_COLS].mean()
    return {c: g.sort_values("date").reset_index(drop=True) for c, g in w.groupby("city")}


def offline_context(recent_weeks: int = 8) -> dict:
    """Last observed weeks + seasonal-average expectation for the following week."""
    cities = {}
    for city, g in _city_weekly_from_dataset().items():
        last = g["date"].max()
        target_week = int((last + pd.Timedelta(weeks=1)).isocalendar().week)
        weeks = g["date"].dt.isocalendar().week.astype(int)
        seasonal = g[(weeks - target_week).abs() <= 1]  # same season +/- 1 week, all years
        cities[city] = {
            "recent_weeks": g.tail(recent_weeks).assign(
                date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).round(2).to_dict("records"),
            "next_week": {
                "rain_fwd1": round(float(seasonal["rainfall_mm"].mean()), 2),
                "temp_fwd1": round(float(seasonal["avg_temp_c"].mean()), 2),
                "hum_fwd1": round(float(seasonal["relative_humidity_pct"].mean()), 2),
            },
        }
    return {"status": "offline_fallback",
            "source": "Stored Phase 2 dataset (seasonal averages; no internet used)",
            "cities": cities}


def weather_context(recent_weeks: int = 8) -> dict:
    """App-facing entry point. NEVER raises; falls back to stored data when offline."""
    try:
        payload, stale = get_payload()
        nxt = next_week_forecast(payload)
        weekly = weekly_by_city(payload)
        cities = {
            c: {"recent_weeks": w.tail(recent_weeks).assign(
                    date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).round(2).to_dict("records"),
                "next_week": nxt.get(c)}
            for c, w in weekly.items()
        }
        return {"status": "cached_stale" if stale else "live",
                "source": payload.get("source", "Open-Meteo"),
                "fetched_at": payload.get("fetched_at"), "cities": cities}
    except Exception as exc:  # any failure -> deterministic local data, app keeps working
        try:
            ctx = offline_context(recent_weeks)
        except Exception as inner:
            return {"status": "unavailable", "source": "none", "cities": {},
                    "detail": f"{type(inner).__name__}: {inner}"}
        ctx["detail"] = f"live weather unavailable ({type(exc).__name__}: {exc})"
        return ctx
