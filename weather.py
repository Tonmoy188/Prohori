"""Live weather ingestion from Open-Meteo Forecast + Historical Weather APIs.
No mock/demo weather data is used. Forecast endpoint is free and requires no API key.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import date, timedelta

import pandas as pd
import requests

import config

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY = "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean"
CITIES = {
    "Dhaka North": (23.81, 90.41),
    "Dhaka South": (23.71, 90.41),
    "Chattogram": (22.34, 91.83),
    "Gazipur": (23.99, 90.42),
    "Narayanganj": (23.62, 90.50),
}
CACHE_FILE = config.ROOT / "cache" / "weather.json"
_lock = threading.Lock()
_mem: dict = {}
_fail_until = 0.0


class WeatherUnavailable(Exception):
    pass


def _get_json(url: str, params: dict):
    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else [data]


def _download():
    """Fetch real observed weather since the historical CSV ends, plus current forecast.

    Open-Meteo Archive has a short publication delay, so the Forecast endpoint's past_days
    overlaps the archive by 7 days. Overlap is deduplicated when daily rows are combined.
    """
    today = pd.Timestamp.now(tz="Asia/Dhaka").date()
    history_start = config.HISTORICAL_END + timedelta(days=7)
    archive_end = today - timedelta(days=6)
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
            raise ValueError("Unexpected Open-Meteo forecast response count")
        archive = []
        if history_start <= archive_end:
            archive = _get_json(ARCHIVE_URL, {
                **coords,
                "start_date": history_start.isoformat(),
                "end_date": archive_end.isoformat(),
            })
            if len(archive) != len(names):
                raise ValueError("Unexpected Open-Meteo archive response count")
        cities = {}
        for i, name in enumerate(names):
            rows = {}
            for source in ((archive[i] if archive else {}).get("daily", {}), forecast[i].get("daily", {})):
                if not source:
                    continue
                for j, day in enumerate(source.get("time", [])):
                    rows[day] = {k: source[k][j] for k in DAILY.split(",")}
            days = sorted(rows)
            cities[name] = {"daily": {"time": days, **{
                k: [rows[d].get(k) for d in days] for k in DAILY.split(",")
            }}}
        return {"fetched_at": time.time(), "source": "Open-Meteo Forecast + Archive API",
                "forecast_days_available": 16, "cities": cities}
    except Exception as exc:
        raise WeatherUnavailable(f"Open-Meteo request failed: {type(exc).__name__}: {exc}") from exc


def get_payload():
    """Return real API data and a stale-cache flag. Never fabricate data when API fails."""
    with _lock:
        if not _mem and CACHE_FILE.exists():
            try:
                _mem.update(json.loads(CACHE_FILE.read_text(encoding="utf-8")))
            except Exception:
                _mem.clear()
        if _mem and time.time() - float(_mem["fetched_at"]) < config.WEATHER_TTL_S:
            return _mem, False
        global _fail_until
        if time.time() < _fail_until:  # recent failure: do not hit the API again on every request
            if _mem:
                return _mem, True
            raise WeatherUnavailable("Open-Meteo recently failed; retrying shortly")
        try:
            payload = _download()
        except WeatherUnavailable:
            _fail_until = time.time() + config.WEATHER_RETRY_S
            if _mem:
                return _mem, True
            raise
        _mem.clear()
        _mem.update(payload)
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass
        return _mem, False


def daily_frame(payload: dict, city: str) -> pd.DataFrame:
    """Return daily rows, identifying forecast days without replacing missing values."""
    if city not in payload["cities"]:
        raise WeatherUnavailable(f"No weather response for {city}")
    x = pd.DataFrame(payload["cities"][city]["daily"]).rename(columns={
        "temperature_2m_max": "max_temp_c",
        "temperature_2m_min": "min_temp_c",
        "temperature_2m_mean": "avg_temp_c",
        "precipitation_sum": "rainfall_mm",
        "relative_humidity_2m_mean": "relative_humidity_pct",
    })
    x["date"] = pd.to_datetime(x.pop("time"))
    x["is_forecast"] = x.date.dt.date >= pd.Timestamp.now(tz="Asia/Dhaka").date()
    return x.dropna(subset=["avg_temp_c", "max_temp_c", "min_temp_c", "rainfall_mm", "relative_humidity_pct"])


def weekly_by_city(payload: dict) -> dict[str, pd.DataFrame]:
    """Daily weather -> Monday-start weekly rows, including forecast-completed current week.

    Only the current week and past weeks are returned for model input. Future weeks are exposed
    separately by /weather/forecast and are not silently treated as observed historical features.
    """
    today = pd.Timestamp.now(tz="Asia/Dhaka").tz_localize(None).normalize()
    out = {}
    for city in payload["cities"]:
        x = daily_frame(payload, city)
        x["date"] = x.date - pd.to_timedelta(x.date.dt.dayofweek, unit="D")
        w = x.groupby("date").agg(
            rainfall_mm=("rainfall_mm", "sum"),
            avg_temp_c=("avg_temp_c", "mean"),
            max_temp_c=("max_temp_c", "mean"),
            min_temp_c=("min_temp_c", "mean"),
            relative_humidity_pct=("relative_humidity_pct", "mean"),
            n_days=("is_forecast", "size"),
            forecast_days=("is_forecast", "sum"),
        ).reset_index()
        # Keep complete weeks only; forecast data completes the current week.
        out[city] = w[(w.n_days == 7) & (w.date <= today)].drop(columns="n_days").reset_index(drop=True)
    return out


def forecast_daily(payload: dict, city: str | None = None):
    """Return the actual provider forecast horizon; never invent 3- or 4-week weather."""
    names = [city] if city else list(payload["cities"])
    result = []
    today = pd.Timestamp.now(tz="Asia/Dhaka").date()
    for name in names:
        x = daily_frame(payload, name)
        x = x[x.date.dt.date >= today].copy()
        x.insert(0, "city", name)
        x["date"] = x.date.dt.strftime("%Y-%m-%d")
        result.extend(x.round(2).to_dict("records"))
    return result


def next_week_forecast(payload: dict) -> dict:
    """Aggregate the provider forecast for the NEXT Monday-Sunday week (t+1) per city, the model's forecast input.
    Returns {} entries only for cities whose 7 forecast days are all available; those get NaN features (model handles it)."""
    today = pd.Timestamp.now(tz="Asia/Dhaka").tz_localize(None).normalize()
    start = today - pd.to_timedelta(today.dayofweek, unit="D") + pd.Timedelta(days=7)
    end = start + pd.Timedelta(days=6)
    out = {}
    for city in payload["cities"]:
        x = daily_frame(payload, city)
        x = x[(x.date >= start) & (x.date <= end)]
        if len(x) == 7:
            out[city] = {"rain_fwd1": float(x.rainfall_mm.sum()), "temp_fwd1": float(x.avg_temp_c.mean()),
                         "hum_fwd1": float(x.relative_humidity_pct.mean())}
    return out


def forecast_last_date(payload: dict, city: str):
    """Last calendar day the provider forecast covers for a city (None if unavailable)."""
    x = daily_frame(payload, city)
    return None if x.empty else x.date.max().date()
