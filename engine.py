"""DengueWatch Phase 2 ML inference engine (101-feature Track 1 schema)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TIERS = ["Low", "Medium", "High", "Critical"]
HORIZONS = (2, 3, 4)
CASE_LAGS = [0, 1, 2, 3, 4, 6, 8, 12, 24, 52]
WEATHER_LAGS = [0, 1, 2, 3, 4, 5, 6, 8]
ROLLING_WINDOWS = [3, 4, 8]
MIN_HISTORY_WEEKS = max(CASE_LAGS) + 1  # lag-52 needs 53 weekly rows per ward

_STATIC = ["city", "ward_id", "population_density_sqkm", "water_logging_risk_index",
           "sanitation_score", "avg_temp_c", "max_temp_c", "min_temp_c", "rainfall_mm",
           "relative_humidity_pct", "temp_range"]
_WX = ["avg_temp_c", "max_temp_c", "min_temp_c", "rainfall_mm", "relative_humidity_pct"]

FEATURE_COLUMNS = (
    _STATIC
    + [f"cases_lag_{l}" for l in CASE_LAGS]
    + [f"cases_roll_{s}_{w}w" for s in ("mean", "std", "max") for w in (3, 4, 8, 12)]
    + ["cases_trend_3v8"]
    + [f"{c}_lag_{l}" for c in _WX for l in WEATHER_LAGS]
    + [f"{c}_roll_mean_{w}w" for c in _WX for w in ROLLING_WINDOWS]
    + ["rain_sum_2w", "rain_sum_4w", "rain_sum_8w", "rain_sum_3to6w",
       "woy_sin", "woy_cos", "month", "ward_vulnerability", "wl_x_sani_gap",
       "wl_x_rain4w", "vul_x_rain8w", "temp_x_humidity"]
)
assert len(FEATURE_COLUMNS) == 101

REQUIRED = ["date", "city", "ward_id", "dengue_cases", "population_density_sqkm",
            "water_logging_risk_index", "sanitation_score", "avg_temp_c", "max_temp_c",
            "min_temp_c", "rainfall_mm", "relative_humidity_pct"]


def validate_horizon(horizon: int) -> int:
    horizon = int(horizon)
    if horizon not in HORIZONS:
        raise ValueError(f"Unsupported forecast horizon: {horizon}. Available: H2, H3, H4.")
    return horizon


def build_features(raw_df: pd.DataFrame):
    """Return (DataFrame with original + engineered columns, 101 feature names).
    Every feature uses only data at or before each row's date (causal)."""
    missing = [c for c in REQUIRED if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required historical columns: {missing}")

    d = raw_df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    if d["date"].isna().any():
        raise ValueError("Invalid dates found in historical dataset")
    d["ward_id"] = d["ward_id"].astype(str).str.strip()
    d = d.sort_values(["ward_id", "date"]).reset_index(drop=True)
    if d.duplicated(["ward_id", "date"]).any():
        raise ValueError("Duplicate ward/date observations found")

    g = d.groupby("ward_id", sort=False)
    new = {}
    d["temp_range"] = d["max_temp_c"] - d["min_temp_c"]

    for lag in CASE_LAGS:
        new[f"cases_lag_{lag}"] = g["dengue_cases"].shift(lag)

    for w in (3, 4, 8, 12):
        new[f"cases_roll_mean_{w}w"] = g["dengue_cases"].transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean())
        new[f"cases_roll_std_{w}w"] = g["dengue_cases"].transform(
            lambda s, w=w: s.rolling(w, min_periods=2).std().fillna(0))
        new[f"cases_roll_max_{w}w"] = g["dengue_cases"].transform(
            lambda s, w=w: s.rolling(w, min_periods=1).max())
    new["cases_trend_3v8"] = new["cases_roll_mean_3w"] - new["cases_roll_mean_8w"]

    for c in _WX:
        for lag in WEATHER_LAGS:
            new[f"{c}_lag_{lag}"] = g[c].shift(lag)
        for w in ROLLING_WINDOWS:
            new[f"{c}_roll_mean_{w}w"] = g[c].transform(
                lambda s, w=w: s.rolling(w, min_periods=1).mean())

    for w in (2, 4, 8):
        new[f"rain_sum_{w}w"] = g["rainfall_mm"].transform(
            lambda s, w=w: s.rolling(w, min_periods=1).sum())
    new["rain_sum_3to6w"] = g["rainfall_mm"].transform(
        lambda s: s.shift(3).rolling(4, min_periods=1).sum())

    week = d["date"].dt.isocalendar().week.astype(int)
    new["woy_sin"] = np.sin(2 * np.pi * week / 52.18)
    new["woy_cos"] = np.cos(2 * np.pi * week / 52.18)
    new["month"] = d["date"].dt.month

    # Verified against Colab-saved features: vulnerability = water-logging x sanitation gap.
    vul = d["water_logging_risk_index"] * (1 - d["sanitation_score"])
    new["ward_vulnerability"] = vul
    new["wl_x_sani_gap"] = d["water_logging_risk_index"] * (1 - d["sanitation_score"])
    new["wl_x_rain4w"] = d["water_logging_risk_index"] * new["rain_sum_4w"]
    new["vul_x_rain8w"] = vul * new["rain_sum_8w"]
    new["temp_x_humidity"] = d["avg_temp_c"] * d["relative_humidity_pct"]

    d = pd.concat([d, pd.DataFrame(new, index=d.index)], axis=1)
    return d, FEATURE_COLUMNS.copy()


def validate_model_features(regressor, classifier) -> list[str]:
    reg = getattr(regressor, "feature_names_in_", None)
    clf = getattr(classifier, "feature_names_in_", None)
    if reg is None or clf is None:
        raise ValueError("Phase 2 models must expose feature_names_in_")
    if list(reg) != list(clf) or list(reg) != FEATURE_COLUMNS:
        raise ValueError("Saved model features do not match the Phase 2 feature schema")
    return list(reg)


def _decode_risk_class(value) -> str:
    if isinstance(value, str) and value.strip().title() in TIERS:
        return value.strip().title()
    try:
        return TIERS[int(value)]  # 0 Low, 1 Medium, 2 High, 3 Critical
    except (ValueError, TypeError, IndexError):
        return "Unknown"


def predict_phase2(raw_df: pd.DataFrame, regressor, classifier, horizon: int) -> pd.DataFrame:
    """Latest-week forecast per ward for one horizon (H+2/3/4 weeks)."""
    horizon = validate_horizon(horizon)
    features = validate_model_features(regressor, classifier)
    engineered, _ = build_features(raw_df)

    counts = engineered.groupby("ward_id").size()
    short = counts[counts < MIN_HISTORY_WEEKS]
    if len(short):
        raise ValueError(
            f"Need >= {MIN_HISTORY_WEEKS} weekly rows per ward (lag-52 feature); "
            f"too short: {short.to_dict()}")

    latest = engineered.groupby("ward_id", sort=False).tail(1).copy()
    X = latest[features]
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Missing values in model inputs: {bad[:8]}")

    cases = np.maximum(np.asarray(regressor.predict(X), dtype=float), 0.0)
    proba = np.asarray(classifier.predict_proba(X), dtype=float)
    classes = list(classifier.classes_)
    if proba.shape[1] != len(classes):
        raise ValueError("Classifier probability dimensions do not match its classes")
    pmap = {_decode_risk_class(c): proba[:, i] for i, c in enumerate(classes)}

    out = latest[["ward_id", "city", "date", "population_density_sqkm"]].copy()
    out["expected_cases"] = cases
    for t in TIERS:
        out[f"p_{t}"] = pmap.get(t, 0.0)
    out["risk_tier"] = [_decode_risk_class(classes[i]) for i in proba.argmax(axis=1)]
    out["horizon_weeks"] = horizon
    out["forecast_week"] = out["date"] + pd.to_timedelta(horizon * 7, unit="D")
    return out.reset_index(drop=True)

def get_model_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return only the ordered 101 model input columns."""
    feature_frame, columns = build_features(raw_df)
    return feature_frame[columns].copy()
