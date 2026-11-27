"""Phase 2 feature builder for the onsite models in models/classifiers and models/v2_onsite/regressors.

Builds the 101 columns the saved sklearn Pipelines expect (case lags, rolling case statistics, weather lags,
rolling weather means, breeding-window rain, season and ward vulnerability). All features are causal and
grouped per ward_id: row t only uses weeks <= t.
"""
import numpy as np
import pandas as pd

WEATHER = ["avg_temp_c", "max_temp_c", "min_temp_c", "rainfall_mm", "relative_humidity_pct"]
CASE_LAGS = [0, 1, 2, 3, 4, 6, 8, 12, 24, 52]
CASE_WINDOWS = [3, 4, 8, 12]
WEATHER_LAGS = [0, 1, 2, 3, 4, 5, 6, 8]
WEATHER_WINDOWS = [3, 4, 8]
STATIC = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]
REQUIRED = ["date", "city", "ward_id", *STATIC, *WEATHER, "dengue_cases"]
MAX_HISTORY_WEEKS = 53  # cases_lag_52 needs 52 earlier weeks


def _roll(s, by, w, how):
    r = s.groupby(by).rolling(w, min_periods=1)
    return getattr(r, how)().reset_index(level=0, drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df]
    if missing:
        raise ValueError(f"Missing columns for feature building: {missing}")
    d = df.sort_values(["ward_id", "date"]).reset_index(drop=True)
    g = d.groupby("ward_id")
    new = {"temp_range": d.max_temp_c - d.min_temp_c}
    for k in CASE_LAGS:
        new[f"cases_lag_{k}"] = g.dengue_cases.shift(k)
    for w in CASE_WINDOWS:
        for how in ("mean", "std", "max"):
            new[f"cases_roll_{how}_{w}w"] = _roll(d.dengue_cases, d.ward_id, w, how)
    new["cases_trend_3v8"] = new["cases_roll_mean_3w"] / (new["cases_roll_mean_8w"] + 1)
    for col in WEATHER:
        for k in WEATHER_LAGS:
            new[f"{col}_lag_{k}"] = g[col].shift(k)
    for col in WEATHER:
        for w in WEATHER_WINDOWS:
            new[f"{col}_roll_mean_{w}w"] = _roll(d[col], d.ward_id, w, "mean")
    for w in (2, 4, 8):
        new[f"rain_sum_{w}w"] = _roll(d.rainfall_mm, d.ward_id, w, "sum")
    new["rain_sum_3to6w"] = _roll(g.rainfall_mm.shift(3), d.ward_id, 4, "sum")
    woy = d.date.dt.isocalendar().week.astype(int)
    new["woy_sin"] = np.sin(2 * np.pi * woy / 52)
    new["woy_cos"] = np.cos(2 * np.pi * woy / 52)
    new["month"] = d.date.dt.month
    vul = d.water_logging_risk_index * (d.population_density_sqkm / 1e4) * (1 - d.sanitation_score)
    new["ward_vulnerability"] = vul
    new["wl_x_sani_gap"] = d.water_logging_risk_index * (1 - d.sanitation_score)
    new["wl_x_rain4w"] = d.water_logging_risk_index * new["rain_sum_4w"]
    new["vul_x_rain8w"] = vul * new["rain_sum_8w"]
    new["temp_x_humidity"] = d.avg_temp_c * d.relative_humidity_pct / 100
    return pd.concat([d, pd.DataFrame(new)], axis=1)


def latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for the newest reported week of every ward (the forecast origin)."""
    recent = df.sort_values(["ward_id", "date"]).groupby("ward_id").tail(MAX_HISTORY_WEEKS)
    feats = build_features(recent)
    return feats.groupby("ward_id", as_index=False).tail(1).reset_index(drop=True)
