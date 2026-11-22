"""Inference engine: feature code and scoring copied unchanged from the training pipeline (dengue_pipeline.py)."""
import numpy as np
import pandas as pd
from scipy import stats

TIERS = ["Low", "Medium", "High", "Critical"]
CASE_BINS = [-1, 1, 5, 11, 10_000]      # Low 0-1, Medium 2-5, High 6-11, Critical >=12 cases/ward/week
CUTS = [1, 5, 11]                       # upper edges of Low / Medium / High (for the count model)
FWD_COLS = ["rain_fwd1", "temp_fwd1", "hum_fwd1"]   # forecast of the NEXT week (t+1); NaN when no forecast exists
NB_ALPHA = 0.2                          # NegBin dispersion (chosen from CV; variance > mean in the data)


def add_forecast_proxy(df, seed=0, rain_cv=0.3, temp_sd=1.5, hum_sd=5.0):
    """TRAINING ONLY. The dataset has no archived forecasts, so 'forecast of week t+1' is approximated by
    week t+1's observed weather plus simulated forecast error. At inference the real Open-Meteo forecast is used."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    g = d.groupby("ward_id")
    d["rain_fwd1"] = (g.rainfall_mm.shift(-1) * (1 + rng.normal(0, rain_cv, len(d)))).clip(lower=0)
    d["temp_fwd1"] = g.avg_temp_c.shift(-1) + rng.normal(0, temp_sd, len(d))
    d["hum_fwd1"] = (g.relative_humidity_pct.shift(-1) + rng.normal(0, hum_sd, len(d))).clip(0, 100)
    return d


def build_features(df):
    """Causal features: at week t only weather observed at weeks <= t is used. Grouped per ward_id."""
    d = df.copy()
    for c in FWD_COLS:                       # forecast columns are supplied by the caller; absent -> NaN (models handle NaN)
        if c not in d:
            d[c] = np.nan
    g = d.groupby("ward_id")
    new = {}
    new["temp_range"] = d.max_temp_c - d.min_temp_c
    for L in range(1, 9):                                           # 1-8 week rainfall lags
        new[f"rain_lag{L}"] = g.rainfall_mm.shift(L)
    for L in [1, 2, 3, 4, 6, 8]:
        new[f"temp_lag{L}"] = g.avg_temp_c.shift(L)
        new[f"hum_lag{L}"] = g.relative_humidity_pct.shift(L)
    for w in [2, 3, 4, 6, 8]:                                       # rolling rainfall totals (weeks t-w+1..t)
        new[f"rain_sum{w}w"] = g.rainfall_mm.transform(lambda s: s.rolling(w, min_periods=w).sum())
    r2 = g.rainfall_mm.shift(2)                                     # breeding window: rain 2-4 weeks ago
    new["rain_sum_lag2to4"] = r2.groupby(d.ward_id).transform(lambda s: s.rolling(3, min_periods=3).sum())
    r3 = g.rainfall_mm.shift(3)
    new["rain_sum_lag3to6"] = r3.groupby(d.ward_id).transform(lambda s: s.rolling(4, min_periods=4).sum())
    new["temp_mean4w"] = g.avg_temp_c.transform(lambda s: s.rolling(4, min_periods=4).mean())
    new["temp_mean8w"] = g.avg_temp_c.transform(lambda s: s.rolling(8, min_periods=8).mean())
    new["hum_mean4w"] = g.relative_humidity_pct.transform(lambda s: s.rolling(4, min_periods=4).mean())
    new["hum_mean8w"] = g.relative_humidity_pct.transform(lambda s: s.rolling(8, min_periods=8).mean())
    new["rainy_weeks_8w"] = g.rainfall_mm.transform(lambda s: (s > 20).astype(float).rolling(8, min_periods=8).sum())
    woy = d.date.dt.isocalendar().week.astype(int).clip(upper=52)
    new["woy_sin"] = np.sin(2 * np.pi * woy / 52)
    new["woy_cos"] = np.cos(2 * np.pi * woy / 52)
    vul = d.water_logging_risk_index * (d.population_density_sqkm / 1e4) * (1 - d.sanitation_score)
    new["ward_vulnerability"] = vul
    new["wl_x_sani_gap"] = d.water_logging_risk_index * (1 - d.sanitation_score)
    new["vul_x_rain2to4"] = vul * new["rain_sum_lag2to4"]
    new["wl_x_rain4w"] = d.water_logging_risk_index * new["rain_sum4w"]
    new["temp_x_hum4w"] = new["temp_mean4w"] * new["hum_mean4w"] / 100
    d = pd.concat([d, pd.DataFrame(new)], axis=1)
    base = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score",
            "avg_temp_c", "max_temp_c", "min_temp_c", "rainfall_mm", "relative_humidity_pct"]
    feats = base + FWD_COLS + list(new.keys())
    return d, feats


def tier_probs_from_mu(mu, alpha=NB_ALPHA):
    mu = np.clip(mu, 1e-6, None)
    n = 1.0 / alpha
    cdf = [stats.nbinom.cdf(k, n, n / (n + mu)) for k in CUTS]
    P = np.column_stack([cdf[0], cdf[1] - cdf[0], cdf[2] - cdf[1], 1 - cdf[2]])
    return np.clip(P, 0, 1)


def predict_proba(bundle, X):
    X = X[bundle["features"]]
    p1 = bundle["clf"].predict_proba(X)
    p2 = tier_probs_from_mu(bundle["reg"].predict(X), bundle["nb_alpha"])
    P = 0.5 * p1 + 0.5 * p2
    return P * np.asarray(bundle["class_weights"])


def predict_cases(bundle, X):
    return bundle["reg"].predict(X[bundle["features"]])


def score_rows(raw_df, bundle):
    """Inference for the dashboard. raw_df: raw weekly rows (>= 8 weeks of history per ward, same columns as the CSV).
    Returns one row per ward-week with probabilities, predicted tier and expected case count for week t + horizon."""
    d, _ = build_features(raw_df.sort_values(["ward_id", "date"]))
    ok = d[[f for f in bundle["features"] if f not in FWD_COLS]].notna().all(axis=1)  # forecast may be missing
    d = d.loc[ok].copy()
    P = predict_proba(bundle, d); P = P / P.sum(1, keepdims=True)
    out = d[["ward_id", "city", "date"]].copy()
    out["forecast_week"] = out.date + pd.Timedelta(weeks=bundle["horizon_weeks"])
    for i, t in enumerate(TIERS): out[f"p_{t}"] = P[:, i]
    out["risk_tier"] = [TIERS[i] for i in P.argmax(1)]
    out["expected_cases"] = predict_cases(bundle, d)
    return out.reset_index(drop=True)
