"""Train reproducible 2/3/4-week dengue models. Run from the project root: python scripts/train_models.py

Inputs for origin week t: weather through week t, ward attributes and (new) a forecast of week t+1.
The dataset has no archived forecasts, so the t+1 forecast is simulated as observed t+1 weather plus noise
(engine.add_forecast_proxy). Both the forecast-aware and the no-forecast models are validated on the same
chronological holdout (2024-2025, with an h-week embargo) and the report keeps both for comparison.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import engine

DATA = ROOT / "data" / "dengue_history.csv"
MODELS = ROOT / "models"
REPORT = ROOT / "training_report.json"
CUTOFF = pd.Timestamp("2024-01-01")
FEATURE_VERSION = "causal_weekly_weather_v3_fwd1"


def prepare():
    df = pd.read_csv(DATA, parse_dates=["date"])
    required = {"date", "city", "ward_id", "population_density_sqkm", "water_logging_risk_index",
                "sanitation_score", "avg_temp_c", "max_temp_c", "min_temp_c", "rainfall_mm",
                "relative_humidity_pct", "dengue_cases"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    df = df.sort_values(["ward_id", "date"]).reset_index(drop=True)
    if df.duplicated(["ward_id", "date"]).any():
        raise ValueError("Duplicate ward_id/date rows found")
    engineered, features = engine.build_features(engine.add_forecast_proxy(df))
    return df, engineered, features


def make_clf():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                          l2_regularization=1.0, random_state=0)


def make_reg():
    return HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05,
                                         max_depth=4, l2_regularization=1.0, random_state=0)


def fit(train, cols):
    return {"clf": make_clf().fit(train[cols], train.target_tier), "reg": make_reg().fit(train[cols], train.target_cases),
            "nb_alpha": engine.NB_ALPHA, "features": cols, "class_weights": [1, 1, 1, 1]}


def evaluate(bundle, test):
    """Accuracy, error and calibration of the production blend on unseen future weeks."""
    P = engine.predict_proba(bundle, test[bundle["features"]])
    P = P / P.sum(1, keepdims=True)
    cases = np.maximum(0, engine.predict_cases(bundle, test[bundle["features"]]))
    y = test.target_tier.to_numpy()
    onehot = np.eye(4)[y]
    recall = recall_score(y, P.argmax(1), labels=[0, 1, 2, 3], average=None, zero_division=0)
    return {
        "tier_accuracy": round(float(accuracy_score(y, P.argmax(1))), 4),
        "tier_macro_f1": round(float(f1_score(y, P.argmax(1), average="macro", zero_division=0)), 4),
        "cases_mae": round(float(mean_absolute_error(test.target_cases, cases)), 4),
        "cases_rmse": round(float(mean_squared_error(test.target_cases, cases) ** 0.5), 4),
        "brier_score": round(float(((P - onehot) ** 2).sum(1).mean()), 4),          # lower = better calibrated probabilities
        "mean_predicted_cases": round(float(cases.mean()), 4),                       # compare with mean_actual_cases (bias)
        "mean_actual_cases": round(float(test.target_cases.mean()), 4),
        "tier_recall": dict(zip(engine.TIERS, [round(float(r), 3) for r in recall])),
    }


def main():
    raw, engineered, features = prepare()
    no_fwd = [f for f in features if f not in engine.FWD_COLS]
    MODELS.mkdir(exist_ok=True)
    reports = {"dataset": str(DATA.relative_to(ROOT)), "rows": int(len(raw)),
               "date_start": str(raw.date.min().date()), "date_end": str(raw.date.max().date()),
               "wards": int(raw.ward_id.nunique()), "feature_count": len(features),
               "feature_version": FEATURE_VERSION,
               "forecast_proxy": "next-week observed weather + simulated error (rain x(1+N(0,.3)), temp N(0,1.5), humidity N(0,5))",
               "horizons": {}}
    for h in (2, 3, 4):
        work = engineered[features].copy()
        work["date"] = raw.date
        work["target_cases"] = raw.groupby("ward_id", sort=False).dengue_cases.shift(-h)
        work = work.dropna(subset=features + ["target_cases"])
        work["target_tier"] = pd.cut(work.target_cases, [-1, 1, 5, 11, np.inf], labels=False).astype(int)
        # Embargo: no training target may fall inside the test period.
        train = work[work.date < CUTOFF - pd.Timedelta(weeks=h)]
        test = work[work.date >= CUTOFF]
        if train.empty or test.empty:
            raise RuntimeError(f"Not enough temporal data for horizon {h}")
        with_fwd = evaluate(fit(train, features), test)
        without = evaluate(fit(train, no_fwd), test)
        info = {"temporal_train_end": str(train.date.max().date()), "temporal_test_start": str(test.date.min().date()),
                "temporal_test_end": str(test.date.max().date()), "test_rows": int(len(test)),
                **with_fwd, "without_forecast_features": without,
                "baseline_mean_mae": round(float(mean_absolute_error(test.target_cases, np.full(len(test), train.target_cases.mean()))), 4)}

        bundle = {**fit(work, features), "horizon_weeks": h, "tiers": engine.TIERS, "case_cuts": engine.CUTS,
                  "trained_through": str(raw.date.max().date()), "training_rows": int(len(work)),
                  "training_dataset": "data/dengue_history.csv", "feature_version": FEATURE_VERSION,
                  "training_note": "Weather through origin week t + forecast of week t+1 (simulated in training, real Open-Meteo forecast at inference)."}
        joblib.dump(bundle, MODELS / f"dengue_h{h}w.joblib")
        reports["horizons"][str(h)] = {**info, "final_training_rows": int(len(work))}
        print(f"{h}-week: with forecast MAE {with_fwd['cases_mae']} F1 {with_fwd['tier_macro_f1']} | "
              f"without MAE {without['cases_mae']} F1 {without['tier_macro_f1']}")
    REPORT.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Saved models in {MODELS} and report at {REPORT}")


if __name__ == "__main__":
    main()
