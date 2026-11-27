"""DengueWatch Phase 2 prediction and operations service (fully offline)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import engine
import operations
import simulator

HORIZONS = (2, 3, 4)
TIERS = ("Low", "Medium", "High", "Critical")
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Critical",
               "0": "Low", "1": "Medium", "2": "High", "3": "Critical",
               "low": "Low", "medium": "Medium", "high": "High", "critical": "Critical"}



def _load_model(path):
    """joblib.load, with support for portable XGBoost classifiers (booster stored as JSON)."""
    obj = joblib.load(path)
    if isinstance(obj, dict) and obj.get("portable_xgb"):
        from sklearn.pipeline import Pipeline
        from xgboost import XGBClassifier
        clf = XGBClassifier(**obj["params"])
        clf.load_model(bytearray(obj["booster"]))
        return Pipeline(list(obj["prefix"]) + [(obj["name"], clf)])
    return obj

class Service:
    """Loads Phase 2 models and serves multi-horizon ward predictions."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root else Path(__file__).resolve().parent
        model_dir = self.root / "models" / "v2_onsite"
        self.dataset_path = self.root / "data" / "onsite_dengue_clean.csv"

        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"Phase 2 dataset not found: {self.dataset_path}")
        self.history = pd.read_csv(self.dataset_path, parse_dates=["date"])
        missing = set(engine.REQUIRED) - set(self.history.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
        self.history["ward_id"] = self.history["ward_id"].astype(str).str.strip()
        self.history = self.history.sort_values(["ward_id", "date"]).reset_index(drop=True)

        self.models: dict[int, dict[str, Any]] = {}
        self._cache: dict[int, pd.DataFrame] = {}
        for h in HORIZONS:
            paths = {"regressor": model_dir / "regressors" / f"final_regressor_h{h}.joblib",
                     "classifier": model_dir / "classifiers" / f"final_classifier_h{h}.joblib"}
            for kind, p in paths.items():
                if not p.is_file():
                    raise FileNotFoundError(f"Missing H{h} {kind}: {p}")
            loaded = {}
            for kind, p in paths.items():
                try:
                    loaded[kind] = _load_model(p)
                except Exception as exc:  # name the exact file and the real underlying error
                    import traceback
                    detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                    raise RuntimeError(
                        f"Could not load H{h} {kind} ({p.name}, {p.stat().st_size:,} bytes): {detail}\n"
                        "Expected scikit-learn==1.6.1 and the xgboost version used for training."
                    ) from exc
            reg, clf = loaded["regressor"], loaded["classifier"]
            # Raises if schemas differ from each other or from the 101-feature schema.
            engine.validate_model_features(reg, clf)
            self.models[h] = {"regressor": reg, "classifier": clf}

        self._features, _ = engine.build_features(self.history.copy())
        self._features = self._features.sort_values(["ward_id", "date"]).reset_index(drop=True)
        counts = self._features.groupby("ward_id").size()
        if (counts < engine.MIN_HISTORY_WEEKS).any():
            raise ValueError(f"Each ward needs >= {engine.MIN_HISTORY_WEEKS} weekly rows of history")

        # Documented assumption: dataset has no bed-capacity column (see operations.py).
        self.bed_capacity = operations.estimate_bed_capacity(
            self.history.groupby("ward_id")["hospital_beds_required"].apply(list).to_dict()
        ) if "hospital_beds_required" in self.history.columns else {}

    @staticmethod
    def _risk_label(value: Any) -> str:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, str):
            value = value.strip().lower()
        return RISK_LABELS.get(value, "Unknown")

    def _predict_horizon(self, horizon: int) -> pd.DataFrame:
        if horizon not in self.models:
            raise ValueError("Horizon must be 2, 3, or 4 weeks")
        if horizon in self._cache:
            return self._cache[horizon].copy()

        m = self.models[horizon]
        features = list(m["regressor"].feature_names_in_)
        latest = self._features.groupby("ward_id", sort=True).tail(1).copy()
        X = latest[features]
        if X.isna().any().any():
            raise ValueError(f"Missing values in model inputs: {X.columns[X.isna().any()].tolist()[:8]}")

        cases = np.maximum(np.asarray(m["regressor"].predict(X), dtype=float), 0.0)
        proba = np.asarray(m["classifier"].predict_proba(X), dtype=float)  # one batch call
        classes = list(m["classifier"].classes_)
        labels = [self._risk_label(c) for c in classes]

        latest["expected_cases"] = np.round(cases, 2)
        latest["risk_tier"] = [labels[i] for i in proba.argmax(axis=1)]
        for t in TIERS:
            col = np.zeros(len(latest))
            for i, lab in enumerate(labels):
                if lab == t:
                    col = proba[:, i]
            latest[f"p_{t.lower()}"] = np.round(col, 4)
        latest["horizon_weeks"] = horizon
        latest["origin_week"] = latest["date"]
        latest["forecast_week"] = latest["date"] + pd.to_timedelta(horizon * 7, unit="D")
        latest = latest.reset_index(drop=True)
        self._cache[horizon] = latest
        return latest.copy()

    @staticmethod
    def _public_row(row: Any) -> dict[str, Any]:
        return {
            "ward_id": str(row.ward_id), "city": str(row.city),
            "risk_tier": str(row.risk_tier), "expected_cases": float(row.expected_cases),
            "probabilities": {"Low": float(row.p_low), "Medium": float(row.p_medium),
                              "High": float(row.p_high), "Critical": float(row.p_critical)},
            "population_density_sqkm": float(row.population_density_sqkm),
        }

    def predict_all(self, horizon: int = 2, city: str | None = None) -> dict[str, Any]:
        pred = self._predict_horizon(int(horizon))
        if city:
            pred = pred[pred["city"].astype(str).str.casefold() == city.strip().casefold()]
        if pred.empty:
            raise KeyError(f"No ward predictions for city: {city}")
        rows = [self._public_row(r) for r in pred.itertuples(index=False)]
        origin = pd.to_datetime(pred["origin_week"]).max()
        return {
            "horizon_weeks": int(horizon),
            "origin_week": str(origin.date()),
            "forecast_week": str((origin + pd.Timedelta(weeks=int(horizon))).date()),
            "dataset_through": str(self.history["date"].max().date()),
            "source": "Phase 2 historical dengue dataset (local models)",
            "count": len(rows),
            "wards": rows,
        }

    def predict_ward(self, ward_id: str, horizon: int = 2) -> dict[str, Any]:
        pred = self._predict_horizon(int(horizon))
        hit = pred[pred["ward_id"].astype(str) == str(ward_id).strip()]
        if hit.empty:
            raise KeyError(f"Ward not found: {ward_id}")
        row = hit.iloc[0]
        out = self._public_row(row)
        out.update({
            "horizon_weeks": int(horizon),
            "origin_week": str(pd.Timestamp(row["origin_week"]).date()),
            "forecast_week": str(pd.Timestamp(row["forecast_week"]).date()),
            "dataset_through": str(self.history["date"].max().date()),
        })
        return out

    def allocate_resources(self, horizon: int = 2, city: str | None = None,
                           available_regional_stock_pct: float = 100.0,
                           total_spraying_crews: int = 0) -> dict[str, Any]:
        pred = self.predict_all(horizon, city)
        alloc = operations.allocate_resources(
            pred["wards"], available_regional_stock_pct=available_regional_stock_pct,
            total_spraying_crews=total_spraying_crews)
        return {"horizon_weeks": int(horizon), "forecast_week": pred["forecast_week"],
                "source": pred["source"], "allocation": alloc}

    def route_overflow(self, horizon: int = 2, city: str | None = None,
                       available_regional_stock_pct: float = 100.0,
                       total_spraying_crews: int = 0,
                       capacity_overrides: dict[str, int] | None = None) -> dict[str, Any]:
        """Allocation + same-city overflow routing (capacity = estimate unless overridden)."""
        base = self.allocate_resources(horizon, city, available_regional_stock_pct, total_spraying_crews)
        wards = base["allocation"]["wards"]
        ids = {w["ward_id"] for w in wards}
        capacity = {k: v for k, v in self.bed_capacity.items() if k in ids}
        capacity.update({str(k): v for k, v in (capacity_overrides or {}).items() if str(k) in ids})
        base["overflow"] = operations.route_hospital_overflow(wards, capacity)
        base["bed_capacity_source"] = "historical 90th-percentile estimate; operator overrides applied"
        return base

    def dispatch(self, horizon: int = 2, city: str | None = None,
                 available_regional_stock_pct: float = 100.0, total_spraying_crews: int = 0,
                 capacity_overrides: dict[str, int] | None = None) -> dict[str, Any]:
        """SMS / WhatsApp / HTML brief payload for every ward."""
        r = self.route_overflow(horizon, city, available_regional_stock_pct,
                                total_spraying_crews, capacity_overrides)
        return operations.build_dispatch_payload(
            r["allocation"], int(horizon), r["forecast_week"], r["overflow"])

    def simulate(self, total_crews: int = 0, deployments: dict[str, int] | None = None,
                 city: str | None = None, capacity_overrides: dict[str, int] | None = None,
                 params: dict[str, float] | None = None) -> dict[str, Any]:
        """Vector-control what-if: model-driven counterfactual over weeks 2-4."""
        latest = self._features.groupby("ward_id", sort=True).tail(1)
        if latest[list(engine.FEATURE_COLUMNS)].isna().any().any():
            raise ValueError("Missing values in model inputs")
        cap = dict(self.bed_capacity)
        cap.update({str(k): v for k, v in (capacity_overrides or {}).items()})
        out = simulator.simulate(latest, {h: self.models[h]["regressor"] for h in HORIZONS},
                                 total_crews, deployments, city, cap, params)
        out["origin_week"] = str(latest["date"].max().date())
        return out

    def available_wards(self) -> list[dict[str, str]]:
        w = self.history[["ward_id", "city"]].drop_duplicates().sort_values(["city", "ward_id"])
        return [{"ward_id": str(r.ward_id), "city": str(r.city)} for r in w.itertuples(index=False)]
