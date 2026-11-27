"""Historical-data-trained dengue model + real observed/current weather ingestion."""
import threading

import joblib
import pandas as pd

import config
import engine
import weather

NOTE = (
    "Estimated dengue cases for the target week using the saved model and its original feature schema. "
    "Historical dengue/weather data is used for training context; current weather comes from Open-Meteo. "
    "Next-week forecast values are model inputs only when the loaded model was trained with those features. "
    "Later daily forecasts are shown as context, not treated as dengue observations. "
    "No live dengue case feed is connected."
)


class Service:
    def __init__(self):
        # Keep the saved estimators unchanged. Build inference inputs from each bundle's
        # own saved feature list; v2 bundles do not contain the three v3 forecast features.
        model_dir = config.ROOT / "models"
        if not all((model_dir / f"dengue_h{h}w.joblib").is_file() for h in config.HORIZONS):
            model_dir = config.ROOT

        self.models = {}
        self.model_forecast_features = {}
        engine_forecast_cols = list(getattr(engine, "FWD_COLS", []))
        for h in config.HORIZONS:
            path = model_dir / f"dengue_h{h}w.joblib"
            if not path.is_file():
                raise FileNotFoundError(f"Missing trained {h}-week model: {path}")
            bundle = joblib.load(path)
            if bundle.get("horizon_weeks") not in (None, h):
                raise ValueError(f"Wrong model loaded for horizon {h}: {path}")

            version = str(bundle.get("feature_version", ""))
            if not (version.startswith("causal_weekly_weather_v2") or
                    version.startswith("causal_weekly_weather_v3")):
                raise ValueError(
                    f"Unsupported model feature version in {path}: {version or 'missing'}"
                )
            features = bundle.get("features")
            if not isinstance(features, (list, tuple)) or not features:
                raise ValueError(f"Model {path} does not contain a valid saved feature list")
            features = list(features)
            for estimator_name in ("clf", "reg"):
                estimator = bundle.get(estimator_name)
                expected_n = getattr(estimator, "n_features_in_", None)
                if expected_n is not None and int(expected_n) != len(features):
                    raise ValueError(
                        f"{path}: {estimator_name} expects {expected_n} features, "
                        f"but bundle lists {len(features)}"
                    )
            self.models[h] = bundle
            self.model_forecast_features[h] = [c for c in engine_forecast_cols if c in features]

        self.wards = pd.read_csv(config.ROOT / "data" / "wards.csv")
        self.history = pd.read_csv(config.DATASET_PATH, parse_dates=["date"])
        self.history_max = self.history.date.max().normalize()

        # Confirm that this engine can actually construct every feature required by
        # every loaded model. This catches a genuine engine/model mismatch at startup.
        _, available_features = engine.build_features(self.history.sort_values(["ward_id", "date"]).tail(20).copy())
        available = set(available_features)
        for h, bundle in self.models.items():
            missing = [c for c in bundle["features"] if c not in available]
            if missing:
                raise ValueError(
                    f"Feature-engine mismatch for {h}-week model ({bundle.get('feature_version')}): "
                    f"engine cannot build {missing[:8]}"
                )

        self.uses_forecast_features = any(self.model_forecast_features.values())
        self._frames, self._stamp, self._cache = None, None, {}
        self._lock = threading.RLock()

    def ward_ids(self):
        return self.wards.ward_id.tolist()

    def frames(self):
        """Historical dengue/weather rows + live weather rows through current forecast-completed week."""
        try:
            payload, stale = weather.get_payload()
        except weather.WeatherUnavailable as exc:
            with self._lock:
                return self._offline_frames(str(exc))
        with self._lock:
            return self._frames_locked(payload, stale)

    def _offline_frames(self, reason):
        """Deterministic fallback when Open-Meteo and the disk cache are both unavailable: score the
        newest week of the local historical dataset instead of failing."""
        if self._stamp != "offline":
            frames = self.history.copy()
            for col in getattr(engine, "FWD_COLS", []):
                frames[col] = float("nan")
            self._frames, self._stamp, self._cache = frames.sort_values(["ward_id", "date"]).reset_index(drop=True), "offline", {}
        return self._frames, {
            "fetched_at_utc": None, "stale": True, "offline": True,
            "source": f"offline fallback: local historical dataset ({reason})",
            "forecast_days_available": 0,
            "historical_dataset_through": str(self.history_max.date()),
            "live_weather_week": None,
        }

    def _frames_locked(self, payload, stale):
        if payload["fetched_at"] != self._stamp:
            weekly = weather.weekly_by_city(payload)
            parts = [self.history.copy()]
            live_parts = []
            for w in self.wards.itertuples():
                if w.city not in weekly:
                    continue
                city_week = weekly[w.city]
                city_week = city_week[city_week.date > self.history_max].copy()
                if city_week.empty:
                    continue
                city_week["ward_id"] = w.ward_id
                city_week["city"] = w.city
                city_week["population_density_sqkm"] = w.population_density_sqkm
                city_week["water_logging_risk_index"] = w.water_logging_risk_index
                city_week["sanitation_score"] = w.sanitation_score
                city_week["dengue_cases"] = float("nan")  # unknown in real time; never fabricate case reports
                live_parts.append(city_week)
            if not live_parts:
                raise weather.WeatherUnavailable("No complete live weather week available for model input")
            parts.append(pd.concat(live_parts, ignore_index=True))
            combined = pd.concat(parts, ignore_index=True, sort=False)
            combined["date"] = pd.to_datetime(combined.date)
            combined = combined.sort_values(["ward_id", "date"]).drop_duplicates(["ward_id", "date"], keep="last")
            # Forecast features are populated only if at least one loaded model
            # was actually trained with them. v2 models remain strictly historical.
            fwd_cols = list(getattr(engine, "FWD_COLS", []))
            for col in fwd_cols:
                combined[col] = float("nan")
            if self.uses_forecast_features and fwd_cols:
                nxt = weather.next_week_forecast(payload)
                live_mask = combined.dengue_cases.isna()
                latest_live = combined.loc[live_mask].groupby("ward_id").date.transform("max")
                newest = live_mask & combined.date.eq(latest_live.reindex(combined.index))
                for col in fwd_cols:
                    if any(col in cols for cols in self.model_forecast_features.values()):
                        combined.loc[newest, col] = combined.loc[newest, "city"].map(
                            lambda c: nxt.get(c, {}).get(col, float("nan")))
            # Each ward needs its complete historical sequence and current weather row.
            self._frames = combined.reset_index(drop=True)
            self._stamp, self._cache = payload["fetched_at"], {}
        meta = {
            "fetched_at_utc": pd.Timestamp(self._stamp, unit="s", tz="UTC").isoformat(),
            "stale": stale,
            "source": payload.get("source", "Open-Meteo"),
            "forecast_days_available": payload.get("forecast_days_available", 16),
            "historical_dataset_through": str(self.history_max.date()),
            "live_weather_week": str(self._frames[self._frames.dengue_cases.isna()].date.max().date()),
            "offline": False,
        }
        return self._frames, meta

    def _weather_context(self, city, h=None):
        """Return current live weather and the provider's actual forward forecast for a city.

        The ML model uses the forecast-completed current week plus historical lag context.
        Future daily forecasts are exposed to the UI/chatbot, but are not falsely injected
        into a model trained without historical forecast snapshots.
        """
        try:
            payload, stale = weather.get_payload()
        except weather.WeatherUnavailable:
            return self._offline_weather_context(city)
        weekly = weather.weekly_by_city(payload).get(city)
        if weekly is None or weekly.empty:
            raise weather.WeatherUnavailable(f"No complete current weather week for {city}")

        current = weekly.sort_values("date").iloc[-1].to_dict()
        current["date"] = pd.Timestamp(current["date"]).strftime("%Y-%m-%d")
        current["forecast_days"] = int(current.get("forecast_days", 0))

        fetched = pd.Timestamp(float(payload["fetched_at"]), unit="s", tz="UTC").isoformat()
        forecast = weather.forecast_daily(payload, city)
        return {
            "city": city,
            "provider": payload.get("source", "Open-Meteo"),
            "fetched_at_utc": fetched,
            "stale_cache": bool(stale),
            "forecast_days_available": int(payload.get("forecast_days_available", 16)),
            "current_week_weather": current,
            "daily_forecast": forecast,
            "model_weather_input": (
                ("The loaded model uses historical weather plus the next-week Open-Meteo forecast "
                 "for its saved forecast features. Future daily forecast values are displayed separately."
                 if (self.model_forecast_features.get(h, []) if h is not None else self.uses_forecast_features) else
                 "The loaded model uses historical/current-week weather features only. "
                 "Future daily forecasts are displayed separately and are not model inputs.")
            ),
        }

    def _offline_weather_context(self, city):
        last = self.history[self.history.city == city].sort_values("date").tail(1)
        if last.empty:
            raise KeyError(city)
        cur = last[["date", "rainfall_mm", "avg_temp_c", "max_temp_c", "min_temp_c", "relative_humidity_pct"]].iloc[0].to_dict()
        cur["date"] = pd.Timestamp(cur["date"]).strftime("%Y-%m-%d")
        cur["forecast_days"] = 0
        return {"city": city, "provider": "offline: local historical dataset", "fetched_at_utc": None,
                "stale_cache": True, "forecast_days_available": 0, "current_week_weather": cur,
                "daily_forecast": [], "model_weather_input": "Offline: newest week of the local historical dataset."}

    def _all(self, h):
        f, meta = self.frames()
        with self._lock:
            if ("all", h) not in self._cache:
                # Only the newest 12 weeks per ward are needed (max lag 8 + rolling 8); identical output, ~10x faster.
                recent = f.sort_values(["ward_id", "date"]).groupby("ward_id").tail(12)
                scored = engine.score_rows(recent, self.models[h])
                latest = scored.sort_values("date").groupby("ward_id", as_index=False).tail(1)
                self._cache[("all", h)] = latest.set_index("ward_id")
            return self._cache[("all", h)], meta

    @staticmethod
    def _row(r, ward_id):
        return {
            "ward_id": ward_id, "city": r.city, "risk_tier": r.risk_tier,
            "expected_cases": round(max(0.0, float(r.expected_cases)), 2),
            "probabilities": {t: round(float(r[f"p_{t}"]), 4) for t in engine.TIERS},
        }

    def predict_all(self, h, city=None):
        scored, meta = self._all(h)
        rows = [self._row(r, ward_id) for ward_id, r in scored.iterrows() if city is None or r.city == city]
        if not rows:
            raise KeyError(city or "No wards available")
        origin = scored.date.max()
        cities = sorted({r["city"] for r in rows})
        weather_contexts = {name: self._weather_context(name, h) for name in cities}
        return {"horizon_weeks": h, "origin_week": str(origin.date()),
                "forecast_week": str((origin + pd.Timedelta(weeks=h)).date()),
                "weather": meta, "weather_contexts": weather_contexts,
                "count": len(rows), "wards": rows}

    def drivers(self, ward_id, h):
        f, _ = self.frames()
        fw = f[f.ward_id == ward_id].sort_values("date").tail(12).copy()  # 12 weeks cover all lag/rolling windows
        ward = self.wards.set_index("ward_id").loc[ward_id]
        base = engine.score_rows(fw, self.models[h]).iloc[-1].expected_cases
        n = len(fw)
        scenarios = {
            "Rainfall (last 9 weeks)": fw.assign(rainfall_mm=[0.0 if i >= n - 9 else v for i, v in enumerate(fw.rainfall_mm)]),
            "Temperature": fw.assign(avg_temp_c=ward.clim_avg_temp_c, max_temp_c=ward.clim_max_temp_c, min_temp_c=ward.clim_min_temp_c),
            "Humidity": fw.assign(relative_humidity_pct=ward.clim_humidity_pct),
        }
        forecast_cols = self.model_forecast_features.get(h, [])
        if forecast_cols:
            scenarios["Next-week forecast"] = fw.assign(**{c: float("nan") for c in forecast_cols})
        return [{"driver": name, "change_in_expected_cases": round(float(base - engine.score_rows(data, self.models[h]).iloc[-1].expected_cases), 3)}
                for name, data in scenarios.items()]

    def _coverage(self, city, origin, h, used):
        """State exactly what part of the prediction window real forecast weather covers."""
        try:
            payload, _ = weather.get_payload()
            daily = weather.forecast_daily(payload, city)
        except weather.WeatherUnavailable:
            daily = []
        last = max((row.get("date") for row in daily if row.get("date")), default=None)
        target_start = origin + pd.Timedelta(weeks=h)
        target_end = target_start + pd.Timedelta(days=6)
        covered = bool(last is not None and pd.Timestamp(last) >= target_end)
        return {
            "weather_forecast_through": str(last) if last else None,
            "target_week": f"{target_start.date()} to {target_end.date()}",
            "target_week_covered_by_weather_forecast": covered,
            "forecast_week_used_by_model": (
                f"{(origin + pd.Timedelta(weeks=1)).date()} to {(origin + pd.Timedelta(days=13)).date()}"
                if used else None
            ),
            "forecast_features_used": used,
            "statement": (
                "This model uses historical/current-week weather and the real next-week forecast as trained features."
                if used else
                "This model was trained without next-week forecast features, so the prediction uses historical/current-week weather only."
            )}

    def predict_ward(self, ward_id, h):
        if ward_id not in set(self.wards.ward_id):
            raise KeyError(ward_id)
        scored, meta = self._all(h)
        key = ("ward", ward_id, h)
        with self._lock:
            if key not in self._cache:
                fw = self._frames[self._frames.ward_id == ward_id].sort_values("date").tail(8)
                self._cache[key] = (
                    self.drivers(ward_id, h),
                    fw[["date", "rainfall_mm", "avg_temp_c", "relative_humidity_pct"]]
                    .assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d")).round(1).to_dict("records"),
                )
            drivers, recent = self._cache[key]
        r = scored.loc[ward_id]
        return {**self._row(r, ward_id), "horizon_weeks": h,
                "origin_week": str(r.date.date()), "forecast_week": str(r.forecast_week.date()),
                "drivers": drivers, "recent_weather": recent, "weather": meta,
                "weather_context": self._weather_context(r.city, h), "note": NOTE,
                "forecast_coverage": self._coverage(
                    r.city, r.date, h,
                    bool(self.model_forecast_features.get(h)) and
                    bool(self._frames[self._frames.ward_id == ward_id].sort_values("date").tail(1)[self.model_forecast_features[h]].notna().all(axis=None))
                )}
