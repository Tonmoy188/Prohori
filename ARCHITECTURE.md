# Architecture

**Goal:** estimate weekly dengue cases and a risk tier (Low/Medium/High/Critical) per ward, 2, 3 and 4 weeks ahead, from weather lags and ward vulnerability.

## Data flow

```text
data/dengue_history.csv (2016-2025, 60 wards)          Open-Meteo Archive + Forecast (2 batched calls, cached 3 h)
        |                                                         |
        |                                      daily -> Monday-start weekly rows (current week completed by forecast days)
        |                                                         |
        +--------------------+------------------------------------+
                             v
        engine.build_features  (grouped by ward_id; causal, uses weeks <= t only)
          rain_lag1-8, rain_sum2/3/4/6/8w, rain_sum_lag2to4 (breeding window), temp/humidity lags + rolling means,
          season sin/cos, ward_vulnerability = waterlogging x density x (1 - sanitation)
          + rain/temp/humidity_fwd1 = real Open-Meteo forecast for next week (t+1)
                             v
   models/dengue_h{2,3,4}w.joblib  (loaded once at startup)
     = HistGradientBoostingClassifier (tier probs)  +  Poisson HistGradientBoostingRegressor -> NegBin tier probs
     final probs = 0.5 * classifier + 0.5 * count model;  tier = argmax
                             v
   FastAPI: /predict /outlook /predict/all /weather /metrics  -->  dashboard
                             +--> /chat (optional Gemini 2.5 Flash, grounded on a compact JSON context)
```

Training: `scripts/train_models.py` -> joblib bundles + `training_report.json`. Inference never retrains.

## Chronological validation split

Target = `dengue_cases` at week t+h (h = 2, 3, 4), shifted per ward. Train on origin weeks before 2024-01-01 minus h weeks (embargo, so no training target falls in the test period); test on 2024-01 to 2025-12. No random shuffling. `training_report.json` (served at `GET /metrics`) holds, per horizon: tier accuracy, macro-F1, per-tier recall, MAE, RMSE, Brier score, mean predicted vs actual cases (calibration bias), a train-mean baseline and the same metrics for the model without forecast features. Final models are refit on all labelled rows after validation.

## How future weather enters the model

Dengue at t+h follows rain 2-4 weeks earlier, so the target week needs no forecast; the 16-day provider forecast covers only about weeks t+1 and t+2 and never the 3-4 week target. The model therefore takes the real forecast for **week t+1** (`*_fwd1`). The dataset has no archived forecasts, so training uses week t+1's observed weather plus simulated forecast error (rain x(1+N(0,0.3)), temp N(0,1.5), humidity N(0,5)). Holdout gain vs the no-forecast model: MAE 1.059 -> 1.042 (3 wk), 1.184 -> 1.121 (4 wk), none at 2 wk. Gains depend on the assumed error level and should be re-checked once real forecast snapshots are collected. Every `/predict` response includes `forecast_coverage` stating the forecast end date, the target week and whether the target week is covered (it is not for 3-4 weeks).

## Model selection rationale

- Gradient boosting handles non-linear, interacting rain x vulnerability effects on ~31k tabular rows, trains in seconds on a CPU and needs no GPU; a deep model was not justified at this data size.
- Case counts are mostly zero and over-dispersed (variance 21.2 vs mean 2.6), so a Poisson regressor with Negative Binomial tier probabilities is blended 50/50 with a direct classifier, which keeps tier probabilities consistent with expected cases.
- Direct multi-horizon models (one per h) avoid compounding error from recursive forecasting and need no future weather.
- Drivers: counterfactual scoring per ward (zero the last 9 weeks of rain / replace temperature or humidity with climatology) -> change in expected cases.
- Limits: no archived forecasts (proxy above); forecast days beyond t+1 are shown as context only; live wards in a city share city-level weather; the holdout shows mild over-prediction (mean predicted 2.55 vs actual 2.32 cases, 3 wk) and weak Medium/Critical recall (~0.4-0.5).

## Compute and API cost controls

| Control | Where |
|---|---|
| Serialized models loaded once; inference scores only the latest 12 weeks per ward (~0.15 s for all 60 wards, was ~1.5 s) | `service.py`, `engine.py` |
| 5 cities fetched in 1 forecast + 1 archive call (batched coordinates); cached in memory + `cache/weather.json` for `WEATHER_TTL_S` (3 h) -> at most 16 weather calls/day | `weather.py` |
| After a failed refresh, serve stale cache and back off `WEATHER_RETRY_S` (5 min) instead of retrying per request | `weather.py` |
| Predictions cached per (ward, horizon) and invalidated only when new weather arrives | `service.py` |
| Numeric prediction never calls an LLM; chat is optional | `chat.py` |
| LLM: Gemini 2.5 Flash, `max_output_tokens=300`, temperature 0.2, 300-char input cap, compact context (7-day forecast, no raw history; ~1.6k chars per ward question), 5-min response cache, 10 requests/min per session, 20 s timeout | `chat.py`, `config.py` |
## Phase 2 operations engine (fully offline)

```text
data/onsite_dengue_clean.csv  (+ data/case_reports/reports.csv uploaded via POST /ops/case-reports)
        v
features_v2.build_features   101 columns: case lags 0-52, case rolling mean/std/max (3/4/8/12 w), weather lags 0-8,
                             rolling weather, rain sums, season sin/cos, ward vulnerability interactions
        v
models/classifiers/final_classifier_h{2,3,4}.joblib     XGB (H2, H4) / HGB (H3)  -> tier probabilities (0..3 -> Low..Critical)
models/v2_onsite/regressors/final_regressor_h{2,3,4}.joblib  HGB (H2) / ExtraTrees (H3, H4) -> expected cases
        v
ops.OpsEngine
  forecast      expected cases, tier, supply need = cases x per-case ratio (least squares on onsite data)
  allocate      priority water-filling of stock_pct x total need; weight = tier rank x density x cases; CRITICAL_DEFICIT flags
  assign_crews  greedy by marginal cases prevented x priority
  route_overflow  beds over capacity moved to wards with headroom in the same city only
  simulate / roi_ranking  ML counterfactual over H2-H4: case and breeding features reduced by crew effect, regressors re-run
        v
dispatch.py  SMS (<=160 chars) / WhatsApp / HTML / PDF  ->  /ops/* endpoints, Operations dashboard page, exports/
```

- **Why water-filling:** it never allocates more than a ward needs or more than the regional stock, and gives every ward a share proportional to its priority weight, so high-risk dense wards fill first without starving the rest.
- **Why counterfactual re-scoring:** the simulator uses the same trained models as the forecast, so crew impact is consistent with the forecast instead of a separate hand-made curve. Effect sizes per crew are configurable (`LARVICIDE_EFFECT_PER_CREW`, `FOGGING_EFFECT_PER_CREW`) and should be calibrated with field data.
- **Offline guarantees:** Phase 2 modules make no network calls. `service.py` falls back to the newest local week when Open-Meteo and the disk cache are unavailable, and `chat.py` returns a grounded template reply when Gemini is not configured or unreachable. `tests/test_ops.py` checks this with the weather provider disabled.
- **Validation:** the rebuilt features reproduce the saved pipelines' inputs (max abs difference ~4e-7); regressor holdout MAE 1.07 / 1.18 / 1.41 (H2 / H3 / H4).
