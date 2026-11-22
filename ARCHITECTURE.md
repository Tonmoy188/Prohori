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