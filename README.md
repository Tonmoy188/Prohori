# Strata | DengueWatch: Bangladesh Dengue Early-Warning System

**Team Strata**, EDU AI Engineering Hackathon 2026 (Preliminary Round, challenge: AI for Urban Dengue Outbreak).

DengueWatch provides public health officers with a **2-, 3- and 4-week early warning** of dengue risk across 60 wards in Dhaka North, Dhaka South, Chattogram, Gazipur and Narayanganj. It combines live weather data (rainfall, temperature and humidity) with ward-level vulnerability indicators to predict dengue risk tiers (**Low / Medium / High / Critical**), expected cases and the main factors influencing each prediction.

**Phase 2 (onsite):** the same backend now turns forecasts into an operational response plan: constrained allocation of saline bags, diagnostic kits and hospital beds under 30-100% stock, same-city hospital overflow routing, a larvicide/fogging crew simulator with cases-prevented-per-crew ROI, and SMS / WhatsApp / HTML / PDF dispatch notices. **All Phase 2 features run fully offline** (no Wi-Fi, no Open-Meteo, no LLM API). See [Phase 2 Operations](#phase-2-operations-offline).

All predictions are **model estimates**, not confirmed dengue cases. For detailed system design, see [ARCHITECTURE.md](ARCHITECTURE.md).


<img width="1600" height="895" alt="WhatsApp Image 2026-11-22 at 1 38 09 PM (3)" src="https://github.com/user-attachments/assets/9a123038-ff2f-4023-884e-d36657a1b7a9" />
<img width="1600" height="766" alt="WhatsApp Image 2026-11-22 at 1 38 09 PM (1)" src="https://github.com/user-attachments/assets/943bec86-4b1b-4787-b21c-1e8e2ffad8ef" />
<img width="1600" height="768" alt="WhatsApp Image 2026-11-22 at 1 38 09 PM" src="https://github.com/user-attachments/assets/d910a7d6-c2b7-4811-883b-38b7ea1a75db" />


## Quick Start

Requires **Python 3.11+**. Internet is optional: live Open-Meteo weather improves the Phase 1 outlook, but every endpoint falls back to local data when offline.

### Windows (Command Prompt or PowerShell)

```bash
# 1. Install (once)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env

# 2. Terminal 1: Start the backend
python -m uvicorn main:app --port 8000

# 3. Terminal 2: Activate the same virtual environment
cd frontend
python -m streamlit run app.py
```

### macOS / Linux

Follow the same steps, using:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then start the backend and dashboard using the commands above.

Open the dashboard at [http://localhost:8501](http://localhost:8501).

Check backend health at [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health). A successful response should include `"status": "ok"`.

The trained models are included in the repository, so **no model training is required to run the application**.

To run tests:

```bash
pip install pytest httpx
python -m pytest tests
```

## How It Meets the Hackathon Requirements

| Requirement | Implementation |
|---|---|
| Weather ingestion and multi-week lags per `ward_id` | `weather.py` (live Open-Meteo integration), `engine.py` (rainfall lags 1–8 weeks, rolling sums, 2–4 week breeding window, temperature and humidity lags, grouped by `ward_id`) |
| Risk prediction with tiers | `models/dengue_h{2,3,4}w.joblib`: trained gradient-boosting classifier and Poisson count model, with Low / Medium / High / Critical tiers |
| Cost and resource efficiency | Serialized models, 3-hour weather cache with back-off, cached predictions and limited, cached chatbot requests |
| Web dashboard with prediction drivers | `frontend/app.py` (Streamlit): ward risk tiles, driver breakdown, weather outlook and model performance |
| Architecture, validation split and model rationale | `ARCHITECTURE.md` |
| Additional features | Model Performance page (MAE/RMSE, chronological validation) and ward-level risk tile grid |

## Phase 2 Operations (offline)

| Phase 2 requirement | Implementation |
|---|---|
| Supply allocation under 30-100% stock, priority for high-risk / high-density wards | `ops.py` `OpsEngine.allocate`: predicted cases x per-case ratios fitted on `data/onsite_dengue_clean.csv` (2.384 saline bags, 1.158 kits, 0.295 beds per case), then priority-weighted water-filling (weight = risk tier x population density x expected cases) |
| Critical deficit warnings | Wards whose allocation falls short of need get `supply_status = CRITICAL_DEFICIT` and appear in `critical_deficit_warnings` |
| Same-city hospital overflow routing | `OpsEngine.route_overflow`: overflow beds go only to wards in the same city with free capacity; returns transfers, ward net deficits and city utilization / headroom |
| Vector-control simulator (2-4 weeks) and ROI | `OpsEngine.simulate` / `roi_ranking`: re-runs the H2/H3/H4 regressors on intervention-adjusted features (recent case and breeding-window features reduced by crew effect); reports cases prevented, beds freed, saline saved and cases prevented per crew |
| Dispatch notices (SMS, WhatsApp, printable brief) | `dispatch.py`: deterministic templates; SMS <= 160 chars; HTML and PDF commissioner briefs (fpdf2) |
| Offline operation | No network call in `features_v2.py`, `ops.py`, `dispatch.py`. Phase 1 endpoints fall back to the local dataset when Open-Meteo and the cache are unavailable; the chatbot falls back to grounded template replies when Gemini is unreachable |
| Sample exports | `exports/` (generated by `python scripts/export_dispatch.py`) |
| Dashboard | **Operations** page in `frontend/app.py`: stock slider, crew count, allocation table, overflow, simulator, ROI ranking and dispatch downloads |

**Phase 2 models** (`models/classifiers/final_classifier_h{2,3,4}.joblib`, `models/v2_onsite/regressors/final_regressor_h{2,3,4}.joblib`) use 101 features built by `features_v2.py` (case lags 0-52, rolling case statistics, weather lags 0-8, rolling weather, seasonality, ward vulnerability interactions). They were saved with **scikit-learn 1.6.1** and the H2/H4 classifiers need **XGBoost**, so both are pinned in `requirements.txt`.

**Forecast origin:** the Phase 2 models need reported case counts, so the newest origin is the last reported week (2025-12-29 in the shipped data). Upload new weekly counts with `POST /ops/case-reports`; they are stored in `data/case_reports/reports.csv` and the engine reloads.

**Bed capacity:** the dataset has no official capacities. Put real numbers in `data/bed_capacity.csv` (`ward_id,bed_capacity`). Without it the fallback is the city's p95 historical weekly bed demand (`BED_CAPACITY_QUANTILE`), split across wards by population density.

**Regenerate sample exports:**

```bash
python scripts/export_dispatch.py --origin 2025-06-16 --horizon 3 --stock 60 --crews 25
```

## How It Works

```text
Dengue history + ward vulnerability + live Open-Meteo weather
        |
        v
Feature engineering
(per-ward weather lags, rolling sums, season, vulnerability)
        |
        v
2-, 3- and 4-week prediction models
(classifier + Poisson count model)
        |
        v
Risk tier + expected cases + prediction drivers
        |
        v
FastAPI backend
        |
        v
Streamlit dashboard + optional Gemini assistant
```

### 1. Data

Historical dengue data comes from `data/dengue_history.csv` (2016–2025, weekly), covering **60 wards across five cities**:

- Dhaka North: 20 wards
- Dhaka South: 20 wards
- Chattogram: 10 wards
- Gazipur: 5 wards
- Narayanganj: 5 wards

Live weather data comes from Open-Meteo, including observed weather and a 16-day forecast.

### 2. Feature Engineering

The system generates:

- Per-ward rainfall lags (1–8 weeks) and rolling sums.
- Temperature and humidity lags.
- Seasonal indicators.
- Ward vulnerability indicators, including waterlogging risk, population density and sanitation.
- Next-week forecast weather features.

### 3. Machine Learning Models

Separate models are used for the 2-, 3- and 4-week prediction horizons.

Models are evaluated using a chronological holdout:

- Training: Data before 2024.
- Testing: 2024–2025.
- No random shuffling is used.

### 4. Prediction Output

Each prediction provides:

- Dengue risk tier.
- Risk probabilities.
- Expected cases (predicted dengue cases for the ward during the target week).
- Main prediction drivers.

Drivers are calculated through sensitivity testing: selected feature groups, such as rainfall, temperature, humidity and forecast weather, are removed to observe changes in expected cases. These indicate what the model relies on, not proven causal relationships.

## Model Validation Results

**Validation period:** 2024–2025 chronological holdout, approximately 6,100 ward-weeks per prediction horizon.

The following results are reported in `training_report.json`.

| Metric | 2 Weeks | 3 Weeks | 4 Weeks |
|---|---:|---:|---:|
| MAE (cases per ward-week) | 0.98 | 1.04 | 1.12 |
| RMSE (cases per ward-week) | 1.94 | 2.05 | 2.19 |
| Baseline MAE (training mean) | 3.15 | 3.16 | 3.17 |
| Tier Accuracy | 0.83 | 0.82 | 0.81 |
| Tier Macro F1 | 0.64 | 0.62 | 0.60 |
| Recall: High | 0.66 | 0.62 | 0.58 |
| Recall: Critical | 0.45 | 0.48 | 0.46 |
| Recall: Low | 0.97 | 0.97 | 0.97 |

Medium and Critical risk tiers are the most challenging to identify, with Critical recall around 0.45–0.48.

During the dry season (November–March), historical data contains very few dengue cases, averaging under 0.1 cases per ward-week. Consequently, Low is the expected prediction for most wards during this period.

### How Forecast Weather Is Validated

The historical dataset does not contain archived weather forecasts.

To approximate forecast conditions during training, next week's observed weather is used with simulated forecast error. This introduces an approximation of forecasting uncertainty, but it cannot fully reproduce the errors of real historical weather forecasts.

Therefore, actual performance during live deployment may differ from the reported validation results. Historical backtesting with real forecast snapshots remains a future improvement.

On the same 4-week holdout, the reported MAE is:

- Without the forecast feature: **1.18**
- With the forecast feature: **1.12**

No measurable benefit was observed at the 2-week horizon.

## Project Structure

```text
main.py                 FastAPI application entry point (Phase 1 + /ops/* endpoints)
service.py              Phase 1 backend service (offline fallback to local data)
engine.py               Phase 1 feature engineering
features_v2.py          Phase 2 101-feature builder
ops.py                  Phase 2 allocation, overflow routing, crew simulator, case-report ingestion
dispatch.py             SMS / WhatsApp / HTML / PDF dispatch templates
weather.py              Weather integration
chat.py                 Gemini chatbot integration
config.py               Application configuration

models/                 Trained joblib models (2, 3 and 4 weeks)
data/                   dengue_history.csv and wards.csv

scripts/
    train_models.py     Phase 1 model training and report generation
    export_dispatch.py  Writes sample dispatch exports to exports/

exports/                Sample SMS, WhatsApp, JSON, HTML and PDF dispatch outputs

frontend/
    app.py              Streamlit dashboard
    api_client.py       Backend API client
    styles.css          Dashboard styling

tests/                  Offline tests

docs/
    screenshots/        Dashboard screenshots

ARCHITECTURE.md         Data flow, validation, model rationale and cost controls
```

To retrain the models:

```bash
python scripts/train_models.py
```

Scikit-learn is pinned to version `1.6.1` because the Phase 2 pipelines were serialized with it (newer versions fail to unpickle them).

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health`, `/stats` | Backend status and chatbot configuration |
| GET | `/wards` | Ward and city information |
| GET | `/weather`, `/weather/forecast` | Weather used by the model and provider forecast |
| GET | `/predict?ward_id=CCC_Ward_01&horizon_weeks=3` | Prediction for one ward and horizon |
| GET | `/outlook?ward_id=CCC_Ward_01` | Predictions for all three horizons |
| GET | `/predict/all?horizon_weeks=3&city=Chattogram` | Predictions for all wards in a city |
| GET | `/metrics` | Model validation metrics |
| POST | `/chat` | Gemini assistant grounded in model output; template reply when offline |
| GET | `/ops/meta` | Origin weeks, per-case ratios, simulator assumptions, bed capacities |
| GET | `/ops/forecast?horizon_weeks=3&origin_week=2025-06-16` | Phase 2 forecast for all wards |
| GET | `/ops/allocation?horizon_weeks=3&stock_pct=60&crews=25` | Constrained supply and crew allocation with deficit warnings |
| GET | `/ops/overflow?horizon_weeks=3&capacity_factor=1.0` | Same-city bed overflow routing and city utilization |
| POST | `/ops/simulate` | What-if crew deployment, body `{"deployments": {"DSCC_Ward_04": 3}}` |
| GET | `/ops/roi?crews_per_ward=1` | Wards ranked by cases prevented per crew |
| GET | `/ops/dispatch?format=sms\|whatsapp\|html\|pdf\|json&city=Gazipur` | Dispatch notices |
| POST | `/ops/case-reports` | Append weekly case reports (list of `{date, ward_id, dengue_cases, ...}`) |

Interactive API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Configuration

Optional environment variables can be configured in `.env`.

| Variable | Purpose / Default |
|---|---|
| `GEMINI_API_KEY` | Optional Gemini chatbot API key |
| `WEATHER_TTL_S` | Weather cache duration (10800 seconds) |
| `WEATHER_RETRY_S` | Weather retry interval (300 seconds) |
| `CHAT_MAX_TOKENS` | Chatbot response token limit (300) |
| `CHAT_CACHE_TTL_S` | Chat response cache duration (300 seconds) |
| `CHAT_RATE_PER_MIN` | Chat request limit (10 per minute) |
| `BED_CAPACITY_QUANTILE` | Fallback bed capacity quantile (0.95) |
| `LARVICIDE_EFFECT_PER_CREW` / `FOGGING_EFFECT_PER_CREW` | Simulator crew effect assumptions (0.15 / 0.10) |
| `MAX_CREWS_PER_WARD` / `CASES_PER_CREW` | Crew cap per ward (5) and cases per required crew (4) |
| `DENGUEWATCH_API_URL` | Dashboard backend URL (`http://127.0.0.1:8000`) |
| `DENGUEWATCH_API_TIMEOUT` | Dashboard API timeout |
| `DENGUEWATCH_CHAT_TIMEOUT` | Chatbot request timeout |

The Gemini API key is only required for the optional chatbot. All prediction and dashboard functionality works without it.

## Limitations

1. **Weather forecast horizon:** Open-Meteo provides forecasts up to approximately 16 days. Only next week's forecast weather is available as a model input. Predictions for weeks 3 and 4 rely primarily on historical weather patterns, lagged features and ward vulnerability rather than direct forecasts for those target weeks.

2. **Historical forecast simulation:** Training uses next week's observed weather with simulated forecast error because archived forecasts are unavailable. Reported validation may differ from live performance. The forecast feature's 4-week MAE improvement (1.18 to 1.12) should be re-evaluated using real historical forecast snapshots.

3. **Shared city-level weather:** Wards within the same city share the city's weather data because ward-level coordinates are not available in the dataset.

4. **No live dengue case feed:** The system does not connect to a live official dengue surveillance feed. Its predictions support public health decision-making but do not replace official surveillance or medical assessment.

5. **Allocation assumptions:** per-case supply ratios are fitted on the onsite dataset, bed capacities fall back to a historical estimate unless `data/bed_capacity.csv` is provided, and simulator crew effects are configurable assumptions, not measured field effects.

## Troubleshooting

- **Backend offline:** Start the backend first and check whether port 8000 is available.
- **`streamlit` not recognized:** Activate the virtual environment and run `python -m streamlit run app.py`.
- **Could not import module `main`:** Run Uvicorn from the project root directory containing `main.py`.
- **HTTP 503 – Live weather unavailable:** Open-Meteo may be unreachable and no cached weather data may be available. Restore internet access and retry.
- **AI Assistant disabled:** Check whether `GEMINI_API_KEY` is configured in `.env`. Predictions work without it.
- **HTTP 429:** The chatbot request limit has been exceeded.
- **Seasonal chart or vulnerability unavailable:** Ensure `frontend/` remains inside the project directory alongside `data/`.
- **Port already in use:** Run Streamlit on another port:

```bash
python -m streamlit run app.py --server.port 8502
```

## AI Disclosure and External Services

### AI Assistants Used by Team Strata

| AI Assistant | How It Was Used |
|---|---|
| **Google Gemini** | Runtime: The dashboard's optional AI Assistant uses the Gemini API (`gemini-2.5-flash`) to explain model outputs. It does not generate predictions or determine risk tiers. Also used for development assistance. |
| **ChatGPT (OpenAI)** | Development assistance, including ideas, code suggestions and debugging. |
| **Claude (Anthropic)** | Code review, debugging, testing and documentation assistance. |
| **Devin (Cognition AI)** | Phase 2 backend: feature rebuild for the onsite models, allocation / overflow / simulator engine, dispatch templates, offline fallbacks, tests and documentation. |

All AI-generated suggestions were reviewed and tested by the team. Prediction values, allocations, routing and dispatch notices are generated by the trained models in `models/` and deterministic code, not by any LLM. No LLM is needed at runtime.

### External APIs and Data Sources

| Service | Purpose | API Key Required |
|---|---|---|
| Open-Meteo Forecast API | Current and forecast weather (up to 16 days) | No |
| Open-Meteo Historical Archive API | Historical and recent observed weather | No |
| Google Gemini API | Optional chatbot functionality | Yes |
| Hackathon dataset (`data/`) | Model training and ward attributes | No |

**Main libraries:** scikit-learn, XGBoost, pandas, NumPy, SciPy, joblib, FastAPI, Uvicorn, Requests, fpdf2, Streamlit, Plotly and google-genai.

SMS and WhatsApp notices are exported as text; sending them would need a provider account (e.g. Twilio or WhatsApp Business API), which is not configured.

## SDG Alignment (Not a Scored Criterion)

- **SDG 3 – Good Health and Well-being:** Supports target 3.3 (neglected tropical diseases) and target 3.d (early warning and risk reduction) through 2–4 week ward-level dengue risk forecasts that can help public health teams prepare preventive interventions.

- **SDG 11 – Sustainable Cities and Communities:** Supports urban resilience through ward-level vulnerability indicators, including waterlogging, population density and sanitation, helping identify neighbourhoods that may require greater attention.

**Scope:** DengueWatch provides predictive estimates and decision-support information. It does not replace official surveillance, guarantee outbreaks or independently allocate public health resources.
