# PROHORI | DengueWatch — Phase 2 Onsite Final

Operational dengue command system for ward commissioners and health-center managers across **60 urban wards** (Dhaka North, Dhaka South, Chattogram, Gazipur, Narayanganj). It extends the Phase 1 early-warning repo (same repository, continuous history, commits tagged `version-onsite-final`) from *predicting* cases to *acting* on them: supply allocation, hospital overflow routing, vector-control simulation and dispatch notices.

- **Live URL:** `<ADD DEPLOYED URL>`
- **Walkthrough video (7 min, uncut):** `<ADD LINK>`
- **Final commit hash (must match video):** `<ADD HASH>`
- Design details, formulas and assumptions: [ARCHITECTURE.md](ARCHITECTURE.md)

## Phase 2 requirements → where they are implemented

| # | Requirement | Implementation | Endpoint / UI page |
|---|---|---|---|
| 1 | Constrained Resource Allocation | `operations.allocate_resources` | `POST /operations/allocate` · **Allocate** |
| 2 | Hospital Surge & Cross-Ward Overflow Routing | `operations.route_hospital_overflow` | `POST /operations/overflow` · **Overflow** |
| 3 | "What-If" Vector Control Simulator (2–4 wk, ROI) | `simulator.simulate` (re-runs the trained regressors) | `POST /operations/simulate` · **Simulate** |
| 4 | Multi-channel Dispatch (SMS / WhatsApp / printable HTML) | `operations.build_*` | `POST /operations/dispatch`, `/dispatch/html`, `/dispatch/export` · **Dispatch** |
| 5 | Offline Resilience & Deterministic Fallback | local models, `weather.offline_context`, template chat | `POST /weather/offline/{on\|off}` · **Weather** |

### 1. Constrained allocation
- Predicted cases (from the H2/H3/H4 models) become exact needs using dataset-verified per-case ratios: saline **2.378**, diagnostic kits **1.147**, beds **0.289** per case (derived from the 82,572 cases in `onsite_dengue_clean.csv`; half-up rounding).
- **Scarcity slider:** Available Regional Stock 30%–100% (values outside are rejected). Total Spraying Crews is an integer input.
- **Prioritisation:** risk tier (Critical > High > Medium > Low) → `population_density_sqkm` → expected cases → ward ID (stable tie-break). Stock is filled in that order, so dense high-risk wards are served first.
- **Deficit flags:** every ward reports required / allocated / deficit / fulfilment % per resource; any ward below 100% is flagged as a critical supply deficit.
- **Crews:** assigned round-robin in priority order, capped per tier (Critical 3, High 2, Medium 1, Low 0 — an operational policy, not dataset-derived).

### 2. Overflow routing
- Wards whose projected `hospital_beds_required` exceeds local capacity send patients to lower-burden wards **in the same city only** (strict city pools). Output: net ward deficit, beds sent/received, remaining spare beds, per-ward and city-wide bed utilisation.
- **Assumption (stated openly):** the dataset has no bed-capacity column, so capacity = 90th percentile of each ward's historical weekly beds required (min 1). Officers can override any ward's capacity in the UI/API.

### 3. Vector-control simulator
- Officers deploy a limited crew pool manually or use auto mode (greedy: each crew goes where the marginal cases prevented is highest).
- **Model-driven counterfactual:** crew effect (diminishing returns, capped) is applied to the transmission and breeding features of each ward, then the *same trained regressors* are re-run for weeks 2, 3 and 4 (effect ramps 50% / 75% / 100%). Cases prevented, beds, saline and kits prevented are reported per week and per ward.
- **Ranking:** wards ranked by ROI = cases prevented per crew; the best first-crew candidates are also listed.
- Effect parameters (`effect_per_crew=0.15`, `max_effect=0.60`, `env_share=0.5`) are explicit assumptions — the dataset contains no intervention outcomes — and can be overridden per request.

### 4. Dispatch
Per-ward SMS (short) and WhatsApp (full) notices plus a printable commissioner brief (HTML; browser *Print → Save as PDF*). Each lists predicted cases, allocated vs required supplies, deficit alerts, assigned spraying crews and transfer notes. Samples are generated into `/exports` (`sms_*.txt`, `whatsapp_*.txt`, `commissioner_brief_*.html`).

### 5. Offline resilience
Forecasting, allocation, routing, simulation and dispatch templates are pure local Python on bundled models/data — **no network call is on their path**. Weather is optional context: Open-Meteo when online, otherwise stale cache, otherwise a deterministic fallback from the stored dataset (`status: offline_fallback`). `POST /weather/offline/on` (or `DENGUEWATCH_OFFLINE=1`) is the live kill-switch. The Gemini chatbot is optional; without a key or when offline, `/chat` returns a deterministic template answer (`source: deterministic_template`).

## Models
Per horizon (2/3/4 weeks): a ward-case **regressor** and a risk-tier **classifier** in `models/v2_onsite/`. Selected on validation macro-F1: H2 XGBoost **0.626**, H3 HistGradientBoosting **0.607**, H4 XGBoost **0.562** (`classifier_manifest.json`). Tiers: Low / Medium / High / Critical.

## Run locally
Python 3.11+ (scikit-learn is pinned to 1.6.1 — the models were pickled with it).

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # optional: add GEMINI_API_KEY (never commit .env)
python -m uvicorn main:app --port 8000                # API docs: http://127.0.0.1:8000/docs

# second terminal
cd frontend && pip install -r requirements.txt && streamlit run app.py   # http://localhost:8501
```
Set `DENGUEWATCH_API_URL` if the API is not on `http://127.0.0.1:8000`. Tests: `pip install pytest httpx && python -m pytest tests`.

### Offline kill-switch demo
```bash
curl -X POST localhost:8000/weather/offline/on        # or disconnect Wi-Fi
curl -X POST localhost:8000/operations/allocate -H "Content-Type: application/json" \
  -d '{"horizon_weeks":2,"available_regional_stock_pct":50,"total_spraying_crews":10}'
```

## Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/stats`, `/wards`, `/cities` | Status and metadata |
| GET | `/predict`, `/predict/all`, `/outlook` | 2/3/4-week ward forecasts and risk tiers |
| POST | `/operations/allocate` | Scarcity-constrained supplies + crews |
| POST | `/operations/overflow` | Same-city bed overflow routing |
| POST | `/operations/simulate` | Crew what-if with ROI ranking |
| POST | `/operations/dispatch` · `/dispatch/html` · `/dispatch/export` | Notices, printable brief, files in `/exports` |
| GET / POST | `/weather` · `/weather/offline/{on\|off}` | Weather context + offline switch |
| POST | `/chat` | Optional assistant (Gemini, else deterministic template) |

## Limitations
1. Bed capacity is an estimate (no capacity column in the data); override with real figures where available.
2. Simulator effect sizes are assumptions, not fitted to intervention data; ROI ranking is relative, not a measured outcome.
3. Resource ratios are dataset-wide averages applied to predicted cases.
4. Outputs are model estimates, not official surveillance; they support, not replace, public-health judgement.

## AI disclosure
- **Claude (Anthropic):** development assistant for code review, debugging, tests and documentation (including this README and ARCHITECTURE.md). All allocation, routing and simulation logic was reviewed by the team.
- **Google Gemini API (`gemini-2.5-flash`):** optional runtime chatbot only. Never used for predictions, allocation, routing, simulation or dispatch math.
- **Open-Meteo:** optional weather data, no key.
- TODO (team): list any other AI tool used, with its purpose.

## SDG alignment
SDG 3 (target 3.3 neglected tropical diseases, 3.d early warning) and SDG 11 (urban resilience): ward-level early warning turned into concrete supply, bed and spraying-crew decisions.
