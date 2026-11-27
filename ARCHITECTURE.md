# Architecture — PROHORI | DengueWatch (Phase 2 Onsite Final)

**Goal:** turn 2/3/4-week ward-level dengue forecasts into operational decisions (supplies, beds, spraying crews, notices) that run **100% locally**.

## System overview
```text
data/onsite_dengue_clean.csv (60 wards, weekly)        models/v2_onsite/{regressors,classifiers}/ (H2,H3,H4)
              |                                                     |
              +--> engine.build_features (causal, weeks <= t) ------+
                                   v
                  service.py  (loads models once; cached per-horizon predictions)
                    expected_cases, risk_tier, population_density_sqkm per ward
        +----------------+---------------------+----------------------+
        v                v                     v                      v
 operations.allocate  operations.route_    simulator.simulate    operations.build_*
 (scarcity + crews)   hospital_overflow    (counterfactual ROI)  (SMS/WhatsApp/HTML)
        +----------------+---------------------+----------------------+
                                   v
                  FastAPI main.py (/operations/*, /predict*, /weather, /chat)
                                   v
                  Streamlit frontend (Allocate · Overflow · Simulate · Dispatch · Weather · Assistant)
```
Optional, never on the core path: Open-Meteo (weather context) and Gemini (chat). `exports/` holds generated sample notices.

## Module responsibilities
| File | Role |
|---|---|
| `engine.py` | Causal weekly feature construction |
| `service.py` | Model loading, multi-horizon prediction, glue to operations/simulator, bed-capacity estimate |
| `operations.py` | Allocation, crew assignment, overflow routing, dispatch templates (pure functions, validated inputs) |
| `simulator.py` | Model-driven vector-control counterfactual + ROI ranking |
| `weather.py` | Optional Open-Meteo context with cache and deterministic offline fallback |
| `chat.py` / `main.py:/chat` | Optional Gemini assistant; template reply when offline/unconfigured |
| `main.py` | FastAPI routes, input validation, error mapping |
| `frontend/` | Streamlit UI calling the API only |

## Models (Track 1, reused for operations)
Per horizon h ∈ {2,3,4}: a regressor (expected weekly cases) and a classifier (Low/Medium/High/Critical), chosen on validation macro-F1: H2 XGBoost 0.626, H3 HistGradientBoosting 0.607, H4 XGBoost 0.562. Direct multi-horizon models avoid compounding recursive error and need no future weather. The XGBoost classifiers are stored as portable JSON boosters and rebuilt on load (`service._load_model`) so pickles stay version-safe.

## Module 1 — Constrained allocation (`allocate_resources`)
1. **Demand:** `need_r = round_half_up(expected_cases × k_r)`; `k` = dataset totals ÷ total cases (82,572): saline 196,333, kits 94,701, beds 23,856 → 2.378 / 1.147 / 0.289.
2. **Supply:** `available_r = min(total_need_r, floor(total_need_r × stock_pct/100))`, `stock_pct ∈ [30,100]` (validated).
3. **Priority order:** risk tier ↓, `population_density_sqkm` ↓, expected cases ↓, ward_id (deterministic).
4. **Fill:** walk wards in priority order, `allocated = min(need, remaining)`; `deficit = need − allocated`; fulfilment %. Any deficit → critical supply warning (ward and aggregate).
5. **Crews:** round-robin in priority order, each ward capped by tier target (Critical 3, High 2, Medium 1, Low 0); unassigned crews reported.
Properties: allocated never exceeds need or pool; allocation sums are exact integers; same inputs give identical output.

## Module 2 — Overflow routing (`route_hospital_overflow`)
- `bed_capacity` per ward = 90th percentile of historical weekly `hospital_beds_required` (floor 1), **overridable**. Rationale: the dataset has no capacity column and median demand is ~0.
- `local_deficit = max(0, beds_required − capacity)`; `spare = max(0, capacity − beds_required)`.
- Deficit wards (largest first) transfer to spare wards **inside the same city pool only** (five pools), lowest-burden receivers first. Outputs: beds sent/received, `net_bed_deficit`, `spare_beds_remaining`, ward and city utilisation %, unplaced overflow when a city pool is exhausted.

## Module 3 — Vector-control simulator (`simulator.simulate`)
Counterfactual re-scoring of the **same trained regressors**:
```text
crew effect   E(c) = min(max_effect, 1 − (1 − effect_per_crew)^c)    # diminishing returns, c ≤ 4/ward
ramp          s_h  = E(c) × {2w: 0.50, 3w: 0.75, 4w: 1.00}            # control builds over weeks
transmission features (case lags 0-4, rolling mean/std/max)  × (1 − s_h)
breeding features     (rain sums, waterlogging×rain)         × (1 − env_share × s_h)
prevented = baseline − counterfactual (≥ 0), summed over weeks 2-4
```
Beds, saline and kits prevented use the per-case ratios above. **ROI = cases prevented ÷ crews** per ward → `roi_ranking`; `best_first_crew_candidates` ranks marginal value of a first crew. Auto mode places each crew greedily where marginal prevented cases are highest; manual mode validates per-ward and total crew limits.
Honest limit: defaults (`effect_per_crew 0.15`, `max_effect 0.60`, `env_share 0.5`) are assumptions, since no intervention outcomes exist in the data; they are exposed and overridable.

## Module 4 — Dispatch (`build_*`)
Deterministic templates per ward: SMS (compact), WhatsApp (full, with transfer note) and an escaped-HTML commissioner brief (print to PDF). Content: predicted cases and tier, allocated vs required for each resource, deficit alerts, assigned crews, overflow transfers. `/operations/dispatch/export` writes `sms_*.txt`, `whatsapp_*.txt`, `commissioner_brief_*.html` to `/exports`.

## Module 5 — Offline resilience
| Capability | Offline behaviour |
|---|---|
| Forecasting | Bundled models + CSV; no network |
| Allocation / routing / simulation / dispatch | Pure local Python (`operations.py`, `simulator.py`) |
| Weather context | live → stale cache → `offline_context` from stored data (never raises) |
| Chat | Gemini only if key set and online; else deterministic template reply |
| Kill-switch | `POST /weather/offline/on`, `DENGUEWATCH_OFFLINE=1`, or real network loss; short weather timeout (4 s) so requests fail fast |
Core math has no dependency on any external API or LLM.

## Validation and reliability
- Inputs validated centrally (`OperationsValidationError` → HTTP 422): finite non-negative numbers, stock 30–100, whole-number crews, five allowed cities, no duplicate wards; negative regressor outputs clamped to 0.
- `tests/` covers model bundle metadata and chat fail-closed behaviour; allocation, routing and simulator invariants are demonstrated live in the walkthrough.
- Secrets live in `.env` (git-ignored); `.env.example` is committed with blank keys.

## Key assumptions and limitations
1. Bed capacity is estimated (P90) unless overridden.
2. Per-case resource ratios are dataset-wide averages.
3. Crew tier targets are operational policy, not data-derived.
4. Simulator effect sizes are assumptions; ROI is a relative ranking tool.
5. Estimates support, not replace, official surveillance and clinical judgement.

## AI disclosure
Claude (Anthropic) assisted with code review, debugging, tests and documentation. Gemini 2.5 Flash is an optional runtime chatbot only and is never used for forecasting, allocation, routing, simulation or dispatch. Team to add any other tools used.
