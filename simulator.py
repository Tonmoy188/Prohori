"""Vector-control what-if simulator (100% local, deterministic).

Counterfactual method (model-driven): for each ward and each horizon h in {2,3,4}
we rebuild the ward's 101-feature row as it would look if larvicide/fogging crews
had cut transmission, then re-run the SAME trained regressor on that row.

  crew effect   E(c)   = E_MAX * (1 - (1 - E_CREW) ** c)      (diminishing returns)
  time ramp     s_h    = E(c) * RAMP[h]                      (effect builds over weeks)
  transmission  features (recent cases, lags 0-4, 3w/4w rolling mean/std/max) x (1 - s_h)
  breeding      features (rain sums, water-logging x rain interactions)       x (1 - ENV_SHARE * s_h)

Cases prevented = baseline - counterfactual (clipped at >= 0, summed over weeks 2-4).
Beds / saline / kits prevented use the dataset-verified per-case ratios.
All parameters are explicit assumptions (not fitted to intervention data, which the
dataset does not contain) and can be overridden per request.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from operations import RESOURCE_RATIOS

HORIZONS = (2, 3, 4)
DEFAULTS = {
    "effect_per_crew": 0.15,   # share of transmission removed by the first crew in a ward
    "max_effect": 0.60,        # control never removes more than this share
    "env_share": 0.5,          # breeding-site features move half as much as case momentum
    "max_crews_per_ward": 4,
}
RAMP = {2: 0.50, 3: 0.75, 4: 1.00}   # larvicide/fogging needs weeks to lower incidence

_CASE_COLS = [f"cases_lag_{l}" for l in (0, 1, 2, 3, 4)] + [
    f"cases_roll_{s}_{w}w" for s in ("mean", "std", "max") for w in (3, 4)]
_ENV_COLS = ["rain_sum_2w", "rain_sum_4w", "rain_sum_8w", "rain_sum_3to6w",
             "wl_x_rain4w", "vul_x_rain8w"]


class SimulationError(ValueError):
    pass


def crew_effect(crews: int, p: Mapping[str, float]) -> float:
    if crews <= 0:
        return 0.0
    return float(min(p["max_effect"], 1 - (1 - p["effect_per_crew"]) ** crews))


def _counterfactual_X(X: pd.DataFrame, s: np.ndarray, env_share: float) -> pd.DataFrame:
    """X: base feature rows (one per ward); s: effect per ward at this horizon."""
    cf = X.copy()
    keep = (1.0 - s)
    for c in _CASE_COLS:
        cf[c] = X[c].to_numpy(float) * keep
    cf["cases_trend_3v8"] = cf["cases_roll_mean_3w"] - X["cases_roll_mean_8w"]
    env_keep = 1.0 - env_share * s
    for c in _ENV_COLS:
        cf[c] = X[c].to_numpy(float) * env_keep
    return cf


def _predict_curves(latest: pd.DataFrame, regressors: Mapping[int, Any], max_c: int,
                    p: Mapping[str, float]) -> np.ndarray:
    """cases[c, h_idx, ward] for crews c = 0..max_c (c=0 is the untouched baseline)."""
    out = np.zeros((max_c + 1, len(HORIZONS), len(latest)))
    for hi, h in enumerate(HORIZONS):
        reg = regressors[h]
        feats = list(reg.feature_names_in_)
        X = latest[feats]
        base = np.maximum(np.asarray(reg.predict(X), float), 0.0)
        out[0, hi] = base
        for c in range(1, max_c + 1):
            s = np.full(len(latest), crew_effect(c, p) * RAMP[h])
            cf = np.maximum(np.asarray(reg.predict(_counterfactual_X(X, s, p["env_share"])),
                                       float), 0.0)
            out[c, hi] = np.minimum(cf, base)   # an intervention can never raise cases
    # enforce monotone non-increasing in crews (tree models can be jagged)
    return np.minimum.accumulate(out, axis=0)


def simulate(latest: pd.DataFrame, regressors: Mapping[int, Any], total_crews: int,
             deployments: Mapping[str, int] | None = None, city: str | None = None,
             bed_capacity: Mapping[str, int] | None = None,
             params: Mapping[str, float] | None = None) -> dict:
    """latest: one feature row per ward (columns ward_id, city, population_density_sqkm + 101 features)."""
    p = {**DEFAULTS, **(params or {})}
    if not 0 < p["effect_per_crew"] < 1 or not 0 < p["max_effect"] <= 1 or not 0 <= p["env_share"] <= 1:
        raise SimulationError("Invalid effect parameters")
    total_crews = int(total_crews)
    if total_crews < 0:
        raise SimulationError("total_crews must be >= 0")

    d = latest.copy()
    d["ward_id"] = d["ward_id"].astype(str)
    if city:
        d = d[d["city"].astype(str).str.casefold() == city.strip().casefold()]
        if d.empty:
            raise KeyError(f"No wards for city: {city}")
    d = d.reset_index(drop=True)
    ids = d["ward_id"].tolist()
    idx = {w: i for i, w in enumerate(ids)}
    cap_max = int(p["max_crews_per_ward"])

    curves = _predict_curves(d, regressors, cap_max, p)        # [c, h, ward]
    totals_by_c = curves.sum(axis=1)                            # [c, ward] cases over weeks 2-4

    crews = np.zeros(len(d), dtype=int)
    mode = "manual"
    if deployments:
        for w, c in deployments.items():
            if str(w) not in idx:
                raise KeyError(f"Unknown ward in deployments: {w}")
            c = int(c)
            if c < 0 or c > cap_max:
                raise SimulationError(f"Crews per ward must be 0..{cap_max} ({w}={c})")
            crews[idx[str(w)]] = c
        if crews.sum() > total_crews:
            raise SimulationError(f"Deployed crews ({int(crews.sum())}) exceed total crews ({total_crews})")
    else:
        mode = "auto_greedy_roi"
        for _ in range(total_crews):               # place each crew where marginal cases prevented is highest
            gain = np.array([totals_by_c[crews[i], i] - totals_by_c[crews[i] + 1, i]
                             if crews[i] < cap_max else -1.0 for i in range(len(d))])
            best = int(np.argmax(gain))
            if gain[best] <= 1e-9:
                break
            crews[best] += 1

    rows = []
    for i, w in enumerate(ids):
        c = int(crews[i])
        base_wk = curves[0, :, i]
        cf_wk = curves[c, :, i]
        prevented = float(base_wk.sum() - cf_wk.sum())
        first = float(totals_by_c[0, i] - totals_by_c[1, i])
        cap = (bed_capacity or {}).get(w)
        rows.append({
            "ward_id": w, "city": str(d.at[i, "city"]),
            "population_density_sqkm": float(d.at[i, "population_density_sqkm"]),
            "crews": c,
            "effect_pct": round(crew_effect(c, p) * 100, 1),
            "baseline_cases_by_week": {f"week_{h}": round(float(base_wk[k]), 2) for k, h in enumerate(HORIZONS)},
            "simulated_cases_by_week": {f"week_{h}": round(float(cf_wk[k]), 2) for k, h in enumerate(HORIZONS)},
            "baseline_cases": round(float(base_wk.sum()), 2),
            "simulated_cases": round(float(cf_wk.sum()), 2),
            "cases_prevented": round(prevented, 2),
            "beds_prevented": round(prevented * RESOURCE_RATIOS["hospital_beds_required"], 2),
            "saline_prevented": round(prevented * RESOURCE_RATIOS["saline_bags_needed"], 1),
            "kits_prevented": round(prevented * RESOURCE_RATIOS["diagnostic_kits_needed"], 1),
            "peak_beds_baseline": round(float(base_wk.max()) * RESOURCE_RATIOS["hospital_beds_required"], 2),
            "peak_beds_simulated": round(float(cf_wk.max()) * RESOURCE_RATIOS["hospital_beds_required"], 2),
            "bed_capacity": cap,
            "roi_cases_per_crew": round(prevented / c, 3) if c else None,
            "first_crew_roi": round(first, 3),   # cases prevented by deploying ONE crew here
        })

    deployed = sorted([r for r in rows if r["crews"] > 0], key=lambda r: (-r["roi_cases_per_crew"], r["ward_id"]))
    for rank, r in enumerate(deployed, 1):
        r["roi_rank"] = rank
    candidates = sorted(rows, key=lambda r: (-r["first_crew_roi"], r["ward_id"]))

    def tot(key):
        return round(sum(r[key] for r in rows), 2)

    weekly = {f"week_{h}": {
        "baseline": round(float(curves[0, k].sum()), 2),
        "simulated": round(float(sum(curves[crews[i], k, i] for i in range(len(d)))), 2)}
        for k, h in enumerate(HORIZONS)}
    return {
        "method": "model-driven counterfactual: trained regressors re-run on crew-adjusted features",
        "parameters": {**p, "ramp_by_horizon": RAMP},
        "mode": mode, "total_crews": total_crews, "crews_deployed": int(crews.sum()),
        "crews_unused": total_crews - int(crews.sum()),
        "summary": {"baseline_cases": tot("baseline_cases"), "simulated_cases": tot("simulated_cases"),
                    "cases_prevented": tot("cases_prevented"), "beds_prevented": tot("beds_prevented"),
                    "saline_prevented": tot("saline_prevented"), "kits_prevented": tot("kits_prevented"),
                    "overall_roi_cases_per_crew": round(tot("cases_prevented") / crews.sum(), 3) if crews.sum() else None},
        "weekly_totals": weekly,
        "roi_ranking": deployed,
        "best_first_crew_candidates": [{k: r[k] for k in ("ward_id", "city", "first_crew_roi", "baseline_cases")}
                                       for r in candidates[:10]],
        "wards": rows,
    }