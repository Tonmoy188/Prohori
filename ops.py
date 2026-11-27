"""Phase 2 operations engine: forecasts -> constrained supply allocation, crew assignment, hospital overflow
routing and the vector-control what-if simulator. Runs fully offline (local CSVs + local joblib models)."""
from __future__ import annotations

import json
import threading
import warnings

import joblib
import numpy as np
import pandas as pd

import config
import features_v2

TIERS = ["Low", "Medium", "High", "Critical"]
TIER_WEIGHT = {"Low": 1.0, "Medium": 2.0, "High": 3.0, "Critical": 4.0}
RESOURCES = ["saline_bags_needed", "diagnostic_kits_needed", "hospital_beds_required"]
FOG_COLS_PREFIX = "cases_"
LARVICIDE_COLS = ["water_logging_risk_index", "ward_vulnerability", "wl_x_sani_gap", "wl_x_rain4w",
                  "vul_x_rain8w", "rain_sum_3to6w"]


def _round_preserving(values, total, caps):
    """Integer allocation that sums to `total` (largest remainder) without exceeding per-ward caps."""
    base = np.minimum(np.floor(values), caps).astype(int)
    left = int(total - base.sum())
    if left > 0:
        frac = np.where(base < caps, values - np.floor(values), -1.0)
        for i in np.argsort(-frac, kind="stable"):
            if left == 0:
                break
            if base[i] < caps[i]:
                base[i] += 1
                left -= 1
    return base


def priority_waterfill(demand, weight, stock):
    """Weighted water-filling: fill_i = min(1, lam * w_i). Higher-priority wards get a higher fill rate,
    every ward with demand receives something, and the total never exceeds stock."""
    demand = np.asarray(demand, float)
    weight = np.asarray(weight, float)
    stock = float(min(stock, demand.sum()))
    if stock <= 0 or demand.sum() <= 0:
        return np.zeros(len(demand), int)
    lo, hi = 0.0, 1.0 / max(weight[demand > 0].min(), 1e-9)
    for _ in range(80):
        lam = (lo + hi) / 2
        if (demand * np.minimum(1.0, lam * weight)).sum() > stock:
            hi = lam
        else:
            lo = lam
    alloc = demand * np.minimum(1.0, hi * weight)
    alloc *= stock / alloc.sum()
    return _round_preserving(np.minimum(alloc, demand), int(round(stock)), demand.astype(int))


class OpsEngine:
    def __init__(self):
        manifest = json.loads((config.ROOT / "models" / "web_asset_manifest.json").read_text(encoding="utf-8"))
        self.manifest = manifest
        self.classifiers, self.regressors = {}, {}
        for h in config.HORIZONS:
            key = f"H{h}"
            clf_path = config.ROOT / "models" / manifest["classification_models"][key]
            reg_path = config.ROOT / "models" / manifest["regression_models"][key]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                self.classifiers[h] = joblib.load(clf_path)
                self.regressors[h] = joblib.load(reg_path)
            for name, model in (("classifier", self.classifiers[h]), ("regressor", self.regressors[h])):
                if not hasattr(model, "feature_names_in_"):
                    raise ValueError(f"{key} {name} has no saved feature names")
        self.features = list(self.regressors[config.HORIZONS[0]].feature_names_in_)
        self.wards = pd.read_csv(config.ROOT / "data" / "wards.csv")
        self._lock = threading.RLock()
        self.reload()

    # ------------------------------------------------------------------ data
    def reload(self):
        base = pd.read_csv(config.ONSITE_DATA_PATH, parse_dates=["date"])
        reports = config.CASE_REPORTS_PATH
        if reports.is_file():
            extra = pd.read_csv(reports, parse_dates=["date"])
            base = pd.concat([base, extra], ignore_index=True, sort=False)
            base = base.sort_values(["ward_id", "date"]).drop_duplicates(["ward_id", "date"], keep="last")
        with self._lock:
            self.data = base.reset_index(drop=True)
            self.ratios = self._fit_ratios(self.data)
            self.capacity = self._bed_capacity(self.data)
            self._cache = {}

    @staticmethod
    def _fit_ratios(d):
        """Supplies per case, fitted from the onsite dataset (least squares through the origin)."""
        x = d.dengue_cases.astype(float)
        return {r: round(float((x * d[r]).sum() / (x * x).sum()), 4) for r in RESOURCES if r in d}

    def _bed_capacity(self, d):
        """Per-ward bed capacity. Uses data/bed_capacity.csv when supplied; otherwise each city's pool is its
        historical p{q} weekly bed demand, split across wards by population density (catchment proxy)."""
        path = config.BED_CAPACITY_PATH
        if path.is_file():
            cap = pd.read_csv(path).set_index("ward_id").bed_capacity.astype(int)
            return cap.reindex(self.wards.ward_id).fillna(1).astype(int)
        weekly = d.groupby(["city", "date"]).hospital_beds_required.sum()
        pool = weekly.groupby("city").quantile(config.BED_CAPACITY_QUANTILE)
        w = self.wards.set_index("ward_id")
        share = w.population_density_sqkm / w.groupby("city").population_density_sqkm.transform("sum")
        return np.ceil(share * w.city.map(pool)).clip(lower=1).astype(int)

    def origins(self):
        dates = self.data.date.drop_duplicates().sort_values()
        return {"first": str(dates.iloc[min(52, len(dates) - 1)].date()), "latest": str(dates.iloc[-1].date())}

    def _origin(self, origin):
        latest = self.data.date.max()
        if origin is None:
            return latest
        o = pd.Timestamp(origin)
        o = o - pd.Timedelta(days=o.dayofweek)
        if o > latest or o < self.data.date.min():
            raise ValueError(f"origin_week must be between {self.data.date.min().date()} and {latest.date()}")
        return o

    def _origin_rows(self, origin):
        key = ("rows", origin)
        if key not in self._cache:
            rows = features_v2.latest_rows(self.data[self.data.date <= origin])
            if len(rows) != len(self.wards):
                raise ValueError(f"Only {len(rows)} wards have data at origin {origin.date()}")
            self._cache[key] = rows
        return self._cache[key]

    # ------------------------------------------------------------- forecast
    def _proba(self, h, X):
        clf = self.classifiers[h]
        P = clf.predict_proba(X[list(clf.feature_names_in_)])
        classes = list(clf.classes_)
        # XGB pipelines were trained on ordinal codes 0..3 = Low..Critical; HGB keeps the string labels.
        names = [TIERS[int(c)] if not isinstance(c, str) else c for c in classes]
        return pd.DataFrame(P, columns=names)[TIERS].to_numpy()

    def _cases(self, h, X):
        return np.clip(self.regressors[h].predict(X[self.features]), 0, None)

    def forecast(self, h, origin=None):
        with self._lock:
            o = self._origin(origin)
            key = ("fc", h, o)
            if key not in self._cache:
                rows = self._origin_rows(o)
                P = self._proba(h, rows)
                cases = self._cases(h, rows)
                out = rows[["ward_id", "city", "population_density_sqkm"]].copy()
                out["origin_week"] = str(o.date())
                out["target_week"] = str((o + pd.Timedelta(weeks=h)).date())
                out["expected_cases"] = cases.round(2)
                out["risk_tier"] = [TIERS[i] for i in P.argmax(1)]
                for i, t in enumerate(TIERS):
                    out[f"p_{t}"] = P[:, i].round(4)
                actual = self.data[self.data.date == o + pd.Timedelta(weeks=h)].set_index("ward_id").dengue_cases
                out["actual_cases"] = out.ward_id.map(actual)  # only known when replaying a past origin
                for r, k in self.ratios.items():
                    out[r] = np.rint(cases * k).astype(int)
                dens = out.population_density_sqkm / out.population_density_sqkm.max()
                out["priority_score"] = (out.risk_tier.map(TIER_WEIGHT) * (0.5 + 0.5 * dens)
                                         * (1 + out.expected_cases / max(out.expected_cases.max(), 1e-9))).round(4)
                self._cache[key] = out
            return self._cache[key].copy()

    # ----------------------------------------------------------- allocation
    def allocate(self, h, stock_pct=100, crews=20, origin=None, scope="region"):
        if not 30 <= stock_pct <= 100:
            raise ValueError("stock_pct must be between 30 and 100")
        f = self.forecast(h, origin)
        groups = [f.index] if scope == "region" else [g.index for _, g in f.groupby("city")]
        stock = {}
        for r in RESOURCES:
            f[f"{r}_allocated"] = 0
            stock[r] = 0
            for idx in groups:
                s = int(np.floor(f.loc[idx, r].sum() * stock_pct / 100))
                stock[r] += s
                f.loc[idx, f"{r}_allocated"] = priority_waterfill(f.loc[idx, r], f.loc[idx, "priority_score"], s)
            f[f"{r}_deficit"] = f[r] - f[f"{r}_allocated"]
        crew_plan = self.assign_crews(h, crews, origin, forecast=f)
        f["crews_required"] = f.ward_id.map(crew_plan["required"]).fillna(0).astype(int)
        f["crews_assigned"] = f.ward_id.map(crew_plan["assigned"]).fillna(0).astype(int)
        f["fill_rate"] = np.where(f[RESOURCES].sum(1) > 0,
                                  f[[f"{r}_allocated" for r in RESOURCES]].sum(1) / f[RESOURCES].sum(1).clip(lower=1), 1.0).round(3)

        def flag(r):
            short = any(r[f"{x}_deficit"] > 0 for x in RESOURCES)
            if short and (r.risk_tier in ("High", "Critical") or r.fill_rate < 0.5):
                return "CRITICAL_DEFICIT"
            return "DEFICIT" if short else "OK"
        f["supply_status"] = f.apply(flag, axis=1)
        f = f.sort_values("priority_score", ascending=False).reset_index(drop=True)
        totals = {r: {"required": int(f[r].sum()), "available_stock": stock[r], "allocated": int(f[f"{r}_allocated"].sum()),
                      "deficit": int(f[f"{r}_deficit"].sum())} for r in RESOURCES}
        totals["spraying_crews"] = {"required": int(f.crews_required.sum()), "available": crews,
                                    "assigned": int(f.crews_assigned.sum()),
                                    "unassigned": int(crews - f.crews_assigned.sum())}
        warnings_ = [f"{r.ward_id} ({r.city}, {r.risk_tier}): short " + ", ".join(
            f"{int(r[f'{x}_deficit'])} {x.split('_')[0]}" for x in RESOURCES if r[f"{x}_deficit"] > 0)
            for _, r in f.iterrows() if r.supply_status == "CRITICAL_DEFICIT"]
        return {"horizon_weeks": h, "origin_week": f.origin_week.iloc[0], "target_week": f.target_week.iloc[0],
                "stock_pct": stock_pct, "scope": scope, "per_case_ratios": self.ratios, "totals": totals,
                "critical_deficit_warnings": warnings_, "wards": _records(f)}

    # -------------------------------------------------------------- crews
    def _counterfactual(self, rows, h, k):
        """Re-score origin rows as if k crews worked the ward this week. Larviciding removes breeding sites
        (scales water-logging/breeding-rain features); fogging cuts adult-vector transmission (scales recent
        case-burden features). Per-crew effects are config assumptions with diminishing returns."""
        lar = (1 - config.LARVICIDE_EFFECT_PER_CREW) ** k
        fog = (1 - config.FOGGING_EFFECT_PER_CREW) ** k
        x = rows.copy()
        for c in self.features:
            if c.startswith(FOG_COLS_PREFIX) and "trend" not in c:
                x[c] = x[c] * fog
            elif c in LARVICIDE_COLS:
                x[c] = x[c] * lar
        return self._cases(h, x)

    def prevented_table(self, h, origin=None, max_crews=None):
        """cases prevented per ward for 0..K crews at horizon h; clipped >= 0 and monotone in crews."""
        K = max_crews or config.MAX_CREWS_PER_WARD
        o = self._origin(origin)
        key = ("prev", h, o, K)
        with self._lock:
            if key not in self._cache:
                rows = self._origin_rows(o)
                base = self._cases(h, rows)
                prev = np.zeros((len(rows), K + 1))
                for k in range(1, K + 1):
                    prev[:, k] = np.clip(base - self._counterfactual(rows, h, k), 0, None)
                prev = np.maximum.accumulate(prev, axis=1)
                self._cache[key] = (rows.ward_id.tolist(), base, prev)
            return self._cache[key]

    def assign_crews(self, h, crews, origin=None, forecast=None):
        """Greedy marginal allocation: each crew goes to the ward with the largest priority-weighted gain in
        cases prevented (summed over 2-4 week horizons). Concave gains make greedy optimal."""
        f = forecast if forecast is not None else self.forecast(h, origin)
        required = {r.ward_id: int(min(config.MAX_CREWS_PER_WARD, np.ceil(r.expected_cases / config.CASES_PER_CREW)))
                    if r.risk_tier != "Low" or r.expected_cases >= config.CASES_PER_CREW else 0 for r in f.itertuples()}
        ids, _, _ = self.prevented_table(h, origin)
        gain = sum(self.prevented_table(hh, origin)[2] for hh in config.HORIZONS)
        weight = f.set_index("ward_id").priority_score.reindex(ids).to_numpy()
        assigned = np.zeros(len(ids), int)
        cap = np.array([required[w] for w in ids])
        for _ in range(int(crews)):
            marginal = np.array([(gain[i, assigned[i] + 1] - gain[i, assigned[i]]) * weight[i]
                                 if assigned[i] < cap[i] else -1 for i in range(len(ids))])
            i = int(marginal.argmax())
            if marginal[i] < 0:
                break
            assigned[i] += 1
        return {"required": required, "assigned": dict(zip(ids, assigned.tolist()))}

    # ----------------------------------------------------------- simulator
    def simulate(self, deployments: dict, origin=None):
        """What-if: deploy crews to selected wards; report 2/3/4-week reductions in cases, beds and saline."""
        unknown = set(deployments) - set(self.wards.ward_id)
        if unknown:
            raise KeyError(", ".join(sorted(unknown)))
        out, totals = [], {}
        for h in config.HORIZONS:
            ids, base, prev = self.prevented_table(h, origin)
            idx = {w: i for i, w in enumerate(ids)}
            tot = {"cases_prevented": 0.0, "beds_freed": 0.0, "saline_saved": 0.0}
            for w, k in deployments.items():
                k = int(min(max(k, 0), config.MAX_CREWS_PER_WARD))
                i = idx[w]
                p = float(prev[i, k])
                row = {"ward_id": w, "horizon_weeks": h, "crews": k, "baseline_cases": round(float(base[i]), 2),
                       "simulated_cases": round(float(base[i]) - p, 2), "cases_prevented": round(p, 2),
                       "beds_freed": round(p * self.ratios["hospital_beds_required"], 2),
                       "saline_saved": round(p * self.ratios["saline_bags_needed"], 2),
                       "roi_cases_per_crew": round(p / k, 3) if k else 0.0}
                out.append(row)
                for t in tot:
                    tot[t] += row[t]
            totals[str(h)] = {t: round(v, 2) for t, v in tot.items()}
        df = pd.DataFrame(out)
        ranking = []
        if not df.empty:
            r = df.groupby("ward_id").agg(crews=("crews", "first"), cases_prevented_2to4w=("cases_prevented", "sum"),
                                          beds_freed_2to4w=("beds_freed", "sum"), saline_saved_2to4w=("saline_saved", "sum"))
            r["roi_cases_per_crew"] = np.where(r.crews > 0, r.cases_prevented_2to4w / r.crews.clip(lower=1), 0)
            ranking = _records(r.sort_values("roi_cases_per_crew", ascending=False).reset_index().round(3))
        return {"origin_week": str(self._origin(origin).date()), "assumptions": self.assumptions(),
                "totals_by_horizon": totals, "by_ward_and_horizon": out, "roi_ranking": ranking}

    def roi_ranking(self, crews_per_ward=1, origin=None, top=60):
        """Rank every ward by cases prevented per crew if it received `crews_per_ward` crews."""
        sim = self.simulate({w: crews_per_ward for w in self.wards.ward_id}, origin)
        return {**sim, "roi_ranking": sim["roi_ranking"][:top]}

    @staticmethod
    def assumptions():
        return {"larvicide_effect_per_crew": config.LARVICIDE_EFFECT_PER_CREW,
                "fogging_effect_per_crew": config.FOGGING_EFFECT_PER_CREW,
                "max_crews_per_ward": config.MAX_CREWS_PER_WARD, "cases_per_crew": config.CASES_PER_CREW,
                "method": "ML counterfactual: re-run the H2/H3/H4 regressors on intervention-adjusted features"}

    # -------------------------------------------------------------- routing
    def route_overflow(self, h, origin=None, capacity_factor=1.0):
        f = self.forecast(h, origin)
        f["bed_capacity"] = np.floor(f.ward_id.map(self.capacity) * capacity_factor).astype(int)
        f["bed_demand"] = f.hospital_beds_required
        f["local_admitted"] = np.minimum(f.bed_demand, f.bed_capacity)
        f["overflow"] = f.bed_demand - f.local_admitted
        f["headroom"] = f.bed_capacity - f.local_admitted
        f["received"] = 0
        f["transferred_out"] = 0
        transfers, cities = [], {}
        for city, g in f.groupby("city"):
            head = g.headroom.to_dict()
            for i in g[g.overflow > 0].sort_values(["overflow", "priority_score"], ascending=False).index:
                need = int(f.at[i, "overflow"])
                # lower-burden receivers first: lowest utilisation, then lowest risk tier
                for j in sorted((j for j in g.index if head[j] > 0 and j != i),
                                key=lambda j: (f.at[j, "local_admitted"] / max(f.at[j, "bed_capacity"], 1),
                                               TIER_WEIGHT[f.at[j, "risk_tier"]], -head[j])):
                    if need == 0:
                        break
                    n = min(need, head[j])
                    head[j] -= n
                    need -= n
                    f.at[j, "received"] += n
                    f.at[i, "transferred_out"] += n
                    transfers.append({"city": city, "from_ward": f.at[i, "ward_id"], "to_ward": f.at[j, "ward_id"],
                                      "patients": int(n)})
            gi = f.loc[g.index]
            cap, dem = int(gi.bed_capacity.sum()), int(gi.bed_demand.sum())
            unrouted = int((gi.overflow - gi.transferred_out).sum())
            cities[city] = {"bed_capacity": cap, "bed_demand": dem, "beds_occupied": dem - unrouted,
                            "utilization": round((dem - unrouted) / cap, 3) if cap else None,
                            "remaining_headroom": cap - (dem - unrouted), "unrouted_deficit": unrouted,
                            "status": "CITY_OVER_CAPACITY" if unrouted else ("SURGE" if dem > 0.85 * cap else "OK")}
        f["net_deficit"] = f.overflow - f.transferred_out
        f["beds_occupied"] = f.local_admitted + f.received
        f["utilization"] = (f.beds_occupied / f.bed_capacity.clip(lower=1)).round(3)
        cols = ["ward_id", "city", "risk_tier", "expected_cases", "bed_demand", "bed_capacity", "local_admitted",
                "overflow", "transferred_out", "received", "net_deficit", "beds_occupied", "utilization"]
        return {"horizon_weeks": h, "origin_week": f.origin_week.iloc[0], "target_week": f.target_week.iloc[0],
                "capacity_factor": capacity_factor,
                "capacity_source": "data/bed_capacity.csv" if config.BED_CAPACITY_PATH.is_file()
                else (f"city pool = p{int(config.BED_CAPACITY_QUANTILE * 100)} of historical weekly city bed demand, "
                      "split across wards by population density"),
                "cities": cities, "transfers": transfers,
                "wards": _records(f[cols].sort_values(["city", "net_deficit", "overflow"], ascending=[True, False, False]))}

    # ---------------------------------------------------------- case reports
    def add_case_reports(self, rows: list[dict]):
        new = pd.DataFrame(rows)
        new["date"] = pd.to_datetime(new.date)
        new["date"] = new.date - pd.to_timedelta(new.date.dt.dayofweek, unit="D")
        unknown = set(new.ward_id) - set(self.wards.ward_id)
        if unknown:
            raise KeyError(", ".join(sorted(unknown)))
        w = self.wards.set_index("ward_id")
        for c in ["city", *features_v2.STATIC]:
            new[c] = new.ward_id.map(w[c])
        last = self.data.sort_values("date").groupby("ward_id").tail(1).set_index("ward_id")
        for c in features_v2.WEATHER:  # missing weather: carry the ward's last observed week forward
            if c not in new:
                new[c] = np.nan
            new[c] = new[c].fillna(new.ward_id.map(last[c]))
        for r, k in self.ratios.items():
            if r not in new:
                new[r] = np.ceil(new.dengue_cases * k).astype(int)
        path = config.CASE_REPORTS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        old = pd.read_csv(path, parse_dates=["date"]) if path.is_file() else pd.DataFrame()
        allr = pd.concat([old, new[self.data.columns.intersection(new.columns)]], ignore_index=True)
        allr.drop_duplicates(["ward_id", "date"], keep="last").to_csv(path, index=False)
        self.reload()
        return {"saved": len(new), "latest_origin_week": self.origins()["latest"]}


def _records(df):
    return json.loads(df.to_json(orient="records", date_format="iso"))
