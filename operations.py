"""DengueWatch Phase 2 operational engine (100% local, deterministic, no network).

Resource allocation under scarcity, spraying-crew assignment, same-city hospital
overflow routing, and SMS / WhatsApp / printable-HTML dispatch notices.
"""

from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from math import ceil, floor, isfinite
from typing import Any, Mapping, Sequence

# Verified aggregate ratios from onsite_dengue_clean.csv (82,572 cases).
RESOURCE_RATIOS = {
    "saline_bags_needed": 196333 / 82572,
    "diagnostic_kits_needed": 94701 / 82572,
    "hospital_beds_required": 23856 / 82572,
}

RISK_ORDER = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Unknown": 0}

# Operational policy (not dataset-derived): crews a ward can usefully absorb per tier.
CREW_TARGET_BY_TIER = {"Critical": 3, "High": 2, "Medium": 1, "Low": 0, "Unknown": 0}

ALLOWED_CITIES = {
    "dhaka north": "Dhaka North",
    "dhaka south": "Dhaka South",
    "chattogram": "Chattogram",
    "gazipur": "Gazipur",
    "narayanganj": "Narayanganj",
}


class OperationsValidationError(ValueError):
    pass


def number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OperationsValidationError(f"{field} must be numeric") from exc
    if not (0 <= result < float("inf")):  # also rejects NaN
        raise OperationsValidationError(f"{field} must be finite and non-negative")
    return result


def integer(value: Any, field: str) -> int:
    result = number(value, field)
    if not result.is_integer():
        raise OperationsValidationError(f"{field} must be a whole number")
    return int(result)


def round_half_up(value: float) -> int:
    return floor(max(0.0, value) + 0.5)


def resource_key(resource: str, suffix: str) -> str:
    if resource.endswith("_needed"):
        return resource[:-7] + "_" + suffix
    if resource.endswith("_required"):
        return resource[:-9] + "_" + suffix
    raise OperationsValidationError(f"Invalid resource: {resource}")


def normalize_ward(row: Mapping[str, Any]) -> dict:
    ward = dict(row)
    ward["ward_id"] = str(ward.get("ward_id", "")).strip()
    city = str(ward.get("city", "")).strip()
    if not ward["ward_id"] or not city:
        raise OperationsValidationError("Every ward requires ward_id and city")
    if city.lower() not in ALLOWED_CITIES:
        raise OperationsValidationError(
            f"Ward {ward['ward_id']}: unknown city '{city}'. "
            f"Allowed: {sorted(ALLOWED_CITIES.values())}"
        )
    ward["city"] = ALLOWED_CITIES[city.lower()]

    # Regressors can output small negatives; clamp instead of crashing. NaN/inf still rejected.
    try:
        cases = float(ward.get("expected_cases", 0))
    except (TypeError, ValueError) as exc:
        raise OperationsValidationError("expected_cases must be numeric") from exc
    if not isfinite(cases):
        raise OperationsValidationError("expected_cases must be finite")
    ward["expected_cases"] = max(0.0, cases)

    ward["population_density_sqkm"] = number(
        ward.get("population_density_sqkm", 0), "population_density_sqkm"
    )
    risk = str(ward.get("risk_tier", "Unknown")).strip().title()
    ward["risk_tier"] = risk if risk in RISK_ORDER else "Unknown"
    return ward


def priority_key(ward: Mapping[str, Any]) -> tuple:
    """Higher risk first, then denser wards, then more cases, then ward_id (stable)."""
    return (
        -RISK_ORDER.get(ward["risk_tier"], 0),
        -ward["population_density_sqkm"],
        -ward["expected_cases"],
        ward["ward_id"],
    )


def calculate_resource_demand(expected_cases: float) -> dict[str, int]:
    cases = max(0.0, float(expected_cases))
    return {r: round_half_up(cases * k) for r, k in RESOURCE_RATIOS.items()}


def _validate_wards(forecasts: Sequence[Mapping[str, Any]]) -> list[dict]:
    wards = [normalize_ward(r) for r in forecasts]
    if not wards:
        raise OperationsValidationError("At least one ward forecast is required")
    ids = [w["ward_id"] for w in wards]
    if len(ids) != len(set(ids)):
        raise OperationsValidationError("Duplicate ward IDs are not allowed")
    return wards


def _assign_crews(ordered: list[dict], crews: int) -> int:
    """Round-robin in priority order, capped at each tier's crew target. Returns unassigned."""
    for w in ordered:
        w["crews_assigned"] = 0
        w["crew_target"] = CREW_TARGET_BY_TIER[w["risk_tier"]]
    remaining = crews
    progressed = True
    while remaining > 0 and progressed:
        progressed = False
        for w in ordered:
            if remaining <= 0:
                break
            if w["crews_assigned"] < w["crew_target"]:
                w["crews_assigned"] += 1
                remaining -= 1
                progressed = True
    return remaining


def allocate_resources(
    forecasts: Sequence[Mapping[str, Any]],
    available_regional_stock_pct: float = 100,
    total_spraying_crews: int = 0,
) -> dict:
    """Allocate scarce regional stock and spraying crews by ward priority."""
    stock_pct = number(available_regional_stock_pct, "available_regional_stock_pct")
    if not 30 <= stock_pct <= 100:
        raise OperationsValidationError("Regional stock must be between 30% and 100%")
    crews = integer(total_spraying_crews, "total_spraying_crews")
    wards = _validate_wards(forecasts)

    for w in wards:
        w.update(calculate_resource_demand(w["expected_cases"]))

    ordered = sorted(wards, key=priority_key)
    resource_totals, warnings = {}, []

    for resource in RESOURCE_RATIOS:
        total_required = sum(w[resource] for w in wards)
        # round() guards float error (e.g. 29.999999999 -> 30)
        available = min(total_required, floor(round(total_required * stock_pct / 100, 9)))
        remaining = available
        for w in ordered:
            required = w[resource]
            allocated = min(required, remaining)
            w[resource_key(resource, "allocated")] = allocated
            w[resource_key(resource, "deficit")] = required - allocated
            w[resource_key(resource, "fulfillment_pct")] = (
                round(allocated / required * 100, 2) if required else 100.0
            )
            remaining -= allocated
        allocated_total = available - remaining
        deficit_total = total_required - allocated_total
        resource_totals[resource] = {
            "required": total_required,
            "available": available,
            "allocated": allocated_total,
            "deficit": deficit_total,
        }
        if deficit_total > 0:
            warnings.append({
                "type": "CRITICAL_SUPPLY_DEFICIT",
                "resource": resource,
                "required": total_required,
                "allocated": allocated_total,
                "deficit": deficit_total,
                "fulfillment_pct": round(allocated_total / total_required * 100, 2),
            })

    unassigned = _assign_crews(ordered, crews)

    for w in wards:
        deficits = [r for r in RESOURCE_RATIOS if w[resource_key(r, "deficit")] > 0]
        w["deficit_resources"] = deficits
        # Requirement: flag whenever ward demand cannot be 100% fulfilled.
        w["supply_deficit_warning"] = bool(deficits)
        w["critical_supply_deficit_warning"] = bool(deficits)
        w["high_priority_deficit"] = bool(deficits) and w["risk_tier"] in {"Critical", "High"}

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "available_regional_stock_pct": stock_pct,
        "total_spraying_crews": crews,
        "assigned_spraying_crews": crews - unassigned,
        "unassigned_spraying_crews": unassigned,
        "resource_ratios": dict(RESOURCE_RATIOS),
        "resource_totals": resource_totals,
        "critical_supply_deficit_warnings": warnings,
        "has_critical_supply_deficit": bool(warnings),
        "wards": ordered,
    }


def estimate_bed_capacity(
    history_beds_by_ward: Mapping[str, Sequence[float]],
    percentile: float = 90,
) -> dict[str, int]:
    """Documented assumption: the dataset has no capacity column, so capacity is
    estimated as the given percentile (default 90th) of each ward's historical weekly beds required,
    with a floor of 1 bed (median weekly demand is ~0, so low percentiles give zero capacity).
    Operators can override any value. Deterministic and offline."""
    out = {}
    for ward_id, vals in history_beds_by_ward.items():
        v = sorted(float(x) for x in vals if isfinite(float(x)))
        if not v:
            continue
        idx = (len(v) - 1) * percentile / 100
        lo, hi = floor(idx), ceil(idx)
        out[str(ward_id)] = max(1, round_half_up(v[lo] + (v[hi] - v[lo]) * (idx - lo)))
    return out


def route_hospital_overflow(
    wards: Sequence[Mapping[str, Any]],
    bed_capacity_by_ward: Mapping[str, int],
) -> dict:
    """Route overflow patients to lower-burden wards within the same city only."""
    rows = _validate_wards(wards)
    capacities = {str(k).strip(): v for k, v in bed_capacity_by_ward.items()}
    unknown = set(capacities) - {r["ward_id"] for r in rows}
    if unknown:
        raise OperationsValidationError(f"Unknown capacity ward IDs: {sorted(unknown)}")

    for r in rows:
        r.update(calculate_resource_demand(r["expected_cases"]))
        cap = capacities.get(r["ward_id"])
        r["bed_capacity"] = integer(cap, "bed_capacity") if cap is not None else None
        for k in ("local_bed_deficit", "net_bed_deficit", "spare_beds_initial",
                  "spare_beds_remaining", "bed_utilization_pct"):
            r[k] = None
        r["beds_sent_out"] = 0
        r["beds_received"] = 0
        if r["bed_capacity"] is not None:
            demand = r["hospital_beds_required"]
            r["local_bed_deficit"] = r["net_bed_deficit"] = max(0, demand - r["bed_capacity"])
            r["spare_beds_initial"] = r["spare_beds_remaining"] = max(0, r["bed_capacity"] - demand)
            r["bed_utilization_pct"] = (
                round(demand / r["bed_capacity"] * 100, 2)
                if r["bed_capacity"] else (100.0 if demand else 0.0)
            )

    by_city = defaultdict(list)
    for r in rows:
        by_city[r["city"]].append(r)

    transfers, summaries = [], []
    for city, city_rows in sorted(by_city.items()):
        configured = [r for r in city_rows if r["bed_capacity"] is not None]
        # Lower-burden first: lowest utilization, then most spare beds.
        donors = sorted(
            (r for r in configured if r["spare_beds_remaining"] > 0),
            key=lambda r: (r["bed_utilization_pct"], -r["spare_beds_remaining"], r["ward_id"]),
        )
        recipients = sorted((r for r in configured if r["net_bed_deficit"] > 0), key=priority_key)

        for rec in recipients:
            for don in donors:
                if rec["net_bed_deficit"] <= 0:
                    break
                moved = min(rec["net_bed_deficit"], don["spare_beds_remaining"])
                if moved <= 0:
                    continue
                transfers.append({
                    "city": city,
                    "from_ward_id": rec["ward_id"],
                    "to_ward_id": don["ward_id"],
                    "patients_transferred": moved,
                })
                rec["net_bed_deficit"] -= moved
                rec["beds_sent_out"] += moved
                don["spare_beds_remaining"] -= moved
                don["beds_received"] += moved

        cap = sum(r["bed_capacity"] for r in configured)
        dem = sum(r["hospital_beds_required"] for r in configured)
        summaries.append({
            "city": city,
            "total_bed_demand": dem,
            "total_bed_capacity": cap,
            "city_wide_bed_utilization_pct": round(dem / cap * 100, 2) if cap else None,
            "initial_local_deficit": sum(r["local_bed_deficit"] for r in configured),
            "unmet_deficit_after_routing": sum(r["net_bed_deficit"] for r in configured),
            "remaining_headroom_after_routing": sum(r["spare_beds_remaining"] for r in configured),
            "patients_transferred": sum(t["patients_transferred"] for t in transfers if t["city"] == city),
            "unconfigured_wards": [r["ward_id"] for r in city_rows if r["bed_capacity"] is None],
        })

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "routing_rule": "Same-city transfers only",
        "transfers": transfers,
        "city_summaries": summaries,
        "wards": rows,
    }


def _res_lines(ward: Mapping[str, Any]) -> list[tuple[str, int, int, int]]:
    out = []
    for r in RESOURCE_RATIOS:
        req = integer(ward.get(r, 0), r)
        alloc = integer(ward.get(resource_key(r, "allocated"), 0), r + "_allocated")
        out.append((r, alloc, req, max(0, req - alloc)))
    return out


def _transfer_note(ward_id: str, overflow: Mapping[str, Any] | None) -> str:
    if not overflow:
        return ""
    parts = []
    for t in overflow.get("transfers", []):
        if t["from_ward_id"] == ward_id:
            parts.append(f"send {t['patients_transferred']} pts to Ward {t['to_ward_id']}")
        elif t["to_ward_id"] == ward_id:
            parts.append(f"receive {t['patients_transferred']} pts from Ward {t['from_ward_id']}")
    return "; ".join(parts)


def build_dispatch_message(
    ward: Mapping[str, Any],
    horizon_weeks: int,
    forecast_week: str,
    overflow: Mapping[str, Any] | None = None,
) -> str:
    """Deterministic WhatsApp-length alert (full detail)."""
    lines = _res_lines(ward)
    res = "; ".join(f"{r}: {a}/{q}" for r, a, q, _ in lines)
    deficits = [f"{r} short by {d}" for r, _, _, d in lines if d > 0]
    warning = ("CRITICAL SUPPLY DEFICIT: " + ", ".join(deficits) + ". Replenishment required."
               if deficits else "No supply deficit.")
    transfer = _transfer_note(ward["ward_id"], overflow)
    return (
        f"DengueWatch | {ward['city']} | Ward {ward['ward_id']} | H+{horizon_weeks} | {forecast_week} | "
        f"Predicted cases: {float(ward['expected_cases']):.1f} | Risk: {ward['risk_tier']} | "
        f"Allocated/Required - {res} | Spraying crews: {ward.get('crews_assigned', 0)} | {warning}"
        + (f" | Overflow: {transfer}" if transfer else "")
    )


def build_sms_message(ward: Mapping[str, Any], horizon_weeks: int, forecast_week: str) -> str:
    """Compact alert for plain SMS."""
    lines = _res_lines(ward)
    short = {"saline_bags_needed": "Saline", "diagnostic_kits_needed": "Kits", "hospital_beds_required": "Beds"}
    res = " ".join(f"{short[r]} {a}/{q}" for r, a, q, _ in lines)
    flag = " DEFICIT!" if any(d for *_, d in lines) else ""
    return (f"DengueWatch Ward {ward['ward_id']} H+{horizon_weeks} {ward['risk_tier']} "
            f"~{float(ward['expected_cases']):.0f} cases. {res}. Crews {ward.get('crews_assigned', 0)}.{flag}")


def build_html_brief(allocation: Mapping[str, Any], horizon_weeks: int, forecast_week: str,
                     overflow: Mapping[str, Any] | None = None) -> str:
    """Printable commissioner brief (self-contained HTML; print to PDF from any browser)."""
    rows = []
    for w in allocation.get("wards", []):
        cells = "".join(f"<td>{a}/{q}</td>" for _, a, q, _ in _res_lines(w))
        alert = "DEFICIT" if w.get("supply_deficit_warning") else "OK"
        rows.append(
            f"<tr class='{'bad' if alert != 'OK' else ''}'><td>{escape(w['city'])}</td>"
            f"<td>{escape(w['ward_id'])}</td><td>{escape(w['risk_tier'])}</td>"
            f"<td>{w['expected_cases']:.1f}</td>{cells}<td>{w.get('crews_assigned', 0)}</td>"
            f"<td>{alert}</td><td>{escape(_transfer_note(w['ward_id'], overflow))}</td></tr>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>DengueWatch Brief</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px}table{border-collapse:collapse;width:100%;"
        "font-size:12px}th,td{border:1px solid #999;padding:4px 6px;text-align:left}th{background:#eee}"
        "tr.bad{background:#fde8e8}@media print{body{margin:8mm}}</style></head><body>"
        f"<h1>DengueWatch Commissioner Brief</h1><p>Forecast: {escape(str(forecast_week))} "
        f"(H+{horizon_weeks} weeks) | Regional stock: {allocation.get('available_regional_stock_pct')}% | "
        f"Crews assigned: {allocation.get('assigned_spraying_crews')}/{allocation.get('total_spraying_crews')}</p>"
        "<table><tr><th>City</th><th>Ward</th><th>Risk</th><th>Pred. cases</th><th>Saline (alloc/req)</th>"
        "<th>Kits (alloc/req)</th><th>Beds (alloc/req)</th><th>Crews</th><th>Alert</th><th>Overflow</th></tr>"
        + "".join(rows) + "</table></body></html>"
    )


def build_dispatch_payload(
    allocation: Mapping[str, Any],
    horizon_weeks: int,
    forecast_week: str,
    overflow: Mapping[str, Any] | None = None,
) -> dict:
    if horizon_weeks not in (2, 3, 4):
        raise OperationsValidationError("Horizon must be 2, 3 or 4 weeks")
    messages = [{
        "ward_id": w["ward_id"],
        "city": w["city"],
        "sms": build_sms_message(w, horizon_weeks, forecast_week),
        "whatsapp": build_dispatch_message(w, horizon_weeks, forecast_week, overflow),
    } for w in allocation.get("wards", [])]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "horizon_weeks": horizon_weeks,
        "forecast_week": forecast_week,
        "ward_messages": messages,
        "html_brief": build_html_brief(allocation, horizon_weeks, forecast_week, overflow),
        "allocation": dict(allocation),
        "overflow": dict(overflow or {}),
    }