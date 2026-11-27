"""Deterministic, template-based dispatch notices (SMS, WhatsApp, HTML and PDF briefs). No LLM or network."""
from __future__ import annotations

import html
from datetime import datetime, timezone

RES_LABEL = {"saline_bags_needed": "Saline bags", "diagnostic_kits_needed": "Diagnostic kits",
             "hospital_beds_required": "Hospital beds"}


def build_notices(alloc: dict, routing: dict) -> list[dict]:
    """Merge allocation and overflow-routing output into one structured notice per ward."""
    route = {w["ward_id"]: w for w in routing["wards"]}
    moves = {}
    for t in routing["transfers"]:
        moves.setdefault(t["from_ward"], []).append(f"{t['patients']} -> {t['to_ward']}")
        moves.setdefault(t["to_ward"], []).append(f"receive {t['patients']} from {t['from_ward']}")
    notices = []
    for w in alloc["wards"]:
        r = route.get(w["ward_id"], {})
        notices.append({
            "ward_id": w["ward_id"], "city": w["city"], "risk_tier": w["risk_tier"],
            "origin_week": alloc["origin_week"], "target_week": alloc["target_week"],
            "horizon_weeks": alloc["horizon_weeks"], "predicted_cases": round(w["expected_cases"], 1),
            "supplies": {k: {"required": int(w[k]), "allocated": int(w[f"{k}_allocated"]),
                             "deficit": int(w[f"{k}_deficit"])} for k in RES_LABEL},
            "crews_assigned": int(w["crews_assigned"]), "crews_required": int(w["crews_required"]),
            "supply_status": w["supply_status"],
            "beds": {"capacity": r.get("bed_capacity"), "net_deficit": r.get("net_deficit", 0),
                     "utilization": r.get("utilization")},
            "transfers": moves.get(w["ward_id"], []),
            "stock_pct": alloc["stock_pct"],
        })
    return notices


def sms(n: dict) -> str:
    s = n["supplies"]
    alert = " DEFICIT!" if n["supply_status"] != "OK" else ""
    text = (f"PROHORI {n['ward_id']} {n['risk_tier'].upper()} wk {n['target_week']}: ~{n['predicted_cases']:.0f} cases."
            f" Saline {s['saline_bags_needed']['allocated']}/{s['saline_bags_needed']['required']},"
            f" Kits {s['diagnostic_kits_needed']['allocated']}/{s['diagnostic_kits_needed']['required']},"
            f" Beds {s['hospital_beds_required']['allocated']}/{s['hospital_beds_required']['required']},"
            f" Crews {n['crews_assigned']}.{alert}")
    return text[:160]


def whatsapp(n: dict) -> str:
    lines = [f"*PROHORI Dengue Dispatch - {n['ward_id']} ({n['city']})*",
             f"Risk: *{n['risk_tier']}* | Target week {n['target_week']} ({n['horizon_weeks']}-week forecast)",
             f"Predicted cases: *{n['predicted_cases']}* (model estimate)", "", "_Supplies (allocated / required)_"]
    for k, lbl in RES_LABEL.items():
        v = n["supplies"][k]
        lines.append(f"- {lbl}: {v['allocated']} / {v['required']}" + (f"  *SHORT {v['deficit']}*" if v["deficit"] else ""))
    lines.append(f"- Spraying crews assigned: {n['crews_assigned']} (needed {n['crews_required']})")
    if n["transfers"]:
        lines.append("- Patient routing: " + "; ".join(n["transfers"]))
    if n["beds"]["net_deficit"]:
        lines.append(f"- *Unrouted bed deficit: {n['beds']['net_deficit']}* - escalate to city health office")
    if n["supply_status"] != "OK":
        lines.append(f"*ALERT: {n['supply_status'].replace('_', ' ')}* (regional stock {n['stock_pct']}%)")
    return "\n".join(lines)


_CSS = """body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#1b1b1b}h1{font-size:20px;margin:0}
.sub{color:#555;font-size:12px;margin-bottom:12px}table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}
th,td{border:1px solid #bbb;padding:4px 6px;text-align:left}th{background:#eef2f5}.Critical{color:#b00020;font-weight:bold}
.High{color:#d35400;font-weight:bold}.Medium{color:#b7950b}.Low{color:#1e8449}.bad{background:#fdecea}
.box{border:1px solid #ccc;padding:8px;margin:8px 0;font-size:12px}@media print{body{margin:8mm}.noprint{display:none}}"""


def html_brief(notices: list[dict], title: str, city_summary: dict | None = None) -> str:
    e = html.escape
    first = notices[0] if notices else {}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = []
    for n in notices:
        s = n["supplies"]
        cls = "bad" if n["supply_status"] != "OK" else ""
        rows.append(
            f"<tr class='{cls}'><td>{e(n['ward_id'])}</td><td class='{e(n['risk_tier'])}'>{e(n['risk_tier'])}</td>"
            f"<td>{n['predicted_cases']}</td>"
            + "".join(f"<td>{s[k]['allocated']} / {s[k]['required']}</td>" for k in RES_LABEL)
            + f"<td>{n['crews_assigned']} / {n['crews_required']}</td><td>{n['beds']['net_deficit'] or 0}</td>"
            f"<td>{e('; '.join(n['transfers']) or '-')}</td><td>{e(n['supply_status'].replace('_', ' '))}</td></tr>")
    city = ""
    if city_summary:
        city = ("<div class='box'><b>City bed pool:</b> " + e(", ".join(
            f"{c}: {v['beds_occupied']}/{v['bed_capacity']} beds ({(v['utilization'] or 0) * 100:.0f}%), "
            f"unrouted deficit {v['unrouted_deficit']}" for c, v in city_summary.items())) + "</div>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{e(title)}</title><style>{_CSS}</style></head><body>
<h1>{e(title)}</h1><div class="sub">Forecast origin {e(str(first.get('origin_week','')))} | target week {e(str(first.get('target_week','')))}
({first.get('horizon_weeks','')}-week horizon) | regional stock {first.get('stock_pct','')}% | generated {stamp} | offline template</div>
{city}<table><tr><th>Ward</th><th>Risk</th><th>Pred. cases</th><th>Saline alloc/req</th><th>Kits alloc/req</th>
<th>Beds alloc/req</th><th>Crews assigned/needed</th><th>Unrouted bed deficit</th><th>Patient routing</th><th>Status</th></tr>
{''.join(rows)}</table><div class="sub">Predicted cases are model estimates, not confirmed reports. Supplies are allocated by
priority-weighted water-filling (risk tier x population density). Print this page (Ctrl+P) for a paper brief.</div></body></html>"""


def pdf_brief(notices: list[dict], title: str, city_summary: dict | None = None) -> bytes:
    from fpdf import FPDF

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(True, 10)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=8)
    if notices:
        n0 = notices[0]
        pdf.cell(0, 5, f"Origin {n0['origin_week']} | target week {n0['target_week']} ({n0['horizon_weeks']}-week) | "
                       f"regional stock {n0['stock_pct']}% | offline template", new_x="LMARGIN", new_y="NEXT")
    if city_summary:
        for c, v in city_summary.items():
            pdf.cell(0, 5, f"{c}: {v['beds_occupied']}/{v['bed_capacity']} beds, unrouted deficit {v['unrouted_deficit']}",
                     new_x="LMARGIN", new_y="NEXT")
    head = ["Ward", "Risk", "Cases", "Saline a/r", "Kits a/r", "Beds a/r", "Crews", "Bed deficit", "Routing", "Status"]
    widths = [32, 18, 14, 22, 20, 20, 16, 20, 80, 35]
    pdf.set_font("Helvetica", "B", 8)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", size=7)
    for n in notices:
        s = n["supplies"]
        vals = [n["ward_id"], n["risk_tier"], f"{n['predicted_cases']}",
                *[f"{s[k]['allocated']}/{s[k]['required']}" for k in RES_LABEL],
                f"{n['crews_assigned']}/{n['crews_required']}", str(n["beds"]["net_deficit"] or 0),
                ("; ".join(n["transfers"]) or "-")[:60], n["supply_status"].replace("_", " ")]
        for v, w in zip(vals, widths):
            pdf.cell(w, 5, v, border=1)
        pdf.ln()
    return bytes(pdf.output())
