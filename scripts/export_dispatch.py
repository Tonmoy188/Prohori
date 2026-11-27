"""Write sample dispatch exports (SMS, WhatsApp, HTML + PDF commissioner briefs) to exports/. Fully offline.

Run from the project root:
    python scripts/export_dispatch.py --origin 2025-06-16 --horizon 3 --stock 60 --crews 25
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import dispatch  # noqa: E402
from ops import OpsEngine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="2025-06-16", help="forecast origin week (default: pre-monsoon-peak 2025)")
    ap.add_argument("--horizon", type=int, default=3, choices=config.HORIZONS)
    ap.add_argument("--stock", type=float, default=60, help="available regional stock, 30-100 %%")
    ap.add_argument("--crews", type=int, default=25)
    ap.add_argument("--out", default=str(config.EXPORTS_DIR))
    a = ap.parse_args()

    ops = OpsEngine()
    alloc = ops.allocate(a.horizon, a.stock, a.crews, a.origin)
    routing = ops.route_overflow(a.horizon, a.origin)
    notices = dispatch.build_notices(alloc, routing)
    out = Path(a.out)
    (out / "briefs").mkdir(parents=True, exist_ok=True)
    (out / "sms_alerts.txt").write_text("\n".join(dispatch.sms(n) for n in notices) + "\n", encoding="utf-8")
    (out / "whatsapp_alerts.txt").write_text("\n\n".join(dispatch.whatsapp(n) for n in notices) + "\n", encoding="utf-8")
    (out / "dispatch_notices.json").write_text(json.dumps(
        {"settings": vars(a) | {"out": None}, "totals": alloc["totals"], "cities": routing["cities"],
         "transfers": routing["transfers"], "notices": notices}, indent=2), encoding="utf-8")
    title = "Commissioner Dispatch Brief - All 60 wards"
    (out / "briefs" / "all_wards.html").write_text(dispatch.html_brief(notices, title, routing["cities"]), encoding="utf-8")
    (out / "briefs" / "all_wards.pdf").write_bytes(dispatch.pdf_brief(notices, title, routing["cities"]))
    for city in sorted({n["city"] for n in notices}):
        sub = [n for n in notices if n["city"] == city]
        slug = city.lower().replace(" ", "_")
        t = f"Commissioner Dispatch Brief - {city}"
        cs = {city: routing["cities"][city]}
        (out / "briefs" / f"{slug}.html").write_text(dispatch.html_brief(sub, t, cs), encoding="utf-8")
        (out / "briefs" / f"{slug}.pdf").write_bytes(dispatch.pdf_brief(sub, t, cs))
    print(f"Wrote {len(notices)} ward notices to {out}")


if __name__ == "__main__":
    main()
