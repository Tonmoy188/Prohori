"""Phase 2 operations engine: allocation, routing, simulator, dispatch and offline behaviour."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import weather  # noqa: E402
from ops import OpsEngine, priority_waterfill  # noqa: E402

ORIGIN = "2025-06-16"


@pytest.fixture(scope="module")
def ops():
    return OpsEngine()


def test_waterfill_respects_stock_and_priority():
    alloc = priority_waterfill([10, 10, 10], [4, 2, 1], 15)
    assert alloc.sum() == 15
    assert (alloc <= 10).all()
    assert alloc[0] >= alloc[1] >= alloc[2] > 0


def test_forecast_shape_and_probabilities(ops):
    for h in config.HORIZONS:
        f = ops.forecast(h, ORIGIN)
        assert len(f) == 60
        assert f.expected_cases.ge(0).all()
        p = f[[f"p_{t}" for t in ("Low", "Medium", "High", "Critical")]].sum(1)
        assert ((p - 1).abs() < 1e-3).all()


@pytest.mark.parametrize("pct", [30, 60, 100])
def test_allocation_never_exceeds_stock_or_demand(ops, pct):
    a = ops.allocate(3, pct, 25, ORIGIN)
    for r, t in a["totals"].items():
        if r == "spraying_crews":
            assert t["assigned"] <= 25
            continue
        assert t["allocated"] <= t["available_stock"] <= t["required"]
    for w in a["wards"]:
        assert 0 <= w["saline_bags_needed_allocated"] <= w["saline_bags_needed"]
    if pct < 100:
        assert a["critical_deficit_warnings"]
    else:
        assert all(w["supply_status"] == "OK" for w in a["wards"])


def test_overflow_routing_stays_within_city(ops):
    r = ops.route_overflow(3, ORIGIN)
    city = {w["ward_id"]: w["city"] for w in r["wards"]}
    assert r["transfers"]
    for t in r["transfers"]:
        assert city[t["from_ward"]] == city[t["to_ward"]] == t["city"]
    for w in r["wards"]:
        assert w["beds_occupied"] <= w["bed_capacity"]
        assert w["net_deficit"] >= 0


def test_simulator_reduces_cases_and_ranks_roi(ops):
    s = ops.simulate({"DSCC_Ward_04": 3, "CCC_Ward_01": 1}, ORIGIN)
    assert all(v["cases_prevented"] > 0 for v in s["totals_by_horizon"].values())
    rois = [r["roi_cases_per_crew"] for r in s["roi_ranking"]]
    assert rois == sorted(rois, reverse=True)


def test_api_core_endpoints_work_offline(monkeypatch):
    from fastapi.testclient import TestClient
    import main

    def down(*_a, **_k):
        raise weather.WeatherUnavailable("network disconnected (test)")
    monkeypatch.setattr(weather, "get_payload", down)
    monkeypatch.setattr(main.bot, "client", None)
    c = TestClient(main.app)
    q = f"origin_week={ORIGIN}&horizon_weeks=3"
    assert c.get(f"/ops/allocation?{q}&stock_pct=50&crews=10").status_code == 200
    assert c.get(f"/ops/overflow?{q}").status_code == 200
    assert c.post("/ops/simulate", json={"deployments": {"DNCC_Ward_01": 2}, "origin_week": ORIGIN}).status_code == 200
    for fmt in ("json", "sms", "whatsapp", "html", "pdf"):
        assert c.get(f"/ops/dispatch?{q}&format={fmt}&city=Gazipur").status_code == 200
    assert c.get("/predict?ward_id=DNCC_Ward_01&horizon_weeks=3").status_code == 200
    reply = c.post("/chat", json={"message": "Risk in DNCC_Ward_01?"}).json()
    assert reply["source"] == "offline_template"
