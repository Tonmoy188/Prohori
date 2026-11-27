"""Prohori (DengueWatch Phase 2) — Streamlit dashboard. Talks only to the FastAPI backend (main.py)."""
import base64
import json
import uuid
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import api_client as api

HERE = Path(__file__).parent
st.set_page_config(page_title="Prohori", page_icon=str(HERE / "assets" / "logo.svg"), layout="wide",
                   initial_sidebar_state="collapsed")
st.logo(str(HERE / "assets" / "wordmark.svg"), size="large")
st.markdown(f"<style>{(HERE / 'styles.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

TIERS = ["Low", "Medium", "High", "Critical"]
TIER_COLORS = {"Low": "#2BAE66", "Medium": "#E8B10C", "High": "#F08A24", "Critical": "#D64545"}
RES = {"saline_bags_needed": ("Saline bags", "saline_bags"),
       "diagnostic_kits_needed": ("Diagnostic kits", "diagnostic_kits"),
       "hospital_beds_required": ("Hospital beds", "hospital_beds")}
HORIZONS = [2, 3, 4]


# ---------------------------------------------------------------- helpers
@st.cache_data
def img(name):
    data = base64.b64encode((HERE / "assets" / f"{name}.svg").read_bytes()).decode()
    return f"data:image/svg+xml;base64,{data}"


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def section(icon, title, text):
    html(f'<div class="sec reveal"><img src="{img(icon)}" alt=""/><div><h2>{title}</h2><p>{text}</p></div></div>')


def label(text):
    html(f'<div class="label reveal">{text}</div>')


def kpi(k, v, s="", tone="#0F9D74", pct=None):
    bar = f'<div class="bar"><i style="width:{max(0, min(100, pct)):.0f}%"></i></div>' if pct is not None else ""
    return f'<div class="kpi reveal" style="--tone:{tone}"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div>{bar}</div>'


def kpis(cards):
    html('<div class="grid">' + "".join(cards) + "</div>")


def tier(t):
    return f'<span class="tier {t}">{t}</span>'


def alert(text, kind="info"):
    html(f'<div class="alert {kind} reveal">{text}</div>')


def fmt(n, d=0):
    return f"{n:,.{d}f}" if isinstance(n, (int, float)) else "–"


def nice(ward_id):
    return str(ward_id).replace("_", " ")


def call(fn, *a, **k):
    try:
        return fn(*a, **k)
    except api.ApiError as e:
        alert(f"<b>Backend error.</b> {e.message}", "bad")
        st.stop()


def chart(fig, h=320):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=30, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans, sans-serif", color="#0E2A23"),
                      legend=dict(orientation="h", y=-0.18), hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E3ECE8")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


@st.cache_data(ttl=300, show_spinner=False)
def c_predict(h, city):
    return api.predict_all(h, city)


@st.cache_data(ttl=300, show_spinner=False)
def c_wards():
    return api.wards()


@st.cache_data(ttl=300, show_spinner=False)
def c_cities():
    return api.cities()


@st.cache_data(ttl=300, show_spinner=False)
def c_outlook(w):
    return api.outlook(w)


def controls(key, stock=False, crews=False):
    """Horizon + city (+ stock % and crews) row; stacks on mobile."""
    cols = st.columns([1.2, 1.2] + ([1.6] if stock else []) + ([1] if crews else []))
    h = cols[0].segmented_control("Forecast horizon", HORIZONS, default=2, format_func=lambda x: f"{x} weeks",
                                  key=f"{key}_h") or 2
    city = cols[1].selectbox("City", ["All cities"] + call(c_cities), key=f"{key}_c")
    out = {"h": h, "city": None if city == "All cities" else city}
    if stock:
        out["stock"] = cols[2].slider("Regional stock available (%)", 30, 100, 100, 5, key=f"{key}_s",
                                      help="Backend accepts 30–100%.")
    if crews:
        out["crews"] = cols[-1].number_input("Spraying crews", 0, 500, 10, 1, key=f"{key}_k")
    return out


# ---------------------------------------------------------------- pages
def home():
    try:
        hl = api.health()
        live = True
    except api.ApiError:
        hl, live = {}, False
    offline = hl.get("offline_mode")
    dot = "dot" if live else "dot off"
    status = "Backend online" if live else "Backend not reachable"
    html(f"""
    <div class="hero reveal">
      <div>
        <div class="brand"><img src="{img('logo')}" alt="Prohori logo"/><span>Prohori</span></div>
        <span class="eyebrow"><i class="{dot}"></i>{status}</span>
        <h1>Stay ahead of dengue,<br/><span>ward by ward.</span></h1>
        <p>Prohori forecasts dengue risk 2–4 weeks ahead for 60 urban wards in Bangladesh and turns it into
        action: supplies, hospital beds, spraying crews and ready-to-send dispatch notices. Core features run fully offline.</p>
        <div class="chips">
          <span class="chip">Data through {hl.get('dataset_through', '–')}</span>
          <span class="chip">{hl.get('wards', '–')} wards</span>
          <span class="chip">Horizons {', '.join(f'{h}w' for h in hl.get('loaded_horizons', [])) or '–'}</span>
          <span class="chip">{'Offline mode' if offline else 'Online'}</span>
        </div>
      </div>
      <img src="{img('hero')}" alt="City protected from mosquitoes"/>
    </div>""")
    if not live:
        alert(f"Start the API first: <code>python -m uvicorn main:app</code> (expected at <code>{api.BASE_URL}</code>).", "bad")
        return

    pred = call(c_predict, 2, None)
    df = pd.DataFrame(pred["wards"])
    hot = df[df.risk_tier.isin(["High", "Critical"])]
    label(f"This fortnight · forecast for week of {pred['forecast_week']}")
    kpis([kpi("Expected cases", fmt(df.expected_cases.sum(), 1), "all wards, 2-week horizon"),
          kpi("High / Critical wards", len(hot), f"of {len(df)} wards", "#D64545" if len(hot) else "#2BAE66"),
          kpi("Top ward", nice(df.sort_values("expected_cases").iloc[-1].ward_id),
              f"{df.expected_cases.max():.1f} expected cases", "#F08A24"),
          kpi("Chat assistant", "Template" if "template only" in hl.get("chatbot", "") else "Gemini",
              hl.get("chatbot", ""), "#7357D8")])

    label("What you can do")
    feats = [("mosquito", "Risk forecast", "Ward-level risk tiers and expected cases.", "forecast"),
             ("supplies", "Allocate supplies", "Split scarce saline, kits and beds by priority.", "allocate"),
             ("hospital", "Route overflow", "Move extra patients within the same city.", "overflow"),
             ("crew", "Simulate crews", "Cases prevented per spraying crew, weeks 2–4.", "simulate"),
             ("dispatch", "Dispatch notices", "SMS, WhatsApp and a printable brief.", "dispatch"),
             ("weather", "Weather context", "Live Open-Meteo or stored seasonal data.", "weather")]
    for row in (feats[:3], feats[3:]):
        cols = st.columns(3)
        for col, (icon, t, d, page) in zip(cols, row):
            with col:
                html(f'<div class="feature reveal"><img src="{img(icon)}"/><h4>{t}</h4><p>{d}</p></div>')
                st.page_link(PAGES[page], label=f"Open {t.lower()} →")

    label("How it works")
    steps = [("Forecast", "Trained regressors and classifiers read 101 local features per ward."),
             ("Prioritise", "High-risk, dense wards are served first when stock is limited."),
             ("Act", "Overflow routing, crew simulation and dispatch notices — no internet needed.")]
    for i, (t, d) in enumerate(steps, 1):
        html(f'<div class="step reveal"><b>{i}</b><div><h5>{t}</h5><p>{d}</p></div></div>')


def forecast():
    section("mosquito", "Risk forecast", "Expected dengue cases and risk tier for every ward.")
    c = controls("fc")
    pred = call(c_predict, c["h"], c["city"])
    df = pd.DataFrame(pred["wards"])
    probs = pd.json_normalize(df["probabilities"]).add_prefix("p_")
    df = pd.concat([df.drop(columns="probabilities"), probs], axis=1)
    counts = df.risk_tier.value_counts().reindex(TIERS, fill_value=0)

    kpis([kpi("Forecast week", pred["forecast_week"], f"origin {pred['origin_week']}"),
          kpi("Expected cases", fmt(df.expected_cases.sum(), 1), f"{len(df)} wards"),
          kpi("Critical", int(counts["Critical"]), "wards", TIER_COLORS["Critical"]),
          kpi("High", int(counts["High"]), "wards", TIER_COLORS["High"])])

    a, b = st.columns([1, 1.6])
    with a:
        label("Risk tiers")
        nz = counts[counts > 0]
        fig = go.Figure(go.Pie(labels=nz.index, values=nz.values, hole=.62, sort=False,
                               marker_colors=[TIER_COLORS[t] for t in nz.index], textinfo="value"))
        chart(fig, 300)
    with b:
        label("Top 12 wards by expected cases")
        top = df.nlargest(12, "expected_cases").sort_values("expected_cases")
        fig = px.bar(top, x="expected_cases", y="ward_id", color="risk_tier", orientation="h",
                     color_discrete_map=TIER_COLORS, category_orders={"risk_tier": TIERS},
                     labels={"expected_cases": "Expected cases", "ward_id": ""})
        chart(fig, 300)

    label("City view")
    cs = df.groupby("city", as_index=False).agg(expected_cases=("expected_cases", "sum"), wards=("ward_id", "count"))
    chart(px.bar(cs, x="city", y="expected_cases", text_auto=".1f", color_discrete_sequence=["#0F9D74"],
                 labels={"expected_cases": "Expected cases", "city": ""}), 260)

    label("All wards")
    show = df[["ward_id", "city", "risk_tier", "expected_cases", "p_Critical", "p_High", "population_density_sqkm"]]
    st.dataframe(show.sort_values("expected_cases", ascending=False), hide_index=True, use_container_width=True,
                 column_config={"ward_id": "Ward", "city": "City", "risk_tier": "Risk",
                                "expected_cases": st.column_config.NumberColumn("Expected cases", format="%.2f"),
                                "p_Critical": st.column_config.ProgressColumn("P(Critical)", min_value=0, max_value=1, format="%.2f"),
                                "p_High": st.column_config.ProgressColumn("P(High)", min_value=0, max_value=1, format="%.2f"),
                                "population_density_sqkm": st.column_config.NumberColumn("Density /km²", format="%d")})
    st.caption(pred["source"])


def ward():
    section("mosquito", "Ward outlook", "One ward, three horizons: 2, 3 and 4 weeks ahead.")
    wl = call(c_wards)
    options = [w["ward_id"] for w in wl]
    city_of = {w["ward_id"]: w["city"] for w in wl}
    wid = st.selectbox("Ward", options, format_func=lambda w: f"{nice(w)} · {city_of[w]}")
    fc = call(c_outlook, wid)["forecasts"]
    cols = st.columns(3)
    for col, f in zip(cols, fc):
        rows = "".join(f'<div class="prob"><span>{t}</span><div class="bar" style="--tone:{TIER_COLORS[t]}">'
                       f'<i style="width:{f["probabilities"][t] * 100:.0f}%"></i></div><span>{f["probabilities"][t]:.0%}</span></div>'
                       for t in TIERS)
        col.markdown(f"""<div class="hcard reveal"><div class="top"><b>{f['horizon_weeks']} weeks ahead</b>{tier(f['risk_tier'])}</div>
            <div class="big">{f['expected_cases']:.1f}</div><div class="s" style="color:#5B7068;font-size:.82rem">
            expected cases · week of {f['forecast_week']}</div>{rows}</div>""", unsafe_allow_html=True)
    label("Trend across horizons")
    d = pd.DataFrame([{"Week": f"{f['horizon_weeks']}w · {f['forecast_week']}", "Expected cases": f["expected_cases"]} for f in fc])
    fig = px.area(d, x="Week", y="Expected cases", markers=True, color_discrete_sequence=["#0F9D74"])
    chart(fig, 260)
    st.caption(f"Density {fc[0]['population_density_sqkm']:,.0f} people/km² · data through {fc[0]['dataset_through']}")


def allocate():
    section("supplies", "Resource allocation", "Share limited saline, diagnostic kits, beds and crews across wards.")
    c = controls("al", stock=True, crews=True)
    r = call(api.allocate, c["h"], c["city"], c["stock"], c["crews"])
    a = r["allocation"]
    cards = []
    for key, (name, _) in RES.items():
        t = a["resource_totals"][key]
        pct = 100 * t["allocated"] / t["required"] if t["required"] else 100
        tone = "#D64545" if t["deficit"] else "#0F9D74"
        cards.append(kpi(name, f"{fmt(t['allocated'])} / {fmt(t['required'])}",
                         f"deficit {fmt(t['deficit'])} · stock {fmt(t['available'])}", tone, pct))
    cards.append(kpi("Spraying crews", f"{a['assigned_spraying_crews']} / {a['total_spraying_crews']}",
                     f"{a['unassigned_spraying_crews']} unassigned", "#F08A24",
                     100 * a["assigned_spraying_crews"] / a["total_spraying_crews"] if a["total_spraying_crews"] else 0))
    kpis(cards)

    if a["has_critical_supply_deficit"]:
        for w in a["critical_supply_deficit_warnings"][:8]:
            name = RES.get(w.get("resource"), (w.get("resource", "Resource"),))[0]
            alert(f"<b>Critical deficit · {name}</b> — {fmt(w.get('allocated'))} of {fmt(w.get('required'))} "
                  f"allocated ({w.get('fulfillment_pct', 0):.0f}%), short by {fmt(w.get('deficit'))}.", "bad")
    else:
        alert(f"No critical supply deficit at {a['available_regional_stock_pct']:.0f}% stock for week of {r['forecast_week']}.", "ok")

    df = pd.DataFrame(a["wards"])
    label("Fulfilment by ward")
    only = st.toggle("Only wards with a deficit", key="al_only")
    if only:
        df = df[df.supply_deficit_warning]
    if df.empty:
        st.info("No wards with a deficit.")
        return
    cols = ["ward_id", "city", "risk_tier", "expected_cases", "saline_bags_allocated", "saline_bags_needed",
            "saline_bags_fulfillment_pct", "diagnostic_kits_allocated", "diagnostic_kits_fulfillment_pct",
            "hospital_beds_allocated", "hospital_beds_fulfillment_pct", "crews_assigned", "critical_supply_deficit_warning"]
    pc = lambda t: st.column_config.ProgressColumn(t, min_value=0, max_value=100, format="%.0f%%")
    st.dataframe(df[cols], hide_index=True, use_container_width=True,
                 column_config={"ward_id": "Ward", "city": "City", "risk_tier": "Risk",
                                "expected_cases": st.column_config.NumberColumn("Cases", format="%.1f"),
                                "saline_bags_allocated": "Saline", "saline_bags_needed": "Saline need",
                                "saline_bags_fulfillment_pct": pc("Saline %"), "diagnostic_kits_allocated": "Kits",
                                "diagnostic_kits_fulfillment_pct": pc("Kits %"), "hospital_beds_allocated": "Beds",
                                "hospital_beds_fulfillment_pct": pc("Beds %"), "crews_assigned": "Crews",
                                "critical_supply_deficit_warning": "Critical"})
    with st.expander("Resource ratios used (per expected case)"):
        st.json(a["resource_ratios"])


def overflow():
    section("hospital", "Hospital overflow", "Wards over bed capacity send patients to wards with spare beds — same city only.")
    c = controls("ov", stock=True)
    with st.expander("Override bed capacity (optional)"):
        st.caption("Leave empty to use the backend's historical estimate.")
        ed = st.data_editor(pd.DataFrame({"ward_id": pd.Series(dtype=str), "bed_capacity": pd.Series(dtype=int)}),
                            num_rows="dynamic", use_container_width=True, key="ov_ed",
                            column_config={"ward_id": st.column_config.SelectboxColumn(
                                "Ward", options=[w["ward_id"] for w in call(c_wards)]),
                                "bed_capacity": st.column_config.NumberColumn("Beds", min_value=0, step=1)})
    overrides = {r.ward_id: int(r.bed_capacity) for r in ed.dropna().itertuples()}
    r = call(api.overflow, c["h"], c["city"], c["stock"], 0, overrides)
    o = r["overflow"]
    cs = pd.DataFrame(o["city_summaries"])
    kpis([kpi("Bed demand", fmt(cs.total_bed_demand.sum()), f"week of {r['forecast_week']}"),
          kpi("Bed capacity", fmt(cs.total_bed_capacity.sum()), "estimated / overridden", "#3B6FD8"),
          kpi("Patients transferred", fmt(cs.patients_transferred.sum()), o["routing_rule"], "#F08A24"),
          kpi("Unmet after routing", fmt(cs.unmet_deficit_after_routing.sum()), "beds still short",
              "#D64545" if cs.unmet_deficit_after_routing.sum() else "#2BAE66")])

    label("City bed utilisation")
    fig = go.Figure()
    fig.add_bar(x=cs.city, y=cs.total_bed_capacity, name="Capacity", marker_color="#BFD2F7")
    fig.add_bar(x=cs.city, y=cs.total_bed_demand, name="Demand", marker_color="#3B6FD8")
    fig.update_layout(barmode="overlay")
    chart(fig, 280)

    label("Transfers")
    if o["transfers"]:
        st.dataframe(pd.DataFrame(o["transfers"]), hide_index=True, use_container_width=True,
                     column_config={"from_ward_id": "From", "to_ward_id": "To", "patients_transferred": "Patients", "city": "City"})
    else:
        alert("No transfers needed — every ward is within its bed capacity.", "ok")

    label("Ward beds")
    wd = pd.DataFrame(o["wards"])[["ward_id", "city", "hospital_beds_required", "bed_capacity", "bed_utilization_pct",
                                   "beds_sent_out", "beds_received", "net_bed_deficit"]]
    st.dataframe(wd, hide_index=True, use_container_width=True,
                 column_config={"ward_id": "Ward", "city": "City", "hospital_beds_required": "Beds needed",
                                "bed_capacity": "Capacity",
                                "bed_utilization_pct": st.column_config.ProgressColumn("Use", min_value=0, max_value=100, format="%.0f%%"),
                                "beds_sent_out": "Sent", "beds_received": "Received", "net_bed_deficit": "Short"})
    st.caption(r.get("bed_capacity_source", ""))


def simulate():
    section("crew", "Vector-control simulator", "What if we send larvicide / fogging crews? Model re-run over weeks 2–4.")
    cols = st.columns([1, 1.2, 1])
    total = cols[0].number_input("Total crews", 0, 500, 10, 1)
    city = cols[1].selectbox("City", ["All cities"] + call(c_cities), key="sim_c")
    mode = cols[2].segmented_control("Placement", ["Auto (best ROI)", "Manual"], default="Auto (best ROI)") or "Auto (best ROI)"
    city = None if city == "All cities" else city
    with st.expander("Model assumptions"):
        e1, e2 = st.columns(2)
        eff = e1.slider("Effect of first crew", 0.05, 0.50, 0.15, 0.01, help="Share of transmission removed by one crew.")
        mx = e2.slider("Maximum total effect", 0.10, 1.00, 0.60, 0.05)
    deployments = None
    if mode == "Manual":
        wl = [w for w in call(c_wards) if not city or w["city"] == city]
        ed = st.data_editor(pd.DataFrame({"ward_id": [w["ward_id"] for w in wl], "city": [w["city"] for w in wl], "crews": 0}),
                            hide_index=True, use_container_width=True, disabled=["ward_id", "city"], height=280,
                            column_config={"crews": st.column_config.NumberColumn("Crews", min_value=0, max_value=4, step=1)})
        deployments = {r.ward_id: int(r.crews) for r in ed.itertuples() if r.crews > 0}
        st.caption(f"{sum(deployments.values())} crews placed by hand · max 4 per ward.")
    s = call(api.simulate, total, city, deployments, eff, mx)
    m = s["summary"]
    kpis([kpi("Cases prevented", fmt(m["cases_prevented"], 2), f"{m['baseline_cases']:.1f} → {m['simulated_cases']:.1f}",
              pct=100 * m["cases_prevented"] / m["baseline_cases"] if m["baseline_cases"] else 0),
          kpi("Beds freed", fmt(m["beds_prevented"], 2), "hospital beds", "#3B6FD8"),
          kpi("Saline saved", fmt(m["saline_prevented"], 2), f"kits {fmt(m['kits_prevented'], 2)}", "#7357D8"),
          kpi("Cases per crew", fmt(m["overall_roi_cases_per_crew"], 3),
              f"{s['crews_deployed']} deployed · {s['crews_unused']} unused", "#F08A24")])

    a, b = st.columns([1.2, 1])
    with a:
        label("Baseline vs with crews")
        wk = pd.DataFrame(s["weekly_totals"]).T.reset_index(names="week")
        wk["week"] = wk["week"].str.replace("week_", "Week ")
        fig = go.Figure()
        fig.add_bar(x=wk.week, y=wk.baseline, name="Baseline", marker_color="#F6B9B9")
        fig.add_bar(x=wk.week, y=wk.simulated, name="With crews", marker_color="#0F9D74")
        chart(fig, 300)
    with b:
        label("Best wards for the first crew")
        st.dataframe(pd.DataFrame(s["best_first_crew_candidates"]).head(10), hide_index=True, use_container_width=True,
                     column_config={"ward_id": "Ward", "city": "City",
                                    "first_crew_roi": st.column_config.NumberColumn("Cases / crew", format="%.3f"),
                                    "baseline_cases": st.column_config.NumberColumn("Baseline", format="%.2f")})
    label("ROI ranking · cases prevented per crew")
    if s["roi_ranking"]:
        rr = pd.DataFrame(s["roi_ranking"])
        st.dataframe(rr[["roi_rank", "ward_id", "city", "crews", "effect_pct", "baseline_cases", "simulated_cases",
                         "cases_prevented", "roi_cases_per_crew", "bed_capacity"]], hide_index=True, use_container_width=True,
                     column_config={"roi_rank": "#", "ward_id": "Ward", "city": "City", "crews": "Crews",
                                    "effect_pct": st.column_config.NumberColumn("Effect", format="%.0f%%"),
                                    "baseline_cases": "Baseline", "simulated_cases": "Simulated",
                                    "cases_prevented": "Prevented",
                                    "roi_cases_per_crew": st.column_config.NumberColumn("Per crew", format="%.3f"),
                                    "bed_capacity": "Bed cap."})
    else:
        st.info("No crews deployed yet — add crews above.")
    st.caption(f"{s['method']} · mode {s['mode']} · origin week {s.get('origin_week', '–')}")


def dispatch():
    section("dispatch", "Dispatch notices", "Ready-to-send SMS, WhatsApp and a printable commissioner brief.")
    c = controls("dp", stock=True, crews=True)
    d = call(api.dispatch, c["h"], c["city"], c["stock"], c["crews"])
    msgs = d["ward_messages"]
    kpis([kpi("Ward notices", len(msgs), f"week of {d['forecast_week']}"),
          kpi("Horizon", f"{d['horizon_weeks']} weeks", "forecast lead time", "#3B6FD8"),
          kpi("Transfers", len(d["overflow"]["transfers"]), "same-city", "#F08A24")])
    ids = [m["ward_id"] for m in msgs]
    wid = st.selectbox("Preview ward", ids, format_func=nice)
    m = next(x for x in msgs if x["ward_id"] == wid)
    a, b = st.columns(2)
    with a:
        label(f"SMS · {len(m['sms'])} characters")
        html(f'<div class="phone reveal"><div class="screen sms"><div class="bubble">{m["sms"]}</div>'
             f'<div class="meta">Prohori</div></div></div>')
    with b:
        label("WhatsApp")
        wa = m["whatsapp"].replace(" | ", "\n").replace("; ", "\n")
        html(f'<div class="phone reveal"><div class="screen"><div class="bubble wa">{wa}</div>'
             f'<div class="meta">✓✓ Prohori</div></div></div>')

    label("Download")
    tag = (c["city"] or "all_cities").lower().replace(" ", "_") + f"_h{c['h']}w"
    sep = "\n" + "=" * 60 + "\n"
    x1, x2, x3, x4 = st.columns(4)
    x1.download_button("SMS .txt", sep.join(x["sms"] for x in msgs), f"sms_{tag}.txt", use_container_width=True)
    x2.download_button("WhatsApp .txt", sep.join(x["whatsapp"] for x in msgs), f"whatsapp_{tag}.txt", use_container_width=True)
    brief = call(api.dispatch_html, c["h"], c["city"], c["stock"], c["crews"])
    x3.download_button("Brief .html", brief, f"commissioner_brief_{tag}.html", "text/html", use_container_width=True)
    x4.download_button("Payload .json", json.dumps(d, indent=2), f"dispatch_{tag}.json", "application/json",
                       use_container_width=True)


    if st.button("Save all to backend /exports", type="primary"):
        res = call(api.dispatch_export, c["h"], c["city"], c["stock"], c["crews"])
        alert("Saved: " + ", ".join(f"<code>{p}</code>" for p in res["exported"]), "ok")
    with st.expander("Preview printable brief (use your browser's Print → Save as PDF)"):
        components.html(brief, height=560, scrolling=True)


def weather():
    section("weather", "Weather context", "Rain, heat and humidity drive mosquito breeding.")
    w = call(api.weather)
    on = st.toggle("Offline mode (no internet calls)", value=w["status"] == "offline_fallback",
                   help="Switches the backend between live Open-Meteo and stored local data.")
    if on != (w["status"] == "offline_fallback"):
        call(api.set_offline, on)
        st.rerun()
    kind = "ok" if w["status"] == "live" else "info"
    alert(f"<b>{w['status'].replace('_', ' ').title()}</b> · {w['source']}" + (f"<br/>{w['detail']}" if w.get("detail") else ""), kind)
    if not w["cities"]:
        return
    city = st.segmented_control("City", list(w["cities"]), default=list(w["cities"])[0]) or list(w["cities"])[0]
    cw = w["cities"][city]
    nx = cw.get("next_week") or {}
    kpis([kpi("Next week rain", f"{fmt(nx.get('rain_fwd1'), 1)} mm", "expected", "#2B8CC4"),
          kpi("Next week temp", f"{fmt(nx.get('temp_fwd1'), 1)} °C", "average", "#F08A24"),
          kpi("Next week humidity", f"{fmt(nx.get('hum_fwd1'), 0)} %", "relative", "#0F9D74")])
    df = pd.DataFrame(cw["recent_weeks"])
    if df.empty:
        return
    label("Recent weeks")
    fig = go.Figure()
    fig.add_bar(x=df.date, y=df.rainfall_mm, name="Rain (mm)", marker_color="#9DD3F0")
    fig.add_scatter(x=df.date, y=df.avg_temp_c, name="Avg temp (°C)", yaxis="y2", line=dict(color="#F08A24", width=3))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
    chart(fig, 300)


def assistant():
    section("dispatch", "Assistant", "Ask about forecasts. Works offline with a local template; Gemini if configured.")
    if "sid" not in st.session_state:
        st.session_state.sid = uuid.uuid4().hex[:16]
        st.session_state.msgs = []
    wl = call(c_wards)
    a, b = st.columns([2, 1])
    wid = a.selectbox("Ward context (optional)", ["None"] + [w["ward_id"] for w in wl], format_func=lambda x: nice(x))
    h = b.segmented_control("Horizon", HORIZONS, default=2, format_func=lambda x: f"{x}w", key="ch_h") or 2
    for m in st.session_state.msgs:
        st.chat_message(m["role"]).markdown(m["text"])
    q = st.chat_input("e.g. Which wards need attention?", max_chars=300)
    if q:
        st.session_state.msgs.append({"role": "user", "text": q})
        st.chat_message("user").markdown(q)
        with st.chat_message("assistant"), st.spinner("Thinking…"):
            try:
                r = api.chat(q, st.session_state.sid, None if wid == "None" else wid, h)
                text = r["reply"] + f"\n\n<sub>{r.get('source', '')}</sub>"
            except api.ApiError as e:
                text = f"⚠️ {e.message}"
            st.markdown(text, unsafe_allow_html=True)
        st.session_state.msgs.append({"role": "assistant", "text": text})


def about():
    section("mosquito", "About", "How Prohori works and what it assumes.")
    for t, d in [("Models", "Phase 2 regressors (expected cases) and classifiers (risk tier) for 2, 3 and 4 weeks, trained on 10 years of ward data."),
                 ("Allocation", "Needs per case come from the dataset's saline, kit and bed ratios. With 30–100% stock, high-risk dense wards go first."),
                 ("Overflow", "Bed capacity is a historical estimate unless overridden. Transfers stay inside the same city."),
                 ("Simulator", "Crews reduce case momentum and breeding features with diminishing returns, ramping over weeks 2–4."),
                 ("Offline", "Forecasts, operations and dispatch need no internet. Weather and Gemini chat are optional.")]:
        html(f'<div class="step reveal"><b>•</b><div><h5>{t}</h5><p>{d}</p></div></div>')
    st.caption(f"API: {api.BASE_URL} · docs at {api.BASE_URL}/docs")


# ---------------------------------------------------------------- navigation
PAGES = {
    "home": st.Page(home, title="Home", icon=":material/home:", default=True),
    "forecast": st.Page(forecast, title="Forecast", icon=":material/monitoring:", url_path="forecast"),
    "ward": st.Page(ward, title="Ward", icon=":material/location_on:", url_path="ward"),
    "allocate": st.Page(allocate, title="Allocate", icon=":material/inventory_2:", url_path="allocate"),
    "overflow": st.Page(overflow, title="Overflow", icon=":material/local_hospital:", url_path="overflow"),
    "simulate": st.Page(simulate, title="Simulate", icon=":material/pest_control:", url_path="simulate"),
    "dispatch": st.Page(dispatch, title="Dispatch", icon=":material/send:", url_path="dispatch"),
    "weather": st.Page(weather, title="Weather", icon=":material/rainy:", url_path="weather"),
    "assistant": st.Page(assistant, title="Assistant", icon=":material/chat:", url_path="assistant"),
    "about": st.Page(about, title="About", icon=":material/info:", url_path="about"),
}
st.navigation(list(PAGES.values()), position="top").run()
html(f'<div class="foot"><img src="{img("logo")}" alt=""/> Prohori · dengue early warning · forecasts are decision support, not diagnosis</div>')
