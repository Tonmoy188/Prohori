"""PROHORI | DengueWatch - Streamlit frontend for the existing FastAPI backend. Run: streamlit run app.py"""
import calendar
import html
import math
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import api_client as api

HERE = Path(__file__).resolve().parent
TIERS = ["Low", "Medium", "High", "Critical"]
TIER_RANK = {t: i for i, t in enumerate(TIERS)}
TIER_COLOR = {"Low": "#2E9B5F", "Medium": "#E8B800", "High": "#F08A24", "Critical": "#D64545"}
HORIZONS = (2, 3, 4)
PAGES = [("Home", "home"), ("Overview", "dashboard"), ("Ward Explorer", "location_city"),
         ("Weather Outlook", "cloud"), ("Prediction Details", "monitoring"), ("AI Assistant", "smart_toy"),
         ("Model Performance", "analytics"), ("About & Resources", "info")]
SUGGESTIONS = ["Which wards have the highest estimated risk in the 3-week outlook?",
               "How is the weather affecting dengue risk right now?",
               "What does a High risk tier mean for a ward?",
               "Compare the 2-week and 4-week outlook for Dhaka North."]

st.set_page_config(page_title="PROHORI | DengueWatch", page_icon=":material/shield:", layout="wide",
                   initial_sidebar_state="auto")
ss = st.session_state
ss.setdefault("page", "Home")
ss.setdefault("horizon", 3)
ss.setdefault("city", "All cities")
ss.setdefault("ward", None)
ss.setdefault("chat", [])
ss.setdefault("sid", "dw-" + uuid.uuid4().hex[:12])


# ------------------------------------------------------------------ data access (cached)
@st.cache_data(ttl=20, show_spinner=False)
def c_health():
    return api.health()


@st.cache_data(ttl=60, show_spinner=False)
def c_stats():
    return api.stats()


@st.cache_data(ttl=600, show_spinner=False)
def c_wards():
    return api.wards()


@st.cache_data(ttl=300, show_spinner="Loading ward predictions...")
def c_all(h, city):
    return api.predict_all(h, city)


@st.cache_data(ttl=300, show_spinner="Loading ward outlook...")
def c_outlook(ward_id):
    return api.outlook(ward_id)


@st.cache_data(ttl=3600, show_spinner=False)
def c_metrics():
    return api.metrics()


@st.cache_data(ttl=600, show_spinner="Loading weather...")
def c_weather(city):
    return api.weather(city)


@st.cache_data(ttl=600, show_spinner=False)
def c_wforecast(city):
    return api.weather_forecast(city)


# ------------------------------------------------------------------ backend CSV files (read from disk, never cached on failure)
WARD_COLS = ["ward_id", "population_density_sqkm", "water_logging_risk_index", "sanitation_score"]


@st.cache_data(show_spinner=False)
def _read_ward_csv(path, mtime):
    """Only successful reads are cached (an exception is never cached). mtime in the key refreshes edited files."""
    return pd.read_csv(path, usecols=WARD_COLS)


@st.cache_data(show_spinner=False)
def _season_profile(path, mtime):
    d = pd.read_csv(path, usecols=["date", "dengue_cases"], parse_dates=["date"]).dropna(subset=["dengue_cases"])
    g = d.groupby(d.date.dt.month).dengue_cases.agg(["mean", "max"])
    return {"mean": {int(m): float(v) for m, v in g["mean"].items()}, "max": {int(m): float(v) for m, v in g["max"].items()},
            "start": str(d.date.min().date()), "end": str(d.date.max().date())}


def find_csv(env_name, filename, required, first_dir=None):
    """Look for a CSV on every call, so a file that appears later is picked up on the next rerun.
    Returns (path | None, notices). Notices list every problem met on the way, even when a fallback worked."""
    cands, notices, tried = [], [], []
    env = os.getenv(env_name, "").strip()
    if env:
        cands.append((Path(env).expanduser(), True))
    if first_dir:
        cands.append((Path(first_dir) / filename, False))
    cands += [(HERE.parent / "data" / filename, False), (HERE / "data" / filename, False)]
    seen = set()
    for path, from_env in cands:
        if str(path) in seen:
            continue
        seen.add(str(path))
        tried.append(str(path))
        if not path.is_file():
            if from_env:
                notices.append(f"{env_name} is set to {path}, but that file does not exist.")
            continue
        try:
            cols = list(pd.read_csv(path, nrows=0).columns)
        except Exception as exc:
            notices.append(f"Could not read {path}: {type(exc).__name__}: {exc}")
            continue
        missing = [c for c in required if c not in cols]
        if missing:
            notices.append(f"{path} is missing column(s): {', '.join(missing)}.")
            continue
        return path, notices
    notices.append("No usable file found. Looked in: " + "; ".join(tried))
    return None, notices


def ward_csv_state():
    """(ward vulnerability DataFrame indexed by ward_id | None, notices)."""
    path, notes = find_csv("DENGUEWATCH_WARDS_CSV", "wards.csv", WARD_COLS)
    if path is None:
        return None, notes
    try:
        return _read_ward_csv(str(path), path.stat().st_mtime).set_index("ward_id"), notes
    except Exception as exc:
        return None, notes + [f"Could not read {path}: {type(exc).__name__}: {exc}"]


def season_state():
    """(monthly historical case profile | None, notices). The history CSV is looked up beside wards.csv first."""
    wpath, _ = find_csv("DENGUEWATCH_WARDS_CSV", "wards.csv", ["ward_id"])
    path, notes = find_csv("DENGUEWATCH_HISTORY_CSV", "dengue_history.csv", ["date", "dengue_cases"],
                           wpath.parent if wpath else None)
    if path is None:
        return None, notes
    try:
        return _season_profile(str(path), path.stat().st_mtime), notes
    except Exception as exc:
        return None, notes + [f"Could not read {path}: {type(exc).__name__}: {exc}"]


def c_meta():
    return ward_csv_state()[0]


def csv_notice(what, meta_ok, notes, key, envs):
    """Visible warning (with a retry button) when a CSV is missing or had problems."""
    if meta_ok and not notes:
        return
    body = "<br>".join(esc(n) for n in notes)
    if not meta_ok:
        note(f"<b>{esc(what)} unavailable.</b><br>{body}<br>Keep this folder inside the backend folder, or set {envs}. "
             "It is read again on every page update, so it appears as soon as the file is available.", "warn")
        st.button("Check again", key=f"csv_retry_{key}", icon=":material/refresh:", on_click=refresh)
    else:
        note(f"<b>{esc(what)} loaded from a fallback location.</b><br>{body}", "warn")


def load(fn, *a):
    try:
        return fn(*a), None
    except api.ApiError as e:
        return None, e


def refresh():
    st.cache_data.clear()


def need(fn, *a):
    data, err = load(fn, *a)
    if err:
        icon = "Backend offline" if err.kind == "offline" else "Could not load data"
        hint = " The backend failed internally; check its console log." if err.status == 500 else ""
        md(f'<div class="note err"><b>{icon}.</b> {esc(err.message)}{hint}</div>')
        st.button("Try again", key=f"retry_{fn.__name__}", on_click=refresh)
        st.stop()
    return data


# ------------------------------------------------------------------ small helpers
def esc(x):
    return html.escape(str(x))


def md(s):
    st.markdown(re.sub(r"\n\s*", "", s.strip()), unsafe_allow_html=True)


def go_to(page, ward=None):
    ss.page = page
    ss["mnav_sel"] = page
    if ward:
        ss.ward = ward


def fmt_date(s):
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").strftime("%d %b %Y")
    except ValueError:
        return str(s)


def fmt_fetched(s):
    """Backend timestamps are UTC; Bangladesh has no DST, so a fixed UTC+6 offset is exact."""
    try:
        t = datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", s)).astimezone(timezone(timedelta(hours=6)))
        return t.strftime("%d %b %Y, %H:%M") + " (Dhaka time)"
    except Exception:
        return str(s)


def pct(p):
    return "<0.1%" if 0 < p < 0.001 else f"{p * 100:.1f}%"


def cases(x):
    return f"{x:.1f}"


def badge(t):
    return f'<span class="badge t-{esc(t)}">{esc(t)}</span>'


def kpis(items):
    md('<div class="kpi-grid">' + "".join(
        f'<div class="kpi k-{tone}"><div class="kpi-label">{esc(l)}</div><div class="kpi-value">{esc(v)}</div>'
        f'<div class="kpi-sub">{esc(s)}</div></div>' for l, v, s, tone in items) + "</div>")


def head(title, sub):
    md(f'<div class="page-head"><h1>{esc(title)}</h1><p>{esc(sub)}</p></div>')


def card(name):
    return st.container(key=f"card_{name}")


def ctitle(t, sub=""):
    md(f'<div class="card-title">{esc(t)}</div>' + (f'<div class="card-sub">{esc(sub)}</div>' if sub else ""))


def note(text, kind=""):
    md(f'<div class="note {kind}">{text}</div>')


def bars(rows, fmt=lambda v: f"{v:.1f}", color=None):
    """rows: (label, value, max, color|None)"""
    out = ""
    for label, v, mx, col in rows:
        w = 0 if not mx else max(2, min(100, abs(v) / mx * 100))
        out += (f'<div class="bar-row"><span>{esc(label)}</span><div class="bar"><i style="width:{w:.0f}%;'
                f'background:{col or color or "#087A50"}"></i></div><b>{fmt(v)}</b></div>')
    md(out)


def style(fig, h=290, legend=False):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=10, b=52 if legend else 8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", showlegend=legend, bargap=0.35,
                      font=dict(family="Plus Jakarta Sans, system-ui, sans-serif", color="#10291F", size=12),
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(gridcolor="#E3F1E9", zeroline=False)
    fig.update_yaxes(gridcolor="#E3F1E9", zeroline=False)
    return fig


def show(fig):
    st.plotly_chart(fig, config={"displayModeBar": False})


def to_df(data):
    rows = [dict(ward_id=w["ward_id"], city=w["city"], tier=w["risk_tier"], cases=w["expected_cases"],
                 **{f"p_{t}": w["probabilities"].get(t, 0.0) for t in TIERS}) for w in data["wards"]]
    df = pd.DataFrame(rows)
    df["rank"] = df.tier.map(TIER_RANK)
    return df.sort_values(["rank", "cases"], ascending=False).reset_index(drop=True)


def ward_list():
    return need(c_wards)["wards"]


def horizon_picker(key):
    v = st.segmented_control("Prediction horizon", HORIZONS, default=ss.horizon,
                             format_func=lambda h: f"{h} Weeks", key=f"hz_{key}")
    if v:
        ss.horizon = v
    return ss.horizon


def city_picker(key, cities):
    opts = ["All cities"] + cities
    v = st.selectbox("City", opts, index=opts.index(ss.city) if ss.city in opts else 0, key=f"city_{key}")
    ss.city = v
    return None if v == "All cities" else v


def logo(size=40):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24"><path d="M12 2 20 5v6c0 5-3.5 9-8 11-4.5-2-8-6-8-11V5z" '
            'fill="#087A50"/><path d="M12 7s-3.5 4-3.5 6.3a3.5 3.5 0 0 0 7 0C15.5 11 12 7 12 7z" fill="#fff"/></svg>')


def brand(dark=False):
    md(f'<div class="brand {"dark" if dark else ""}">{logo()}<div><b>PROHORI</b><span>DengueWatch</span></div></div>')


def status_pill():
    h, err = load(c_health)
    if err:
        return '<span class="pill off"><i></i>Backend offline</span>'
    return '<span class="pill"><i></i>Backend connected</span>'


def weather_stale_pill(meta):
    return '<span class="pill warn"><i></i>Cached weather (provider unreachable)</span>' if meta.get("stale") else ""


# ------------------------------------------------------------------ shell (CSS + navigation)
css = (HERE / "styles.css").read_text(encoding="utf-8") if (HERE / "styles.css").is_file() else ""
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
if ss.page == "Home":
    st.markdown("<style>[data-testid='stSidebar'],[data-testid='stExpandSidebarButton'],"
                "[data-testid='stSidebarCollapsedControl']{display:none!important}"
                "[class*='st-key-topnav'] [data-testid='stColumn']:first-child{margin-right:auto}</style>",
                unsafe_allow_html=True)
else:
    with st.sidebar:
        brand()
        for name, icon in PAGES:
            st.button(name, key=f"nav_{name}", icon=f":material/{icon}:", on_click=go_to, args=(name,),
                      type="primary" if ss.page == name else "secondary")
        md(status_pill())
        md('<div class="side-foot"><b>Predict. Prepare. Protect.</b>Model estimates, not confirmed cases.</div>')
if ss.page != "Home":
    with st.container(key="mnav"):  # visible on phones/tablets only (see styles.css)
        ss["mnav_sel"] = ss.page
        st.selectbox("Menu", [n for n, _ in PAGES], key="mnav_sel", label_visibility="collapsed",
                     on_change=lambda: go_to(ss.mnav_sel))


# ------------------------------------------------------------------ pages
def page_home():
    with st.container(key="topnav"):
        cols = st.columns([3, 1, 1, 1.3, 1, 1.3])
        with cols[0]:
            brand(dark=True)
        for c, (label, target, kind) in zip(cols[1:], [("Home", "Home", "secondary"), ("Dashboard", "Overview", "secondary"),
                                                       ("Ward Explorer", "Ward Explorer", "secondary"),
                                                       ("About", "About & Resources", "secondary"),
                                                       ("Get Started", "Overview", "primary")]):
            with c:
                st.button(label, key=f"top_{label}", on_click=go_to, args=(target,), type=kind)
    with st.container(key="hero"):
        left, right = st.columns([1.35, 1])
        with left:
            md('<div class="hero"><h1>Dengue Early Warning Dashboard</h1><div class="tag">Predict. Prepare. Protect.</div>'
               '<p>Ward-level dengue risk estimates 2 to 4 weeks ahead, built from multi-week rainfall lags, temperature, '
               'humidity and local vulnerability. Designed to help health teams focus on the wards that need attention first.</p></div>')
            b1, b2, _ = st.columns([1.3, 1, 1.2])
            with b1:
                st.button("Explore Dashboard", key="cta_explore", type="primary", on_click=go_to, args=("Overview",))
            with b2:
                st.button("Learn More", key="cta_more", on_click=go_to, args=("About & Resources",))
        with right:
            with card("snapshot"):
                data, err = load(c_all, 3, None)
                ctitle("Latest model run (3-week outlook)", "Model estimates, not confirmed cases")
                if err:
                    note(f"<b>No live data.</b> {esc(err.message)}", "warn")
                else:
                    df = to_df(data)
                    cnt = df.tier.value_counts()
                    bars([(t, int(cnt.get(t, 0)), len(df), TIER_COLOR[t]) for t in TIERS], fmt=lambda v: f"{int(v)} wards")
                    md(f'<div class="card-sub">Forecast week of {fmt_date(data["forecast_week"])} - '
                       f'{len(df)} wards evaluated</div>')
                    if (df.tier == "Low").all():
                        md('<div class="card-sub">All evaluated wards are Low in this model run. '
                           'The Overview page explains the seasonal context.</div>')
    md('<div class="feat-grid">' + "".join(
        f'<div class="feat">{svg}<div><b>{t}</b><span>{d}</span></div></div>' for svg, t, d in [
            ('<svg viewBox="0 0 24 24" fill="#087A50"><rect x="3" y="12" width="4" height="9" rx="1"/><rect x="10" y="7" width="4" height="14" rx="1"/><rect x="17" y="3" width="4" height="18" rx="1"/></svg>',
             "Multi-Week Prediction", "2, 3 and 4 week outlook per ward"),
            ('<svg viewBox="0 0 24 24" fill="#087A50"><path d="M12 2a7 7 0 0 0-7 7c0 5 7 13 7 13s7-8 7-13a7 7 0 0 0-7-7zm0 9.5A2.5 2.5 0 1 1 12 6.500a2.5 2.5 0 0 1 0 5z"/></svg>',
             "Ward-Level Risk Analysis", "Rank wards and compare vulnerability"),
            ('<svg viewBox="0 0 24 24" fill="#087A50"><path d="M7 18a4 4 0 0 1-.5-8 5.500 5.500 0 0 1 10.600 1A3.500 3.500 0 0 1 17 18z"/></svg>',
             "Weather Insights", "Recent rainfall, temperature and humidity (Open-Meteo)"),
            ('<svg viewBox="0 0 24 24" fill="#087A50"><path d="M12 2 20 5v6c0 5-3.500 9-8 11-4.500-2-8-6-8-11V5zm-1.500 14 6-6-1.400-1.400-4.600 4.600-1.900-1.900L7.200 13z"/></svg>',
             "AI-Powered Assistant", "Ask questions grounded in model output")])
       + "</div>")


def fc_days_text(cw):
    """Describe how many days of a weekly weather aggregate are forecast values (field comes from the backend)."""
    n = (cw or {}).get("forecast_days")
    if n is None:
        return ""
    n = int(n)
    return f"includes {n} forecast day{'s' if n != 1 else ''}" if n else "no forecast days"


def seasonal_card(df, cur, h):
    """Explains the live result using the backend's historical case data. States 'all Low' only if the response says so."""
    target = pd.Timestamp(cur["forecast_week"])
    mname = calendar.month_name[target.month]
    n = len(df)
    n_low = int((df.tier == "Low").sum())
    prof, notes = season_state()
    with card("season"):
        ctitle("Seasonal context", f"Target week of {fmt_date(cur['forecast_week'])} ({h}-week horizon)")
        if n_low == n:
            note(f"<b>All {n} evaluated wards are Low risk</b> in this {h}-week model run. "
                 "That is the model's output for these wards, not a statement that dengue risk is zero.")
        else:
            note(f"<b>{n - n_low} of {n} evaluated wards are above Low</b> in this {h}-week model run "
                 f"({n_low} are Low).")
        if prof is None:
            csv_notice("Historical case data", False, notes, "season",
                       "<code>DENGUEWATCH_HISTORY_CSV</code> (full path of dengue_history.csv)")
            return
        mean, mx = prof["mean"], prof["max"]
        tm_mean = mean.get(target.month)
        if tm_mean is not None:
            peak_m = max(mean, key=mean.get)
            share = (tm_mean / mean[peak_m] * 100) if mean[peak_m] else 0
            md(f'<div class="card-sub">In the historical data ({fmt_date(prof["start"])} to {fmt_date(prof["end"])}), '
               f'<b>{esc(mname)}</b> averaged <b>{tm_mean:.2f}</b> cases per ward-week (highest single ward-week: '
               f'{mx[target.month]:.0f}). The peak month, {calendar.month_name[peak_m]}, averaged {mean[peak_m]:.1f}. '
               f'{esc(mname)} is {"less than 1%" if share < 1 else f"{share:.0f}%"} of the peak-month average.</div>')
        months = list(range(1, 13))
        fig = go.Figure(go.Bar(x=[calendar.month_abbr[m] for m in months], y=[mean.get(m, 0) for m in months],
                               marker_color=["#087A50" if m == target.month else "#9ED8BA" for m in months],
                               hovertemplate="%{x}: %{y:.2f} cases per ward-week<extra></extra>"))
        fig.update_yaxes(title="Avg cases per ward-week")
        show(style(fig, 260))
        st.caption("Historical context only: reported case counts from the dataset's years. The model estimates above "
                   "come from recent weather, multi-week lags and ward vulnerability.")


def page_overview():
    wards = ward_list()
    cities = sorted({w["city"] for w in wards})
    head("Overview Dashboard", "Estimated dengue risk across wards, from weather lags and ward vulnerability")
    c1, c2, c3 = st.columns([1.2, 1.8, 0.7])
    with c1:
        city = city_picker("ov", cities)
    with c2:
        h = horizon_picker("ov")
    with c3:
        md('<div style="height:1.7rem"></div>')
        st.button("Refresh", key="ov_refresh", icon=":material/refresh:", on_click=refresh)
    multi = {x: need(c_all, x, city) for x in HORIZONS}
    cur = multi[h]
    df = to_df(cur)
    meta = cur["weather"]
    md(status_pill() + " " + weather_stale_pill(meta) +
       f' <span class="card-sub">Weather last updated {esc(fmt_fetched(meta.get("fetched_at_utc", "")))}</span>')
    hc = int(df.tier.isin(["High", "Critical"]).sum())
    kpis([("Wards evaluated", len(df), f'{"All cities" if not city else city}', "Low"),
          ("High / Critical wards", hc, "Estimated risk tier", "High" if hc else "Low"),
          ("Estimated cases", f"{df.cases.sum():.1f}", "Model estimate, all wards in view", "Medium"),
          ("Prediction horizon", f"{h} weeks", f'Week of {fmt_date(cur["forecast_week"])}', "Low")])

    a, b = st.columns(2)
    with a, card("tiers"):
        ctitle("Risk-tier distribution", f"Wards per predicted tier, {h}-week horizon")
        cnt = df.tier.value_counts()
        ys = [int(cnt.get(t, 0)) for t in TIERS]
        fig = go.Figure(go.Bar(x=TIERS, y=ys, marker_color=[TIER_COLOR[t] for t in TIERS], text=ys, textposition="outside"))
        fig.update_yaxes(range=[0, max(ys) * 1.2 + 1], title="Wards")
        show(style(fig))
    with b, card("multi"):
        ctitle("Multi-week forecast comparison", "Total estimated cases across wards in view")
        totals = [to_df(multi[x]).cases.sum() for x in HORIZONS]
        hic = [int(to_df(multi[x]).tier.isin(["High", "Critical"]).sum()) for x in HORIZONS]
        fig = go.Figure(go.Bar(x=[f"{x} weeks" for x in HORIZONS], y=totals, text=[f"{v:.1f}" for v in totals],
                               textposition="outside", customdata=hic,
                               marker_color=["#087A50" if x == h else "#9ED8BA" for x in HORIZONS],
                               hovertemplate="%{x}: %{y:.1f} est. cases<br>High/Critical wards: %{customdata}<extra></extra>"))
        fig.update_yaxes(range=[0, max(totals) * 1.2 + 1], title="Estimated cases")
        show(style(fig))
    a, b = st.columns(2)
    with a, card("top"):
        ctitle("Top wards by estimated cases", f"{h}-week horizon, colored by predicted tier")
        top = df.nlargest(8, "cases").iloc[::-1]
        fig = go.Figure(go.Bar(x=top.cases, y=top.ward_id, orientation="h", marker_color=[TIER_COLOR[t] for t in top.tier],
                               customdata=top.tier, hovertemplate="%{y}: %{x:.2f} est. cases (%{customdata})<extra></extra>"))
        fig.update_xaxes(title="Estimated cases")
        show(style(fig, 320))
    with b, card("prob"):
        ctitle("Risk probability breakdown", "Average model probability per tier across wards (not a confidence interval)")
        ys = [df[f"p_{t}"].mean() * 100 for t in TIERS]
        fig = go.Figure(go.Bar(x=TIERS, y=ys, marker_color=[TIER_COLOR[t] for t in TIERS],
                               text=[f"{v:.1f}%" for v in ys], textposition="outside"))
        fig.update_yaxes(range=[0, 115], title="Mean probability (%)")
        show(style(fig, 320))

    seasonal_card(df, cur, h)

    with card("weather"):
        ctxs = cur.get("weather_contexts") or {}
        ctitle("City weather summary", "Open-Meteo, city level. Wards in a city share the same weather inputs. Full detail on the Weather Outlook page.")
        if not ctxs:
            note("The backend returned no weather context.", "warn")
        else:
            md('<div class="kpi-grid">' + "".join(
                f'<div class="kpi"><div class="kpi-label">{esc(c)}</div>'
                f'<div class="kpi-value">{x["current_week_weather"]["avg_temp_c"]:.1f} C</div>'
                f'<div class="kpi-sub">Humidity {x["current_week_weather"]["relative_humidity_pct"]:.0f}% - '
                f'Rain {x["current_week_weather"]["rainfall_mm"]:.1f} mm<br>Current-week average, '
                f'{esc(fc_days_text(x["current_week_weather"]))}</div></div>' for c, x in ctxs.items()) + "</div>")
            wc = st.selectbox("Daily forecast for", list(ctxs), key="ov_wc") if len(ctxs) > 1 else next(iter(ctxs))
            daily = ctxs[wc].get("daily_forecast") or []
            if daily:
                show(style(forecast_fig(daily), 300, True))
                st.caption(f'Provider forecast covers {len(daily)} days, through {fmt_date(daily[-1]["date"])}. '
                           'Forecast weather is context only; it is not weather for the 3-4 week target week.')
            else:
                note("No daily forecast available for this city.", "warn")
    st.caption("All values are model estimates for ward-weeks, not confirmed dengue cases.")


def forecast_fig(daily):
    d = pd.DataFrame(daily)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=d.date, y=d.rainfall_mm, name="Rainfall (mm)", marker_color="#7FB7E6", secondary_y=False)
    fig.add_scatter(x=d.date, y=d.avg_temp_c, name="Avg temp (C)", mode="lines+markers", line=dict(color="#F08A24"),
                    secondary_y=True)
    fig.update_yaxes(title="Rainfall (mm)", secondary_y=False)
    fig.update_yaxes(title="Temperature (C)", secondary_y=True, showgrid=False)
    return fig


def meta_row(ward_id):
    m = c_meta()
    return None if m is None or ward_id not in m.index else m.loc[ward_id]


def page_wards():
    wards = ward_list()
    cities = sorted({w["city"] for w in wards})
    head("Ward Explorer", "Explore estimated dengue risk by ward")
    c1, c2, c3 = st.columns([1, 1.4, 1.4])
    with c1:
        city = city_picker("wx", cities)
    with c3:
        h = horizon_picker("wx")
    data = need(c_all, h, city)
    df = to_df(data)
    ids = df.ward_id.tolist()
    with c2:
        sel = st.selectbox("Search or select a ward", ids, index=ids.index(ss.ward) if ss.ward in ids else 0,
                           key=f"wsel_{city}_{h}")
    ss.ward = sel
    meta, meta_notes = ward_csv_state()
    left, right = st.columns([3, 2])
    with left, card("rank"):
        ctitle("Ward ranking", f"{len(df)} wards, {h}-week horizon, highest tier then estimated cases first")
        head_cells = ["#", "Ward", "City", "Risk tier", "Est. cases"] + (["Density /km2", "Waterlogging", "Sanitation"] if meta is not None else [])
        body = ""
        for i, r in df.iterrows():
            extra = ""
            if meta is not None and r.ward_id in meta.index:
                m = meta.loc[r.ward_id]
                extra = (f'<td>{int(m.population_density_sqkm):,}</td><td>{m.water_logging_risk_index:.2f}</td>'
                         f'<td>{m.sanitation_score:.2f}</td>')
            body += (f'<tr class="{"sel" if r.ward_id == sel else ""}"><td>{i + 1}</td><td><b>{esc(r.ward_id)}</b></td>'
                     f'<td>{esc(r.city)}</td><td>{badge(r.tier)}</td><td>{cases(r.cases)}</td>{extra}</tr>')
        md('<div class="tbl-wrap"><table class="tbl"><tr>' + "".join(f"<th>{c}</th>" for c in head_cells) + f"</tr>{body}</table></div>")
        csv_notice("Ward vulnerability values (served by no API endpoint)", meta is not None, meta_notes, "wx",
                   "<code>DENGUEWATCH_WARDS_CSV</code> (full path of wards.csv)")
    with right:
        with card("legend"):
            ctitle("Risk legend and ward grid", "Each tile is one ward. This is a grid, not a geographic map.")
            md('<div class="legend">' + "".join(f'<span style="--c:{TIER_COLOR[t]}">{t}</span>' for t in reversed(TIERS)) + "</div>")
            for c in sorted(df.city.unique()):
                sub = df[df.city == c].sort_values("ward_id")
                md(f'<div class="card-sub"><b>{esc(c)}</b></div><div class="tiles">' + "".join(
                    f'<div class="tile {r.tier} {"sel" if r.ward_id == sel else ""}" title="{esc(r.ward_id)}: {r.tier}">'
                    f'{esc(r.ward_id.split("_")[-1])}</div>' for r in sub.itertuples()) + "</div>")
        with card("selected"):
            r = df[df.ward_id == sel].iloc[0]
            ctitle(sel, f"{r.city} - {h}-week outlook")
            md(f'{badge(r.tier)} &nbsp; <b>{cases(r.cases)}</b> estimated cases')
            m = meta_row(sel)
            if m is not None:
                bars([("Waterlogging risk", m.water_logging_risk_index, 1, None), ("Sanitation score", m.sanitation_score, 1, None),
                      ("Population density", m.population_density_sqkm, c_meta().population_density_sqkm.max(), None)],
                     fmt=lambda v: f"{v:,.0f}" if v > 5 else f"{v:.2f}")
            st.button("Open prediction details", key="wx_open", type="primary", on_click=go_to, args=("Prediction Details", sel))
    note("Ward boundaries or coordinates are not part of the dataset, so this page ranks wards instead of drawing a map.")


def gauge_svg(tier):
    cx, cy, r = 110, 110, 88
    arcs = ""
    for i, t in enumerate(TIERS):
        a0, a1 = math.pi * (1 - i / 4), math.pi * (1 - (i + 1) / 4)
        p = lambda a: (cx + r * math.cos(a), cy - r * math.sin(a))
        (x0, y0), (x1, y1) = p(a0), p(a1)
        arcs += (f'<path d="M{x0:.1f},{y0:.1f} A{r},{r} 0 0 1 {x1:.1f},{y1:.1f}" stroke="{TIER_COLOR[t]}" stroke-width="22" '
                 f'fill="none" opacity="{1 if t == tier else 0.2}"/>')
    return (f'<svg class="gauge" viewBox="0 0 220 135" role="img" aria-label="Predicted risk tier {esc(tier)}">{arcs}'
            f'<text x="110" y="100" text-anchor="middle" font-size="28" font-weight="800" fill="{TIER_COLOR[tier]}">{esc(tier)}</text>'
            f'<text x="110" y="124" text-anchor="middle" font-size="13" fill="#5A7266">predicted risk tier</text></svg>')


def page_prediction():
    wards = ward_list()
    ids = [w["ward_id"] for w in wards]
    city_of = {w["ward_id"]: w["city"] for w in wards}
    c1, c2, c3 = st.columns([1.6, 1.6, 0.8])
    with c1:
        ward = st.selectbox("Ward", ids, index=ids.index(ss.ward) if ss.ward in ids else 0,
                            format_func=lambda x: f"{x} - {city_of[x]}", key="pd_ward")
    with c2:
        h = horizon_picker("pd")
    with c3:
        md('<div style="height:1.7rem"></div>')
        st.button("Back to Overview", key="pd_back", icon=":material/arrow_back:", on_click=go_to, args=("Overview",))
    ss.ward = ward
    out = need(c_outlook, ward)
    by_h = {x["horizon_weeks"]: x for x in out["horizons"]}
    p = by_h[h]
    head(f"Prediction Details - {ward}", "Dengue risk estimate for the selected ward and horizon")
    md('<div class="ctx">' + "".join(f"<span>{esc(s)}</span>" for s in [
        f'Ward: {ward}', f'City: {p["city"]}', f'Horizon: {h} weeks',
        f'Weather through week of {fmt_date(p["origin_week"])}', f'Target week of {fmt_date(p["forecast_week"])}']) + "</div>")

    a, b, c = st.columns([1, 1.05, 1.25])
    with a, card("risk"):
        ctitle("Predicted risk")
        md(gauge_svg(p["risk_tier"]))
        md(f'<div style="text-align:center"><div class="big">{cases(p["expected_cases"])}</div>'
           '<div class="card-sub">estimated cases in the target week (model estimate, not confirmed cases)</div></div>')
    with b, card("probs"):
        ctitle("Tier probabilities", "Model probability per tier. Not a confidence interval.")
        bars([(t, p["probabilities"].get(t, 0.0), 1, TIER_COLOR[t]) for t in TIERS], fmt=pct)
        ctitle("Forecast summary")
        md('<table class="tbl" style="min-width:0"><tr><th>Horizon</th><th>Risk tier</th><th>Est. cases</th></tr>' + "".join(
            f'<tr class="{"sel" if x == h else ""}"><td>{x} weeks</td><td>{badge(by_h[x]["risk_tier"])}</td>'
            f'<td>{cases(by_h[x]["expected_cases"])}</td></tr>' for x in HORIZONS if x in by_h) + "</table>")
    with c, card("drivers"):
        ctitle("Model driver breakdown", "Change in estimated cases when that factor is neutralised in the model")
        drv = p.get("drivers") or []
        if drv:
            mx = max(abs(d["change_in_expected_cases"]) for d in drv) or 1
            bars([(d["driver"], d["change_in_expected_cases"], mx, "#087A50" if d["change_in_expected_cases"] >= 0 else "#7A9BB0")
                  for d in drv], fmt=lambda v: f"{v:+.2f}")
            top = max(drv, key=lambda d: abs(d["change_in_expected_cases"]))
            verb = "raises" if top["change_in_expected_cases"] >= 0 else "lowers"
            note(f'<b>{esc(top["driver"])}</b> has the largest modelled effect: it {verb} this ward\'s estimate by '
                 f'{abs(top["change_in_expected_cases"]):.2f} cases. Values are case counts, not confidence percentages.')
        else:
            note("The backend returned no driver breakdown for this ward.", "warn")
        m = meta_row(ward)
        ctitle("Ward vulnerability inputs")
        meta_df, meta_notes = ward_csv_state()
        if m is None and meta_df is not None:
            note(f"{esc(ward)} is not listed in the ward CSV, so its vulnerability inputs cannot be shown.", "warn")
        elif m is None:
            csv_notice("Ward vulnerability values", False, meta_notes, "pd",
                       "<code>DENGUEWATCH_WARDS_CSV</code> (full path of wards.csv)")
        else:
            bars([("Waterlogging risk", m.water_logging_risk_index, 1, None), ("Sanitation score", m.sanitation_score, 1, None)],
                 fmt=lambda v: f"{v:.2f}")
            md(f'<div class="card-sub">Population density: <b>{int(m.population_density_sqkm):,}</b> per km2</div>')

    a, b = st.columns([1.2, 1])
    with a, card("recent"):
        ctitle("Recent weekly weather", "Weeks starting Monday. Past weeks feed the model's multi-week lag features.")
        rw = pd.DataFrame(p.get("recent_weather") or [])
        if rw.empty:
            note("No recent weather returned.", "warn")
        else:
            tabs = st.tabs(["Rainfall", "Temperature", "Humidity"])
            specs = [("rainfall_mm", "Rainfall (mm)", "#7FB7E6", True), ("avg_temp_c", "Avg temp (C)", "#F08A24", False),
                     ("relative_humidity_pct", "Humidity (%)", "#087A50", False)]
            for tab, (col, lab, colr, isbar) in zip(tabs, specs):
                with tab:
                    fig = go.Figure(go.Bar(x=rw.date, y=rw[col], marker_color=colr) if isbar
                                    else go.Scatter(x=rw.date, y=rw[col], mode="lines+markers", line=dict(color=colr)))
                    fig.update_yaxes(title=lab)
                    show(style(fig, 250))
    with b, card("wx"):
        w = p.get("weather_context") or {}
        cw = w.get("current_week_weather") or {}
        ctitle(f'Weather - {p["city"]}', "City-level Open-Meteo data, not ward-specific")
        if cw:
            md('<div class="wx">' + "".join(f"<div><small>{l}</small><b>{v}</b></div>" for l, v in [
                ("Avg temp (current week)", f'{cw["avg_temp_c"]:.1f} C'), ("Humidity (current week)", f'{cw["relative_humidity_pct"]:.0f}%'),
                ("Rain (current week)", f'{cw["rainfall_mm"]:.1f} mm')]) + "</div>")
            md(f'<div class="card-sub">Weekly aggregate for the week of {fmt_date(cw.get("date", ""))}; '
               f'{esc(fc_days_text(cw))}.</div>')
        daily = w.get("daily_forecast") or []
        if daily:
            show(style(forecast_fig(daily), 230, True))
        fc = p.get("forecast_coverage") or {}
        if fc:
            covered = fc.get("target_week_covered_by_weather_forecast")
            note(f'{esc(fc.get("statement", ""))} Target week {esc(fc.get("target_week", ""))}: '
                 f'{"covered" if covered else "not covered"} by the weather forecast'
                 f'{" (forecast ends " + fmt_date(fc["weather_forecast_through"]) + ")" if fc.get("weather_forecast_through") else ""}.')
    with st.expander("How to read this prediction"):
        st.write(p.get("note", ""))
        st.write("Tier probabilities describe how the model splits likelihood across risk tiers. They are not confidence "
                 "intervals. Historical accuracy is on the Model Performance page.")


def fmt_short(s):
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").strftime("%d %b")
    except ValueError:
        return str(s)


def city_weeks(rows):
    """/weather repeats the same city-level values on every ward row. Collapse to one row per week and report whether
    all wards of the city really did share identical values."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df, True
    cols = ["rainfall_mm", "avg_temp_c", "max_temp_c", "min_temp_c", "relative_humidity_pct"]
    same = bool((df.groupby("date")[cols].nunique(dropna=False).max() <= 1).all())
    return df.groupby("date", as_index=False)[cols].first().sort_values("date").reset_index(drop=True), same


def week_forecast_days(week_start, daily):
    """Daily forecast rows (today onward, per the backend) that fall inside the Monday-Sunday week."""
    ws = pd.Timestamp(week_start)
    we = ws + pd.Timedelta(days=6)
    return [d for d in daily if ws <= pd.Timestamp(d["date"]) <= we and d.get("is_forecast", True)]


def weekly_fig(weeks, series, n_forecast_days):
    """series: (column, name, colour, light colour, kind). The last point is the current week; when it contains
    forecast days it is drawn lighter (bars) or as an open marker (lines)."""
    x = [f"Wk {fmt_short(d)}" for d in weeks.date]
    last = len(x) - 1
    fig = go.Figure()
    for col, name, colr, light, kind in series:
        if kind == "bar":
            fig.add_bar(x=x, y=weeks[col], name=name,
                        marker_color=[light if (i == last and n_forecast_days) else colr for i in range(len(x))])
        else:
            fig.add_scatter(x=x, y=weeks[col], name=name, mode="lines+markers", line=dict(color=colr),
                            marker=dict(symbol=["circle-open" if (i == last and n_forecast_days) else "circle" for i in range(len(x))],
                                        size=9))
    return fig


def page_weather():
    wards = ward_list()
    cities = sorted({w["city"] for w in wards})
    head("Weather Outlook", "City-level Open-Meteo weather: recent weeks, the current week and the provider's daily forecast")
    default = ss.city if ss.city in cities else ("Dhaka North" if "Dhaka North" in cities else cities[0])
    c1, c2 = st.columns([1.4, 0.7])
    with c1:
        city = st.selectbox("City", cities, index=cities.index(default), key="wo_city")
    with c2:
        md('<div style="height:1.7rem"></div>')
        st.button("Refresh", key="wo_refresh", icon=":material/refresh:", on_click=refresh)
    wx = need(c_weather, city)
    fc = need(c_wforecast, city)
    meta, daily = wx["weather"], fc.get("daily_forecast") or []
    weeks, same = city_weeks(wx["recent_live_weekly_weather"])
    if weeks.empty:
        note("The backend returned no recent weekly weather for this city.", "warn")
        st.stop()
    cur_date = meta.get("live_weather_week")
    cw = (weeks[weeks.date == cur_date].iloc[-1] if cur_date in set(weeks.date) else weeks.iloc[-1])
    n_f = len(week_forecast_days(cw["date"], daily)) if daily else None
    span = f'{fmt_short(cw["date"])} to {fmt_short((pd.Timestamp(cw["date"]) + pd.Timedelta(days=6)).strftime("%Y-%m-%d"))}'
    stale = bool(meta.get("stale") or fc.get("stale_cache"))
    md(status_pill() + " " + (weather_stale_pill({"stale": True}) if stale else "") +
       f' <span class="card-sub">Fetched {esc(fmt_fetched(fc.get("fetched_at_utc", "")))} - '
       f'source: {esc(fc.get("source") or meta.get("source", "Open-Meteo"))}</span>')
    if stale:
        note("The weather provider could not be reached on the last refresh, so the backend is serving its previously "
             "fetched real response. Values may be older than the fetch time suggests.", "warn")
    if not same:
        note("Wards in this city did not all share identical weather values; the first ward's values are shown.", "warn")

    comp = ("composition unknown (no daily forecast returned)" if n_f is None else
            f"{7 - n_f} past day{'s' if 7 - n_f != 1 else ''} + {n_f} forecast day{'s' if n_f != 1 else ''}")
    kpis([(f"Rainfall, week of {fmt_short(cw['date'])}", f"{cw.rainfall_mm:.1f} mm", comp, ""),
          ("Avg temperature, current week", f"{cw.avg_temp_c:.1f} C", f"Weekly mean; max {cw.max_temp_c:.1f}, min {cw.min_temp_c:.1f}", ""),
          ("Avg humidity, current week", f"{cw.relative_humidity_pct:.0f}%", "Weekly mean of daily humidity", ""),
          ("Daily forecast", f"{len(daily)} days", (f"{fmt_short(daily[0]['date'])} to {fmt_short(daily[-1]['date'])}" if daily else "none returned"), "")])
    if n_f:
        note(f"<b>The current week ({esc(span)}) is a mix, not purely observed weather.</b> {7 - n_f} day(s) come from the "
             f"provider's archive/past-days data and {n_f} day(s) (today onward) are forecast values. The backend uses this "
             "forecast-completed week for the model's lag features. Day counts are derived from the daily forecast list below.")
    elif n_f == 0:
        note(f"The current week ({esc(span)}) contains no forecast days; all 7 days are past data.")

    with card("wo_weekly"):
        ctitle("Recent weekly weather: past weeks and the current week",
               f"Monday-start weeks since the dengue dataset ends ({fmt_date(meta.get('historical_dataset_through', ''))}). "
               f"The API returns the latest {len(weeks)} weeks. Weather only, no dengue cases.")
        tabs = st.tabs(["Rainfall", "Temperature", "Humidity"])
        specs = [
            [("rainfall_mm", "Rainfall (mm)", "#7FB7E6", "#BFDDF3", "bar")],
            [("avg_temp_c", "Avg temp", "#F08A24", "#F08A24", "line"), ("max_temp_c", "Max temp", "#D64545", "#D64545", "line"),
             ("min_temp_c", "Min temp", "#4F8DB8", "#4F8DB8", "line")],
            [("relative_humidity_pct", "Humidity (%)", "#087A50", "#8CCDB0", "bar")]]
        ylab = ["Rainfall (mm)", "Temperature (C)", "Humidity (%)"]
        for tab, ser, yl in zip(tabs, specs, ylab):
            with tab:
                fig = weekly_fig(weeks, ser, n_f)
                fig.update_yaxes(title=yl)
                show(style(fig, 270, len(ser) > 1))
        st.caption("The last point is the current week." +
                   (f" It is drawn lighter (bars) or as an open marker (lines) because it includes {n_f} forecast day(s)." if n_f else ""))

    with card("wo_daily"):
        if not daily:
            ctitle("Daily forecast")
            note("The backend returned no daily forecast for this city.", "warn")
        else:
            ctitle("Daily forecast from the weather provider",
                   f"{len(daily)} days, {fmt_date(daily[0]['date'])} through {fmt_date(daily[-1]['date'])}. "
                   f"Provider nominal range: {fc.get('provider_forecast_days', '?')} days. Forecast values, not observations.")
            show(style(forecast_fig(daily), 300, True))
            rows = ""
            for d in daily:
                wk = (pd.Timestamp(d["date"]) - pd.Timedelta(days=pd.Timestamp(d["date"]).dayofweek)).strftime("%Y-%m-%d")
                rows += (f'<tr class="{"sel" if wk == cw["date"] else ""}"><td>{esc(fmt_date(d["date"]))}</td>'
                         f'<td>{"Forecast" if d.get("is_forecast", True) else "Past"}</td><td>{esc(fmt_short(wk))}</td>'
                         f'<td>{d["min_temp_c"]:.1f}</td><td>{d["max_temp_c"]:.1f}</td><td>{d["avg_temp_c"]:.1f}</td>'
                         f'<td>{d["rainfall_mm"]:.1f}</td><td>{d["relative_humidity_pct"]:.0f}</td></tr>')
            md('<div class="tbl-wrap"><table class="tbl"><tr><th>Date</th><th>Type</th><th>Week of</th><th>Min C</th><th>Max C</th>'
               f'<th>Avg C</th><th>Rain mm</th><th>Humidity %</th></tr>{rows}</table></div>')
            st.caption("Highlighted rows belong to the current week. Rows are daily values; the weekly charts above sum or average them.")

    with card("wo_limits"):
        ctitle("Source, coverage and limitations")
        last = fmt_date(daily[-1]["date"]) if daily else "n/a"
        st.markdown(
            f"- **Source:** {fc.get('source') or meta.get('source', 'Open-Meteo')}. Fetched {fmt_fetched(fc.get('fetched_at_utc', ''))}. "
            f"Status: {'serving a cached response because the provider was unreachable' if stale else 'fresh response'}.\n"
            f"- **Forecast coverage:** {len(daily)} daily forecast days (through {last}). Dengue predictions look 2 to 4 weeks ahead, "
            "so their target weeks are beyond this forecast; the Prediction Details page states coverage for each ward.\n"
            f"- **City-level proxy:** one Open-Meteo location per city. Every ward in {city} receives these same values.\n"
            f"- **Not on this page:** dengue case history (the dataset ends {fmt_date(meta.get('historical_dataset_through', ''))}) "
            "and past forecast snapshots, which the backend does not store.\n"
            "- **Weekly aggregates:** weeks start on Monday; the current week may include forecast days, as flagged above.")
        st.button("Open prediction details", key="wo_pd", on_click=go_to, args=("Prediction Details",))


def page_chat():
    wards = ward_list()
    ids = [w["ward_id"] for w in wards]
    head("AI Assistant", "Ask about dengue risk, weather and predictions. Answers are grounded in backend model output.")
    st_, err = load(c_stats)
    if err:
        note(f"<b>Cannot check the assistant.</b> {esc(err.message)}", "err")
        st.button("Try again", key="chat_retry", on_click=refresh)
        st.stop()
    ok = bool(st_.get("configured"))
    if not ok:
        note("<b>The AI assistant is not enabled.</b> The backend has no chatbot API key configured. "
             "Add the key to the backend's <code>.env</code> and restart it. Predictions and dashboards work without it.", "warn")
    c1, c2, c3 = st.columns([1.6, 1.2, 0.8])
    with c1:
        ward = st.selectbox("Ward (optional)", [None] + ids, format_func=lambda x: "Detect from my question" if x is None else x,
                            key="chat_ward", disabled=not ok)
    with c2:
        hz = st.selectbox("Horizon (optional)", [None, 2, 3, 4], format_func=lambda x: "Detect from my question" if x is None else f"{x} weeks",
                          key="chat_hz", disabled=not ok)
    with c3:
        md('<div style="height:1.7rem"></div>')
        st.button("Clear chat", key="chat_clear", on_click=lambda: ss.update(chat=[]), disabled=not ss.chat)
    if ok and not ss.chat:
        st.caption("Suggested questions")
        for i, q in enumerate(SUGGESTIONS):
            st.button(q, key=f"sug_{i}", on_click=lambda q=q: ss.update(pending=q))
    for m in ss.chat:
        if m["role"] == "error":
            note(esc(m["content"]), "err")
        else:
            with st.chat_message(m["role"], avatar=":material/shield:" if m["role"] == "assistant" else None):
                st.write(m["content"])
    q = st.chat_input("Type your question...", max_chars=300, disabled=not ok) or ss.pop("pending", None)
    if q and ok:
        ss.chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant", avatar=":material/shield:"):
            with st.spinner("Checking the latest model output..."):
                try:
                    r = api.chat(q, ss.sid, ward, hz)
                    ss.chat.append({"role": "assistant", "content": r["reply"]})
                    st.write(r["reply"])
                except api.ApiError as e:
                    msg = {429: "You are sending messages too quickly. Wait a minute and try again.",
                           503: f"The assistant is unavailable: {e.message}",
                           404: f"The backend did not recognise that ward or city: {e.message}"}.get(
                        e.status, e.message if e.kind != "http" or e.status != 500 else
                        "The backend could not get an answer from the AI provider (HTTP 500). Try again shortly.")
                    ss.chat.append({"role": "error", "content": msg})
                    note(esc(msg), "err")
        if ss.chat[-1]["role"] == "assistant":
            st.caption("AI-generated from model estimates. Not medical advice and not confirmed case counts.")


def page_performance():
    head("Model Performance & Methodology", "Historical validation of the saved models on a chronological holdout")
    m = need(c_metrics)
    hz = m["horizons"]
    h = horizon_picker("mp")
    r = hz.get(str(h))
    if not r:
        note("No validation metrics for this horizon in the training report.", "warn")
        st.stop()
    note("<b>Historical validation, not prediction confidence.</b> These metrics measure how the model performed on past "
         f'weeks ({fmt_date(r["temporal_test_start"])} to {fmt_date(r["temporal_test_end"])}). They do not describe how certain '
         "any single current prediction is.")
    kpis([("Tier accuracy", f'{r["tier_accuracy"] * 100:.1f}%', f'{r["test_rows"]:,} ward-weeks tested', "Low"),
          ("Macro F1", f'{r["tier_macro_f1"]:.3f}', "Average across the four tiers", "Low"),
          ("MAE (cases)", f'{r["cases_mae"]:.2f}', f'Train-mean baseline {r["baseline_mean_mae"]:.2f}', "Medium"),
          ("RMSE (cases)", f'{r["cases_rmse"]:.2f}', "Penalises large misses", "Medium"),
          ("Brier score", f'{r["brier_score"]:.3f}', "Lower is better", "Medium")])
    a, b = st.columns(2)
    with a, card("recall"):
        ctitle("Per-tier recall", f"Share of actual ward-weeks in each tier that the {h}-week model caught")
        rec = r["tier_recall"]
        fig = go.Figure(go.Bar(x=TIERS, y=[rec.get(t, 0) * 100 for t in TIERS], marker_color=[TIER_COLOR[t] for t in TIERS],
                               text=[f'{rec.get(t, 0) * 100:.1f}%' for t in TIERS], textposition="outside"))
        fig.update_yaxes(range=[0, 115], title="Recall (%)")
        show(style(fig))
        low = min(TIERS, key=lambda t: rec.get(t, 0))
        st.caption(f"Weakest tier in this horizon: {low} ({rec.get(low, 0) * 100:.1f}% recall). Treat that tier's calls with extra care.")
    with b, card("hcomp"):
        ctitle("Horizon comparison", "Accuracy and macro F1 by horizon")
        xs = [f"{x} weeks" for x in HORIZONS if str(x) in hz]
        fig = go.Figure()
        fig.add_bar(x=xs, y=[hz[str(x)]["tier_accuracy"] for x in HORIZONS if str(x) in hz], name="Tier accuracy", marker_color="#087A50")
        fig.add_bar(x=xs, y=[hz[str(x)]["tier_macro_f1"] for x in HORIZONS if str(x) in hz], name="Macro F1", marker_color="#9ED8BA")
        fig.update_layout(barmode="group")
        fig.update_yaxes(range=[0, 1])
        show(style(fig, legend=True))
    a, b = st.columns(2)
    with a, card("err"):
        ctitle("Case-count error by horizon", "MAE and RMSE in cases per ward-week, with the train-mean baseline MAE")
        ks = [x for x in HORIZONS if str(x) in hz]
        fig = go.Figure()
        for key, name, col in [("cases_mae", "MAE", "#087A50"), ("cases_rmse", "RMSE", "#F08A24"), ("baseline_mean_mae", "Baseline MAE", "#B7C7BD")]:
            fig.add_bar(x=[f"{x} weeks" for x in ks], y=[hz[str(x)][key] for x in ks], name=name, marker_color=col)
        fig.update_layout(barmode="group")
        show(style(fig, legend=True))
    with b, card("fwd"):
        ctitle("With vs without next-week forecast features", f"{h}-week model, same holdout")
        w = r.get("without_forecast_features")
        if w:
            rows = [("Tier accuracy", "tier_accuracy", "{:.3f}"), ("Macro F1", "tier_macro_f1", "{:.3f}"), ("MAE", "cases_mae", "{:.3f}"),
                    ("RMSE", "cases_rmse", "{:.3f}"), ("Brier score", "brier_score", "{:.3f}")]
            md('<div class="tbl-wrap"><table class="tbl" style="min-width:0"><tr><th>Metric</th><th>With forecast</th><th>Without</th></tr>' +
               "".join(f"<tr><td>{n}</td><td>{f.format(r[k])}</td><td>{f.format(w[k])}</td></tr>" for n, k, f in rows) + "</table></div>")
            st.caption("The training report notes the forecast feature was trained on simulated forecast error, so its benefit should be re-checked with archived real forecasts.")
        else:
            note("No comparison available.", "warn")
    with card("dataset"):
        ctitle("Dataset and methodology")
        md(f'<div class="kpi-grid"><div class="kpi"><div class="kpi-label">Training data</div><div class="kpi-value">{m["rows"]:,}</div>'
           f'<div class="kpi-sub">ward-week rows</div></div><div class="kpi"><div class="kpi-label">Coverage</div><div class="kpi-value">{m["wards"]} wards</div>'
           f'<div class="kpi-sub">{fmt_date(m["date_start"])} to {fmt_date(m["date_end"])}</div></div>'
           f'<div class="kpi"><div class="kpi-label">Features</div><div class="kpi-value">{m["feature_count"]}</div>'
           f'<div class="kpi-sub">{esc(m["feature_version"])}</div></div></div>')
        st.markdown(
            f"- **Chronological split.** For the {h}-week model, training ends {fmt_date(r['temporal_train_end'])} and testing starts "
            f"{fmt_date(r['temporal_test_start'])}. There is no random shuffling, so the test never sees the future.\n"
            "- **Weather lags.** Rainfall, temperature and humidity are turned into multi-week lag and rolling features per ward, "
            "matching the 2-4 week breeding and incubation delay.\n"
            "- **Forecast proxy.** " + m.get("forecast_proxy", "") + ".\n"
            "- **Risk tiers.** Low, Medium, High and Critical are derived from weekly case counts per ward.")


def page_about():
    head("About DengueWatch", "Building a safer, healthier Bangladesh")
    a, b = st.columns([1.4, 1])
    with a, card("about"):
        ctitle("Purpose")
        st.write("PROHORI | DengueWatch gives public health officers an early-warning view of dengue risk by ward. Heavy rain "
                 "does not cause cases overnight: mosquito breeding, maturation and virus incubation create a delay of roughly "
                 "2 to 4 weeks. The platform uses that delay to flag wards before outbreaks peak.")
        ctitle("How it works")
        st.markdown("1. **Weather in.** Open-Meteo rainfall, temperature and humidity are turned into multi-week lag features per ward.\n"
                    "2. **Model.** Saved machine-learning models estimate weekly cases and risk-tier probabilities for 2, 3 and 4 weeks ahead.\n"
                    "3. **Explain.** Each ward shows a driver breakdown, weather context and the model's historical validation.")
    with b, card("limits"):
        ctitle("Important limitations")
        st.markdown("- Predictions are model estimates, not confirmed dengue cases. No live case feed is connected.\n"
                    "- Weather is city level, so wards in one city share weather inputs.\n"
                    "- Weather forecasts reach about 16 days, so 3-4 week outlooks are not driven by forecast weather.\n"
                    "- Tier probabilities are not confidence intervals.\n"
                    "- Ward boundaries are not in the dataset, so there is no map.\n"
                    "- Not a substitute for official surveillance.")
    with card("resources"):
        ctitle("Resources")
        r1, r2, r3, _ = st.columns([1, 1, 1, 1])
        with r1:
            st.link_button("Backend API docs", f"{api.BASE_URL}/docs")
        with r2:
            st.link_button("Backend health", f"{api.BASE_URL}/health")
        with r3:
            st.link_button("Open-Meteo", "https://open-meteo.com/")
        st.button("View model performance", key="about_perf", on_click=go_to, args=("Model Performance",))


{"Home": page_home, "Overview": page_overview, "Ward Explorer": page_wards, "Weather Outlook": page_weather, "Prediction Details": page_prediction,
 "AI Assistant": page_chat, "Model Performance": page_performance, "About & Resources": page_about}[ss.page]()
