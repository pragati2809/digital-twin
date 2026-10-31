"""
Digital Twin- Bearing Health Digital Twin (CWRU vibration data)
==============================================================
Pages:
  1. Live Monitor      - live KPIs, diagnosis and trends (InfluxDB), offline replay fallback
  2. Model Analytics   - champion model metrics, confusion matrix, ROC/PR, evaluation protocol
  3. Fault Simulator   - pick a real CWRU window, add noise, see waveform/spectra/prediction
  4. Data Explorer     - recordings, window table, per-class feature distributions
  5. System Health     - service checks, alert log, downloadable reports
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from influxdb_client import InfluxDBClient
    HAS_INFLUX = True
except ImportError:
    HAS_INFLUX = False

from src.data import (CLASS_LABELS, CLASS_NAMES, WINDOW_SIZE, get_window, load_all_recordings,
                      recording_summary)
from src.features import DEFECT_ORDERS, FEATURE_NAMES, TARGET_FS, extract_features, features_to_frame
from src.replay_publisher import DEMO_SCENARIO, build_payload, load_window_table, window_stream

st.set_page_config(
    page_title="Digital Twin- Bearing Health Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
  code, pre, .mono            { font-family: 'JetBrains Mono', monospace; }

  /* ── App Background ───────────────────────────────────── */
  .stApp { background: #080c14; color: #c9d1e0; }
     .block-container { padding-top: 4.5rem !important; }

  /* ── Hide default Streamlit chrome ───────────────────── */
  #MainMenu, footer { visibility: hidden; }
  .stDeployButton  { display: none; }

  /* ── Scrollbar ──────────────────────────────────────────  */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #080c14; }
  ::-webkit-scrollbar-thumb { background: #1e3050; border-radius: 4px; }

  /* ── Sidebar ────────────────────────────────────────────  */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1120 0%, #080e1c 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
  }
  section[data-testid="stSidebar"] .stRadio label {
    color: #7a8fa8 !important; font-size: 0.87rem; padding: 3px 0;
    transition: color 0.2s;
  }
  section[data-testid="stSidebar"] .stRadio label:hover { color: #e2e8f0 !important; }

  /* ── Brand header ────────────────────────────────────── */
  .brand-wrap {
    display: flex; align-items: center; gap: 10px;
    padding: 18px 0 14px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 18px;
  }
  .brand-logo {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
    box-shadow: 0 4px 14px rgba(59,130,246,0.4);
  }
  .brand-name { font-size: 1.05rem; font-weight: 800; color: #e2e8f0; line-height: 1.1; }
  .brand-tag  { font-size: 0.65rem; color: #3a5272; letter-spacing: 0.12em; text-transform: uppercase; }
  .brand-ver  { font-size: 0.6rem; color: #2a3f58; }

  /* ── Pulse Status ─────────────────────────────────────── */
  .sys-pill {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 5px 13px; border-radius: 20px;
    font-size: 0.73rem; font-weight: 600;
    margin-bottom: 6px;
  }
  .pill-ok   { background: rgba(16,185,129,0.09); border: 1px solid rgba(16,185,129,0.28); color: #10b981; }
  .pill-warn { background: rgba(245,158,11,0.09); border: 1px solid rgba(245,158,11,0.28); color: #f59e0b; }
  .pill-err  { background: rgba(239,68,68,0.09);  border: 1px solid rgba(239,68,68,0.28);  color: #ef4444; }
  .sys-dot   { width: 7px; height: 7px; border-radius: 50%; }
  .dot-ok    { background: #10b981; animation: blink 2s infinite; }
  .dot-warn  { background: #f59e0b; animation: blink 1.5s infinite; }
  .dot-err   { background: #ef4444; }
  @keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.25;} }

  /* ── Nav label ───────────────────────────────────────── */
  .nav-lbl {
    font-size: 0.65rem; font-weight: 700; color: #2c3f58;
    letter-spacing: 0.12em; text-transform: uppercase;
    margin: 16px 0 5px;
  }

  /* ── Sidebar footer ──────────────────────────────────── */
  .sb-footer {
    font-size: 0.68rem; color: #233448;
    border-top: 1px solid rgba(255,255,255,0.04);
    padding-top: 14px; line-height: 1.9; margin-top: 8px;
  }
  .sb-footer b { color: #2e4a6a; }

  /* ── Connection chips ────────────────────────────────── */
  .conn-chip {
    display: flex; align-items: center; gap: 7px;
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px; padding: 5px 11px;
    font-size: 0.71rem; color: #405570; margin-bottom: 5px;
  }
  .cdot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
  .con  { background: #10b981; }
  .idle { background: #f59e0b; }
  .off  { background: #ef4444; }

  /* ── Page topbar ─────────────────────────────────────── */
  .topbar {
    display: flex; align-items: flex-start; justify-content: space-between;
    margin-bottom: 24px;
  }
  .page-title    { font-size: 1.5rem; font-weight: 800; color: #e2e8f0; margin: 0; }
  .page-subtitle { font-size: 0.80rem; color: #445a72; margin-top: 3px; }
  .ts-pill {
    font-size: 0.7rem; color: #304862;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.05); border-radius: 8px;
    padding: 5px 12px; white-space: nowrap; margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── KPI Cards ───────────────────────────────────────── */
  .kpi {
    background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 18px 20px;
    position: relative; overflow: hidden;
    transition: border-color 0.25s, transform 0.15s;
  }
  .kpi:hover { border-color: rgba(59,130,246,0.3); transform: translateY(-1px); }
  .kpi::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 14px 14px 0 0;
  }
  .k-blue::before   { background: linear-gradient(90deg, #3b82f6, #06b6d4); }
  .k-green::before  { background: linear-gradient(90deg, #10b981, #34d399); }
  .k-amber::before  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
  .k-red::before    { background: linear-gradient(90deg, #ef4444, #f87171); }
  .k-purple::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
  .k-cyan::before   { background: linear-gradient(90deg, #06b6d4, #67e8f9); }
  .kpi-ico  { position: absolute; top: 16px; right: 16px; font-size: 1.3rem; opacity: 0.35; }
  .kpi-lbl  { font-size: 0.69rem; font-weight: 700; color: #3a5272;
               letter-spacing: 0.09em; text-transform: uppercase; margin-bottom: 7px; }
  .kpi-val  { font-size: 1.65rem; font-weight: 800; color: #e2e8f0; line-height: 1; }
  .kpi-unit { font-size: 0.82rem; font-weight: 400; color: #506888; margin-left: 3px; }
  .kpi-sub  { font-size: 0.70rem; color: #2e4460; margin-top: 5px; }
  .kpi-delta-up   { font-size: 0.70rem; color: #34d399; margin-top: 4px; }
  .kpi-delta-down { font-size: 0.70rem; color: #f87171; margin-top: 4px; }

  /* ── Status Badge ─────────────────────────────────────── */
  .badge {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 7px 18px; border-radius: 999px;
    font-size: 0.82rem; font-weight: 700; letter-spacing: 0.06em;
  }
  .b-normal   { background: rgba(16,185,129,0.13); color: #10b981; border: 1px solid rgba(16,185,129,0.32); }
  .b-warning  { background: rgba(245,158,11,0.13); color: #f59e0b; border: 1px solid rgba(245,158,11,0.32); }
  .b-critical {
    background: rgba(239,68,68,0.13); color: #ef4444; border: 1px solid rgba(239,68,68,0.4);
    animation: pulse-crit 1.8s infinite;
  }
  @keyframes pulse-crit {
    0%,100%{ box-shadow: 0 0 0 0 rgba(239,68,68,0.35); }
    50%    { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
  }

  /* ── Risk Gauge Bar ──────────────────────────────────── */
  .risk-wrap   { background: rgba(255,255,255,0.04); border-radius: 999px; height: 8px; overflow: hidden; margin: 8px 0; }
  .risk-fill   { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #10b981 0%, #f59e0b 55%, #ef4444 100%); transition: width 0.6s ease; }

  /* ── Section divider & header ────────────────────────── */
  .sec-div  { border: none; border-top: 1px solid rgba(255,255,255,0.05); margin: 24px 0; }
  .sec-hdr  {
    font-size: 0.68rem; font-weight: 700; color: #2c3f58;
    letter-spacing: 0.13em; text-transform: uppercase;
    margin-bottom: 14px; display: flex; align-items: center; gap: 8px;
  }
  .sec-hdr::after { content: ''; flex: 1; height: 1px; background: rgba(255,255,255,0.04); }

  /* ── Alert banners ───────────────────────────────────── */
  .al-ok   { background: rgba(16,185,129,0.06); border-left: 3px solid #10b981; border-radius: 0 10px 10px 0; padding: 11px 15px; color: #6ee7b7; font-size: 0.83rem; margin: 10px 0; }
  .al-warn { background: rgba(245,158,11,0.06); border-left: 3px solid #f59e0b; border-radius: 0 10px 10px 0; padding: 11px 15px; color: #fcd34d; font-size: 0.83rem; margin: 10px 0; }
  .al-crit { background: rgba(239,68,68,0.06);  border-left: 3px solid #ef4444; border-radius: 0 10px 10px 0; padding: 11px 15px; color: #fca5a5; font-size: 0.83rem; margin: 10px 0; }
  .al-info { background: rgba(59,130,246,0.06); border-left: 3px solid #3b82f6; border-radius: 0 10px 10px 0; padding: 11px 15px; color: #93c5fd; font-size: 0.83rem; margin: 10px 0; }

  /* ── Inference result box ────────────────────────────── */
  .infer-box {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 22px; margin-top: 18px;
  }

  /* ── Summary stat box ────────────────────────────────── */
  .stat-box {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 16px 20px; text-align: center;
  }
  .stat-val  { font-size: 2rem; font-weight: 800; color: #e2e8f0; }
  .stat-lbl  { font-size: 0.68rem; color: #3a5272; text-transform: uppercase; letter-spacing: 0.1em; }

  /* ── Sweep result table ──────────────────────────────── */
  .sweep-crit { color: #f87171; font-weight: 700; }
  .sweep-warn { color: #fcd34d; font-weight: 600; }
  .sweep-ok   { color: #34d399; }

  /* ── Footer bar ──────────────────────────────────────── */
  .footer-bar {
    margin-top: 40px; padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.05);
    font-size: 0.68rem; color: #213040; text-align: center; line-height: 1.8;
  }

  /* ── Streamlit overrides ─────────────────────────────── */
  .stDataFrame { border-radius: 12px !important; overflow: hidden; }
  .stSlider > div { color: #7a8fa8; }
  div[data-testid="metric-container"] { background: transparent; }
  .stTabs [data-baseweb="tab"] { color: #506888; font-size: 0.83rem; }
  .stTabs [aria-selected="true"] { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "my-super-secret-auth-token")
INFLUX_ORG = os.getenv("INFLUX_ORG", "digital-twin-org")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "machine_data")

CLASS_COLORS = {"normal": "#10b981", "inner_race": "#ef4444", "ball": "#f59e0b", "outer_race": "#8b5cf6"}


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_bundle():
    p = ROOT / "models" / "best_model.joblib"
    return joblib.load(str(p)) if p.exists() else None


@st.cache_data
def load_metadata():
    p = ROOT / "models" / "metadata.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def load_comparison():
    p = ROOT / "reports" / "model_comparison.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource
def load_recordings():
    return load_all_recordings(str(ROOT / "data" / "cwru"))


@st.cache_data
def load_windows():
    return load_window_table(str(ROOT / "data" / "cwru_windows.csv"))


def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add model outputs (failure_prob, pred_class, p_<class>) to a feature table."""
    bundle = load_bundle()
    proba = bundle["pipeline"].predict_proba(df[FEATURE_NAMES])
    out = df.copy()
    out["failure_prob"] = 1.0 - proba[:, CLASS_NAMES.index("normal")]
    out["pred_class"] = proba.argmax(1)
    for i, c in enumerate(CLASS_NAMES):
        out[f"p_{c}"] = proba[:, i]
    return out


@st.cache_data
def demo_sequence() -> pd.DataFrame:
    """One full loop of the demo scenario on held-out TEST windows, scored by the real model."""
    df = load_windows()
    n = sum(k for _, k in DEMO_SCENARIO)
    stream = window_stream(df, "demo", "test")
    rows = [build_payload(next(stream), i + 1) for i in range(n)]
    seq = pd.DataFrame(rows)
    seq["class_id"] = seq["true_class"]
    out = predict_df(seq)
    out["predicted_failure"] = (out["failure_prob"] >= load_metadata().get("optimal_threshold", 0.5)).astype(int)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# INFLUXDB
# ══════════════════════════════════════════════════════════════════════════════
def _influx_client():
    return InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG, timeout=3000)


def influx_alive() -> bool:
    if not HAS_INFLUX:
        return False
    try:
        with _influx_client() as c:
            return bool(c.ping())
    except Exception:
        return False


def fetch_live_history(minutes: int = 30):
    """Recent 'health' rows (pivoted, oldest to newest) or None when InfluxDB is empty/unreachable."""
    if not HAS_INFLUX:
        return None
    q = (f'from(bucket: "{INFLUX_BUCKET}") |> range(start: -{minutes}m) '
         f'|> filter(fn: (r) => r._measurement == "health") '
         f'|> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")')
    try:
        with _influx_client() as c:
            tables = c.query_api().query(q)
        rows = [rec.values for t in tables for rec in t.records]
        if not rows:
            return None
        df = pd.DataFrame(rows).rename(columns={"_time": "time"}).sort_values("time").reset_index(drop=True)
        return df
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def thresholds():
    meta = load_metadata()
    st_th = meta.get("status_thresholds", {})
    thr = float(meta.get("optimal_threshold", 0.5))
    return float(st_th.get("normal_max", thr * 0.6)), float(st_th.get("warning_max", thr)), thr


def status_classify(prob: float):
    normal_max, warn_max, _ = thresholds()
    if prob < normal_max:
        return "NORMAL", "b-normal", "al-ok", "✅ <b>Bearing healthy</b> - vibration signature matches normal operation."
    if prob < warn_max:
        return "WARNING", "b-warning", "al-warn", "⚠️ <b>Elevated risk</b> - vibration signature is drifting from normal. Monitor closely."
    return "CRITICAL", "b-critical", "al-crit", "🚨 <b>Bearing fault detected</b> - schedule inspection / maintenance."


def render_kpi(col, label, value, unit, sub, color, icon=""):
    with col:
        st.markdown(f"""
        <div class="kpi k-{color}">
          <div class="kpi-ico">{icon}</div>
          <div class="kpi-lbl">{label}</div>
          <div class="kpi-val">{value}<span class="kpi-unit">{unit}</span></div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)


def risk_bar_html(pct: float) -> str:
    return f'<div class="risk-wrap"><div class="risk-fill" style="width:{max(0, min(100, pct))}%;"></div></div>'


def header(title: str, subtitle: str):
    now = datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M:%S UTC")
    st.markdown(f"""
    <div class="topbar"><div><div class="page-title">{title}</div>
    <div class="page-subtitle">{subtitle}</div></div><div class="ts-pill">🕐 {now}</div></div>
    """, unsafe_allow_html=True)


def info(msg: str, kind: str = "al-info"):
    st.markdown(f'<div class="{kind}">{msg}</div>', unsafe_allow_html=True)


def dark_fig(w=7, h=3.2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0d1526")
    ax.set_facecolor("#0d1526")
    for s in ax.spines.values():
        s.set_color("#243247")
    ax.tick_params(colors="#9fb0c8")
    ax.xaxis.label.set_color("#9fb0c8")
    ax.yaxis.label.set_color("#9fb0c8")
    ax.title.set_color("#e2e8f0")
    return fig, ax


def get_live_or_demo():
    """(history_df, is_live). History has: time, failure_prob, pred_class, label, rms, kurtosis, ..."""
    live = fetch_live_history(30)
    if live is not None and len(live):
        if "predicted_failure" not in live.columns:
            live["predicted_failure"] = (live["failure_prob"] >= thresholds()[2]).astype(int)
        return live, True
    seq = demo_sequence()
    pos = int(time.time() // 2) % len(seq)
    idx = [(pos - k) % len(seq) for k in range(min(60, len(seq)))][::-1]
    hist = seq.iloc[idx].reset_index(drop=True).copy()
    hist["time"] = pd.date_range(end=datetime.now(timezone.utc), periods=len(hist), freq="2s")
    return hist, False


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
bundle = load_bundle()
meta = load_metadata()
live_df, live_mode = get_live_or_demo() if bundle is not None else (None, False)

with st.sidebar:
    st.markdown("""
    <div class="brand-wrap"><div class="brand-logo">🏭</div><div>
      <div class="brand-name">Digital Twin</div><div class="brand-tag">Bearing Health Twin</div>
      <div class="brand-ver">v3.0 · CWRU vibration</div></div></div>
    """, unsafe_allow_html=True)

    if bundle is None:
        st.markdown('<div class="sys-pill pill-err"><div class="sys-dot dot-err"></div>Model not found</div>', unsafe_allow_html=True)
    elif live_mode:
        st.markdown('<div class="sys-pill pill-ok"><div class="sys-dot dot-ok"></div>Live stream active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sys-pill pill-warn"><div class="sys-dot dot-warn"></div>Offline replay mode</div>', unsafe_allow_html=True)

    st.markdown('<div class="nav-lbl">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("nav", ["🖥️  Live Monitor", "📊  Model Analytics", "🔮  Fault Simulator",
                            "🗄️  Data Explorer", "📈  System Health"], label_visibility="collapsed")
    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)
    st.markdown('<div class="nav-lbl">Auto-Refresh (Live Monitor)</div>', unsafe_allow_html=True)
    auto_refresh = st.toggle("Enable auto-refresh", value=os.getenv("AUTO_REFRESH", "1") == "1")
    refresh_secs = st.select_slider("Interval (seconds)", options=[2, 5, 10, 15, 30], value=5, disabled=not auto_refresh)
    st.markdown(f"""
    <div class="sb-footer"><b>Digital Twin</b> v3.0<br>
      Model: {meta.get('best_model_name', 'n/a')}<br>
      Data: CWRU bearing, 12 kHz drive-end<br>
      Last refresh: {datetime.now().strftime('%H:%M:%S')}</div>
    """, unsafe_allow_html=True)

if bundle is None:
    st.error("Trained model not found (models/best_model.joblib). Run `python src/train.py` first.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 - LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════
if page.endswith("Live Monitor"):
    header("Live Bearing Monitor", "Vibration windows → 18 features → ML diagnosis · Normal / Inner race / Ball / Outer race")
    if live_mode:
        info("🟢 <b>Live mode</b> - showing data from InfluxDB (MQTT → inference service).")
    else:
        info("🟡 <b>Offline replay</b> - no live data in InfluxDB. Showing real held-out CWRU test windows scored by the trained "
             "model, looping through a demo scenario. Start the stack (<code>docker compose up</code>) for the live stream.", "al-warn")

    last = live_df.iloc[-1]
    prob = float(last["failure_prob"])
    status_label, badge_cls, alert_cls, alert_msg = status_classify(prob)
    pred_name = CLASS_LABELS[CLASS_NAMES[int(last["pred_class"])]]
    true_name = CLASS_LABELS[CLASS_NAMES[int(last["true_class"])]] if "true_class" in last and pd.notna(last.get("true_class")) else "n/a"

    st.markdown(f'<span class="badge {badge_cls}">● &nbsp;{status_label}</span>', unsafe_allow_html=True)
    info(alert_msg, alert_cls)

    k = st.columns(6)
    render_kpi(k[0], "Failure Probability", f"{prob * 100:.1f}", "%", status_label, "cyan", "🎯")
    render_kpi(k[1], "Diagnosis", pred_name.replace(" Fault", ""), "", f"Ground truth: {true_name.replace(' Fault', '')}", "red" if prob >= thresholds()[2] else "green", "🔎")
    render_kpi(k[2], "Vibration RMS", f"{float(last['rms']):.3f}", "g", "Overall vibration level", "blue", "📳")
    render_kpi(k[3], "Kurtosis", f"{float(last['kurtosis']):.2f}", "", "Impulsiveness (normal ≈ 3)", "amber", "⚡")
    render_kpi(k[4], "Crest Factor", f"{float(last['crest_factor']):.2f}", "", "Peak / RMS", "purple", "📈")
    n_alerts = int((live_df["failure_prob"] >= thresholds()[0]).sum())
    render_kpi(k[5], "Alerts (window)", f"{n_alerts}", "", f"of last {len(live_df)} windows", "red", "🚨")

    st.markdown(risk_bar_html(prob / max(thresholds()[2], 1e-6) * 100), unsafe_allow_html=True)
    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">Telemetry & Predictions</div>', unsafe_allow_html=True)

    h = live_df.set_index("time")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Failure probability vs alarm threshold")
        chart = h[["failure_prob"]].copy()
        chart["threshold"] = thresholds()[2]
        st.line_chart(chart, height=220)
    with c2:
        st.caption("Class probabilities")
        cols = [f"p_{c}" for c in CLASS_NAMES if f"p_{c}" in h.columns]
        if cols:
            st.line_chart(h[cols].rename(columns=lambda c: c[2:]), height=220)
    c3, c4 = st.columns(2)
    with c3:
        st.caption("Vibration RMS (g)")
        st.line_chart(h[["rms"]], height=200)
    with c4:
        st.caption("Kurtosis and crest factor")
        st.line_chart(h[["kurtosis", "crest_factor"]], height=200)
    if "label" in h.columns:
        st.caption("Ground truth (1 = faulty) vs model decision")
        st.line_chart(h[["label", "predicted_failure"]], height=140)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 - MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page.endswith("Model Analytics"):
    header("Model Analytics", "Champion model selection, held-out test metrics and evaluation protocol")
    m = meta.get("metrics_test", {})
    m4 = meta.get("metrics_test_4class", {})
    k = st.columns(5)
    render_kpi(k[0], "Champion model", meta.get("best_model_name", "n/a"), "", "Chosen on validation set", "blue", "🏆")
    render_kpi(k[1], "Fault-detection F1", f"{m.get('f1', 0):.3f}", "", f"threshold {meta.get('optimal_threshold', 0):.2f}", "green", "🎯")
    render_kpi(k[2], "Precision / Recall", f"{m.get('precision', 0):.2f} / {m.get('recall', 0):.2f}", "", "fault vs normal", "cyan", "⚖️")
    render_kpi(k[3], "4-class accuracy", f"{m4.get('accuracy', 0) * 100:.1f}", "%", f"macro-F1 {m4.get('macro_f1', 0):.3f}", "purple", "🧩")
    render_kpi(k[4], "ROC-AUC", f"{m.get('roc_auc', 0):.3f}", "", f"PR-AUC {m.get('pr_auc', 0):.3f}", "amber", "📉")

    st.markdown("&nbsp;")
    info("<b>Read these numbers with care.</b> " + meta.get("caveat", ""), "al-warn")
    st.caption("Split protocol: " + meta.get("split_protocol", "n/a"))

    st.markdown('<div class="sec-hdr">Model comparison (held-out test split)</div>', unsafe_allow_html=True)
    comp = load_comparison()
    if comp is not None:
        show = comp[["model", "deployable", "threshold", "val_f1", "test_precision", "test_recall", "test_f1",
                     "test_roc_auc", "test_accuracy_4class", "test_macro_f1_4class"]]
        st.dataframe(show, hide_index=True)
        st.caption("Rule Baseline and Isolation Forest are binary-only reference models; the deployed model is a 4-class classifier.")

    c1, c2 = st.columns(2)
    for col, img in ((c1, "confusion_matrix.png"), (c2, "roc_pr_curves.png")):
        p = ROOT / "reports" / img
        if p.exists():
            col.image(str(p))
    c3, c4 = st.columns(2)
    for col, img in ((c3, "feature_importance.png"), (c4, "class_balance.png")):
        p = ROOT / "reports" / img
        if p.exists():
            col.image(str(p))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 - FAULT SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page.endswith("Fault Simulator"):
    header("Fault Simulator", "Pick a real CWRU vibration window, degrade it with noise, and watch the twin diagnose it")
    recs = load_recordings()
    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        fid = st.selectbox("Recording", list(recs.keys()),
                           format_func=lambda i: f"{i}.mat · {CLASS_LABELS[recs[i].class_name]} · {recs[i].load_hp} HP")
    rec = recs[fid]
    n_win = (rec.n_samples - WINDOW_SIZE) // 512 + 1
    with c2:
        win_idx = st.slider("Window index (time position in recording)", 0, n_win - 1, n_win // 2)
    with c3:
        snr = st.select_slider("Added noise (SNR, dB)", options=["off", 30, 20, 10, 5, 0, -5], value="off")

    window = get_window(rec, win_idx).copy()
    if snr != "off":
        rng = np.random.default_rng(1234)
        window = window + rng.normal(0, np.sqrt(np.mean((window - window.mean()) ** 2)) / (10 ** (float(snr) / 20)), window.size)

    feats = extract_features(window, TARGET_FS, rec.rpm)
    pred = predict_df(pd.DataFrame([feats]))
    prob = float(pred["failure_prob"].iloc[0])
    pcls = CLASS_NAMES[int(pred["pred_class"].iloc[0])]
    status_label, badge_cls, alert_cls, alert_msg = status_classify(prob)

    k = st.columns(4)
    render_kpi(k[0], "Failure probability", f"{prob * 100:.1f}", "%", status_label, "cyan", "🎯")
    render_kpi(k[1], "Diagnosis", CLASS_LABELS[pcls].replace(" Fault", ""), "", f"Actual: {CLASS_LABELS[rec.class_name].replace(' Fault', '')}", "green" if pcls == rec.class_name else "red", "🔎")
    render_kpi(k[2], "RMS", f"{feats['rms']:.3f}", "g", "", "blue", "📳")
    render_kpi(k[3], "Kurtosis", f"{feats['kurtosis']:.2f}", "", "", "amber", "⚡")
    info(alert_msg, alert_cls)
    st.bar_chart(pred[[f"p_{c}" for c in CLASS_NAMES]].rename(columns=lambda c: c[2:]).T, height=160)

    t = np.arange(window.size) / TARGET_FS * 1000
    w1, w2 = st.columns(2)
    with w1:
        st.caption("Waveform (ms vs g)")
        st.line_chart(pd.DataFrame({"acceleration": window - window.mean()}, index=np.round(t, 2)), height=220)
    with w2:
        st.caption("Amplitude spectrum (Hz)")
        freqs = np.fft.rfftfreq(window.size, 1 / TARGET_FS)
        spec = np.abs(np.fft.rfft((window - window.mean()) * np.hanning(window.size))) / window.size
        st.line_chart(pd.DataFrame({"amplitude": spec}, index=np.round(freqs, 1)), height=220)

    from scipy import signal as sps
    from src.features import _ENV_SOS
    env = np.abs(sps.hilbert(sps.sosfiltfilt(_ENV_SOS, window - window.mean())))
    env_spec = np.abs(np.fft.rfft((env - env.mean()) * np.hanning(env.size))) / env.size
    mask = freqs <= 400
    shaft = rec.rpm / 60
    st.caption("Envelope spectrum (0-400 Hz). Bearing defect frequencies for this shaft speed: " +
               ", ".join(f"{k.replace('env_', '').upper()} ≈ {v * shaft:.0f} Hz" for k, v in DEFECT_ORDERS.items()))
    st.line_chart(pd.DataFrame({"envelope amplitude": env_spec[mask]}, index=np.round(freqs[mask], 1)), height=200)
    with st.expander("All 18 extracted features"):
        st.dataframe(pd.DataFrame({"feature": FEATURE_NAMES, "value": [feats[f] for f in FEATURE_NAMES]}), hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 - DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page.endswith("Data Explorer"):
    header("Data Explorer", "CWRU bearing recordings, windowed feature table and class separation")
    recs = load_recordings()
    wins = load_windows()
    st.markdown('<div class="sec-hdr">Recordings</div>', unsafe_allow_html=True)
    st.dataframe(recording_summary(recs), hide_index=True)

    k = st.columns(4)
    for col, (i, c) in zip(k, enumerate(CLASS_NAMES)):
        render_kpi(col, CLASS_LABELS[c], f"{int((wins['class_id'] == i).sum())}", "", "windows", ["green", "red", "amber", "purple"][i], "📦")

    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">Feature distribution by class</div>', unsafe_allow_html=True)
    feat = st.selectbox("Feature", FEATURE_NAMES, index=FEATURE_NAMES.index("kurtosis"))
    fig, ax = dark_fig(8, 3.6)
    data = [wins.loc[wins["class_id"] == i, feat].to_numpy() for i in range(4)]
    bp = ax.boxplot(data, patch_artist=True, tick_labels=[CLASS_LABELS[c].replace(" Fault", "") for c in CLASS_NAMES])
    for patch, c in zip(bp["boxes"], CLASS_NAMES):
        patch.set_facecolor(CLASS_COLORS[c]); patch.set_alpha(0.75)
    for el in ("whiskers", "caps", "medians", "fliers"):
        for line in bp[el]:
            line.set_color("#9fb0c8")
    ax.set_ylabel(feat)
    st.pyplot(fig)

    st.markdown('<div class="sec-hdr">Window table</div>', unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    cls_sel = f1.multiselect("Classes", CLASS_NAMES, default=CLASS_NAMES)
    split_sel = f2.multiselect("Split", ["train", "val", "test", "drop"], default=["train", "val", "test"])
    view = wins[wins["class_name"].isin(cls_sel) & wins["split"].isin(split_sel)]
    st.caption(f"{len(view)} windows")
    st.dataframe(view.head(500), hide_index=True)
    st.download_button("Download filtered windows (CSV)", view.to_csv(index=False), "cwru_windows_filtered.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 - SYSTEM HEALTH
# ══════════════════════════════════════════════════════════════════════════════
else:
    header("System Health", "Service checks, alert history and downloadable reports")
    influx_ok = influx_alive()
    n_mat = len(list((ROOT / "data" / "cwru").glob("*.mat")))
    checks = [
        ("Trained model artifact", (ROOT / "models" / "best_model.joblib").exists()),
        ("Model metadata", (ROOT / "models" / "metadata.json").exists()),
        (f"CWRU recordings ({n_mat} .mat files)", n_mat >= 4),
        (f"InfluxDB reachable ({INFLUX_URL})", influx_ok),
        ("Live telemetry present (last 30 min)", live_mode),
    ]
    for name, ok in checks:
        info(("✅ " if ok else "⚠️ ") + name, "al-ok" if ok else "al-warn")

    st.markdown('<div class="sec-hdr">Alert log (recent windows)</div>', unsafe_allow_html=True)
    log = live_df[live_df["failure_prob"] >= thresholds()[0]].copy()
    if len(log):
        log["diagnosis"] = log["pred_class"].astype(int).map(lambda i: CLASS_LABELS[CLASS_NAMES[i]])
        cols = [c for c in ["time", "diagnosis", "failure_prob", "rms", "kurtosis"] if c in log.columns]
        st.dataframe(log[cols].iloc[::-1].head(100), hide_index=True)
    else:
        st.write("No alerts in the current window.")
    if not live_mode:
        st.caption("Offline replay mode: log shows the demo scenario, not live telemetry.")

    st.markdown('<div class="sec-hdr">Downloads</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    for col, rel, label in ((d1, "reports/model_comparison.csv", "Model comparison (CSV)"),
                            (d2, "reports/metrics.json", "Metrics (JSON)"),
                            (d3, "models/metadata.json", "Model metadata (JSON)")):
        p = ROOT / rel
        if p.exists():
            col.download_button(label, p.read_bytes(), p.name)

st.markdown(f"""
<div class="footer-bar"><b>Digital Twin</b> v3.0 &nbsp;·&nbsp; Digital Twin for Bearing Health Monitoring &nbsp;·&nbsp; HCL Project P_222<br>
Model: {meta.get('best_model_name', 'n/a')} &nbsp;·&nbsp; Data: CWRU Bearing Data Center (drive-end, 12 kHz) &nbsp;·&nbsp; Stack: Streamlit · InfluxDB · MQTT · Docker</div>
""", unsafe_allow_html=True)

# Auto-refresh runs AFTER the page has rendered (only on the Live Monitor page).
if auto_refresh and page.endswith("Live Monitor"):
    time.sleep(refresh_secs)
    st.rerun()
