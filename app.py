"""
IntrinsiQ — McMillin Analytics
Blended Intrinsic Value Calculator
Step 1: Clean working base
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from src.data_fetcher import DataFetcher, invalidate_ticker_cache
from src.models import (
    dcf_valuation, ddm_valuation, ev_ebitda_valuation,
    pe_relative_valuation, regression_valuation, inverse_dcf,
    compute_blended_value, sensitivity_dcf,
    compute_dynamic_wacc, suggest_dcf_assumptions, sanitize_net_debt,
    excess_return_valuation, pbv_relative_valuation, is_financial_sector, is_reit_sector,
    MODEL_LABELS, MODEL_COLORS,
)
from src.visualizations import (
    create_price_history_chart, create_valuation_waterfall,
    create_sensitivity_heatmap, create_dcf_projection_chart,
    create_upside_gauge, create_inverse_dcf_chart,
)
from src.peers import get_peers_for_ticker

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IntrinsiQ | McMillin Analytics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stDataFrame { overflow-x: auto !important; }
@media (max-width: 768px) {
    .block-container { padding: 8px 6px 40px 6px !important; max-width: 100% !important; }
    [data-testid="stMetricValue"] { font-size: 14px !important; line-height: 1.2 !important; }
    [data-testid="stMetricLabel"] { font-size: 10px !important; }
    [data-testid="stMetricDelta"] { font-size: 10px !important; }
    .stTabs [data-baseweb="tab-list"] { overflow-x: auto !important; flex-wrap: nowrap !important; }
    .stTabs [data-baseweb="tab"] { padding: 6px 10px !important; font-size: 11px !important; white-space: nowrap !important; flex-shrink: 0 !important; }
    section[data-testid="stSidebar"] > div { padding: 8px 10px !important; }
    h1 { font-size: 20px !important; } h2 { font-size: 16px !important; } h3 { font-size: 14px !important; }
    .stButton > button { font-size: 12px !important; padding: 6px 8px !important; }
    .stSlider { width: 100% !important; }
    hr { margin: 8px 0 !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
for k, v in [
    ("analysis_done", False),
    ("last_ticker", ""),
    ("metrics", {}),
    ("peer_data", []),
    ("sensitivity_df", None),
    ("disclaimer_ok", False),
    # Slider defaults — only set once, then driven by presets/user
    ("w_dcf", 35),
    ("w_ddm", 10),
    ("w_ev",  20),
    ("w_pe",  20),
    ("w_reg", 15),
    ("active_preset", "⚖️ Balanced"),
    ("wacc_data", {}),
    ("dcf_sugg",  {}),
    # DCF/DDM slider defaults (overridden by dynamic WACC after first analyze)
    ("s_wacc", 9.0),
    ("s_coe",  9.5),
    ("s_g1",   10.0),
    ("s_g2",   6.0),
    ("s_tg",   2.5),
    ("s_dg",   5.0),
    ("s_manual_fcf", 0.0),
    ("s_use_mfcf",   False),
    ("_is_financial", False),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sector weight presets ──────────────────────────────────────────────────
WEIGHT_PRESETS = {
    # Balanced: good starting point for unknown companies
    "⚖️ Balanced":            dict(dcf=35, ddm=10, ev=20, pe=20, reg=15),

    # Tech/Growth: DCF-heavy, FCF is the story; no DDM (AAPL/MSFT/NVDA/GOOGL)
    # EV/EBITDA less relevant for high-multiple tech; regression for sanity check
    "💻 Tech / Growth":       dict(dcf=50, ddm=0,  ev=10, pe=30, reg=10),

    # Dividend/Value: heavy DDM + P/E for yield-focused companies (KO, PEP, JNJ)
    "💰 Dividend / Value":    dict(dcf=15, ddm=40, ev=15, pe=20, reg=10),

    # Industrial/Energy: balanced — EV/EBITDA dominant but DCF valid for
    # pure manufacturers (HON, RTX, LMT). Captive finance detection below
    # auto-reduces DCF weight for companies like CAT and DE.
    "🏭 Industrial / Energy": dict(dcf=25, ddm=5, ev=35, pe=20, reg=15),

    # Financials: handled by specialist models (excess return + P/BV)
    # Sliders here mainly control DDM and P/E weighting
    "🏦 Financials":          dict(dcf=0,  ddm=20, ev=0,  pe=35, reg=15),

    # Healthcare: DCF for pipeline value; P/E for mature pharma; EV/EBITDA for comps
    "🏥 Healthcare":          dict(dcf=40, ddm=5,  ev=20, pe=25, reg=10),

    # Consumer Staples: DDM + EV/EBITDA dominant — DCF undervalues brand moats
    # KO/PEP/PG/CL trade at premium FCF multiples the market assigns to stability
    "🛒 Consumer Staples":    dict(dcf=15, ddm=35, ev=25, pe=20, reg=5),

    # Comm Services: mix — streaming/digital = DCF; telco = DDM; EV/EBITDA for all
    "📡 Comm. Services":      dict(dcf=30, ddm=15, ev=25, pe=20, reg=10),

    # Utilities/REIT: DDM-heavy (high yield, regulated); DCF for rate base growth
    "⚡ Utilities / REIT":    dict(dcf=15, ddm=40, ev=20, pe=15, reg=10),

    # Consumer Discretionary: growth-oriented, FCF lumpy; P/E and regression
    "🛍️ Consumer Disc.":      dict(dcf=40, ddm=0,  ev=20, pe=25, reg=15),

    # Basic Materials: EV/EBITDA dominant; commodity cycles make DCF noisy
    "⛏️ Basic Materials":     dict(dcf=20, ddm=10, ev=40, pe=20, reg=10),

    # Quant: for when you trust the peer data more than assumptions
    "🔢 Quant / Regression":  dict(dcf=10, ddm=5,  ev=20, pe=20, reg=45),
}

SECTOR_TO_PRESET = {
    "Technology":             "💻 Tech / Growth",
    "Consumer Discretionary": "🛍️ Consumer Disc.",
    "Consumer Cyclical":      "🛍️ Consumer Disc.",
    "Healthcare":             "🏥 Healthcare",
    "Financials":             "🏦 Financials",
    "Financial Services":     "🏦 Financials",
    "Insurance":              "🏦 Financials",
    "Consumer Staples":       "🛒 Consumer Staples",
    "Consumer Defensive":     "🛒 Consumer Staples",
    "Energy":                 "🏭 Industrial / Energy",
    "Industrials":            "🏭 Industrial / Energy",
    "Basic Materials":        "⛏️ Basic Materials",
    "Communication Services": "📡 Comm. Services",
    "Utilities":              "⚡ Utilities / REIT",
    "Real Estate":            "⚡ Utilities / REIT",
}

def apply_preset(preset_name):
    """Store preset as pending — applied before sliders render on next rerun."""
    st.session_state["_pending_preset"] = preset_name

# ── Apply any pending values BEFORE widgets render ────────────────────────
if "_pending_preset" in st.session_state:
    _p = WEIGHT_PRESETS.get(st.session_state["_pending_preset"], WEIGHT_PRESETS["⚖️ Balanced"])
    st.session_state["w_dcf"] = _p["dcf"]
    st.session_state["w_ddm"] = _p["ddm"]
    st.session_state["w_ev"]  = _p["ev"]
    st.session_state["w_pe"]  = _p["pe"]
    st.session_state["w_reg"] = _p["reg"]
    del st.session_state["_pending_preset"]

if "_pending_wacc" in st.session_state:
    st.session_state["s_wacc"] = st.session_state.pop("_pending_wacc")
if "_pending_coe" in st.session_state:
    st.session_state["s_coe"]  = st.session_state.pop("_pending_coe")
if "_pending_g1" in st.session_state:
    st.session_state["s_g1"]   = st.session_state.pop("_pending_g1")
if "_pending_g2" in st.session_state:
    st.session_state["s_g2"]   = st.session_state.pop("_pending_g2")
if "_pending_tg" in st.session_state:
    st.session_state["s_tg"]   = st.session_state.pop("_pending_tg")
if "_pending_dcf" in st.session_state:
    st.session_state["w_dcf"]  = st.session_state.pop("_pending_dcf")
if "_pending_ddm" in st.session_state:
    st.session_state["w_ddm"]  = st.session_state.pop("_pending_ddm")
if "_pending_ev" in st.session_state:
    st.session_state["w_ev"]   = st.session_state.pop("_pending_ev")
if "_pending_reg" in st.session_state:
    st.session_state["w_reg"]  = st.session_state.pop("_pending_reg")
if "_pending_pe" in st.session_state:
    st.session_state["w_pe"]   = st.session_state.pop("_pending_pe")

# ── Disclaimer gate ────────────────────────────────────────────────────────
if not st.session_state.disclaimer_ok:
    st.markdown(
        """
        <div style="max-width:680px;margin:60px auto;padding:32px 36px;
                    background:#1a1200;border:1px solid #92400e;border-radius:12px;
                    font-size:13px;line-height:1.8;color:#fcd34d;">
            <div style="font-size:18px;font-weight:700;color:#fbbf24;margin-bottom:16px;">
                ⚠️  Important Disclaimer — Please Read Before Using IntrinsiQ
            </div>
            <p>IntrinsiQ is developed by <strong>McMillin Analytics</strong> for
            <strong>educational and research purposes only</strong>. Nothing in this
            application constitutes investment advice, a buy/sell recommendation, or a
            solicitation to purchase or sell any security.</p>
            <p><strong>All valuations are simplified models using estimated inputs.</strong>
            Results are highly sensitive to assumptions and should never be the sole basis
            for an investment decision. A model output does not mean the market is wrong.</p>
            <p>Investing involves <strong>risk including the possible loss of principal</strong>.
            Past performance is not indicative of future results. McMillin Analytics is not a
            registered investment advisor, broker-dealer, or financial planner.
            Always consult a qualified financial professional before making any investment decisions.</p>
            <p style="margin-bottom:0;">By clicking below you acknowledge these limitations and agree
            that McMillin Analytics bears no liability for any investment outcomes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        if st.button("✓  I Understand — Enter IntrinsiQ", type="primary", use_container_width=True):
            st.session_state.disclaimer_ok = True
            st.rerun()
    st.stop()

# ── Cached fetchers ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_metrics(ticker):
    return DataFetcher().get_key_metrics(ticker)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_peers(tickers_tuple):
    return DataFetcher().get_peers_data(list(tickers_tuple))

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker):
    hist, _, _ = DataFetcher().get_history(ticker, "2y")
    return hist

# ── Helpers ────────────────────────────────────────────────────────────────
def fmt(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"${v:,.{d}f}"

def fmt_large(v):
    if v is None: return "N/A"
    if abs(v) >= 1e12: return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:  return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:  return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"

def fmt_pct(v, d=1):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
    return f"{v*100:+.{d}f}%"

# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 IntrinsiQ")
    st.caption("McMillin Analytics · Blended Valuation Engine")
    st.divider()

    # Ticker
    ticker_input = st.text_input("Stock Ticker", value="JNJ", max_chars=10,
                                  placeholder="e.g. AAPL, KO, JPM")
    ticker = ticker_input.strip().upper()
    col_a, col_r = st.columns([4, 1])
    with col_a:
        analyze_btn = st.button("▶  ANALYZE", type="primary", use_container_width=True)
    with col_r:
        if st.button("🔄", use_container_width=True, help="Force refresh — clears cached data for this ticker and fetches fresh prices"):
            invalidate_ticker_cache(ticker)
            st.session_state.analysis_done = False
            st.session_state.last_ticker = ""
            st.session_state["_force_analyze"] = True
            st.rerun()

    st.divider()

    # Model weights
    st.markdown("**Model Weights**")

    # Preset buttons — clicking immediately applies, no separate button needed
    preset_names = list(WEIGHT_PRESETS.keys())
    active = st.session_state.get("active_preset", "⚖️ Balanced")

    # Show auto-suggested preset if analysis done
    if st.session_state.analysis_done and st.session_state.metrics.get("sector"):
        sec = st.session_state.metrics["sector"]
        auto = SECTOR_TO_PRESET.get(sec, "⚖️ Balanced")
        st.caption(f"📐 Auto-applied for **{sec}**")

    # Preset pills — 2 columns of buttons
    cols = st.columns(2)
    for i, name in enumerate(preset_names):
        is_active = (name == active)
        label = f"✓ {name}" if is_active else name
        if cols[i % 2].button(label, key=f"preset_{i}", use_container_width=True,
                               type="primary" if is_active else "secondary"):
            apply_preset(name)
            st.session_state["active_preset"] = name
            st.rerun()

    st.caption("Active preset shown with ✓ — sliders override after selection")
    st.divider()

    # Sliders — show current % next to each label
    # No default values here — session state drives these entirely
    w_dcf = st.slider("DCF",          0, 100, step=5, key="w_dcf")
    w_ddm = st.slider("DDM",          0, 100, step=5, key="w_ddm")
    w_ev  = st.slider("EV/EBITDA",    0, 100, step=5, key="w_ev")
    w_pe  = st.slider("P/E Relative", 0, 100, step=5, key="w_pe")
    w_reg = st.slider("Regression",   0, 100, step=5, key="w_reg")

    total_w = w_dcf + w_ddm + w_ev + w_pe + w_reg
    # Visual weight bar
    if total_w > 0:
        bar_html = (
            f'<div style="display:flex;height:6px;border-radius:4px;overflow:hidden;margin:4px 0 8px 0;">'
            f'<div style="width:{w_dcf/total_w*100:.0f}%;background:#3b82f6;" title="DCF"></div>'
            f'<div style="width:{w_ddm/total_w*100:.0f}%;background:#8b5cf6;" title="DDM"></div>'
            f'<div style="width:{w_ev/total_w*100:.0f}%;background:#10b981;" title="EV/EBITDA"></div>'
            f'<div style="width:{w_pe/total_w*100:.0f}%;background:#f59e0b;" title="P/E"></div>'
            f'<div style="width:{w_reg/total_w*100:.0f}%;background:#ef4444;" title="Regression"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;font-size:9px;color:#94a3b8;">'
            f'<span>■ DCF {w_dcf}%</span><span>■ DDM {w_ddm}%</span>'
            f'<span>■ EV {w_ev}%</span><span>■ PE {w_pe}%</span><span>■ Reg {w_reg}%</span>'
            f'</div>'
        )
        st.markdown(bar_html, unsafe_allow_html=True)
    if total_w == 100:
        st.success(f"✓ Weights sum to 100%")
    else:
        st.warning(f"Weights: {total_w}% — auto-renormalized")

    st.divider()

    # DCF assumptions — auto-populated from computed WACC after Analyze
    st.markdown("**DCF Assumptions**")
    if st.session_state.wacc_data:
        wd = st.session_state.wacc_data
        st.caption(
            f"📐 Computed: WACC **{wd['wacc']*100:.2f}%** | "
            f"CoE **{wd['cost_of_equity']*100:.2f}%** | "
            f"β **{wd['beta']}** | "
            f"E/D **{wd['equity_weight']*100:.0f}%/{wd['debt_weight']*100:.0f}%**"
        )
    wacc     = st.slider("WACC (%)",           4.0, 20.0, step=0.25, key="s_wacc") / 100
    g_stage1 = st.slider("Growth S1 5yr (%)", -10.0, 60.0, step=0.5,  key="s_g1")  / 100
    g_stage2 = st.slider("Growth S2 5yr (%)", -10.0, 40.0, step=0.5,  key="s_g2")  / 100
    term_g   = st.slider("Terminal Growth (%)", 0.5,  5.0,  step=0.25, key="s_tg")  / 100

    st.markdown("**DDM / Cost of Equity**")
    coe = st.slider("Cost of Equity (%)", 4.0, 20.0, step=0.25, key="s_coe") / 100
    dg  = st.slider("Dividend Growth (%)", 0.0, 15.0, step=0.5,  key="s_dg")  / 100

    st.divider()

    # Manual FCF override
    # FCF override lives in DCF tab — initialize here so analyze block can read it
    m_fcf    = st.session_state.get("s_manual_fcf", 0.0)
    use_mfcf = st.session_state.get("s_use_mfcf", False)

    # Peers
    st.markdown("**Peer Companies**")
    max_peers = st.slider("Max peers", 4, 12, 6, 1)

    # Cache
    with st.expander("⚙️ Cache"):
        if st.button("Clear Cache"):
            DataFetcher().clear_cache()
            st.cache_data.clear()
            st.session_state.analysis_done = False
            st.success("Cleared.")
        s = DataFetcher().get_cache_status()
        st.caption(f"{s.get('entries', 0)} entries · {s.get('size_mb', 0):.1f} MB")


# ── ANALYZE ────────────────────────────────────────────────────────────────
if st.session_state.pop("_force_analyze", False):
    analyze_btn = True

if analyze_btn:
    st.session_state.last_ticker = ticker

    with st.spinner(f"Fetching {ticker}..."):
        m = fetch_metrics(ticker)

    if not m or not m.get("name"):
        st.error(f"Could not fetch data for **{ticker}**. Check the symbol.")
        st.stop()
    if not m.get("current_price"):
        st.warning(f"⚠️ Live price unavailable for **{ticker}** — using last known price from cache if available.")

    # Manual FCF override
    if use_mfcf and m_fcf != 0:
        m["free_cashflow"] = m_fcf

    # Compute dynamic WACC from company fundamentals
    wacc_data = compute_dynamic_wacc(
        beta=m.get("beta", 1.0) or 1.0,
        total_debt=m.get("total_debt", 0) or 0,
        market_cap=m.get("market_cap", 1e9) or 1e9,
        sector=m.get("sector", ""),
    )
    st.session_state.wacc_data = wacc_data

    # Suggest growth rate defaults from company fundamentals
    dcf_sugg = suggest_dcf_assumptions(m)
    st.session_state.dcf_sugg = dcf_sugg

    # Push computed values into slider session state keys (pending pattern)
    st.session_state["_pending_wacc"] = round(wacc_data["wacc"] * 100, 2)
    st.session_state["_pending_coe"]  = round(wacc_data["cost_of_equity"] * 100, 2)
    st.session_state["_pending_g1"]   = dcf_sugg.get("g1", 10.0)
    st.session_state["_pending_g2"]   = dcf_sugg.get("g2", 6.0)
    st.session_state["_pending_tg"]   = dcf_sugg.get("terminal", 2.5)

    # Fetch peers
    with st.spinner("Loading peers..."):
        peer_tickers = get_peers_for_ticker(
            ticker,
            {"sector": m.get("sector", ""), "industry": m.get("industry", "")},
            max_peers,
        )
        peer_tickers = [t for t in peer_tickers if t != ticker][:max_peers]
        peers = fetch_peers(tuple(peer_tickers))

    # Save to session
    # Auto-apply sector preset when ticker changes
    sector = m.get("sector", "")
    auto_preset = SECTOR_TO_PRESET.get(sector, "⚖️ Balanced")
    apply_preset(auto_preset)
    st.session_state["active_preset"] = auto_preset

    # For financials: override weights to use financial-specific models
    if is_financial_sector(sector):
        st.session_state["_pending_preset"] = "🏦 Financials"
        st.session_state["active_preset"]   = "🏦 Financials"
    st.session_state["_is_financial"] = is_financial_sector(sector)

    # Auto-zero DDM if dividend yield below 0.5% — DDM meaningless for non-payers
    div_y_raw = m.get("dividend_yield", 0) or 0
    div_y_pct = div_y_raw if div_y_raw > 1 else div_y_raw * 100
    if div_y_pct < 1.0:  # zero DDM for yields below 1% — too small to be meaningful
        current_ddm = st.session_state.get("w_ddm", 0)
        if current_ddm > 0:
            # Use _pending_ pattern — direct widget key assignment crashes Streamlit
            st.session_state["_pending_dcf"] = st.session_state.get("w_dcf", 35) + current_ddm
            st.session_state["_pending_ddm"] = 0

    # ── Captive finance detection ─────────────────────────────────────────
    # Companies with captive finance arms (CAT, DE, CNH, AGCO) report massive
    # debt from their lending subsidiaries. This inflates D/E and destroys DCF
    # net debt calculations. Detect by: sector is Industrials/Consumer Disc AND
    # debt-to-equity > 2.0. For these, slash DCF weight and boost EV/EBITDA.
    de_raw = m.get("debt_to_equity", 0) or 0
    # yfinance returns debtToEquity as percent (250 = 2.5x), normalize
    de_ratio = de_raw / 100 if de_raw > 20 else de_raw
    # Captive finance only applies to heavy equipment/auto OEMs with lending arms
    # Raise threshold to 3.0x and exclude pure-play EV/tech companies
    CAPTIVE_EXCLUSIONS = {"TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI"}
    captive_sectors = ("Industrials", "Consumer Discretionary", "Consumer Cyclical")
    if sector in captive_sectors and de_ratio > 3.0 and ticker not in CAPTIVE_EXCLUSIONS:
        current_dcf = st.session_state.get("w_dcf", 25)
        current_ev  = st.session_state.get("w_ev", 35)
        current_reg = st.session_state.get("w_reg", 15)
        if current_dcf > 10:
            # Use _pending_ pattern — direct widget key assignment crashes Streamlit
            shift = current_dcf - 10
            st.session_state["_pending_dcf"] = 10
            st.session_state["_pending_ev"]  = min(current_ev + int(shift * 0.6), 55)
            st.session_state["_pending_reg"] = min(current_reg + int(shift * 0.4), 30)
        st.session_state["_captive_finance_flag"] = True
        st.session_state["_captive_de_ratio"] = round(de_ratio, 2)
    else:
        st.session_state["_captive_finance_flag"] = False
        st.session_state["_captive_de_ratio"] = None

    # ── Regulated / Capital-Intensive detection ──────────────────────────
    # Telcos (AT&T, VZ) and Utilities (NEE, DUK) share the same problem:
    # massive regulated debt destroys DCF. Value is driven by yield, rate
    # environment, and peer EV/EBITDA — not discounted cash flows.
    TELCO_TICKERS = {"T", "VZ", "TMUS", "LUMN", "TDS", "USM"}
    de_ratio_cs = de_raw / 100 if de_raw > 20 else de_raw
    is_utility  = sector in ("Utilities",)
    is_telco    = (sector in ("Communication Services",) and de_ratio_cs > 1.5) or ticker in TELCO_TICKERS
    is_regulated = (is_utility or is_telco) and not is_financial_sector(sector)
    if is_regulated:
        if is_utility:
            # Utilities: lean on DDM (stable growing dividends) + EV/EBITDA + regression
            st.session_state["_pending_dcf"] = 10
            st.session_state["_pending_ddm"] = 30
            st.session_state["_pending_ev"]  = 25
            st.session_state["_pending_reg"] = 25
            st.session_state["_pending_pe"]  = 10
        else:
            # Telcos: regression and EV/EBITDA dominant
            st.session_state["_pending_dcf"] = 10
            st.session_state["_pending_ddm"] = 20
            st.session_state["_pending_ev"]  = 30
            st.session_state["_pending_reg"] = 30
            st.session_state["_pending_pe"]  = 10
        st.session_state["_telco_flag"] = True
        st.session_state["_is_utility"] = is_utility
    else:
        st.session_state["_telco_flag"] = False
        st.session_state["_is_utility"] = False

    # ── REIT detection ───────────────────────────────────────────────────
    # REITs use FFO (Funds from Operations) not GAAP earnings. Depreciation
    # on real estate is added back, making reported earnings look terrible.
    # DCF and P/E massively understate value. EV/EBITDA, DDM, and regression
    # are the only reliable anchors for REITs.
    industry = m.get("industry", "")
    is_reit = is_reit_sector(sector, industry)
    if is_reit:
        div_y_reit = m.get("dividend_yield", 0) or 0
        div_y_reit = div_y_reit if div_y_reit > 1 else div_y_reit * 100
        # DDM consistently undervalues REITs — cost of equity too high, growth too
        # conservative, doesn't capture NAV appreciation or rent roll upside.
        # EV/EBITDA and regression are the reliable anchors for all REIT types.
        if div_y_reit >= 3.0:
            # Income REITs (O): modest DDM weight, EV/EBITDA + regression dominant
            st.session_state["_pending_dcf"] = 5
            st.session_state["_pending_ddm"] = 15
            st.session_state["_pending_ev"]  = 45
            st.session_state["_pending_reg"] = 30
            st.session_state["_pending_pe"]  = 5
        else:
            # Growth REITs (PLD, AMT): DDM minimal, EV/EBITDA + regression only
            st.session_state["_pending_dcf"] = 5
            st.session_state["_pending_ddm"] = 5
            st.session_state["_pending_ev"]  = 50
            st.session_state["_pending_reg"] = 35
            st.session_state["_pending_pe"]  = 5
        st.session_state["_reit_flag"] = True
    else:
        st.session_state["_reit_flag"] = False

    st.session_state.metrics       = m
    st.session_state.peer_data     = peers
    st.session_state.sensitivity_df = None  # recomputed live
    st.session_state.analysis_done  = True
    st.session_state["_collapse_sidebar"] = True
    st.rerun()


# ── RESULTS ────────────────────────────────────────────────────────────────
# Auto-collapse sidebar on mobile after analyze
if st.session_state.get('_collapse_sidebar'):
    st.session_state['_collapse_sidebar'] = False
    js = ('<script>'
          'if(window.innerWidth < 768){'
          'const b=window.parent.document.querySelector("[data-testid=\\"collapsedControl\\"]");'
          'if(b)b.click();}'
          '</script>')
    st.markdown(js, unsafe_allow_html=True)

if not st.session_state.analysis_done:
    st.title("🔬 IntrinsiQ")
    st.markdown("**McMillin Analytics** · Multi-Method Blended Intrinsic Value Calculator")
    st.info("← Enter a ticker in the sidebar and click **ANALYZE**")
    st.stop()

m      = st.session_state.metrics
peers  = st.session_state.peer_data
price  = m.get("current_price", 0)
fcf    = m.get("free_cashflow", 0) or 0

# Sanitize net debt — caps captive-finance debt for industrials, auto, etc.
raw_net_debt = m.get("net_debt", 0) or 0
net_debt, nd_capped, nd_cap_mult = sanitize_net_debt(
    raw_net_debt,
    m.get("ebitda", 0) or 0,
    m.get("sector", ""),
)

# ── Live model computation — reruns on every slider change ─────────────────
if fcf > 0:
    dcf_r = dcf_valuation(
        base_fcf=fcf, wacc=wacc,
        growth_rate_stage1=g_stage1, growth_rate_stage2=g_stage2,
        terminal_growth=term_g,
        net_debt=net_debt,
        shares_outstanding=m.get("shares_outstanding", 1),
    )
else:
    dcf_r = {"error": "No positive FCF", "intrinsic_value": None}

# Standard models
mr = {
    "dcf":         dcf_r,
    "ddm":         ddm_valuation(m.get("dividend_rate", 0) or 0, dg, coe),
    "ev_ebitda":   ev_ebitda_valuation(m.get("ebitda", 0) or 0, peers, net_debt, m.get("shares_outstanding", 1)),
    "pe_relative": pe_relative_valuation(m.get("eps_forward") or m.get("eps", 0) or 0, peers),
    "regression":  regression_valuation(m, peers),
}

# Financial sector: add specialist models and override weights
is_fin = st.session_state.get("_is_financial", False) or is_financial_sector(m.get("sector", ""))

# Captive finance warning banner
if st.session_state.get("_captive_finance_flag"):
    de = st.session_state.get("_captive_de_ratio", "")
    st.warning(
        f"⚠️ **Captive Finance Detected** (D/E: {de}x) — {m.get('name', ticker)} operates a "
        f"captive lending arm that inflates reported debt. DCF weight auto-reduced; "
        f"EV/EBITDA and Regression weighted higher for more reliable valuation.",
        icon=None
    )
if st.session_state.get("_telco_flag"):
    if st.session_state.get("_is_utility"):
        st.warning(
            f"⚡ **Regulated Utility Detected** — {m.get('name', ticker)} carries heavy infrastructure "
            f"debt typical of regulated utilities. DCF weight reduced; DDM, EV/EBITDA and Regression "
            f"weighted higher for more reliable valuation.",
            icon=None
        )
        # Growth utility caveat — low yield + premium sector suggests market is pricing
        # future capacity (renewables, AI power demand) not captured in current financials
        div_y_check = m.get("dividend_yield", 0) or 0
        div_y_check = div_y_check if div_y_check > 1 else div_y_check * 100
        if div_y_check < 3.0:
            st.info(
                f"🌱 **Growth Utility Note** — {m.get('name', ticker)} trades at a significant premium "
                f"to traditional utility peers due to its renewables pipeline and AI/data center power "
                f"demand narrative. Like high-growth story stocks, fundamental models may materially "
                f"understate market value. Use this valuation as a floor, not a target.",
                icon=None
            )
    else:
        st.warning(
            f"📡 **Telecom Detected** — {m.get('name', ticker)} is a capital-intensive, regulated carrier. "
            f"DCF is unreliable due to heavy infrastructure debt. Weights shifted to Regression and EV/EBITDA.",
            icon=None
        )
if st.session_state.get("_reit_flag"):
    st.warning(
        f"🏢 **REIT Detected** — {m.get('name', ticker)} reports FFO (Funds from Operations), not GAAP "
        f"earnings. Real estate depreciation makes DCF and P/E unreliable. Weights shifted to "
        f"EV/EBITDA, DDM, and Regression — the standard valuation approach for REITs.",
        icon=None
    )
    # Growth REIT caveat — low yield signals market is pricing rent roll upside,
    # contracted infrastructure growth, or NAV premium not in current financials
    div_y_reit_disp = m.get("dividend_yield", 0) or 0
    div_y_reit_disp = div_y_reit_disp if div_y_reit_disp > 1 else div_y_reit_disp * 100
    if div_y_reit_disp < 3.5:
        st.info(
            f"📈 **Growth REIT Note** — {m.get('name', ticker)} trades at a premium to NAV due to "
            f"embedded rent roll upside, contracted long-term revenue, or sector scarcity value. "
            f"Fundamental models may materially understate market value. "
            f"Use this valuation as a floor, not a target.",
            icon=None
        )
if is_fin:
    mr["excess_return"] = excess_return_valuation(
        book_value_per_share=m.get("book_value", 0) or 0,
        roe=m.get("roe", 0) or 0,
        cost_of_equity=coe,
        growth_rate=g_stage1,
        terminal_growth=term_g,
    )
    # Add cost_of_equity to metrics for pbv model
    m_with_coe = {**m, "cost_of_equity": coe}
    mr["pbv_relative"] = pbv_relative_valuation(m_with_coe, peers)

    # Force weights: zero out DCF and EV/EBITDA, use financial models
    weights = {
        "dcf":           0,
        "ddm":           w_ddm / 100,
        "ev_ebitda":     0,
        "pe_relative":   w_pe  / 100,
        "regression":    w_reg / 100,
        "excess_return": 0.35,   # anchor weight for residual income
        "pbv_relative":  0.25,   # anchor weight for P/BV relative
    }

weights = {"dcf": w_dcf/100, "ddm": w_ddm/100, "ev_ebitda": w_ev/100, "pe_relative": w_pe/100, "regression": w_reg/100}
bl      = compute_blended_value(mr, weights)
bv      = bl.get("blended_value")
upside  = ((bv / price) - 1) * 100 if bv and price else None

# Sensitivity recomputes live too (fast — pure math)
sens = sensitivity_dcf(
    base_fcf=fcf, net_debt=net_debt,
    shares_outstanding=m.get("shares_outstanding", 1),
    terminal_growth=term_g,
) if fcf > 0 else None

# Inverse DCF — solve for implied growth rate at current market price
inv_r = inverse_dcf(
    current_price=price, base_fcf=fcf, wacc=wacc,
    terminal_growth=term_g,
    net_debt=net_debt,
    shares_outstanding=m.get("shares_outstanding", 1),
) if fcf > 0 and price else {"error": "Requires positive FCF", "implied_growth": None}

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"## {m['ticker']} — {m.get('name', '')}")
    st.caption(f"{m.get('sector','—')} · {m.get('industry','—')} · Mkt Cap: {fmt_large(m.get('market_cap'))}")
with col2:
    cache_tag = "🟡 Cached" if m.get("_from_cache") else "🟢 Live"
    st.caption(cache_tag)
    if bv:
        color = "green" if (upside or 0) > 5 else ("red" if (upside or 0) < -5 else "orange")
        st.markdown(f"### :{color}[${bv:.2f}]")
        st.caption(f"vs ${price:.2f} market · {upside:+.1f}%")

st.divider()

# Key metrics — 3 cols on mobile, 6 on desktop
div_y = m.get("dividend_yield", 0) or 0
div_y_pct = div_y if div_y > 1 else div_y * 100
# Flag if yield looks abnormally high (>15%) — likely bad data
div_y_display = f"{div_y_pct:.2f}% ⚠️" if div_y_pct > 15 else (f"{div_y_pct:.2f}%" if div_y else "—")
r1c1, r1c2, r1c3 = st.columns(3)
with r1c1: st.metric("Price",   fmt(price))
with r1c2: st.metric("Mkt Cap", fmt_large(m.get("market_cap")))
with r1c3: st.metric("Fwd P/E", f"{m.get('pe_forward',0):.1f}x" if m.get("pe_forward") else "—")
r2c1, r2c2, r2c3 = st.columns(3)
fcf_display = "N/A (Bank)" if is_financial_sector(m.get("sector","")) else fmt_large(m.get("free_cashflow"))
with r2c1: st.metric("FCF", fcf_display)
with r2c2: st.metric("Div Yield", div_y_display)
with r2c3: st.metric("Beta",      f"{m.get('beta',0):.2f}" if m.get("beta") else "—")

st.divider()

# Net debt cap warning
if nd_capped:
    st.warning(
        f"⚠️ **Net debt capped** at {nd_cap_mult}× EBITDA (${net_debt/1e9:.1f}B) — "
        f"raw balance sheet debt was ${raw_net_debt/1e9:.1f}B, likely includes captive finance arm "
        f"(e.g. equipment lending). DCF/DDM use capped figure.",
        icon=None,
    )

# WACC breakdown — 3 cols per row for mobile friendliness
if st.session_state.wacc_data:
    wd = st.session_state.wacc_data
    wc1,wc2,wc3 = st.columns(3)
    with wc1: st.metric("WACC", f"{wd['wacc']*100:.2f}%", help="Weighted Avg Cost of Capital")
    with wc2: st.metric("Cost of Equity", f"{wd['cost_of_equity']*100:.2f}%", help="CAPM: Rf + Beta x ERP")
    with wc3: st.metric("Cost of Debt", f"{wd['after_tax_debt']*100:.2f}%", help="After-tax cost of debt")
    wc4,wc5,_ = st.columns(3)
    with wc4: st.metric("Equity Weight", f"{wd['equity_weight']*100:.1f}%", help="Market cap / (Market cap + Debt)")
    with wc5: st.metric("Beta", f"{wd['beta']:.2f}", help="Market sensitivity from yfinance")
    st.divider()

# Model results
st.markdown("**Model Results**")

if is_fin:
    # Financial sector — show specialist model row
    st.caption("🏦 Financial sector — using Excess Return and P/BV Relative models")
    fin_cols = st.columns(4)
    fin_models = [
        ("excess_return", "Excess Return"),
        ("pbv_relative",  "P/BV Relative"),
        ("ddm",           "DDM"),
        ("pe_relative",   "P/E Relative"),
    ]
    for col, (key, label) in zip(fin_cols, fin_models):
        r = mr.get(key, {})
        with col:
            if r and not r.get("error") and r.get("intrinsic_value"):
                v = r["intrinsic_value"]
                diff = ((v/price)-1)*100 if price else 0
                st.metric(label, fmt(v), f"{diff:+.1f}%")
            else:
                err = (r.get("error","") or "")[:28] if r else "Not run"
                st.metric(label, "—", err)
else:
    # 3 cols row 1, 2 cols row 2 — works on mobile and desktop
    model_items = list(MODEL_LABELS.items())
    row1 = st.columns(3)
    row2 = st.columns(3)
    for col, (key, label) in zip(row1 + row2, model_items):
        r = mr.get(key, {})
        with col:
            if r and not r.get("error") and r.get("intrinsic_value"):
                v = r["intrinsic_value"]
                diff = ((v/price)-1)*100 if price else 0
                st.metric(label, fmt(v), f"{diff:+.1f}%")
            else:
                st.metric(label, "—", (r.get("error","Not run") or "Not run")[:30] if r else "Not run")

st.divider()

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Overview", "🔮 DCF Detail", "🏢 Comps", "🔄 Inverse DCF", "🌡️ Sensitivity", "🏢 About"])

with tab1:
    hist = fetch_history(m["ticker"])
    if hist is not None and not hist.empty:
        st.plotly_chart(create_price_history_chart(hist, m["ticker"], bv), use_container_width=True)
    if upside is not None:
        st.plotly_chart(create_upside_gauge(upside, m["ticker"]), use_container_width=True)
    if bl.get("contributions"):
        st.plotly_chart(create_valuation_waterfall(bl["contributions"], price, m["ticker"]), use_container_width=True)

with tab2:
    # Financial sector — show residual income detail instead
    if is_fin:
        er = mr.get("excess_return", {})
        pb = mr.get("pbv_relative", {})
        st.markdown("**Excess Return (Residual Income) Model**")
        if not er.get("error") and er.get("intrinsic_value"):
            e1,e2,e3,e4 = st.columns(4)
            with e1: st.metric("Intrinsic Value",   fmt(er["intrinsic_value"]))
            with e2: st.metric("Book Value/Share",   fmt(er.get("book_value_per_share")))
            with e3: st.metric("ROE",               f"{er.get('roe',0)*100:.1f}%")
            with e4: st.metric("Premium to Book",   f"{er.get('premium_to_book',0)*100:+.1f}%",
                                help="Positive = ROE > CoE, bank creates value. Negative = ROE < CoE.")
            pv_ex = er.get("excess_return_pv", 0)
            pv_tv = er.get("terminal_value_pv", 0)
            bv    = er.get("book_value_per_share", 0)
            st.caption(
                f"Book Value: {fmt(bv)} + "
                f"PV Excess Returns (5yr): {fmt(pv_ex)} + "
                f"PV Terminal: {fmt(pv_tv)} = {fmt(er['intrinsic_value'])}"
            )
        else:
            st.warning(f"Excess Return model: {er.get('error','failed')}")

        st.divider()
        st.markdown("**P/BV Relative Valuation**")
        if not pb.get("error") and pb.get("intrinsic_value"):
            p1,p2,p3 = st.columns(3)
            with p1: st.metric("Intrinsic Value",    fmt(pb["intrinsic_value"]))
            with p2: st.metric("Fair P/BV Multiple", f"{pb.get('pb_multiple_used',0):.2f}x")
            with p3: st.metric("ROE/CoE Ratio",      f"{pb.get('roe_coe_ratio',0):.2f}x",
                                help="ROE/CoE > 1 justifies premium to book value")
        else:
            st.warning(f"P/BV model: {pb.get('error','failed')}")

    else:
    # Non-financial DCF detail
     if True:
      dcf_r = mr.get("dcf", {})
      if dcf_r.get("error"):
        st.warning(f"DCF unavailable: {dcf_r['error']}")
        st.info("Use **Manual FCF Override** below. Enter the TTM Free Cash Flow from Yahoo Finance → Financials → Cash Flow Statement.")
      else:
        d1,d2,d3,d4 = st.columns(4)
        with d1: st.metric("DCF Value",     fmt(dcf_r.get("intrinsic_value")))
        with d2: st.metric("Enterprise Val",fmt_large(dcf_r.get("enterprise_value")))
        with d3: st.metric("Equity Value",  fmt_large(dcf_r.get("equity_value")))
        with d4: st.metric("Terminal Val %",f"{dcf_r.get('tv_pct',0)*100:.1f}%")
        if dcf_r.get("projected_fcfs"):
            st.plotly_chart(create_dcf_projection_chart(
                dcf_r["projected_fcfs"], dcf_r["pv_fcfs"], dcf_r["pv_tv"], m["ticker"]
            ), use_container_width=True)

    with st.expander("🔧 Manual FCF Override", expanded=False):
        st.caption("Use if DCF shows 'No positive FCF' or the value looks wrong. Find TTM FCF on Yahoo Finance → Financials → Cash Flow.")
        new_fcf = st.number_input("Free Cash Flow ($)", value=st.session_state.s_manual_fcf,
                                   step=1e8, format="%.0f", key="fcf_input")
        use_new = st.checkbox("Apply manual FCF", value=st.session_state.s_use_mfcf, key="fcf_checkbox")
        if st.button("Apply & Re-analyze", type="primary", key="fcf_apply"):
            st.session_state["s_manual_fcf"] = new_fcf
            st.session_state["s_use_mfcf"]   = use_new
            # Clear cache for this ticker so it re-fetches with override
            st.session_state.analysis_done = False
            st.rerun()
        if st.session_state.s_use_mfcf and st.session_state.s_manual_fcf:
            st.success(f"Using manual FCF: ${st.session_state.s_manual_fcf/1e9:.2f}B")

    with st.expander("🩺 FCF Diagnostic", expanded=False):
        fd = DataFetcher()
        fin_d, _, _ = fd.get_financials(m["ticker"])
        if fin_d and fin_d.get("cash_flow") is not None:
            cf = fin_d["cash_flow"]
            rows = []
            for idx in cf.index:
                first_val = next((float(v) for v in cf.loc[idx] if pd.notna(v)), None)
                rows.append({"Row": str(idx), "Value": f"${first_val:,.0f}" if first_val is not None else "N/A"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            fcf_cf, _ = DataFetcher._extract_fcf_from_cashflow(cf, m["ticker"])
            c1, c2 = st.columns(2)
            with c1: st.metric("FCF from Yahoo info", fmt_large(m.get("free_cashflow")))
            with c2: st.metric("FCF from CF statement", fmt_large(fcf_cf) if fcf_cf else "Not extracted")
        else:
            st.error("Cash flow not returned — possible rate limit.")

with tab3:
    if not peers:
        st.warning("No peer data loaded.")
    else:
        cols = ["ticker","name","current_price","pe_trailing","pe_forward","ev_ebitda","ps_ratio","pb_ratio"]
        df = pd.DataFrame([{k: p.get(k) for k in cols} for p in peers])
        df.columns = [c.replace("_"," ").title() for c in df.columns]
        st.dataframe(df, use_container_width=True, hide_index=True)

with tab4:
    if inv_r.get("error"):
        st.warning(f"Inverse DCF: {inv_r['error']}")
    else:
        ig = inv_r.get("implied_growth", 0)
        delta = ig - g_stage1

        i1,i2,i3,i4 = st.columns(4)
        with i1: st.metric("Implied FCF Growth",  f"{ig*100:.2f}%",
                            help="The FCF growth rate the market is pricing in at the current share price")
        with i2: st.metric("Your S1 Assumption",  f"{g_stage1*100:.1f}%",
                            help="Your Stage 1 growth assumption from the sidebar slider")
        with i3: st.metric("Difference",          f"{delta*100:+.2f}%",
                            help="Positive = market is more optimistic than you. Negative = you are more optimistic.")
        with i4: st.metric("WACC Used",            f"{wacc*100:.2f}%")

        # Interpretation
        if delta > 0.02:
            st.success(f"✅ Market implies **{ig*100:.1f}%** growth — {delta*100:.1f}% above your assumption of {g_stage1*100:.1f}%. "
                       f"The market is pricing in more optimism than your model.")
        elif delta < -0.02:
            st.info(f"💡 Market implies **{ig*100:.1f}%** growth — {abs(delta)*100:.1f}% below your assumption of {g_stage1*100:.1f}%. "
                    f"If your estimate is right, the stock may be undervalued.")
        else:
            st.warning(f"⚖️ Market-implied growth ({ig*100:.1f}%) closely matches your assumption ({g_stage1*100:.1f}%). "
                       f"The stock appears fairly valued on a DCF basis.")

        # Scenario chart
        if inv_r.get("scenarios"):
            st.plotly_chart(
                create_inverse_dcf_chart(inv_r["scenarios"], ig, price, m["ticker"]),
                use_container_width=True
            )
            # Scenario table
            scen_df = pd.DataFrame({
                "FCF Growth Rate": [f"{g*100:.0f}%" for g in sorted(inv_r["scenarios"])],
                "Implied Price":   [fmt(v) for v in [inv_r["scenarios"][g] for g in sorted(inv_r["scenarios"])]],
                "vs Market":       [f"{((v/price)-1)*100:+.1f}%" for v in [inv_r["scenarios"][g] for g in sorted(inv_r["scenarios"])]],
            })
            st.dataframe(scen_df, use_container_width=True, hide_index=True)

with tab5:
    if sens is not None:
        st.plotly_chart(create_sensitivity_heatmap(sens, price, m["ticker"]), use_container_width=True)
    else:
        st.warning("Sensitivity requires positive FCF.")



with tab6:
    desc      = m.get("description", "")
    ceo       = m.get("ceo", "")
    city      = m.get("city", "")
    state     = m.get("state", "")
    country   = m.get("country", "")
    website   = m.get("website", "")
    employees = m.get("employees")

    st.markdown(f"### {m.get('name', ticker)}")
    loc_parts = [x for x in [city, state, country] if x]
    st.caption(" · ".join(filter(None, [" · ".join(loc_parts), m.get("sector",""), m.get("industry","")])))

    f1, f2, f3 = st.columns(3)
    with f1: st.metric("CEO", ceo or "—")
    with f2: st.metric("Employees", f"{employees:,.0f}" if employees else "—")
    with f3:
        if website:
            st.markdown("**Website**")
            st.markdown(f"[{website.replace('https://','').replace('http://','').rstrip('/')}]({website})")
        else:
            st.metric("Website", "—")

    st.divider()

    if desc:
        st.markdown("**About**")
        if len(desc) > 500:
            st.write(desc[:500] + "...")
            with st.expander("Read more"):
                st.write(desc)
        else:
            st.write(desc)
    else:
        st.info("Company description not available.")

    st.divider()

    target_mean = m.get("target_mean_price")
    target_low  = m.get("target_low_price")
    target_high = m.get("target_high_price")
    rating      = m.get("analyst_rating", "")
    count       = m.get("analyst_count", 0)
    if target_mean:
        st.markdown("**Analyst Consensus**")
        a1, a2, a3, a4 = st.columns(4)
        upside_str = f"{((target_mean/price)-1)*100:+.1f}% vs market" if price else ""
        with a1: st.metric("Mean Target",  fmt(target_mean), upside_str)
        with a2: st.metric("Low Target",   fmt(target_low))
        with a3: st.metric("High Target",  fmt(target_high))
        with a4: st.metric("Rating", rating.replace("-"," ").title() if rating else "—",
                            f"{count} analysts" if count else "")

# Footer
st.divider()
st.markdown(
    '<div style="text-align:center;font-size:10px;color:#64748b;line-height:1.9;">'
    '<strong>IntrinsiQ · McMillin Analytics</strong><br>'
    'For educational and research purposes only — not investment advice.<br>'
    'Investing involves risk including loss of principal. '
    'Not a registered investment advisor. '
    'Always consult a qualified financial professional before making investment decisions.<br>'
    '© 2025 McMillin Analytics · All rights reserved.</div>',
    unsafe_allow_html=True,
)
