"""
Visualization module — all Plotly charts for IntrinsiQ.
Dark theme, professional financial aesthetics.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Shared theme ──────────────────────────────────────────
BG_PRIMARY   = "#0a0e17"
BG_SECONDARY = "#111827"
BG_CARD      = "#161d2e"
BORDER       = "#1e293b"
TEXT_PRIMARY  = "#e2e8f0"
TEXT_MUTED    = "#64748b"
ACCENT_BLUE  = "#3b82f6"
ACCENT_TEAL  = "#06b6d4"
ACCENT_GREEN = "#10b981"
ACCENT_RED   = "#ef4444"
ACCENT_AMBER = "#f59e0b"
ACCENT_PURPLE= "#a855f7"

MODEL_COLORS = {
    "dcf":        ACCENT_BLUE,
    "ddm":        ACCENT_PURPLE,
    "ev_ebitda":  ACCENT_TEAL,
    "pe_relative": ACCENT_AMBER,
    "regression": ACCENT_GREEN,
}
MODEL_LABELS = {
    "dcf": "DCF",
    "ddm": "DDM",
    "ev_ebitda": "EV/EBITDA",
    "pe_relative": "P/E Relative",
    "regression": "Regression",
}

LAYOUT_BASE = dict(
    paper_bgcolor=BG_CARD,
    plot_bgcolor=BG_CARD,
    font=dict(family="DM Mono, monospace", color=TEXT_PRIMARY, size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(
        gridcolor=BORDER, gridwidth=1,
        linecolor=BORDER, tickfont=dict(color=TEXT_PRIMARY)
    ),
    yaxis=dict(
        gridcolor=BORDER, gridwidth=1,
        linecolor=BORDER, tickfont=dict(color=TEXT_PRIMARY)
    ),
    hoverlabel=dict(
        bgcolor=BG_SECONDARY,
        bordercolor=BORDER,
        font=dict(color=TEXT_PRIMARY, family="DM Mono, monospace")
    ),
)


def _apply_layout(fig, title="", height=420):
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=14, color=TEXT_PRIMARY), x=0.0, xanchor="left"),
        height=height,
    )
    return fig


# ══════════════════════════════════════════════════════════
#  Price History Chart
# ══════════════════════════════════════════════════════════

def create_price_history_chart(hist_df, ticker: str, intrinsic_value: float = None) -> go.Figure:
    if hist_df is None or hist_df.empty:
        return go.Figure()

    fig = go.Figure()

    # Candlestick or line depending on data length
    if len(hist_df) > 30:
        fig.add_trace(go.Scatter(
            x=hist_df.index, y=hist_df["Close"],
            name="Price", line=dict(color=ACCENT_BLUE, width=2),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.08)",
            hovertemplate="$%{y:.2f}<extra>%{x|%b %d, %Y}</extra>"
        ))
    else:
        fig.add_trace(go.Candlestick(
            x=hist_df.index,
            open=hist_df["Open"], high=hist_df["High"],
            low=hist_df["Low"], close=hist_df["Close"],
            increasing_line_color=ACCENT_GREEN,
            decreasing_line_color=ACCENT_RED,
            name="OHLC"
        ))

    if intrinsic_value and intrinsic_value > 0:
        fig.add_hline(
            y=intrinsic_value,
            line_dash="dot", line_color=ACCENT_GREEN,
            annotation_text=f"  Intrinsic: ${intrinsic_value:.2f}",
            annotation_font_color=ACCENT_GREEN,
        )

    return _apply_layout(fig, f"{ticker} — Price History", height=380)


# ══════════════════════════════════════════════════════════
#  Blended Value Waterfall / Bar
# ══════════════════════════════════════════════════════════

def create_valuation_waterfall(
    contributions: dict,
    current_price: float,
    ticker: str,
) -> go.Figure:
    if not contributions:
        return go.Figure()

    labels = [MODEL_LABELS.get(k, k) for k in contributions]
    values = [v["value"] for v in contributions.values()]
    weights = [v["effective_weight"] for v in contributions.values()]
    colors = [MODEL_COLORS.get(k, ACCENT_BLUE) for k in contributions]

    blended = sum(v["contribution"] for v in contributions.values())

    # Add blended and current price bars
    labels += ["Blended IV", "Market Price"]
    values += [blended, current_price]
    colors += [ACCENT_GREEN, ACCENT_RED if current_price > blended else ACCENT_AMBER]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"${v:.2f}" for v in values],
        textposition="outside",
        textfont=dict(size=11, color=TEXT_PRIMARY),
        hovertemplate="%{y}: <b>$%{x:.2f}</b><extra></extra>",
        customdata=weights + [1.0, 1.0],
    ))

    # Vertical line at current price
    fig.add_vline(x=current_price, line_dash="dash", line_color=TEXT_MUTED, line_width=1)

    _apply_layout(fig, f"{ticker} — Valuation Bridge", height=350)
    fig.update_layout(xaxis_title="Intrinsic Value per Share ($)")
    return fig


# ══════════════════════════════════════════════════════════
#  Sensitivity Heatmap
# ══════════════════════════════════════════════════════════

def create_sensitivity_heatmap(
    df: pd.DataFrame,
    current_price: float,
    ticker: str,
) -> go.Figure:
    if df is None or df.empty:
        return go.Figure()

    # Color: green = above market, red = below
    relative = df.values / current_price - 1  # upside/downside matrix

    fig = go.Figure(go.Heatmap(
        z=relative * 100,
        x=df.columns.tolist(),
        y=df.index.tolist(),
        colorscale=[
            [0.0,  "#7f1d1d"],
            [0.35, "#ef4444"],
            [0.5,  "#1e293b"],
            [0.65, "#10b981"],
            [1.0,  "#064e3b"],
        ],
        zmid=0,
        text=[[f"${v:.0f}\n({(v/current_price-1)*100:+.0f}%)" for v in row] for row in df.values],
        texttemplate="%{text}",
        textfont=dict(size=9, color="white"),
        hovertemplate="WACC: %{x}<br>Growth: %{y}<br><b>$%{text}</b><extra></extra>",
        showscale=True,
        colorbar=dict(
            title=dict(text="vs Market (%)", font=dict(color=TEXT_PRIMARY)),
            tickfont=dict(color=TEXT_PRIMARY),
            bgcolor=BG_CARD,
            bordercolor=BORDER,
        ),
    ))

    _apply_layout(fig, f"{ticker} DCF Sensitivity — Intrinsic Value vs Market Price", height=420)
    fig.update_layout(
        xaxis_title="WACC",
        yaxis_title="FCF Growth Rate (Stage 1)",
    )
    return fig


# ══════════════════════════════════════════════════════════
#  Comps Scatter / Bar
# ══════════════════════════════════════════════════════════

def create_comps_chart(
    peer_data: list,
    target_metrics: dict,
    x_metric: str = "ev_ebitda",
    y_metric: str = "pe_trailing",
) -> go.Figure:
    if not peer_data:
        return go.Figure()

    tickers = []
    x_vals = []
    y_vals = []
    sizes = []

    for p in peer_data:
        x = p.get(x_metric)
        y = p.get(y_metric)
        if x and y and x > 0 and y > 0 and x < 200 and y < 300:
            tickers.append(p.get("ticker", "?"))
            x_vals.append(x)
            y_vals.append(y)
            mc = p.get("market_cap") or 1e9
            sizes.append(max(6, min(30, mc / 1e10)))

    fig = go.Figure()

    # Peers
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="markers+text",
        text=tickers,
        textposition="top center",
        textfont=dict(size=9, color=TEXT_MUTED),
        marker=dict(
            size=sizes, color=ACCENT_BLUE,
            opacity=0.7,
            line=dict(color=BORDER, width=1)
        ),
        name="Peers",
        hovertemplate="<b>%{text}</b><br>" + x_metric + ": %{x:.1f}<br>" + y_metric + ": %{y:.1f}<extra></extra>",
    ))

    # Target
    tx = target_metrics.get(x_metric)
    ty = target_metrics.get(y_metric)
    if tx and ty and tx > 0 and ty > 0:
        fig.add_trace(go.Scatter(
            x=[tx], y=[ty],
            mode="markers+text",
            text=[target_metrics.get("ticker", "TARGET")],
            textposition="top center",
            textfont=dict(size=11, color=ACCENT_GREEN, family="DM Mono"),
            marker=dict(
                size=18, color=ACCENT_GREEN,
                symbol="star",
                line=dict(color="white", width=1.5)
            ),
            name="Target",
            hovertemplate="<b>%{text} ★</b><br>" + x_metric + ": %{x:.1f}<br>" + y_metric + ": %{y:.1f}<extra></extra>",
        ))

    x_label = x_metric.replace("_", "/").upper()
    y_label = y_metric.replace("_", "/").upper()
    _apply_layout(fig, f"Comps — {x_label} vs {y_label}", height=420)
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
    return fig


# ══════════════════════════════════════════════════════════
#  DCF Cash Flow Projection
# ══════════════════════════════════════════════════════════

def create_dcf_projection_chart(
    projected_fcfs: list,
    pv_fcfs: list,
    pv_tv: float,
    ticker: str,
) -> go.Figure:
    if not projected_fcfs:
        return go.Figure()

    years = list(range(1, len(projected_fcfs) + 1))
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Projected Free Cash Flows", "Present Value Breakdown"],
        horizontal_spacing=0.12,
    )
    fig.update_annotations(font=dict(color=TEXT_PRIMARY))

    # Left: FCF bars
    fig.add_trace(go.Bar(
        x=[f"Y{y}" for y in years], y=[f / 1e6 for f in projected_fcfs],
        marker_color=ACCENT_BLUE,
        name="FCF ($M)",
        hovertemplate="Y%{x}: $%{y:.0f}M<extra></extra>",
    ), row=1, col=1)

    # Right: PV breakdown
    labels = [f"Y{y}" for y in years] + ["Terminal\nValue"]
    values = [p / 1e6 for p in pv_fcfs] + [pv_tv / 1e6]
    clrs = [ACCENT_BLUE] * len(pv_fcfs) + [ACCENT_TEAL]

    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=clrs,
        name="PV ($M)",
        hovertemplate="%{x}: $%{y:.0f}M<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=f"{ticker} — DCF Projection", font=dict(size=14, color=TEXT_PRIMARY), x=0.0),
        height=380,
        showlegend=False,
    )
    fig.update_yaxes(title_text="$M", row=1, col=1, gridcolor=BORDER)
    fig.update_yaxes(title_text="$M", row=1, col=2, gridcolor=BORDER)
    return fig


# ══════════════════════════════════════════════════════════
#  Inverse DCF — Implied Growth Scenarios
# ══════════════════════════════════════════════════════════

def create_inverse_dcf_chart(
    scenarios: dict,
    implied_growth: float,
    current_price: float,
    ticker: str,
) -> go.Figure:
    if not scenarios:
        return go.Figure()

    gs = sorted(scenarios.keys())
    prices = [scenarios[g] for g in gs]

    fig = go.Figure()

    # Line
    fig.add_trace(go.Scatter(
        x=[g * 100 for g in gs],
        y=prices,
        mode="lines+markers",
        line=dict(color=ACCENT_BLUE, width=2),
        marker=dict(size=6, color=ACCENT_BLUE),
        name="Implied DCF Price",
        hovertemplate="Growth %{x:.0f}%: $%{y:.2f}<extra></extra>",
    ))

    # Current market price line
    fig.add_hline(
        y=current_price,
        line_dash="dash", line_color=ACCENT_RED,
        annotation_text=f"  Market: ${current_price:.2f}",
        annotation_font_color=ACCENT_RED,
    )

    # Implied growth marker
    if implied_growth is not None:
        fig.add_vline(
            x=implied_growth * 100,
            line_dash="dot", line_color=ACCENT_GREEN,
            annotation_text=f"  Implied: {implied_growth*100:.1f}%",
            annotation_font_color=ACCENT_GREEN,
            annotation_textangle=0,
        )

    _apply_layout(fig, f"{ticker} — Inverse DCF: Price vs. Implied FCF Growth Rate", height=380)
    fig.update_layout(
        xaxis_title="FCF Growth Rate (%)",
        yaxis_title="Intrinsic Value per Share ($)",
    )
    return fig


# ══════════════════════════════════════════════════════════
#  Comps Table (peer multiples bar chart)
# ══════════════════════════════════════════════════════════

def create_comps_multiples_bar(
    peer_data: list,
    target_metrics: dict,
    metric: str = "ev_ebitda",
) -> go.Figure:
    tickers, values, colors = [], [], []
    target_val = target_metrics.get(metric)
    target_ticker = target_metrics.get("ticker", "TARGET")
    median_val = None

    valid = [(p.get("ticker", "?"), p.get(metric)) for p in peer_data
             if p.get(metric) and 0 < p.get(metric, 0) < 300]

    if not valid:
        return go.Figure()

    import statistics
    vals = [v for _, v in valid]
    try:
        median_val = statistics.median(vals)
    except Exception:
        median_val = sum(vals) / len(vals)

    for t, v in sorted(valid, key=lambda x: x[1]):
        tickers.append(t)
        values.append(v)
        colors.append(ACCENT_BLUE)

    # Insert target
    if target_val and target_val > 0:
        tickers.append(f"★ {target_ticker}")
        values.append(target_val)
        colors.append(ACCENT_GREEN)

    label = metric.upper().replace("_", "/")
    fig = go.Figure(go.Bar(
        x=tickers, y=values,
        marker_color=colors,
        text=[f"{v:.1f}x" for v in values],
        textposition="outside",
        textfont=dict(size=10, color=TEXT_PRIMARY),
        hovertemplate="%{x}: <b>%{y:.1f}x</b><extra></extra>",
    ))

    if median_val:
        fig.add_hline(
            y=median_val,
            line_dash="dot", line_color=ACCENT_AMBER,
            annotation_text=f"  Median: {median_val:.1f}x",
            annotation_font_color=ACCENT_AMBER,
        )

    _apply_layout(fig, f"Comparable {label} Multiples", height=350)
    fig.update_layout(xaxis_title="Company", yaxis_title=f"{label} Multiple")
    return fig


# ══════════════════════════════════════════════════════════
#  Regression Feature Importance
# ══════════════════════════════════════════════════════════

def create_regression_importance_chart(feature_importance: dict, ticker: str) -> go.Figure:
    if not feature_importance:
        return go.Figure()

    sorted_items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    labels = [k.replace("_", " ").title() for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    norm = max(values) if max(values) > 0 else 1
    colors = [f"rgba(59,130,246,{0.4 + 0.6 * v/norm})" for v in values]

    fig = go.Figure(go.Bar(
        y=labels, x=values,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=10, color=TEXT_PRIMARY),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))

    _apply_layout(fig, f"{ticker} — Regression Feature Coefficients", height=320)
    fig.update_layout(xaxis_title="Absolute Coefficient Weight")
    return fig


# ══════════════════════════════════════════════════════════
#  Gauge Chart — Upside/Downside
# ══════════════════════════════════════════════════════════

def create_upside_gauge(upside_pct: float, ticker: str) -> go.Figure:
    clamped = max(-100, min(200, upside_pct))
    color = ACCENT_GREEN if clamped > 10 else (ACCENT_RED if clamped < -10 else ACCENT_AMBER)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=clamped,
        number=dict(suffix="%", font=dict(size=32, color=color, family="DM Mono, monospace")),
        delta=dict(reference=0, suffix="%", font=dict(size=16)),
        gauge=dict(
            axis=dict(
                range=[-100, 200],
                tickwidth=1,
                tickcolor=BORDER,
                tickfont=dict(color=TEXT_MUTED),
            ),
            bar=dict(color=color, thickness=0.3),
            bgcolor=BG_CARD,
            borderwidth=0,
            steps=[
                dict(range=[-100, -10], color="rgba(239,68,68,0.12)"),    # red = overvalued
                dict(range=[-10, 10],   color="rgba(245,158,11,0.12)"),   # amber = fairly valued
                dict(range=[10, 200],   color="rgba(16,185,129,0.12)"),   # green = upside
            ],
            threshold=dict(
                line=dict(color=TEXT_MUTED, width=2),
                thickness=0.75,
                value=0,
            ),
        ),
        title=dict(text=f"{ticker} — Upside / Downside", font=dict(color=TEXT_MUTED, size=12)),
    ))

    fig.update_layout(
        paper_bgcolor=BG_CARD,
        height=260,
        margin=dict(l=20, r=20, t=20, b=10),
        font=dict(color=TEXT_PRIMARY),
    )
    return fig


# ══════════════════════════════════════════════════════════
#  DDM Chart
# ══════════════════════════════════════════════════════════

def create_ddm_chart(result: dict, ticker: str) -> go.Figure:
    if not result or result.get("error"):
        return go.Figure()

    pv_divs = result.get("pv_dividends", [])
    pv_tv = result.get("pv_terminal", 0)

    if not pv_divs:
        return go.Figure()

    years = [f"Y{i+1}" for i in range(len(pv_divs))] + ["Terminal"]
    values = [d / 1e6 if d > 1e6 else d for d in pv_divs] + [pv_tv / 1e6 if pv_tv > 1e6 else pv_tv]
    scale = "($M)" if any(v > 1e3 for v in pv_divs) else "($)"
    colors = [ACCENT_PURPLE] * len(pv_divs) + [ACCENT_TEAL]

    fig = go.Figure(go.Bar(
        x=years, y=values,
        marker_color=colors,
        text=[f"${v:.1f}" for v in values],
        textposition="outside",
        hovertemplate="%{x}: $%{y:.2f}<extra></extra>",
    ))

    _apply_layout(fig, f"{ticker} — DDM: PV of Dividends {scale}", height=340)
    return fig
